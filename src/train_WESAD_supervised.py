import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Iterable, List, Sequence, Tuple
from types import SimpleNamespace
import ast

from scipy.signal import butter, filtfilt, find_peaks
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datasets import Dataset as hgDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.utils.checkpoint as checkpoint_utils

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score
)
from datetime import datetime
from tqdm import tqdm
from NormWear.main_model import NormWearModel
from NormWear.modules.normwear import NormWear, cwt_wrap
from NormWear.pretrain_pipeline.misc import NativeScalerWithGradNormCount as NativeScaler
from NormWear.downstream_pipeline.engine_finetune import train_one_epoch

def per_class_precision_recall_from_cm(cm):
    cm = np.asarray(cm)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    return precision, recall

class MLP_Handcraft(nn.Module):
    def __init__(self, input_shape, num_class):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class), 
        )

    def forward(self, x):
        return self.net(x)

class NormwearMLP(nn.Module):
    def __init__(self, embedding_length, num_class, normwear_weight_path=None):
        super().__init__()

        if normwear_weight_path is None:
            self.normwear = NormWear(img_size=(387,65), patch_size=(9,5),mask_scheme='random',mask_prob=0.8,use_cwt=True,nvar=4, comb_freq=False)
            self.normwear.train()
        else:
            self.normwear = NormWear(img_size=(387,65), patch_size=(9,5),mask_scheme='random',mask_prob=0.8,use_cwt=True,nvar=4, comb_freq=False)
            self.normwear.load_state_dict(torch.load(normwear_weight_path, map_location=torch.device('cpu')))
            self.normwear.train()
        self.net = nn.Sequential(
            nn.Linear(embedding_length, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class), 
        )

    def forward(self, x):
        embedding = self.normwear.get_signal_embedding(x, hidden_out=False, device=x.device)
        embedding = embedding.mean(dim=2).reshape(embedding.shape[0], -1).squeeze(0)
        return self.net(embedding)

CWT_CAT_M_PLUS_DEFAULTS = {
    "embed_dim": 512,
    "depth": 10,
    "heads": 8,
    "channel_heads": 8,
    "dropout": 0.1,
    "drop_path": 0.05,
}


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtGridBlock(nn.Module):
    """Small ConvNeXt-style local block over the CWT patch grid."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AxialCWTBlock(nn.Module):
    """Factorized attention over CWT frequency and time axes."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.freq_norm = nn.LayerNorm(dim)
        self.freq_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(dim)
        self.time_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time_patches, freq_patches, dim]
        batch_size, num_channels, num_time, num_freq, dim = x.shape

        freq_tokens = x.reshape(batch_size * num_channels * num_time, num_freq, dim)
        freq_normed = self.freq_norm(freq_tokens)
        freq_update, _ = self.freq_attn(freq_normed, freq_normed, freq_normed, need_weights=False)
        x = x + self.drop_path(freq_update.reshape(batch_size, num_channels, num_time, num_freq, dim))

        time_tokens = x.permute(0, 1, 3, 2, 4).reshape(batch_size * num_channels * num_freq, num_time, dim)
        time_normed = self.time_norm(time_tokens)
        time_update, _ = self.time_attn(time_normed, time_normed, time_normed, need_weights=False)
        time_update = time_update.reshape(batch_size, num_channels, num_freq, num_time, dim).permute(0, 1, 3, 2, 4)
        x = x + self.drop_path(time_update)

        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x


class ChannelFusionBlock(nn.Module):
    """Attention across the five physiological channels."""

    def __init__(self, dim: int, num_heads: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=2.0, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_tokens = x.mean(dim=(2, 3))
        normed = self.norm(channel_tokens)
        update, _ = self.attn(normed, normed, normed, need_weights=False)
        update = self.drop_path(update)
        channel_tokens = channel_tokens + update
        channel_tokens = channel_tokens + self.drop_path(self.mlp(self.mlp_norm(channel_tokens)))
        return x + channel_tokens[:, :, None, None, :]


class AttentiveGridPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_time, num_freq, dim = x.shape
        tokens = x.reshape(batch_size, num_channels, num_time * num_freq, dim)
        weights = torch.softmax(self.score(self.norm(tokens)).squeeze(-1), dim=-1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=2)


class CWTCATEncoder(nn.Module):
    """CWT Conv-Axial Transformer Small encoder.

    Input is a chunk of CWT windows with shape [B, 5, 3, 388, 65].
    Output is one 768-d vector per physiological channel: [B, 5, 768].
    """

    def __init__(
        self,
        num_channels: int = 5,
        embed_dim: int = 384,
        output_dim: int = 768,
        depth: int = 6,
        num_heads: int = 6,
        channel_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.05,
        fusion_every: int = 2,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=(9, 5), stride=(9, 5))
        self.patch_hw = (43, 13)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_channels, 1, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, self.patch_hw[0], 1, embed_dim))
        self.freq_embed = nn.Parameter(torch.zeros(1, 1, 1, self.patch_hw[1], embed_dim))
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.local_stem = nn.Sequential(
            ConvNeXtGridBlock(embed_dim, drop_path=dpr[0] if depth else 0.0),
            ConvNeXtGridBlock(embed_dim, drop_path=dpr[1] if depth > 1 else 0.0),
        )
        self.axial_blocks = nn.ModuleList(
            [
                AxialCWTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[layer_idx],
                )
                for layer_idx in range(depth)
            ]
        )
        self.channel_fusion = nn.ModuleDict(
            {
                str(layer_idx): ChannelFusionBlock(embed_dim, channel_heads, dropout, dpr[layer_idx])
                for layer_idx in range(depth)
                if (layer_idx + 1) % fusion_every == 0
            }
        )
        self.pool = AttentiveGridPool(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_adapter = nn.Linear(embed_dim, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.freq_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, cwt_chunk: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, deriv_channels, cwt_len, num_scales = cwt_chunk.shape
        x = cwt_chunk.reshape(batch_size * num_channels, deriv_channels, cwt_len, num_scales)
        x = self.patch_embed(x)
        if tuple(x.shape[-2:]) != self.patch_hw:
            raise RuntimeError(f"CWT-CAT-S expected patch grid {self.patch_hw}, got {tuple(x.shape[-2:])}")
        x = self.local_stem(x)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, num_channels, self.patch_hw[0], self.patch_hw[1], self.embed_dim)
        x = x + self.channel_embed + self.time_embed + self.freq_embed
        for layer_idx, block in enumerate(self.axial_blocks):
            x = block(x)
            layer_key = str(layer_idx)
            if layer_key in self.channel_fusion:
                x = self.channel_fusion[layer_key](x)
        x = self.pool(x)
        x = self.output_adapter(self.output_norm(x))
        return x


class VitaStressCWTCATMultiTask(nn.Module):
    def __init__(
        self,
        shared_dropout: float = 0.2,
        window_chunk_size: int = 1,
        grad_checkpoint: bool = False,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        channel_heads: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.window_chunk_size = max(1, int(window_chunk_size))
        self.grad_checkpoint = grad_checkpoint
        self.encoder = CWTCATEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            channel_heads=channel_heads,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.row_projector = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(15 * 128, 256),
            nn.ReLU(),
            nn.Dropout(shared_dropout),
        )
        self.stress_binary_head = nn.Linear(256, 1)
        # self.stress_three_class_head = nn.Linear(256, 3)

    @property
    def patch_grid(self) -> Tuple[int, int]:
        return self.encoder.patch_hw

    def encode_window_chunk(self, cwt_chunk: torch.Tensor) -> torch.Tensor:
        return self.encoder(cwt_chunk)

    def temporal_slope(self, features: torch.Tensor) -> torch.Tensor:
        num_windows = features.shape[1]
        positions = torch.arange(num_windows, device=features.device, dtype=features.dtype)
        centered = positions - positions.mean()
        denom = centered.pow(2).sum().clamp_min(1e-6)
        centered = centered.view(1, num_windows, 1, 1)
        mean_features = features.mean(dim=1, keepdim=True)
        numer = ((features - mean_features) * centered).sum(dim=1)
        return numer / denom

    def summarize_temporal_features(self, per_window_features: torch.Tensor) -> torch.Tensor:
        mean_features = per_window_features.mean(dim=1)
        std_features = per_window_features.std(dim=1, unbiased=False)
        slope_features = self.temporal_slope(per_window_features)
        return torch.cat((mean_features, std_features, slope_features), dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, num_windows, num_channels, seq_len = x.shape
        per_window_features = []
        for start in range(0, num_windows, self.window_chunk_size):
            end = min(start + self.window_chunk_size, num_windows)
            chunk = x[:, start:end]
            chunk_windows = end - start
            with torch.amp.autocast(device_type="cuda", enabled=False):
                cwt = cwt_wrap(
                    chunk.reshape(batch_size * chunk_windows * num_channels, seq_len).float(),
                    0.1,
                    64,
                )
            _, n_deriv, cwt_len, n_scales = cwt.shape
            cwt = cwt.reshape(batch_size * chunk_windows, num_channels, n_deriv, cwt_len, n_scales)
            if self.grad_checkpoint and self.training:
                chunk_features = checkpoint_utils.checkpoint(self.encode_window_chunk, cwt, use_reentrant=False)
            else:
                chunk_features = self.encode_window_chunk(cwt)
            chunk_features = chunk_features.reshape(batch_size, chunk_windows, num_channels, 768)
            per_window_features.append(chunk_features)

        per_window_features = torch.cat(per_window_features, dim=1)
        summary_rows = self.summarize_temporal_features(per_window_features)
        projected_rows = self.row_projector(summary_rows)
        z_shared = self.shared_trunk(projected_rows.reshape(batch_size, -1))
        outputs = self.stress_binary_head(z_shared).reshape(-1),
        return outputs, z_shared


if __name__ == "__main__":
    epochs = 35
    batch_size=64
    lr=0.0001
    window_size = 40
    step_size = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeding_length = 4608
    feature_length = ((340 - window_size) // step_size + 1) * window_size

    df = pd.read_parquet("./data/WESAD/WESAD_Data_60s_6s_norm.parquet")

    num_baseline = len(df[(df["Label"] == 0) | (df["Label"] == 3)])
    num_stress = len(df[df["Label"] == 1])
    num_amusement = len(df[df["Label"] == 2])
    total = num_baseline + num_stress + num_amusement

    print("Num baseline labels: " + str(num_baseline))
    print("Num stress labels: " + str(num_stress))
    print("Num amusement labels: " + str(num_amusement))
    print("Total labels: " + str(total))

    SUBJECT_IDS = (
        [f"S{i}" for i in range(2, 12)] +
        [f"S{i}" for i in range(13, 18)]
    )

    args = SimpleNamespace(
        accum_iter=1,
        warmup_epochs=2,
        min_lr = 1e-7,
        lr = lr,
        weight_decay = 0.05,
        epochs=epochs,
    )
    
    for opt in ["NormWear"]: #, "Handcraft", "NormWear_Embed", "NormWear"]:
        for mode in ["Bin_stress"]: #, "Bin_stress", "Tri_stress_amuse"]:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f'./run/{timestamp}_{opt}_{mode}'
            writer = SummaryWriter(log_dir=log_dir)

            avg_loso_f1 = 0
            avg_loso_acc = 0
            for sid in SUBJECT_IDS:
                tr_data = df[df["PID"] != sid]
                val_data = df[df["PID"] == sid]

                input_dim = feature_length
                num_class = 2

                tr_X_tensor = torch.tensor(np.vstack(tr_data["Handcraft"].to_numpy()), dtype=torch.float32)
                tr_X_tensor = tr_X_tensor.unfold(1, window_size, step_size).flatten(1)
                val_X_tensor = torch.tensor(np.vstack(val_data["Handcraft"].to_numpy()), dtype=torch.float32)
                val_X_tensor = val_X_tensor.unfold(1, window_size, step_size).flatten(1)

                if opt == "NormWear_Embed":
                    tr_X_tensor = [[[a.astype(np.float32) for a in b] for b in c] for c in tr_data["Normwear_Input"]]
                    val_X_tensor = [[[a.astype(np.float32) for a in b] for b in c] for c in val_data["Normwear_Input"]]
                    tr_X_tensor = torch.tensor(np.vstack(tr_X_tensor), dtype=torch.float32)
                    val_X_tensor = torch.tensor(np.vstack(val_X_tensor), dtype=torch.float32)
                    input_dim = embeding_length
                elif opt == "NormWear":
                    cwt_data_file = "./data/WESAD/WESAD_CWT_NormWear_Data_6s_6s.parquet"
                    hf_dataset = hgDataset.from_parquet(cwt_data_file)
                    hf_dataset.set_format(type="torch", columns=["Normwear_CWT", "Label"])

                    train_dataset = hf_dataset.filter(lambda row: row["PID"] != sid)
                    val_dataset = hf_dataset.filter(lambda row: row["PID"] == sid)

                    if mode == "Bin_stress":
                        train_dataset = train_dataset.map(lambda row: {"Label": 0 if row["Label"] > 1 else row["Label"]})
                        val_dataset = val_dataset.map(lambda row: {"Label": 0 if row["Label"] > 1 else row["Label"]})

                    input_dim = embeding_length

                tr_y_tensor = torch.empty(0)
                val_y_tensor = torch.empty(0)
                if mode == "Bin_stress":
                    tr_y_tensor = torch.tensor(np.where(tr_data["Label"] > 1, 0, tr_data["Label"]))
                    val_y_tensor = torch.tensor(np.where(val_data["Label"] > 1, 0, val_data["Label"]))
                elif mode == "Tri_stress_amuse":
                    tr_y_tensor = torch.tensor(np.where(tr_data["Label"] > 2, 0, tr_data["Label"]))
                    val_y_tensor = torch.tensor(np.where(val_data["Label"] > 2, 0, val_data["Label"]))
                    num_class = 3

                if opt != "NormWear":
                    train_dataset = TensorDataset(tr_X_tensor, tr_y_tensor)
                    val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model = MLP_Handcraft(input_shape=input_dim, num_class=num_class).to(device)
                if opt == "NormWear":
                    model = NormwearMLP(embedding_length=input_dim, num_class=num_class, normwear_weight_path="./src/NormWear/normwear_pretrain_ckpt.pth").to(device)                    

                class_weights = torch.tensor([total / (num_class * num_baseline), 
                                              total / (num_class * num_stress), 
                                              total / (num_class * num_amusement)], 
                                              dtype=torch.float32).to(device)
                if mode == "Bin_stress":
                    class_weights = torch.tensor([total / (num_class * (num_baseline + num_amusement)), 
                                                  total / (num_class * num_stress)], 
                                                  dtype=torch.float32).to(device)

                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                scaler = NativeScaler()
                
                best_val_loss = float('inf')
                best_val_acc = 0
                best_val_f1 = 0
                for epoch in tqdm(range(epochs)):
                    model.train()
                    train_loss = 0.0
                    
                    # for batch_X, batch_y in train_loader:
                    #     batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    #     print(batch_X.shape)

                    #     logits = model(batch_X)
                    #     loss = criterion(logits, batch_y.long())

                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     optimizer.step()
                    #     train_loss += loss.item() * batch_X.size(0)

                    met = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, scaler, args = args)

                    train_loss /= len(train_loader.dataset)
                    
                    # 4. Validation & Metrics Phase
                    model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []
                    
                    with torch.no_grad():
                        # for batch_X, batch_y in val_loader:
                        #     batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            
                        #     logits = model(batch_X)
                        #     loss = criterion(logits, batch_y.long())
                        #     val_loss += loss.item() * batch_X.size(0)
                            
                        #     # Convert logits to binary predictions
                        #     preds = logits.argmax(dim=1)
                            
                        #     all_preds.append(preds.cpu().numpy())
                        #     all_targets.append(batch_y.cpu().numpy())

                        for batch in val_loader:
                            batch_X, batch_y = batch["Normwear_CWT"].to(device).reshape(-1, 6, 3, 389, 65), batch["Label"].to(device)
                            
                            logits = model(batch_X)
                            loss = criterion(logits, batch_y.long())
                            val_loss += loss.item() * batch_X.size(0)
                            
                            # Convert logits to binary predictions
                            preds = logits.argmax(dim=1)
                            
                            all_preds.append(preds.cpu().numpy())
                            all_targets.append(batch_y.cpu().numpy())
                            
                    val_loss /= len(val_loader.dataset)

                    all_preds = np.concatenate(all_preds)
                    all_targets = np.concatenate(all_targets)

                    val_f1 = f1_score(all_targets, all_preds, zero_division=0, average = "micro")
                    val_weighted_f1 = f1_score(all_targets, all_preds, zero_division=0, average = "weighted")
                    val_acc = accuracy_score(all_targets, all_preds)
                    
                    # Log metrics to TensorBoard
                    writer.add_scalar(f'{sid}/Loss/Train', train_loss, epoch)
                    writer.add_scalar(f'{sid}/Loss/Validation', val_loss, epoch)
                    writer.add_scalar(f'{sid}/F1', val_f1, epoch)
                    writer.add_scalar(f'{sid}/WeightedF1', val_weighted_f1, epoch)
                    writer.add_scalar(f'{sid}/ACC', val_acc, epoch)
                    
                    torch.save(model.state_dict(), os.path.join(log_dir, f'model_latest_{sid}.pth'))
                    
                    # Save Best Loss Model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_loss_{sid}.pth'))
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_acc_{sid}.pth'))
                    
                    if val_weighted_f1 > best_val_f1:
                        best_val_f1 = val_weighted_f1
                        torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_f1_{sid}.pth'))

                avg_loso_f1 += best_val_f1
                avg_loso_acc += best_val_acc
            avg_loso_f1 /= len(SUBJECT_IDS)
            avg_loso_acc /= len(SUBJECT_IDS)
            print(f"Average f1 for LOSO: {avg_loso_f1} | ACC: {avg_loso_acc}")

            writer.close()