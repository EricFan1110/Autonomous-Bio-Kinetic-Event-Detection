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
from NormWear.downstream_pipeline.engine_finetune import train_one_epoch, evaluate

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




def balance_pid_groups_two_class(df, target_prop=0.70):
    sampled_chunks = []

    # Group by PID to process each individual ID separately
    for pid, group in df.groupby("PID"):
        df_0 = group[group["Label"] != 1]
        df_other = group[group["Label"] == 1]

        n_0 = len(df_0)
        n_other = len(df_other)

        # Edge case: If either category is entirely missing, we can't form a 70/30 split
        if n_0 == 0 or n_other == 0:
            continue

        # Mathematical requirement: n_0 / (n_0 + n_other) = 0.70 -> n_0 = (7/3) * n_other
        # Check whether we have an excess of 0s or an excess of 'other' labels
        if n_0 / (n_0 + n_other) > target_prop:
            # Too many 0s: Keep all 'others', downsample 0s
            sample_size_other = n_other
            sample_size_0 = int(round((target_prop / (1 - target_prop)) * n_other))
        else:
            # Too many others: Keep all 0s, downsample 'others'
            sample_size_0 = n_0
            sample_size_other = int(round(((1 - target_prop) / target_prop) * n_0))

        # Safeguard against rounding errors resulting in 0 samples
        if sample_size_0 == 0 or sample_size_other == 0:
            continue

        # Randomly sample from both subsets based on calculated sizes
        sampled_0 = df_0.sample(n=sample_size_0, random_state=42)
        sampled_other = df_other.sample(n=sample_size_other, random_state=42)

        # Combine them back for this specific PID
        sampled_chunks.append(pd.concat([sampled_0, sampled_other]))

    # Merge all balanced PID groups back into a single DataFrame
    return pd.concat(sampled_chunks, ignore_index=True) if sampled_chunks else pd.DataFrame()

if __name__ == "__main__":
    epochs = 100
    batch_size=32
    lr=0.0001
    window_size = 40
    step_size = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeding_length = 4608
    feature_length = ((340 - window_size) // step_size + 1) * window_size
    undersample = None
    df = pd.read_parquet("/diniuvol/jonah/Eric/data/WESAD/WESAD_Data_6s_6s.parquet")
    val_method = "tr_val_test" # "loso", "tr_val_test"
    train_ids = ['S3','S4','S5','S9','S10','S11','S14','S15','S16','S17']
    val_ids = ['S7','S8']
    test_ids = ['S2','S6','S13']

    if undersample is not None:
        df = balance_pid_groups_two_class(df, target_prop=undersample)
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
    
    for opt in ["Handcraft"]: #, "Handcraft"]:
        for mode in ["Bin_stress"]: #, "Bin_stress", "Tri_stress_amuse"]:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f'/diniuvol/jonah/Eric/run/{timestamp}_{opt}_{mode}'
            writer = SummaryWriter(log_dir=log_dir)

            avg_loso_f1 = 0
            avg_loso_acc = 0
            if val_method != "loso":
                SUBJECT_IDS = [0]
            for sid in SUBJECT_IDS:
                print(sid)
                tr_data = df[df["PID"] != sid]
                val_data = df[df["PID"] == sid]

                if val_method == "tr_val_test":
                    tr_data = df[df["PID"].isin(train_ids)]
                    val_data = df[df["PID"].isin(val_ids)]
                    test_data = df[df["PID"].isin(test_ids)]

                input_dim = feature_length
                num_class = 2

                tr_X_tensor = torch.tensor(np.vstack(tr_data["Handcraft"].to_numpy()), dtype=torch.float32)
                tr_X_tensor = tr_X_tensor.unfold(1, window_size, step_size).flatten(1)
                val_X_tensor = torch.tensor(np.vstack(val_data["Handcraft"].to_numpy()), dtype=torch.float32)
                val_X_tensor = val_X_tensor.unfold(1, window_size, step_size).flatten(1)
                if val_method == "tr_val_test":
                    test_X_tensor = torch.tensor(np.vstack(test_data["Handcraft"].to_numpy()), dtype=torch.float32)
                    test_X_tensor = test_X_tensor.unfold(1, window_size, step_size).flatten(1)
                    test_y_tensor = torch.tensor(np.where(test_data["Label"] > 1, 0, test_data["Label"]))
                    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)

                if opt == "NormWear_Embed":
                    tr_X_tensor = np.vstack(tr_data["Normwewar_Embed"].values)
                    val_X_tensor = np.vstack(val_data["Normwewar_Embed"].values)
                    tr_X_tensor = torch.tensor(tr_X_tensor, dtype=torch.float32)
                    val_X_tensor = torch.tensor(val_X_tensor, dtype=torch.float32)

                    if val_method == "tr_val_test":
                        test_X_tensor = np.vstack(test_data["Normwewar_Embed"].values)
                        test_X_tensor = torch.tensor(test_X_tensor, dtype=torch.float32)
                        test_y_tensor = torch.tensor(np.where(test_data["Label"] > 1, 0, test_data["Label"]))
                        test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size)
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

                    tr_X_list = []
                    tr_y_list = []
                    for sid in tr_data["PID"].unique():
                        sid_data = tr_data[tr_data["PID"] == sid]
                        sid_X = np.vstack(sid_data["Normwear_Input"].values)
                        sid_X = torch.tensor(sid_X, dtype=torch.float32).reshape(-1, 6, 390).unfold(0, 10, 1)
                        sid_X = sid_X.transpose(1, 3).transpose(2,3)
                        tr_X_list.append(sid_X)
                        tr_y_data_= np.where(sid_data["Label"] > 1, 0, sid_data["Label"])
                        tr_y_list.append(torch.tensor(tr_y_data_[-sid_X.shape[0]:]))
                    tr_X_tensor = torch.cat(tr_X_list).to(torch.float16)
                    tr_y_tensor = torch.cat(tr_y_list)

                    val_X_list = []
                    val_y_list = []
                    for sid in val_data["PID"].unique():
                        sid_data = val_data[val_data["PID"] == sid]
                        sid_X = np.vstack(sid_data["Normwear_Input"].values)
                        sid_X = torch.tensor(sid_X, dtype=torch.float32).reshape(-1, 6, 390).unfold(0, 10, 1)
                        sid_X = sid_X.transpose(1, 3).transpose(2,3)
                        val_X_list.append(sid_X)
                        val_y_data_= np.where(sid_data["Label"] > 1, 0, sid_data["Label"])
                        val_y_list.append(torch.tensor(val_y_data_[-sid_X.shape[0]:]))
                    val_X_tensor = torch.cat(val_X_list).to(torch.float16)
                    val_y_tensor = torch.cat(val_y_list)

                    if val_method == "tr_val_test":
                        test_X_list = []
                        test_y_list = []
                        for sid in test_data["PID"].unique():
                            sid_data = test_data[test_data["PID"] == sid]
                            sid_X = np.vstack(sid_data["Normwear_Input"].values)
                            sid_X = torch.tensor(sid_X, dtype=torch.float32).reshape(-1, 6, 390).unfold(0, 10, 1)
                            sid_X = sid_X.transpose(1, 3).transpose(2,3)
                            test_X_list.append(sid_X)
                            test_y_data_= np.where(sid_data["Label"] > 1, 0, sid_data["Label"])
                            test_y_list.append(torch.tensor(test_y_data_[-sid_X.shape[0]:]))
                        test_X_tensor = torch.cat(test_X_list).to(torch.float16)
                        test_y_tensor = torch.cat(test_y_list)
                        test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size)

                    input_dim = embeding_length

                tr_y_tensor = torch.empty(0)
                val_y_tensor = torch.empty(0)
                if mode == "Bin_stress":
                    tr_y_tensor = torch.tensor(np.where(tr_data["Label"] > 1, 0, tr_data["Label"]))[:tr_X_tensor.shape[0]]
                    val_y_tensor = torch.tensor(np.where(val_data["Label"] > 1, 0, val_data["Label"]))[:val_X_tensor.shape[0]]
                elif mode == "Tri_stress_amuse":
                    tr_y_tensor = torch.tensor(np.where(tr_data["Label"] > 2, 0, tr_data["Label"]))
                    val_y_tensor = torch.tensor(np.where(val_data["Label"] > 2, 0, val_data["Label"]))
                    num_class = 3

                train_dataset = TensorDataset(tr_X_tensor, tr_y_tensor)
                val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model = MLP_Handcraft(input_shape=input_dim, num_class=num_class).to(device)
                # if opt == "NormWear":
                    # model = NormwearMLP(embedding_length=input_dim, num_class=num_class, normwear_weight_path="./src/NormWear/normwear_pretrain_ckpt.pth").to(device) 
                    # model.load_state_dict(torch.load("/diniuvol/jonah/Eric/run/20260609-184146_NormWear_Bin_stress/model_latest_S13.pth", map_location=device))                  

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
                
                best_val_loss = float('inf')
                best_val_acc = 0
                best_val_f1 = 0

                best_test_f1 = 0
                best_test_acc = 0
                for epoch in range(epochs):
                    print(f"Epoch {epoch+1}/{epochs}")
                    model.train()
                    train_loss = 0.0
                    
                    for batch_X, batch_y in tqdm(train_loader):
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()

                        logits = model(batch_X)
                        loss = criterion(logits, batch_y.long())

                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * batch_X.size(0)

                    train_loss /= len(train_loader.dataset)
                    
                    # 4. Validation & Metrics Phase
                    model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []

                    with torch.no_grad():
                        for batch_X, batch_y in tqdm(val_loader):
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            
                            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                                logits = model(batch_X)
                                loss = criterion(logits, batch_y.long())
                            val_loss += loss.item() * batch_X.size(0)
                            
                            # Convert logits to binary predictions
                            preds = logits.argmax(dim=1)
                            
                            all_preds.append(preds.cpu().numpy())
                            all_targets.append(batch_y.cpu().numpy())

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

                    val_f1 = f1_score(all_targets, all_preds, zero_division=0, average = "macro")
                    val_weighted_f1 = f1_score(all_targets, all_preds, zero_division=0, average = "weighted")
                    val_acc = accuracy_score(all_targets, all_preds)




                    if val_method == "tr_val_test":
                        test_loss = 0.0
                        test_preds = []
                        test_targets = []
                        with torch.no_grad():
                            for batch_X, batch_y in tqdm(test_loader):
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                                    logits = model(batch_X)
                                    loss = criterion(logits, batch_y.long())
                                test_loss += loss.item() * batch_X.size(0)
                                preds = logits.argmax(dim=1)
                                test_preds.append(preds.cpu().numpy())
                                test_targets.append(batch_y.cpu().numpy())
                        test_loss /= len(test_loader.dataset)
                        test_preds = np.concatenate(test_preds)
                        test_targets = np.concatenate(test_targets)
                        test_f1 = f1_score(test_targets, test_preds, zero_division=0, average = "macro")
                        test_weighted_f1 = f1_score(test_targets, test_preds, zero_division=0, average = "weighted")
                        test_acc = accuracy_score(test_targets, test_preds)
                        if test_f1 > best_test_f1:
                            best_test_f1 = test_f1
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc


                    
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
            print(f"Best test f1: {best_test_f1} | Best test acc: {best_test_acc}")

            writer.close()