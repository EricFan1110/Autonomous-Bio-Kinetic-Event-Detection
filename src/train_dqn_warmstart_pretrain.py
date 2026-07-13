import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import stable_baselines3
from NormWear.modules.normwear import NormWear, cwt_wrap
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score
)
import os

class Config_Events:
    past_window = 10
    step_size = 1
    prediction_window = 5

    wait_penalty = -0.002
    correct_base = 1.0
    correct_lead = 0.35
    wrong_base = -1.5
    wrong_lead = -0.75
    no_commit = -0.20

@torch.compile
class MLP(nn.Module):
    def __init__(self, input_shape, num_class=2):
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

@torch.compile
class NormwearMLP(nn.Module):
    def __init__(self, embedding_length, num_class=2, normwear_weight_path=None):
        super().__init__()

        if normwear_weight_path is None:
            self.normwear = NormWear(img_size=(387,65), patch_size=(9,5),mask_scheme='random',mask_prob=0.8,use_cwt=True,nvar=4, comb_freq=False)
            self.normwear.train()
        else:
            self.normwear = NormWear(img_size=(387,65), patch_size=(9,5),mask_scheme='random',mask_prob=0.8,use_cwt=True,nvar=4, comb_freq=False)
            self.normwear.load_state_dict(torch.load(normwear_weight_path, map_location=torch.device('cpu')))
            print(f"Loaded NormWear weights from {normwear_weight_path}")
            self.normwear.train()
        
        self.embed_1 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.embed_2 = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_head = nn.Linear(256, num_class)

        # self.net = nn.Sequential(
        #     nn.Linear(embedding_length * 3, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_class), 
        # )

    def forward(self, x):
        bn, window_size, nvar, L = x.shape
        per_window_features = []
        for window in range(window_size):
            x_window = x[:, window, :, :] # [batch_size, sensors, seq_len]

            cwt_res = cwt_wrap(x_window.reshape(bn*nvar, L), 0.1, 64)
            _, n_, new_L, n_scale = cwt_res.shape
            spec = cwt_res.view(bn, nvar, n_, new_L, n_scale).to(x.dtype)

            embedding = self.normwear.get_signal_embedding(spec.to(x.device), hidden_out=False, device=x.device)
            embedding = embedding.mean(dim=2)
            per_window_features.append(embedding)
        per_window_features = torch.stack(per_window_features, dim=1).to(x.device)
        mean_features = per_window_features.mean(dim=1)
        std_features = per_window_features.std(dim=1, unbiased=False)
        slope_features = self.temporal_slope(per_window_features)

        combined_features = torch.stack([mean_features, std_features, slope_features], dim=1)
        embed = self.embed_1(combined_features.flatten(1,2))
        embed = self.embed_2(embed.flatten(1))

        return self.output_head(embed), embed

    def temporal_slope(self, features: torch.Tensor) -> torch.Tensor:
        num_windows = features.shape[1]
        positions = torch.arange(num_windows, device=features.device, dtype=features.dtype)
        centered = positions - positions.mean()
        denom = centered.pow(2).sum().clamp_min(1e-6)
        centered = centered.view(1, num_windows, 1, 1)
        _mean_feature = features.mean(dim=1, keepdim=True)
        numer = ((features - _mean_feature) * centered).sum(dim=1)
        return numer / denom
    
class StressDetectionEnv_WESAD_Events(gym.Env):
    data = pd.read_parquet("/diniuvol/jonah/Eric/data/WESAD/WESAD_Data_6s_6s.parquet")

    def __init__(self, config = Config_Events(), seed = None):
        super(StressDetectionEnv_WESAD_Events, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)
        
        self.action_space = spaces.Discrete(2)
        self.feature_count = 4608
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=[256 + 1 + 9], dtype=np.float32)

        self.timestep = 0
        self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))
        self.train_ids = ['S3','S4','S5','S9','S10','S11','S14','S15','S16','S17']
        self.val_ids = ['S7','S8']
        self.test_ids = ['S2','S6','S13']

        self.pretrained_model = NormwearMLP(4608, 2).cuda()
        self.pretrained_model.load_state_dict(torch.load(f"/home/jonah/Eric/AutonomousBioKineticEventDetection/pretrained_normwear.pth"))
        self.pretrained_model.eval()

        wesad_root_dir = "/diniuvol/jonah/Eric/data/WESAD/"
        target_keys = set(["Base", "TSST", "Medi 1", "Fun", "Medi 2"])
        self.event_intervals = dict()
        for pid in self.train_ids + self.val_ids + self.test_ids:
            events_df = pd.read_csv(f"{wesad_root_dir}{pid}/{pid}_quest.csv", delimiter=';')
            order_row = events_df.iloc[0]
            start_row = events_df.iloc[1]
            end_row = events_df.iloc[2]
            intervals = dict()
            
            for col in events_df.columns[1:]:
                key_name = str(order_row[col]).strip()
                if key_name not in target_keys:
                    continue
                start_val = float(start_row[col])
                end_val = float(end_row[col])
                intervals[key_name] = (start_val, end_val)

            self.event_intervals[pid] = intervals


    def reset(self, train_id, event_, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        tr_pid = train_id
        self.env_data = StressDetectionEnv_WESAD_Events.data[StressDetectionEnv_WESAD_Events.data["PID"] == tr_pid]
        self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)

        events = list(self.event_intervals[tr_pid].keys())
        event_idx = event_
        event = events[event_idx]
        start_time, end_time = self.event_intervals[tr_pid][event]

        if event_idx > 0:
            _, prev_end = self.event_intervals[tr_pid][events[event_idx - 1]]
        else:
            prev_end = 0.0

        # if event_idx < len(events) - 1:
        #     next_start, _ = self.event_intervals[tr_pid][events[event_idx + 1]]
        # else:
        #     next_start = self.env_data["Time"].iloc[-1]

        self.env_data = self.env_data[self.env_data['Time'].between(int(prev_end * 60 - 54), int(end_time * 60))]
        self.stress_label = event == 'TSST'
        self.stress_label_idx = len(self.env_data)
        if self.stress_label:
            skip_rows = self.env_data.iloc[20:]
            self.stress_label_idx = skip_rows[skip_rows["Label"] == 1].index[-1] - self.env_data.index[0]

        self.env_data_ = torch.tensor(np.array(self.env_data["Normwear_Input"].tolist(), dtype=np.float32))
        self.env_data_ = self.env_data_.reshape(-1, 6, 390).unfold(0, 10, 1)
        self.env_data_ = self.env_data_.transpose(1, 3).transpose(2,3)
        
        with torch.amp.autocast(device_type="cuda"):
            with torch.no_grad():
                logits, obs = self.pretrained_model(torch.tensor(self.env_data_[self.timestep]).detach().clone().unsqueeze(0).to(torch.float16).cuda())
        stress_prob = F.softmax(logits, dim=1)

        self.past_prediction_buffer = np.zeros((len(self.env_data_), 1))
        self.past_prediction_buffer[0, :] = stress_prob[0][1].item()

        obs = np.concatenate((obs.cpu().numpy(), np.expand_dims(self._get_past_features_metrics(), 0)), axis=1)

        info = {"pred": stress_prob.argmax(dim=1).item()}

        return obs, info

    def step(self, action):
        terminated = False
        reward = self.config.wait_penalty
        truncation = False
        info = {}
        self.timestep += 1

        # Played till end of environment without detecting stress
        if self.timestep >= len(self.env_data_) - 1:
            terminated = True

        with torch.amp.autocast(device_type="cuda"):
            with torch.no_grad():
                logits, obs = self.pretrained_model(torch.tensor(self.env_data_[self.timestep]).detach().clone().unsqueeze(0).to(torch.float16).cuda())
        stress_prob = F.softmax(logits, dim=1)

        self.past_prediction_buffer[1:, :] = self.past_prediction_buffer[0:-1, :]
        self.past_prediction_buffer[0, :] = stress_prob[0][1].item()

        obs = np.concatenate((obs.cpu().numpy(), np.expand_dims(self._get_past_features_metrics(), 0)), axis=1)

        info["pred"] = stress_prob.argmax(dim=1).item()

        return obs, reward, terminated, truncation, info
    def _get_past_features_metrics(self):
        buf = self.past_prediction_buffer.flatten()
        mean_10 = np.mean(buf[:min(10, self.timestep + 1)])
        std_10 = np.std(buf[:min(10, self.timestep + 1)])
        delta = buf[0] - buf[1]
        y_10 = buf[:min(10, 2)][::-1] 
        x_10 = np.arange(len(y_10))
        slope_10 = np.polyfit(x_10, y_10, 1)[0]
        span = min(10, self.timestep)
        alpha = 2 / (span + 1)
        chronological_buf = buf[:self.timestep + 1][::-1] 

        ema_array = np.zeros_like(chronological_buf)
        ema_array[0] = chronological_buf[0]
        for i in range(1, len(chronological_buf)):
            ema_array[i] = alpha * chronological_buf[i] + (1 - alpha) * ema_array[i-1]
        ema = ema_array[-1]

        persist_3 = float(1.0 - np.prod(1.0 - buf[0:min(3, self.timestep + 1)]))

        mean_all = np.mean([buf[:self.timestep + 1]])
        std_all = np.std(buf[:self.timestep + 1])

        y_all = buf[:max(self.timestep + 1, 2)][::-1]
        x_all = np.arange(len(y_all))
        slope_all = np.polyfit(x_all, y_all, 1)[0]
        return np.array([buf[0], mean_10, std_10, delta, slope_10, ema, persist_3, mean_all, std_all, slope_all])

if __name__ == "__main__":
    epochs = 80
    batch_size=64
    lr=0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ids = ['S3','S4','S5','S9','S10','S11','S14','S15','S16','S17']
    val_ids = ['S7','S8']
    test_ids = ['S2','S6','S13']

    env = StressDetectionEnv_WESAD_Events()

    # Commented out code below is for generating the dataset and saving it to a file.
    # train_X = []
    # train_y = []

    # val_X = []
    # val_y = []

    # test_X = []
    # test_y = []

    # for event_id in range(5):
    #     for tr_id in train_ids:
    #         obs, info = env.reset(tr_id, event_id)
    #         print(f"Reset environment for {tr_id}. Event: {event_id}")
    #         train_X.append(obs)
    #         train_y.append(info['pred'])
    #         done = False
    #         while not done:
    #             obs, reward, terminated, truncation, info = env.step(0)
    #             done = terminated or truncation

    #             train_X.append(obs)
    #             train_y.append(info['pred'])

    # for event_id in range(5):
    #     for val_id in val_ids:
    #         obs, info = env.reset(val_id, event_id)
    #         print(f"Reset environment for {val_id}. Event: {event_id}")
    #         val_X.append(obs)
    #         val_y.append(info['pred'])
    #         done = False
    #         while not done:
    #             obs, reward, terminated, truncation, info = env.step(0)
    #             done = terminated or truncation
                
    #             val_X.append(obs)
    #             val_y.append(info['pred'])
    
    # for event_id in range(5):
    #     for test_id in test_ids:
    #         obs, info = env.reset(test_id, event_id)
    #         print(f"Reset environment for {test_id}. Event: {event_id}")
    #         test_X.append(obs)
    #         test_y.append(info['pred'])
    #         done = False
    #         while not done:
    #             obs, reward, terminated, truncation, info = env.step(0)
    #             done = terminated or truncation
                
    #             test_X.append(obs)
    #             test_y.append(info['pred'])
    
    # train_X_tensor = torch.tensor(np.vstack(train_X), dtype=torch.float32)
    # train_y_tensor = torch.tensor(np.vstack(train_y), dtype=torch.int32)
    # print(f"Num 0: {np.sum(train_y_tensor.numpy() == 0)}, Num 1: {np.sum(train_y_tensor.numpy() == 1)}")
    # val_X_tensor = torch.tensor(np.vstack(val_X), dtype=torch.float32)
    # val_y_tensor = torch.tensor(np.vstack(val_y), dtype=torch.int32)
    # test_X_tensor = torch.tensor(np.vstack(test_X), dtype=torch.float32)
    # test_y_tensor = torch.tensor(np.vstack(test_y), dtype=torch.int32)

    # tensordict = {
    #     "train_X": train_X_tensor,
    #     "train_y": train_y_tensor,
    #     "val_X": val_X_tensor,
    #     "val_y": val_y_tensor,
    #     "test_X": test_X_tensor,
    #     "test_y": test_y_tensor
    # }
    # torch.save(tensordict, f"/diniuvol/jonah/Eric/data/WESAD/dqn_warmstart_pretrain_data.pt")

    tensordict = torch.load(f"/diniuvol/jonah/Eric/data/WESAD/dqn_warmstart_pretrain_data.pt")
    train_X_tensor = tensordict["train_X"]
    train_y_tensor = tensordict["train_y"].squeeze()
    val_X_tensor = tensordict["val_X"]
    val_y_tensor = tensordict["val_y"].squeeze()
    test_X_tensor = tensordict["test_X"]
    test_y_tensor = tensordict["test_y"].squeeze()

    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights = torch.tensor([0.5651, 4.3414], dtype=torch.float32).to(device)
    model = MLP(266).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_f1 = 0
    best_val_weighted_f1 = 0

    best_test_loss = float('inf')
    best_test_f1 = 0
    best_test_weighted_f1 = 0
    best_test_acc = 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'/diniuvol/jonah/Eric/run/{timestamp}_dqn_warmstart_pretrain_mlp'
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        test_loss = 0.0

        val_preds = []
        val_targets = []
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
                logits = model(batch_X)
                loss = criterion(logits, batch_y.long())
                val_loss += loss.item() * batch_X.size(0)
                
                # Convert logits to binary predictions
                preds = logits.argmax(dim=1)
                
                val_preds.append(preds.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

            for batch_X, batch_y in tqdm(test_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y.long())
                test_loss += loss.item() * batch_X.size(0)
                
                # Convert logits to binary predictions
                preds = logits.argmax(dim=1)
                
                test_preds.append(preds.cpu().numpy())
                test_targets.append(batch_y.cpu().numpy())
        

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0, average = "macro")
        val_weighted_f1 = f1_score(val_targets, val_preds, zero_division=0, average = "weighted")
        val_acc = accuracy_score(val_targets, val_preds)

        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_f1 = f1_score(test_targets, test_preds, zero_division=0, average = "macro")
        test_weighted_f1 = f1_score(test_targets, test_preds, zero_division=0, average = "weighted")
        test_acc = accuracy_score(test_targets, test_preds)
    
        val_loss /= len(val_loader.dataset)
        test_loss /= len(test_loader.dataset)

        writer.add_scalar(f'Loss/Train', train_loss, epoch)
        writer.add_scalar(f'Loss/Validation', val_loss, epoch)
        writer.add_scalar(f'Loss/Test', test_loss, epoch)
        writer.add_scalar(f'F1/Val', val_f1, epoch)
        writer.add_scalar(f'F1/Test', test_f1, epoch)
        writer.add_scalar(f'WeightedF1/Val', val_weighted_f1, epoch)
        writer.add_scalar(f'WeightedF1/Test', test_weighted_f1, epoch)
        writer.add_scalar(f'ACC/Val', val_acc, epoch)
        writer.add_scalar(f'ACC/Test', test_acc, epoch)

        torch.save(model.state_dict(), os.path.join(log_dir, f'model_latest.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_val_loss.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_val_acc.pth'))
        
        if val_weighted_f1 > best_val_weighted_f1:
            best_val_weighted_f1 = val_weighted_f1
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_val_weighted_f1.pth'))
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_val_macro_f1.pth'))
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_test_loss.pth'))
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_test_acc.pth'))
        
        if test_weighted_f1 > best_test_weighted_f1:
            best_test_weighted_f1 = test_weighted_f1
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_test_weighted_f1.pth'))
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_best_test_macro_f1.pth'))
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}, Val Weighted F1: {val_weighted_f1:.4f}, Test Weighted F1: {test_weighted_f1:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()
