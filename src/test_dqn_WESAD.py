import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import stable_baselines3
import matplotlib.pyplot as plt
from NormWear.modules.normwear import NormWear, cwt_wrap
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score
)

class Config:
    past_window = 40
    step_size = 5
    env_length = 150
    early_detection_window = 10
    late_detection_window = 2
    prediction_window = 5

    r_true_positive = 1.0
    r_false_positive = -1.0
    r_true_negative = 1.0
    r_false_negative = -1.0

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
class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(DQN, self).__init__()
        
        # Calculate the flattened size of the observation
        flat_size = np.prod(obs_shape)
        
        # Simple Multi-Layer Perceptron (MLP)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = self.network(x)

        return x

class MLP(nn.Module):
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
        return self.net(x), x

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

# class StressDetectionEnv_WESAD(gym.Env):
#     data = pd.read_parquet("./data/WESAD/WESAD_Data_60s_6s_norm.parquet")

#     def __init__(self, val_id, config = Config(), seed = None):
#         super(StressDetectionEnv_WESAD, self).__init__()
#         self.config = config
#         self.rng = np.random.default_rng(seed=seed)
#         feature_size = int(((340 - config.past_window) // config.step_size + 1) * config.past_window)
        
#         self.action_space = spaces.Discrete(2)
#         self.feature_count = np.array(StressDetectionEnv_WESAD.data["Handcraft"].tolist()).shape[1]
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
#                                             shape=[feature_size + 2], dtype=np.float32)

#         self.timestep = 0
#         self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))
#         self.val_pid = val_id

        
#         self.pretrained_model = MLP(feature_size, 2)
#         self.pretrained_model.load_state_dict(torch.load(f"/home/general/Documents/Eric/Autonomous-Bio-Kinetic-Event-Detection/run/20260519-153218_Handcraft_Bin_stress/model_best_f1_{val_id}.pth"))
        
#         #weights = torch.load(f"/home/general/Documents/Eric/Autonomous-Bio-Kinetic-Event-Detection/run/20260519-113441_Handcraft_Bin_stress/model_best_f1_{val_id}.pth")
#         # new_state_dict = {}
#         # for sb3_key, pretrained_weight in zip(self.pretrained_model.state_dict().keys(), weights.values()):
#         #     new_state_dict[sb3_key] = pretrained_weight
#         # self.pretrained_model.load_state_dict(new_state_dict)
#         self.pretrained_model.eval()

#     def reset(self, seed = None, options = None):
#         if seed is not None:
#             self.rng = np.random.default_rng(seed=seed)
#         self.timestep = 0

#         tr_pid = self.rng.choice(StressDetectionEnv_WESAD.data[StressDetectionEnv_WESAD.data["PID"] != self.val_pid]["PID"].unique())
#         self.env_data = StressDetectionEnv_WESAD.data[StressDetectionEnv_WESAD.data["PID"] == tr_pid]
#         self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)

#         self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))

#         self.stress_label = np.where(self.env_data["Label"] > 1, 0, self.env_data["Label"]).any()

#         self.env_data_ = torch.tensor(np.array(self.env_data["Handcraft"].tolist(), dtype=np.float32)).unfold(1, self.config.past_window, self.config.step_size).flatten(1)
#         obs = self.env_data_[self.timestep + self.config.past_window]
        
#         time_label = self.env_data["Label"].iloc[self.timestep]

#         with torch.no_grad():
#             logits = self.pretrained_model(torch.tensor(obs).unsqueeze(0))
#         stress_prob = F.softmax(logits, dim=1)

#         # prob_reshaped = stress_prob.reshape(1,2)
#         obs = np.concatenate((obs.unsqueeze(0), stress_prob), axis=1)

#         info = {"prob": stress_prob[0][1].item() - time_label}

#         return obs, info

#     def step(self, action):
#         terminated = False
#         reward = 0.0
#         truncation = False
#         info = {}

#         self.past_prediction_buffer[1:, :] = self.past_prediction_buffer[0:-1, :]
#         self.past_prediction_buffer[0, :] = action

#         # Detect Stress
#         if action == 1:
#             terminated = True

#             if self.stress_label == 1:
#                 reward = self.config.r_true_positive - (self.timestep / (25 + self.timestep))
#             else:
#                 reward = self.config.r_false_positive + (self.timestep / (25 + self.timestep))

#         # Played till end of environment without detecting stress
#         if self.timestep >= len(self.env_data) - self.config.past_window - 2:
#             terminated = True
#             if action == 0:
#                 if self.stress_label == 0:
#                     reward = self.config.r_true_negative
#                 else:
#                     reward = self.config.r_false_negative - (self.timestep / (25 + self.timestep))

#         self.timestep += 1
#         obs = self.env_data_[self.timestep + self.config.past_window]

#         if self.timestep >= self.config.env_length:
#             terminated = True

#         # time_label = 1
#         # if self.env_data[["Baseline", "Amusement", "Meditation"]].iloc[self.timestep].sum(axis=0) > self.env_data['Stress'].iloc[self.timestep].item():
#         #     time_label = 0

#         with torch.no_grad():
#             logits = self.pretrained_model(torch.tensor(obs).unsqueeze(0))
#         stress_prob = F.softmax(logits, dim=1)
#         # prob_reshaped = stress_prob.reshape(1,2)
#         obs = np.concatenate((obs.unsqueeze(0), stress_prob), axis=1)
#         info["prob"] = stress_prob[0][1].item() # - time_label

#         return obs, reward, terminated, truncation, info

class StressDetectionEnv_WESAD_Events(gym.Env):
    data = pd.read_parquet("/diniuvol/jonah/Eric/data/WESAD/WESAD_Data_6s_6s.parquet")

    def __init__(self, train_end = False, config = Config_Events(), seed = None):
        super(StressDetectionEnv_WESAD_Events, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)

        self.train_end = train_end
        
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
        self.pretrained_model.load_state_dict(torch.load(f"/home/jonah/Eric/AutonomousBioKineticEventDetection/model_best_f1_S13.pth"))
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
            skip_rows = self.env_data.iloc[20:] if not self.train_end else self.env_data.iloc[:-20]
            idx = -1 if self.train_end else 0
            self.stress_label_idx = skip_rows[skip_rows["Label"] == 1].index[idx] - self.env_data.index[0]

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

        info = {"pred": stress_prob.argmax(dim=1).item(), "done": False}

        return obs, info

    def step(self, action):
        terminated = False
        reward = self.config.wait_penalty
        truncation = False
        info = {"done": False}
        self.timestep += 1

        # Detect Stress
        if action == 1:
            terminated = True

            if self.stress_label == 1:
                reward = self.config.correct_base + max(self.config.correct_lead * ((-1) * abs(self.stress_label_idx - 1 - self.timestep) + self.stress_label_idx) / max(1, self.stress_label_idx - 1), 0)
            else:
                reward = self.config.wrong_base + self.config.wrong_lead * (self.stress_label_idx - 1 - self.timestep) / max(1, self.stress_label_idx - 1)

        # Played till end of environment without detecting stress
        if self.timestep >= len(self.env_data_) - 1:
            info['done'] = True
            terminated = True
            if action == 0:
                if self.stress_label == 0:
                    reward = self.config.correct_base
                else:
                    reward = self.config.no_commit

        with torch.amp.autocast(device_type="cuda"):
            with torch.no_grad():
                logits, obs = self.pretrained_model(torch.tensor(self.env_data_[self.timestep]).detach().clone().unsqueeze(0).to(torch.float16).cuda())
        stress_prob = F.softmax(logits, dim=1)

        self.past_prediction_buffer[1:, :] = self.past_prediction_buffer[0:-1, :]
        self.past_prediction_buffer[0, :] = stress_prob[0][1].item()

        obs = np.concatenate((obs.cpu().numpy(), np.expand_dims(self._get_past_features_metrics(), 0)), axis=1)

        info["prob"] = stress_prob[0][1].item()

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
    
class StressDetectionEnv_WESAD_Events_Handcraft(gym.Env):
    data = pd.read_parquet("/diniuvol/jonah/Eric/data/WESAD/WESAD_Data_60s_6s_norm.parquet")

    def __init__(self, config = Config_Events(), seed = None):
        super(StressDetectionEnv_WESAD_Events_Handcraft, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)
        
        self.action_space = spaces.Discrete(2)
        self.feature_count = 4608
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=[4608*3 + 5 + 2], dtype=np.float32)

        self.timestep = 0
        self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))
        self.train_ids = ['S3','S4','S5','S9','S10','S11','S14','S15','S16','S17']
        self.val_ids = ['S7','S8']
        self.test_ids = ['S2','S6','S13']

        self.pretrained_model = MLP(((340 - 40) // 5 + 1) * 40, 2).cuda()
        self.pretrained_model.load_state_dict(torch.load(f"/home/jonah/Eric/AutonomousBioKineticEventDetection/run/20260602-213916_Handcraft_Bin_stress/model_best_f1_0.pth"))
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


    def reset(self, subject_id, event_idx, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        tr_pid = subject_id
        self.env_data = StressDetectionEnv_WESAD_Events_Handcraft.data[StressDetectionEnv_WESAD_Events_Handcraft.data["PID"] == tr_pid]
        self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)

        events = list(self.event_intervals[tr_pid].keys())
        event = events[event_idx]
        start_time, end_time = self.event_intervals[tr_pid][event]

        if event_idx > 0:
            _, prev_end = self.event_intervals[tr_pid][events[event_idx - 1]]
        else:
            prev_end = 0.0

        if event_idx < len(events) - 1:
            next_start, _ = self.event_intervals[tr_pid][events[event_idx + 1]]
        else:
            next_start = 10000.0

        self.env_data = self.env_data[self.env_data['Time'].between(int(prev_end * 60 - 54), int(next_start * 60))]

        self.stress_label = event == 'TSST'
        self.stress_label_idx = len(self.env_data)
        if self.stress_label:
            skip_rows = self.env_data.iloc[20:]
            self.stress_label_idx = skip_rows[skip_rows["Label"] == 1].index[0] - self.env_data.index[0]

        self.env_data_ = torch.tensor(np.array(self.env_data["Handcraft"].tolist(), dtype=np.float32))
        self.env_data_ = self.env_data_.unfold(1, 40, 5).flatten(1)
        # self.env_data_ = self.env_data_.transpose(1, 3).transpose(2,3)
        
        with torch.amp.autocast(device_type="cuda"):
            with torch.no_grad():
                logits, obs = self.pretrained_model(torch.tensor(self.env_data_[self.timestep]).detach().clone().unsqueeze(0).to(torch.float16).cuda())
        stress_prob = F.softmax(logits, dim=1)

        self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))
        self.past_prediction_buffer[0, :] = stress_prob[0][1].item()

        obs = np.concatenate((obs.cpu().numpy(), np.expand_dims(self._get_past_features_metrics(), 0)), axis=1)

        info = {"prob": stress_prob[0][1].item(), "time_label": self.env_data["Label"].iloc[self.timestep + 9].item(), 'done': False}

        return obs, info

    def step(self, action):
        terminated = False
        reward = self.config.wait_penalty
        truncation = False
        info = {'done': False}
        self.timestep += 1

        # Detect Stress
        if action == 1:
            terminated = True

            if self.stress_label == 1:
                reward = self.config.correct_base + self.config.correct_lead * (self.stress_label_idx - 1 - self.timestep) / max(1, self.stress_label_idx - 1)
            else:
                reward = self.config.wrong_base + self.config.wrong_lead * (self.stress_label_idx - 1 - self.timestep) / max(1, self.stress_label_idx - 1)

        # Played till end of environment without detecting stress
        if self.timestep >= len(self.env_data_) - 1:
            info['done'] = True
            terminated = True
            if action == 0:
                if self.stress_label == 0:
                    reward = self.config.correct_base
                else:
                    reward = self.config.no_commit

        with torch.amp.autocast(device_type="cuda"):
            with torch.no_grad():
                logits, obs = self.pretrained_model(torch.tensor(self.env_data_[self.timestep]).detach().clone().unsqueeze(0).to(torch.float16).cuda())
        stress_prob = F.softmax(logits, dim=1)

        self.past_prediction_buffer[1:, :] = self.past_prediction_buffer[0:-1, :]
        self.past_prediction_buffer[0, :] = stress_prob[0][1].item()

        obs = np.concatenate((obs.cpu().numpy(), np.expand_dims(self._get_past_features_metrics(), 0)), axis=1)

        info["prob"] = stress_prob[0][1].item()
        # info['time_label'] = self.env_data["Label"].iloc[self.timestep + 9].item()

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
    SUBJECT_IDS = (
        [f"S{i}" for i in range(2, 12)] +
        [f"S{i}" for i in range(13, 18)]
    )
    train_ids = ['S3','S4','S5','S9','S10','S11','S14','S15','S16','S17']
    val_ids = ['S7','S8']
    test_ids = ['S2','S6','S13']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "Normwear"
    env_class = StressDetectionEnv_WESAD_Events

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"/diniuvol/jonah/Eric/run/20260630-202722_dqn_Handcraft/"
    
    # pred_start = stable_baselines3.DQN.load(f"/diniuvol/jonah/Eric/run/20260630-215829_dqn_Handcraft/model.zip", device = device)
    # pred_end = pred_start

    pred_start = stable_baselines3.DQN.load(f"/home/jonah/Eric/AutonomousBioKineticEventDetection/dqn_normwear_start.zip", device = device)
    pred_end = stable_baselines3.DQN.load(f"/home/jonah/Eric/AutonomousBioKineticEventDetection/dqn_normwear_end.zip", device = device)
    leads = []

    val_pred = []
    val_actual = []
    for subject_id in val_ids:
        env = env_class()

        for i in range(5):
            step = 0
            obs, info = env.reset(subject_id, i)
            b_pred_start = True
            model = pred_start

            actions = []
            labels = env.env_data["Label"].tolist()
            preds = []
            predicted_stress = 0
            while True:
                if b_pred_start:
                    action, _states = pred_start.predict(obs, deterministic=True)
                else:
                    action, _states = pred_end.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                actions.append(action)
                preds.append(info["prob"])
                if action == 1 and b_pred_start:
                    predicted_stress = 1

                step += 1

                if (terminated and not b_pred_start) or info['done']:
                    break
                
                if terminated and b_pred_start:
                    b_pred_start = False
                    model = pred_end
            val_actual.append(env.stress_label)
            if env.stress_label == 1:
                leads.append(env.stress_label_idx - step)
            val_pred.append(predicted_stress)
            print(actions, subject_id, i)
            plt.plot(actions, label='Actions', color='blue')
            plt.plot(labels, label='Labels', color='red')

            plt.title(f'{subject_id}')
            plt.xlabel('Timestep(6s)')
            plt.ylabel('Prob')

            plt.legend()
            plt.savefig(f'val_{subject_id}_event_{i}_{mode}.png')
            plt.close()

            plt.plot(actions, label='Actions', color='blue')
            plt.plot(labels, label='Labels', color='red')
            plt.plot(preds, label='Predictions', color='green')

            plt.title(f'{subject_id}')
            plt.xlabel('Timestep(6s)')
            plt.ylabel('Prob')

            plt.legend()
            plt.savefig(f'val_{subject_id}_event_{i}_{mode}_w_pred.png')
            plt.close()
    val_acc = accuracy_score(val_actual, val_pred)
    val_f1_macro = f1_score(val_actual, val_pred, average = "macro")
    val_f1_weighted = f1_score(val_actual, val_pred, average = "weighted")
    print(val_actual, val_pred)
    print(f"Val acc: {val_acc}  Val F1 Macro: {val_f1_macro}   Val F1 Weighted: {val_f1_weighted}")
    print(leads)
    leads = []
    
    test_pred = []
    test_actual = []
    for subject_id in test_ids:
        env = env_class()

        for i in range(5):
            step = 0
            obs, info = env.reset(subject_id, i)
            b_pred_start = True
            model = pred_start
            predicted_stress = 0

            actions = []
            labels = env.env_data["Label"].tolist()
            preds = []
            while True:
                if b_pred_start:
                    action, _states = pred_start.predict(obs, deterministic=True)
                else:
                    action, _states = pred_end.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                actions.append(action)
                preds.append(info["prob"])

                step += 1
                if action == 1 and b_pred_start:
                    predicted_stress = 1

                if (terminated and not b_pred_start) or info['done']:
                    break

                if terminated and b_pred_start:
                    b_pred_start = False
                    model = pred_end
            test_actual.append(env.stress_label)
            if env.stress_label == 1:
                leads.append(env.stress_label_idx - step)
            test_pred.append(predicted_stress)
            
            plt.plot(actions, label='Actions', color='blue')
            plt.plot(labels, label='Labels', color='red')

            plt.title(f'{subject_id}')
            plt.xlabel('Timestep(6s)')
            plt.ylabel('Prob')

            plt.legend()
            plt.savefig(f'test_{subject_id}_event_{i}_{mode}.png')
            plt.close()

            plt.plot(actions, label='Actions', color='blue')
            plt.plot(labels, label='Labels', color='red')
            plt.plot(preds, label='Predictions', color='green')

            plt.title(f'{subject_id}')
            plt.xlabel('Timestep(6s)')
            plt.ylabel('Prob')

            plt.legend()
            plt.savefig(f'test_{subject_id}_event_{i}_{mode}_w_pred.png')
            plt.close()
    test_acc = accuracy_score(test_actual, test_pred)
    test_f1_macro = f1_score(test_actual, test_pred, average = "macro")
    test_f1_weighted = f1_score(test_actual, test_pred, average = "weighted")
    print(test_actual, test_pred)
    print(f"Test acc: {test_acc}  Test F1 Macro: {test_f1_macro}   Test F1 Weighted: {test_f1_weighted}")
    
    print(leads)


