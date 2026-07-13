import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import stable_baselines3
from NormWear.modules.normwear import NormWear, cwt_wrap
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class InfoLoggingCallback(BaseCallback):
    def __init__(self, info_key="prob", verbose=0):
        """
        :param info_key: The key in the info dictionary you want to log.
        """
        super().__init__(verbose)
        self.info_key = info_key

    def _on_step(self) -> bool:
        # Access the 'infos' list from the local variables
        infos = self.locals.get("infos", [])
        
        # Extract the probability from all environments (if using multiple)
        probs = []
        for info in infos:
            if self.info_key in info:
                probs.append(info[self.info_key])
        
        # If we found the probability, log it
        if probs:
            # Average the value in case of multiple vectorized environments
            mean_prob = np.mean(probs)
            
            # Record it to TensorBoard under a custom heading (e.g., "custom/prob")
            self.logger.record(f"custom/{self.info_key}", mean_prob)
            
        return True # Return True to continue training

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
        self.pretrained_model.load_state_dict(torch.load(f"/home/jonah/Eric/AutonomousBioKineticEventDetection/pretrained_normwear.pth"))
        self.pretrained_model.eval()

        wesad_root_dir = "/diniuvol/jonah/Eric/data/WESAD/"
        target_keys = set(["Base", "TSST", "Medi 1", "Fun", "Medi 2"])
        self.event_intervals = dict()
        for pid in self.train_ids: # + self.val_ids + self.test_ids:
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


    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        tr_pid = self.rng.choice(self.train_ids)
        self.env_data = StressDetectionEnv_WESAD_Events.data[StressDetectionEnv_WESAD_Events.data["PID"] == tr_pid]
        self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)

        events = list(self.event_intervals[tr_pid].keys())
        event_idx = self.rng.integers(0, len(self.event_intervals[tr_pid]))
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

        info = {"prob": stress_prob[0][1].item()}

        return obs, info

    def step(self, action):
        terminated = False
        reward = self.config.wait_penalty
        truncation = False
        info = {}
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


if __name__ == "__main__":
    SUBJECT_IDS = (
        [f"S{i}" for i in range(2, 12)] +
        [f"S{i}" for i in range(13, 18)]
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    name = f"{timestamp}_dqn_normwear_warmstart"

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"/diniuvol/jonah/Eric/run/{name}", 
        name_prefix="dqn_model",
    )

    env = StressDetectionEnv_WESAD_Events()

    policy_kwargs = dict(net_arch=[512, 256])
    model = stable_baselines3.DQN(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        exploration_initial_eps=0.8,
        exploration_final_eps=0.05,
        tensorboard_log=f"/diniuvol/jonah/Eric/run/{name}"
    )

    # If there's a pretrained model, load its weights into the current model for warmstart
    pretrained_weights = torch.load("/diniuvol/jonah/Eric/run/20260707-110445_dqn_warmstart_pretrain_mlp/model_latest.pth")
    sb3_expected_keys = model.q_net.state_dict().keys()
    print(f"SB3 expected keys: {sb3_expected_keys}  Pretrained keys: {pretrained_weights.keys()}")
    new_state_dict = {}
    for sb3_key, pretrained_weight in zip(sb3_expected_keys, pretrained_weights.values()):
        new_state_dict[sb3_key] = pretrained_weight

    model.q_net.load_state_dict(new_state_dict, strict=True)
    model.q_net_target.load_state_dict(model.q_net.state_dict())

    logging_callback = InfoLoggingCallback(info_key="prob")

    model.learn(total_timesteps=500000, callback=checkpoint_callback, log_interval=500)
    model.save(f"/diniuvol/jonah/Eric/run/{name}/model")

