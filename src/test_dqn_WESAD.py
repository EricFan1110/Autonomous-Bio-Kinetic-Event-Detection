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
from vit_pytorch import ViT
import stable_baselines3
import matplotlib.pyplot as plt

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
        return self.net(x)

class StressDetectionEnv_WESAD(gym.Env):
    data = pd.read_parquet("./data/WESAD/WESAD_Data_60s_6s_norm.parquet")

    def __init__(self, val_id, config = Config(), seed = None):
        super(StressDetectionEnv_WESAD, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)
        feature_size = int(((340 - config.past_window) // config.step_size + 1) * config.past_window)
        
        self.action_space = spaces.Discrete(2)
        self.feature_count = np.array(StressDetectionEnv_WESAD.data["Handcraft"].tolist()).shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=[feature_size + 2], dtype=np.float32)

        self.timestep = 0
        self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))
        self.val_pid = val_id

        
        self.pretrained_model = MLP(feature_size, 2)
        self.pretrained_model.load_state_dict(torch.load(f"/home/general/Documents/Eric/Autonomous-Bio-Kinetic-Event-Detection/run/20260519-153218_Handcraft_Bin_stress/model_best_f1_{val_id}.pth"))
        
        #weights = torch.load(f"/home/general/Documents/Eric/Autonomous-Bio-Kinetic-Event-Detection/run/20260519-113441_Handcraft_Bin_stress/model_best_f1_{val_id}.pth")
        # new_state_dict = {}
        # for sb3_key, pretrained_weight in zip(self.pretrained_model.state_dict().keys(), weights.values()):
        #     new_state_dict[sb3_key] = pretrained_weight
        # self.pretrained_model.load_state_dict(new_state_dict)
        self.pretrained_model.eval()

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        tr_pid = self.rng.choice(StressDetectionEnv_WESAD.data[StressDetectionEnv_WESAD.data["PID"] != self.val_pid]["PID"].unique())
        self.env_data = StressDetectionEnv_WESAD.data[StressDetectionEnv_WESAD.data["PID"] == tr_pid]
        self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)

        self.past_prediction_buffer = np.zeros((self.config.prediction_window, 1))

        self.stress_label = np.where(self.env_data["Label"] > 1, 0, self.env_data["Label"]).any()

        self.env_data_ = torch.tensor(np.array(self.env_data["Handcraft"].tolist(), dtype=np.float32)).unfold(1, self.config.past_window, self.config.step_size).flatten(1)
        obs = self.env_data_[self.timestep + self.config.past_window]
        
        time_label = self.env_data["Label"].iloc[self.timestep]

        with torch.no_grad():
            logits = self.pretrained_model(torch.tensor(obs).unsqueeze(0))
        stress_prob = F.softmax(logits, dim=1)

        # prob_reshaped = stress_prob.reshape(1,2)
        obs = np.concatenate((obs.unsqueeze(0), stress_prob), axis=1)

        info = {"prob": stress_prob[0][1].item() - time_label}

        return obs, info

    def step(self, action):
        terminated = False
        reward = 0.0
        truncation = False
        info = {}

        self.past_prediction_buffer[1:, :] = self.past_prediction_buffer[0:-1, :]
        self.past_prediction_buffer[0, :] = action

        # Detect Stress
        if action == 1:
            terminated = True

            if self.stress_label == 1:
                reward = self.config.r_true_positive - (self.timestep / (25 + self.timestep))
            else:
                reward = self.config.r_false_positive + (self.timestep / (25 + self.timestep))

        # Played till end of environment without detecting stress
        if self.timestep >= len(self.env_data) - self.config.past_window - 2:
            terminated = True
            if action == 0:
                if self.stress_label == 0:
                    reward = self.config.r_true_negative
                else:
                    reward = self.config.r_false_negative - (self.timestep / (25 + self.timestep))

        self.timestep += 1
        obs = self.env_data_[self.timestep + self.config.past_window]

        if self.timestep >= self.config.env_length:
            terminated = True

        # time_label = 1
        # if self.env_data[["Baseline", "Amusement", "Meditation"]].iloc[self.timestep].sum(axis=0) > self.env_data['Stress'].iloc[self.timestep].item():
        #     time_label = 0

        with torch.no_grad():
            logits = self.pretrained_model(torch.tensor(obs).unsqueeze(0))
        stress_prob = F.softmax(logits, dim=1)
        # prob_reshaped = stress_prob.reshape(1,2)
        obs = np.concatenate((obs.unsqueeze(0), stress_prob), axis=1)
        info["prob"] = stress_prob[0][1].item() # - time_label

        return obs, reward, terminated, truncation, info

if __name__ == "__main__":
    SUBJECT_IDS = (
        [f"S{i}" for i in range(2, 12)] +
        [f"S{i}" for i in range(13, 18)]
    )

    subject_id = SUBJECT_IDS[14]

    for subject_id in SUBJECT_IDS:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"run/20260414-165939_dqn_PID_with_pretrained_pred/{subject_id}/"
        writer = SummaryWriter(file_name + f"{timestamp}")

        model = stable_baselines3.DQN.load(f"{file_name}"+ "model")

        env = StressDetectionEnv_WESAD(subject_id, seed = 2)
        step = 0
        obs, info = env.reset()

        actions = []
        labels = []
        preds = []
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            writer.add_scalar("val/Action", action, step)
            writer.add_scalar("val/Label", info["time_label"], step)
            writer.add_scalar("val/Prediction", info["prob"], step)

            actions.append(action)
            labels.append(info["time_label"])
            preds.append(info["prob"])

            step += 1

            if terminated:
                break
        
        plt.plot(actions, label='Actions', color='blue')
        plt.plot(labels, label='Labels', color='red')

        plt.title(f'Validation {subject_id}')
        plt.xlabel('Timestep(min)')
        plt.ylabel('Prob')

        plt.legend()
        plt.savefig(file_name + f'val_{subject_id}.png')
        plt.close()

        plt.plot(actions, label='Actions', color='blue')
        plt.plot(labels, label='Labels', color='red')
        plt.plot(preds, label='Predictions', color='green')

        plt.title(f'Validation {subject_id}')
        plt.xlabel('Timestep(min)')
        plt.ylabel('Prob')

        plt.legend()
        plt.savefig(file_name + f'val_{subject_id}_w_pred.png')
        plt.close()



