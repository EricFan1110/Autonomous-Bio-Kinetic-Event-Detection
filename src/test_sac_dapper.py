import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from vit_pytorch import ViT
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime

class Config:
    past_window = 60
    env_length = 373
    prediction_window = 12

    prediction_std_threshold = 0.05
    rew_l2_sensitivity = 3

    rew_mode = "dense"
    rew_func = "timed"
    remove_dup = True
    decide_terminate = True
    decide_terminate_threshold = 0.9

validation_subjects = [1005, 1014, 1019, 1021, 1028, 1030, 2001, 2006, 2010, 2017, 2021, 2025, 3003, 3008, 3011, 3022, 3023, 3117]

class StressDetectionEnv(gym.Env):
    data = pd.read_parquet("./data/Dapper/dapper_dqn_10s_step_62min_norm.parquet")
    data = data[data["PID"].isin(validation_subjects)]
    data['Time'] = pd.to_datetime(data['Time'])

    def __init__(self, config = Config(), seed = None):
        super(StressDetectionEnv, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)

        if config.decide_terminate:
            self.action_space = spaces.Box(low = np.array([1, 1, 0]), high = np.array([5, 5, 1]), shape = (3,), dtype = np.float32)
        else:
            self.action_space = spaces.Box(low = 1, high = 5, shape = (2,), dtype = np.float32)
        self.feature_count = np.array(StressDetectionEnv.data["Features"].tolist()).shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(config.past_window * self.feature_count // 6 + config.prediction_window * 2, ), dtype=np.float32)
        
        self.timestep = 0
        self.past_prediction_buffer = np.zeros((Config().prediction_window, 2))
        self.idxs = set()

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        if len(set(StressDetectionEnv.data["Label Index"].unique()) - self.idxs) <= 0:
            return None, {}
        self.label_idx = self.rng.choice(list(set(StressDetectionEnv.data["Label Index"].unique()) - self.idxs))
        self.idxs.add(self.label_idx)

        self.env_data = StressDetectionEnv.data[StressDetectionEnv.data["Label Index"] == self.label_idx]
        self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)
        self.current_pid = self.env_data["PID"].iloc[0]

        if len(self.env_data) < self.config.env_length - 2:
            return self.reset(seed=seed)

        self.valence = self.env_data["Valence"].iloc[0]
        self.arousal = self.env_data["Arousal"].iloc[0]
        self.label = np.array([self.valence, self.arousal], dtype=np.float32)

        self.past_prediction_buffer = np.zeros((Config().prediction_window, 2))

        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        reward = 0.0
        truncation = False
        info = {}

        pred = action
        if self.config.decide_terminate:
            pred = action[0:-1]

        self.past_prediction_buffer[1:, :] = self.past_prediction_buffer[0:-1, :]
        self.past_prediction_buffer[0, :] = pred
        std = np.std(self.past_prediction_buffer, axis = 0)

        if self.config.rew_mode == "dense":
            if self.config.rew_func == "timed":
                reward = (len(self.env_data) - (self.timestep + self.config.past_window) + 1) / len(self.env_data) * np.exp(-self.config.rew_l2_sensitivity * np.linalg.norm(pred - self.label)) 
            elif self.config.rew_func == "l2":
                reward = -np.linalg.norm(pred - self.label)
        
        if not self.config.decide_terminate:
            if std[0] < self.config.prediction_std_threshold and std[1] < self.config.prediction_std_threshold and self.timestep > self.config.prediction_window:
                terminated = True
                if self.config.rew_func == "timed":
                    reward = (len(self.env_data) - (self.timestep + self.config.past_window) + 1) * np.exp(-self.config.rew_l2_sensitivity * np.linalg.norm(pred - self.label))
                elif self.config.rew_func == "l2":
                    reward = -np.linalg.norm(pred - self.label)
        else:
            terminated = action[-1] > self.config.decide_terminate_threshold
            if terminated:
                if self.config.rew_func == "timed":
                    reward = (len(self.env_data) - (self.timestep + self.config.past_window) + 1) * np.exp(-self.config.rew_l2_sensitivity * np.linalg.norm(pred - self.label))
                elif self.config.rew_func == "l2":
                    reward = -np.linalg.norm(pred - self.label)

        if self.timestep >= len(self.env_data) - self.config.past_window - 1:
            terminated = True
            if self.config.rew_func == "timed":
                reward = (len(self.env_data) - (self.timestep + self.config.past_window) + 1) * np.exp(-self.config.rew_l2_sensitivity * np.linalg.norm(pred - self.label))
            elif self.config.rew_func == "l2":
                reward = -np.linalg.norm(pred - self.label)

        self.timestep += 1
        
        return self._get_obs(), reward, terminated, truncation, info
    
    def _get_obs(self):
        obs = np.array(self.env_data["Features"].iloc[self.timestep : self.timestep + self.config.past_window].tolist(), dtype=np.float32)
        if self.config.remove_dup:
            obs_ind = np.arange(0, obs.shape[0], 6)
            obs = obs[obs_ind]
        obs = np.concat([obs.flatten(), (self.past_prediction_buffer.flatten() - 1) / 4.0])
        return obs

if __name__ == "__main__":
    file_name = "20260327-012631_dense_timed_rew_05std_with_past_pred_BiggerModels"

    model = SAC.load(f"{file_name}")

    env = StressDetectionEnv(seed = 2)

    current_episode_reward = 0
    episode_rewards_history = deque()
    terminated_timestep = 0

    obs, info = env.reset()
    label_idx = env.label_idx
    #writer = SummaryWriter(f"./test/{file_name}/{label_idx}")
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        #writer.add_scalar(f"valence/{env.valence}", action[0], env.timestep)
        #writer.add_scalar(f"arousal/{env.arousal}", action[1], env.timestep)

        current_episode_reward += reward
        
        if terminated or truncated:
            episode_rewards_history.append(current_episode_reward)
            avg_reward = np.mean(episode_rewards_history)
            #writer.add_scalar(f"reward", avg_reward, env.timestep)
            terminated_timestep += env.timestep

            obs, info = env.reset()
            current_episode_reward = 0
            if obs is None:
                break

            label_idx = env.label_idx
            #writer.close()
            #writer = SummaryWriter(f"./test/{file_name}/{label_idx}")

    print(f"Average reward: {np.mean(episode_rewards_history)}")
    print(f"Average terminated timestep: {terminated_timestep / len(episode_rewards_history)}")
    env.close()
