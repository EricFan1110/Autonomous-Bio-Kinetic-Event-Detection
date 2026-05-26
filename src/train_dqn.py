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

from stable_baselines3.common.callbacks import BaseCallback

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

class StressDetectionEnv(gym.Env):
    # data = pd.read_parquet("./data/Dapper/dapper_dqn_norm.parquet")
    # data['Time'] = pd.to_datetime(data['Time'])
    feature_df = pd.read_csv("./data/WESAD/WESAD_table_norm.csv")
    feature_df["Features"] = feature_df['Features'].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
    embed_df = pd.read_parquet("./data/WESAD/WESAD_table_Normwear_Encoding_60s.parquet")

    combine_data = pd.merge(embed_df, feature_df, on=['Time', "PID"], how='inner')
    combine_data["Features"] = [np.concatenate([a, b]) for a, b in zip(combine_data['Features_x'], combine_data['Features_y'])]
    combine_data["Baseline"] = combine_data["Baseline_x"]
    combine_data["Stress"] = combine_data["Stress_x"]
    combine_data["Amusement"] = combine_data["Amusement_x"]
    combine_data["Meditation"] = combine_data["Meditation_x"]
    combine_data["Valence"] = combine_data["Valence_x"]
    combine_data["Arousal"] = combine_data["Arousal_x"]

    def __init__(self, config = Config(), seed = None):
        super(StressDetectionEnv, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)
        
        self.action_space = spaces.Discrete(2)
        self.feature_count = np.array(StressDetectionEnv.data["Features"].tolist()).shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(config.past_window, self.feature_count), dtype=np.float32)

        self.timestep = 0

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        label_idx = self.rng.choice(StressDetectionEnv.data["Label Index"].unique())
        self.env_data = StressDetectionEnv.data[StressDetectionEnv.data["Label Index"] == label_idx]
        self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)
        if len(self.env_data) < self.config.env_length - 2:
            return self.reset(seed=seed)

        self.stress_label = self.env_data["Stress"].iloc[0]

        obs = np.array(self.env_data["Features"].iloc[self.timestep : self.timestep + self.config.past_window].tolist(), dtype=np.float32) # shape: (T, Feature dim)
        return obs, {}

    def step(self, action):
        terminated = False
        reward = 0.0
        truncation = False
        info = {}

        # Detect Stress
        if action == 1:
            terminated = True

            if self.stress_label == 1:
                reward = self.config.r_true_positive
            else:
                reward = self.config.r_false_positive

        # Played till end of environment without detecting stress
        if self.timestep >= len(self.env_data) - self.config.past_window - 1:
            terminated = True
            if action == 0:
                if self.stress_label == 0:
                    reward = self.config.r_true_negative
                else:
                    reward = self.config.r_false_negative

        self.timestep += 1
        obs = np.array(self.env_data["Features"].iloc[self.timestep : self.timestep + self.config.past_window].tolist(), dtype=np.float32)

        return obs, reward, terminated, truncation, info

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

        max_time = self.env_data["Time"].iloc[-1]
        time = random.randrange(self.config.past_window, len(self.env_data) - self.config.env_length - self.config.past_window, 1)
        self.env_data = self.env_data.iloc[time - self.config.past_window : time + self.config.env_length]

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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for pid in SUBJECT_IDS:
        name = f"{timestamp}_dqn_PID_with_pretrained_pred/{pid}"

        env = StressDetectionEnv_WESAD(pid)
        # pretrained_weights = torch.load("./run/20260409-161205_Both_Bin_stress/model_latest.pth")

        policy_kwargs = dict(net_arch=[512, 256])
        model = stable_baselines3.DQN(
            "MlpPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            verbose=1,
            exploration_initial_eps=0.8,
            exploration_final_eps=0.05,
            tensorboard_log=f"./run/{name}"
        )

        # sb3_expected_keys = model.q_net.state_dict().keys()
        # new_state_dict = {}
        # for sb3_key, pretrained_weight in zip(sb3_expected_keys, pretrained_weights.values()):
        #     new_state_dict[sb3_key] = pretrained_weight

        # model.q_net.load_state_dict(new_state_dict, strict=True)
        # model.q_net_target.load_state_dict(model.q_net.state_dict())

        logging_callback = InfoLoggingCallback(info_key="prob")

        model.learn(total_timesteps=200000, callback=logging_callback, log_interval=500)
        model.save(f"./run/{name}/model")

