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

data = pd.read_parquet("./data/Dapper/norm_dapper_1m.parquet")
data['Time'] = pd.to_datetime(data['Time'])

skipped_subject = [1024, 2011]

class Config:
    past_window = 16
    env_length = 120
    early_detection_window = 10
    late_detection_window = 2
    prediction_window = 5

    r_true_positive = 1.0
    r_false_positive = -0.5
    r_true_negative = 0.001
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
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def push_batch(self, states, actions, rewards, next_states, dones):
        # Insert a batch of transitions from the parallel environments
        for i in range(len(states)):
            self.buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class StressDetectionEnv(gym.Env):
    def __init__(self, config = Config(), seed = None):
        super(StressDetectionEnv, self).__init__()
        self.config = config
        self.rng = np.random.default_rng(seed=seed)
        
        # Define Action Space: 0 = Not Stressed, 1 = Stressed
        self.action_space = spaces.Discrete(2)
        self.feature_count = np.array(data["Features"].tolist()).shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(config.past_window, self.feature_count), dtype=np.float32)
        
        self.current_subject = 0
        self.timestep = 0

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.timestep = 0

        selected = False
        while not selected:
            self.current_subject = self.rng.choice(data["PID"].unique())
            while self.current_subject in skipped_subject:
                self.current_subject = self.rng.choice(data["PID"].unique())

            subject_data = data[data["PID"] == self.current_subject]
            subject_data = subject_data.sort_values("Time").reset_index(drop=True)

            self.env_data = pd.DataFrame()
            try_counter = 0
            while len(self.env_data) < self.config.env_length + 1:
                try_counter += 1
                if try_counter > 1000:
                    break

                start_time = self.rng.choice(data["Time"].unique())
                end_time = start_time + pd.Timedelta(minutes=self.config.env_length + 1 + 30)
                self.env_data = subject_data[(subject_data["Time"] >= start_time) & (subject_data["Time"] < end_time)]
                self.env_data = self.env_data.sort_values("Time").reset_index(drop=True)
                if len(self.env_data[self.env_data["Label"] == 1].index.tolist()) > 0:
                    if min(self.env_data[self.env_data["Label"] == 1].index.tolist()) < self.config.past_window:
                        continue

                if len(self.env_data) >= self.config.env_length + 1:
                    selected = True

        num_stress_label = len(self.env_data[self.env_data["Label"] == 1].index.tolist())
        self.stress_labels_time = np.inf
        if num_stress_label > 0:
            self.stress_labels_time = min(self.env_data[self.env_data["Label"] == 1].index.tolist())

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
            
            t_diff = self.stress_labels_time - (self.timestep + self.config.past_window)
            if t_diff <= self.config.early_detection_window and t_diff >= -self.config.late_detection_window:
                reward = self.config.r_true_positive
            else:
                reward = self.config.r_false_positive
        else:
            t_diff = self.stress_labels_time - (self.timestep + self.config.past_window)
            if t_diff > self.config.early_detection_window:
                reward = self.config.r_true_negative
            elif t_diff < -self.config.late_detection_window:
                reward = self.config.r_false_negative
                terminated = True

        # Played till end of environment without detecting stress
        if self.timestep >= self.config.env_length - self.config.past_window:
            truncation = True

        self.timestep += 1
        obs = np.array(self.env_data["Features"].iloc[self.timestep : self.timestep + self.config.past_window].tolist(), dtype=np.float32)

        return obs, reward, terminated, truncation, info

def make_env(seed=None):
    def _init():
        env = StressDetectionEnv(seed=seed)
        return env
    return _init

def train_dqn_vector(vec_env, total_steps=50000):
    # Hyperparameters
    BATCH_SIZE = 128       # Increased batch size since we gather data faster
    GAMMA = 0.99
    LR = 1e-4
    MEMORY_SIZE = 100000
    TARGET_UPDATE_FREQ = 10 
    
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 10000 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # For VectorEnvs, we use single_observation_space and single_action_space
    single_obs_shape = vec_env.single_observation_space.shape
    n_actions = vec_env.single_action_space.n
    num_envs = vec_env.num_envs

    policy_net = ViT(
        image_size = single_obs_shape,
        patch_size = 8,
        num_classes = n_actions,
        dim = 256,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        channels = 1
    ).to(device) # DQN(single_obs_shape, n_actions).to(device)
    target_net = ViT(
        image_size = single_obs_shape,
        patch_size = 8,
        num_classes = n_actions,
        dim = 256,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        channels = 1
    ).to(device) # DQN(single_obs_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    loss_fn = nn.SmoothL1Loss()

    # Reset all environments once at the beginning
    states, _ = vec_env.reset()
    
    # Track episodic returns manually for logging
    episode_returns = np.zeros(num_envs)

    past_prediction_buffer = np.zeros((num_envs, Config().prediction_window))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"run/{timestamp}")

    for step in range(total_steps):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * step / EPSILON_DECAY)
                  
        # --- Batched Action Selection ---
        if random.random() > epsilon:
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(device)
                states_tensor = states_tensor.unsqueeze(1)
                q_values = policy_net(states_tensor)


                # probs = torch.softmax(q_values, dim = 1).cpu().numpy()
                # past_prediction_buffer[:, 1:] = past_prediction_buffer[:, 0:-1]
                # past_prediction_buffer[:, 0] = probs[:, 1]
                # p = 1 - np.prod(1 - past_prediction_buffer, axis = 1)
                # actions = (random.random() < p).astype(int)


                actions = q_values.argmax(dim=1).cpu().numpy()
        else:
            actions = vec_env.action_space.sample()

        # --- Step All Environments ---
        next_states, rewards, terminations, truncations, infos = vec_env.step(actions)
        dones = terminations | truncations
        episode_returns += rewards

        # --- Handle Gymnasium VectorEnv Auto-Resets ---
        # Copy next_states to modify them without affecting the env's internal arrays
        real_next_states = next_states.copy()
        
        writer.add_scalar("Reward", episode_returns.mean(), step * num_envs)
        writer.add_scalar("Env terminated", dones.sum(), step * num_envs)
        writer.add_scalar("Accuracy/True Positive", (rewards == Config().r_true_positive).sum(), step * num_envs)
        writer.add_scalar("Accuracy/False Positive", (rewards == Config().r_false_positive).sum(), step * num_envs)
        writer.add_scalar("Accuracy/True Negative", (rewards == Config().r_true_negative).sum(), step * num_envs)
        writer.add_scalar("Accuracy/False Negative", (rewards == Config().r_false_negative).sum(), step * num_envs)
        writer.add_scalar("Epsilon", epsilon, step * num_envs)

        # If any environment finished, extract its true final state
        for i, d in enumerate(dones):
            if d:
                if step % 50 == 0:
                    print(f"Step {step} | Env {i} completed | Reward: {episode_returns[i]:.2f} | Epsilon: {epsilon:.3f}")
                        
                # Reset the return tracker for this sub-environment
                episode_returns[i] = 0 

        # --- Store in Memory ---
        memory.push_batch(states, actions, rewards, real_next_states, dones)
        
        # We assign `states = next_states` (the raw output from step), 
        # so the next loop starts with the newly reset state if an env finished.
        states = next_states

        # --- Optimize Model ---
        if len(memory) >= BATCH_SIZE:
            b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(BATCH_SIZE)
            
            states_t = torch.FloatTensor(b_states).to(device)
            actions_t = torch.LongTensor(b_actions).unsqueeze(1).to(device)
            rewards_t = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(b_next_states).to(device)
            dones_t = torch.FloatTensor(b_dones).unsqueeze(1).to(device)

            states_t = states_t.unsqueeze(1)
            current_q_values = policy_net(states_t).gather(1, actions_t)

            with torch.no_grad():
                next_states_t = next_states_t.unsqueeze(1)
                max_next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)
                expected_q_values = rewards_t + (GAMMA * max_next_q_values * (1 - dones_t))

            loss = loss_fn(current_q_values, expected_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

        # --- Update Target Network ---
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net

if __name__ == "__main__":
    num_parallel_envs = 128
    vec_env = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_parallel_envs)])
    # 3. Train!
    trained_model = train_dqn_vector(vec_env, total_steps=50000)
    
    vec_env.close()