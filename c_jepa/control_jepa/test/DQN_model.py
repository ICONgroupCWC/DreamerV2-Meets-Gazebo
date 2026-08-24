import os
import matplotlib
import torch
import csv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from typing import Union
plt.ion()
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class SkipFrame(gym.Wrapper):
    """
    A wrapper for skipping frames in the environment to speed up training.

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.

        skip (int) : The number of frames to skip.
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # Executes the action for the specified number of frames, accumulating rewards.
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info
# =========================================================
# NORMAL ARRAY-BASED REPLAY BUFFER (REPLACEMENT ONLY)
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.new_state = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, new_state, terminated):
        self.state[self.ptr] = (state * 255).astype(np.uint8)
        self.new_state[self.ptr] = (new_state * 255).astype(np.uint8)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.state[idx], dtype=torch.float32, device=device) / 255.0
        new_states = torch.tensor(self.new_state[idx], dtype=torch.float32, device=device) / 255.0
        actions = torch.tensor(self.action[idx], dtype=torch.long, device=device)
        rewards = torch.tensor(self.reward[idx], dtype=torch.float32, device=device)
        terminateds = torch.tensor(self.terminated[idx], dtype=torch.float32, device=device)

        return states, actions, rewards, new_states, terminateds


# =========================================================
# SKIP FRAME (UNCHANGED)
# =========================================================
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


# =========================================================
# DQN NETWORK (UNCHANGED)
# =========================================================
class DQN(nn.Module):
    """
    Defines the neural network architecture for the DQN agent.

    Parameters:
        in_dim (tuple) : The shape of the input state (channels, height, width).

        out_dim (int) : The number of possible actions.
    """
    def __init__(self, in_dim: tuple, out_dim: int):
        super().__init__()
        cannel_n, height, width = in_dim
        if height != 84 or width != 84:
            error_text = f"DQN model requires input of a (84, 84)-shape. \
                           Input of a ({height, width})-shape was passed."
            raise ValueError(error_text)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=cannel_n, out_channels=16,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, input):
        return self.net(input)


# =========================================================
# AGENT (UNCHANGED EXCEPT BUFFER)
# =========================================================
class Agent:
    def __init__(
        self,
        state_space_shape: int,
        action_n: int,
        load_state: str = "",
        load_model: str = None,
        double_q: bool = False,
        gamma: float = 0.95,
        epsilon: float = 1,
        epsilon_decay: float = 0.9999925,
        epsilon_min: float = 0.05
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_shape = state_space_shape
        self.action_n = action_n
        self.load_state = load_state
        self.double_q = double_q
        self.save_dir = './training/saved_models/'
        self.log_dir = './training/logs/'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.updating_net = DQN(self.state_shape, self.action_n).to(self.device)
        self.frozen_net = DQN(self.state_shape, self.action_n).to(self.device)
        self.frozen_net.load_state_dict(self.updating_net.state_dict())

        self.optimizer = torch.optim.Adam(self.updating_net.parameters(), lr=0.0002)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # 🔴 ONLY CHANGE: NORMAL REPLAY BUFFER
        self.buffer = ReplayBuffer(300000, self.state_shape)

        self.act_taken = 0
        self.n_updates = 0

    def store(self, state, action, reward, new_state, terminated):
        self.buffer.add(state, action, reward, new_state, terminated)

    def get_samples(self, batch_size: int):
        return self.buffer.sample(batch_size, self.device)

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.updating_net(state)
            action_idx = torch.argmax(q_values, dim=1).item()

        action_onehot = np.zeros(self.action_n, dtype=np.float32)
        action_onehot[action_idx] = 1.0

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.act_taken += 1
        return action_onehot
    
    def take_action_(self, state: Union[np.ndarray, torch.Tensor]):
        """
        Chooses an action based on the epsilon-greedy policy.

        Parameters:
            state (numpy.ndarray | torch.Tensor) : The current state of 
            the environment.

        Returns:
            action_idx (torch.Tensor) : The action chosen by the agent.
        """
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device
                ).unsqueeze(0)
            action_values = self.updating_net(state)
            action_idx = torch.argmax(action_values, axis=1).item()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        self.act_taken += 1
        return action_idx

    def update_net(self, batch_size: int):
        if self.buffer.size < batch_size:
            return None, 0.0

        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)

        q_values = self.updating_net(states)
        td_est = q_values[np.arange(batch_size), actions]

        with torch.no_grad():
            if self.double_q:
                next_actions = torch.argmax(self.updating_net(new_states), dim=1)
                q_next = self.frozen_net(new_states)[np.arange(batch_size), next_actions]
            else:
                q_next = self.frozen_net(new_states).max(1)[0]

            td_tar = rewards + (1 - terminateds) * self.gamma * q_next

        loss = self.loss_fn(td_est, td_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_est, loss.item()

    def save(self, save_dir: str, save_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + save_name + f"_{self.act_taken}.pt"
        torch.save({
            'upd_model_state_dict': self.updating_net.state_dict(),
            'frz_model_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_number': self.act_taken,
            'epsilon': self.epsilon
        }, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_dir: str, model_name: str):
        """
        Loads a saved model and its parameters.

        Parameters:
            load_dir (str) : The directory from which the model should be loaded.

            model_name (str) : The name of the file containing the saved model.
        """
        loaded_model = torch.load(load_dir+model_name)
        upd_net_param = loaded_model['upd_model_state_dict']
        frz_net_param = loaded_model['frz_model_state_dict']
        opt_param = loaded_model['optimizer_state_dict']
        self.updating_net.load_state_dict(upd_net_param)
        self.frozen_net.load_state_dict(frz_net_param)
        self.optimizer.load_state_dict(opt_param)
        if self.load_state == 'eval':
            self.updating_net.eval()
            self.frozen_net.eval()
            self.epsilon_min = 0
            self.epsilon = 0
        elif self.load_state == 'train':
            self.updating_net.train()
            self.frozen_net.train()
            self.act_taken = loaded_model['action_number']
            self.epsilon = loaded_model['epsilon']
        else:
            raise ValueError(f"Unknown load state. Should be either 'eval' or 'train'.")

    def write_log(
        self,
        date_list: list,
        time_list: list,
        reward_list: list,
        length_list: list,
        loss_list: list,
        epsilon_list: list,
        log_filename: str = 'default_log.csv'
    ):
       
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        rows = [['date']+date_list,
                ['time']+time_list,
                ['reward']+reward_list,
                ['length']+length_list,
                ['loss']+loss_list,
                ['epsilon']+epsilon_list]
        with open(self.log_dir+log_filename, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)    
            csvwriter.writerows(rows)


def plot_reward(episode_num: int, reward_list: list, n_steps: int):
    """
    Plots the reward progression over episodes.

    Parameters:
        episode_num (int) : The current episode number.

        reward_list (list) : A list of rewards obtained in all episodes so far.

        n_steps (int) : The number of steps taken so far.
    """
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(f'Episode #{episode_num}: {n_steps} steps, \
                  reward {mean_eval_reward}±{std_eval_reward}')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())
    if len(rewards_tensor) >= 50:
        reward_f = torch.clone(rewards_tensor[:50])
        means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(49)*torch.mean(reward_f), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)