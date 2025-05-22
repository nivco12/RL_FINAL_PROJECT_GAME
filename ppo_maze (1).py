
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
from gym import spaces
from matplotlib.colors import ListedColormap
import pandas as pd

# --- Maze Environment ---
class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 2],  # 2 = goal
        ])
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.goal_pos = (3, 6)
        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # up, down, left, right

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_obs()

    def step(self, action):
        y, x = self.agent_pos
        if action == 0 and y > 0: y -= 1      # up
        elif action == 1 and y < self.grid.shape[0] - 1: y += 1    # down
        elif action == 2 and x > 0: x -= 1    # left
        elif action == 3 and x < self.grid.shape[1] - 1: x += 1    # right

        if self.grid[y, x] == 1:  # wall
            y, x = self.agent_pos

        self.agent_pos = (y, x)
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        y, x = self.agent_pos
        return np.array([y / (self.grid.shape[0] - 1), x / (self.grid.shape[1] - 1)], dtype=np.float32)

    def get_render_frame(self):
        grid_disp = np.copy(self.grid)
        y, x = self.agent_pos
        grid_disp[y, x] = 9  # agent
        return grid_disp

# --- PPO Model ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(self.policy(x), dim=-1), self.value(x)

# --- GAE Advantage Estimation ---
def compute_returns(rewards, values, gamma=0.99, lam=0.95):
    advantages, gae = [], 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns

# --- PPO Update ---
def ppo_update(model, optimizer, states, actions, old_probs, returns, advantages, eps_clip=0.2):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions)
    old_probs = torch.tensor(old_probs)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    for _ in range(4):  # epochs
        new_probs, values = model(states)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        old_log_probs = torch.log(old_probs)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        loss_actor = -torch.min(ratio * advantages, clip_adv).mean()
        loss_critic = (returns - values.squeeze()).pow(2).mean()

        loss = loss_actor + 0.5 * loss_critic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- Animation Function ---
def animate_maze(frames, episode):
    cmap = ListedColormap([
        '#1f77b4',  # 0 = path (blue)
        '#2ca02c',  # 1 = wall (green)
        '#ffdf00',  # 2 = goal (yellow)
        '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',
        '#d62728'   # 9 = agent (red)
    ])
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=9)

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=True)
    plt.title(f"Episode {episode} Agent Path")
    plt.axis('off')
    plt.show()

# --- Training Loop ---
env = MazeEnv()
model = ActorCritic(2, env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

reward_history = []  # Track rewards over episodes

for episode in range(100):
    state = env.reset()
    done = False

    states, actions, rewards, probs, values = [], [], [], [], []
    episode_frames = [env.get_render_frame()]

    while not done:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        dist, value = model(state_tensor)
        action_dist = torch.distributions.Categorical(dist)
        action = action_dist.sample().item()

        next_state, reward, done, _ = env.step(action)

        episode_frames.append(env.get_render_frame())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        probs.append(dist[0][action].item())
        values.append(value.item())

        state = next_state

    advs, rets = compute_returns(rewards, values)
    ppo_update(model, optimizer, states, actions, probs, rets, advs)

    total_reward = sum(rewards)
    reward_history.append(total_reward)  # Save reward
    print(f"✅ Episode {episode} finished. Total Reward: {total_reward:.2f}")

    if episode % 20 == 0:
        animate_maze(episode_frames, episode)

# --- Plot the Learning Curve ---
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label='Total Reward per Episode')
plt.plot(pd.Series(reward_history).rolling(10).mean(), label='Smoothed Reward (window=10)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
