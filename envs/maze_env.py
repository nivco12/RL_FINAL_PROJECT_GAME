import numpy as np
import gym
from gym import spaces

class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 2],
        ])
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.goal_pos = (3, 6)
        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, **kwargs):
        self.agent_pos = self.start_pos
        return self._get_obs(), {}

    def step(self, action):
        y, x = self.agent_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.grid.shape[0] - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.grid.shape[1] - 1: x += 1

        if self.grid[y, x] == 1:
            y, x = self.agent_pos

        self.agent_pos = (y, x)
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        y, x = self.agent_pos
        return np.array([y / (self.grid.shape[0] - 1), x / (self.grid.shape[1] - 1)], dtype=np.float32)
