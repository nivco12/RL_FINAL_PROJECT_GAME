import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap


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
        self.max_steps = 100  # ✅ max steps per episode
        self.current_step = 0

        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, **kwargs):
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1  # ✅ increment step count

        y, x = self.agent_pos
        if action == 0 and y > 0: y -= 1      # Up
        elif action == 1 and y < self.grid.shape[0] - 1: y += 1  # Down
        elif action == 2 and x > 0: x -= 1    # Left
        elif action == 3 and x < self.grid.shape[1] - 1: x += 1  # Right

        if self.grid[y, x] == 1:
            y, x = self.agent_pos  # Hit a wall

        self.agent_pos = (y, x)
        done = self.agent_pos == self.goal_pos or self.current_step >= self.max_steps
        reward = 1.0 if self.agent_pos == self.goal_pos else -0.01
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        y, x = self.agent_pos
        return np.array([y / (self.grid.shape[0] - 1), x / (self.grid.shape[1] - 1)], dtype=np.float32)

    def get_frame(self):
        frame = self.grid.copy()
        y, x = self.agent_pos
        frame[y, x] = 9  # Use 9 to represent the agent
        return frame
    

    
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


