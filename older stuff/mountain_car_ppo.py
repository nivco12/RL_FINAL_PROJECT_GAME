import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time

# --- Hyperparameters ---
policy_lr = 3e-4
value_lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
update_epochs = 4
batch_size = 5000
total_iterations = 100

# --- Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.fc(x)

# --- Value Network ---
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)

# --- Helper function to select action ---
def select_action(policy_net, obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    logits = policy_net(obs)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# --- Play using a trained policy ---
def watch(policy_net, env, episodes=5):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            time.sleep(0.02)

            with torch.no_grad():
                action, _ = select_action(policy_net, obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1} total reward: {total_reward}")
    env.close()

# --- Training ---
def train():
    # Two environments
    env_train = gym.make("CartPole-v1", render_mode=None)
    env_watch = gym.make("CartPole-v1", render_mode="human")

    obs_dim = env_train.observation_space.shape[0]
    n_actions = env_train.action_space.n

    policy_net = PolicyNetwork(obs_dim, n_actions)
    value_net = ValueNetwork(obs_dim)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=policy_lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=value_lr)

    returns_history = []

    for iteration in range(total_iterations):
        obs_list = []
        action_list = []
        logprob_list = []
        reward_list = []
        done_list = []
        value_list = []
        next_value_list = []

        obs, info = env_train.reset()
        steps = 0
        ep_rewards = []

        while steps < batch_size:
            action, logprob = select_action(policy_net, obs)
            value = value_net(torch.tensor(obs, dtype=torch.float32))

            next_obs, reward, terminated, truncated, _ = env_train.step(action)

            next_value = value_net(torch.tensor(next_obs, dtype=torch.float32))

            obs_list.append(obs)
            action_list.append(action)
            logprob_list.append(logprob)
            reward_list.append(reward)
            done_list.append(terminated or truncated)
            value_list.append(value.item())
            next_value_list.append(next_value.item())

            obs = next_obs
            ep_rewards.append(reward)

            steps += 1

            if terminated or truncated:
                obs, info = env_train.reset()

        # --- Compute Advantages with GAE ---
        advantages = []
        gae = 0
        for t in reversed(range(len(reward_list))):
            mask = 1.0 - done_list[t]
            delta = reward_list[t] + gamma * next_value_list[t] * mask - value_list[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(value_list, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.from_numpy(np.array(obs_list)).float()
        action_tensor = torch.tensor(action_list, dtype=torch.int64)
        old_logprobs = torch.stack(logprob_list).detach()

        # --- PPO Update ---
        for _ in range(update_epochs):
            logits = policy_net(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_logprobs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - old_logprobs)

            # Policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # Value loss
            values = value_net(obs_tensor)
            value_loss = ((returns - values) ** 2).mean()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

        avg_return = np.sum(ep_rewards) / (len(done_list) // 2)
        returns_history.append(avg_return)

        print(f"Iteration {iteration+1}/{total_iterations}")
        print(f"  Avg Return: {avg_return:.2f}")
        print(f"  Policy Loss: {policy_loss.item():.4f}")
        print(f"  Value Loss: {value_loss.item():.4f}")
        print(f"  Entropy: {entropy.item():.4f}")

        # Save model every 10 iterations
        if (iteration + 1) % 10 == 0:
            torch.save(policy_net.state_dict(), "ppo_cartpole_gae.pth")

    # Final save
    torch.save(policy_net.state_dict(), "ppo_cartpole_gae.pth")

    print("✅ Training finished! Plotting training curve...")
    plt.plot(returns_history)
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Training Progress (CartPole with GAE)')
    plt.grid()
    plt.show()

    print("✅ Watching trained agent...")
    policy_net.load_state_dict(torch.load("ppo_cartpole_gae.pth"))
    policy_net.eval()
    watch(policy_net, env_watch, episodes=5)

    env_train.close()
    env_watch.close()

if __name__ == "__main__":
    train()
