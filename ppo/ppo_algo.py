import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset
import gym
import time
from envs.maze_env import animate_maze, MazeEnv
import matplotlib.pyplot as plt
import os
from datetime import datetime

# code taken from :
# https://www.datacamp.com/tutorial/proximal-policy-optimization



class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

def create_agent(input_dim, action_dim, hidden_dim, dropout):
    actor = BackboneNetwork(input_dim, hidden_dim, action_dim, dropout)
    critic = BackboneNetwork(input_dim, hidden_dim, 1, dropout)
    return ActorCritic(actor, critic)

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def calculate_surrogate_loss(actions_log_prob_old, actions_log_prob_new, epsilon, advantages):
    policy_ratio = (actions_log_prob_new - actions_log_prob_old).exp()
    surrogate1 = policy_ratio * advantages
    surrogate2 = torch.clamp(policy_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return torch.min(surrogate1, surrogate2)

def calculate_losses(surrogate_loss, entropy, entropy_coeff, returns, value_pred):
    policy_loss = -(surrogate_loss + entropy_coeff * entropy).sum()
    value_loss = nn.functional.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss

def forward_pass(env, agent, discount_factor):
    states, actions, old_log_probs, values, rewards = [], [], [], [], []
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_pred, value_pred = agent(state_tensor)
        action_prob = torch.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state_tensor)
        actions.append(action)
        old_log_probs.append(dist.log_prob(action))
        values.append(value_pred)
        rewards.append(reward)
        state = next_state

    states = torch.cat(states)
    actions = torch.cat(actions)
    old_log_probs = torch.cat(old_log_probs)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    return states, actions, old_log_probs, returns, advantages, sum(rewards)

def update_policy(agent, optimizer, states, actions, old_log_probs, returns, advantages,
                  epsilon, entropy_coeff, ppo_steps=4, batch_size=128):
    dataset = TensorDataset(states, actions, old_log_probs.detach(), returns.detach(), advantages.detach())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(ppo_steps):
        for s_batch, a_batch, logp_old_batch, ret_batch, adv_batch in dataloader:
            action_pred, value_pred = agent(s_batch)
            action_prob = torch.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            logp_new = dist.log_prob(a_batch)

            # Detach here ensures graph is not reused
            surrogate = calculate_surrogate_loss(logp_old_batch.detach(), logp_new, epsilon, adv_batch.detach())
            entropy = dist.entropy().detach()  # ❗️detach this too

            policy_loss, value_loss = calculate_losses(surrogate, entropy, entropy_coeff, ret_batch, value_pred.squeeze(-1))

            total_loss = policy_loss + value_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return policy_loss.item(), value_loss.item()


def evaluate(env, agent):
    agent.eval()
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = agent(state_tensor)
            action = torch.argmax(torch.softmax(action_pred, dim=-1), dim=-1)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        state = next_state
    return total_reward

def plot_train_rewards(train_rewards):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Training Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_test_rewards(test_rewards):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Testing Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label='Value Losses')
    plt.plot(policy_losses, label='Policy Losses')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def watch_trained_agent(env_name, agent, n_episodes=5):
    env = gym.make(env_name, render_mode="human")
    agent.eval()
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"\n🎮 Watching Episode {ep+1}")

        while not done:
            env.render()
            time.sleep(0.03)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = agent(state_tensor)
                action_prob = torch.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

        print(f"🏁 Episode Reward: {total_reward:.2f}")
    env.close()



    
def watch_trained_maze_agent(env, agent, n_episodes=1):
    agent.eval()

    for ep in range(n_episodes):
        state, _ = env.reset()
        frames = [env.get_frame()]
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = agent(state_tensor)
                action_prob = torch.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(env.get_frame())
            total_reward += reward

        print(f"🏁 Episode {ep+1} Reward: {total_reward:.2f}")
        animate_maze(frames, ep+1)



# def save_model(agent, env_name, directory="models"):
#     os.makedirs(directory, exist_ok=True)
#     date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
#     path = os.path.join(directory, f"{env_name}_{date_str}.pth")
#     torch.save(agent.state_dict(), path)
#     print(f"💾 Model saved to {path}")
#     return path

# def load_model(agent, path):
#     agent.load_state_dict(torch.load(path))
#     agent.eval()
#     print(f"✅ Loaded model from {path}")



def run_ppo(config, env_train, env_test):
    env = config["environment"]
    input_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n
    agent = create_agent(input_dim, action_dim, config["hidden_dim"], config["dropout"])
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []

    for episode in range(1, config["total_episodes"] + 1):
        states, actions, old_log_probs, returns, advantages, train_reward = forward_pass(env_train, agent, config["gamma"])
        policy_loss, value_loss = update_policy(agent, optimizer, states, actions, old_log_probs, returns, advantages,
                                                config["epsilon"], config["entropy_coefficient"], config["ppo_steps"])
        test_reward = evaluate(env_test, agent)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        if episode % config["print_interval"] == 0:
            print(f"Episode: {episode:3} | Train: {np.mean(train_rewards[-10:]):.2f} | Test: {np.mean(test_rewards[-10:]):.2f}")

        if env == "maze":
            if config["early_stopping"]:
                if np.mean(test_rewards[-10:]) >= config["reward_threshold_maze"]:
                    print(f"✅ Solved in {episode} episodes!")
                    break

        if env == "CartPole-v1":
            if config["early_stopping"]:
              if np.mean(test_rewards[-10:]) >= config["reward_threshold_CartPole-v1"]:
                print(f"✅ Solved in {episode} episodes!")
                break
        
        
        if env == "Acrobot-v1":
            if config["early_stopping"]:
              if np.mean(test_rewards[-10:]) >= config["reward_threshold_Acrobot-v1"]:
                print(f"✅ Solved in {episode} episodes!")
                break


        # if np.mean(test_rewards[-10:]) >= config["reward_threshold"]:
        #     print(f"✅ Solved in {episode} episodes!")
        #     break


    plot_train_rewards(train_rewards)
    plot_test_rewards(test_rewards)
    plot_losses(policy_losses, value_losses)


     # === Save Rewards and Plot ===
    if config["save_rewards_nrz"] == True:
        env_name = config["environment"]
        rewards_filename = f"{env_name}_ppo_rewards.npz"
        np.savez(rewards_filename, ppo=train_rewards)
        print(f"📁 Saved PPO rewards to: {rewards_filename}")

    # Watch trained agent play
    if (env == "maze"):
        test_env_for_anim = MazeEnv()
        watch_trained_maze_agent(test_env_for_anim, agent, n_episodes=1)        

    else:
        watch_trained_agent(config["environment"], agent, n_episodes=3)
    
    # if (config["save_model"] == True):
    #       # Save the model with game name and timestamp
    #       save_model(agent, config["environment"])

    return train_rewards, test_rewards, policy_losses, value_losses
