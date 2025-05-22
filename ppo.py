import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
from gym import spaces
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as distributions
from gym.wrappers import RecordVideo
import os
import time

# code taken from :
# https://www.datacamp.com/tutorial/proximal-policy-optimization

# --- Environments ---
env_train = gym.make('CartPole-v1')
env_test = gym.make('CartPole-v1')  

#env_test = gym.make('CartPole-v1', render_mode="human")  

# --- Backbone Network ---
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

# --- Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

# --- Agent Creation ---
def create_agent(hidden_dimensions, dropout):
    INPUT_FEATURES = env_train.observation_space.shape[0]
    ACTOR_OUTPUT_FEATURES = env_train.action_space.n
    CRITIC_OUTPUT_FEATURES = 1
    actor = BackboneNetwork(INPUT_FEATURES, hidden_dimensions, ACTOR_OUTPUT_FEATURES, dropout)
    critic = BackboneNetwork(INPUT_FEATURES, hidden_dimensions, CRITIC_OUTPUT_FEATURES, dropout)
    agent = ActorCritic(actor, critic)
    return agent

# --- Utility Functions ---
def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def calculate_surrogate_loss(actions_log_probability_old, actions_log_probability_new, epsilon, advantages):
    advantages = advantages.detach()
    policy_ratio = (actions_log_probability_new - actions_log_probability_old).exp()
    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(policy_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss

def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    value_loss = torch.nn.functional.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss

def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, episode_reward

def forward_pass(env, agent, optimizer, discount_factor):
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    state, _ = env.reset()
    agent.train()
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        states.append(state_tensor)
        action_pred, value_pred = agent(state_tensor)
        action_prob = torch.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
        state = next_state
    states = torch.cat(states)
    actions = torch.cat(actions)
    actions_log_probability = torch.cat(actions_log_probability)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, actions_log_probability, advantages, returns

def update_policy(agent, states, actions, actions_log_probability_old, advantages, returns, optimizer, ppo_steps, epsilon, entropy_coefficient):
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(states, actions, actions_log_probability_old, advantages, returns)
    batch_dataset = DataLoader(training_results_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for _ in range(ppo_steps):
        for states_batch, actions_batch, actions_log_probability_old_batch, advantages_batch, returns_batch in batch_dataset:
            action_pred, value_pred = agent(states_batch)
            value_pred = value_pred.squeeze(-1)
            action_prob = torch.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            entropy = dist.entropy()
            actions_log_probability_new = dist.log_prob(actions_batch)
            surrogate_loss = calculate_surrogate_loss(
                actions_log_probability_old_batch,
                actions_log_probability_new,
                epsilon,
                advantages_batch
            )
            policy_loss, value_loss = calculate_losses(
                surrogate_loss,
                entropy,
                entropy_coefficient,
                returns_batch,
                value_pred
            )
            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, agent):
    agent.eval()
    rewards = []
    done = False
    episode_reward = 0
    state, _ = env.reset()
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = agent(state_tensor)
            action_prob = torch.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward
        state = next_state
    return episode_reward

def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Training Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y', label='Reward Threshold')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_test_rewards(test_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Testing Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='y', label='Reward Threshold')
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




#dont use, slows down the training
def watch_trained_agent(env, agent, n_episodes=5):
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"\n🎥 Watching Episode {ep+1}")
        time.sleep(1)

        while not done:
            env.render()  # 👈 display the environment
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = agent(state_tensor)
                action_prob = torch.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            state = next_state

            time.sleep(0.05)  # 👈 small delay to make it viewable

        print(f"Episode Reward: {total_reward:.2f}")

    env.close()

def record_agent(env_name, agent, video_folder, video_name, n_episodes=1):
    # Create a brand new env instance with video recording
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_name)

    agent.eval()

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = agent(state_tensor)
                action_prob = torch.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state = next_state

    env.close()

# --- Main Training ---
def run_ppo():
    MAX_EPISODES = 100
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 0.001

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []

    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    #     # --- Watch agent BEFORE training ---
    # print("\n🎬 Watching agent BEFORE training...")
    # watch_trained_agent(env_test, agent, n_episodes=1)

    print("\n🎬 Recording BEFORE training...")
    record_agent('CartPole-v1', agent, video_folder="videos_before", video_name="before_training", n_episodes=1)



    for episode in range(1, MAX_EPISODES + 1):
        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(env_train, agent, optimizer, DISCOUNT_FACTOR)
        policy_loss, value_loss = update_policy(agent, states, actions, actions_log_probability, advantages, returns, optimizer, PPO_STEPS, EPSILON, ENTROPY_COEFFICIENT)
        test_reward = evaluate(env_test, agent)

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))

        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:3.1f} | Mean Test Rewards: {mean_test_rewards:3.1f} | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
        
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'✅ Reached reward threshold in {episode} episodes!')
            break

    plot_train_rewards(train_rewards, REWARD_THRESHOLD)
    plot_test_rewards(test_rewards, REWARD_THRESHOLD)
    plot_losses(policy_losses, value_losses)

    #   # --- Watch agent AFTER training ---
    # print("\n🎬 Watching agent AFTER training...")
    # watch_trained_agent(env_test, agent, n_episodes=5)
    
    print("\n🎬 Recording AFTER training...")
    record_agent('CartPole-v1', agent, video_folder="videos_after", video_name="after_training", n_episodes=1)



if __name__ == "__main__":
    run_ppo()
