import gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
episodes = 5000
reward_threshold = 480
window_size = 10

# Fine discretization for better control
bins = [10, 20, 10, 20]

# Moving average smoothing
def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def discretize(obs, bins, lower, upper):
    ratios = [(obs[i] - lower[i]) / (upper[i] - lower[i]) for i in range(len(obs))]
    new_obs = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def get_discretization_params(env_name):
    if env_name == "CartPole-v1":
        bins = [10, 20, 10, 20]
        upper = [4.8, 5, 0.418, 5]
        lower = [-4.8, -5, -0.418, -5]
    elif env_name == "Acrobot-v1":
        bins = [10, 10, 10, 10, 10, 10]
        upper = [1.0]*6
        lower = [-1.0]*6
    else:
        raise NotImplementedError(f"No discretization defined for {env_name}")
    return bins, lower, upper



# Check if solved
def check_solved(reward_list, threshold=reward_threshold, window=window_size):
    if len(reward_list) >= window:
        recent_avg = np.mean(reward_list[-window:])
        return recent_avg >= threshold
    return False


def q_learning(env):
    bins, lower, upper = get_discretization_params(env.spec.id)
    q_table = np.zeros(bins + [env.action_space.n])
    rewards = []

    alpha = 0.5
    min_alpha = 0.05
    alpha_decay = 0.999

    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.999

    gamma = 0.99

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins, lower, upper)
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins, lower, upper)
            best_next = np.max(q_table[next_state])
            q_table[state + (action,)] += alpha * (reward + gamma * best_next - q_table[state + (action,)])
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        alpha = max(min_alpha, alpha * alpha_decay)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if check_solved(rewards):
            print(f"✅ Q-learning solved in episode {ep+1} with avg reward {np.mean(rewards[-100:]):.2f}")

    return rewards

def sarsa(env):
    bins, lower, upper = get_discretization_params(env.spec.id)
    q_table = np.zeros(bins + [env.action_space.n])
    rewards = []

    alpha = 0.5
    min_alpha = 0.05
    alpha_decay = 0.999

    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.999

    gamma = 0.99

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins, lower, upper)
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])
        total_reward = 0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins, lower, upper)
            next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[next_state])
            q_table[state + (action,)] += alpha * (reward + gamma * q_table[next_state + (next_action,)] - q_table[state + (action,)])
            state = next_state
            action = next_action
            total_reward += reward

        rewards.append(total_reward)
        alpha = max(min_alpha, alpha * alpha_decay)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if check_solved(rewards):
            print(f"✅ SARSA solved in episode {ep+1} with avg reward {np.mean(rewards[-100:]):.2f}")

    return rewards


TRAIN = 1
LOAD  = 0
MODE = LOAD
# GAME = "CartPole-v1"
GAME = "Acrobot-v1"
# === Main Execution ===
if __name__ == "__main__":
    if MODE == TRAIN:
        
        # env = gym.make("CartPole-v1")
        env = gym.make(GAME)
        print("Running Q-Learning...")
        q_rewards = q_learning(env)

        print("Running SARSA...")
        sarsa_rewards = sarsa(env)

        env.close()

        # Save raw results
        if GAME == "CartPole-v1":
            np.savez("cartpole_rewards.npz", q_learning=q_rewards, sarsa=sarsa_rewards)

        if GAME == "Acrobot-v1":
            np.savez("acrobot_rewards.npz", q_learning=q_rewards, sarsa=sarsa_rewards)


        # Plot with smoothing
        plt.plot(moving_average(q_rewards), label="Q-Learning (smoothed)")
        plt.plot(moving_average(sarsa_rewards), label="SARSA (smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("CartPole: Reward vs. Episode (Q-Learning vs SARSA)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cartpole_stable_smoothed.png")
        plt.show()


    if MODE == LOAD:
                

        
        # === Load rewards ===
        if GAME == "CartPole-v1":
            ppo_data = np.load("CartPole-v1_ppo_rewards.npz")
            traditional_data = np.load("cartpole_rewards.npz")

        if GAME == "Acrobot-v1":
            ppo_data = np.load("Acrobot-v1_ppo_rewards.npz")
            traditional_data = np.load("acrobot_rewards.npz")

        ppo_rewards = ppo_data["ppo"]
        q_rewards = traditional_data["q_learning"]
        sarsa_rewards = traditional_data["sarsa"]

        # === Plot all with smoothing ===
        plt.figure(figsize=(12, 8))
        plt.plot(moving_average(q_rewards), label="Q-Learning (smoothed)")
        plt.plot(moving_average(sarsa_rewards), label="SARSA (smoothed)")
        plt.plot(moving_average(ppo_rewards), label="PPO (smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{GAME}: Reward vs. Episode (Q-Learning vs SARSA vs PPO)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig("cartpole_all_smoothed.png")
        plt.show()