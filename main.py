import yaml
from ppo.ppo_algo import run_ppo
from envs.maze_env import MazeEnv
from envs.gym_env import get_gym_env

### ADD GRAPHS !!!



def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()

    if config["environment"] == "maze":
        env_train = MazeEnv()
        env_test = MazeEnv()
    else:
        env_train = get_gym_env(config["environment"])
        env_test = get_gym_env(config["environment"])

    run_ppo(config, env_train, env_test)
