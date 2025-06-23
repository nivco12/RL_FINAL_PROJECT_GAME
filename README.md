# Proximal Policy Optimization

## ***Introduction***
In this project, we implemented the Proximal Policy Optimization (PPO) algorithm to solve both a custom-designed grid-based maze environment and a standard continuous control task from the OpenAI Gymnasium library. PPO is a state-of-the-art reinforcement learning algorithm known for its stability and efficiency in policy updates. The goal of the project was to demonstrate the adaptability and performance of PPO across different types of environments—discrete and continuous action spaces—while gaining insights into the agent’s learning behavior and generalization capabilities.


## ***Project Setup and Usage***
### 1. Create and Activate a Virtual Environment
```
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
```

### 2. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the Environment
Edit the config.yaml file to set up your desired environment and training parameters. You can adjust:

- Environment name (e.g., CartPole-v1, Acrobot-v1)

- PPO hyperparameters (learning rate, total number of episodes, etc.)


### 4. . Run the Main Script

```
python main.py
```

The script will automatically load the settings from config.yaml, initialize the environment, and begin training the PPO agent.

After training completes, three graphs will appear sequentially: training rewards, test rewards, and losses. Once all graphs are closed, a demo will launch showing the trained agent playing the game specified in the config.yml file.

