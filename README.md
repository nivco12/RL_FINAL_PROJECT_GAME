# Proximal Policy Optimization


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

- Environment name (e.g., CartPole-v1, LunarLander-v2)

- PPO hyperparameters (learning rate, total number of episodes, etc.)

(Logging and saving options)

### 4. . Run the Main Script

```
python main.py
```

The script will automatically load the settings from config.yaml, initialize the environment, and begin training the PPO agent.

