import gym
from gym import spaces
import numpy as np
import serial
import time

class RealRobotEnv(gym.Env):
    def __init__(self, port="COM3", baudrate=115200, timeout=1):
        super().__init__()

        # Serial connection to Arduino
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # wait for Arduino to reboot

        # Observation: pitch angle, normalized [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Action: continuous value to control motors [-1, 1]  - -1: backward, 0: stop, 1:forward
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # For simulation control
        self.current_pitch = 0.0
        self.max_pitch = 0.7  # ~40 degrees in radians

    def reset(self, **kwargs):
        self._send_command("RESET")
        self.current_pitch = self._read_pitch()
        return np.array([self.current_pitch], dtype=np.float32), {}

    def step(self, action):
        # Clip and send action to Arduino
        motor_cmd = float(np.clip(action[0], -1.0, 1.0))
        self._send_command(f"ACTION:{motor_cmd:.3f}")

        # Read new state
        self.current_pitch = self._read_pitch()

        # Compute reward (closer to upright = better)
        reward = 1.0 - abs(self.current_pitch)

        # Episode done if robot falls
        done = abs(self.current_pitch) > self.max_pitch
        truncated = False

        return np.array([self.current_pitch], dtype=np.float32), reward, done, truncated, {}

    def _read_pitch(self):
        try:
            self.ser.reset_input_buffer()
            self.ser.write(b"GET_PITCH\n")
            line = self.ser.readline().decode().strip()
            pitch = float(line)  # Expecting Arduino to send single float
            # Normalize pitch to [-1, 1] based on max_pitch
            return np.clip(pitch / self.max_pitch, -1.0, 1.0)
        except Exception as e:
            print(f"[ERROR] Failed to read pitch: {e}")
            return 0.0

    def _send_command(self, msg):
        try:
            self.ser.write((msg + "\n").encode())
        except Exception as e:
            print(f"[ERROR] Failed to send command: {e}")

    def close(self):
        self.ser.close()


###################
## Example usage ##
###################

# from robot_env import RealRobotEnv

# env = RealRobotEnv(port="COM3")  # Adjust port

# obs, _ = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Obs: {obs}, Reward: {reward}")

# env.close()


####################################
## Maping action to pwm in arduino##
####################################
# // Received action value from Python, e.g. -0.5 to 0.5
# float action = received_action_value;
# int pwm = int(action * 255);  // Scale to motor driver range

# if (pwm >= 0) {
#   analogWrite(motor_forward_pin, pwm);
#   analogWrite(motor_backward_pin, 0);
# } else {
#   analogWrite(motor_forward_pin, 0);
#   analogWrite(motor_backward_pin, -pwm);
# }
