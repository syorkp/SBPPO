import gym
from gym import spaces
import json
import numpy as np
from fish_env import ContinuousNaturalisticEnvironment
import copy
from Tools.make_gif import make_gif


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([0.0, -0.6283185307179586]), high=np.array([10.0, 0.6283185307179586]))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 3, 2), dtype=np.int)
        self.current_configuration_location = "1"
        self.environment_params = self.load_configuration_files()
        self.environment = ContinuousNaturalisticEnvironment(self.environment_params, True)

        self.total_reward = 0
        self.total_steps = 0

        self.all_impulses = []
        self.all_angles = []
        self.episode_number = 0

        self.save_frames = False
        self.frame_buffer = []

    def load_configuration_files(self):
        # with open(f"{self.current_configuration_location}_learning.json", 'r') as f:
        #     params = json.load(f)
        with open(f"{self.current_configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return env

    def step(self, action):
        # Execute one time step within the environment
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        action = [action[0]*10, action[1]]
        o1, r, new_internal_state, d, self.frame_buffer = self.environment.simulation_step(action, frame_buffer=self.frame_buffer, save_frames=self.save_frames, activations=sa)
        self.total_reward += r
        self.total_steps += 1

        self.all_impulses.append(action[0])
        self.all_angles.append(action[1])

        if self.total_steps >= 500:
            d = True
        return o1, r, d, {}

    def reset(self):
        if self.save_frames:
            # Create the GIF
            make_gif(self.frame_buffer, f"./episodes/episode-{str(self.episode_number)}.gif",
                     duration=len(self.frame_buffer) * 0.03, true_image=True)
            self.frame_buffer = []
            self.save_frames = False
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.

        if np.isnan(np.sum(self.all_impulses)):
            x = True

        # Reset the state of the environment to an initial state
        total_reward = copy.copy(self.total_reward)
        print(f"\nEpisode: {self.episode_number}")
        print(f"Total reward: {self.total_reward}")
        print(f"Number of prey caught: {self.environment.prey_caught}")
        print(f"Average impulse: {np.mean(self.all_impulses)}")
        print(f"Average angle: {np.mean(self.all_angles)}")
        ep_length = copy.copy(self.environment.num_steps)
        self.environment.reset()
        o1, r, new_internal_state, d, frame_buffer = self.environment.simulation_step([4.0, 0], [], False, sa)
        self.total_steps = 0
        self.total_reward = 0

        self.all_impulses = []
        self.all_angles = []
        self.episode_number += 1
        if self.episode_number % 20 == 0 and self.episode_number != 0:
            self.save_frames = True
        return o1

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...


def my_env_function():
    """Creates, and returns an instance of the environment, under the gym api above."""
    return CustomEnv()
