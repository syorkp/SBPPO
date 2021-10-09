import gym
from gym import spaces
import json
import numpy as np
from Environments.fish_env import DiscreteNaturalisticEnvironment
import copy
from Tools.make_gif import make_gif


class DiscreteEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rendering_frequency, trial_name):
        super(DiscreteEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(10)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 3, 2), dtype=np.int)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(4, 60, 3), dtype=np.int)
        self.current_configuration_location = "1"
        self.environment_params = self.load_configuration_files()
        self.environment = DiscreteNaturalisticEnvironment(self.environment_params, True)

        self.rendering_frequency = rendering_frequency
        self.total_reward = 0
        self.total_steps = 0

        self.all_actions = []
        self.episode_number = 0

        self.save_frames = False
        self.frame_buffer = []

        self.trial_name = trial_name

    def load_configuration_files(self):
        # with open(f"{self.current_configuration_location}_learning.json", 'r') as f:
        #     params = json.load(f)
        with open(f"{self.current_configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return env

    def step(self, action):
        # Execute one time step within the environment
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        o1, r, new_internal_state, d, self.frame_buffer = self.environment.simulation_step(action, frame_buffer=self.frame_buffer, save_frames=self.save_frames, activations=sa)
        # o1 = np.reshape(o11, (4, 60, 3))
        # o1 = o1.astype("uint8")
        self.total_reward += r
        self.total_steps += 1

        self.all_actions.append(action)

        if self.total_steps >= 1000:
            d = True
        return o1, r, d, {}

    def reset(self):
        if self.save_frames:
            # Create the GIF
            make_gif(self.frame_buffer, f"./Training-Output/{self.trial_name}/episodes/episode-{str(self.episode_number)}.gif",
                     duration=len(self.frame_buffer) * 0.03, true_image=True)
            self.frame_buffer = []
            self.save_frames = False
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.

        # Reset the state of the environment to an initial state
        total_reward = copy.copy(self.total_reward)
        print(f"\nEpisode: {self.episode_number}")
        print(f"Total reward: {self.total_reward}")
        print(f"Number of prey caught: {self.environment.prey_caught}")
        ep_length = copy.copy(self.environment.num_steps)
        self.environment.reset()
        o1, r, new_internal_state, d, frame_buffer = self.environment.simulation_step(0, [], False, sa)
        self.total_steps = 0
        self.total_reward = 0

        self.all_actions = []
        self.episode_number += 1
        if self.episode_number % self.rendering_frequency == 0 and self.episode_number != 0:
            self.save_frames = True
        return o1

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...
