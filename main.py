import gym

from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from environment import CustomEnv

# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)
env = CustomEnv()
check_env(env)

model = PPO2(MlpPolicy, env, verbose=0, n_steps=1000, full_tensorboard_log=True, nminibatches=1)
model.learn(total_timesteps=250000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()