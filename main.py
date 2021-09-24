# import gym

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
#from stable_baselines.common.env_checker import check_env
from policy import CustomPolicy

#from additional_logging import TensorboardCallback

from environment import CustomEnv

# multiprocess environment
env = make_vec_env(CustomEnv, n_envs=8, env_kwargs={"rendering_frequency": 200})
# env = CustomEnv(rendering_frequency=100)
# check_env(env)

model = PPO2(CustomPolicy, env, n_steps=1000, full_tensorboard_log=False, nminibatches=4, tensorboard_log='./ppo_tensorboard/')#, policy_kwargs={"data_format":"NCHW"})
# model = PPO2.load("ppo2_cartpole", env=env)

model.learn(total_timesteps=2500000)#, callback=TensorboardCallback())
model.save("ppo_simfish")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo_simfish")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
