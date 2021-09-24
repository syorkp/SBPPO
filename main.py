from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback

from additional_logging import TensorboardCallback

from Networks.continuous_policy import CustomPolicy
from Networks.reflected_policy import ReflectedPolicy

from Environments.continuous_environment import ContinuousEnv
from Environments.discrete_environment import DiscreteEnv

# Callbacks for saving
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./model_checkpoints/')
env_log_callback = TensorboardCallback()

env = make_vec_env(ContinuousEnv, n_envs=8, env_kwargs={"rendering_frequency": 200})

model = PPO2(ReflectedPolicy, env, n_steps=1000, full_tensorboard_log=False, nminibatches=4, tensorboard_log='./ppo_tensorboard/')#, policy_kwargs={"data_format":"NCHW"})

model.learn(total_timesteps=2500000)#, callback=[env_log_callback, checkpoint_callback])
model.save("ppo_simfish")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo_simfish")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
