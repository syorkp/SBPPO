import os

from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback

from additional_logging import TensorboardCallback

from Networks.reflected_policy import ReflectedPolicy

from Environments.continuous_environment import ContinuousEnv
from Environments.discrete_environment import DiscreteEnv


trial_name = "DiscreteReflected2"

if not os.path.exists(f"Output/{trial_name}/"):
    os.makedirs(f"Output/{trial_name}/")
    os.makedirs(f'Output/{trial_name}/model_checkpoints/')
    os.makedirs(f'Output/{trial_name}/ppo_tensorboard/')
    os.makedirs(f'Output/{trial_name}/episodes/')


# Callbacks for saving
# checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f'./Output/{trial_name}/model_checkpoints/')
# env_log_callback = TensorboardCallback()

env = make_vec_env(DiscreteEnv, n_envs=8, env_kwargs={"rendering_frequency": 200, "trial_name": trial_name})

model = PPO2(ReflectedPolicy, env, n_steps=1000, full_tensorboard_log=False, nminibatches=4, tensorboard_log=f'./Output/{trial_name}/ppo_tensorboard/')#, policy_kwargs={"data_format":"NCHW"})

model.learn(total_timesteps=2500000)#, callback=checkpoint_callback)#, callback=[env_log_callback, checkpoint_callback])


# model = PPO2.load("ppo_simfish")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
