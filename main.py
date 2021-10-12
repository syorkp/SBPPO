import os

import tensorflow as tf

from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
#from stable_baselines.common.policies import LstmPolicy
# from stable_baselines.common.callbacks import CheckpointCallback

from Networks.reflected_policy import ReflectedPolicy
from Environments.continuous_environment import ContinuousEnv
from Environments.discrete_environment import DiscreteEnv
# from additional_logging import LoggingCallback, SavingCallback

trial_name = "ContinuousSigmaSaving"

if not os.path.exists(f"Training-Output/{trial_name}/"):
    os.makedirs(f"Training-Output/{trial_name}/")
    os.makedirs(f'Training-Output/{trial_name}/model_checkpoints/')
    os.makedirs(f'Training-Output/{trial_name}/ppo_tensorboard/')
    os.makedirs(f'Training-Output/{trial_name}/episodes/')


# Callbacks for saving
# checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f'./Training-Output/{trial_name}/model_checkpoints/')
# env_log_callback = TensorboardCallback()

env = make_vec_env(ContinuousEnv, n_envs=1, env_kwargs={"rendering_frequency": 100, "trial_name": trial_name})  # TODO: Try without vectorised environment. Try without all other imports
model = PPO2(ReflectedPolicy, env, n_steps=1000, full_tensorboard_log=False, nminibatches=1,
             tensorboard_log=f'./Training-Output/{trial_name}/ppo_tensorboard/')#, policy_kwargs={"data_format":"NCHW"})

model.learn(total_timesteps=10000)#, callback=saving_callback)#, callback=[env_log_callback, checkpoint_callback])
saver = tf.train.Saver(max_to_keep=10)
saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-100000.cptk")

# model = PPO2.load("Training-Output/DiscreteReflected3")

# saver = tf.train.Saver(max_to_keep=10)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-100000.cptk")
#
# model.learn(total_timesteps=100000)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-200000.cptk")
#
# model.learn(total_timesteps=100000)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-300000.cptk")
#
# model.learn(total_timesteps=100000)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-400000.cptk")
#
# model.learn(total_timesteps=100000)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-500000.cptk")
#
# model.learn(total_timesteps=100000)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-600000.cptk")
#
# model.learn(total_timesteps=100000)
# saver.save(model.sess, f"./Training-Output/{trial_name}/model_checkpoints/model-700000.cptk")

# model = PPO2.load("ppo_simfish")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
