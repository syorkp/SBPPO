
import tensorflow as tf
import numpy as np

from stable_baselines.common.callbacks import BaseCallback


class SavingCallback(BaseCallback):
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(SavingCallback, self).__init__(verbose)
        self.saver = tf.train.Saver(max_to_keep=5)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('prey_caught', self.training_env.envs[0].env.environment.prey_caught)
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True


        if self.training_env.envs[0].environment.total_steps == 1000 and self.training_env.envs[0].environment.total_episodes % 200 == 0:
            self.saver.save(self.sess, f"model-{str(self.training_env.envs[0].environment.total_episodes)}.cptk")




class LoggingCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(LoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('prey_caught', self.training_env.envs[0].env.environment.prey_caught)
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True


        if self.training_env.envs[0].environment.total_steps == 1000:
            prey_caught = self.training_env.envs[env].env.environment.prey_caught
            summary = tf.Summary(
                value=[tf.Summary.Value(tag='prey_caught', simple_value=prey_caught / self.training_env.num_envs)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
        # Log scalar value (here a random variable)
        # summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
        # self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        prey_caught = 0
        for env in range(self.training_env.num_envs):
            prey_caught += self.training_env.envs[env].env.environment.prey_caught
        summary = tf.Summary(value=[tf.Summary.Value(tag='prey_caught', simple_value=prey_caught/self.training_env.num_envs)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

