import numpy as np
from abc import ABC, abstractmethod

import gym
import tensorflow.compat.v1 as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn, RecurrentActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common.tf_layers import lstm, linear

# Problems: Passing in correct obs, reflected architecture - would need to modify the distribution or create custom

def batch_to_seq(tensor_batch, n_batch, n_steps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies

    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_steps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies
    """
    if flat:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps])
    else:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]


def seq_to_batch(tensor_sequence, flat=False):
    """
    Transform a sequence of Tensors, into a batch of Tensors for recurrent policies

    :param tensor_sequence: (TensorFlow Tensor) The input tensor to batch
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) batch of Tensors for recurrent policies
    """
    shape = tensor_sequence[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        n_hidden = tensor_sequence[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
    else:
        return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])


class CustomPolicy(RecurrentActorCriticPolicy):

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, state_shape=(2 * 512, ), reuse=reuse, scale=True)

        n_lstm = 512
        state_shape = (2 * n_lstm,)
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = self.create_network("model", ob_space, rnn_cell)
            # TODO: pass in internal state and prev action
            # TODO: Build to be symmetric
            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def create_network(self, my_scope, obs_space, rnn_cell):

        self.num_arms = obs_space.shape[0]  # Rays for each eye

        self.reshaped_observation = tf.reshape(self.processed_obs, shape=[-1, self.num_arms, 3, 2],
                                               name="reshaped_observation")
        #            ----------        Non-Reflected       ---------            #

        self.left_eye = self.reshaped_observation[:, :, :, 0]
        self.right_eye = self.reshaped_observation[:, :, :, 1]

        # Convolutional Layers
        self.conv1l = tf.layers.conv1d(inputs=self.left_eye, filters=16, kernel_size=16, strides=4, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv1l')
        self.conv2l = tf.layers.conv1d(inputs=self.conv1l, filters=8, kernel_size=8, strides=2, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv2l')
        self.conv3l = tf.layers.conv1d(inputs=self.conv2l, filters=8, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv3l')
        self.conv4l = tf.layers.conv1d(inputs=self.conv3l, filters=64, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv4l')

        self.conv1r = tf.layers.conv1d(inputs=self.right_eye, filters=16, kernel_size=16, strides=4, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv1r')
        self.conv2r = tf.layers.conv1d(inputs=self.conv1r, filters=8, kernel_size=8, strides=2, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv2r')
        self.conv3r = tf.layers.conv1d(inputs=self.conv2r, filters=8, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv3r')
        self.conv4r = tf.layers.conv1d(inputs=self.conv3r, filters=64, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv4r')

        self.conv4l_flat = tf.layers.flatten(self.conv4l)
        self.conv4r_flat = tf.layers.flatten(self.conv4r)

        self.conv_with_states = tf.concat(
            [self.conv4l_flat, self.conv4r_flat], 1, name="flattened_conv")
        activ = tf.nn.relu

        # Recurrent Layer
        self.rnn_in = activ(linear(self.conv_with_states, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
        # self.rnn_in = tf.layers.dense(self.conv_with_states, 512, activation=tf.nn.relu,
        #                               kernel_initializer=tf.orthogonal_initializer,
        #                               trainable=True, name=my_scope + '_rnn_in')
        self.convFlat = batch_to_seq(self.rnn_in, self.n_env, self.n_steps)
        self.masks = batch_to_seq(self.dones_ph, self.n_env, self.n_steps)

        self.rnn, self.snew = lstm(self.convFlat, self.masks, self.states_ph, 'lstm1', n_hidden=512,
                                             layer_norm=False)
        self.rnn_output = seq_to_batch(self.rnn)

        self.action_stream, self.value_stream = tf.split(self.rnn_output, 2, 1)

        return self.action_stream, self.value_stream

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """tf.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """tf.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

