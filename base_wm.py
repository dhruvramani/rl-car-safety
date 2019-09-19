import os
import gym 
import numpy as np
import tensorflow as tf
from gym.envs.box2d import CarRacing 

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_video

from tqdm import tqdm
from utils import SubprocVecEnv

make_env = lambda : CarRacing(
    grayscale=0,
    show_info_panel=0,
    discretize_actions="hard",
    frames_per_state=1,
    num_lanes=1,
    num_tracks=1)

# TODO : TRAIN ON MULTIPLE AGENTS
# TODO : RESET AFTER CERTAIN STEPS
# TODO : Write testing function for env_model
# TODO : Write the tree code
# TODO : Write code for safety check

def inject_additional_input(layer, inputs, name, mode="multi_additive"):
  """Injects the additional input into the layer.

  Args:
    layer: layer that the input should be injected to.
    inputs: inputs to be injected.
    name: TF scope name.
    mode: how the infor should be added to the layer:
      "concat" concats as additional channels.
      "multiplicative" broadcasts inputs and multiply them to the channels.
      "multi_additive" broadcasts inputs and multiply and add to the channels.

  Returns:
    updated layer.

  Raises:
    ValueError: in case of unknown mode.
  """
  layer_shape = common_layers.shape_list(layer)
  input_shape = common_layers.shape_list(inputs)
  zeros_mask = tf.zeros(layer_shape, dtype=tf.float32)
  if mode == "concat":
    emb = common_video.encode_to_shape(inputs, layer_shape, name)
    layer = tf.concat(values=[layer, emb], axis=-1)
  elif mode == "multiplicative":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mask = tf.layers.dense(input_reshaped, filters, name=name)
    input_broad = input_mask + zeros_mask
    layer *= input_broad
  elif mode == "multi_additive":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mul = tf.layers.dense(input_reshaped, filters, name=name + "_mul")
    layer *= tf.nn.sigmoid(input_mul)
    input_add = tf.layers.dense(input_reshaped, filters, name=name + "_add")
    layer += input_add
  else:
    raise ValueError("Unknown injection mode: %s" % mode)

  return layer

class EnvModel(object):
    def __init__(self, obs_shape, action_dim, num_rewards=1, n_envs=16, is_policy=False, has_rewards=False, 
        hidden_size=64, n_layers=6, dropout_p=0.1, activation_fn=tf.nn.relu, max_ep_len=50000,
        l2_clip=10.0, softmax_clip=0.03, reward_coeff=0.1, should_summary=True, log_interval=100):
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.layers = n_layers
        self.dropout_p = dropout_p
        self.activation_fn = activation_fn
        self.l2_clip = l2_clip
        self.softmax_clip = softmax_clip
        self.reward_coeff = reward_coeff
        self.n_envs = n_envs
        self.max_ep_len = max_ep_len
        self.log_interval = log_interval

        self.is_policy = is_policy
        self.has_rewards = has_rewards

        self.width, self.height, self.depth = self.obs_shape

        self.states_ph = tf.placeholder(tf.float32, [None, self.width, self.height, self.depth])
        self.actions_ph = tf.placeholder(tf.uint8, [None, 1])
        self.actions_oph = tf.one_hot(self.actions_ph, depth=action_dim)
        self.target_states = tf.placeholder(tf.float32, [None, self.width, self.height, self.depth])
        if(self.has_rewards):
            self.target_rewards = tf.placeholder(tf.uint8, [None, self.num_rewards])
        
        # NOTE - Implement policy and value parts later
        with tf.variable_scope("env_model"):
            self.state_pred, self.reward_pred, _, _ = self.network()

        # NOTE - Change this maybe to video_l2_loss
        self.state_loss = tf.math.maximum(tf.reduce_mean(tf.pow(self.state_pred - self.target_states, 2)), self.l2_clip)
        self.loss = self.state_loss

        if(self.has_rewards):
            self.reward_loss = tf.math.maximum(tf.reduce_mean(tf.losses.softmax_cross_entropy(self.tw_one_hot, self.reward_pred)), self.softmax_clip)
            self.loss = self.loss + (self.reward_coeff * self.reward_loss)

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        if should_summary:
            tf.summary.scalar('loss', self.loss)
            if(self.has_rewards):
                tf.summary.scalar('image_loss', self.state_loss)
                tf.summary.scalar('reward_loss', self.reward_loss)

    def generate_data(self, envs, max_ep_len, n_envs):
        states = envs.reset()
        for frame_idx in range(max_ep_len):
            states = states.reshape(1, self.width, self.height, self.depth)
            #actions = [envs.action_space.sample() for _ in range(n_envs)]
            actions = envs.action_space.sample()
            #actions, _, _ = actor_critic.act(states)
            next_states, rewards, dones, _ = envs.step(actions)
            next_states = next_states.reshape(1, self.width, self.height, self.depth)

            yield frame_idx, states, actions, rewards, next_states, dones
            states = next_states

    def network(self):
        def middle_network(layer):
            # Run a stack of convolutions.
            x = layer
            kernel1 = (3, 3)
            filters = common_layers.shape_list(x)[-1]
            for i in range(2):
              with tf.variable_scope("layer%d" % i):
                y = tf.nn.dropout(x, 1.0 - 0.5)
                y = tf.layers.conv2d(y, filters, kernel1, activation=self.activation_fn,
                                     strides=(1, 1), padding="SAME")
                if i == 0:
                  x = y
                else:
                  x = common_layers.layer_norm(x + y)
            return x

        batch_size = tf.shape(self.states_ph)[0]

        filters = self.hidden_size
        kernel2 = (4, 4)
        action = self.actions_oph#[0] NOTE - might remove this

        # Normalize states. - NOTE might remove the list comprehension
        #states = [common_layers.standardize_images(f) for f in self.states_ph]
        #stacked_states = tf.concat(states, axis=-1)

        stacked_states = common_layers.standardize_images(self.states_ph)
        inputs_shape = common_layers.shape_list(stacked_states)

        # Using non-zero bias initializer below for edge cases of uniform inputs.
        x = tf.layers.dense(
            stacked_states, filters, name="inputs_embed",
            bias_initializer=tf.random_normal_initializer(stddev=0.01))
        x = common_attention.add_timing_signal_nd(x)

        # Down-stride.
        layer_inputs = [x]
        for i in range(self.layers):
          with tf.variable_scope("downstride%d" % i):
            layer_inputs.append(x)
            x = tf.nn.dropout(x, 1.0 - self.dropout_p)
            x = common_layers.make_even_size(x)
            if i < 2:
              filters *= 2
            x = common_attention.add_timing_signal_nd(x)
            x = tf.layers.conv2d(x, filters, kernel2, activation=self.activation_fn,
                                 strides=(2, 2), padding="SAME")
            x = common_layers.layer_norm(x)

        if self.is_policy:
          with tf.variable_scope("policy"):
            x_flat = tf.layers.flatten(x)
            policy_pred = tf.layers.dense(x_flat, self.action_dim)
            value_pred = tf.layers.dense(x_flat, 1)
            value_pred = tf.squeeze(value_pred, axis=-1)
        else:
          policy_pred, value_pred = None, None

        #if self.has_actions:
        x = inject_additional_input(x, action, "action_enc", "multi_additive")

        # Inject latent if present. Only for stochastic models.
        target_states = common_layers.standardize_images(self.target_states)

        x_mid = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = middle_network(x)

        # Up-convolve.
        layer_inputs = list(reversed(layer_inputs))
        for i in range(self.layers):
          with tf.variable_scope("upstride%d" % i):
            x = tf.nn.dropout(x, 1.0 - 0.1)
            if i >= self.layers - 2:
              filters //= 2
            x = tf.layers.conv2d_transpose(
                x, filters, kernel2, activation=self.activation_fn,
                strides=(2, 2), padding="SAME")
            y = layer_inputs[i]
            shape = common_layers.shape_list(y)
            x = x[:, :shape[1], :shape[2], :]
            x = common_layers.layer_norm(x + y)
            x = common_attention.add_timing_signal_nd(x)

        # Cut down to original size.
        x = x[:, :inputs_shape[1], :inputs_shape[2], :]
        x_fin = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        
        x = tf.layers.dense(x, self.depth, name="logits")

        reward_pred = None
        if self.has_rewards:
          # Reward prediction based on middle and final logits.
          reward_pred = tf.concat([x_mid, x_fin], axis=-1)
          reward_pred = tf.nn.relu(tf.layers.dense(
              reward_pred, 128, name="reward_pred"))
          reward_pred = tf.squeeze(reward_pred, axis=1)  # Remove extra dims
          reward_pred = tf.squeeze(reward_pred, axis=1)  # Remove extra dims

        return x, reward_pred, policy_pred, value_pred

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            losses = []
            all_rewards = []
            save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
            saver = tf.train.Saver(var_list=save_vars)

            train_writer = tf.summary.FileWriter('./env_logs/train/', graph=sess.graph)
            summary_op = tf.summary.merge_all()

            envs = make_env()
            #envs = [make_env() for i in range(self.n_envs)]
            #envs = SubprocVecEnv(envs)

            for idx, states, actions, rewards, next_states, dones in tqdm(
                self.generate_data(envs, self.max_ep_len, self.n_envs), total=self.max_ep_len):
                actions = np.array(actions)
                actions = np.reshape(actions, (-1, 1))

                if(self.has_rewards):
                    target_reward = reward_to_target(rewards)
                    loss, reward_loss, state_loss, summary, _ = sess.run([self.loss, self.reward_loss, self.state_loss,
                        summary_op, self.opt], feed_dict={
                        self.states_ph: states,
                        self.actions_ph: actions,
                        self.target_states: next_states,
                        self.target_rewards: target_reward
                    })
                else :
                    loss, summary, _ = sess.run([self.loss, summary_op, self.opt], feed_dict={
                        self.states_ph: states,
                        self.actions_ph: actions,
                        self.target_states: next_states,
                    })

                if idx % self.log_interval == 0:
                    if(self.has_rewards):
                        print('%i => Loss : %.4f, Reward Loss : %.4f, Image Loss : %.4f' % (idx, loss, reward_loss, state_loss))
                    else :
                        print('%i => Loss : %.4f' % (idx, loss))

                train_writer.add_summary(summary, idx)

            saver.save(sess, 'weights/env_model.ckpt')
            print('Environment model saved')
