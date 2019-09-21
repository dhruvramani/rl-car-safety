import os
import gym 
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utils import SubprocVecEnv, make_env

g_env_model = None
def get_cache_loaded_env_model(sess, ob_shape, action_dim, config, path):
    global g_env_model
    if g_env_model is None:
        g_env_model = SmallEnvModel(ob_shape, action_dim, config)
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='small_env_model')
        loader = tf.train.Saver(var_list=save_vars)
        loader.restore(sess, path)
        print('Small Env model restored')

    return g_env_model

def change_state(env, car_state):
    env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1] = car_state
    new_obs = env.render("state_pixels")
    env._update_state(new_obs)

class SmallEnvModel(object):
    def __init__(self, obs_shape, action_dim, config):
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config

        self.hidden_size = config.s_hidden_size
        self.layers = config.s_n_layers
        self.dropout_p = config.dropout_p
        
        if(config.activation_fn == 'relu'):
            self.activation_fn = tf.nn.relu
        elif (config.activation_fn == 'tanh'):
            self.activation_fn = tf.nn.tanh
        
        self.l2_clip = config.l2_clip
        self.softmax_clip = config.softmax_clip
        self.reward_coeff = config.reward_coeff
        self.n_envs = 1
        self.max_ep_len = config.s_max_ep_len
        self.log_interval = config.log_interval

        self.has_rewards = False
        self.num_rewards = 1

        self.states_ph = tf.placeholder(tf.float32, [1, self.obs_shape])
        self.actions_ph = tf.placeholder(tf.uint8, [1])
        self.actions_oph = tf.cast(tf.one_hot(self.actions_ph, depth=action_dim), tf.float32)
        self.inputs = tf.concat([self.states_ph, self.actions_oph], axis=-1)
        
        self.target_states = tf.placeholder(tf.float32, [None, self.obs_shape])
        if(self.has_rewards):
            self.target_rewards = tf.placeholder(tf.uint8, [None, self.num_rewards])
        
        # NOTE - Implement policy and value parts later
        with tf.variable_scope("small_env_model"):
            self.state_pred, self.reward_pred = self.network()

        # NOTE - Change this maybe to video_l2_loss
        self.state_loss = tf.math.maximum(tf.reduce_sum(tf.pow(self.state_pred - self.target_states, 2)), self.l2_clip)
        self.loss = self.state_loss

        if(self.has_rewards):
            self.reward_loss = tf.math.maximum(tf.reduce_mean(tf.losses.softmax_cross_entropy(self.tw_one_hot, self.reward_pred)), self.softmax_clip)
            self.loss = self.loss + (self.reward_coeff * self.reward_loss)

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        if(self.has_rewards):
            tf.summary.scalar('image_loss', self.state_loss)
            tf.summary.scalar('reward_loss', self.reward_loss)

    def generate_data(self, envs):
    	states = envs.reset()
    	obs = np.asarray([envs.car.hull.position[0], envs.car.hull.position[1], envs.car.hull.angle, envs.car.hull.linearVelocity[0], envs.car.hull.linearVelocity[1]])
    	action_choice = 0
    	for frame_idx in range(1, self.max_ep_len):
    		new_action_choice = frame_idx % 300
    		if(new_action_choice == 0):
    			print("Changing policy")
    			states = envs.reset()
    			obs = np.asarray([envs.car.hull.position[0], envs.car.hull.position[1], envs.car.hull.angle, envs.car.hull.linearVelocity[0], envs.car.hull.linearVelocity[1]])
    			if(action_choice == 2):
    				action_choice = 0
    			else:
    				action_choice += 1   

    		if(action_choice == 0):
    			actions = envs.action_space.sample()

    		if(action_choice == 1):
    			if(frame_idx % 2 == 0):
    				actions = 1 
    			else:
    				actions = 3

    		if(action_choice == 2):
    			if(frame_idx % 2 == 0):
    				actions = 1 
    			else:
    				actions = 3

    		next_states, rewards, dones, _ = envs.step(actions)
    		next_obs = np.asarray([envs.car.hull.position[0], envs.car.hull.position[1], envs.car.hull.angle, envs.car.hull.linearVelocity[0], envs.car.hull.linearVelocity[1]])
    		obs = obs.reshape(1, self.obs_shape)
    		next_obs = next_obs.reshape(1, self.obs_shape)
    		yield frame_idx, obs, actions, rewards, next_obs, dones
    		states = next_states
    		obs = next_obs
    		if(self.n_envs == 1 and dones == True):
    			states = envs.reset()

    def imagine(self, sess, env, action):
        action = np.array(action)
        action = np.reshape(action, (1))
        obs = np.asarray([env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1]])
        obs = obs.reshape(1, self.obs_shape)
        next_pred_ob = sess.run(self.state_pred, feed_dict={self.states_ph : obs, self.actions_ph : action})
        next_pred_ob = next_pred_ob.reshape(self.obs_shape).tolist()
        return next_pred_ob

    def network(self):
    	x = self.inputs
    	for _ in range(self.layers):
    		x = tf.layers.dense(x, self.hidden_size, activation=self.activation_fn)
    	x = tf.layers.dense(x, self.obs_shape)
    	return x, None

    def train(self, world_model_path):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            losses = []
            all_rewards = []
            save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='small_env_model')
            saver = tf.train.Saver(var_list=save_vars)

            train_writer = tf.summary.FileWriter('./small_env_logs/train/', graph=sess.graph)
            summary_op = tf.summary.merge_all()

            if(self.n_envs == 1):
                envs = make_env()()
            else:
                envs = [make_env() for i in range(self.n_envs)]
                envs = SubprocVecEnv(envs)

            for idx, states, actions, rewards, next_states, dones in tqdm(
                self.generate_data(envs), total=self.max_ep_len):
                actions = np.array(actions)
                actions = np.reshape(actions, (1))

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
                    saver.save(sess, '{}/small_env_model.ckpt'.format(world_model_path))
                    print('Environment model saved')

                train_writer.add_summary(summary, idx)
            envs.close()
