import os
import sys
import gym
import time
import copy
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from base_wm import EnvModel

class ImaginedNode(object):
    def __init__(self, imagined_state, imagined_reward=None):
        self.imagined_state  = imagined_state
        self.imagined_reward = imagined_reward
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

def car_outside(env):
    right = env.info['count_right'] > 0
    left  = env.info['count_left']  > 0
    if (left|right).sum() == 0:
        return True
    else:
        return False

def imagine(sess, world_model, obs, action):
    action = np.array(action)
    action = np.reshape(action, (1, 1))
    obs = obs.reshape(1, world_model.width, world_model.height, world_model.depth)    
    next_pred_ob = sess.run(world_model.state_pred, feed_dict={world_model.states_ph : obs, world_model.actions_ph : action})
    next_pred_ob = next_pred_ob.reshape(world_model.width, world_model.height, world_model.depth)
    return next_pred_ob

# TODO : Make changes
def search_node(root, base_state):
    if(root is not None):
        #print(root.imagined_state.reshape(nc, nw, nh))
        imagined_state = copy.deepcopy(root.imagined_state)
        imagined_state = imagined_state.reshape(nc, nw, nh)
        #imagined_state[np.where(imagined_state == 2.0)] = 1.0
        if(np.array_equal(imagined_state, base_state) and root.imagined_reward != END_REWARD):
            return True
        for child in root.children:
            found = search_node(child, base_state)
            if(found == True):
                return found
    return False

def generate_tree(sess, state, config, world_model, count=0):
    nw, nh, nc = world_model.width, world_model.height, world_model.depth
    action_dim = world_model.action_dim

    state = state.reshape(nw, nh, nc)
    node = ImaginedNode(state)
    if(count > config.tree_size):
        node.children.extend([None for i in range(action_dim)])
        return node

    for action in range(action_dim):
        imagined_state = imagine(sess, world_model, state, action)
        if(np.array_equal(state, imagined_state)):
            node.add_child(None)
            continue
        node.add_child(generate_tree(sess, imagined_state, config, world_model, count + 1))

    return node

def safe_action(agent, tree, base_state, unsafe_action, world_model):
    possible_actions = [i for i in range(world_model.action_dim) if i != unsafe_action]
    imagined_states =  {a : tree.children[a].imagined_state for a in possible_actions if search_node(tree.children[a], base_state) == True}
    values = {a : agent.critique(imagined_states[a].reshape(1, nw, nh, nc)) for a in imagined_states.keys()}
    max_a = max(values.keys(), key=lambda a:values[a])
    return [max_a]

def get_node(root, state):
    #state = state.reshape(nc, nw, nh)
    current_node = copy.deepcopy(root)
    queue = []
    queue.append(current_node)
    while len(queue) != 0:
        current_node = queue.pop(0)
        curr_state = copy.deepcopy(current_node.imagined_state)
        #curr_state = curr_state.reshape(nc, nw, nh)

        if(np.array_equal(curr_state, state)):
            return current_node

        for child in current_node.children:
            if(child is not None):
                queue.append(child)
    return None
