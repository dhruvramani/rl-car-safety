import os
import sys
import gym
import time
import copy
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# NOTE : edit it up!

class ImaginedNode(object):
    def __init__(self, imagined_state, imagined_reward):
        self.imagined_state  = imagined_state
        self.imagined_reward = imagined_reward
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

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

def generate_tree(sess, state, reward=-1, count=0):
    # TODO : Recursion count 1, allow END REWARD
    nc, nw, nh = ob_space
    num_actions = ac_space.n
    # NOTE : Change this
    num_rewards = len(sokoban_rewards)

    env_model = get_cache_loaded_env_model(sess, ob_space, num_actions)

    imagination = ImaginationCore(num_actions, num_rewards,
                ob_space, env_model)

    state = state.reshape(-1, nw, nh, nc)
    node = ImaginedNode(state, reward)
    if(reward == END_REWARD or count > MAX_TREE_STEPS):
        node.children.extend([None for i in range(num_actions)])
        return node

    for action in range(num_actions):
        imagined_states, imagined_rewards = imagination.imagine(state, sess, action)
        imagined_state, imagined_reward = imagined_states[0][0, 0, :, :], sokoban_rewards[np.argmax(imagined_rewards[0], axis=1)[0]]
        if(np.array_equal(state.reshape(nw, nh), imagined_state)):
            node.add_child(None)
            continue
        imagined_state = imagined_state.reshape(-1, nw, nh, nc)
        node.add_child(generate_tree(sess, imagined_state, imagined_reward, count + 1))

    return node

def safe_action(agent, tree, base_state, unsafe_action):
    possible_actions = [i for i in range(ac_space.n) if i != unsafe_action]
    imagined_states =  {a : tree.children[a].imagined_state for a in possible_actions if search_node(tree.children[a], base_state) == True}
    values = {a : agent.critique(imagined_states[a].reshape(1, nw, nh, nc)) for a in imagined_states.keys()}
    for a in possible_actions:
        if(tree.children[a] is not None and tree.children[a].imagined_reward == END_REWARD):
            values[a] = END_REWARD
    max_a = max(values.keys(), key=lambda a:values[a])
    return [max_a]

def get_node(root, state):
    state = state.reshape(nc, nw, nh)
    current_node = copy.deepcopy(root)
    queue = []
    queue.append(current_node)
    while len(queue) != 0:
        current_node = queue.pop(0)
        curr_state = copy.deepcopy(current_node.imagined_state)
        curr_state = curr_state.reshape(nc, nw, nh)

        if(np.array_equal(curr_state, state)):
            return current_node

        for child in current_node.children:
            if(child is not None):
                queue.append(child)

    return None
