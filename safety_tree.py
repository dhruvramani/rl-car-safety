import os
import sys
import gym
import time
import copy
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from small_network import *

def car_outside(env):
    right = env.info['count_right'] > 0
    left  = env.info['count_left']  > 0
    if (left|right).sum() == 0:
        return True
    else:
        return False

def change_state(env, car_state):
    env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1] = car_state
    new_obs = env.render("state_pixels")
    env._update_state(new_obs)

class ImaginedNode(object):
    def __init__(self, imagined_state, imagined_reward=None):
        self.imagined_state  = imagined_state
        self.imagined_reward = imagined_reward
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

# TODO : Make changes
def is_unsafe(root, env):
    if(root is not None):
        imagined_state = root.imagined_state
        change_state(env, imagined_state)
        
        if(car_outside(env) == True):
            return True
        
        unsafe_founds = []
        for child in root.children:
            found = is_unsafe(child, env)
            if(found == True):
                return found

    return False

def generate_tree(state, config, env, count=0):
    action_dim = 5
    node = ImaginedNode(state)
    if(count > config.tree_size):
        node.children.extend([None for i in range(action_dim)])
        return node

    init_state = state
    change_state(env, init_state)
    for action in range(1, action_dim-1):
        obs, _, _, _ = env.step(action)
        state = [env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1]]
        if(False not in [state[i] == init_state[i] for i in range(len(state))]):
            node.add_child(None)
            continue
        node.add_child(generate_tree(state, config, env, count + 1))
        change_state(env, init_state)
    change_state(env, init_state)
    return node

def move_node(root, action):
    return root.children[action]

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


def safe_action(agent, tree, base_state, unsafe_action, world_model):
    possible_actions = [i for i in range(world_model.action_dim) if i != unsafe_action]
    imagined_states =  {a : tree.children[a].imagined_state for a in possible_actions if search_node(tree.children[a], base_state) == True}
    values = {a : agent.critique(imagined_states[a].reshape(1, nw, nh, nc)) for a in imagined_states.keys()}
    max_a = max(values.keys(), key=lambda a:values[a])
    return [max_a]
