import os
import gym
import copy
import numpy as np
import tensorflow as tf

from config import argparser
from gym.envs.box2d import CarRacing 
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl import trainer_model_based

import base_wm
import small_network
from utils import make_env as env_fn, printstar
from safety_tree import *

import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# NOTE : At every Kth step, create a new tree with K time steps 
#       and check if it leads to a state in which 
#       the car is off-track _is_outside()

# NOTE : Will have to install envs again after changing

g_env_model = None

def main(config):
    global env_fn
    env = env_fn()()
    env.reset()
    
    action_dim = 5
    ob_shape = env.observation_space.shape
    world_model_path = os.path.expanduser(os.path.join(config.model_dir, config.world_model_type + "_" + config.world_model_path))

    if(config.train_world_model):
        if(config.world_model_type == "small"):
            ob_shape = 5
            env_model = small_network.SmallEnvModel(ob_shape, action_dim, config)
        elif(config.world_model_type == "base"):        
            env_model = base_wm.EnvModel(ob_shape, action_dim, config)

        if(not os.path.exists(world_model_path) or config.train_world_model):
            if(not os.path.exists(world_model_path)):
                os.mkdir(world_model_path)
            printstar("Training Base Policy")
            env_model.train(world_model_path)

    if(config.debug):
        debug_test(env, config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if(config.eval_world_model):
            if(config.world_model_type == "small"):
                env_model = small_network.get_cache_loaded_env_model(sess, 5, action_dim, config, world_model_path + '/small_env_model.ckpt')
            elif(config.world_model_type == "base"):        
                env_model = base_wm.get_cache_loaded_env_model(sess, ob_shape, action_dim, config, world_model_path + '/env_model.ckpt')
            evaluate_world_model(env, sess, env_model, config)        

        if(config.eval_safety):
            evaluate_agent(env, config)

def evaluate_world_model(env, sess, world_model, config, policy=None):
    printstar("Testing World Model")
    obs = env.reset()
    for t in range(config.max_eval_iters):
        if(policy is None):
            if(t % 2 == 0):
                action = 1
            else:
                action = 2
        
        if(config.world_model_type == "base"):
            next_pred_ob = world_model.magine(sess, obs, action)
            if(config.debug):
                obs = obs.reshape(world_model.width, world_model.height, world_model.depth)
                imgplot = plt.imshow(obs)
                plt.savefig('./figs/obs.png')

            imgplot = plt.imshow(next_pred_ob)
            plt.savefig('./figs/world_model_eval.png')
            
            if(config.debug):
                printstar("Obs\n\n{}".format(obs))
                printstar("Next Pred\n\n{}".format(next_pred_ob))

        elif(config.world_model_type == "small"):
            next_pred_ob = world_model.imagine(sess, env, action)
            small_network.change_state(env, next_pred_ob)

        env.render()
        obs, reward, dones, info = env.step(action)
        inp = input("E to exit : ").lower()
        if(inp == "e"):
            break

def evaluate_agent(env, config, policy=None, safety_graph=None, path=None):
    printstar("Testing Agent")
    obs = env.reset()
    state = [env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1]]
    tree = generate_tree(state, config, env)
    for t in range(1, config.max_eval_iters):
        if(t % 2 == 0):
            action = 1
        else:
            action = 2
        state = [env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1]]
        if(t % config.tree_size == 0):
            print("Generating diff tree")
            tree = generate_tree(state, config, env)

        next_node = move_node(tree, action)
        if(is_unsafe(next_node, env)):
            print("Unsafe action")
        print(action)
        obs, reward, dones, info = env.step(action)
        env.render()

def debug_test(env, config):
    printstar("Debug")
    obs = env.reset()
    init_car = [env.car.hull.position[0], env.car.hull.position[1], env.car.hull.angle, env.car.hull.linearVelocity[0], env.car.hull.linearVelocity[1]]#, env.info]
    print(init_car[3][0])
    obs, reward, dones, info = env.step(1)
    for _ in range(100):
        obs, reward, dones, info = env.step(1)
        env.render()
        obs, reward, dones, info = env.step(3)
        env.render()
    
    _ = input("Moving state")
    print(env._is_outside())
    small_network.change_state(env, init_car)
    for _ in range(2):
        obs, reward, dones, info = env.step(env.action_space.sample())
        env.render()
    print(env._is_outside())
    _ = input("Moved state")

if __name__ == '__main__':
    config = argparser()
    mp.set_start_method('spawn', force=True)
    main(config)
