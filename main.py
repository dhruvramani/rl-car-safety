import os
import gym
import numpy as np
import tensorflow as tf

from config import argparser
from gym.envs.box2d import CarRacing 
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl import trainer_model_based

from base_wm import EnvModel
from utils import make_env as env_fn, printstar
import multiprocessing as mp

# NOTE : At every Kth step, create a new tree with K time steps 
#       and check if it leads to a state in which 
#       the car is off-track _is_outside()

# NOTE : Will have to install envs again after changing

def main(config):
    global env_fn
    env = env_fn()()
    env.reset()
    
    action_dim = 5
    ob_shape = env.observation_space.shape
    world_model_path = os.path.expanduser(os.path.join(config.model_dir, config.world_model_type + "_" + config.world_model_path))

    if(config.train_world_model):
        if(config.world_model_type == "t2t"):
            # NOTE : Incomplete implementation; gave up temprroily
            hp = trainer_model_based_params.rlmb_base()
            hp.game = "CarRacing"
            hp.epochs = 1
            hp.ppo_epochs_num = 0 
            trainer_model_based.training_loop(env_fn, hp, world_model_path)  

        elif(config.world_model_type == "base"):        
            env_model = EnvModel(ob_shape, action_dim, config)
            if(not os.path.exists(world_model_path) or config.train_world_model):
                if(not os.path.exists(world_model_path)):
                    os.mkdir(world_model_path)
                printstar("Training Base Policy")
                env_model.train(world_model_path)

    if(config.eval_safety):
        evaluate_agent(env, config)

def evaluate_agent(env, config, policy=None, safety_graph=None, path=None):
    printstar("Testing Agent")
    obs = env.reset()
    for t in range(config.max_eval_iters):
        if(policy is None):
            action = env.action_space.sample()
        obs, reward, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    config = argparser()
    mp.set_start_method('spawn', force=True)
    main(config)
