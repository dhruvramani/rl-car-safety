import gym
import numpy as np
import tensorflow as tf

from config import argparser
from gym.envs.box2d import CarRacing 

def main(config):

    env_fn = lambda : CarRacing(
        grayscale=1,
        show_info_panel=0,
        discretize_actions="hard",
        frames_per_state=4,
        num_lanes=1,
        num_tracks=1)
    env = env_fn()

    if(config.eval_safety):
        evaluate_agent(env, config)

def evaluate_agent(env, config, policy=None, safety_graph=None, path=None):
    obs = env.reset()
    for t in range(config.max_eval_iters):
        if(policy is None):
            action = env.action_space.sample()
        obs, reward, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    config = argparser()
    main(config)
