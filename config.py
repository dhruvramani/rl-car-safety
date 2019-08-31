import os
import argparse

def str2bool(v):
    return v.lower() == 'true'

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

def argparser():
    parser = argparse.ArgumentParser("Model based safety",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_world_model', type=str2bool, default=True)

    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--policy_path', type=str, default='CarRacingPolicy')
    parser.add_argument('--world_model_path', type=str, default="CarRacingWorldModel")

    parser.add_argument('--eval_safety', type=str2bool, default=True)
    parser.add_argument('--tree_size', type=int, default=10)
    parser.add_argument('--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('--max_eval_iters', type=int, default=int(1e3))
    
    parser.add_argument('--render', type=str2bool, default=True, help='Render frames')
    parser.add_argument('--debug', type=str2bool, default=False, help='See debugging info')

    args = parser.parse_args()
    return args
