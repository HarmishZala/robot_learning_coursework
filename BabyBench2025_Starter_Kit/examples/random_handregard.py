"""
Example: Random policy for hand-regard baseline
"""
import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_handregard.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    
    steps = 0
    obs, _ = env.reset()
    while steps < args.train_for:
        steps += 1
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            obs,_ = env.reset()

    env.close()

if __name__ == '__main__':
    main()
