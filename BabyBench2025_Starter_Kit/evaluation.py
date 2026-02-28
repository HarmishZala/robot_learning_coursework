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
import babybench.utils as bb_utils
import babybench.eval as bb_eval

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_test_installation.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--render', default=True,  type=bool,
                        help='Renders a video for each episode during the evaluation.')
    parser.add_argument('--duration', default=1000, type=int,
                        help='Total timesteps per evaluation episode')
    parser.add_argument('--episodes', default=10, type=int,
                        help='Number of evaluation episode')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config, training=False)
    env.reset()

    # Initialize evaluation object
    evaluation = bb_eval.EVALS[config['behavior']](
        env=env,
        duration=args.duration,
        render=args.render,
        save_dir=config['save_dir'],
    )

    # Preview evaluation of training log
    evaluation.eval_logs()

    for ep_idx in range(args.episodes):
        print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')

        # Reset environment and evaluation
        obs, _ = env.reset()
        evaluation.reset()

        for t_idx in range(args.duration):

            # Select action
            action = env.action_space.sample()

            # ---------------------------------------------------# 
            #                                                    #
            # TODO REPLACE WITH CALL TO YOUR TRAINED POLICY HERE #
            # action = policy(obs)                               #
            #                                                    #
            # ---------------------------------------------------#

            # Perform step in simulation
            obs, _, _, _, info = env.step(action)

            # Perform evaluations of step
            evaluation.eval_step(info)
            
        evaluation.end(episode=ep_idx)

if __name__ == '__main__':
    main()
