"""
record_clip.py
--------------
Record a short MP4 clip of the trained PPO agent (MIMo) touching itself.
The clip uses BabyBench's own evaluation_img() which gives a nice 3-panel
view: corner camera + top camera + touch sensor heatmap.

Usage:
    env MUJOCO_GL=egl python record_clip.py --seed 0 --steps 500
    env MUJOCO_GL=egl python record_clip.py --seed 1 --steps 300

Output:
    results/self_touch/seed_<seed>/videos/mimo_clip_seed<seed>.mp4
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.append(".")
sys.path.append("..")
import mimoEnv                         # noqa: F401  (registers gym envs)
import babybench.utils as bb_utils
from stable_baselines3 import PPO


def record(seed: int, steps: int, config_path: str, fps: int = 30):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed_dir   = os.path.join(config["save_dir"], f"seed_{seed}")
    model_path = os.path.join(seed_dir, "model.zip")

    if not os.path.exists(model_path):
        print(f"[error] No model at {model_path}")
        sys.exit(1)

    video_dir  = os.path.join(seed_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    save_path  = os.path.join(video_dir, f"mimo_clip_seed{seed}.mp4")

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    config_seed = dict(config)
    config_seed["save_dir"] = seed_dir
    env = bb_utils.make_env(config_seed, training=False)
    obs, _ = env.reset()

    print(f"Recording {steps} steps → {save_path}")
    frames = []
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        # BabyBench composite frame: corner + top + touch map
        img = bb_utils.evaluation_img(env, up="side2", down="touches_with_hands")
        frames.append(img)

        if terminated or truncated:
            obs, _ = env.reset()

        if step % 50 == 0:
            print(f"  step {step}/{steps}", flush=True)

    env.close()

    bb_utils.evaluation_video(frames, save_name=save_path, frame_rate=fps)
    print(f"\nDone! Clip saved to: {save_path}")
    print(f"  {len(frames)} frames @ {fps} fps = {len(frames)/fps:.1f} s")


def main():
    parser = argparse.ArgumentParser(description="Record MIMo self-touch clip")
    parser.add_argument("--seed",   default=0,    type=int,
                        help="Which trained seed to load")
    parser.add_argument("--steps",  default=500,  type=int,
                        help="Number of environment steps to record")
    parser.add_argument("--fps",    default=30,   type=int,
                        help="Frame rate of output video")
    parser.add_argument("--config", default="examples/config_selftouch.yml",
                        help="BabyBench config YAML")
    args = parser.parse_args()

    record(seed=args.seed, steps=args.steps,
           config_path=args.config, fps=args.fps)


if __name__ == "__main__":
    main()
