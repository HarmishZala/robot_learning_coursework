"""
Evaluation script for the BabyBench Self-Touch PPO agent.
Loads a trained model, runs evaluation episodes, and saves:
  - BabyBench trajectory logs (native format)
  - A per-episode touch CSV with body-zone breakdown (for analysis.py)

Usage:
    python evaluate_selftouch.py --seed 0
    python evaluate_selftouch.py --seed 0 --episodes 10 --duration 1000
    python evaluate_selftouch.py --all_seeds   (runs seeds 0, 1, 2)
"""

import os
import sys
import csv
import argparse
import yaml
import numpy as np

sys.path.append(".")
sys.path.append("..")
import mimoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils
import babybench.eval as bb_eval
from stable_baselines3 import PPO


def get_sim_dt(env) -> float:
    """Read the effective sim timestep from the env (base_dt × frame_skip)."""
    try:
        base_dt    = float(env.unwrapped.model.opt.timestep)
        frame_skip = int(getattr(env.unwrapped, "frame_skip", 1))
        return base_dt * frame_skip
    except Exception:
        return 0.004  # MuJoCo default fallback

# ─── Body-zone classification (mirrors ppo_selftouch.py) ─────────────────────

ZONE_WEIGHTS = {
    "head":   0.40,
    "torso":  0.30,
    "arm":    0.15,
    "hand":   0.10,
    "leg":    0.05,
    "foot":   0.00,
}

def classify_body(name):
    n = name.lower()
    if any(h in n for h in ["head", "eye"]):            return "head"
    if any(t in n for t in ["body", "hip", "torso"]):   return "torso"
    if "finger" in n or "hand" in n:                    return "hand"
    if "upper_arm" in n or "lower_arm" in n:            return "arm"
    if "foot" in n:                                      return "foot"
    if "leg" in n:                                       return "leg"
    return "other"


# ─── Touch logger ─────────────────────────────────────────────────────────────

class TouchLogger:
    """
    Counts per-zone touches during an evaluation episode.

    Parameters
    ----------
    sim_dt : float
        Effective simulation timestep in seconds (base_dt × frame_skip).
        Used to convert step counts to real time (minutes) for touches_per_min.
        Read from the env — never hardcoded.
    """

    def __init__(self, sim_dt: float):
        self.sim_dt = sim_dt
        self.reset()

    def reset(self):
        self.counts = {z: 0 for z in ZONE_WEIGHTS}
        self.steps  = 0

    def record(self, env_unwrapped):
        """Call once per timestep after env.step()."""
        self.steps += 1
        try:
            touch_outputs = env_unwrapped.touch.sensor_outputs
        except AttributeError:
            return
        for body_id, force_vectors in touch_outputs.items():
            magnitudes = np.linalg.norm(force_vectors, axis=-1)
            if not np.any(magnitudes > 1e-6):
                continue
            try:
                body_name = env_unwrapped.model.body(body_id).name
            except Exception:
                continue
            zone = classify_body(body_name)
            if zone in self.counts:
                self.counts[zone] += 1

    def summary(self, seed: int, episode: int) -> dict:
        total        = sum(self.counts.values())
        duration_min = (self.steps * self.sim_dt) / 60.0
        tpm          = total / duration_min if duration_min > 0 else 0.0
        return {
            "episode":          episode,
            "seed":             seed,
            "steps":            self.steps,
            "sim_dt":           self.sim_dt,
            "touches_head":     self.counts["head"],
            "touches_torso":    self.counts["torso"],
            "touches_arm":      self.counts["arm"],
            "touches_hand":     self.counts["hand"],
            "touches_leg":      self.counts["leg"],
            "touches_foot":     self.counts["foot"],
            "total_touches":    total,
            "touches_per_min":  round(tpm, 2),
        }


# ─── Evaluate one seed ────────────────────────────────────────────────────────

def evaluate_seed(config, seed, episodes, duration, render):
    seed_dir = os.path.join(config["save_dir"], f"seed_{seed}")
    config_seed = dict(config)
    config_seed["save_dir"] = seed_dir

    model_path = os.path.join(seed_dir, "model.zip")
    if not os.path.exists(model_path):
        print(f"  [skip] No trained model found for seed {seed} at {model_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Evaluating seed {seed}  |  {episodes} episodes × {duration} steps")
    print(f"{'='*60}")

    model = PPO.load(model_path)
    env   = bb_utils.make_env(config_seed, training=False)
    env.reset()

    # BabyBench native evaluator (trajectory logs + optional video)
    evaluation = bb_eval.EVALS[config["behavior"]](
        env      = env,
        duration = duration,
        render   = render,
        save_dir = seed_dir,
    )
    evaluation.eval_logs()  # summarise training logs

    # Our own per-zone touch logger — read sim_dt from the live env
    sim_dt       = get_sim_dt(env)
    touch_logger = TouchLogger(sim_dt)
    csv_path = os.path.join(seed_dir, "logs", f"episode_log_seed{seed}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["episode", "seed", "steps", "sim_dt",
                  "touches_head", "touches_torso", "touches_arm",
                  "touches_hand", "touches_leg",   "touches_foot",
                  "total_touches", "touches_per_min"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep_idx in range(episodes):
            print(f"  Episode {ep_idx + 1}/{episodes} ...", end=" ", flush=True)
            obs, _ = env.reset()
            evaluation.reset()
            touch_logger.reset()

            for _ in range(duration):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _, info = env.step(action)
                evaluation.eval_step(info)
                touch_logger.record(env.unwrapped)

            row = touch_logger.summary(seed=seed, episode=ep_idx + 1)
            writer.writerow(row)
            f.flush()
            print(f"touches={row['total_touches']:>4}  ({row['touches_per_min']:.1f}/min)")

            evaluation.end(episode=ep_idx)

    env.close()
    print(f"\n  Touch log saved → {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Self-Touch PPO agent")
    parser.add_argument("--config",     default="examples/config_selftouch.yml", type=str)
    parser.add_argument("--seed",       default=0,    type=int,
                        help="Which seed model to evaluate (ignored if --all_seeds)")
    parser.add_argument("--all_seeds",  action="store_true",
                        help="Evaluate all seeds (0, 1, 2) sequentially")
    parser.add_argument("--render",     default=True, type=bool)
    parser.add_argument("--duration",   default=1000, type=int,
                        help="Timesteps per evaluation episode")
    parser.add_argument("--episodes",   default=10,   type=int,
                        help="Number of evaluation episodes per seed")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seeds = [0, 1, 2] if args.all_seeds else [args.seed]
    for s in seeds:
        evaluate_seed(config, s, args.episodes, args.duration, args.render)

    print("\nEvaluation complete. Run analysis.py to generate figures.")


if __name__ == "__main__":
    main()
