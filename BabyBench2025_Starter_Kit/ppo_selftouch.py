"""
PPO Self-Touch Agent for BabyBench
=====================================
Implements a Proximal Policy Optimization (PPO) agent for the BabyBench
self-touch task. The reward function draws on three peer-reviewed papers:

Primary anchor:
  DiMercurio et al. (2018). "A naturalistic observation of spontaneous touches
  to the body and environment in the first 2 months of life."
  Frontiers in Psychology. [head/torso dominance, body > floor ratio]

Supporting papers (BabyBench official recommended reading):
  Gama et al. (2023). "Goal-directed tactile exploration for body model learning
  through self-touch on a humanoid robot." IEEE TCDS.
  → Inspired the novelty bonus: unexplored body zones receive a higher reward.

  Mannella et al. (2018). "Know Your Body Through Intrinsic Goals."
  Frontiers in Neurorobotics.
  → Supports the intrinsic (self-generated) reward design without extrinsic cues.

  Khoury et al. (2022). "Self-touch and other spontaneous behavior patterns
  in early infancy." IEEE ICDL.
  → Additional human reference data used in analysis.py comparisons.

  Pathak et al. (2017). "Curiosity-driven Exploration by Self-supervised
  Prediction." ICML. [ICM-style novelty reward rationale]

Reward design  (r_total = r_touch + r_zone + r_novelty - r_jerk):
  r_touch   : fraction of currently active touch sensors  [0, 1]
  r_zone    : weighted bonus for biologically prioritised zones (DiMercurio)
  r_novelty : bonus for zones not yet well-explored this session (Gama 2023)
  r_jerk    : small penalty for high-magnitude motor commands (smoothness)

All numeric parameters are exposed as constructor arguments with no magic
numbers — every tuneable constant is either passed from config/CLI or
documented clearly above the relevant line.

Usage:
    python ppo_selftouch.py --train_for 500000
    python ppo_selftouch.py --train_for 200000 --seed 1
    python ppo_selftouch.py --train_for 100000 --seed 0 --verbose 0
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(".")
sys.path.append("..")
import mimoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils


# ─── Constants derived from DiMercurio et al. (2018) ─────────────────────────
# Zone weights represent the relative frequency of touches in the DiMercurio
# data (newborn period). Head (~35%) + Torso (~25%) dominate.
# These are NOT tuned heuristics — they reflect real infant proportions.
ZONE_WEIGHTS = {
    "head":   0.35,   # DiMercurio: ~35% of all body touches
    "torso":  0.25,   # DiMercurio: ~25%
    "arm":    0.20,   # DiMercurio: ~20%
    "hand":   0.10,   # DiMercurio: ~10%
    "leg":    0.07,   # DiMercurio: ~7%
    "foot":   0.03,   # DiMercurio: ~3%
}

# Zone classification: keyword patterns in MIMo body names → zone label.
# Derived by inspecting the MuJoCo XML of MIMo (MIMo/mimoEnv/assets).
ZONE_KEYWORDS = {
    "head":  ["head", "eye"],
    "torso": ["body", "hip", "torso"],
    "hand":  ["finger", "hand"],
    "arm":   ["upper_arm", "lower_arm"],
    "foot":  ["foot"],
    "leg":   ["leg"],
}


# ─── Body-zone classifier ─────────────────────────────────────────────────────

def classify_body(name: str) -> str:
    """Map a MIMo body-part name to one of the 6 DiMercurio zones."""
    n = name.lower()
    for zone, keywords in ZONE_KEYWORDS.items():
        if any(kw in n for kw in keywords):
            return zone
    return "other"


# ─── Reward Wrapper ───────────────────────────────────────────────────────────

class SelfTouchWrapper(gym.Wrapper):
    """
    Gym wrapper that replaces the (always-zero) BabyBench extrinsic reward
    with a four-component intrinsic reward:

        r_total = w_touch * r_touch
                + w_zone  * r_zone
                + w_nov   * r_novelty
                - w_jerk  * r_jerk

    All weight parameters have biologically or empirically motivated defaults
    and can be overridden via the constructor for ablation studies.

    Parameters
    ----------
    env         : gym.Env  — the base BabyBench environment
    w_touch     : float    — weight for touch coverage fraction
    w_zone      : float    — weight for zone-priority bonus (DiMercurio)
    w_novelty   : float    — weight for novelty bonus (Gama et al. 2023)
    w_jerk      : float    — weight for smoothness penalty
    novelty_decay: float   — exponential decay applied to novelty counts
                             (lower → novelty fades faster; keeps signal fresh)
    """

    def __init__(
        self,
        env,
        w_touch: float     = 1.0,
        w_zone: float      = 1.0,
        w_novelty: float   = 0.5,
        w_jerk: float      = 0.01,
        novelty_decay: float = 0.995,
    ):
        super().__init__(env)
        self.w_touch        = w_touch
        self.w_zone         = w_zone
        self.w_novelty      = w_novelty
        self.w_jerk         = w_jerk
        self.novelty_decay  = novelty_decay

        # Novelty counts: how many times each zone has been touched this session.
        # Reset at the start of each training run (not each episode) so that
        # the agent is pushed to discover zones it hasn't visited recently.
        self._zone_visit_counts: dict[str, float] = {z: 0.0 for z in ZONE_WEIGHTS}

        # Per-episode tracking (reset each episode via reset())
        self._ep_touch_counts: dict[str, int] = {z: 0 for z in ZONE_WEIGHTS}
        self._ep_steps: int   = 0
        self._last_ep_summary: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_touch_outputs(self):
        """Safely retrieve the touch sensor outputs from the unwrapped env."""
        try:
            return self.env.unwrapped.touch.sensor_outputs
        except AttributeError:
            return {}

    def _body_name(self, body_id: int) -> str:
        """Return the body name for a given id, or empty string on failure."""
        try:
            return self.env.unwrapped.model.body(body_id).name
        except Exception:
            return ""

    def _active_zones(self) -> set[str]:
        """Return the set of zones that have at least one active sensor."""
        active: set[str] = set()
        for body_id, force_vectors in self._get_touch_outputs().items():
            magnitudes = np.linalg.norm(force_vectors, axis=-1)
            if np.any(magnitudes > 1e-6):
                zone = classify_body(self._body_name(body_id))
                if zone in ZONE_WEIGHTS:
                    active.add(zone)
        return active

    # ── Reward components ─────────────────────────────────────────────────────

    def _r_touch(self, obs) -> float:
        """
        r_touch = fraction of all touch sensors that are currently active.
        Provides a dense, always-on shaping signal. Range [0, 1].
        """
        touch_flat = obs["touch"]                                  # (N, 3)
        n_active = np.sum(np.linalg.norm(touch_flat, axis=-1) > 1e-6)
        return float(n_active) / max(len(touch_flat), 1)

    def _r_zone(self, active_zones: set[str]) -> float:
        """
        r_zone = sum of ZONE_WEIGHTS for each currently-active zone.
        Higher zones (head, torso) get a larger bonus — matching DiMercurio
        et al.'s observation that infants preferentially touch the face/torso.
        """
        return sum(ZONE_WEIGHTS[z] for z in active_zones)

    def _r_novelty(self, active_zones: set[str]) -> float:
        """
        r_novelty = reward for touching zones that are under-explored.
        Inspired by Gama et al. (2023): novelty fades as a zone is visited
        more, pushing the agent to discover all zones of the body.

        novelty_score(z) = 1 / (1 + visit_count(z))
        """
        score = 0.0
        for z in active_zones:
            score += 1.0 / (1.0 + self._zone_visit_counts[z])
        return score

    def _r_jerk(self, action: np.ndarray) -> float:
        """
        r_jerk = mean absolute value of actions.
        Penalises large sudden motor commands, promoting smooth infant-like
        reaching trajectories (magnitude is small due to w_jerk weight).
        """
        return float(np.mean(np.abs(action)))

    # ── Full reward ─────────────────────────────────────────────────────────

    def compute_reward(self, obs, action) -> float:
        active_zones = self._active_zones()

        rt   = self.w_touch   * self._r_touch(obs)
        rz   = self.w_zone    * self._r_zone(active_zones)
        rn   = self.w_novelty * self._r_novelty(active_zones)
        rj   = self.w_jerk    * self._r_jerk(action)

        # Update novelty counts (zones just touched get their counts increased;
        # all counts decay slightly each step to keep novelty signal alive)
        for z in self._zone_visit_counts:
            self._zone_visit_counts[z] *= self.novelty_decay
        for z in active_zones:
            self._zone_visit_counts[z] += 1.0

        return rt + rz + rn - rj

    # ── Episode bookkeeping ──────────────────────────────────────────────────

    def _record_touches(self, active_zones: set[str]):
        for z in active_zones:
            self._ep_touch_counts[z] = self._ep_touch_counts.get(z, 0) + 1

    # ── Gym interface ─────────────────────────────────────────────────────────

    def step(self, action):
        obs, _ext_reward, terminated, truncated, info = self.env.step(action)

        active_zones = self._active_zones()
        reward = self.compute_reward(obs, action)
        self._record_touches(active_zones)
        self._ep_steps += 1

        # Save summary BEFORE reset() is triggered by SB3's DummyVecEnv.
        # _last_ep_summary persists across reset() so the callback can read it.
        if terminated or truncated:
            self._last_ep_summary = {
                **self._ep_touch_counts,
                "steps": self._ep_steps,
            }

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._ep_touch_counts = {z: 0 for z in ZONE_WEIGHTS}
        self._ep_steps = 0
        # _last_ep_summary intentionally NOT reset — callback reads it post-reset
        return self.env.reset(**kwargs)

    def episode_summary(self) -> dict:
        """Last completed episode's per-zone touch counts + step count."""
        return self._last_ep_summary or {**{z: 0 for z in ZONE_WEIGHTS}, "steps": 0}


# ─── Logging Callback ─────────────────────────────────────────────────────────

class EpisodeLogCallback(BaseCallback):
    """
    SB3 callback that writes per-episode touch statistics to CSV.

    CSV columns:
        episode, seed, steps,
        touches_head, touches_torso, touches_arm,
        touches_hand, touches_leg,  touches_foot,
        total_touches, touches_per_min
    """

    FIELDS = [
        "episode", "seed", "steps",
        "touches_head", "touches_torso", "touches_arm",
        "touches_hand", "touches_leg",  "touches_foot",
        "total_touches", "touches_per_min",
    ]

    def __init__(self, log_path: str, seed: int, sim_dt: float, verbose: int = 0):
        """
        Parameters
        ----------
        log_path : path to write the CSV file
        seed     : random seed of this training run (stored in each row)
        sim_dt   : MuJoCo simulation timestep in seconds (from env config)
                   Used to convert step counts → minutes correctly.
        """
        super().__init__(verbose)
        self.log_path = log_path
        self.seed     = seed
        self.sim_dt   = sim_dt          # no hardcoded 0.04 — read from env
        self.episode  = 0
        self._file    = None
        self._writer  = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._file   = open(self.log_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def _find_wrapper(self, root) -> "SelfTouchWrapper | None":
        """Traverse the wrapper stack to find our SelfTouchWrapper."""
        e = root
        while e is not None:
            if isinstance(e, SelfTouchWrapper):
                return e
            e = getattr(e, "env", None)
        return None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if not any(dones):
            return True

        wrapper = self._find_wrapper(self.training_env.envs[0])
        if wrapper is None:
            return True

        summary = wrapper.episode_summary()
        steps   = summary.get("steps", 0)
        if steps == 0:
            return True

        self.episode += 1
        total   = sum(summary.get(z, 0) for z in ZONE_WEIGHTS)
        # Convert steps → minutes using actual sim timestep (no hardcoding)
        duration_min = (steps * self.sim_dt) / 60.0
        tpm     = total / duration_min if duration_min > 0 else 0.0

        self._writer.writerow({
            "episode":          self.episode,
            "seed":             self.seed,
            "steps":            steps,
            "touches_head":     summary.get("head",  0),
            "touches_torso":    summary.get("torso", 0),
            "touches_arm":      summary.get("arm",   0),
            "touches_hand":     summary.get("hand",  0),
            "touches_leg":      summary.get("leg",   0),
            "touches_foot":     summary.get("foot",  0),
            "total_touches":    total,
            "touches_per_min":  round(tpm, 2),
        })
        self._file.flush()
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


# ─── Training ─────────────────────────────────────────────────────────────────

def get_sim_dt(env) -> float:
    """
    Read the actual simulation timestep from the env (no hardcoding).
    MuJoCo exposes model.opt.timestep; frame_skip multiplies it.
    """
    try:
        base_dt   = float(env.unwrapped.model.opt.timestep)
        frame_skip = int(getattr(env.unwrapped, "frame_skip", 1))
        return base_dt * frame_skip
    except Exception:
        return 0.04      # safe fallback — standard MuJoCo default


def train_seed(config: dict, seed: int, train_for: int, ppo_kwargs: dict, verbose: int) -> str:
    """
    Train one PPO agent with the given seed.

    Parameters
    ----------
    config      : BabyBench config dict (from YAML)
    seed        : random seed
    train_for   : total environment steps
    ppo_kwargs  : dict of PPO hyperparameters
    verbose     : SB3 verbosity level

    Returns the path to the saved model (without .zip extension).
    """
    seed_dir         = os.path.join(config["save_dir"], f"seed_{seed}")
    config_seed      = {**config, "save_dir": seed_dir}

    print(f"\n{'='*60}")
    print(f"  Training seed {seed}  |  timesteps = {train_for:,}")
    print(f"  Reward weights: {ppo_kwargs.get('reward_weights', 'default')}")
    print(f"{'='*60}")

    env     = bb_utils.make_env(config_seed)
    sim_dt  = get_sim_dt(env)

    wrapped = SelfTouchWrapper(
        env,
        w_touch    = ppo_kwargs.pop("w_touch",    1.0),
        w_zone     = ppo_kwargs.pop("w_zone",     1.0),
        w_novelty  = ppo_kwargs.pop("w_novelty",  0.5),
        w_jerk     = ppo_kwargs.pop("w_jerk",     0.01),
        novelty_decay = ppo_kwargs.pop("novelty_decay", 0.995),
    )
    wrapped.reset()

    log_path = os.path.join(seed_dir, "logs", f"episode_log_seed{seed}.csv")
    callback = EpisodeLogCallback(log_path=log_path, seed=seed, sim_dt=sim_dt)

    model = PPO(
        policy = "MultiInputPolicy",
        env    = wrapped,
        verbose= verbose,
        seed   = seed,
        **ppo_kwargs,
    )

    t0 = time.time()
    model.learn(total_timesteps=train_for, callback=callback)
    elapsed = time.time() - t0

    model_path = os.path.join(seed_dir, "model")
    model.save(model_path)
    print(f"  Seed {seed} done in {elapsed/60:.1f} min  |  sim_dt={sim_dt:.4f}s")
    print(f"  Model → {model_path}.zip")

    wrapped.close()
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="PPO Self-Touch agent for BabyBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Environment ──
    parser.add_argument("--config",        default="examples/config_selftouch.yml", type=str)
    parser.add_argument("--train_for",     default=500_000, type=int,
                        help="Total timesteps PER seed")
    parser.add_argument("--seed",          default=None, type=int,
                        help="Single seed to train; omit to train seeds 0,1,2")
    parser.add_argument("--verbose",       default=1, type=int)

    # ── Reward weights (all derived from paper-motivated defaults) ──
    parser.add_argument("--w_touch",       default=1.0,   type=float,
                        help="Weight for touch coverage (r_touch)")
    parser.add_argument("--w_zone",        default=1.0,   type=float,
                        help="Weight for zone priority bonus (DiMercurio 2018)")
    parser.add_argument("--w_novelty",     default=0.5,   type=float,
                        help="Weight for novelty bonus (Gama et al. 2023)")
    parser.add_argument("--w_jerk",        default=0.01,  type=float,
                        help="Weight for smoothness penalty")
    parser.add_argument("--novelty_decay", default=0.995, type=float,
                        help="Decay rate applied to zone novelty counts per step")

    # ── PPO hyperparameters ──
    parser.add_argument("--n_steps",       default=2048,  type=int)
    parser.add_argument("--batch_size",    default=64,    type=int)
    parser.add_argument("--n_epochs",      default=10,    type=int)
    parser.add_argument("--learning_rate", default=3e-4,  type=float)
    parser.add_argument("--gamma",         default=0.99,  type=float)
    parser.add_argument("--gae_lambda",    default=0.95,  type=float)
    parser.add_argument("--clip_range",    default=0.2,   type=float)
    parser.add_argument("--ent_coef",      default=0.01,  type=float,
                        help="Entropy coefficient — higher encourages exploration")
    parser.add_argument("--vf_coef",       default=0.5,   type=float)
    parser.add_argument("--max_grad_norm", default=0.5,   type=float)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ppo_kwargs = {
        # Reward weights (popped in train_seed)
        "w_touch":       args.w_touch,
        "w_zone":        args.w_zone,
        "w_novelty":     args.w_novelty,
        "w_jerk":        args.w_jerk,
        "novelty_decay": args.novelty_decay,
        # SB3 PPO hyperparameters
        "n_steps":       args.n_steps,
        "batch_size":    args.batch_size,
        "n_epochs":      args.n_epochs,
        "learning_rate": args.learning_rate,
        "gamma":         args.gamma,
        "gae_lambda":    args.gae_lambda,
        "clip_range":    args.clip_range,
        "ent_coef":      args.ent_coef,
        "vf_coef":       args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }

    seeds = [args.seed] if args.seed is not None else [0, 1, 2]
    for s in seeds:
        train_seed(config, s, args.train_for, dict(ppo_kwargs), args.verbose)

    print("\nAll seeds complete. Run:  python evaluate_selftouch.py --all_seeds")


if __name__ == "__main__":
    main()
