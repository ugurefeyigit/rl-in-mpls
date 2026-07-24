"""Train the MaskablePPO traffic-engineering agent.

Usage:
    python scripts/train.py                       # full run per configs/training.yaml
    python scripts/train.py --timesteps 30000     # short sanity run
    python scripts/train.py --tag exp1 --seed 7   # tagged run, custom seed

Outputs:
    models/<tag>/checkpoint_*.zip     periodic checkpoints
    models/<tag>/best_model.zip       best mean eval reward
    models/<tag>/final_model.zip      model at the end of training
    runs/<tag>/                       TensorBoard logs (incl. reward components)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mplssim.factory import get_training_config
from mplssim.rl.env import MplsTeEnv

ROOT = Path(__file__).resolve().parents[1]


class RewardComponentLogger(BaseCallback):
    """Log mean per-step reward components and action stats to TensorBoard."""

    def __init__(self, log_every: int = 2048) -> None:
        super().__init__()
        self.log_every = log_every
        self.acc: dict[str, list[float]] = defaultdict(list)
        self.noop_count = 0
        self.reroute_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            comps = info.get("reward_components")
            if comps:
                for k, v in comps.items():
                    self.acc[k].append(v)
            dec = info.get("decoded_action")
            if dec:
                if dec["type"] == "noop":
                    self.noop_count += 1
                else:
                    self.reroute_count += 1
        if self.num_timesteps % self.log_every < self.training_env.num_envs:
            for k, vals in self.acc.items():
                if vals:
                    self.logger.record(f"reward_components/{k}", float(np.mean(vals)))
            total = max(1, self.noop_count + self.reroute_count)
            self.logger.record("actions/noop_fraction", self.noop_count / total)
            self.acc.clear()
            self.noop_count = self.reroute_count = 0
        return True


def make_env(scenario: str, base_seed: int, rank: int,
             reward_overrides: dict[str, float] | None = None):
    def _init():
        env = MplsTeEnv(scenario=scenario, base_seed=base_seed + rank * 10_000,
                        reward_overrides=reward_overrides)
        return Monitor(env)
    return _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default="ppo_te")
    ap.add_argument("--scenario", type=str, default=None)
    ap.add_argument("--zero-weight", nargs="*", default=[],
                    help="reward weights forced to 0 during training (ablations), "
                         "e.g. --zero-weight reroute flap")
    args = ap.parse_args()
    overrides = {name: 0.0 for name in args.zero_weight}

    cfg = get_training_config()
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    timesteps = args.timesteps if args.timesteps is not None else int(cfg["total_timesteps"])
    scenario = args.scenario or cfg["scenario"]
    n_envs = int(cfg["n_envs"])

    model_dir = ROOT / "models" / args.tag
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(scenario, seed, i, overrides) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(cfg["eval_scenarios"][0], 900_000 + seed, 0, overrides)])

    ppo = cfg["ppo"]
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        policy_kwargs=dict(ppo["policy_kwargs"]),
        seed=seed,
        device=cfg.get("device", "cpu"),
        tensorboard_log=str(ROOT / "runs"),
        verbose=1,
    )

    callbacks = [
        RewardComponentLogger(),
        CheckpointCallback(
            save_freq=max(1, int(cfg["checkpoint_freq"]) // n_envs),
            save_path=str(model_dir), name_prefix="checkpoint",
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(model_dir),
            eval_freq=max(1, int(cfg["eval_freq"]) // n_envs),
            n_eval_episodes=3,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name=args.tag,
                progress_bar=False)
    model.save(model_dir / "final_model")
    print(f"saved {model_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
