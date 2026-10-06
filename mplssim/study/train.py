"""Training loop for every learner of the post-V2 study.

The masked-bandit branch reproduces the closed study's training loop
(:func:`mplssim.experiments.trainers_v2.train_experiment`) step for step:
same vector size, same order of predict / step / observe / update, same
update cadence, same exact-component reward. The differences are deliberate
and limited to what the new experiments need:

* any hyper-parameter may be overridden from an experiment registry entry;
* environment variants (:mod:`mplssim.study.variants`) can be trained;
* the masked Q-learner (``gamma > 0``) and PPO with arbitrary settings;
* per-step JSONL telemetry is not written (per-episode records are), which
  keeps a 400k-transition run at a few MB instead of ~90 MB.

Integrity checks (legal actions under the authoritative mask, exact reward
component sums, finite values, solver convergence, protected-class safety,
governed seed roots) are enforced by :class:`AuditedV2Env` exactly as in the
closed study: any violation aborts the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mplssim.experiments.learning_common import AuditedV2Env, SeedLedger
from mplssim.experiments.masked_bandit import MaskedContextualBandit
from mplssim.rl.reward_v2 import components_sum
from mplssim.study.learners import MaskedQLearner
from mplssim.study.provenance import run_record

ALGORITHMS = ("masked_bandit", "masked_q", "maskable_ppo")


def build_vec(env_factory: Callable[..., Any], scenario: str, root: int, n_envs: int,
              ledger: SeedLedger) -> tuple[DummyVecEnv, list[AuditedV2Env]]:
    audited: list[AuditedV2Env] = []

    def make(rank: int):
        def _f():
            env = AuditedV2Env(env_factory(scenario=scenario, root_seed=root,
                                           worker_rank=rank), seed_ledger=ledger)
            audited.append(env)
            return Monitor(env)
        return _f

    return DummyVecEnv([make(r) for r in range(n_envs)]), audited


class EpisodeLog:
    """Per-episode return / no-op / move statistics, flushed line by line."""

    def __init__(self, path: Path, n_envs: int) -> None:
        self.fh = path.open("w", encoding="utf-8")
        self.acc = [self._blank() for _ in range(n_envs)]
        self.count = 0

    @staticmethod
    def _blank() -> dict[str, float]:
        return {"return": 0.0, "length": 0, "noops": 0, "accepted": 0}

    def record(self, transitions: int, actions, rewards, dones, infos) -> None:
        for w, info in enumerate(infos):
            a = self.acc[w]
            a["return"] += float(rewards[w])
            a["length"] += 1
            a["noops"] += int(int(actions[w]) == 0)
            a["accepted"] += int(info["decoded_action"].get("accepted", False))
            if dones[w]:
                self.fh.write(json.dumps({"transitions": int(transitions), "worker": w,
                                          "episode_seed": int(info["episode_seed"]),
                                          **a}) + "\n")
                self.fh.flush()
                self.count += 1
                self.acc[w] = self._blank()

    def close(self) -> None:
        self.fh.close()


def _check_integrity(envs: list[AuditedV2Env]) -> dict[str, int]:
    keys = envs[0].integrity.as_dict().keys()
    tot = {k: sum(e.integrity.as_dict()[k] for e in envs) for k in keys}
    bad = {k: v for k, v in tot.items() if k != "aggregate_transitions" and v}
    if bad:
        raise RuntimeError(f"integrity counters nonzero: {bad}")
    return tot


class _PpoCallback(BaseCallback):
    def __init__(self, log: EpisodeLog, ckpt_dir: Path, every: int, target: int) -> None:
        super().__init__(verbose=0)
        self.log, self.ckpt_dir, self.every, self.target = log, ckpt_dir, every, target
        self.saved: list[str] = []

    def _on_step(self) -> bool:
        masks = np.asarray(self.locals["action_masks"], dtype=bool)
        acts = np.asarray(self.locals["actions"]).reshape(-1)
        if not np.all(masks[np.arange(len(acts)), acts]):
            raise RuntimeError("MaskablePPO sampled an illegal action")
        self.log.record(self.num_timesteps, acts, self.locals["rewards"],
                        self.locals["dones"], self.locals["infos"])
        if self.num_timesteps % self.every == 0:
            p = self.ckpt_dir / f"checkpoint_{self.num_timesteps:09d}.zip"
            self.model.save(str(p))
            self.saved.append(p.name)
        # Train a complete final rollout when the budget is an exact multiple.
        if self.num_timesteps < self.target:
            return True
        return self.num_timesteps % (self.model.n_steps * self.model.n_envs) == 0 \
            and self.num_timesteps == self.target


def train_run(*, run_dir: Path, algorithm: str, root: int, config: dict[str, Any],
              env_factory: Callable[..., Any], env_spec: dict[str, Any],
              transitions: int, checkpoint_interval: int, n_envs: int = 16,
              scenario: str = "random_day", run_id: str = "") -> dict[str, Any]:
    """Train one learner; write checkpoints, episode log and manifest."""
    if algorithm not in ALGORITHMS:
        raise ValueError(algorithm)
    if transitions % n_envs or checkpoint_interval % n_envs or transitions % checkpoint_interval:
        raise ValueError("budgets must be exact multiples of n_envs and the checkpoint interval")
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir()
    torch.set_num_threads(1)
    ledger = SeedLedger()
    vec, audited = build_vec(env_factory, scenario, root, n_envs, ledger)
    manifest = run_record(kind="training", run_id=run_id, algorithm=algorithm,
                          root_seed=root, config=config, env=env_spec,
                          transitions=transitions, checkpoint_interval=checkpoint_interval,
                          n_envs=n_envs, scenario=scenario, status="running",
                          observation_dim=int(vec.observation_space.shape[0]),
                          action_dim=int(vec.action_space.n))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log = EpisodeLog(run_dir / "episodes.jsonl", n_envs)
    t0 = time.perf_counter()
    saved: list[str] = []
    diag: dict[str, Any] = {}
    try:
        if algorithm == "maskable_ppo":
            model = MaskablePPO(
                "MlpPolicy", vec, learning_rate=float(config["learning_rate"]),
                n_steps=int(config["n_steps"]), batch_size=int(config["batch_size"]),
                n_epochs=int(config["n_epochs"]), gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]), clip_range=float(config["clip_range"]),
                ent_coef=float(config["ent_coef"]), vf_coef=float(config["vf_coef"]),
                max_grad_norm=float(config["max_grad_norm"]),
                policy_kwargs=dict(config["policy_kwargs"]), seed=int(root),
                device="cpu", verbose=0)
            cb = _PpoCallback(log, ckpt, checkpoint_interval, transitions)
            model.learn(total_timesteps=transitions, callback=cb, progress_bar=False)
            done_transitions = int(model.num_timesteps)
            saved = cb.saved
            diag = {k: float(v) for k, v in model.logger.name_to_value.items()
                    if k.startswith("train/")}
            diag["parameters"] = int(sum(p.numel() for p in model.policy.parameters()))
        else:
            cls = MaskedContextualBandit if algorithm == "masked_bandit" else MaskedQLearner
            learner = cls(int(vec.observation_space.shape[0]), int(vec.action_space.n),
                          torch.device("cpu"), int(root), config)
            use_next = algorithm == "masked_q"
            obs = vec.reset()
            every = int(config["update_every_vector_steps"])
            for vstep in range(1, transitions // n_envs + 1):
                masks = get_action_masks(vec)
                acts = learner.predict(obs, masks, deterministic=False)
                new_obs, rewards, dones, infos = vec.step(acts)
                exact = np.asarray([components_sum(i["reward_components"]) for i in infos],
                                   dtype=np.float32)
                if use_next:
                    next_masks = get_action_masks(vec)
                    succ = new_obs.copy()
                    for w in np.flatnonzero(dones):
                        succ[w] = infos[w]["terminal_observation"]
                    learner.observe_transitions(obs, acts, masks, exact, succ,
                                                next_masks, dones.astype(bool))
                else:
                    learner.observe(obs, acts, masks, exact)
                if vstep % every == 0:
                    learner.update()
                log.record(learner.transitions, acts, exact, dones, infos)
                obs = new_obs
                if learner.transitions % checkpoint_interval == 0:
                    p = ckpt / f"checkpoint_{learner.transitions:09d}.pt"
                    if algorithm == "masked_q":
                        learner.save_policy(p)
                    else:
                        MaskedQLearner.save_policy(learner, p)  # same inference payload
                    saved.append(p.name)
            done_transitions = learner.transitions
            diag = {"updates": learner.updates, "final_epsilon": learner.epsilon(),
                    "last_loss": learner.last_loss,
                    "parameters": int(sum(p.numel() for p in learner.network.parameters()))}
        if done_transitions != transitions:
            raise RuntimeError(f"stopped at {done_transitions} != {transitions}")
        roots = {int(r["root_seed"]) for r in ledger.records}
        if roots != {int(root)}:
            raise RuntimeError(f"episode seeds used roots {roots}")
        integ = _check_integrity(audited)
        wall = time.perf_counter() - t0
        manifest.update(status="completed", wall_time_seconds=wall,
                        transitions_per_second=transitions / wall, integrity=integ,
                        diagnostics=diag, checkpoints=saved, episodes=log.count,
                        unique_episode_seeds=len({r["episode_seed"] for r in ledger.records}))
        return manifest
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                        wall_time_seconds=time.perf_counter() - t0)
        raise
    finally:
        log.close()
        vec.close()
        (run_dir / "episode_seeds.json").write_text(json.dumps(ledger.records))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))


def load_policy(path: Path, algorithm: str):
    """Inference policy for a study checkpoint."""
    if algorithm == "maskable_ppo":
        from mplssim.experiments.trainers_v2 import MaskablePpoLearner
        return MaskablePpoLearner.load(path, device=torch.device("cpu"))
    return MaskedQLearner.load_policy(path)
