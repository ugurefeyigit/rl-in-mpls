"""Learner adapters and shared training orchestration for V2 experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecEnv

from mplssim.experiments.learning_common import (
    AuditedV2Env,
    MetricsWriter,
    SeedLedger,
    create_run_directory,
    hardware_inventory,
    make_audited_env,
    require_exact_vector_budget,
    resolve_device,
    validate_checkpoint_sidecar,
    validate_training_root,
    verified_environment_record,
    write_checkpoint_sidecar,
)
from mplssim.experiments.masked_bandit import MaskedContextualBandit
from mplssim.rl.reward_v2 import components_sum


ALGORITHMS: tuple[str, ...] = ("maskable_ppo", "masked_bandit")


def require_algorithm(name: str) -> str:
    if name not in ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {', '.join(ALGORITHMS)}, got {name!r}")
    return name


class MaskablePpoLearner:
    """Thin adapter that requires masks on every inference call."""

    def __init__(
        self,
        env: VecEnv,
        device: torch.device,
        seed: int,
        ppo_config: Mapping[str, Any],
        tensorboard_log: str | None,
    ) -> None:
        cfg = ppo_config
        self.model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            ent_coef=float(cfg["ent_coef"]),
            vf_coef=float(cfg["vf_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            policy_kwargs=dict(cfg["policy_kwargs"]),
            seed=int(seed),
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=0,
        )

    @classmethod
    def from_model(cls, model: MaskablePPO) -> "MaskablePpoLearner":
        learner = cls.__new__(cls)
        learner.model = model
        return learner

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def transitions(self) -> int:
        return int(self.model.num_timesteps)

    def predict(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        actions, _ = self.model.predict(
            observations,
            deterministic=bool(deterministic),
            action_masks=np.asarray(masks, dtype=bool),
        )
        return np.asarray(actions, dtype=np.int64).reshape(-1)

    def learn(self, total_timesteps: int, callback=None) -> None:
        self.model.learn(
            total_timesteps=int(total_timesteps),
            callback=callback,
            progress_bar=False,
        )

    def save(self, path: Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
        env: VecEnv | None = None,
    ) -> "MaskablePpoLearner":
        model = MaskablePPO.load(str(path), device=device, env=env)
        return cls.from_model(model)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


class TrainingMetricRecorder:
    """Flatten vector batches into auditable step and episode streams."""

    def __init__(self, writer: MetricsWriter, algorithm: str, n_envs: int) -> None:
        self.writer = writer
        self.algorithm = algorithm
        self.n_envs = int(n_envs)
        self.episodes = [
            {
                "return": 0.0,
                "length": 0,
                "noops": 0,
                "reward_components": {},
            }
            for _ in range(n_envs)
        ]

    def record_batch(
        self,
        aggregate_end: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
        masks: np.ndarray,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        first = int(aggregate_end) - self.n_envs + 1
        for worker in range(self.n_envs):
            info = infos[worker]
            decoded = info["decoded_action"]
            exact_reward = components_sum(info["reward_components"])
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(dones[worker]) and not truncated
            if terminated:
                raise RuntimeError(
                    "V2 training episode terminated instead of truncating")
            row: dict[str, Any] = {
                "aggregate_transition": first + worker,
                "worker_rank": worker,
                "algorithm": self.algorithm,
                "episode_seed": int(info["episode_seed"]),
                "action": int(actions[worker]),
                "action_type": decoded["type"],
                "action_accepted": bool(decoded.get("accepted", False)),
                "valid_action_count": int(np.sum(masks[worker])),
                "reward": exact_reward,
                "learner_reward": float(rewards[worker]),
                "reward_component_sum_exact": True,
                "terminated": terminated,
                "truncated": truncated,
            }
            row.update(_jsonable(info["metrics"]))
            row.update({
                f"rc_{key}": float(value)
                for key, value in info["reward_components"].items()
            })
            if diagnostics:
                row.update({
                    f"algorithm_{key}": _jsonable(value)
                    for key, value in diagnostics.items()
                })
            self.writer.write_step(row)

            episode = self.episodes[worker]
            episode["return"] += exact_reward
            episode["length"] += 1
            episode["noops"] += int(int(actions[worker]) == 0)
            for key, value in info["reward_components"].items():
                episode["reward_components"][key] = (
                    episode["reward_components"].get(key, 0.0) + float(value))
            if dones[worker]:
                summary = {
                    "algorithm": self.algorithm,
                    "worker_rank": worker,
                    "episode_seed": int(info["episode_seed"]),
                    "return": episode["return"],
                    "length": episode["length"],
                    "noops": episode["noops"],
                    "noop_fraction": episode["noops"] / episode["length"],
                    "reward_components": episode["reward_components"],
                    "episode_totals": _jsonable(info["episode_totals"]),
                    "terminated": terminated,
                    "truncated": truncated,
                }
                self.writer.write_episode(summary)
                self.episodes[worker] = {
                    "return": 0.0,
                    "length": 0,
                    "noops": 0,
                    "reward_components": {},
                }


def build_training_vec(
    scenario: str,
    root_seed: int,
    n_envs: int,
    seed_ledger: SeedLedger,
) -> tuple[DummyVecEnv, list[AuditedV2Env]]:
    """Build CPU-based DummyVecEnv workers through the sole V2 factory."""
    audited: list[AuditedV2Env] = []

    def worker(rank: int):
        def construct():
            env = make_audited_env(
                scenario=scenario,
                root_seed=root_seed,
                worker_rank=rank,
                seed_ledger=seed_ledger,
            )
            audited.append(env)
            return Monitor(env)
        return construct

    vec = DummyVecEnv([worker(rank) for rank in range(n_envs)])
    return vec, audited


def _integrity_totals(envs: list[AuditedV2Env]) -> dict[str, int]:
    keys = tuple(envs[0].integrity.as_dict()) if envs else ()
    return {
        key: sum(env.integrity.as_dict()[key] for env in envs)
        for key in keys
    }


def _save_checkpoint(
    learner: MaskablePpoLearner | MaskedContextualBandit,
    run_directory: Path,
    algorithm: str,
    aggregate_transitions: int,
    environment_record: dict[str, Any],
    run_config: dict[str, Any],
    device: dict[str, Any],
) -> Path:
    suffix = ".zip" if algorithm == "maskable_ppo" else ".pt"
    payload = (
        run_directory / "checkpoints"
        / f"checkpoint_{aggregate_transitions:09d}{suffix}"
    )
    payload.parent.mkdir(parents=True, exist_ok=True)
    learner.save(payload)
    sidecar = write_checkpoint_sidecar(
        payload,
        algorithm=algorithm,
        aggregate_transitions=aggregate_transitions,
        environment_record=environment_record,
        run_config=run_config,
        device=device,
    )
    validate_checkpoint_sidecar(
        payload, sidecar, expected_algorithm=algorithm,
        require_clean_source=bool(
            run_config.get("require_clean_checkout", True)))
    return payload


def should_continue_ppo_rollout(
    *,
    num_timesteps: int,
    target_transitions: int,
    rollout_transitions: int,
) -> bool:
    """Let SB3 train a complete final rollout; stop an exact partial budget."""
    if int(num_timesteps) < int(target_transitions):
        return True
    return (
        int(num_timesteps) == int(target_transitions)
        and int(num_timesteps) % int(rollout_transitions) == 0
    )


class PpoExperimentCallback(BaseCallback):
    """Exact-budget, masked rollout audit and periodic checkpoint callback."""

    def __init__(
        self,
        *,
        learner: MaskablePpoLearner,
        recorder: TrainingMetricRecorder,
        target_transitions: int,
        checkpoint_interval: int,
        run_directory: Path,
        environment_record: dict[str, Any],
        run_config: dict[str, Any],
        device: dict[str, Any],
    ) -> None:
        super().__init__(verbose=0)
        self.learner = learner
        self.recorder = recorder
        self.target_transitions = int(target_transitions)
        self.checkpoint_interval = int(checkpoint_interval)
        self.run_directory = run_directory
        self.environment_record = environment_record
        self.run_config = run_config
        self.device_record = device
        self.saved: list[str] = []

    def _on_step(self) -> bool:
        masks = np.asarray(self.locals["action_masks"], dtype=bool)
        actions = np.asarray(self.locals["actions"], dtype=np.int64).reshape(-1)
        if not np.all(masks[np.arange(len(actions)), actions]):
            raise RuntimeError("MaskablePPO selected an invalid rollout action")
        diagnostics = {
            key.replace("train/", ""): value
            for key, value in self.model.logger.name_to_value.items()
            if key.startswith("train/")
        }
        self.recorder.record_batch(
            self.num_timesteps,
            actions,
            np.asarray(self.locals["rewards"]),
            np.asarray(self.locals["dones"]),
            self.locals["infos"],
            masks,
            diagnostics,
        )
        if self.num_timesteps % self.checkpoint_interval == 0:
            path = _save_checkpoint(
                self.learner, self.run_directory, "maskable_ppo",
                self.num_timesteps, self.environment_record, self.run_config,
                self.device_record)
            self.saved.append(str(path))
        return should_continue_ppo_rollout(
            num_timesteps=self.num_timesteps,
            target_transitions=self.target_transitions,
            rollout_transitions=int(self.model.n_steps * self.model.n_envs),
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=1, allow_nan=False),
        encoding="utf-8",
    )


def train_experiment(
    *,
    algorithm: str,
    run_directory: Path,
    root_seed: int,
    scenario: str,
    aggregate_transitions: int,
    n_envs: int,
    checkpoint_interval: int,
    requested_device: str,
    learning_config: dict[str, Any],
    purpose: str = "meaningful",
    require_clean_checkout: bool = True,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Execute one new, fail-closed V2 learner run."""
    algorithm = require_algorithm(algorithm)
    validate_training_root(root_seed)
    from mplssim.experiments.learning_common import validate_run_purpose
    validate_run_purpose(
        purpose, aggregate_transitions, checkpoint_interval)
    require_exact_vector_budget(aggregate_transitions, n_envs)
    require_exact_vector_budget(checkpoint_interval, n_envs)
    if aggregate_transitions % checkpoint_interval:
        raise ValueError("aggregate budget must be divisible by checkpoint interval")
    device_selection = resolve_device(requested_device)
    environment_record = verified_environment_record(
        root_seed, require_clean=require_clean_checkout)
    run_dir = create_run_directory(Path(run_directory))
    seed_ledger = SeedLedger()
    vec, audited_envs = build_training_vec(
        scenario, root_seed, n_envs, seed_ledger)
    run_config = {
        "learning_version": learning_config["version"],
        "algorithm": algorithm,
        "purpose": purpose,
        "environment_version": "v2",
        "root_seed": int(root_seed),
        "scenario": scenario,
        "aggregate_transitions": int(aggregate_transitions),
        "n_envs": int(n_envs),
        "checkpoint_interval": int(checkpoint_interval),
        "requested_device": requested_device,
        "resolved_device": device_selection.resolved,
        "require_clean_checkout": bool(require_clean_checkout),
        "command": list(command) if command is not None else None,
        "ppo": None,
        "masked_bandit": (
            learning_config["masked_bandit"]
            if algorithm == "masked_bandit" else None),
    }
    if algorithm == "maskable_ppo":
        from mplssim.factory import get_training_config
        run_config["ppo"] = get_training_config()["ppo"]
    _write_json(run_dir / "run_config.json", run_config)
    manifest: dict[str, Any] = {
        "format": "v2-learning-run-v1",
        "status": "running",
        "algorithm": algorithm,
        "environment_record": environment_record,
        "run_config": run_config,
        "device": device_selection.as_dict(),
        "hardware": hardware_inventory(),
    }
    _write_json(run_dir / "manifest.json", manifest)

    start = time.perf_counter()
    saved: list[str] = []
    diagnostics: dict[str, Any] = {}
    writer = MetricsWriter(run_dir)
    recorder = TrainingMetricRecorder(writer, algorithm, n_envs)
    try:
        if device_selection.resolved == "cuda":
            torch.cuda.reset_peak_memory_stats(device_selection.torch_device)
        if algorithm == "masked_bandit":
            learner = MaskedContextualBandit(
                observation_dim=int(vec.observation_space.shape[0]),
                action_dim=int(vec.action_space.n),
                device=device_selection.torch_device,
                seed=root_seed,
                config=learning_config["masked_bandit"],
            )
            observations = vec.reset()
            vector_steps = aggregate_transitions // n_envs
            update_every = int(
                learning_config["masked_bandit"]["update_every_vector_steps"])
            for vector_step in range(1, vector_steps + 1):
                masks = get_action_masks(vec)
                actions = learner.predict(
                    observations, masks, deterministic=False)
                new_observations, rewards, dones, infos = vec.step(actions)
                exact_rewards = np.asarray([
                    components_sum(info["reward_components"]) for info in infos
                ], dtype=np.float32)
                learner.observe(observations, actions, masks, exact_rewards)
                update_result = (
                    learner.update() if vector_step % update_every == 0 else None)
                if update_result is not None:
                    diagnostics = update_result
                recorder.record_batch(
                    learner.transitions, actions, rewards, dones, infos, masks,
                    diagnostics if diagnostics else None)
                observations = new_observations
                if learner.transitions % checkpoint_interval == 0:
                    path = _save_checkpoint(
                        learner, run_dir, algorithm, learner.transitions,
                        environment_record, run_config,
                        device_selection.as_dict())
                    saved.append(str(path))
            completed_transitions = learner.transitions
            diagnostics = {
                **diagnostics,
                "updates": learner.updates,
                "final_epsilon": learner.epsilon(),
                "replay_size": len(learner.replay),
            }
        else:
            learner = MaskablePpoLearner(
                env=vec,
                device=device_selection.torch_device,
                seed=root_seed,
                ppo_config=run_config["ppo"],
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            callback = PpoExperimentCallback(
                learner=learner,
                recorder=recorder,
                target_transitions=aggregate_transitions,
                checkpoint_interval=checkpoint_interval,
                run_directory=run_dir,
                environment_record=environment_record,
                run_config=run_config,
                device=device_selection.as_dict(),
            )
            learner.learn(aggregate_transitions, callback=callback)
            completed_transitions = learner.transitions
            saved = callback.saved
            diagnostics = {
                key: _jsonable(value)
                for key, value in learner.model.logger.name_to_value.items()
                if key.startswith("train/")
            }
        if completed_transitions != aggregate_transitions:
            raise RuntimeError(
                f"run stopped at {completed_transitions}, expected "
                f"{aggregate_transitions} aggregate transitions")
        if any(
            int(record["root_seed"]) != int(root_seed)
            for record in seed_ledger.records
        ):
            raise RuntimeError(
                "recorded episode seed used a root other than the governed "
                f"training root {root_seed}")
        integrity = _integrity_totals(audited_envs)
        if any(integrity[key] for key in (
            "invalid_action_attempts", "mask_disagreements",
            "reward_mismatches", "nonfinite_values", "solver_failures",
            "protected_safety_failures",
        )):
            raise RuntimeError(f"run integrity counters are nonzero: {integrity}")
        wall_time = time.perf_counter() - start
        manifest.update({
            "status": "completed",
            "aggregate_transitions": completed_transitions,
            "wall_time_seconds": wall_time,
            "aggregate_transitions_per_second": completed_transitions / wall_time,
            "integrity": integrity,
            "algorithm_diagnostics": diagnostics,
            "checkpoints": saved,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device_selection.torch_device))
                if device_selection.resolved == "cuda" else None),
        })
        return manifest
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "wall_time_seconds": time.perf_counter() - start,
            "integrity": _integrity_totals(audited_envs),
            "checkpoints": saved,
        })
        raise
    finally:
        writer.close()
        vec.close()
        _write_json(run_dir / "episode_seeds.json", seed_ledger.records)
        _write_json(run_dir / "manifest.json", manifest)
