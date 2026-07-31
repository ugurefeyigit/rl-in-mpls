"""Shared, V2-only experiment contracts for governed learner comparisons."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "learning_v2.yaml"
MEANINGFUL_TRANSITIONS = 400_000
MEANINGFUL_CHECKPOINT_INTERVAL = 50_000
RUN_PURPOSES = ("meaningful", "smoke", "benchmark")
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.(?:zip|pt)")


def load_learning_config(path: Path = LEARNING_CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the explicit V2 learning contract."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("version") != "learning-v2.0":
        raise ValueError(f"{path}: unsupported learning configuration version")
    if cfg.get("environment_version") != "v2":
        raise ValueError(f"{path}: learning comparison must construct V2")
    if tuple(cfg.get("algorithms", ())) != ("maskable_ppo", "masked_bandit"):
        raise ValueError(f"{path}: algorithm registry must contain only governed learners")
    return cfg


def validate_training_root(root_seed: int) -> None:
    """Enforce the active subset of preregistered training roots."""
    cfg = load_learning_config()
    registered = {int(value) for value in cfg["training_roots"]}
    active = {int(value) for value in cfg["active_training_roots"]}
    permitted = registered & active
    if int(root_seed) not in permitted:
        raise ValueError(
            "this task permits only active preregistered training roots "
            f"{sorted(permitted)}, got {root_seed}")


def validate_run_purpose(
    purpose: str,
    aggregate_transitions: int,
    checkpoint_interval: int,
) -> None:
    """Keep disposable runs distinct from the preregistered comparison."""
    if purpose not in RUN_PURPOSES:
        raise ValueError(f"run purpose must be one of {RUN_PURPOSES}, got {purpose!r}")
    if purpose == "meaningful" and (
        int(aggregate_transitions) != MEANINGFUL_TRANSITIONS
        or int(checkpoint_interval) != MEANINGFUL_CHECKPOINT_INTERVAL
    ):
        raise ValueError(
            "meaningful runs require exactly 400000 aggregate transitions "
            "and a 50000-transition checkpoint interval")


def validate_evaluation_seeds(seeds: list[int] | tuple[int, ...]) -> None:
    """Permit continuity seeds and reject holdout access before construction."""
    cfg = load_learning_config()
    holdout = set(int(x) for x in cfg["holdout_seeds"])
    requested = [int(x) for x in seeds]
    forbidden = sorted(set(requested) & holdout)
    if forbidden:
        raise ValueError(f"forbidden holdout seed(s): {forbidden}")
    continuity = set(int(x) for x in cfg["continuity_seeds"])
    outside = sorted(set(requested) - continuity)
    if outside:
        raise ValueError(f"evaluation seeds must be continuity seeds: {outside}")
    if len(requested) != len(set(requested)):
        raise ValueError("continuity evaluation seeds must not contain duplicates")


def create_run_directory(path: Path) -> Path:
    """Create a never-before-used run directory and return its absolute path."""
    target = path.resolve()
    if target.exists():
        raise FileExistsError(f"run directory already exists: {target}")
    target.mkdir(parents=True)
    return target


def vector_step_transition_count(vector_steps: int, n_envs: int) -> int:
    """Convert vector-environment calls to aggregate environment transitions."""
    if vector_steps < 0 or n_envs <= 0:
        raise ValueError("vector_steps must be nonnegative and n_envs positive")
    return int(vector_steps) * int(n_envs)


def require_exact_vector_budget(aggregate_transitions: int, n_envs: int) -> None:
    """Reject vector counts that cannot stop on the exact aggregate budget."""
    if aggregate_transitions <= 0 or n_envs <= 0:
        raise ValueError("aggregate transitions and n_envs must be positive")
    if aggregate_transitions % n_envs:
        raise ValueError(
            f"aggregate budget {aggregate_transitions} is not divisible by "
            f"{n_envs} environments")


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    resolved: str
    torch_device: torch.device
    cuda_available: bool
    cuda_device_name: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
        }


def resolve_device(requested: str) -> DeviceSelection:
    """Resolve ``auto|cuda|cpu`` without claiming unavailable acceleration."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but this PyTorch build cannot use CUDA")
    resolved = "cuda" if available and requested in {"auto", "cuda"} else "cpu"
    name = torch.cuda.get_device_name(0) if resolved == "cuda" else None
    return DeviceSelection(
        requested=requested,
        resolved=resolved,
        torch_device=torch.device(resolved),
        cuda_available=available,
        cuda_device_name=name,
    )


class SeedLedger:
    """Append-only record of every environment reset and derived seed."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._owners: dict[int, tuple[int, int, int]] = {}

    def record(self, reset_info: dict[str, Any]) -> None:
        row = dict(reset_info)
        row.setdefault("episode_index", len(self.records))
        identity = (
            int(row["root_seed"]),
            int(row["worker_rank"]),
            int(row["episode_index"]),
        )
        episode_seed = int(row["episode_seed"])
        owner = self._owners.get(episode_seed)
        if owner is not None and owner != identity:
            raise RuntimeError(
                f"derived episode seed collision: {episode_seed} belongs to "
                f"{owner} and {identity}")
        self._owners[episode_seed] = identity
        self.records.append({
            "root_seed": identity[0],
            "worker_rank": identity[1],
            "episode_index": identity[2],
            "episode_seed": episode_seed,
            "scenario": row["scenario"],
        })


@dataclass
class IntegrityCounters:
    aggregate_transitions: int = 0
    invalid_action_attempts: int = 0
    mask_disagreements: int = 0
    reward_mismatches: int = 0
    nonfinite_values: int = 0
    solver_failures: int = 0
    protected_safety_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


class AuditedV2Env(gym.Wrapper):
    """V2 wrapper enforcing mask, reward, solver, and seed contracts."""

    def __init__(self, env: gym.Env, seed_ledger: SeedLedger) -> None:
        from mplssim.rl.env_v2 import MplsTeEnvV2

        if not isinstance(env, MplsTeEnvV2):
            raise TypeError("AuditedV2Env requires MplsTeEnvV2")
        super().__init__(env)
        self.seed_ledger = seed_ledger
        self.integrity = IntegrityCounters()
        self._pre_action_mask: np.ndarray | None = None
        self._governed_root_seed = int(env.root_seed)
        self._governed_worker_rank = int(env.worker_rank)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        # SB3 seeds vector workers as model_seed + rank.  That seed is valid
        # for generic Gym environments, but V2 already derives independent
        # episode seeds from one governed root plus worker_rank.  Forwarding
        # SB3's value would count rank twice and put different algorithms on
        # different training scenarios.  Model RNG seeding remains handled by
        # SB3; the experiment wrapper deliberately preserves the V2 root.
        observation, info = self.env.reset(seed=None, options=options)
        if (
            int(info["root_seed"]) != self._governed_root_seed
            or int(info["worker_rank"]) != self._governed_worker_rank
        ):
            raise RuntimeError(
                "V2 training reset changed the governed root or worker rank")
        row = dict(info)
        row["episode_index"] = int(self.env._episode_index) - 1
        self.seed_ledger.record(row)
        self._pre_action_mask = self.env.action_masks().copy()
        return observation, info

    def action_masks(self) -> np.ndarray:
        mask = np.asarray(self.env.action_masks(), dtype=bool)
        self._pre_action_mask = mask.copy()
        return mask

    def step(self, action: int):
        from mplssim.rl.reward_v2 import components_sum

        action = int(action)
        mask = (self._pre_action_mask if self._pre_action_mask is not None
                else np.asarray(self.env.action_masks(), dtype=bool))
        if action < 0 or action >= len(mask) or not bool(mask[action]):
            self.integrity.invalid_action_attempts += 1
            raise RuntimeError(f"invalid action {action} selected under authoritative mask")
        if action > 0:
            d_idx, p_idx = divmod(action - 1, int(self.env.k))
            if bool(self.env.eng._protected[d_idx]):
                projected = self.env.eng.projected_gross_bottleneck(d_idx, p_idx)
                limit = float(self.env.engine_cfg.protected_projected_max_util)
                if projected > limit:
                    self.integrity.protected_safety_failures += 1
                    raise RuntimeError(
                        f"protected action {action} projects utilization "
                        f"{projected} above {limit}")

        observation, reward, terminated, truncated, info = self.env.step(action)
        actual_mask = np.asarray(self.env.action_masks(), dtype=bool)
        reported_mask = np.asarray(info["action_mask"], dtype=bool)
        if not np.array_equal(actual_mask, reported_mask):
            self.integrity.mask_disagreements += 1
            raise RuntimeError("post-step action mask disagrees with authoritative mask")

        component_total = components_sum(info["reward_components"])
        if component_total != float(reward):
            self.integrity.reward_mismatches += 1
            raise RuntimeError(
                f"reward component mismatch: {component_total!r} != {reward!r}")

        metrics = info["metrics"]
        numeric = [float(reward)]
        numeric.extend(
            float(value) for value in metrics.values()
            if isinstance(value, (int, float, np.integer, np.floating)))
        numeric.extend(float(value) for value in info["reward_components"].values())
        if not np.all(np.isfinite(numeric)):
            self.integrity.nonfinite_values += 1
            raise FloatingPointError("non-finite V2 reward or metric")

        cfg = self.env.engine_cfg.flow_solver
        if (int(info["flow_solver_iterations_max"]) > int(cfg.max_iterations)
                or float(metrics["flow_solver_residual"]) > 2.0 * float(cfg.tolerance)):
            self.integrity.solver_failures += 1
            raise RuntimeError("V2 flow solver integrity failure")

        self.integrity.aggregate_transitions += 1
        self._pre_action_mask = actual_mask.copy()
        return observation, reward, terminated, truncated, info


def make_audited_env(
    scenario: str,
    root_seed: int,
    worker_rank: int,
    seed_ledger: SeedLedger,
) -> AuditedV2Env:
    """Construct the experiment's only permitted environment type."""
    from mplssim.experiments.v2_factory import make_env_v2

    return AuditedV2Env(
        make_env_v2(
            scenario=scenario,
            root_seed=int(root_seed),
            worker_rank=int(worker_rank),
        ),
        seed_ledger=seed_ledger,
    )


def validate_source_identity(
    source: dict[str, Any],
    *,
    require_clean: bool,
) -> None:
    """Require an immutable Git commit and, for governed work, a clean tree."""
    commit = source.get("git_commit")
    if not isinstance(commit, str) or _FULL_SHA_RE.fullmatch(commit) is None:
        raise RuntimeError(f"source Git commit is not a full SHA: {commit!r}")
    if require_clean and source.get("git_dirty") is not False:
        raise RuntimeError(
            "governed training requires a clean committed checkout")


def verified_environment_record(
    root_seed: int,
    *,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Return the live V2 identity and signed-off definition-pin proof."""
    from mplssim.experiments.v2_factory import (
        assert_training_pin,
        build_environment_metadata,
        git_metadata,
    )

    validate_training_root(root_seed)
    pin = assert_training_pin()
    environment = build_environment_metadata(
        root_seed=int(root_seed), worker_rank=0)
    source = git_metadata()
    validate_source_identity(source, require_clean=require_clean)
    return {
        "environment": environment,
        "training_pin": pin,
        "source": source,
    }


class MetricsWriter:
    """Streaming step and episode records without retaining a full run in RAM."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._steps = gzip.open(
            self.run_directory / "training_steps.jsonl.gz",
            "wt", encoding="utf-8", compresslevel=6)
        self._episodes = (
            self.run_directory / "training_episodes.jsonl").open(
                "w", encoding="utf-8")

    @staticmethod
    def _line(record: dict[str, Any]) -> str:
        return json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"

    def write_step(self, record: dict[str, Any]) -> None:
        self._steps.write(self._line(record))

    def write_episode(self, record: dict[str, Any]) -> None:
        self._episodes.write(self._line(record))
        self._episodes.flush()

    def close(self) -> None:
        self._steps.close()
        self._episodes.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def sha256_file(path: Path) -> str:
    """Raw SHA-256 for binary experiment artifacts."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sidecar_path(payload_path: Path) -> Path:
    return Path(f"{payload_path}.metadata.json")


def write_checkpoint_sidecar(
    payload_path: Path,
    *,
    algorithm: str,
    aggregate_transitions: int,
    environment_record: dict[str, Any],
    run_config: dict[str, Any],
    device: dict[str, Any],
) -> Path:
    """Write the hash and complete identity required to reload a checkpoint."""
    payload = Path(payload_path)
    if not payload.is_file():
        raise FileNotFoundError(payload)
    metadata = {
        "format": "v2-learning-checkpoint-v1",
        "algorithm": algorithm,
        "aggregate_transitions": int(aggregate_transitions),
        "payload": payload.name,
        "payload_sha256": sha256_file(payload),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": environment_record["source"],
        "environment_record": environment_record,
        "run_config": run_config,
        "device": device,
    }
    sidecar = checkpoint_sidecar_path(payload)
    sidecar.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    return sidecar


def validate_checkpoint_sidecar(
    payload_path: Path,
    sidecar_path: Path | None = None,
    *,
    expected_algorithm: str,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    """Fail closed on checkpoint hash, algorithm, environment, or pin drift."""
    from mplssim.experiments.v2_factory import (
        assert_training_pin,
        git_metadata,
        validate_environment_metadata,
    )

    payload = Path(payload_path)
    sidecar = sidecar_path or checkpoint_sidecar_path(payload)
    metadata = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    if metadata.get("format") != "v2-learning-checkpoint-v1":
        raise ValueError(f"{sidecar}: unsupported checkpoint metadata format")
    if metadata.get("algorithm") != expected_algorithm:
        raise ValueError(
            f"checkpoint algorithm {metadata.get('algorithm')!r} != "
            f"{expected_algorithm!r}")
    if metadata.get("payload") != payload.name:
        raise ValueError(
            f"checkpoint payload name {metadata.get('payload')!r} != "
            f"{payload.name!r}")
    filename_match = _CHECKPOINT_RE.fullmatch(payload.name)
    if (
        filename_match is not None
        and int(filename_match.group(1))
        != int(metadata.get("aggregate_transitions", -1))
    ):
        raise ValueError(
            "checkpoint filename transition does not match checkpoint metadata")
    run_config = metadata.get("run_config", {})
    if run_config.get("algorithm") != expected_algorithm:
        raise ValueError("checkpoint run_config algorithm mismatch")
    if run_config.get("environment_version") != "v2":
        raise ValueError("checkpoint run_config is not V2")
    validate_training_root(int(run_config.get("root_seed", -1)))
    stored_source = metadata.get("source")
    if stored_source != metadata.get("environment_record", {}).get("source"):
        raise ValueError("checkpoint source identity records disagree")
    validate_source_identity(
        stored_source or {}, require_clean=require_clean_source)
    current_source = git_metadata()
    validate_source_identity(
        current_source, require_clean=require_clean_source)
    if stored_source["git_commit"] != current_source["git_commit"]:
        raise ValueError(
            "checkpoint source SHA does not match the current checkout")
    actual_hash = sha256_file(payload)
    if actual_hash != metadata.get("payload_sha256"):
        raise ValueError(
            f"checkpoint SHA-256 mismatch: stored "
            f"{metadata.get('payload_sha256')}, actual {actual_hash}")
    assert_training_pin()
    validate_environment_metadata(
        metadata["environment_record"]["environment"])
    return metadata


def hardware_inventory() -> dict[str, Any]:
    """Best-effort host and library inventory without claiming unusable CUDA."""
    try:
        import psutil
        ram_bytes: int | None = int(psutil.virtual_memory().total)
        physical_cores: int | None = psutil.cpu_count(logical=False)
    except (ImportError, OSError):
        ram_bytes = None
        physical_cores = None
    gpu_rows: list[dict[str, str]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15, check=True,
        )
        for line in result.stdout.splitlines():
            name, memory_mib, driver = [part.strip() for part in line.split(",", 2)]
            gpu_rows.append({
                "name": name,
                "memory_mib": memory_mib,
                "driver_version": driver,
            })
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    packages = {}
    for name in (
        "numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib",
        "pandas", "scipy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "physical_cores": physical_cores,
        "logical_cores": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "visible_nvidia_gpus": gpu_rows,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "packages": packages,
    }
