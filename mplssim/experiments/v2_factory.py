"""Explicit V2 construction and fail-closed environment metadata.

Governing documents: docs/RL_ENVIRONMENT_V2_MIGRATION_PLAN.md ("Selection
architecture", "Compatibility safeguards") and
docs/RL_MODEL_TRAINING_CONTRACT.md ("Checkpoint metadata").

Version selection is always explicit. Nothing here is reachable without asking
for V2 by name, no default anywhere changes, and a version is *never* inferred
from a checkpoint's tensor shapes — 586 vs 604 would look like a shape mismatch
long after the real problem (a different candidate-path table pointed at by the
same action numbers) had already invalidated the run.

The metadata contract exists because V2 keeps V1's action numbering while
changing what those numbers mean. Actions 37-40 and 61-64 address D10 and D16;
their candidate 3 is a different router sequence in V2, and applying rule 7 of
the candidate spec reorders equal-cost candidates for D4, D5, D7, D10, D13 and
D15. A checkpoint that only recorded "Discrete(69)" would load happily and be
silently wrong, so the ordered router-sequence table is part of the identity and
any mismatch is fatal.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from mplssim.core.topology import CONFIG_DIR
from mplssim.paths.candidates_v2 import build_candidate_table
from mplssim.rl.env_v2 import MplsTeEnvV2, episode_seed_for
from mplssim.rl.reward_v2 import RewardConfigV2, load_reward_config_v2
from mplssim.sim.engine_v2 import (
    ACTION_VERSION,
    CONFIG_VERSION,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    REWARD_VERSION,
    SEED_VERSION,
    TRANSITION_VERSION,
    EngineConfigV2,
    SimulationEngineV2,
    load_engine_config_v2,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every configuration file whose content is part of the V2 environment identity.
HASHED_CONFIGS: tuple[str, ...] = (
    "topology.yaml",
    "traffic_classes.yaml",
    "scenarios.yaml",
    "experiments/rl_env_v2.yaml",
    "experiments/rl_observation_v2.yaml",
    "experiments/rl_reward_v2.yaml",
)


class MetadataMismatchError(ValueError):
    """Stored V2 metadata disagrees with the live environment. Always fatal.

    Never repaired by padding an observation, reordering actions, falling back
    to V1 or rewriting the stored record.
    """


# --------------------------------------------------------------- construction
def make_engine_v2(scenario: str, episode_seed: int,
                   cfg: EngineConfigV2 | None = None) -> SimulationEngineV2:
    """Build a V2 engine for one episode from an explicit episode seed."""
    from mplssim.factory import get_scenarios, get_topology, get_traffic_config
    scenarios = get_scenarios()
    if scenario not in scenarios:
        raise KeyError(f"unknown scenario '{scenario}' (have: {sorted(scenarios)})")
    return SimulationEngineV2(
        topo=get_topology(), traffic_cfg=get_traffic_config(),
        scenario=scenarios[scenario], episode_seed=int(episode_seed),
        cfg=cfg or load_engine_config_v2(),
    )


def make_env_v2(
    scenario: str = "random_day",
    root_seed: int = 0,
    worker_rank: int = 0,
    engine_cfg: EngineConfigV2 | None = None,
    reward_cfg: RewardConfigV2 | None = None,
    include_time_of_day: bool = False,
) -> MplsTeEnvV2:
    """Build the V2 Gym environment."""
    return MplsTeEnvV2(
        scenario=scenario, root_seed=root_seed, worker_rank=worker_rank,
        engine_cfg=engine_cfg, reward_cfg=reward_cfg,
        include_time_of_day=include_time_of_day,
    )


# -------------------------------------------------------------------- hashing
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hashes() -> dict[str, str]:
    """SHA-256 of every configuration file in the V2 identity, by posix key."""
    out: dict[str, str] = {}
    for rel in HASHED_CONFIGS:
        path = CONFIG_DIR / rel
        if not path.exists():
            raise FileNotFoundError(f"config {path} required for V2 metadata")
        out[f"configs/{rel}"] = sha256_file(path)
    return out


def git_metadata() -> dict[str, Any]:
    """Current commit and dirty-tree status, or ``unknown`` outside a checkout."""
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                check=True, timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "git_commit": commit or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": None if status is None else bool(status.strip()),
    }


# ------------------------------------------------------------------- metadata
def build_environment_metadata(
    env: MplsTeEnvV2 | None = None,
    engine_cfg: EngineConfigV2 | None = None,
    root_seed: int | None = None,
    worker_rank: int | None = None,
) -> dict[str, Any]:
    """Complete V2 environment identity record.

    Written before any V2 result and compared field by field on load.
    """
    from mplssim.factory import get_topology, get_traffic_config
    cfg = engine_cfg or (env.engine_cfg if env is not None else load_engine_config_v2())
    reward_cfg = env.reward_cfg if env is not None else load_reward_config_v2()
    topo = get_topology()
    traffic_cfg = get_traffic_config()

    table = build_candidate_table(
        topo, traffic_cfg.demands, k=cfg.k_paths,
        max_hop_factor=cfg.max_hop_factor,
        delay_factor=cfg.candidate_delay_factor,
        delay_additive_ms=cfg.candidate_delay_additive_ms,
    )
    demand_ids = [d.id for d in traffic_cfg.demands]
    n_demands, k = len(demand_ids), cfg.k_paths

    versions = env.environment_versions() if env is not None else {
        "environment": ENVIRONMENT_VERSION,
        "observation": OBSERVATION_VERSION,
        "action": ACTION_VERSION,
        "reward": REWARD_VERSION,
        "transition": TRANSITION_VERSION,
        "config": CONFIG_VERSION,
        "seed_protocol": SEED_VERSION,
    }
    obs_dim = (int(env.observation_space.shape[0]) if env is not None
               else 2 * topo.n_dlinks + (8 + 5 * k) * n_demands)

    meta: dict[str, Any] = dict(versions)
    meta.update({
        "environment_class": "mplssim.rl.env_v2.MplsTeEnvV2",
        "engine_class": "mplssim.sim.engine_v2.SimulationEngineV2",
        "observation_dim": obs_dim,
        "action_dim": 1 + n_demands * k,
        "n_demands": n_demands,
        "k_paths": k,
        "n_dlinks": topo.n_dlinks,
        "demand_ids": demand_ids,
        "candidate_paths": {d: [list(p) for p in table[d]] for d in demand_ids},
        "config_hashes": config_hashes(),
        "engine_config": {
            "control_interval_min": cfg.control_interval_min,
            "micro_ticks_per_interval": cfg.micro_ticks_per_interval,
            "k_paths": cfg.k_paths,
            "max_hop_factor": cfg.max_hop_factor,
            "minimum_te_dwell_steps": cfg.minimum_te_dwell_steps,
            "reversal_window_steps": cfg.reversal_window_steps,
            "max_te_changes_per_interval": cfg.max_te_changes_per_interval,
            "candidate_delay_factor": cfg.candidate_delay_factor,
            "candidate_delay_additive_ms": cfg.candidate_delay_additive_ms,
            "protected_projected_max_util": cfg.protected_projected_max_util,
            "flow_solver": {
                "damping": cfg.flow_solver.damping,
                "tolerance": cfg.flow_solver.tolerance,
                "max_iterations": cfg.flow_solver.max_iterations,
            },
            "worker_stride": cfg.worker_stride,
        },
        "reward_config": {
            "delivered": reward_cfg.delivered,
            "protected_disconnect": reward_cfg.protected_disconnect,
            "unprotected_disconnect": reward_cfg.unprotected_disconnect,
            "sla_severity": reward_cfg.sla_severity,
            "max_util": reward_cfg.max_util,
            "overload": reward_cfg.overload,
            "util_free_threshold": reward_cfg.util_free_threshold,
            "util_span": reward_cfg.util_span,
            "potential_coefficient": reward_cfg.potential_coefficient,
            "potential_gamma": reward_cfg.potential_gamma,
            "potential_scale": reward_cfg.potential_scale,
            "move_fixed": reward_cfg.move_fixed,
            "move_volume_share": reward_cfg.move_volume_share,
            "move_edge_divergence": reward_cfg.move_edge_divergence,
            "move_reversal": reward_cfg.move_reversal,
            "invalid": reward_cfg.invalid,
        },
        "seed": {
            "protocol": "root_seed + worker_rank + worker_stride*episode_index",
            "worker_stride": cfg.worker_stride,
            "max_worker_rank": cfg.worker_stride - 1,
            "root_seed": (root_seed if root_seed is not None
                          else (env.root_seed if env is not None else None)),
            "worker_rank": (worker_rank if worker_rank is not None
                            else (env.worker_rank if env is not None else None)),
        },
    })
    meta.update(git_metadata())
    return meta


#: Fields compared on load. Deliberately excludes git status and the per-run
#: seed block, which legitimately differ between the run that trained a
#: checkpoint and the run that loads it.
IDENTITY_FIELDS: tuple[str, ...] = (
    "environment", "observation", "action", "reward", "transition", "config",
    "seed_protocol", "environment_class", "engine_class", "observation_dim",
    "action_dim", "n_demands", "k_paths", "n_dlinks", "demand_ids",
    "candidate_paths", "config_hashes", "engine_config", "reward_config",
)


def validate_environment_metadata(stored: dict[str, Any],
                                  current: dict[str, Any] | None = None) -> None:
    """Compare stored metadata against the live environment. Fail closed.

    Raises :class:`MetadataMismatchError` naming the first offending field, with
    enough detail to identify what actually changed.
    """
    current = current or build_environment_metadata()
    for field_name in IDENTITY_FIELDS:
        if field_name not in stored:
            raise MetadataMismatchError(
                f"stored V2 metadata is missing required field {field_name!r}")
        want, got = current[field_name], stored[field_name]
        if want == got:
            continue
        if field_name == "candidate_paths":
            detail = _candidate_path_diff(got, want)
        elif field_name == "config_hashes":
            detail = "; ".join(
                f"{key}: stored {got.get(key)!r} != current {want.get(key)!r}"
                for key in sorted(set(got) | set(want))
                if got.get(key) != want.get(key))
        else:
            detail = f"stored {got!r} != current {want!r}"
        raise MetadataMismatchError(
            f"V2 environment metadata mismatch in {field_name!r}: {detail}. "
            f"Refusing to load: V2 never pads, truncates, reorders or falls "
            f"back to V1.")


def _candidate_path_diff(stored: Any, current: Any) -> str:
    if not isinstance(stored, dict):
        return f"stored candidate table has type {type(stored).__name__}"
    lines = []
    for demand_id in sorted(set(stored) | set(current)):
        s, c = stored.get(demand_id), current.get(demand_id)
        if s == c:
            continue
        if s is None or c is None:
            lines.append(f"{demand_id}: stored={s} current={c}")
            continue
        for p_idx, (sp, cp) in enumerate(zip(s, c)):
            if list(sp) != list(cp):
                lines.append(f"{demand_id} candidate {p_idx}: "
                             f"stored {'-'.join(sp)} != current {'-'.join(cp)}")
        if len(s) != len(c):
            lines.append(f"{demand_id}: {len(s)} stored candidates vs {len(c)} current")
    return "; ".join(lines) if lines else "tables differ"


__all__ = [
    "MetadataMismatchError",
    "make_engine_v2",
    "make_env_v2",
    "episode_seed_for",
    "config_hashes",
    "git_metadata",
    "build_environment_metadata",
    "validate_environment_metadata",
    "IDENTITY_FIELDS",
]
