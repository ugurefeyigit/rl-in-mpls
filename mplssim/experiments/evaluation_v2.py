"""Shared deterministic V2 evaluation for learners and existing baselines."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mplssim.baselines import make_baseline
from mplssim.experiments.learning_common import (
    create_run_directory,
    SeedLedger,
    make_audited_env,
    resolve_device,
    validate_checkpoint_sidecar,
    validate_evaluation_seeds,
)
from mplssim.rl.reward_v2 import COMPONENT_ORDER, components_sum


BASELINE_ALGORITHMS: tuple[str, ...] = ("static", "greedy", "cspf")
LEARNER_ALGORITHMS: tuple[str, ...] = ("maskable_ppo", "masked_bandit")


class V2BaselineAdapter:
    """Read-only compatibility view for the repository's existing controllers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    @property
    def demand_volumes(self) -> np.ndarray:
        return self._engine.demand_offered

    @property
    def _path_links(self):
        return self._engine._cand_links

    @property
    def _path_costs(self):
        return self._engine._cand_cost

    def path_bottleneck_util(self, d_idx: int, p_idx: int) -> float:
        links = self._engine._cand_links[d_idx][p_idx]
        return float(np.max(self._engine.link_util[links])) if len(links) else 0.0

    def validate_action(
        self, d_idx: int, p_idx: int, source: str = "rl",
    ) -> tuple[bool, str]:
        return self._engine.validate_te_action(d_idx, p_idx)


def choose_baseline_action(
    controller: Any,
    engine: Any,
    authoritative_mask: np.ndarray,
) -> int:
    """Submit the first existing-controller proposal legal in V2, else no-op."""
    mask = np.asarray(authoritative_mask, dtype=bool)
    for d_idx, p_idx in controller.decide(V2BaselineAdapter(engine)):
        action = 1 + int(d_idx) * int(engine.k) + int(p_idx)
        if 0 <= action < len(mask) and bool(mask[action]):
            return action
    return 0


def _device_name(policy: Any | None) -> str:
    if policy is None or not hasattr(policy, "device"):
        return "cpu"
    return str(policy.device)


def run_evaluation_episode(
    *,
    algorithm: str,
    scenario: str,
    seed: int,
    policy: Any | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one full deterministic V2 episode and return steps plus summary."""
    validate_evaluation_seeds([seed])
    if algorithm not in BASELINE_ALGORITHMS + LEARNER_ALGORITHMS:
        raise ValueError(f"unknown V2 evaluation algorithm {algorithm!r}")
    if algorithm in LEARNER_ALGORITHMS and policy is None:
        raise ValueError(f"{algorithm} evaluation requires a loaded policy")

    ledger = SeedLedger()
    env = make_audited_env(
        scenario=scenario, root_seed=seed, worker_rank=0, seed_ledger=ledger)
    observation, _ = env.reset(options={"episode_seed": seed})
    controller = (
        make_baseline(algorithm, seed=seed)
        if algorithm in BASELINE_ALGORITHMS else None)
    records: list[dict[str, Any]] = []
    action_counts: Counter[int] = Counter()
    terminated = truncated = False
    decision_seconds = 0.0
    mask_seconds = 0.0
    wall_start = time.perf_counter()

    while not (terminated or truncated):
        mask_start = time.perf_counter()
        mask = env.action_masks()
        mask_seconds += time.perf_counter() - mask_start
        engine = env.unwrapped.eng
        dwell_active = int(np.sum(engine.te_dwell_remaining > 0))
        dwell_mean = float(np.mean(engine.te_dwell_remaining))

        decision_start = time.perf_counter()
        if controller is not None:
            action = choose_baseline_action(controller, engine, mask)
        else:
            predicted = policy.predict(
                observation[None, :], mask[None, :], deterministic=True)
            action = int(np.asarray(predicted).reshape(-1)[0])
        decision_seconds += time.perf_counter() - decision_start
        if not bool(mask[action]):
            raise RuntimeError(
                f"{algorithm} selected invalid evaluation action {action}")

        moved_mbps = 0.0
        if action > 0:
            demand_index = (action - 1) // engine.k
            moved_mbps = float(engine.demand_offered[demand_index])
        observation, reward, terminated, truncated, info = env.step(action)
        if not info["decoded_action"].get("accepted", False):
            moved_mbps = 0.0
        metrics = info["metrics"]
        record: dict[str, Any] = {
            "algorithm": algorithm,
            "scenario": scenario,
            "seed": int(seed),
            "episode_seed": int(info["episode_seed"]),
            "step_index": len(records),
            "action": int(action),
            "action_type": info["decoded_action"]["type"],
            "action_accepted": bool(
                info["decoded_action"].get("accepted", False)),
            "valid_action_count": int(np.sum(mask)),
            "reward": float(reward),
            "moved_mbps": moved_mbps,
            "dwell_active_demands": dwell_active,
            "dwell_remaining_mean": dwell_mean,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        record.update({
            key: value for key, value in metrics.items()
            if key != "failed_links"
        })
        record["n_failed_links"] = len(metrics["failed_links"])
        record.update({
            f"rc_{key}": float(value)
            for key, value in info["reward_components"].items()
        })
        records.append(record)
        action_counts[int(action)] += 1

    wall_time = time.perf_counter() - wall_start
    frame = pd.DataFrame(records)
    n_demands = int(env.unwrapped.n_demands)
    hours = (
        float(frame["t_min"].iloc[-1]) / 60.0 if len(frame) else 0.0)
    component_columns = [
        column for column in frame.columns if column.startswith("rc_")]
    component_sums = {
        column[3:]: float(frame[column].sum())
        for column in component_columns
    }
    reward_rows_exact = all(
        components_sum({
            name: float(row[f"rc_{name}"]) for name in COMPONENT_ORDER
        }) == float(row["reward"])
        for _, row in frame.iterrows()
    )
    accepted = int(frame["accepted_te_changes"].sum())
    reversals = int(frame["te_reversals"].sum())
    interval_seconds = (
        float(frame["t_min"].iloc[0]) * 60.0 if len(frame) else 300.0)
    protected_counts = frame["protected_disconnected_demands"]
    disconnected_counts = frame["disconnected_demands"]
    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "scenario": scenario,
        "seed": int(seed),
        "episode_seed": int(seed),
        "episode_length": int(len(frame)),
        "truncated": bool(truncated),
        "terminated": bool(terminated),
        "operational_return": float(frame["reward"].sum()),
        "reward_components": component_sums,
        "reward_component_sum_exact": bool(reward_rows_exact),
        "offered_gbit_total": float(
            (frame["offered_mbps"] * interval_seconds / 1000.0).sum()),
        "delivered_gbit_total": float(
            (frame["delivered_mbps"] * interval_seconds / 1000.0).sum()),
        "delivered_ratio_mean": float(frame["delivered_ratio"].mean()),
        "sla_violations_demand_intervals": int(frame["sla_violations"].sum()),
        "protected_disconnection_demand_intervals": int(protected_counts.sum()),
        "unprotected_disconnection_demand_intervals": int(
            (disconnected_counts - protected_counts).sum()),
        "max_utilization_peak": float(frame["max_util"].max()),
        "max_utilization_mean": float(frame["max_util"].mean()),
        "link_utilization_mean": float(frame["mean_util"].mean()),
        "congested_link_intervals": int(frame["congested_links"].sum()),
        "overload_ratio_mean": float(frame["overload_ratio"].mean()),
        "delay_ms_mean": float(frame["mean_delay_ms"].mean()),
        "delay_ms_max": float(frame["max_delay_ms"].max()),
        "loss_ratio_mean": float(frame["loss_ratio"].mean()),
        "accepted_te_changes": accepted,
        "reroutes_per_hour": accepted / hours if hours else 0.0,
        "te_reversals": reversals,
        "flaps_per_demand": reversals / n_demands,
        "moved_mbps_total": float(frame["moved_mbps"].sum()),
        "dwell_active_demand_intervals": int(
            frame["dwell_active_demands"].sum()),
        "dwell_remaining_mean": float(frame["dwell_remaining_mean"].mean()),
        "rejected_te_requests": int(frame["rejected_te_requests"].sum()),
        "frr_changes": int(frame["frr_changes"].sum()),
        "frr_disconnections": int(frame["frr_disconnections"].sum()),
        "recovery_restorations": int(frame["recovery_restorations"].sum()),
        "action_distribution": {
            str(action): count for action, count in sorted(action_counts.items())},
        "noop_frequency": int(action_counts[0]) / len(frame),
        "invalid_action_attempts": env.integrity.invalid_action_attempts,
        "mask_disagreements": env.integrity.mask_disagreements,
        "solver_iterations_mean": float(
            frame["flow_solver_iterations_max"].mean()),
        "solver_iterations_max": int(
            frame["flow_solver_iterations_max"].max()),
        "solver_convergence_failures": env.integrity.solver_failures,
        "protected_safety_failures":
            env.integrity.protected_safety_failures,
        "mean_decision_time_ms": 1000.0 * decision_seconds / len(frame),
        "mean_mask_time_ms": 1000.0 * mask_seconds / len(frame),
        "wall_time_seconds": wall_time,
        "resolved_device": _device_name(policy),
    }
    return frame, summary


def select_checkpoint(
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preregistered selection: valid max mean return, then earlier exact tie."""
    valid = [row for row in checkpoint_rows if bool(row.get("valid"))]
    if not valid:
        raise ValueError("no valid checkpoints are available for selection")
    return min(
        valid,
        key=lambda row: (
            -float(row["mean_operational_return"]),
            int(row["checkpoint_transition"]),
        ),
    )


def write_episode_outputs(
    output_directory: Path,
    steps: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Persist one evaluation episode in machine-readable formats."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{summary['algorithm']}_{summary['scenario']}_"
        f"seed{summary['seed']}")
    steps.to_csv(output_directory / f"{stem}_steps.csv", index=False)
    (output_directory / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")


def load_policy_checkpoint(
    payload_path: Path,
    *,
    algorithm: str,
    requested_device: str,
    require_clean_source: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Validate a learner checkpoint before constructing its inference policy."""
    if algorithm not in LEARNER_ALGORITHMS:
        raise ValueError(f"{algorithm!r} is not a learner checkpoint type")
    metadata = validate_checkpoint_sidecar(
        payload_path, expected_algorithm=algorithm,
        require_clean_source=require_clean_source)
    device = resolve_device(requested_device).torch_device
    if algorithm == "masked_bandit":
        from mplssim.experiments.masked_bandit import MaskedContextualBandit
        policy = MaskedContextualBandit.load(payload_path, device=device)
    else:
        from mplssim.experiments.trainers_v2 import MaskablePpoLearner
        policy = MaskablePpoLearner.load(payload_path, device=device)
    return policy, metadata


def evaluate_algorithm_matrix(
    *,
    algorithm: str,
    policy: Any | None,
    scenarios: list[str],
    seeds: list[int],
    output_directory: Path,
    write_steps: bool,
) -> pd.DataFrame:
    """Evaluate one learner checkpoint or baseline over a paired matrix."""
    validate_evaluation_seeds(seeds)
    output = create_run_directory(Path(output_directory))
    if write_steps:
        (output / "steps").mkdir(exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            steps, summary = run_evaluation_episode(
                algorithm=algorithm,
                scenario=scenario,
                seed=int(seed),
                policy=policy,
            )
            summaries.append(summary)
            if write_steps:
                path = (
                    output / "steps"
                    / f"{algorithm}_{scenario}_seed{seed}_steps.csv.gz"
                )
                steps.to_csv(path, index=False, compression="gzip")
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "episode_summary.csv", index=False)
    (output / "episode_summary.json").write_text(
        json.dumps(summaries, indent=1, allow_nan=False), encoding="utf-8")
    return frame
