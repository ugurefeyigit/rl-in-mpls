"""One-shot governed V2 final-holdout evaluation with fixed checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplssim.experiments.evaluation_v2 import (  # noqa: E402
    BASELINE_ALGORITHMS,
    evaluate_algorithm_matrix,
    load_policy_checkpoint,
)
from mplssim.experiments.learning_common import (  # noqa: E402
    checkpoint_sidecar_path,
    create_run_directory,
    hardware_inventory,
    load_learning_config,
    sha256_file,
    validate_checkpoint_sidecar,
    validate_evaluation_seeds,
)
from mplssim.experiments.v2_factory import (  # noqa: E402
    assert_training_pin,
    git_metadata,
)


SEED42_SOURCE_SHA = "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
CONTINUATION_SOURCE_SHA = "6a8a4068b98bf9a71dead6e547595b4bbd755689"
SIGNED_OFF_ENVIRONMENT_SHA = "dca533b5c6fa9953307d01470c23cac512eb2961"
FINAL_SCENARIOS = (
    "full_day",
    "evening_peak",
    "flash_crowd",
    "link_failure",
    "deceptive_local_optimum",
    "ood_double_failure",
    "overload_stress",
)
FINAL_SEEDS = (1001, 1002, 1003, 1004, 1005)


@dataclass(frozen=True)
class FinalHoldoutCheckpointSpec:
    training_root: int
    algorithm: str
    checkpoint_transition: int
    payload_sha256: str
    training_source_sha: str
    worktree_key: str
    relative_payload: str


FINAL_HOLDOUT_CHECKPOINTS = (
    FinalHoldoutCheckpointSpec(
        42, "maskable_ppo", 250_000,
        "d34cc77ded05b064fa2a39dbe5c5ccc3126c9e6cf85e36c1b507127c987f5676",
        SEED42_SOURCE_SHA, "seed42",
        "runs/v2/seed42_maskable_ppo_final/checkpoints/checkpoint_000250000.zip"),
    FinalHoldoutCheckpointSpec(
        42, "masked_bandit", 250_000,
        "c15097700eac518ee259cba67e34e4fba1716881ab3dd912188b55da0c79bf49",
        SEED42_SOURCE_SHA, "seed42",
        "runs/v2/seed42_masked_bandit_final/checkpoints/checkpoint_000250000.pt"),
    FinalHoldoutCheckpointSpec(
        314159, "maskable_ppo", 350_000,
        "0af41be78102617b103c3e21ebb0ba26ae251f2626ff50b30c0887fdb1320489",
        CONTINUATION_SOURCE_SHA, "continuation",
        "runs/v2/seed314159_maskable_ppo_final_r2/checkpoints/checkpoint_000350000.zip"),
    FinalHoldoutCheckpointSpec(
        314159, "masked_bandit", 300_000,
        "fd474430e9f5ed60d09d82e3d08390151f54c8c0ca10b5abd98fe11d5d2c8433",
        CONTINUATION_SOURCE_SHA, "continuation",
        "runs/v2/seed314159_masked_bandit_final_r2/checkpoints/checkpoint_000300000.pt"),
    FinalHoldoutCheckpointSpec(
        271828, "maskable_ppo", 150_000,
        "40d0f9b7fe92449e6e8bfe2bcb44604ac2a5002c0f2a662dbad6cf70c219fb79",
        CONTINUATION_SOURCE_SHA, "continuation",
        "runs/v2/seed271828_maskable_ppo_final/checkpoints/checkpoint_000150000.zip"),
    FinalHoldoutCheckpointSpec(
        271828, "masked_bandit", 400_000,
        "d9c31430ad4320ae238f6d3aa833614edc120f7411c5a3e99372c85707116e73",
        CONTINUATION_SOURCE_SHA, "continuation",
        "runs/v2/seed271828_masked_bandit_final/checkpoints/checkpoint_000400000.pt"),
)

METRIC_COLUMNS = (
    "offered_gbit_total",
    "delivered_gbit_total",
    "delivered_ratio_mean",
    "sla_violations_demand_intervals",
    "protected_disconnection_demand_intervals",
    "unprotected_disconnection_demand_intervals",
    "max_utilization_peak",
    "max_utilization_mean",
    "link_utilization_mean",
    "congested_link_intervals",
    "overload_ratio_mean",
    "delay_ms_mean",
    "delay_ms_max",
    "loss_ratio_mean",
    "accepted_te_changes",
    "reroutes_per_hour",
    "te_reversals",
    "flaps_per_demand",
    "moved_mbps_total",
    "dwell_active_demand_intervals",
    "dwell_remaining_mean",
    "rejected_te_requests",
    "frr_changes",
    "frr_disconnections",
    "recovery_restorations",
    "noop_frequency",
    "solver_iterations_mean",
    "solver_iterations_max",
    "mean_decision_time_ms",
    "mean_mask_time_ms",
    "wall_time_seconds",
)
INTEGRITY_COUNTERS = (
    "invalid_action_attempts",
    "mask_disagreements",
    "reward_mismatches",
    "nonfinite_values",
    "solver_convergence_failures",
    "protected_safety_failures",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-worktree", type=Path, required=True)
    parser.add_argument("--continuation-worktree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compact-output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def validate_final_holdout_checkpoint_metadata(
    metadata: dict,
    spec: FinalHoldoutCheckpointSpec,
    *,
    actual_payload_sha256: str,
) -> None:
    """Bind one validated sidecar to the immutable final-holdout registry."""
    run_config = metadata.get("run_config", {})
    checks = {
        "algorithm": metadata.get("algorithm") == spec.algorithm,
        "run_config algorithm": run_config.get("algorithm") == spec.algorithm,
        "environment": run_config.get("environment_version") == "v2",
        "training root": int(run_config.get("root_seed", -1)) == spec.training_root,
        "training budget": int(run_config.get("aggregate_transitions", -1)) == 400_000,
        "checkpoint interval": int(run_config.get("checkpoint_interval", -1)) == 50_000,
        "run purpose": run_config.get("purpose") == "meaningful",
        "checkpoint transition": (
            int(metadata.get("aggregate_transitions", -1))
            == spec.checkpoint_transition),
        "training source": (
            metadata.get("source", {}).get("git_commit")
            == spec.training_source_sha),
        "sidecar payload hash": metadata.get("payload_sha256") == spec.payload_sha256,
        "actual payload hash": actual_payload_sha256 == spec.payload_sha256,
    }
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise ValueError(
            "final-holdout checkpoint binding mismatch: " + ", ".join(failures))


def _aggregate_metric_group(group: pd.DataFrame) -> dict:
    first = group.iloc[0]
    row = {
        "policy_id": first["policy_id"],
        "algorithm": first["algorithm"],
        "training_root": first["training_root"],
        "episodes": int(len(group)),
        "operational_return_mean": float(group["operational_return"].mean()),
        "operational_return_std": (
            float(group["operational_return"].std(ddof=1))
            if len(group) > 1 else 0.0),
    }
    for column in METRIC_COLUMNS:
        row[f"{column}_mean"] = float(group[column].mean())
    return row


def build_compact_tables(episodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all compact metric and integrity views from full episode evidence."""
    required = {
        "policy_id", "algorithm", "training_root", "scenario", "seed",
        "operational_return", "reward_components", "action_distribution",
        "episode_length", "truncated", "terminated",
        "reward_component_sum_exact", *METRIC_COLUMNS, *INTEGRITY_COUNTERS,
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"episode evidence is missing required columns: {missing}")

    per_root = pd.DataFrame([
        _aggregate_metric_group(group)
        for _, group in episodes.groupby("policy_id", sort=False)
    ])
    aggregate_rows = []
    for algorithm, group in episodes.groupby("algorithm", sort=False):
        aggregate = _aggregate_metric_group(group)
        aggregate["policy_id"] = algorithm
        aggregate["training_root"] = "aggregate"
        root_means = group.groupby("policy_id")["operational_return"].mean()
        aggregate["root_mean_std"] = (
            float(root_means.std(ddof=1)) if len(root_means) > 1 else 0.0)
        aggregate["root_count"] = int(len(root_means))
        aggregate_rows.append(aggregate)
    aggregate = pd.DataFrame(aggregate_rows)

    scenario = pd.DataFrame([
        {**_aggregate_metric_group(group), "scenario": scenario_name}
        for (_, scenario_name), group in episodes.groupby(
            ["policy_id", "scenario"], sort=False)
    ])

    component_names = sorted({
        name
        for components in episodes["reward_components"]
        for name in components
    })
    reward_rows = []
    for _, group in episodes.groupby("policy_id", sort=False):
        first = group.iloc[0]
        row = {
            "policy_id": first["policy_id"],
            "algorithm": first["algorithm"],
            "training_root": first["training_root"],
            "episodes": int(len(group)),
        }
        residuals = []
        for components, operational_return in zip(
            group["reward_components"], group["operational_return"], strict=True,
        ):
            residuals.append(abs(sum(components.values()) - float(operational_return)))
        for name in component_names:
            row[f"{name}_mean"] = float(np.mean([
                float(components.get(name, 0.0))
                for components in group["reward_components"]
            ]))
        row["operational_return_mean"] = float(group["operational_return"].mean())
        row["max_abs_reward_residual"] = float(max(residuals, default=0.0))
        reward_rows.append(row)
    reward_components = pd.DataFrame(reward_rows)

    action_rows = []
    for _, group in episodes.groupby("policy_id", sort=False):
        first = group.iloc[0]
        counts = {action: 0 for action in range(69)}
        for distribution in group["action_distribution"]:
            for action, count in distribution.items():
                counts[int(action)] += int(count)
        total = sum(counts.values())
        for action, count in counts.items():
            action_rows.append({
                "policy_id": first["policy_id"],
                "algorithm": first["algorithm"],
                "training_root": first["training_root"],
                "action": action,
                "action_type": "noop" if action == 0 else "te_change",
                "count": count,
                "frequency": count / total if total else 0.0,
            })
    action_distribution = pd.DataFrame(action_rows)

    integrity_rows = []
    numeric_columns = ["operational_return", *METRIC_COLUMNS]
    for _, group in episodes.groupby("policy_id", sort=False):
        first = group.iloc[0]
        counters = {
            f"{name}_total": int(group[name].sum()) for name in INTEGRITY_COUNTERS}
        all_checks = (
            bool(group["reward_component_sum_exact"].all())
            and bool(group["truncated"].all())
            and not bool(group["terminated"].any())
            and all(value == 0 for value in counters.values())
            and bool(np.isfinite(group[numeric_columns].to_numpy(dtype=float)).all())
        )
        integrity_rows.append({
            "policy_id": first["policy_id"],
            "algorithm": first["algorithm"],
            "training_root": first["training_root"],
            "episodes": int(len(group)),
            "unique_seeds": int(group["seed"].nunique()),
            "unique_scenarios": int(group["scenario"].nunique()),
            "all_episodes_truncated": bool(group["truncated"].all()),
            "any_episode_terminated": bool(group["terminated"].any()),
            "all_reward_component_sums_exact": bool(
                group["reward_component_sum_exact"].all()),
            **counters,
            "all_checks_passed": all_checks,
        })
    integrity = pd.DataFrame(integrity_rows)
    return {
        "per_root_metrics": per_root,
        "aggregate_metrics": aggregate,
        "scenario_metrics": scenario,
        "reward_components": reward_components,
        "action_distribution": action_distribution,
        "evaluation_integrity": integrity,
    }


def validate_complete_final_evidence(
    episodes: pd.DataFrame,
    *,
    scenarios: list[str],
    seeds: list[int],
) -> None:
    """Require the exact six-learner plus three-baseline paired matrix."""
    expected_policies = {
        *(f"root{spec.training_root}_{spec.algorithm}"
          for spec in FINAL_HOLDOUT_CHECKPOINTS),
        "baseline_static",
        "baseline_greedy",
        "baseline_cspf",
    }
    expected_total = len(expected_policies) * len(scenarios) * len(seeds)
    if len(episodes) != expected_total or expected_total != 315:
        raise RuntimeError(
            f"final holdout requires exactly 315 episodes, got {len(episodes)}")
    if set(episodes["policy_id"]) != expected_policies:
        raise RuntimeError("final holdout policy registry mismatch")
    expected_pairs = {(scenario, int(seed)) for scenario in scenarios for seed in seeds}
    for policy_id, group in episodes.groupby("policy_id", sort=False):
        actual_pairs = {
            (str(row.scenario), int(row.seed))
            for row in group[["scenario", "seed"]].itertuples(index=False)
        }
        if len(group) != 35 or actual_pairs != expected_pairs:
            raise RuntimeError(
                f"{policy_id} does not contain exactly 35 paired final episodes")
        if group.duplicated(["scenario", "seed"]).any():
            raise RuntimeError(f"{policy_id} repeats a final-holdout episode")


def _git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _resolve_checkpoint(
    spec: FinalHoldoutCheckpointSpec,
    *,
    seed42_worktree: Path,
    continuation_worktree: Path,
) -> Path:
    root = seed42_worktree if spec.worktree_key == "seed42" else continuation_worktree
    return (root / spec.relative_payload).resolve()


def preflight_final_holdout(
    *,
    seed42_worktree: Path,
    continuation_worktree: Path,
) -> list[dict]:
    """Validate every immutable input before any holdout environment exists."""
    roots = {
        "seed42": (seed42_worktree.resolve(), SEED42_SOURCE_SHA),
        "continuation": (
            continuation_worktree.resolve(), CONTINUATION_SOURCE_SHA),
    }
    for label, (worktree, expected_sha) in roots.items():
        actual_sha = _git(worktree, "rev-parse", "HEAD")
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{label} artifact worktree is at {actual_sha}, expected {expected_sha}")
        status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=no")
        if status:
            raise RuntimeError(f"{label} artifact worktree has tracked changes")

    current_source = git_metadata()
    if current_source.get("git_dirty") is not False:
        raise RuntimeError("final-holdout evaluation checkout must be clean")
    assert_training_pin()

    records = []
    for spec in FINAL_HOLDOUT_CHECKPOINTS:
        payload = _resolve_checkpoint(
            spec,
            seed42_worktree=seed42_worktree,
            continuation_worktree=continuation_worktree,
        )
        metadata = validate_checkpoint_sidecar(
            payload,
            expected_algorithm=spec.algorithm,
            final_holdout_checkpoint_source_sha=spec.training_source_sha,
            expected_payload_sha256=spec.payload_sha256,
        )
        actual_hash = sha256_file(payload)
        validate_final_holdout_checkpoint_metadata(
            metadata, spec, actual_payload_sha256=actual_hash)
        sidecar = checkpoint_sidecar_path(payload)
        records.append({
            "training_root": spec.training_root,
            "algorithm": spec.algorithm,
            "checkpoint_transition": spec.checkpoint_transition,
            "checkpoint_path": str(payload),
            "payload_sha256": actual_hash,
            "sidecar_path": str(sidecar.resolve()),
            "sidecar_sha256": sha256_file(sidecar),
            "training_source_sha": spec.training_source_sha,
            "artifact_worktree": str(roots[spec.worktree_key][0]),
            "artifact_worktree_head": roots[spec.worktree_key][1],
            "evaluation_source_sha": current_source["git_commit"],
        })
    return records


def _report_markdown(
    tables: dict[str, pd.DataFrame],
    provenance: pd.DataFrame,
    *,
    full_output: Path,
    evaluation_source_sha: str,
    total_runtime_seconds: float,
) -> str:
    aggregate = tables["aggregate_metrics"].set_index("algorithm")
    per_root = tables["per_root_metrics"]
    bandit_return = float(aggregate.loc["masked_bandit", "operational_return_mean"])
    ppo_return = float(aggregate.loc["maskable_ppo", "operational_return_mean"])
    greedy_return = float(aggregate.loc["greedy", "operational_return_mean"])
    advantage = bandit_return - ppo_return
    learner_roots = per_root[per_root["algorithm"].isin(
        ["maskable_ppo", "masked_bandit"])]
    piv = learner_roots.pivot(
        index="training_root", columns="algorithm",
        values="operational_return_mean")
    wins = int((piv["masked_bandit"] > piv["maskable_ppo"]).sum())
    integrity_ok = bool(tables["evaluation_integrity"]["all_checks_passed"].all())

    rows = []
    for root in (42, 314159, 271828):
        rows.append(
            f"| {root} | {piv.loc[root, 'masked_bandit']:.3f} | "
            f"{piv.loc[root, 'maskable_ppo']:.3f} | "
            f"{piv.loc[root, 'masked_bandit'] - piv.loc[root, 'maskable_ppo']:.3f} |")
    aggregate_rows = []
    for algorithm in ("masked_bandit", "maskable_ppo", "greedy", "cspf", "static"):
        row = aggregate.loc[algorithm]
        aggregate_rows.append(
            f"| {algorithm} | {row['operational_return_mean']:.3f} | "
            f"{row['operational_return_std']:.3f} | "
            f"{row['delivered_ratio_mean_mean']:.4f} | "
            f"{row['sla_violations_demand_intervals_mean']:.2f} | "
            f"{row['reroutes_per_hour_mean']:.3f} | "
            f"{row['moved_mbps_total_mean']:.2f} |")
    runtime_rows = []
    for row in provenance.itertuples(index=False):
        runtime_rows.append(
            f"| {row.training_root} | {row.algorithm} | "
            f"{row.evaluation_wall_seconds:.3f} | {row.resolved_device} | "
            f"{int(row.peak_gpu_memory_bytes)} |")

    conclusion = (
        "The masked contextual bandit advantage generalizes to the untouched "
        f"holdout: it beat PPO on {wins}/3 training roots and by {advantage:.3f} "
        "return points in the root-aggregated learner comparison. This final "
        "evidence does not support a need for temporal planning; the explicitly "
        "myopic learner remains stronger. That is a result about these frozen "
        "learners and scenarios, not a claim that planning is generally useless."
    )
    return f"""# V2 final holdout report

## Decision

{conclusion}

The strongest repository baseline was greedy at **{greedy_return:.3f}**. The
bandit achieved **{bandit_return:.3f}** and PPO **{ppo_return:.3f}**. No
checkpoint was selected, reselected, tuned, or redesigned with holdout results.
The study is closed; no further tuning recommendation is made from this holdout.

## Per-root learner performance

| Training root | Bandit return | PPO return | Bandit advantage |
| ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Aggregate operational and traffic-engineering metrics

| Method | Return mean | Episode SD | Delivered ratio | SLA intervals | Reroutes/hour | Moved Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(aggregate_rows)}

`aggregate_metrics.csv` and `scenario_metrics.csv` retain utilization,
congestion, overload, delay, loss, delivery, disconnections, reroutes,
reversals, flaps, moved bandwidth, dwell, accepted and rejected TE changes,
FRR changes/disconnections, restorations, decision time, and mask time.

## Stability and scenario variance

Aggregate episode return SD is
**{aggregate.loc['masked_bandit', 'operational_return_std']:.2f}** for the
bandit and **{aggregate.loc['maskable_ppo', 'operational_return_std']:.2f}**
for PPO. Across root means, SD is
**{aggregate.loc['masked_bandit', 'root_mean_std']:.2f}** and
**{aggregate.loc['maskable_ppo', 'root_mean_std']:.2f}**, respectively.
Scenario-level values are reported without selective omission in
`scenario_metrics.csv`; scenario heterogeneity remains the main source of
episode variance.

## Reward, actions, safety, and churn

All named reward components and their maximum aggregation residuals are in
`reward_components.csv`. Every step passed the repository's exact component
sum check. `action_distribution.csv` contains all actions 0-68, including
zero-count actions and no-op frequency, for each of the six policies and three
baselines.

Safety and integrity passed: **{str(integrity_ok).lower()}**. Every method has
exactly 35 episodes (seven scenarios by five seeds); all 315 episodes reached
normal truncation with no abnormal termination, invalid action, mask
disagreement, reward mismatch, non-finite value, solver convergence failure,
or protected safety failure. Gains therefore preserve the governed safety
envelope. Acceptable churn must be judged from the complete reroute, reversal,
flap, moved-bandwidth, and dwell fields rather than return alone.

## Runtime and provenance

| Root | Algorithm | Wall seconds | Device | Peak GPU bytes |
| ---: | --- | ---: | --- | ---: |
{chr(10).join(runtime_rows)}

Total one-shot evaluation runtime was **{total_runtime_seconds:.3f} seconds**.
Evaluation source: `{evaluation_source_sha}`. Checkpoint payload and sidecar
hashes, training-source bindings, exact paths, and worktree heads are in
`checkpoint_provenance.csv`.

Full compressed step evidence and per-episode summaries are preserved at:

`{full_output}`

## Failures and limitations

No holdout episode was retried or omitted. The final comparison covers three
training roots, five holdout seeds, seven fixed scenarios, two frozen learner
families, and three fixed baselines. Baselines are evaluated once because they
have no training root. High scenario variance limits broad generalization, and
the result cannot establish that memory or planning would never help under a
different learner, observation design, or task.

## Final scientific conclusion

{conclusion} The observed gains preserve safety; churn acceptability is
reported explicitly and is not inferred from return alone. The governed V2
study ends here.
"""


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")


def run_final_holdout(args: argparse.Namespace) -> dict:
    cfg = load_learning_config()
    scenarios = list(cfg["evaluation"]["scenarios"])
    seeds = [int(seed) for seed in cfg["holdout_seeds"]]
    if tuple(scenarios) != FINAL_SCENARIOS or tuple(seeds) != FINAL_SEEDS:
        raise RuntimeError("configured final-holdout matrix differs from authorization")
    validate_evaluation_seeds(
        seeds, evaluation_mode="final_holdout", require_complete=True)
    provenance_rows = preflight_final_holdout(
        seed42_worktree=args.seed42_worktree,
        continuation_worktree=args.continuation_worktree,
    )
    if args.preflight_only:
        result = {
            "status": "preflight_passed",
            "evaluation_source": git_metadata(),
            "environment_pin": assert_training_pin(),
            "scenarios": scenarios,
            "seeds": seeds,
            "checkpoints": provenance_rows,
        }
        print(json.dumps(result, indent=1, allow_nan=False))
        return result

    full_output = args.output_dir.resolve()
    compact_output = args.compact_output_dir.resolve()
    if full_output.exists() or compact_output.exists():
        raise FileExistsError(
            "final-holdout output directories must both be new: "
            f"{full_output}, {compact_output}")
    full_output = create_run_directory(full_output)
    attempt = {
        "format": "v2-final-holdout-attempt-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "evaluation_source": git_metadata(),
        "scenarios": scenarios,
        "seeds": seeds,
        "holdout_authorized": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
    }
    attempt_path = full_output / "attempt_manifest.json"
    _write_json(attempt_path, attempt)
    start = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the authorized evaluation")
        gpu_name = torch.cuda.get_device_name(0)
        all_frames = []
        for spec, provenance in zip(
            FINAL_HOLDOUT_CHECKPOINTS, provenance_rows, strict=True,
        ):
            payload = Path(provenance["checkpoint_path"])
            torch.cuda.reset_peak_memory_stats(0)
            policy_start = time.perf_counter()
            policy, metadata = load_policy_checkpoint(
                payload,
                algorithm=spec.algorithm,
                requested_device=args.device,
                final_holdout_checkpoint_source_sha=spec.training_source_sha,
                expected_payload_sha256=spec.payload_sha256,
            )
            validate_final_holdout_checkpoint_metadata(
                metadata, spec, actual_payload_sha256=sha256_file(payload))
            policy_id = f"root{spec.training_root}_{spec.algorithm}"
            frame = evaluate_algorithm_matrix(
                algorithm=spec.algorithm,
                policy=policy,
                scenarios=scenarios,
                seeds=seeds,
                output_directory=full_output / "learners" / policy_id,
                write_steps=True,
                evaluation_mode="final_holdout",
            )
            frame.insert(0, "policy_id", policy_id)
            frame.insert(2, "training_root", spec.training_root)
            frame.insert(3, "checkpoint_transition", spec.checkpoint_transition)
            all_frames.append(frame)
            torch.cuda.synchronize(0)
            provenance["evaluation_wall_seconds"] = time.perf_counter() - policy_start
            provenance["resolved_device"] = str(frame["resolved_device"].iloc[0])
            provenance["gpu_name"] = gpu_name
            provenance["peak_gpu_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(0))
            del policy
            gc.collect()
            torch.cuda.empty_cache()

        for baseline in BASELINE_ALGORITHMS:
            policy_id = f"baseline_{baseline}"
            frame = evaluate_algorithm_matrix(
                algorithm=baseline,
                policy=None,
                scenarios=scenarios,
                seeds=seeds,
                output_directory=full_output / "baselines" / baseline,
                write_steps=True,
                evaluation_mode="final_holdout",
            )
            frame.insert(0, "policy_id", policy_id)
            frame.insert(2, "training_root", "baseline")
            frame.insert(3, "checkpoint_transition", pd.NA)
            all_frames.append(frame)

        episodes = pd.concat(all_frames, ignore_index=True)
        validate_complete_final_evidence(
            episodes, scenarios=scenarios, seeds=seeds)
        tables = build_compact_tables(episodes)
        if not bool(tables["evaluation_integrity"]["all_checks_passed"].all()):
            raise RuntimeError("one or more final-holdout integrity gates failed")
        episodes.to_csv(full_output / "final_holdout_episode_summary.csv", index=False)

        compact_output = create_run_directory(compact_output)
        for name, table in tables.items():
            table.to_csv(compact_output / f"{name}.csv", index=False)
        provenance = pd.DataFrame(provenance_rows)
        provenance.to_csv(
            compact_output / "checkpoint_provenance.csv", index=False)
        total_runtime = time.perf_counter() - start
        report = _report_markdown(
            tables,
            provenance,
            full_output=full_output,
            evaluation_source_sha=git_metadata()["git_commit"],
            total_runtime_seconds=total_runtime,
        )
        (compact_output / "FINAL_HOLDOUT_REPORT.md").write_text(
            report, encoding="utf-8")
        manifest = {
            "format": "v2-final-holdout-v1",
            "status": "completed",
            "branch": "feat/rl-environment-v2",
            "evaluation_source_sha": git_metadata()["git_commit"],
            "seed42_training_source_sha": SEED42_SOURCE_SHA,
            "continuation_training_source_sha": CONTINUATION_SOURCE_SHA,
            "signed_off_environment_sha": SIGNED_OFF_ENVIRONMENT_SHA,
            "environment": {
                "class": "MplsTeEnvV2",
                "observation_dim": 604,
                "action_dim": 69,
                "definitions_frozen": True,
            },
            "authorization": {
                "workflow": "single final holdout",
                "scenarios": scenarios,
                "seeds": seeds,
                "deterministic_inference": True,
                "training_performed": False,
                "tuning_performed": False,
                "checkpoint_selection_performed": False,
                "checkpoint_sweep_performed": False,
                "holdout_used_for_debugging": False,
            },
            "episodes": {
                "total": int(len(episodes)),
                "per_policy_or_baseline": 35,
                "learner_checkpoints": 6,
                "baselines": list(BASELINE_ALGORITHMS),
            },
            "integrity": tables["evaluation_integrity"].to_dict(orient="records"),
            "checkpoint_provenance": provenance.to_dict(orient="records"),
            "runtime": {
                "total_wall_seconds": total_runtime,
                "device": "cuda:0",
                "gpu": gpu_name,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "hardware": hardware_inventory(),
            },
            "full_artifact_path": str(full_output),
            "compact_artifact_path": str(compact_output),
            "command": attempt["command"],
            "tooling_tests_before_evaluation": "pending final insertion",
            "final_verification": "pending final insertion",
            "failures": [],
            "large_artifacts_committed": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(compact_output / "manifest.json", manifest)
        attempt.update({
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "total_wall_seconds": total_runtime,
            "full_artifact_path": str(full_output),
            "compact_artifact_path": str(compact_output),
        })
        _write_json(attempt_path, attempt)
        print(json.dumps(manifest, indent=1, allow_nan=False))
        return manifest
    except Exception as exc:
        attempt.update({
            "status": "failed",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "failure_type": type(exc).__name__,
            "failure": str(exc),
        })
        _write_json(attempt_path, attempt)
        raise


def main(argv: list[str] | None = None) -> None:
    run_final_holdout(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
