"""Select V2 learner checkpoints and compare them with existing baselines."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplssim.experiments.evaluation_v2 import (
    BASELINE_ALGORITHMS,
    evaluate_algorithm_matrix,
    load_policy_checkpoint,
    select_checkpoint,
)
from mplssim.experiments.learning_common import (
    create_run_directory,
    load_learning_config,
    validate_training_root,
)


_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.(?:zip|pt)$")
_GOVERNED_CHECKPOINT_TRANSITIONS = set(range(50_000, 400_001, 50_000))


def checkpoint_transition_from_path(path: Path) -> int:
    match = _CHECKPOINT_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"not a governed checkpoint filename: {path}")
    return int(match.group(1))


def validate_checkpoint_schedule(checkpoints: list[Path]) -> None:
    """Require every preregistered periodic checkpoint, with no omissions."""
    transitions = [
        checkpoint_transition_from_path(checkpoint)
        for checkpoint in checkpoints
    ]
    if (
        len(transitions) != len(set(transitions))
        or set(transitions) != _GOVERNED_CHECKPOINT_TRANSITIONS
    ):
        raise ValueError(
            "meaningful comparison requires the exact eight-checkpoint "
            "schedule from 50000 through 400000 transitions")


def validate_meaningful_checkpoint_metadata(
    checkpoint: Path,
    metadata: dict[str, Any],
    algorithm: str,
) -> None:
    """Bind checkpoint selection to this task's preregistered run contract."""
    transition = checkpoint_transition_from_path(checkpoint)
    run_config = metadata.get("run_config", {})
    if metadata.get("algorithm") != algorithm:
        raise ValueError("checkpoint metadata algorithm mismatch")
    if run_config.get("algorithm") != algorithm:
        raise ValueError("checkpoint run_config algorithm mismatch")
    if run_config.get("environment_version") != "v2":
        raise ValueError("checkpoint is not from V2")
    validate_training_root(int(run_config.get("root_seed", -1)))
    if int(run_config.get("aggregate_transitions", -1)) != 400_000:
        raise ValueError(
            "checkpoint must come from a 400000-transition training run")
    if int(run_config.get("checkpoint_interval", -1)) != 50_000:
        raise ValueError("checkpoint interval must be 50000 transitions")
    if run_config.get("purpose") != "meaningful":
        raise ValueError("checkpoint run purpose must be meaningful")
    if int(metadata.get("aggregate_transitions", -1)) != transition:
        raise ValueError("checkpoint filename transition disagrees with metadata")
    if transition not in range(50_000, 400_001, 50_000):
        raise ValueError("checkpoint transition is outside the governed schedule")


def checkpoint_summaries_are_valid(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return all(
        bool(row["reward_component_sum_exact"])
        and int(row["invalid_action_attempts"]) == 0
        and int(row["mask_disagreements"]) == 0
        and int(row["solver_convergence_failures"]) == 0
        and int(row["protected_safety_failures"]) == 0
        and bool(row["truncated"])
        and not bool(row["terminated"])
        for row in rows
    )


def aggregate_comparison(episodes: pd.DataFrame) -> pd.DataFrame:
    """Compact all-scenario/seed means using actual V2 summary field names."""
    rows = []
    for algorithm, group in episodes.groupby("algorithm", sort=False):
        rows.append({
            "algorithm": algorithm,
            "episodes": int(len(group)),
            "operational_return_mean": float(group["operational_return"].mean()),
            "operational_return_std": float(
                group["operational_return"].std(ddof=1))
                if len(group) > 1 else 0.0,
            "sla_violations_demand_intervals_mean": float(
                group["sla_violations_demand_intervals"].mean()),
            "reroutes_per_hour_mean": float(group["reroutes_per_hour"].mean()),
            "max_utilization_mean": float(
                group["max_utilization_mean"].mean()),
            "delivered_ratio_mean": float(group["delivered_ratio_mean"].mean()),
            "te_reversals_mean": float(group["te_reversals"].mean()),
            "protected_disconnection_demand_intervals_mean": float(
                group["protected_disconnection_demand_intervals"].mean()),
            "unprotected_disconnection_demand_intervals_mean": float(
                group["unprotected_disconnection_demand_intervals"].mean()),
            "invalid_action_attempts_total": int(
                group["invalid_action_attempts"].sum()),
            "mask_disagreements_total": int(
                group["mask_disagreements"].sum()),
            "solver_convergence_failures_total": int(
                group["solver_convergence_failures"].sum()),
            "wall_time_seconds_total": float(group["wall_time_seconds"].sum()),
        })
    return pd.DataFrame(rows)


def evaluate_checkpoint_sweep(
    *,
    algorithm: str,
    run_directory: Path,
    output_directory: Path,
    scenarios: list[str],
    seeds: list[int],
    requested_device: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    suffix = ".zip" if algorithm == "maskable_ppo" else ".pt"
    checkpoints = sorted(
        (Path(run_directory) / "checkpoints").glob(f"checkpoint_*{suffix}"),
        key=checkpoint_transition_from_path,
    )
    validate_checkpoint_schedule(checkpoints)
    selection_rows: list[dict[str, Any]] = []
    all_episodes: list[pd.DataFrame] = []
    for checkpoint in checkpoints:
        transition = checkpoint_transition_from_path(checkpoint)
        checkpoint_output = output_directory / f"checkpoint_{transition:09d}"
        try:
            policy, metadata = load_policy_checkpoint(
                checkpoint,
                algorithm=algorithm,
                requested_device=requested_device,
            )
            validate_meaningful_checkpoint_metadata(
                checkpoint, metadata, algorithm)
            episodes = evaluate_algorithm_matrix(
                algorithm=algorithm,
                policy=policy,
                scenarios=scenarios,
                seeds=seeds,
                output_directory=checkpoint_output,
                write_steps=False,
            )
            records = episodes.to_dict(orient="records")
            valid = checkpoint_summaries_are_valid(records)
            row = {
                "algorithm": algorithm,
                "checkpoint": str(checkpoint),
                "checkpoint_transition": transition,
                "mean_operational_return": float(
                    episodes["operational_return"].mean()),
                "valid": valid,
                "integrity_failure": None if valid else (
                    "one or more evaluation integrity checks failed"),
                "payload_sha256": metadata["payload_sha256"],
            }
            episodes.insert(0, "checkpoint_transition", transition)
            all_episodes.append(episodes)
        except Exception as exc:
            row = {
                "algorithm": algorithm,
                "checkpoint": str(checkpoint),
                "checkpoint_transition": transition,
                "mean_operational_return": float("-inf"),
                "valid": False,
                "integrity_failure": f"{type(exc).__name__}: {exc}",
                "payload_sha256": None,
            }
        selection_rows.append(row)
    selected = select_checkpoint(selection_rows)
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(output_directory / "checkpoint_selection.csv", index=False)
    safe_rows = [
        {
            **row,
            "mean_operational_return": (
                row["mean_operational_return"]
                if np.isfinite(row["mean_operational_return"]) else None),
        }
        for row in selection_rows
    ]
    (output_directory / "checkpoint_selection.json").write_text(
        json.dumps({
            "rows": safe_rows,
            "selected": selected,
            "rule": (
                "highest mean operational return over all seven scenarios and "
                "continuity seeds 101-105; exact ties choose earlier"),
        }, indent=1, allow_nan=False),
        encoding="utf-8",
    )
    episodes_frame = pd.concat(all_episodes, ignore_index=True)
    episodes_frame.to_csv(
        output_directory / "checkpoint_episode_summary.csv", index=False)
    return selected, selection_frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-run", type=Path, required=True)
    parser.add_argument("--bandit-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = create_run_directory(args.output_dir)
    cfg = load_learning_config()
    scenarios = list(cfg["evaluation"]["scenarios"])
    seeds = list(cfg["continuity_seeds"])
    selections: dict[str, dict[str, Any]] = {}
    curves = []
    for algorithm, run_dir in (
        ("maskable_ppo", args.ppo_run),
        ("masked_bandit", args.bandit_run),
    ):
        sweep_dir = output / "sweeps" / algorithm
        sweep_dir.mkdir(parents=True)
        selected, curve = evaluate_checkpoint_sweep(
            algorithm=algorithm,
            run_directory=run_dir,
            output_directory=sweep_dir,
            scenarios=scenarios,
            seeds=seeds,
            requested_device=args.device,
        )
        selections[algorithm] = selected
        curves.append(curve)

    comparison_frames: list[pd.DataFrame] = []
    for algorithm in ("maskable_ppo", "masked_bandit"):
        checkpoint = Path(selections[algorithm]["checkpoint"])
        policy, _ = load_policy_checkpoint(
            checkpoint, algorithm=algorithm, requested_device=args.device)
        comparison_frames.append(evaluate_algorithm_matrix(
            algorithm=algorithm,
            policy=policy,
            scenarios=scenarios,
            seeds=seeds,
            output_directory=output / "selected" / algorithm,
            write_steps=True,
        ))
    for baseline in BASELINE_ALGORITHMS:
        comparison_frames.append(evaluate_algorithm_matrix(
            algorithm=baseline,
            policy=None,
            scenarios=scenarios,
            seeds=seeds,
            output_directory=output / "baselines" / baseline,
            write_steps=True,
        ))

    episodes = pd.concat(comparison_frames, ignore_index=True)
    episodes.to_csv(output / "comparison_episode_summary.csv", index=False)
    compact = aggregate_comparison(episodes)
    compact.to_csv(output / "comparison.csv", index=False)
    (output / "comparison.json").write_text(
        json.dumps({
            "checkpoint_selection": selections,
            "aggregate": compact.to_dict(orient="records"),
            "episodes": episodes.to_dict(orient="records"),
        }, indent=1, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(curves, ignore_index=True).to_csv(
        output / "learning_curve.csv", index=False)
    (output / "comparison_manifest.json").write_text(
        json.dumps({
            "training_runs": {
                "maskable_ppo": str(args.ppo_run),
                "masked_bandit": str(args.bandit_run),
            },
            "selected_checkpoints": selections,
            "scenarios": scenarios,
            "seeds": seeds,
            "holdout_accessed": False,
        }, indent=1, allow_nan=False),
        encoding="utf-8",
    )
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
