"""Bounded V2 throughput benchmark for vector counts and usable devices."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplssim.experiments.learning_common import (
    create_run_directory,
    load_learning_config,
)
from mplssim.experiments.trainers_v2 import ALGORITHMS, train_experiment


def select_benchmark_configuration(
    rows: list[dict],
    *,
    meaningful_budget: int = 400_000,
    checkpoint_interval: int = 50_000,
) -> dict:
    """Select best complete two-learner score among exact-budget counts."""
    counts = sorted({int(row["n_envs"]) for row in rows})
    eligible_counts = [
        count for count in counts
        if meaningful_budget % count == 0 and checkpoint_interval % count == 0
    ]
    candidates = []
    devices = sorted({str(row["device"]) for row in rows})
    for device in devices:
        for count in eligible_counts:
            matching = [
                row for row in rows
                if row["status"] == "completed"
                and int(row["n_envs"]) == count
                and str(row["device"]) == device
            ]
            by_algorithm = {row["algorithm"]: row for row in matching}
            if set(by_algorithm) != set(ALGORITHMS):
                continue
            rates = [
                float(by_algorithm[algorithm]["transitions_per_second"])
                for algorithm in ALGORITHMS
            ]
            if any(rate <= 0 or not math.isfinite(rate) for rate in rates):
                continue
            candidates.append({
                "n_envs": count,
                "device": device,
                "combined_transitions_per_second": sum(rates) / len(rates),
                "per_algorithm": {
                    algorithm: float(
                        by_algorithm[algorithm]["transitions_per_second"])
                    for algorithm in ALGORITHMS
                },
            })
    if not candidates:
        raise ValueError("no complete, stable, exact-budget benchmark candidate")
    best = min(
        candidates,
        key=lambda row: (-row["combined_transitions_per_second"], row["n_envs"]),
    )
    return {
        **best,
        "eligible_counts": eligible_counts,
        "selection_rule": (
            "highest arithmetic mean transitions/s across both learners among "
            "counts dividing 400000 and 50000 exactly"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-counts", nargs="*", type=int, default=[8, 12, 16])
    parser.add_argument("--vector-steps", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = create_run_directory(args.output_dir)
    cfg = load_learning_config()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows: list[dict] = []
    for device in devices:
        for n_envs in args.vector_counts:
            transitions = int(args.vector_steps) * int(n_envs)
            for algorithm in ALGORITHMS:
                run_dir = output / f"{algorithm}_{device}_{n_envs}env"
                try:
                    manifest = train_experiment(
                        algorithm=algorithm,
                        run_directory=run_dir,
                        root_seed=42,
                        scenario="random_day",
                        aggregate_transitions=transitions,
                        n_envs=n_envs,
                        checkpoint_interval=transitions,
                        requested_device=device,
                        learning_config=cfg,
                        purpose="benchmark",
                    )
                    rows.append({
                        "algorithm": algorithm,
                        "n_envs": n_envs,
                        "device": manifest["device"]["resolved"],
                        "aggregate_transitions": transitions,
                        "wall_time_seconds": manifest["wall_time_seconds"],
                        "transitions_per_second":
                            manifest["aggregate_transitions_per_second"],
                        "status": manifest["status"],
                    })
                except Exception as exc:
                    rows.append({
                        "algorithm": algorithm,
                        "n_envs": n_envs,
                        "device": device,
                        "aggregate_transitions": transitions,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "failure": str(exc),
                    })
    selection = select_benchmark_configuration(rows)
    (output / "benchmark.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    (output / "selection.json").write_text(
        json.dumps(selection, indent=1), encoding="utf-8")
    print(json.dumps(selection, indent=1))


if __name__ == "__main__":
    main()
