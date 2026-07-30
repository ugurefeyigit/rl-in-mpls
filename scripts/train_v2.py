"""Train one governed V2 learner with shared masks, seeds, metrics, and checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplssim.experiments.learning_common import load_learning_config
from mplssim.experiments.trainers_v2 import ALGORITHMS, train_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--purpose", choices=("meaningful", "smoke"), default="meaningful")
    parser.add_argument("--root-seed", type=int, default=42)
    parser.add_argument("--scenario", default="random_day")
    parser.add_argument("--transitions", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_learning_config()
    training = cfg["training"]
    manifest = train_experiment(
        algorithm=args.algorithm,
        run_directory=args.run_dir,
        root_seed=args.root_seed,
        scenario=args.scenario,
        aggregate_transitions=(
            args.transitions if args.transitions is not None
            else int(training["aggregate_transitions"])),
        n_envs=args.n_envs,
        checkpoint_interval=(
            args.checkpoint_interval if args.checkpoint_interval is not None
            else int(training["checkpoint_interval"])),
        requested_device=args.device,
        learning_config=cfg,
        purpose=args.purpose,
        command=[
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv if argv is not None else sys.argv[1:]),
        ],
    )
    print(json.dumps({
        "status": manifest["status"],
        "algorithm": manifest["algorithm"],
        "aggregate_transitions": manifest["aggregate_transitions"],
        "wall_time_seconds": manifest["wall_time_seconds"],
        "aggregate_transitions_per_second":
            manifest["aggregate_transitions_per_second"],
    }, indent=1))


if __name__ == "__main__":
    main()
