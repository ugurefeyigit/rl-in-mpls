"""Deterministic paired evaluation for a V2 learner checkpoint or baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mplssim.experiments.evaluation_v2 import (
    BASELINE_ALGORITHMS,
    LEARNER_ALGORITHMS,
    evaluate_algorithm_matrix,
    load_policy_checkpoint,
)
from mplssim.experiments.learning_common import load_learning_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", required=True,
        choices=BASELINE_ALGORITHMS + LEARNER_ALGORITHMS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-steps", action="store_true")
    args = parser.parse_args(argv)
    if args.algorithm in LEARNER_ALGORITHMS and args.checkpoint is None:
        parser.error("--checkpoint is required for learner evaluation")
    if args.algorithm in BASELINE_ALGORITHMS and args.checkpoint is not None:
        parser.error("--checkpoint is not valid for baseline evaluation")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_learning_config()
    policy = None
    if args.checkpoint is not None:
        policy, _ = load_policy_checkpoint(
            args.checkpoint,
            algorithm=args.algorithm,
            requested_device=args.device,
        )
    frame = evaluate_algorithm_matrix(
        algorithm=args.algorithm,
        policy=policy,
        scenarios=args.scenarios or list(cfg["evaluation"]["scenarios"]),
        seeds=args.seeds or list(cfg["continuity_seeds"]),
        output_directory=args.output_dir,
        write_steps=not args.no_steps,
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
