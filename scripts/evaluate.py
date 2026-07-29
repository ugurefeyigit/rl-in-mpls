"""Paired multi-seed evaluation of RL vs baselines.

Every (scenario, seed) pair produces byte-identical offered traffic and
scripted events for all algorithms, so differences are attributable to
routing decisions alone.

Usage:
    python scripts/evaluate.py                                   # default suite
    python scripts/evaluate.py --scenarios evening_peak --seeds 101 102
    python scripts/evaluate.py --model ppo_te --algorithms rl static greedy cspf

Outputs (results/):
    eval_summary.csv     one row per (scenario, algorithm, seed)
    eval_stats.csv       mean/std/95% CI per (scenario, algorithm)
    eval_steps_<scenario>_seed<N>_<algo>.csv   per-step records for figures
    eval_summary.json    machine-readable copy incl. paired deltas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats as sps

from mplssim.experiments.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DEFAULT_SCENARIOS = ["full_day", "evening_peak", "flash_crowd", "link_failure",
                     "deceptive_local_optimum", "ood_double_failure", "overload_stress"]
DEFAULT_SEEDS = [101, 102, 103, 104, 105]
DEFAULT_ALGOS = ["static", "greedy", "cspf", "random", "rl"]


def ci95(x: np.ndarray) -> float:
    """Half-width of the 95% t-interval.

    Returns 0.0 for fewer than two samples, where the interval is undefined
    rather than zero-width; the companion ``<metric>_n`` column records the
    sample size so a 0.0 here can be told apart from a genuinely tight
    interval.
    """
    if len(x) < 2:
        return 0.0
    return float(sps.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_te")
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--algorithms", nargs="*", default=DEFAULT_ALGOS)
    ap.add_argument("--no-safety", action="store_true",
                    help="disable the RL safety filter (experimental mode)")
    ap.add_argument("--keep-steps-seed", type=int, default=None,
                    help="seed whose per-step records are saved (default: first)")
    ap.add_argument("--prefix", default="eval",
                    help="output filename prefix (default 'eval')")
    ap.add_argument("--env-version", choices=["v1", "v2"], default="v1",
                    help="environment version (default v1). v2 emits the V2 "
                         "environment identity record; V2 controller evaluation "
                         "belongs to the separate training task.")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.env_version == "v2":
        # Version selection is explicit and never inferred. This path only
        # validates and publishes the V2 identity: the paired suite below is
        # V1-semantic (V1 reward, V1 candidate paths, V1 event timing), so
        # running it under a V2 label would silently mix problem definitions.
        from mplssim.experiments.v2_factory import (
            build_environment_metadata, validate_environment_metadata)
        meta = build_environment_metadata()
        validate_environment_metadata(meta)
        out_dir = RESULTS / "environment_v2_validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "environment_v2.json").write_text(
            json.dumps(meta, indent=1), encoding="utf-8")
        print(f"wrote {out_dir / 'environment_v2.json'}")
        print("V2 environment metadata validated. This script's paired suite is "
              "V1-semantic; run 'python scripts/validate_env_v2.py --all' for the "
              "V2 pre-training validation.")
        return
    model = None
    if "rl" in args.algorithms:
        from server.session import load_model
        model = load_model(args.model)
        print(f"loaded model {args.model}")

    # A repeated seed would put duplicate (scenario, seed) keys in the summary
    # frame, which makes the paired-delta .loc lookups expand instead of
    # aligning one-to-one. Reject it rather than silently produce wrong deltas.
    if len(set(args.seeds)) != len(args.seeds):
        dupes = sorted({s for s in args.seeds if args.seeds.count(s) > 1})
        ap.error(f"--seeds contains duplicates {dupes}; paired deltas require "
                 f"one row per (scenario, seed)")

    keep_seed = args.keep_steps_seed if args.keep_steps_seed is not None else args.seeds[0]
    summaries: list[dict] = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            for algo in args.algorithms:
                df, summary = run_episode(
                    algo, scenario, seed, model=model,
                    safety_filter=not args.no_safety,
                )
                summaries.append(summary)
                print(f"  {scenario:26s} seed {seed} {algo:7s} "
                      f"reward={summary['reward_sum']:9.1f} "
                      f"maxU={summary['max_util_mean']:.3f} "
                      f"sla={summary['sla_violations_total']:5d} "
                      f"rr={summary['reroutes_total']:4d}")
                if seed == keep_seed:
                    df.insert(0, "algorithm", algo)
                    path = RESULTS / f"{args.prefix}_steps_{scenario}_seed{seed}_{algo}.csv"
                    df.to_csv(path, index=False)

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(RESULTS / f"{args.prefix}_summary.csv", index=False)

    # per-(scenario, algorithm) statistics across seeds
    metrics = ["reward_sum", "max_util_mean", "max_util_peak", "mean_delay_ms",
               "p95_delay_ms", "loss_ratio_mean", "delivered_ratio_mean",
               "sla_violations_total", "reroutes_total", "flaps_total",
               "time_above_90pct", "recovery_steps", "mean_decision_time_ms"]
    rows = []
    for (scen, algo), g in sdf.groupby(["scenario", "algorithm"]):
        row = {"scenario": scen, "algorithm": algo, "n_seeds": len(g)}
        for mkey in metrics:
            if mkey not in g.columns:
                continue
            vals = g[mkey].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            row[f"{mkey}_mean"] = float(np.mean(vals))
            row[f"{mkey}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{mkey}_ci95"] = ci95(vals)
            # Per-metric sample count. `n_seeds` counts the rows in the group,
            # but a metric that is None for some seeds (recovery_steps is None
            # whenever a scenario has no failure) is averaged over fewer.
            # Recording the real n stops a mean being read as if it used every
            # seed, and marks where ci95 returned 0.0 for want of a sample.
            row[f"{mkey}_n"] = int(len(vals))
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(RESULTS / f"{args.prefix}_stats.csv", index=False)

    # paired RL-vs-baseline deltas (same scenario+seed)
    paired = {}
    if "rl" in args.algorithms:
        rl = sdf[sdf.algorithm == "rl"].set_index(["scenario", "seed"])
        for base in [a for a in args.algorithms if a != "rl"]:
            b = sdf[sdf.algorithm == base].set_index(["scenario", "seed"])
            common = rl.index.intersection(b.index)
            deltas = {}
            for mkey in ("reward_sum", "max_util_mean", "sla_violations_total",
                         "mean_delay_ms", "loss_ratio_mean"):
                d = (rl.loc[common, mkey] - b.loc[common, mkey]).to_numpy(dtype=float)
                deltas[mkey] = {"mean": float(np.mean(d)), "ci95": ci95(d)}
            paired[f"rl_minus_{base}"] = deltas

    (RESULTS / f"{args.prefix}_summary.json").write_text(json.dumps({
        "config": vars(args), "stats": rows, "paired_deltas": paired,
    }, indent=1), encoding="utf-8")

    print("\n==== mean over seeds ====")
    view = stats_df[stats_df.columns[stats_df.columns.str.contains(
        "scenario|algorithm|reward_sum_mean|max_util_mean_mean|sla_violations_total_mean|reroutes_total_mean")]]
    print(view.to_string(index=False))
    # name the files actually written (the prefix is configurable)
    print(f"\nwrote {RESULTS / f'{args.prefix}_summary.csv'}, "
          f"{args.prefix}_stats.csv, {args.prefix}_summary.json")
    stale = sorted(p.name for p in RESULTS.glob(f"{args.prefix}_steps_*.csv")
                   if not any(f"_{scen}_seed{keep_seed}_" in p.name
                              for scen in args.scenarios))
    if stale:
        print(f"note: {len(stale)} older {args.prefix}_steps_*.csv file(s) in "
              f"results/ were NOT part of this run and were left untouched;\n"
              f"      figures built from the prefix will still pick them up: "
              f"{', '.join(stale[:4])}{' ...' if len(stale) > 4 else ''}")


if __name__ == "__main__":
    main()

