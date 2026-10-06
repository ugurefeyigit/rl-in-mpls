"""Exact check: deterministic baselines reproduce the closed study's holdout table.

Runs static/greedy/cspf on the 7 scenarios x holdout seeds 1001-1005 (unless
already present) and compares every (policy, scenario) mean and SD of the
operational return, SLA violations and moved bandwidth with
results/v2_final_holdout/scenario_metrics.csv. Exit code 1 on any mismatch
> 1e-9. Writes experiments/processed/baseline_reproduction.csv.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/raw/references_historical_holdout"


def main() -> int:
    if len(list(OUT.glob("*__seed*.json"))) < 105:
        subprocess.run([sys.executable, str(ROOT / "scripts/study/run_oracles.py"),
                        "--policies", "static", "greedy", "cspf", "--seedset",
                        "historical_holdout", "--out", str(OUT)], check=True)
    sys.path.insert(0, str(ROOT / "scripts/study"))
    from run_oracles import load_dir
    mine = load_dir(OUT)
    m = mine.groupby(["algorithm", "scenario"]).agg(
        return_mean=("operational_return", "mean"), return_std=("operational_return", "std"),
        sla_mean=("sla_violations_demand_intervals", "mean"),
        moved_mean=("moved_mbps_total", "mean")).reset_index()
    h = pd.read_csv(ROOT / "results/v2_final_holdout/scenario_metrics.csv")
    h = h[h.algorithm.isin(["static", "greedy", "cspf"])].rename(columns={
        "operational_return_mean": "hist_return_mean", "operational_return_std": "hist_return_std",
        "sla_violations_demand_intervals_mean": "hist_sla_mean",
        "moved_mbps_total_mean": "hist_moved_mean"})
    j = m.merge(h[["algorithm", "scenario", "hist_return_mean", "hist_return_std",
                   "hist_sla_mean", "hist_moved_mean"]], on=["algorithm", "scenario"])
    for c in ("return_mean", "return_std", "sla_mean", "moved_mean"):
        j[f"abs_diff_{c}"] = (j[c] - j[f"hist_{c}"]).abs()
    out = ROOT / "experiments/processed/baseline_reproduction.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    j.to_csv(out, index=False)
    worst = j[[c for c in j.columns if c.startswith("abs_diff")]].to_numpy().max()
    ok = len(j) == 21 and worst <= 1e-9
    print(f"{len(j)} cells compared; worst absolute difference {worst:.3e}; "
          f"{'EXACT REPRODUCTION' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
