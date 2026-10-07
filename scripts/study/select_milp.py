"""Select MILP-track's configuration on VALIDATION seeds, then run it on TEST.

    python scripts/study/select_milp.py

Reads the validation grid in experiments/raw/references_validation/milp_grid
(written by run_oracles.py), picks the configuration with the highest mean
validation return (ties: the untuned default first, then the order below),
writes selection.json next to the grid, and runs the selected configuration
on the 140 test episodes into experiments/raw/references_milp_tuned/.
The test results play no role in the selection.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "study"))
from run_oracles import load_dir  # noqa: E402

GRID = ROOT / "experiments/raw/references_validation/milp_grid"
OUT = ROOT / "experiments/raw/references_milp_tuned"
CONFIGS = ["milp_g020_largest", "milp_g000_largest", "milp_g050_largest", "milp_g100_largest",
           "milp_g000_smallest", "milp_g020_smallest", "milp_g050_smallest", "milp_g100_smallest"]
EXPECTED = 35  # 7 scenarios x 5 validation seeds


def main() -> None:
    d = load_dir(GRID)
    counts = d.groupby("algorithm").size()
    missing = [c for c in CONFIGS if counts.get(c, 0) < EXPECTED]
    if missing:
        raise SystemExit(f"grid incomplete: {missing}")
    means = d.groupby("algorithm").operational_return.mean()
    best = max(CONFIGS, key=lambda c: (round(float(means[c]), 9), -CONFIGS.index(c)))
    sel = {"selected": best, "validation_mean": {c: float(means[c]) for c in CONFIGS},
           "rule": "highest mean validation return; ties to the untuned default, then list order"}
    (GRID / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))
    subprocess.run([sys.executable, str(ROOT / "scripts/study/run_oracles.py"), "--policies", best,
                    "--seedset", "test", "--out", str(OUT)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
