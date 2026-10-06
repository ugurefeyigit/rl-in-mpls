"""Evaluate a reproduced governed V2 run under both protocols.

historical: select on HISTORICAL_CONTINUITY (101-105), test on HISTORICAL_HOLDOUT (1001-1005)
            -- exactly the closed study's procedure, for comparison with its tables.
study:      select on VALIDATION (2001-2005), test on TEST (3001-3020).

    python scripts/study/eval_repro.py .worktrees/repro/runs/repro/seed42_masked_bandit
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mplssim.study.evalrun import select_and_test  # noqa: E402

run = Path(sys.argv[1]).resolve()
manifest = json.loads((run / "manifest.json").read_text())
if manifest.get("status") != "completed":
    raise SystemExit(f"{run} is not a completed run")
alg = manifest["algorithm"]
root = manifest["run_config"]["root_seed"]
rid = f"E0_repro__{'bandit' if alg == 'masked_bandit' else 'ppo'}__r{root}"
for proto, sel, test in (("historical", "historical_continuity", "historical_holdout"),
                         ("study", "validation", "test")):
    res = select_and_test(run, alg, rid, None, select_on=sel, test_on=test)
    (run / "eval" / f"selection_{proto}.json").write_text(json.dumps(res, indent=1))
    print(proto, json.dumps(res))
