"""Load-aware job queue: keep at most N heavy study/training processes running.

    python scripts/study/scheduler.py --slots 4 --ids E1_main E2_horizon ...

Arguments may be run ids or family ids (expanded in registry order). Jobs whose
evaluation summary already exists are skipped, so the queue is restartable.
Each job's stdout/stderr goes to runs/study/logs/<run_id>.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mplssim.study.registry import RUNS_DIR, all_runs  # noqa: E402

HEAVY = ("train_v2.py", "scripts/study/job.py", "run_seqdiag.py", "run_oracles.py")


def heavy_count() -> int:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines()
               if line.startswith("python") and any(h in line for h in HEAVY))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--ids", nargs="+", required=True)
    a = p.parse_args()
    runs = all_runs()
    order: list[str] = []
    for i in a.ids:
        order += [r for r in runs if r == i or runs[r].family == i]
    logs = RUNS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for rid in order:
        if (runs[rid].run_dir / "eval" / "selection.json").exists():
            continue
        while heavy_count() >= a.slots:
            time.sleep(20)
        with (logs / f"{rid}.log").open("a") as fh:
            subprocess.Popen([sys.executable, str(ROOT / "scripts/study/job.py"), rid],
                             cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        print(time.strftime("%H:%M:%S"), "started", rid, flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
