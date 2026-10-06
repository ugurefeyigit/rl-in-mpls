"""Train (if needed), select on validation seeds and evaluate on test seeds, for one
registry run.

    python scripts/study/job.py E2_horizon__qg09__r42
    python scripts/study/job.py --list                # every registered run id
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mplssim.study.evalrun import select_and_test  # noqa: E402
from mplssim.study.registry import all_runs  # noqa: E402
from mplssim.study.train import train_run  # noqa: E402
from mplssim.study.variants import make_variant_factory  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_id", nargs="?")
    p.add_argument("--list", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()
    runs = all_runs()
    if a.list:
        print("\n".join(runs))
        return
    spec = runs[a.run_id]
    factory = make_variant_factory(spec.env)
    manifest_path = spec.run_dir / "manifest.json"
    done = manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "completed"
    if not done and not a.eval_only:
        if spec.run_dir.exists():  # an interrupted attempt: keep it, start a fresh one
            spec.run_dir.rename(spec.run_dir.with_name(spec.run_dir.name + ".interrupted"))
        train_run(run_dir=spec.run_dir, algorithm=spec.algorithm, root=spec.root,
                  config=spec.config, env_factory=factory, env_spec=spec.env,
                  transitions=spec.transitions, checkpoint_interval=spec.checkpoint_interval,
                  n_envs=spec.n_envs, scenario=spec.scenario, run_id=spec.run_id)
    result = select_and_test(spec.run_dir, spec.algorithm, spec.run_id, factory)
    (spec.run_dir / "eval" / "selection.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
