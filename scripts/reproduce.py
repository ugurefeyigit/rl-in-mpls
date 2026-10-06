"""One-command reproduction of the post-V2 study.

    python scripts/reproduce.py --suite sanity   # ~10 min on 4 CPU cores
    python scripts/reproduce.py --suite core     # ~3 h: one root of the main comparison + references
    python scripts/reproduce.py --suite full     # ~20 h on 4 cores: every registered run, all
                                                 # references, diagnostics, tables, figures, paper
    python scripts/reproduce.py --suite analysis # rebuild tables/figures/paper from existing results

Every step is an ordinary script that can be run on its own; this file only
sequences them. Steps are idempotent (completed runs and cached evaluations
are skipped), so an interrupted suite can simply be restarted. Runtimes are
measured on the machine in docs/EXPERIMENT_LOG.md (4-core Xeon 2.1 GHz, no GPU).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)}


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def sanity() -> None:
    # 1. fast tests (everything except the slow trainer-equivalence test)
    run([PY, "-m", "pytest", "-q", "-m", "not slow", "tests/test_study.py",
         "tests/test_env_v2.py", "tests/test_reward_v2.py", "tests/test_transition_v2.py"])
    # 2. deterministic baselines reproduce the closed study's holdout table exactly
    run([PY, "scripts/study/check_baseline_reproduction.py"])
    # 3. both learners train end-to-end on a tiny budget and evaluate
    with tempfile.TemporaryDirectory() as tmp:
        code = (
            "from pathlib import Path\n"
            "from mplssim.experiments.v2_factory import make_env_v2\n"
            "from mplssim.study.registry import base_config\n"
            "from mplssim.study.train import train_run\n"
            "from mplssim.study.evalrun import load_any_policy\n"
            "from mplssim.study.episode import LearnerPolicy, run_episode\n"
            "for alg in ('masked_bandit', 'maskable_ppo'):\n"
            f"    d = Path({tmp!r}) / alg\n"
            "    m = train_run(run_dir=d, algorithm=alg, root=42, config=base_config(alg),\n"
            "                  env_factory=make_env_v2, env_spec={'variant': 'base'},\n"
            "                  transitions=8192, checkpoint_interval=8192)\n"
            "    ck = sorted((d / 'checkpoints').iterdir())[-1]\n"
            "    s, _ = run_episode(LearnerPolicy(load_any_policy(ck, alg), alg), 'link_failure', 2001)\n"
            "    print(alg, m['status'], round(m['transitions_per_second'], 1), 'tr/s',\n"
            "          'return', round(s['operational_return'], 2))\n")
        run([PY, "-c", code])
    print("\nSANITY SUITE PASSED")


def references(seedset: str, policies: list[str], seeds: list[str] | None = None) -> None:
    cmd = [PY, "scripts/study/run_oracles.py", "--policies", *policies, "--seedset", seedset,
           "--out", "experiments/raw/references"]
    if seeds:
        cmd += ["--seeds", *seeds]
    run(cmd)


def analysis() -> None:
    run([PY, "scripts/study/analyze.py"])
    run([PY, "scripts/make_paper_figures.py"])
    if (ROOT / "paper" / "main.tex").exists():
        try:
            run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=ROOT / "paper")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"paper not compiled ({exc}); install TeX Live + latexmk")


def core() -> None:
    for rid in ("E1_main__bandit__r161803", "E1_main__ppo__r161803"):
        run([PY, "scripts/study/job.py", rid])
    references("test", ["static", "greedy", "cspf", "noop", "random_valid", "milp_track",
                        "oracle_h1"])
    run([PY, "scripts/study/run_seqdiag.py", "--seeds", "4001",
         "--out", "experiments/raw/seqdiag_greedy"])
    analysis()


def full() -> None:
    run([PY, "scripts/study/check_baseline_reproduction.py"])
    run([PY, "scripts/study/scheduler.py", "--slots", "4", "--ids", "E1_main", "E2_horizon",
         "E3_ppo_tuning", "E3b_ppo_best", "E4_delay", "E5_budget"])
    references("test", ["static", "greedy", "cspf", "noop", "random_valid", "milp_track",
                        "oracle_h1"])
    references("test", ["oracle_h3", "oracle_h6"], seeds=[str(s) for s in range(3001, 3006)])
    for frozen in ([], ["--frozen"]):
        out = "experiments/raw/seqdiag_greedy" + ("_frozen" if frozen else "")
        run([PY, "scripts/study/run_seqdiag.py", "--out", out, *frozen])
    run([PY, "scripts/study/run_mask_audit.py"])
    print("E0 (governed CPU reproduction of the closed study) is run separately; see "
          "docs/REPRODUCIBILITY.md section 4.")
    analysis()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--suite", choices=("sanity", "core", "full", "analysis"), required=True)
    {"sanity": sanity, "core": core, "full": full, "analysis": analysis}[p.parse_args().suite]()


if __name__ == "__main__":
    main()
