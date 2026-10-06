"""Checkpoint evaluation, preregistered selection and test evaluation of a run.

Protocol (fixed before any new result was produced):

1. Every saved checkpoint is evaluated deterministically on the 7 evaluation
   scenarios x ``VALIDATION`` seeds.
2. The checkpoint with the highest mean validation return is selected (ties:
   the earlier checkpoint) -- the same rule as the closed study, applied to a
   seed set that is never reported.
3. The selected checkpoint *and* the final checkpoint are evaluated on the
   7 scenarios x ``TEST`` seeds. Both are reported; the final checkpoint is
   the selection-free robustness check.

Results are cached per (checkpoint, seed set) as CSV so evaluation can be
resumed and audited.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from mplssim.study.episode import LearnerPolicy, evaluate_matrix
from mplssim.study.protocol import EVAL_SCENARIOS, SEED_SETS

_CKPT = re.compile(r"checkpoint_(\d+)\.(zip|pt)$")


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in (run_dir / "checkpoints").iterdir():
        m = _CKPT.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_any_policy(path: Path, algorithm: str) -> Any:
    """Load study or governed (closed-study) checkpoints for inference on CPU."""
    torch.set_num_threads(1)
    if path.suffix == ".zip":
        from mplssim.experiments.trainers_v2 import MaskablePpoLearner
        return MaskablePpoLearner.load(path, device=torch.device("cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == "masked-contextual-bandit-v1":
        from mplssim.experiments.masked_bandit import MaskedContextualBandit
        return MaskedContextualBandit.load(path, device=torch.device("cpu"))
    from mplssim.study.learners import MaskedQLearner
    return MaskedQLearner.load_policy(path)


def eval_checkpoint(run_dir: Path, algorithm: str, step: int, path: Path, seedset: str,
                    env_factory: Callable[..., Any] | None, run_id: str,
                    scenarios: tuple[str, ...] = EVAL_SCENARIOS) -> pd.DataFrame:
    cache = run_dir / "eval" / f"{seedset}__ckpt{step:09d}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    policy = LearnerPolicy(load_any_policy(path, algorithm), name=algorithm)
    df = evaluate_matrix(policy, list(scenarios), list(SEED_SETS[seedset]), env_factory)
    df.insert(0, "checkpoint", step)
    df.insert(0, "run_id", run_id)
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def select_and_test(run_dir: Path, algorithm: str, run_id: str,
                    env_factory: Callable[..., Any] | None = None,
                    select_on: str = "validation", test_on: str = "test") -> dict[str, Any]:
    ckpts = checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    val = pd.concat([eval_checkpoint(run_dir, algorithm, s, p, select_on, env_factory, run_id)
                     for s, p in ckpts])
    curve = val.groupby("checkpoint")["operational_return"].mean()
    best = max(curve.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    final = ckpts[-1][0]
    paths = dict(ckpts)
    tests = []
    for role, step in (("selected", best), ("final", final)):
        df = eval_checkpoint(run_dir, algorithm, step, paths[step], test_on, env_factory, run_id)
        df = df.copy()
        df["role"] = role
        tests.append(df)
    test = pd.concat(tests)
    test.to_csv(run_dir / "eval" / f"{test_on}__selected_and_final.csv", index=False)
    curve.rename("mean_return").to_csv(run_dir / "eval" / f"{select_on}__curve.csv")
    return {"run_id": run_id, "selected_checkpoint": int(best), "final_checkpoint": int(final),
            "validation_curve": {int(k): float(v) for k, v in curve.items()},
            "test_mean_selected": float(test[test.role == "selected"].operational_return.mean()),
            "test_mean_final": float(test[test.role == "final"].operational_return.mean())}
