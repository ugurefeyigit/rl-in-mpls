"""Per-decision inference cost of every policy (CPU time, single thread).

Measures process CPU time (not wall time) per decision over the states of
two test episodes, so the numbers are robust to other jobs sharing the
machine. Writes experiments/raw/benchmark/inference.json.

    python scripts/study/benchmark_inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from case_study import run_dir  # noqa: E402
from mplssim.study.episode import LearnerPolicy, make_eval_env  # noqa: E402
from mplssim.study.evalrun import checkpoints, load_any_policy  # noqa: E402
from mplssim.study.provenance import run_record  # noqa: E402
from run_oracles import make_policy  # noqa: E402

torch.set_num_threads(1)


def states(scenario: str, seed: int):
    env = make_eval_env(scenario, seed)
    raw = env.unwrapped
    obs, _ = env.reset(options={"episode_seed": seed})
    out, trunc = [], False
    while not trunc:
        mask = env.action_masks()
        out.append((obs.copy(), mask.copy(), raw.eng.fast_clone()))
        obs, _, _, trunc, _ = env.step(0)
    return env, out


def bench(policy, env, sts, repeat: int = 1) -> dict:
    raw = env.unwrapped
    times = []
    for obs, mask, eng in sts:
        raw.eng = eng.fast_clone()
        t0 = time.process_time()
        for _ in range(repeat):
            policy.act(obs, mask, raw)
        times.append((time.process_time() - t0) / repeat)
    t = np.array(times) * 1000
    return {"mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)), "n": len(t)}


def main() -> None:
    env, sts = states("evening_peak", 3001)
    sts = sts[:60]
    res = {}
    for rid in ("E0_repro__bandit__r42", "E0_repro__ppo__r42"):
        d = run_dir(rid)
        path = checkpoints(d)[-1][1]
        alg = "maskable_ppo" if path.suffix == ".zip" else "masked_bandit"
        learner = load_any_policy(path, alg)
        net = learner.model.policy if alg == "maskable_ppo" else learner.network
        res[alg] = {**bench(LearnerPolicy(learner, alg), env, sts, repeat=20),
                    "parameters": int(sum(p.numel() for p in net.parameters()))}
    for name, rep in (("greedy", 20), ("cspf", 20), ("static", 20), ("milp_track", 1),
                      ("oracle_h1", 1)):
        res[name] = bench(make_policy(name, 3001), env, sts, repeat=rep)
        res[name]["parameters"] = 0
    out = ROOT / "experiments" / "raw" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference.json").write_text(json.dumps(run_record(
        kind="inference_benchmark", scenario="evening_peak", seed=3001, states=len(sts),
        timer="process_time", results=res), indent=1, default=str))
    for k, v in res.items():
        print(f"{k:14s} {v['mean_ms']:9.3f} ms  p95 {v['p95_ms']:9.3f} ms  params {v['parameters']}")


if __name__ == "__main__":
    main()
