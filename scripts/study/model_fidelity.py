"""How well does a trained policy approximate the exact myopic optimum?

For each named run's selected checkpoint, follow the policy on test episodes
and, every ``--every`` steps, compute the exact (clairvoyant one-interval)
reward of every legal action by cloning the simulator. Reports per state:

* the reward of the policy's action, of no-op, and of the best action;
* whether the policy's action is the exact myopic optimum (top-1 agreement);
* for value learners: predicted vs. exact immediate reward of all legal actions
  (pooled R^2 after removing the per-state mean, and per-state Spearman rank).

    python scripts/study/model_fidelity.py --runs E0_repro__bandit__r42 E0_repro__ppo__r42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from case_study import run_dir  # noqa: E402
from mplssim.study.episode import make_eval_env  # noqa: E402
from mplssim.study.evalrun import checkpoints, load_any_policy  # noqa: E402
from mplssim.study.oracles import simulate  # noqa: E402
from mplssim.study.protocol import EVAL_SCENARIOS  # noqa: E402


def analyse(run_id: str, seeds: list[int], every: int) -> pd.DataFrame:
    d = run_dir(run_id)
    sel = [s for s in sorted((d / "eval").glob("selection*.json")) if "historical" not in s.name]
    step = json.loads(sel[-1].read_text())["selected_checkpoint"]
    path = dict(checkpoints(d))[step]
    alg = "maskable_ppo" if path.suffix == ".zip" else "masked_bandit"
    learner = load_any_policy(path, alg)
    rows = []
    for scenario in EVAL_SCENARIOS:
        for seed in seeds:
            env = make_eval_env(scenario, seed)
            raw = env.unwrapped
            obs, _ = env.reset(options={"episode_seed": seed})
            t, trunc = 0, False
            while not trunc:
                mask = env.action_masks()
                a = int(np.asarray(learner.predict(obs[None], mask[None], deterministic=True))[0])
                if t % every == 0:
                    legal = np.flatnonzero(mask)
                    true = np.array([simulate(raw, [int(x)], horizon=1)[0] for x in legal])
                    row = {"run_id": run_id, "scenario": scenario, "seed": seed, "step": t,
                           "n_legal": len(legal), "r_policy": float(true[legal == a][0]),
                           "r_noop": float(true[legal == 0][0]), "r_best": float(true.max()),
                           "policy_is_noop": a == 0,
                           "best_is_noop": bool(true[legal == 0][0] >= true.max() - 1e-12),
                           "agree_top1": bool(true[legal == a][0] >= true.max() - 1e-12)}
                    if alg == "masked_bandit":
                        with torch.no_grad():
                            pred = learner.network(torch.as_tensor(obs[None])).numpy()[0][legal]
                        row["pred_centered"] = json.dumps((pred - pred.mean()).round(5).tolist())
                        row["true_centered"] = json.dumps((true - true.mean()).round(5).tolist())
                        row["spearman"] = float(spearmanr(pred, true).statistic) \
                            if np.std(true) > 0 and np.std(pred) > 0 else np.nan
                        row["pred_policy_minus_noop"] = float(pred[legal == a][0] - pred[legal == 0][0])
                    rows.append(row)
                obs, _, _, trunc, _ = env.step(a)
                t += 1
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003])
    p.add_argument("--every", type=int, default=3)
    a = p.parse_args()
    out = ROOT / "experiments" / "raw" / "model_fidelity"
    out.mkdir(parents=True, exist_ok=True)
    for rid in a.runs:
        f = out / f"{rid}.csv"
        if f.exists():
            continue
        df = analyse(rid, a.seeds, a.every)
        df.to_csv(f, index=False)
        reg = (df.r_best - df.r_policy)
        print(rid, f"states={len(df)} top1={df.agree_top1.mean():.3f} "
              f"regret1={reg.mean():.4f} noop_share={df.policy_is_noop.mean():.3f} "
              f"best_noop_share={df.best_is_noop.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
