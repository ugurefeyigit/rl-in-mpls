"""Step-level trajectories of several policies on one test episode (failure analysis).

    python scripts/study/case_study.py --scenario deceptive_local_optimum --seed 3007 \
        --runs E0_repro__bandit__r42 E0_repro__ppo__r42 --refs greedy milp_track oracle_h1

Learner checkpoints are the *selected* checkpoints of the named runs (from
their selection records). Writes experiments/raw/case_studies/<scenario>_seed<seed>.csv
with one row per (policy, step): action, decoded demand/path, reward and its
components, max utilization, SLA violations, delivered ratio; plus, for
learners, the bandit's predicted immediate reward or PPO's action probability
of the chosen action and of no-op.
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

from mplssim.study.collect import REPRO, STUDY  # noqa: E402
from mplssim.study.episode import LearnerPolicy, run_episode  # noqa: E402
from mplssim.study.evalrun import checkpoints, load_any_policy  # noqa: E402
from run_oracles import make_policy  # noqa: E402


class Explained(LearnerPolicy):
    """Learner policy that records its own scores for the chosen action."""

    def __init__(self, learner, name):
        super().__init__(learner, name)
        self.trace = []

    def act(self, observation, mask, env):
        a = super().act(observation, mask, env)
        rec = {"chosen": a}
        if hasattr(self.learner, "network"):
            with torch.no_grad():
                q = self.learner.network(torch.as_tensor(observation[None, :])).numpy()[0]
            q[~mask] = -np.inf
            rec.update(score_chosen=float(q[a]), score_noop=float(q[0]),
                       score_margin=float(q[a] - np.max(np.delete(q, a))))
        else:
            pol = self.learner.model.policy
            obs_t, _ = pol.obs_to_tensor(observation[None, :])
            with torch.no_grad():
                p = pol.get_distribution(obs_t, action_masks=mask[None, :]).distribution.probs
            p = p.numpy()[0]
            rec.update(prob_chosen=float(p[a]), prob_noop=float(p[0]),
                       entropy=float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        self.trace.append(rec)
        return a


def run_dir(run_id: str) -> Path:
    fam, tag, r = run_id.split("__")
    if fam == "E0_repro":
        alg = "masked_bandit" if tag == "bandit" else "maskable_ppo"
        return REPRO / f"seed{r[1:]}_{alg}"
    return STUDY / fam / run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--refs", nargs="*", default=[])
    a = p.parse_args()
    frames = []
    for rid in a.runs:
        d = run_dir(rid)
        sel = sorted((d / "eval").glob("selection*.json"))
        sel = [s for s in sel if "historical" not in s.name] or sel
        step = json.loads(sel[-1].read_text())["selected_checkpoint"]
        path = dict(checkpoints(d))[step]
        alg = "maskable_ppo" if path.suffix == ".zip" else "masked_bandit"
        pol = Explained(load_any_policy(path, alg), rid)
        _, steps = run_episode(pol, a.scenario, a.seed, keep_steps=True)
        steps = pd.concat([steps, pd.DataFrame(pol.trace)], axis=1)
        steps["policy"] = rid
        frames.append(steps)
    for name in a.refs:
        _, steps = run_episode(make_policy(name, a.seed), a.scenario, a.seed, keep_steps=True)
        steps["policy"] = name
        frames.append(steps)
    out = ROOT / "experiments" / "raw" / "case_studies"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df["demand_idx"] = np.where(df.action > 0, (df.action - 1) // 4, -1)
    df["path_idx"] = np.where(df.action > 0, (df.action - 1) % 4, -1)
    df.to_csv(out / f"{a.scenario}_seed{a.seed}.csv", index=False)
    print(df.groupby("policy").reward.sum())


if __name__ == "__main__":
    main()
