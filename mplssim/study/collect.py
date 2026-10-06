"""Collect raw evaluation outputs into tidy, version-controlled tables.

Inputs (large artifacts stay in ignored ``runs/`` / ``.worktrees/``):
* governed CPU reproductions: ``.worktrees/repro/runs/repro/seed<root>_<alg>/``
* study runs: ``runs/study/<family>/<run_id>/``
* reference policies: ``experiments/raw/references*/*.json``

Outputs:
* ``experiments/raw/learner_eval/<run_id>/``: every evaluation CSV, the
  selection records, the training manifest and per-episode training log of a
  run (small; checkpoints are not copied).
* ``experiments/processed/episodes_<seedset>.csv``: one row per
  (policy instance, scenario, seed) with all episode metrics.
* ``experiments/processed/validation_curves.csv``, ``runs.csv``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPRO = REPO / ".worktrees" / "repro" / "runs" / "repro"
STUDY = REPO / "runs" / "study"
RAW = REPO / "experiments" / "raw"
PROCESSED = REPO / "experiments" / "processed"

#: Display policy name for every (family, tag); E0/E1 learners pool into the
#: primary comparison because their trainers are bit-identical (tested).
POLICY_OF_TAG = {"bandit": "bandit", "ppo": "ppo"}


def _policy(family: str, tag: str) -> str:
    if family in ("E0_repro", "E1_main"):
        return POLICY_OF_TAG[tag]
    return f"{family.split('_')[0]}:{tag}"


def _copy_run(run_dir: Path, run_id: str) -> Path:
    dst = RAW / "learner_eval" / run_id
    dst.mkdir(parents=True, exist_ok=True)
    for f in (run_dir / "eval").glob("*"):
        if f.suffix in (".csv", ".json"):
            shutil.copy2(f, dst / f.name)
    for name in ("manifest.json", "episodes.jsonl", "training_episodes.jsonl",
                 "episode_seeds.json"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dst / name)
    return dst


def discover_runs() -> list[dict]:
    runs = []
    for d in sorted(REPRO.glob("seed*_*")):
        if not (d / "eval").exists():
            continue
        m = json.loads((d / "manifest.json").read_text())
        alg = m["algorithm"]
        tag = "bandit" if alg == "masked_bandit" else "ppo"
        root = int(m["run_config"]["root_seed"])
        runs.append(dict(run_id=f"E0_repro__{tag}__r{root}", family="E0_repro", tag=tag,
                         algorithm=alg, root=root, dir=d, manifest=m))
    for d in sorted(STUDY.glob("E*/*__*__r*")):
        if d.name.endswith(".interrupted") or not (d / "eval").exists():
            continue
        fam, tag, r = d.name.split("__")
        m = json.loads((d / "manifest.json").read_text())
        runs.append(dict(run_id=d.name, family=fam, tag=tag, algorithm=m["algorithm"],
                         root=int(r[1:]), dir=d, manifest=m))
    return runs


def collect(copy: bool = True) -> dict[str, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frames: dict[str, list[pd.DataFrame]] = {}
    curves, meta = [], []
    for r in discover_runs():
        src = _copy_run(r["dir"], r["run_id"]) if copy else r["dir"] / "eval"
        pol = _policy(r["family"], r["tag"])
        for seedset in ("test", "historical_holdout"):
            f = src / f"{seedset}__selected_and_final.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["policy"], df["family"], df["tag"], df["root"] = pol, r["family"], r["tag"], r["root"]
                df["kind"] = "learner"
                frames.setdefault(seedset, []).append(df)
        for sel in ("validation", "historical_continuity"):
            for f in sorted(src.glob(f"{sel}__ckpt*.csv")):
                df = pd.read_csv(f)
                curves.append({"run_id": r["run_id"], "policy": pol, "family": r["family"],
                               "tag": r["tag"], "root": r["root"], "seedset": sel,
                               "checkpoint": int(df["checkpoint"].iloc[0]),
                               "mean_return": float(df["operational_return"].mean()),
                               "episodes": len(df)})
        m = r["manifest"]
        meta.append({"run_id": r["run_id"], "policy": pol, "family": r["family"], "tag": r["tag"],
                     "algorithm": r["algorithm"], "root": r["root"],
                     "wall_time_seconds": m.get("wall_time_seconds"),
                     "transitions_per_second": m.get("transitions_per_second",
                                                     m.get("aggregate_transitions_per_second")),
                     "transitions": m.get("transitions", m.get("aggregate_transitions")),
                     "git_commit": m.get("git_commit", m.get("environment_record", {})
                                         .get("source", {}).get("commit")),
                     "parameters": (m.get("diagnostics") or {}).get("parameters"),
                     "config": json.dumps(m.get("config") or m.get("run_config", {}).get("ppo")
                                          or m.get("run_config", {}).get("masked_bandit"))})
    for refdir, seedset in ((RAW / "references", "test"),
                            (RAW / "references_historical_holdout", "historical_holdout")):
        rows = []
        for f in sorted(refdir.glob("*__seed*.json")):
            s = json.loads(f.read_text())
            comps = s.pop("reward_components")
            s.pop("env", None)
            s.update({f"rc_{k}": v for k, v in comps.items()})
            rows.append(s)
        if rows:
            df = pd.DataFrame(rows)
            df["policy"] = df["algorithm"]
            df["family"], df["tag"], df["root"], df["role"] = "reference", df["algorithm"], -1, "na"
            df["kind"] = ["oracle" if a.startswith("oracle") else "baseline" for a in df["algorithm"]]
            frames.setdefault(seedset, []).append(df)
    out = {}
    for seedset, fs in frames.items():
        out[f"episodes_{seedset}"] = pd.concat(fs, ignore_index=True)
    out["validation_curves"] = pd.DataFrame(curves)
    out["runs"] = pd.DataFrame(meta)
    for name, df in out.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
    return out
