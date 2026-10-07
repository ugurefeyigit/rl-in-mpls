"""Machine-readable experiment registry (``experiments/registry/*.yaml``).

Each registry file defines one experiment family::

    id: E2_horizon
    description: ...
    env: {variant: base}                  # or {variant: delayed, delay_steps: 1}
    budget: {transitions: 400000, checkpoint_interval: 50000, n_envs: 16}
    roots: [42, 314159, 271828]
    runs:
      - {tag: qg09, algorithm: masked_q, overrides: {gamma: 0.9}}

A run's stable identifier is ``<family>__<tag>__r<root>``. Its fully resolved
configuration -- base learner defaults from the governed configuration files
plus the registry overrides -- is stored in the run manifest, so no setting is
implicit or scattered across scripts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mplssim.core.topology import CONFIG_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "experiments" / "registry"
RUNS_DIR = REPO_ROOT / "runs" / "study"


def base_config(algorithm: str) -> dict[str, Any]:
    """Governed defaults: learning_v2.yaml (bandit / Q) or training.yaml (PPO)."""
    if algorithm in ("masked_bandit", "masked_q"):
        cfg = yaml.safe_load((CONFIG_DIR / "experiments" / "learning_v2.yaml").read_text())
        out = copy.deepcopy(cfg["masked_bandit"])
        if algorithm == "masked_q":
            out.update(gamma=0.0, target_update_every=250)
        return out
    if algorithm == "maskable_ppo":
        return copy.deepcopy(yaml.safe_load((CONFIG_DIR / "training.yaml").read_text())["ppo"])
    raise KeyError(algorithm)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    family: str
    tag: str
    algorithm: str
    root: int
    config: dict[str, Any]
    env: dict[str, Any]
    transitions: int
    checkpoint_interval: int
    n_envs: int
    scenario: str = "random_day"
    eval_env: dict[str, Any] = field(default_factory=lambda: {"variant": "base"})
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.family / self.run_id


def load_family(path: Path | str) -> list[RunSpec]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REGISTRY_DIR / path
    raw = yaml.safe_load(path.read_text())
    fam = raw["id"]
    budget = raw.get("budget", {})
    out = []
    for run in raw["runs"]:
        roots = run.get("roots", raw["roots"])
        b = {**budget, **run.get("budget", {})}
        for root in roots:
            cfg = _merge(base_config(run["algorithm"]), run.get("overrides", {}))
            out.append(RunSpec(
                run_id=f"{fam}__{run['tag']}__r{root}", family=fam, tag=run["tag"],
                algorithm=run["algorithm"], root=int(root), config=cfg,
                env=dict(run.get("env", raw.get("env", {"variant": "base"}))),
                eval_env=dict(run.get("eval_env", raw.get("eval_env",
                                                          run.get("env", raw.get("env", {"variant": "base"}))))),
                transitions=int(b.get("transitions", 400_000)),
                checkpoint_interval=int(b.get("checkpoint_interval", 50_000)),
                n_envs=int(b.get("n_envs", 16)),
                extra={k: v for k, v in run.items()
                       if k not in ("tag", "algorithm", "overrides", "roots", "budget", "env",
                                    "eval_env")},
            ))
    ids = [r.run_id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate run ids")
    return out


def all_runs() -> dict[str, RunSpec]:
    runs: dict[str, RunSpec] = {}
    for p in sorted(REGISTRY_DIR.glob("*.yaml")):
        for r in load_family(p):
            if r.run_id in runs:
                raise ValueError(f"duplicate run id across families: {r.run_id}")
            runs[r.run_id] = r
    return runs
