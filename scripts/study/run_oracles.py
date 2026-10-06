"""Evaluate non-learning reference policies on a seed set (paired with learners).

    python scripts/study/run_oracles.py --policies static greedy cspf noop random_valid \
        --seedset test --out experiments/raw/references
    python scripts/study/run_oracles.py --policies oracle_h1 --seedset test --out ...
    python scripts/study/run_oracles.py --policies oracle_h3 oracle_h6 --seeds 3001 3002 ...

Results are written per (policy, scenario, seed) so long oracle runs resume.
Oracles are CLAIRVOYANT (they clone the simulator, including future traffic).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from mplssim.study.episode import (BaselinePolicy, NoopPolicy,  # noqa: E402
                                   UniformValidPolicy, run_episode)
from mplssim.study.oracles import RolloutOracle  # noqa: E402
from mplssim.study.protocol import EVAL_SCENARIOS, SEED_SETS  # noqa: E402
from mplssim.study.provenance import run_record  # noqa: E402
from mplssim.study.variants import make_variant_factory  # noqa: E402


def make_policy(name: str, seed: int):
    if name == "noop":
        return NoopPolicy()
    if name == "random_valid":
        return UniformValidPolicy(seed=seed)
    if name.startswith("oracle_h"):
        return RolloutOracle(horizon=int(name[len("oracle_h"):]))
    return BaselinePolicy(name, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--seedset", default="test")
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--scenarios", nargs="+", default=list(EVAL_SCENARIOS))
    p.add_argument("--env", default='{"variant": "base"}')
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    seeds = a.seeds or list(SEED_SETS[a.seedset])
    env_spec = json.loads(a.env)
    factory = make_variant_factory(env_spec)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for name in a.policies:
        for scenario in a.scenarios:
            for seed in seeds:
                path = a.out / f"{name}__{scenario}__seed{seed}.json"
                if path.exists():
                    continue
                summary, _ = run_episode(make_policy(name, seed), scenario, seed, factory)
                summary["env"] = env_spec
                path.write_text(json.dumps(summary))
                print(f"{name} {scenario} {seed} R={summary['operational_return']:.2f} "
                      f"t={time.perf_counter()-t0:.0f}s", flush=True)
    (a.out / f"run_record_{'_'.join(a.policies)}.json").write_text(json.dumps(run_record(
        kind="reference_policies", args=vars(a), seeds=seeds,
        wall_seconds=time.perf_counter() - t0), indent=1, default=str))


def load_dir(path: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(path).glob("*__seed*.json")):
        s = json.loads(f.read_text())
        comps = s.pop("reward_components")
        s.update({f"rc_{k}": v for k, v in comps.items()})
        rows.append(s)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
