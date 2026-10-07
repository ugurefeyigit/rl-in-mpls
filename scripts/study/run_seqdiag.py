"""Sequentiality diagnostic (clairvoyant action-value rollouts) on DIAGNOSTIC seeds.

Example:
    python scripts/study/run_seqdiag.py --reference greedy --seeds 4001 4002 4003 \
        --out experiments/raw/seqdiag_greedy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mplssim.study.episode import BaselinePolicy, NoopPolicy  # noqa: E402
from mplssim.study.oracles import MyopicOracle  # noqa: E402
from mplssim.study.protocol import DIAGNOSTIC, EVAL_SCENARIOS  # noqa: E402
from mplssim.study.provenance import run_record  # noqa: E402
from mplssim.study.seqdiag import DEFAULT_HORIZONS, trajectory_diagnostic  # noqa: E402
from mplssim.study.variants import make_variant_factory  # noqa: E402


def reference_policy(name: str, seed: int):
    if name == "noop":
        return NoopPolicy()
    if name == "oracle_h1":
        return MyopicOracle()
    return BaselinePolicy(name, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="greedy")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DIAGNOSTIC))
    p.add_argument("--scenarios", nargs="+", default=list(EVAL_SCENARIOS))
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--h-max", type=int, default=24)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frozen", action="store_true",
                   help="hold traffic and link state fixed during rollouts")
    p.add_argument("--continuation", default="noop", choices=["noop", "greedy"],
                   help="policy after the first action in each rollout")
    p.add_argument("--env", default='{"variant": "base"}',
                   help="environment variant spec (JSON), e.g. '{\"variant\": \"delayed\", \"delay_steps\": 1}'")
    a = p.parse_args()
    env_spec = json.loads(a.env)
    factory = None if env_spec.get("variant", "base") == "base" else make_variant_factory(env_spec)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for scenario in a.scenarios:
        for seed in a.seeds:
            path = a.out / f"{scenario}_seed{seed}.csv"
            if path.exists():
                continue
            df = trajectory_diagnostic(reference_policy(a.reference, seed), scenario, seed,
                                       every=a.every, h_max=a.h_max,
                                       horizons=DEFAULT_HORIZONS, frozen=a.frozen,
                                       env_factory=factory, continuation=a.continuation)
            df.to_csv(path, index=False)
            print(f"{scenario} {seed} states={len(df)} t={time.perf_counter()-t0:.0f}s",
                  flush=True)
    (a.out / "run_record.json").write_text(json.dumps(run_record(
        kind="seqdiag", args=vars(a), env=env_spec, wall_seconds=time.perf_counter() - t0),
        indent=1, default=str))


if __name__ == "__main__":
    main()
