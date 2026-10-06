"""Large-sample action-mask audit; writes experiments/raw/mask_audit/summary.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mplssim.study.maskaudit import CHECKS, run_audit  # noqa: E402
from mplssim.study.protocol import EVAL_SCENARIOS  # noqa: E402
from mplssim.study.provenance import run_record  # noqa: E402

out = Path("experiments/raw/mask_audit")
out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
episodes = [(s, 5000 + i) for s in EVAL_SCENARIOS for i in range(6)]
episodes += [("random_day", 6000 + i) for i in range(40)]
# An untrained MaskablePPO policy (random logits) on the V2 spaces: masked
# probability mass on illegal actions must be exactly zero.
from sb3_contrib import MaskablePPO  # noqa: E402
from mplssim.experiments.v2_factory import make_env_v2  # noqa: E402
ppo = MaskablePPO("MlpPolicy", make_env_v2("full_day"), seed=0, device="cpu",
                  policy_kwargs={"net_arch": [256, 256]})
counts = run_audit(episodes, p_noop=0.5, rng_seed=0, ppo_policy=ppo)
summary = {"counts": dict(counts), "violations": {c: counts.get(c, 0) for c in CHECKS},
           "all_checks_passed": all(counts.get(c, 0) == 0 for c in CHECKS),
           "episodes": episodes}
(out / "summary.json").write_text(json.dumps(run_record(
    kind="mask_audit", wall_seconds=time.perf_counter() - t0, **summary), indent=1, default=str))
print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))
