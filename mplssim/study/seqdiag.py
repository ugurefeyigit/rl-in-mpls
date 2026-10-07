"""Sequentiality diagnostics: does the immediate reward rank actions correctly?

For a state ``s`` visited by a reference policy, and every legal action ``a``,
we roll the exact (cloned) simulator forward for ``H_max`` control intervals:
first ``a``, then no-op. Prefix sums give the H-interval return ``G_H(s, a)``
for every ``H <= H_max`` from a single rollout. With ``a = 0`` (no-op) as the
reference,

    Delta_H(s, a) = G_H(s, a) - G_H(s, 0)

is the H-interval advantage of making change ``a`` now and then leaving the
network alone. ``Delta_1`` is exactly the immediate-reward advantage the
contextual bandit is trained to predict.

If ``argmax_a Delta_1`` almost always equals ``argmax_a Delta_H`` and the
regret of the myopic choice under the H-step criterion is small, then -- for
this environment and this family of continuations -- acting on immediate
reward loses little, and the sequential machinery of RL has little to work
with. The diagnostic is clairvoyant (exact future traffic) and its
continuation is fixed, so it measures the structure of the problem, not the
performance of any learner.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mplssim.study.episode import Policy, make_eval_env
from mplssim.study.oracles import simulate

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)


def state_diagnostic(env: Any, mask: np.ndarray, h_max: int,
                     horizons: tuple[int, ...], frozen: bool = False,
                     continuation: str = "noop") -> dict[str, Any]:
    """Per-state action-value table and summary statistics."""
    legal = [int(a) for a in np.flatnonzero(mask)]
    curves: dict[int, np.ndarray] = {}
    for a in legal:
        _, rewards = simulate(env, [a], horizon=h_max, frozen=frozen,
                              continuation=continuation)
        curves[a] = np.cumsum(rewards)
    length = min(len(c) for c in curves.values())
    hs = [h for h in horizons if h <= length]
    out: dict[str, Any] = {"n_legal": len(legal), "rollout_length": length}
    base = {h: curves[0][h - 1] for h in hs}
    delta = {h: np.array([curves[a][h - 1] - base[h] for a in legal]) for h in hs}
    acts = np.array(legal)

    def best(h: int) -> int:
        d = delta[h]
        i = int(np.argmax(d))
        # exact ties resolve to no-op (index 0 of ``legal`` is action 0)
        return int(acts[i]) if d[i] > 1e-12 else 0

    a1 = best(1)
    i1 = legal.index(a1)
    for h in hs:
        ah = best(h)
        d = delta[h]
        out[f"best_a_h{h}"] = ah
        out[f"best_delta_h{h}"] = float(d.max(initial=0.0))
        out[f"myopic_delta_h{h}"] = float(d[i1])
        out[f"myopic_regret_h{h}"] = float(max(d.max(), 0.0) - d[i1])
        out[f"agree_h{h}"] = bool(ah == a1)
        if h > 1 and np.std(d) > 0 and np.std(delta[1]) > 0:
            out[f"spearman_h1_h{h}"] = float(spearmanr(delta[1], d).statistic)
        else:
            out[f"spearman_h1_h{h}"] = float("nan")
        out[f"n_sacrifice_h{h}"] = int(np.sum((delta[1] < 0) & (d > 0)))
        out[f"n_misleading_h{h}"] = int(np.sum((delta[1] > 0) & (d < 0)))
        out[f"best_is_sacrifice_h{h}"] = bool(ah != 0 and delta[1][legal.index(ah)] < 0)
    return out


def trajectory_diagnostic(policy: Policy, scenario: str, seed: int, every: int = 4,
                          h_max: int = 24, horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                          env_factory: Callable[..., Any] | None = None,
                          frozen: bool = False, continuation: str = "noop") -> pd.DataFrame:
    """Run ``policy`` for one episode; diagnose every ``every``-th state."""
    env = make_eval_env(scenario, seed, env_factory)
    raw = env.unwrapped
    obs, _ = env.reset(options={"episode_seed": int(seed)})
    policy.reset()
    rows = []
    step = 0
    truncated = False
    while not truncated:
        mask = env.action_masks()
        if step % every == 0:
            row = state_diagnostic(raw, mask, h_max, horizons, frozen=frozen,
                                   continuation=continuation)
            row.update({"scenario": scenario, "seed": int(seed), "step": step,
                        "t_min": float(raw.eng.t_min),
                        "reference_policy": policy.name, "frozen": frozen,
                        "continuation": continuation})
            rows.append(row)
        action = int(policy.act(obs, mask, raw))
        obs, _, _, truncated, _ = env.step(action)
        step += 1
    return pd.DataFrame(rows)
