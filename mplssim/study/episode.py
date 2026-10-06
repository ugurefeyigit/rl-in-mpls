"""Generic deterministic evaluation episodes for any policy on any V2 variant.

:func:`mplssim.experiments.evaluation_v2.run_evaluation_episode` is the
governed evaluator of the closed V2 study. It hard-wires the governed seed
sets and learner families. This module evaluates *arbitrary* policies
(learners, baselines, clairvoyant oracles) on *arbitrary* seed sets and
environment variants, while computing the per-episode summary with the same
definitions. ``tests/test_study.py`` asserts that both evaluators produce
identical summaries for a governed policy on a governed seed.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from mplssim.experiments.evaluation_v2 import choose_baseline_action
from mplssim.experiments.learning_common import AuditedV2Env, SeedLedger
from mplssim.rl.reward_v2 import COMPONENT_ORDER, components_sum


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int: ...


class LearnerPolicy:
    """Deterministic (greedy) inference for any learner exposing ``predict``."""

    def __init__(self, learner: Any, name: str) -> None:
        self.learner = learner
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int:
        out = self.learner.predict(observation[None, :], mask[None, :],
                                   deterministic=True)
        return int(np.asarray(out).reshape(-1)[0])


class BaselinePolicy:
    """The repository's static / greedy / cspf controllers, exactly as governed."""

    def __init__(self, name: str, seed: int = 0) -> None:
        from mplssim.baselines import make_baseline
        self.name = name
        self._seed = int(seed)
        self._make = make_baseline
        self.controller = make_baseline(name, seed=self._seed)

    def reset(self) -> None:
        self.controller = self._make(self.name, seed=self._seed)

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int:
        return choose_baseline_action(self.controller, env.eng, mask)


class NoopPolicy:
    name = "noop"

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int:
        return 0


class UniformValidPolicy:
    """Uniformly random legal action (including no-op), seeded per episode."""

    def __init__(self, seed: int = 0, name: str = "random_valid") -> None:
        self.name = name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int:
        valid = np.flatnonzero(mask)
        return int(valid[int(self.rng.integers(len(valid)))])


def make_eval_env(scenario: str, seed: int,
                  env_factory: Callable[..., Any] | None = None) -> AuditedV2Env:
    """Audited V2 (or variant) environment for one evaluation episode."""
    if env_factory is None:
        from mplssim.experiments.v2_factory import make_env_v2
        env_factory = make_env_v2
    raw = env_factory(scenario=scenario, root_seed=int(seed), worker_rank=0)
    return AuditedV2Env(raw, seed_ledger=SeedLedger())


def run_episode(policy: Policy, scenario: str, seed: int,
                env_factory: Callable[..., Any] | None = None,
                keep_steps: bool = False) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Run one full deterministic episode; return (summary, optional step frame)."""
    env = make_eval_env(scenario, seed, env_factory)
    raw = env.unwrapped
    observation, _ = env.reset(options={"episode_seed": int(seed)})
    policy.reset()
    records: list[dict[str, Any]] = []
    actions: Counter[int] = Counter()
    decision_s = 0.0
    truncated = terminated = False
    wall = time.perf_counter()
    while not (terminated or truncated):
        mask = env.action_masks()
        eng = raw.eng
        dwell_active = int(np.sum(eng.te_dwell_remaining > 0))
        dwell_mean = float(np.mean(eng.te_dwell_remaining))
        t0 = time.perf_counter()
        action = int(policy.act(observation, mask, raw))
        decision_s += time.perf_counter() - t0
        if not bool(mask[action]):
            raise RuntimeError(f"{policy.name} selected invalid action {action}")
        moved = float(eng.demand_offered[(action - 1) // eng.k]) if action > 0 else 0.0
        observation, reward, terminated, truncated, info = env.step(action)
        if not info["decoded_action"].get("accepted", False):
            moved = 0.0
        m = info["metrics"]
        rec = {k: v for k, v in m.items() if k != "failed_links"}
        rec.update({
            "step_index": len(records), "action": action,
            "action_accepted": bool(info["decoded_action"].get("accepted", False)),
            "valid_action_count": int(np.sum(mask)), "reward": float(reward),
            "moved_mbps": moved, "dwell_active_demands": dwell_active,
            "dwell_remaining_mean": dwell_mean,
            "n_failed_links": len(m["failed_links"]),
        })
        rec.update({f"rc_{k}": float(v) for k, v in info["reward_components"].items()})
        records.append(rec)
        actions[action] += 1
    frame = pd.DataFrame(records)
    summary = summarize_episode(frame, policy.name, scenario, int(seed), env, actions)
    summary["mean_decision_time_ms"] = 1000.0 * decision_s / len(frame)
    summary["wall_time_seconds"] = time.perf_counter() - wall
    return summary, (frame if keep_steps else None)


def summarize_episode(frame: pd.DataFrame, algorithm: str, scenario: str, seed: int,
                      env: AuditedV2Env, actions: Counter) -> dict[str, Any]:
    """Per-episode metrics with the governed evaluator's definitions."""
    n_demands = int(env.unwrapped.n_demands)
    hours = float(frame["t_min"].iloc[-1]) / 60.0
    interval_s = float(frame["t_min"].iloc[0]) * 60.0
    exact = all(
        components_sum({n: float(r[f"rc_{n}"]) for n in COMPONENT_ORDER}) == float(r["reward"])
        for _, r in frame.iterrows())
    accepted = int(frame["accepted_te_changes"].sum())
    reversals = int(frame["te_reversals"].sum())
    prot = frame["protected_disconnected_demands"]
    disc = frame["disconnected_demands"]
    integ = env.integrity
    return {
        "algorithm": algorithm, "scenario": scenario, "seed": seed,
        "episode_length": int(len(frame)),
        "operational_return": float(frame["reward"].sum()),
        "reward_components": {c: float(frame[f"rc_{c}"].sum()) for c in COMPONENT_ORDER},
        "reward_component_sum_exact": bool(exact),
        "offered_gbit_total": float((frame["offered_mbps"] * interval_s / 1000.0).sum()),
        "delivered_gbit_total": float((frame["delivered_mbps"] * interval_s / 1000.0).sum()),
        "delivered_ratio_mean": float(frame["delivered_ratio"].mean()),
        "sla_violations_demand_intervals": int(frame["sla_violations"].sum()),
        "protected_disconnection_demand_intervals": int(prot.sum()),
        "unprotected_disconnection_demand_intervals": int((disc - prot).sum()),
        "max_utilization_peak": float(frame["max_util"].max()),
        "max_utilization_mean": float(frame["max_util"].mean()),
        "congested_link_intervals": int(frame["congested_links"].sum()),
        "overload_ratio_mean": float(frame["overload_ratio"].mean()),
        "delay_ms_mean": float(frame["mean_delay_ms"].mean()),
        "loss_ratio_mean": float(frame["loss_ratio"].mean()),
        "accepted_te_changes": accepted,
        "reroutes_per_hour": accepted / hours if hours else 0.0,
        "te_reversals": reversals,
        "flaps_per_demand": reversals / n_demands,
        "moved_mbps_total": float(frame["moved_mbps"].sum()),
        "rejected_te_requests": int(frame["rejected_te_requests"].sum()),
        "frr_changes": int(frame["frr_changes"].sum()),
        "noop_frequency": int(actions[0]) / len(frame),
        "invalid_action_attempts": integ.invalid_action_attempts,
        "mask_disagreements": integ.mask_disagreements,
        "reward_mismatches": integ.reward_mismatches,
        "nonfinite_values": integ.nonfinite_values,
        "solver_convergence_failures": integ.solver_failures,
        "protected_safety_failures": integ.protected_safety_failures,
        "solver_iterations_max": int(frame["flow_solver_iterations_max"].max()),
    }


def evaluate_matrix(policy: Policy, scenarios: list[str], seeds: list[int],
                    env_factory: Callable[..., Any] | None = None) -> pd.DataFrame:
    """Evaluate one policy over a paired scenario x seed matrix."""
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            summary, _ = run_episode(policy, scenario, int(seed), env_factory)
            comps = summary.pop("reward_components")
            summary.update({f"rc_{k}": v for k, v in comps.items()})
            rows.append(summary)
    return pd.DataFrame(rows)
