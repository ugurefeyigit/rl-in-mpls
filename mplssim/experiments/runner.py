"""Episode runner used by evaluation, experiments, and the live server.

`run_episode` executes one full scenario with a given controller and returns
(per-step records, summary). Both RL and baselines run through the SAME
engine and the SAME reward computation, on identical seeded traffic — paired
comparison by construction.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from mplssim.baselines import make_baseline
from mplssim.factory import engine_config_from_training, make_engine
from mplssim.rl.env import MplsTeEnv
from mplssim.rl.reward import compute_reward


def run_episode(
    algorithm: str,
    scenario: str,
    seed: int,
    model: Any | None = None,
    safety_filter: bool = True,
    deterministic: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one episode. ``algorithm``: static | greedy | cspf | random | rl.

    For ``rl`` a loaded sb3 model must be supplied.
    """
    if algorithm == "rl":
        return _run_rl(scenario, seed, model, safety_filter, deterministic)
    return _run_baseline(algorithm, scenario, seed)


def _record(interval: dict, reward: float, comps: dict, extra: dict | None = None) -> dict:
    rec = {k: v for k, v in interval.items() if k != "failed_links"}
    rec["n_failed_links"] = len(interval["failed_links"])
    rec["reward"] = reward
    rec.update({f"rc_{k}": v for k, v in comps.items()})
    if extra:
        rec.update(extra)
    return rec


def _run_baseline(name: str, scenario: str, seed: int) -> tuple[pd.DataFrame, dict]:
    eng = make_engine(scenario, seed=seed, cfg=engine_config_from_training())
    ctl = make_baseline(name, seed=seed)
    records: list[dict] = []
    decision_times: list[float] = []
    while not eng.done:
        t0 = time.perf_counter()
        moves = ctl.decide(eng)
        decision_times.append(time.perf_counter() - t0)
        rerouted = False
        log_mark = len(eng.action_log)
        for d_idx, p_idx in moves:
            ok, _ = eng.apply_action(
                d_idx, p_idx, source=ctl.name if ctl.name != "random" else "rl"
            )
            rerouted = rerouted or ok
        # flap flag from the controller's own actions only (FRR excluded),
        # mirroring how the RL env attributes reroute/flap penalties
        flapped = any(a.is_flap for a in eng.action_log[log_mark:] if a.accepted)
        interval = eng.step_interval()
        reward, comps = compute_reward(interval, rerouted, flapped, invalid=False)
        records.append(_record(interval, reward, comps))
    df = pd.DataFrame(records)
    summary = summarize_records(df, algorithm=name, scenario=scenario, seed=seed, engine=eng)
    summary["mean_decision_time_ms"] = 1000.0 * float(np.mean(decision_times))
    return df, summary


def _run_rl(scenario: str, seed: int, model: Any, safety_filter: bool,
            deterministic: bool) -> tuple[pd.DataFrame, dict]:
    if model is None:
        raise ValueError("rl algorithm requires a loaded model")
    env = MplsTeEnv(scenario=scenario, base_seed=seed, safety_filter=safety_filter)
    obs, _ = env.reset(options={"episode_seed": seed})
    records: list[dict] = []
    decision_times: list[float] = []
    invalid_count = 0
    done = False
    while not done:
        mask = env.action_masks()
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        decision_times.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        dec = info["decoded_action"]
        if dec.get("accepted") is False:
            invalid_count += 1
        records.append(_record(
            info["metrics"], reward, info["reward_components"],
            extra={"action": dec["action"], "action_type": dec["type"]},
        ))
    df = pd.DataFrame(records)
    summary = summarize_records(df, algorithm="rl", scenario=scenario, seed=seed, engine=env.eng)
    summary["mean_decision_time_ms"] = 1000.0 * float(np.mean(decision_times))
    summary["invalid_actions"] = invalid_count
    return df, summary


def summarize_records(df: pd.DataFrame, algorithm: str, scenario: str,
                      seed: int, engine: Any = None) -> dict[str, Any]:
    """Aggregate one episode into presentation-ready scalar metrics."""
    n = len(df)
    out: dict[str, Any] = {
        "algorithm": algorithm,
        "scenario": scenario,
        "seed": seed,
        "steps": n,
        "reward_sum": float(df["reward"].sum()),
        "reward_mean": float(df["reward"].mean()),
        "max_util_peak": float(df["max_util"].max()),
        "max_util_mean": float(df["max_util"].mean()),
        "mean_util": float(df["mean_util"].mean()),
        "util_std_mean": float(df["util_std"].mean()),
        "jain_fairness_mean": float(df["jain_fairness"].mean()),
        "mean_delay_ms": float(df["mean_delay_ms"].mean()),
        "p95_delay_ms": float(df["p95_delay_ms"].mean()),
        "max_delay_ms": float(df["max_delay_ms"].max()),
        "loss_ratio_mean": float(df["loss_ratio"].mean()),
        "delivered_ratio_mean": float(df["delivered_ratio"].mean()),
        # Mbps * seconds = megabits; /1000 converts to gigabits exactly once.
        # (V1 published files divided twice — values there are Gbit/1000; the
        # column name is unchanged so corrected re-runs are comparable.)
        "dropped_gbit_total": float(
            ((df["offered_mbps"] - df["carried_mbps"]) * 5 * 60 / 1000).sum()
        ),
        "sla_violation_steps": int((df["sla_violations"] > 0).sum()),
        "sla_violations_total": int(df["sla_violations"].sum()),
        "priority_sla_success_mean": float(df["priority_sla_success"].mean()),
        "congested_link_steps": int(df["congested_links"].sum()),
        "time_above_80pct": float((df["max_util"] >= 0.8).mean()),
        "time_above_90pct": float((df["max_util"] >= 0.9).mean()),
        "time_above_100pct": float((df["max_util"] >= 1.0).mean()),
        "reroutes_total": int(df["reroutes"].sum()),
        "flaps_total": int(df["flaps"].sum()),
        "frr_events_total": int(df["frr_events"].sum()),
        "disconnected_steps": int((df["disconnected_demands"] > 0).sum()),
    }
    # Failure recovery time: steps from first failed-link step until SLA
    # violations return to zero (NaN if the scenario has no failure).
    fail_steps = df.index[df["n_failed_links"] > 0]
    if len(fail_steps) > 0:
        f0 = int(fail_steps[0])
        post = df.loc[f0:]
        ok = post.index[post["sla_violations"] == 0]
        out["recovery_steps"] = int(ok[0] - f0) if len(ok) > 0 else n - f0
    else:
        out["recovery_steps"] = None
    if engine is not None:
        out["path_changes_per_demand"] = float(np.mean(engine.path_change_count))
    return out
