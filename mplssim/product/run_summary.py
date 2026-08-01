"""Per-environment episode summaries for `POST /api/export/save-run`.

`mplssim.experiments.runner.summarize_records` is a **V1** summarizer. It reads
columns the V1 interval record has and the frozen V2 interval record does not:
`jain_fairness`, `p95_delay_ms`, `priority_sla_success`, `carried_mbps`,
`reroutes`, `flaps`, `frr_events` — and it reads `engine.path_change_count`,
which the V2 engine does not keep.

There were two ways to make save-run work under V2. Padding the V2 record with
zeros for the columns it does not have would have produced a row that *looks*
like a V1 result and is not one, so V2 gets its own summarizer over its own
columns instead. A V2 row and a V1 row therefore have different fields, and each
carries `environment_version` so nothing downstream can average them by accident.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: Cumulative movement counters. Kept separate: controller TE changes, FRR
#: protection moves and post-recovery restorations are three different things.
_V2_MOVEMENT_COLUMNS: tuple[str, ...] = (
    "accepted_te_changes", "rejected_te_requests", "te_reversals",
    "frr_changes", "frr_disconnections", "recovery_restorations",
)


def _interval_seconds(df: pd.DataFrame) -> float:
    """Control-interval length read from the trace, never hardcoded."""
    if len(df) >= 2 and "t_min" in df:
        return float(df["t_min"].iloc[1] - df["t_min"].iloc[0]) * 60.0
    return 300.0


def summarize_v2_records(df: pd.DataFrame, algorithm: str, scenario: str,
                         seed: int, training_root: int | None = None,
                         checkpoint_id: str | None = None) -> dict[str, Any]:
    """Aggregate one V2 episode from the V2 interval record only."""
    n = len(df)
    out: dict[str, Any] = {
        "environment_version": "v2",
        "algorithm": algorithm,
        "scenario": scenario,
        "seed": seed,
        "training_root": training_root,
        "checkpoint_id": checkpoint_id,
        "steps": n,
        "reward_sum": float(df["reward"].sum()),
        "reward_mean": float(df["reward"].mean()),
        "max_util_peak": float(df["max_util"].max()),
        "max_util_mean": float(df["max_util"].mean()),
        "gross_max_util_peak": (float(df["gross_max_util"].max())
                                if "gross_max_util" in df else None),
        "mean_util": float(df["mean_util"].mean()),
        "util_std_mean": float(df["util_std"].mean()),
        "mean_delay_ms": float(df["mean_delay_ms"].mean()),
        "max_delay_ms": float(df["max_delay_ms"].max()),
        "loss_ratio_mean": float(df["loss_ratio"].mean()),
        "delivered_ratio_mean": float(df["delivered_ratio"].mean()),
        "overload_ratio_mean": (float(df["overload_ratio"].mean())
                                if "overload_ratio" in df else None),
        "sla_violation_steps": int((df["sla_violations"] > 0).sum()),
        "sla_violations_peak": int(df["sla_violations"].max()),
        "sla_violation_fraction_mean": (float(df["sla_violation_fraction"].mean())
                                        if "sla_violation_fraction" in df else None),
        "congested_link_steps": int((df["congested_links"] > 0).sum()),
        "time_above_80pct": float((df["max_util"] >= 0.8).mean()),
        "time_above_90pct": float((df["max_util"] >= 0.9).mean()),
        "time_above_100pct": float((df["max_util"] >= 1.0).mean()),
        "disconnected_steps": int((df["disconnected_demands"] > 0).sum()),
        "protected_disconnected_steps": (
            int((df["protected_disconnected_demands"] > 0).sum())
            if "protected_disconnected_demands" in df else None),
        # Offered traffic above delivered, converted to gigabits exactly once.
        "dropped_gbit_total": (
            float(((df["offered_mbps"] - df["delivered_mbps"])
                   * _interval_seconds(df) / 1000).sum())
            if {"offered_mbps", "delivered_mbps"} <= set(df.columns) else None),
    }
    for column in _V2_MOVEMENT_COLUMNS:
        out[f"{column}_total"] = int(df[column].sum()) if column in df else None

    # Recovery time: intervals from the first failed-link interval until SLA
    # violations return to zero. None when the episode has no failure.
    fail_steps = df.index[df["n_failed_links"] > 0] if "n_failed_links" in df else []
    if len(fail_steps) > 0:
        first = int(fail_steps[0])
        post = df.loc[first:]
        clear = post.index[post["sla_violations"] == 0]
        out["recovery_steps"] = int(clear[0] - first) if len(clear) > 0 else n - first
    else:
        out["recovery_steps"] = None
    out["not_measured"] = [
        "jain_fairness", "p95_delay_ms", "priority_sla_success",
        "path_changes_per_demand",
    ]
    out["not_measured_reason"] = (
        "The frozen V2 interval record does not carry these V1 quantities. They "
        "are reported as absent rather than padded with zeros.")
    return out


def summarize_session_runner(runner: Any, scenario: str, seed: int,
                             environment: str,
                             training_root: int | None) -> dict[str, Any]:
    """Summarize one live-session runner for save-run, per environment."""
    df = pd.DataFrame([
        {**record["metrics"], "reward": record["reward"],
         "n_failed_links": record["n_failed_links"]}
        for record in runner.history
    ])
    if environment == "v2":
        summary = summarize_v2_records(
            df, algorithm=runner.algorithm, scenario=scenario, seed=seed,
            training_root=training_root,
            checkpoint_id=getattr(runner, "checkpoint_id", None))
    else:
        from mplssim.experiments.runner import summarize_records

        summary = summarize_records(df, runner.algorithm, scenario, seed,
                                    engine=runner.eng)
        summary["environment_version"] = "v1"
    summary["cumulative_reward"] = round(float(runner.cumulative_reward), 4)
    summary["record_class"] = "live_demonstration"
    summary["is_evidence"] = False
    summary["evidence_reason"] = (
        "A saved live run is a demonstration record. It is not holdout evidence "
        "and is never merged with the closed study's numbers.")
    return summary


__all__ = ["summarize_session_runner", "summarize_v2_records"]
