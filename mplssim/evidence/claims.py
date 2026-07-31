"""The single place where the V2 study's scientific numbers are computed.

Nothing downstream — API, frontend, documentation script — does arithmetic on the
frozen evidence. It all comes through here, so there is exactly one implementation
to test against the committed files.

Two rules this module exists to enforce:

1. **Root-aware aggregation.** Learner aggregates are the unweighted mean over the
   three training-root means. Pooling the 105 episodes would treat episodes as
   independent training roots. Baselines have no training root, ran once, and carry
   `root_count == 1` with zero root spread.
2. **No silent grain-mixing.** Where two defensible statistics share a name — no-op
   share, wall time — both are returned under distinct keys with their grain stated.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mplssim.evidence import identity
from mplssim.evidence.loader import Continuity, FinalHoldout, Seed42

#: The frozen conclusions, in the words the study closed with. The last two must
#: always travel together: the result is a negative finding about this formulation,
#: not a general claim about planning.
CONCLUSIONS: tuple[str, ...] = (
    "The final holdout ran exactly once, over 315 episodes: 35 per learner "
    "checkpoint or baseline.",
    "The masked contextual bandit reached a mean holdout return of 18.221 against "
    "MaskablePPO's 9.036, an advantage of 9.185. Greedy was the strongest "
    "repository baseline at -2.327.",
    "The bandit won all three training roots and six of seven scenarios. PPO "
    "retained a 1.107-point lead in deceptive_local_optimum, preserved here as a "
    "negative result against an across-the-board bandit claim.",
    "All safety and integrity checks passed. Bandit and PPO both averaged about "
    "2.148 reroutes/hour; the bandit had fewer reversals and flaps but moved more "
    "bandwidth than PPO.",
    "The frozen evidence does not positively support a need for temporal planning "
    "in this formulation: the explicitly myopic learner remained stronger.",
    "This is not evidence that planning is generally irrelevant to MPLS or traffic "
    "engineering. It is a result about these frozen learners, scenarios, reward and "
    "observation design only.",
    "No training, tuning, sweep, reselection, redesign, retry, or policy debugging "
    "used holdout results.",
)

#: Metric columns that are meaningful to average across roots.
_SKIP_FOR_ROOT_MEAN = {"policy_id", "algorithm", "training_root", "episodes",
                       "operational_return_std", "scenario"}


def _learner_rows(per_root: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    mask = per_root["algorithm"] == algorithm
    if algorithm in identity.LEARNER_ALGORITHMS:
        mask &= per_root["training_root"].astype(str) != "baseline"
    return per_root[mask]


# ------------------------------------------------------------- aggregation
def root_aggregate(per_root: pd.DataFrame, algorithm: str) -> dict[str, Any]:
    """Aggregate a method over training roots, never over pooled episodes."""
    rows = _learner_rows(per_root, algorithm)
    if rows.empty:
        raise KeyError(f"no per-root rows for algorithm {algorithm!r}")

    out: dict[str, Any] = {"algorithm": algorithm}
    for col in rows.columns:
        if col in _SKIP_FOR_ROOT_MEAN or not pd.api.types.is_numeric_dtype(rows[col]):
            continue
        out[col] = float(rows[col].mean())

    returns = rows["operational_return_mean"]
    out["root_count"] = int(len(rows))
    out["root_mean_std"] = float(returns.std(ddof=1)) if len(rows) > 1 else 0.0
    out["episodes"] = int(rows["episodes"].sum())
    out["episodes_per_root"] = int(rows["episodes"].iloc[0])
    out["is_learner"] = algorithm in identity.LEARNER_ALGORITHMS
    return out


def aggregate_table(fh: FinalHoldout) -> list[dict[str, Any]]:
    """The five-method aggregate comparison, ordered best return first.

    Episode dispersion is read from the frozen aggregate table rather than
    recomputed: it is the pooled standard deviation over all of a method's
    episodes, which is not the mean of the per-root standard deviations.
    """
    frozen = fh.aggregate.set_index("algorithm")
    rows = []
    for algo in identity.ALL_ALGORITHMS:
        row = root_aggregate(fh.per_root, algo)
        row["operational_return_std"] = float(frozen.loc[algo, "operational_return_std"])
        row["episode_std_grain"] = "pooled over every episode of this method"
        rows.append(row)
    return sorted(rows, key=lambda r: r["operational_return_mean"], reverse=True)


def episode_accounting(fh: FinalHoldout) -> dict[str, Any]:
    """How the 315 episodes were composed, and the assertion that it ran once."""
    ep = fh.manifest["episodes"]
    auth = fh.manifest["authorization"]
    return {
        "total": int(ep["total"]),
        "per_policy": int(ep["per_policy_or_baseline"]),
        "policies": identity.POLICY_COUNT,
        "learner_checkpoints": int(ep["learner_checkpoints"]),
        "baselines": len(ep["baselines"]),
        "scenarios": len(identity.SCENARIOS),
        "seeds": list(identity.HOLDOUT_SEEDS),
        "ran_once": auth.get("workflow") == "single final holdout",
        "deterministic_inference": bool(auth.get("deterministic_inference")),
    }


# --------------------------------------------------------- learner comparison
def learner_comparison(fh: FinalHoldout) -> dict[str, Any]:
    """Bandit versus PPO at the aggregate and per-root grain."""
    bandit = root_aggregate(fh.per_root, "masked_bandit")
    ppo = root_aggregate(fh.per_root, "maskable_ppo")

    roots: list[dict[str, Any]] = []
    for root in identity.TRAINING_ROOTS:
        sel = fh.per_root["training_root"].astype(str) == str(root)
        b = float(fh.per_root[sel & (fh.per_root.algorithm == "masked_bandit")]
                  .operational_return_mean.iloc[0])
        p = float(fh.per_root[sel & (fh.per_root.algorithm == "maskable_ppo")]
                  .operational_return_mean.iloc[0])
        prov = fh.provenance[fh.provenance["training_root"].astype(str) == str(root)]
        roots.append({
            "training_root": root,
            "bandit": b,
            "ppo": p,
            "advantage": b - p,
            "winner": "masked_bandit" if b > p else "maskable_ppo",
            "bandit_checkpoint": int(prov[prov.algorithm == "masked_bandit"]
                                     .checkpoint_transition.iloc[0]),
            "ppo_checkpoint": int(prov[prov.algorithm == "maskable_ppo"]
                                  .checkpoint_transition.iloc[0]),
        })

    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "bandit_return": bandit["operational_return_mean"],
        "ppo_return": ppo["operational_return_mean"],
        "advantage": bandit["operational_return_mean"] - ppo["operational_return_mean"],
        "bandit_root_mean_std": bandit["root_mean_std"],
        "ppo_root_mean_std": ppo["root_mean_std"],
        "bandit_episode_std": float(
            _learner_rows(fh.per_root, "masked_bandit").operational_return_std.mean()),
        "ppo_episode_std": float(
            _learner_rows(fh.per_root, "maskable_ppo").operational_return_std.mean()),
        "roots": roots,
        "roots_won": sum(r["winner"] == "masked_bandit" for r in roots),
        "roots_total": len(roots),
    }


def scenario_comparison(fh: FinalHoldout) -> list[dict[str, Any]]:
    """Per-scenario bandit-versus-PPO, root-averaged before comparison."""
    rows: list[dict[str, Any]] = []
    for scen in identity.SCENARIOS:
        sel = fh.scenario["scenario"] == scen
        b_rows = fh.scenario[sel & (fh.scenario.algorithm == "masked_bandit")]
        p_rows = fh.scenario[sel & (fh.scenario.algorithm == "maskable_ppo")]
        b = float(b_rows.operational_return_mean.mean())
        p = float(p_rows.operational_return_mean.mean())
        baselines = {
            algo: float(fh.scenario[sel & (fh.scenario.algorithm == algo)]
                        .operational_return_mean.iloc[0])
            for algo in identity.BASELINE_ALGORITHMS
        }
        rows.append({
            "scenario": scen,
            "bandit": b,
            "ppo": p,
            "advantage": b - p,
            "winner": "masked_bandit" if b > p else "maskable_ppo",
            "root_count": int(len(b_rows)),
            "episodes_per_root": int(b_rows.episodes.iloc[0]),
            "baselines": baselines,
        })
    return rows


# ------------------------------------------------------------ reward integrity
def reward_reconciliation(fh: FinalHoldout) -> dict[str, Any]:
    """Recompute the 12-component sum for every policy and report the residual."""
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for _, r in fh.reward_components.iterrows():
        comps = {c: float(r[f"{c}_mean"]) for c in identity.REWARD_COMPONENTS}
        total = sum(comps.values())
        ret = float(r["operational_return_mean"])
        worst = max(worst, abs(total - ret))
        rows.append({
            "policy_id": r["policy_id"],
            "algorithm": r["algorithm"],
            "training_root": str(r["training_root"]),
            "components": comps,
            "sum": total,
            "operational_return_mean": ret,
            "residual": abs(total - ret),
            "reported_max_abs_residual": float(r["max_abs_reward_residual"]),
        })
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "rows": rows,
        "component_names": list(identity.REWARD_COMPONENTS),
        "max_residual": worst,
        "reported_max_abs_residual": float(fh.reward_components
                                           ["max_abs_reward_residual"].max()),
        "exact": worst < 1e-9,
    }


# ------------------------------------------------- grains that look alike
def noop_shares(fh: FinalHoldout) -> dict[str, Any]:
    """No-op share at both published grains, each labelled.

    `pooled_step_share` counts action 0 across all 3,300 recorded steps.
    `episode_mean_share` averages each episode's own no-op frequency.
    The final-holdout report quotes the pooled figure.
    """
    pooled: dict[str, float] = {}
    for algo in identity.ALL_ALGORITHMS:
        rows = fh.actions[fh.actions.algorithm == algo]
        total = float(rows["count"].sum())
        noop = float(rows[rows.action == identity.NOOP_ACTION]["count"].sum())
        pooled[algo] = noop / total if total else 0.0

    episode_mean = {
        algo: root_aggregate(fh.per_root, algo)["noop_frequency_mean"]
        for algo in identity.ALL_ALGORITHMS
    }
    return {
        "pooled_step_share": pooled,
        "episode_mean_share": episode_mean,
        "pooled_grain": "action 0 count / all recorded steps, pooled over episodes",
        "episode_grain": "mean over episodes of each episode's own no-op frequency",
        "steps_per_policy": int(fh.actions.groupby("policy_id")["count"].sum().iloc[0]),
    }


def runtime_summary(fh: FinalHoldout) -> dict[str, Any]:
    """Runner wall time and the six per-checkpoint times, kept apart.

    `total_runner_wall_seconds` covers the whole one-shot run including the three
    baselines and setup. `checkpoint_wall_seconds_sum` covers only the six learner
    evaluations. They are different quantities and differ by design.
    """
    rt = fh.manifest["runtime"]
    prov = fh.provenance
    return {
        "total_runner_wall_seconds": float(rt["total_wall_seconds"]),
        "checkpoint_wall_seconds_sum": float(prov["evaluation_wall_seconds"].sum()),
        "total_grain": "whole one-shot runner, including the three baselines and setup",
        "checkpoint_grain": "sum of the six learner-checkpoint evaluations only",
        "device": rt["device"],
        "gpu": rt["gpu"],
        "torch": rt["torch"],
        "cuda_runtime": rt["cuda_runtime"],
        "peak_gpu_memory_bytes_min": int(prov["peak_gpu_memory_bytes"].min()),
        "peak_gpu_memory_bytes_max": int(prov["peak_gpu_memory_bytes"].max()),
        "hardware": rt.get("hardware", {}),
    }


# ------------------------------------------------------------ churn / safety
_CHURN_FIELDS = {
    "reroutes_per_hour": "reroutes_per_hour_mean",
    "te_reversals": "te_reversals_mean",
    "flaps_per_demand": "flaps_per_demand_mean",
    "moved_mbps_total": "moved_mbps_total_mean",
    "accepted_te_changes": "accepted_te_changes_mean",
    "rejected_te_requests": "rejected_te_requests_mean",
    "dwell_active_demand_intervals": "dwell_active_demand_intervals_mean",
    "dwell_remaining_mean": "dwell_remaining_mean_mean",
    "frr_changes": "frr_changes_mean",
    "frr_disconnections": "frr_disconnections_mean",
    "recovery_restorations": "recovery_restorations_mean",
}


def churn_summary(fh: FinalHoldout) -> dict[str, dict[str, float]]:
    """Route churn per method, including the bandit's higher moved bandwidth."""
    out: dict[str, dict[str, float]] = {}
    for algo in identity.ALL_ALGORITHMS:
        agg = root_aggregate(fh.per_root, algo)
        out[algo] = {name: float(agg[col]) for name, col in _CHURN_FIELDS.items()}
    return out


def safety_summary(fh: FinalHoldout) -> dict[str, Any]:
    """Safety and integrity status, plus the identical-accounting checks."""
    integ = fh.integrity
    counters = {c: int(integ[c].astype(float).sum()) for c in (
        "invalid_action_attempts_total", "mask_disagreements_total",
        "reward_mismatches_total", "nonfinite_values_total",
        "solver_convergence_failures_total", "protected_safety_failures_total")}
    prot = fh.per_root["protected_disconnection_demand_intervals_mean"]
    unprot = fh.per_root["unprotected_disconnection_demand_intervals_mean"]
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "all_checks_passed": bool(
            integ["all_checks_passed"].astype(str).str.lower().eq("true").all()),
        "policies": int(len(integ)),
        "counters": counters,
        "all_episodes_truncated": bool(
            integ["all_episodes_truncated"].astype(str).str.lower().eq("true").all()),
        "any_episode_terminated": bool(
            integ["any_episode_terminated"].astype(str).str.lower().eq("true").any()),
        "protected_disconnection_identical_across_methods": bool(prot.nunique() == 1),
        "unprotected_disconnection_identical_across_methods": bool(unprot.nunique() == 1),
        "protected_disconnection_demand_intervals": float(prot.iloc[0]),
        "unprotected_disconnection_demand_intervals": float(unprot.iloc[0]),
        "rejected_te_requests_total": float(fh.per_root["rejected_te_requests_mean"].sum()),
        "rows": [
            {"policy_id": r["policy_id"], "algorithm": r["algorithm"],
             "training_root": str(r["training_root"]), "episodes": int(r["episodes"]),
             "unique_seeds": int(r["unique_seeds"]),
             "unique_scenarios": int(r["unique_scenarios"]),
             "all_checks_passed": str(r["all_checks_passed"]).lower() == "true"}
            for _, r in integ.iterrows()
        ],
    }


def provenance_table(fh: FinalHoldout) -> list[dict[str, Any]]:
    """Checkpoint identity, binding and per-checkpoint runtime."""
    return [
        {
            "training_root": int(r["training_root"]),
            "algorithm": r["algorithm"],
            "checkpoint_transition": int(r["checkpoint_transition"]),
            "payload_sha256": r["payload_sha256"],
            "sidecar_sha256": r["sidecar_sha256"],
            "checkpoint_path": r["checkpoint_path"],
            "training_source_sha": r["training_source_sha"],
            "evaluation_source_sha": r["evaluation_source_sha"],
            "artifact_worktree": r["artifact_worktree"],
            "artifact_worktree_head": r["artifact_worktree_head"],
            "evaluation_wall_seconds": float(r["evaluation_wall_seconds"]),
            "resolved_device": r["resolved_device"],
            "gpu_name": r["gpu_name"],
            "peak_gpu_memory_bytes": int(r["peak_gpu_memory_bytes"]),
        }
        for _, r in fh.provenance.iterrows()
    ]


def action_distribution(fh: FinalHoldout) -> list[dict[str, Any]]:
    """Every action 0-68 for every policy, zero counts included."""
    return [
        {"policy_id": r["policy_id"], "algorithm": r["algorithm"],
         "training_root": str(r["training_root"]), "action": int(r["action"]),
         "action_type": r["action_type"], "count": int(r["count"]),
         "frequency": float(r["frequency"])}
        for _, r in fh.actions.iterrows()
    ]


def holdout_summary(fh: FinalHoldout) -> dict[str, Any]:
    """Everything the final-holdout stage asserts, in one payload."""
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "source_sha": identity.EVALUATION_SOURCE_SHA,
        "artifact_path": str(fh.directory),
        "full_artifact_path": fh.manifest.get("full_artifact_path"),
        "episodes": episode_accounting(fh),
        "comparison": learner_comparison(fh),
        "aggregate": aggregate_table(fh),
        "safety": safety_summary(fh),
        "churn": churn_summary(fh),
        "runtime": runtime_summary(fh),
        "noop": noop_shares(fh),
        "conclusions": list(CONCLUSIONS),
    }


# ------------------------------------------------------- development evidence
def development_summary(cont: Continuity) -> dict[str, Any]:
    """Three-root continuity results. Development stage — never a holdout claim."""
    agg = cont.aggregate.set_index("algorithm")
    return {
        "stage": identity.STAGE_DEVELOPMENT,
        "study": "three-root continuity",
        "source_sha": identity.CONTINUATION_SOURCE_SHA,
        "artifact_path": str(cont.directory),
        "evaluation_seeds": list(identity.CONTINUITY_SEEDS),
        "holdout_accessed": bool(cont.manifest["holdout_accessed"]),
        "bandit_return": float(agg.loc["masked_bandit", "root_operational_return_mean"]),
        "ppo_return": float(agg.loc["maskable_ppo", "root_operational_return_mean"]),
        "methods": [
            {"algorithm": a,
             "root_return_mean": float(agg.loc[a, "root_operational_return_mean"]),
             "root_return_std": float(agg.loc[a, "root_operational_return_std"]),
             "episode_return_mean": float(agg.loc[a, "episode_operational_return_mean"]),
             "episode_return_std": float(agg.loc[a, "episode_operational_return_std"]),
             "roots": int(agg.loc[a, "roots"]),
             "episodes": int(agg.loc[a, "episodes"]),
             "delivered_ratio": float(agg.loc[a, "delivered_ratio_mean"]),
             "sla_violations_demand_intervals": float(
                 agg.loc[a, "sla_violations_demand_intervals_mean"]),
             "reroutes_per_hour": float(agg.loc[a, "reroutes_per_hour_mean"]),
             "moved_mbps_total": float(agg.loc[a, "moved_mbps_total_mean"])}
            for a in agg.index
        ],
        "caption": (
            "Development / continuity evidence, evaluated on seeds 101-105. It selected "
            "the checkpoints. It is not final-holdout evidence and must not be read as "
            "generalization."
        ),
    }


def learning_curves(cont: Continuity) -> dict[str, Any]:
    """Per-root checkpoint curves. Selection happened here, never on the holdout."""
    series: list[dict[str, Any]] = []
    for root in identity.TRAINING_ROOTS:
        for algo in identity.LEARNER_ALGORITHMS:
            rows = cont.learning_curves[
                (cont.learning_curves.training_root.astype(str) == str(root))
                & (cont.learning_curves.algorithm == algo)
            ].sort_values("checkpoint_transition")
            selected = rows[rows["selected"].astype(str).str.lower() == "true"]
            series.append({
                "training_root": root,
                "algorithm": algo,
                "points": [
                    {"transition": int(r["checkpoint_transition"]),
                     "return": float(r["mean_operational_return"]),
                     "valid": str(r["valid"]).lower() == "true",
                     "selected": str(r["selected"]).lower() == "true"}
                    for _, r in rows.iterrows()
                ],
                "selected_transition": int(selected.checkpoint_transition.iloc[0])
                if not selected.empty else None,
            })
    return {
        "stage": identity.STAGE_DEVELOPMENT,
        "source_sha": identity.CONTINUATION_SOURCE_SHA,
        "series": series,
        "rule": "preregistered highest-valid-return checkpoint on continuity seeds 101-105",
        "caption": (
            "Development / continuity learning curves. These are not holdout results and "
            "no holdout episode influenced any point or any selection on this chart."
        ),
    }


def pilot_summary(pilot: Seed42) -> dict[str, Any]:
    """The seed-42 single-root pilot. Development stage."""
    comp = pilot.comparison.set_index("algorithm")
    return {
        "stage": identity.STAGE_DEVELOPMENT,
        "study": "seed-42 pilot",
        "source_sha": identity.SEED42_SOURCE_SHA,
        "artifact_path": str(pilot.directory),
        "methods": [
            {"algorithm": a,
             "operational_return_mean": float(comp.loc[a, "operational_return_mean"]),
             "operational_return_std": float(comp.loc[a, "operational_return_std"]),
             "episodes": int(comp.loc[a, "episodes"])}
            for a in comp.index
        ],
        "curve": [
            {"algorithm": r["algorithm"],
             "transition": int(r["checkpoint_transition"]),
             "return": float(r["mean_operational_return"]),
             "valid": str(r["valid"]).lower() == "true"}
            for _, r in pilot.learning_curve.iterrows()
        ],
        "caption": (
            "Single-training-root pilot on continuity seeds 101-105. Not a "
            "generalization claim and not holdout evidence."
        ),
    }


# ------------------------------------------------------------- disclosures
def disclosures(fh: FinalHoldout, cont: Continuity, pilot: Seed42) -> list[dict[str, Any]]:
    """Invalidated, superseded and repaired runs, stated rather than buried.

    Three distinct statuses, never collapsed: `invalidated` runs are scientifically
    void; `superseded` runs were valid and were replaced for source-identity reasons;
    `repaired` events are tooling problems fixed before the holdout.
    """
    out: list[dict[str, Any]] = []

    for key, item in (pilot.manifest.get("invalidated_or_superseded_runs") or {}).items():
        invalid = key.startswith("invalid")
        out.append({
            "kind": "invalidated" if invalid else "superseded",
            "stage": identity.STAGE_DEVELOPMENT,
            "study": "seed-42 pilot",
            "title": key.replace("_", " "),
            "summary": item["reason"],
            "path": item["path"],
            "preserved": True,
            "used_in_reported_results": False,
        })

    for item in cont.manifest.get("superseded_valid_runs") or []:
        out.append({
            "kind": "superseded",
            "stage": identity.STAGE_DEVELOPMENT,
            "study": "three-root continuity",
            "title": f"root 314159 {item['algorithm']} at {item['source_sha'][:7]}",
            "summary": item["reason"],
            "path": item["path"],
            "preserved": True,
            "used_in_reported_results": False,
        })

    for item in cont.manifest.get("failures") or []:
        out.append({
            "kind": "failed",
            "stage": identity.STAGE_DEVELOPMENT,
            "study": "three-root continuity",
            "title": item["stage"].replace("_", " "),
            "summary": item["reason"],
            "path": item.get("evidence", ""),
            "preserved": item["status"] != "failed_before_run_directory_creation",
            "used_in_reported_results": False,
        })

    for item in fh.manifest.get("pre_holdout_events") or []:
        out.append({
            "kind": "repaired",
            "stage": identity.STAGE_FINAL_HOLDOUT,
            "study": "final holdout",
            "title": item["stage"].replace("_", " "),
            "summary": item["reason"],
            "path": item.get("repair_commit", ""),
            "preserved": True,
            "used_in_reported_results": False,
        })

    return out
