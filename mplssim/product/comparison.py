"""The paired decision comparison.

`pairing.synchronization` answers *may* we compare. This module answers *what*
the comparison actually shows, once the answer is yes.

The rules it keeps are the ones that separate a comparison from an advert:

- **No verdict without a proof.** When `pairing` refuses, this module produces
  lanes with no comparative field at all — not a greyed-out verdict, not a
  verdict with a caveat. The refusal is the payload.
- **No percentage on a signed score.** Operational return is a signed simulation
  score; a ratio of two signed numbers is meaningless and is never computed. The
  difference is reported as a signed absolute gap, in the score's own units.
- **A lead is not a conclusion.** One paired live run at one seed is a
  demonstration. Every verdict here carries `is_evidence: False` and the reason,
  and the frozen study record is never merged into it.
- **Movement is separated from protection.** V2 keeps controller TE changes, FRR
  protection moves and post-recovery restorations in three counters. They are
  reported as three counters here too, and never summed into "reroutes".
"""

from __future__ import annotations

from typing import Any

from mplssim.display import CITY_NAMES
from mplssim.product import pairing

#: Metrics compared row by row. `better` says which direction is an improvement,
#: or None where the product refuses to call a direction good or bad.
COMPARED_METRICS: tuple[tuple[str, str, str, str | None], ...] = (
    ("max_util", "Busiest link", "share", "lower"),
    ("mean_util", "Mean link load", "share", None),
    ("delivered_ratio", "Delivered traffic", "share", "higher"),
    ("sla_violations", "Demands violating SLA now", "count", "lower"),
    ("sla_violation_fraction", "Share of demands violating SLA", "share", "lower"),
    ("congested_links", "Congested directed links", "count", "lower"),
    ("disconnected_demands", "Disconnected demands", "count", "lower"),
    ("protected_disconnected_demands", "Protected demands disconnected", "count",
     "lower"),
    ("mean_delay_ms", "Mean demand delay", "ms", "lower"),
    ("loss_ratio", "Loss ratio", "share", "lower"),
    ("overload_ratio", "Offered load above capacity", "share", "lower"),
)

#: Movement counters, kept apart on purpose. V1 has one `reroutes` counter; V2
#: has three, and collapsing them would misattribute protection to the policy.
MOVEMENT_COUNTERS: tuple[tuple[str, str, str], ...] = (
    ("accepted_te_changes", "Controller TE changes", "controller"),
    ("rejected_te_requests", "TE requests rejected", "controller"),
    ("te_reversals", "TE reversals", "controller"),
    ("frr_changes", "FRR protection moves", "protection"),
    ("frr_disconnections", "FRR disconnections", "protection"),
    ("recovery_restorations", "Restorations after recovery", "recovery"),
    ("reroutes", "TE reroutes (V1 counter)", "controller"),
    ("flaps", "Flaps (V1 counter)", "controller"),
    ("frr_events", "FRR events (V1 counter)", "protection"),
)

DEMONSTRATION_NOTE = (
    "This is one paired live run at one seed. It demonstrates behaviour; it is "
    "not a holdout result and never becomes one. The study's conclusions live in "
    "the frozen final-holdout record and are read only from there.")

NO_PERCENTAGE_NOTE = (
    "Operational return is a signed simulation score, so the gap is reported in "
    "score units. No percentage difference is computed from signed values.")


# --------------------------------------------------------------------- lanes
def _action_summary(runner: Any) -> dict[str, Any]:
    """What this lane did in the interval it last completed."""
    decision = runner.last_decision
    if not decision:
        return {"available": False,
                "reason": "This lane has not completed an interval yet.",
                "kind": None, "text": "—"}
    if "action" in decision:
        action = int(decision["action"])
        decoded = decision.get("decoded") or {}
        if action == 0:
            text = "No TE change"
        elif decoded.get("demand"):
            # V1's decoded action carries src/dst; V2's does not. The endpoints
            # are named only when the payload actually has them — a city pair
            # invented here would be a fabricated fact about the move.
            endpoints = ""
            if decoded.get("src") and decoded.get("dst"):
                endpoints = (f"{CITY_NAMES.get(decoded['src'], decoded['src'])} → "
                             f"{CITY_NAMES.get(decoded['dst'], decoded['dst'])} ")
            text = (f"{endpoints}{decoded['demand']} "
                    f"to candidate {decoded.get('path_idx')}")
        else:
            text = f"Action {action}"
        return {
            "available": True,
            "kind": "single_action",
            "action": action,
            "is_noop": action == 0,
            "text": text,
            "decoded": decoded,
            "accepted": decoded.get("accepted"),
            "validator_reason": decoded.get("reason"),
            "explanation": decision.get("explanation"),
        }
    moves = decision.get("moves") or []
    return {
        "available": True,
        "kind": "baseline_moves",
        "action": None,
        "is_noop": not moves,
        "n_moves": len(moves),
        "text": (", ".join(f"{m['demand']}→p{m['path_idx']}" for m in moves)
                 if moves else "No TE change"),
        "moves": moves,
        "accepted": (all(m.get("accepted") for m in moves) if moves else None),
        "validator_reason": next((m.get("reason") for m in moves
                                  if not m.get("accepted")), None),
        "explanation": decision.get("explanation"),
    }


def _movement_totals(runner: Any) -> dict[str, Any]:
    """Cumulative movement over the run, counter by counter, never summed."""
    rows: dict[str, dict[str, Any]] = {}
    for key, label, attribution in MOVEMENT_COUNTERS:
        total = 0
        seen = False
        for record in runner.history:
            value = record["metrics"].get(key)
            if value is None:
                continue
            seen = True
            total += int(value)
        if seen:
            rows[key] = {"label": label, "attribution": attribution, "total": total}
    return rows


def _latest_metrics(runner: Any) -> dict[str, Any]:
    history = runner.eng.metrics_history
    return dict(history[-1]) if history else {}


def _reward_components(runner: Any) -> dict[str, Any]:
    decision = runner.last_decision or {}
    components = decision.get("components") or {}
    order = list(decision.get("component_order") or components.keys())
    return {
        "available": bool(components),
        "order": order,
        "values": {name: components[name] for name in order if name in components},
        "reason": None if components else "No interval has been rewarded yet.",
    }


def lane(runner: Any, index: int) -> dict[str, Any]:
    """One side of the comparison, complete and self-describing."""
    return {
        "position": "a" if index == 0 else "b",
        "token": "A" if index == 0 else "B",
        "algorithm": runner.algorithm,
        "environment_version": getattr(runner, "environment_version", "v1"),
        "family": ("learner" if getattr(runner, "checkpoint", None) is not None
                   or runner.algorithm == "rl" else "baseline"),
        "checkpoint_id": getattr(runner, "checkpoint_id", None),
        "output_semantics": getattr(runner, "output_semantics", "none"),
        "step": int(runner.eng.step_count),
        "steps_recorded": len(runner.history),
        "cumulative_reward": round(float(runner.cumulative_reward), 4),
        "interval_reward": (round(float(runner.history[-1]["reward"]), 6)
                            if runner.history else None),
        "action": _action_summary(runner),
        "metrics": _latest_metrics(runner),
        "reward_components": _reward_components(runner),
        "movement": _movement_totals(runner),
        # Kept for the pre-existing lane_details consumers.
        "last_decision": runner.last_decision,
    }


# ----------------------------------------------------------------- divergence
def _first_divergent_step(runners: list[Any]) -> dict[str, Any]:
    """The earliest completed interval where the two lanes chose differently.

    Read from each lane's own recorded history, so it is a measurement rather
    than a claim reconstructed after the fact.
    """
    a, b = runners[0], runners[1]
    depth = min(len(a.history), len(b.history))
    if depth == 0:
        return {"available": False,
                "reason": "Neither lane has completed an interval yet."}
    for i in range(depth):
        moved_a = _moved_in(a, i)
        moved_b = _moved_in(b, i)
        if moved_a != moved_b:
            return {
                "available": True,
                "step": int(a.history[i]["step"]),
                "t_min": float(a.history[i]["t_min"]),
                "a_moved": moved_a,
                "b_moved": moved_b,
                "note": ("The first interval in which the two lanes made "
                         "different movement decisions. Everything before it is "
                         "identical by construction."),
            }
    return {"available": False,
            "reason": (f"The lanes have made the same movement decision in every "
                       f"one of the {depth} completed interval(s) so far.")}


def _moved_in(runner: Any, index: int) -> bool:
    """Did this lane's controller accept a TE change in interval `index`?"""
    metrics = runner.history[index]["metrics"]
    if "accepted_te_changes" in metrics:
        return int(metrics["accepted_te_changes"]) > 0
    return int(metrics.get("reroutes", 0)) > 0


def _metric_rows(runners: list[Any]) -> list[dict[str, Any]]:
    a_metrics, b_metrics = _latest_metrics(runners[0]), _latest_metrics(runners[1])
    rows = []
    for key, label, unit, better in COMPARED_METRICS:
        if key not in a_metrics or key not in b_metrics:
            continue
        a_value, b_value = float(a_metrics[key]), float(b_metrics[key])
        gap = round(a_value - b_value, 6)
        if better is None or gap == 0:
            leader = None
        elif better == "lower":
            leader = "a" if a_value < b_value else "b"
        else:
            leader = "a" if a_value > b_value else "b"
        rows.append({
            "key": key, "label": label, "unit": unit, "better": better,
            "a": a_value, "b": b_value, "gap": gap, "leader": leader,
        })
    return rows


def _return_verdict(runners: list[Any]) -> dict[str, Any]:
    a, b = runners[0], runners[1]
    a_total = round(float(a.cumulative_reward), 4)
    b_total = round(float(b.cumulative_reward), 4)
    gap = round(a_total - b_total, 4)
    if gap == 0:
        leader, sentence = None, "Both lanes are level on cumulative return."
    else:
        winner = a if gap > 0 else b
        leader = "a" if gap > 0 else "b"
        sentence = (f"{winner.algorithm} leads this run by {abs(gap):.4f} points "
                    f"of cumulative operational return.")
    return {
        "a": a_total, "b": b_total, "gap": gap, "leader": leader,
        "sentence": sentence,
        "unit": "signed operational return",
        "percentage": None,
        "percentage_reason": NO_PERCENTAGE_NOTE,
        "is_evidence": False,
        "evidence_reason": DEMONSTRATION_NOTE,
    }


# -------------------------------------------------------------------- payload
def comparison_state(session: Any) -> dict[str, Any]:
    """`GET /api/simulation/comparison` and the `comparison` block of a moment."""
    state = pairing.synchronization(session)
    runners = list(session.runners)
    state["lane_details"] = [{
        "algorithm": runner.algorithm,
        "last_decision": runner.last_decision,
        "cumulative_reward": round(float(runner.cumulative_reward), 4),
    } for runner in runners]
    state["compared_metric_keys"] = [key for key, _, _, _ in COMPARED_METRICS]
    state["movement_counter_keys"] = [key for key, _, _ in MOVEMENT_COUNTERS]
    state["demonstration_note"] = DEMONSTRATION_NOTE

    if not state.get("comparison"):
        state["detail"] = {
            "available": False,
            "reason": state.get("reason"),
            "lanes": [lane(runner, i) for i, runner in enumerate(runners)],
        }
        return state

    if not state.get("matched"):
        # A refusal is the payload. No verdict, no metric table, no gap — a
        # reader must not be able to lift a number out of a broken comparison.
        state["detail"] = {
            "available": False,
            "reason": state.get("reason"),
            "mismatched_fields": list(state.get("mismatched_fields") or []),
            "proof": state.get("proof"),
            "lanes": [{k: v for k, v in lane(runner, i).items()
                       if k not in ("metrics", "movement", "reward_components")}
                      for i, runner in enumerate(runners)],
        }
        return state

    state["detail"] = {
        "available": True,
        "proof": state.get("proof"),
        "reason": state.get("reason"),
        "lanes": [lane(runner, i) for i, runner in enumerate(runners)],
        "metric_rows": _metric_rows(runners),
        "divergence": _first_divergent_step(runners),
        "verdict": _return_verdict(runners),
        "movement_note": (
            "Controller TE changes, FRR protection moves and post-recovery "
            "restorations are three separate counters. They are never summed, "
            "because protection is not a policy decision."),
    }
    return state


__all__ = ["COMPARED_METRICS", "DEMONSTRATION_NOTE", "MOVEMENT_COUNTERS",
           "NO_PERCENTAGE_NOTE", "comparison_state", "lane"]
