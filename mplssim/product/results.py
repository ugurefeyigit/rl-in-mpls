"""One results surface over three record classes that must never be merged.

The product shows results from three places, and the single most damaging thing
it could do is average them:

1. ``live_demonstration`` — the run on screen right now. One seed, one pass,
   unaudited, still moving.
2. ``retained_demonstration`` — runs archived by *Reset run* in this server
   process. Same status as the live run, just finished.
3. ``governed_evidence`` — the closed V2 study's frozen record. It is read
   through ``/api/v2/*`` and **is not loaded here at all**; this module emits a
   pointer to it and the reason it stays separate.

So the payload is three labelled sections with three separate grains, never one
table, never one mean. Every non-evidence row carries ``is_evidence: False`` and
a reason, and no cross-class aggregate exists anywhere in this module — there is
no function here that could compute one.

Retention lifetime (decided in Part 2, recorded in docs/ADR-003):

- a **reset run** archives the run it replaces;
- a **full reset** hands the session's archive to the process-level store, so
  the results surface survives returning to the configuration form;
- a **server restart** drops everything, deliberately. Persisting a
  demonstration number to disk is the first step towards someone citing it, and
  these numbers are not evidence. Anything worth keeping goes through
  ``POST /api/export/save-run``, which labels its rows ``live``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mplssim.display import scenario_label
from mplssim.evidence.identity import REWARD_COMPONENTS

#: Process-scoped archive of runs whose session has been fully reset. Cleared on
#: restart by construction: it is a list in memory and nothing writes it to disk.
_PROCESS_RETAINED: list[dict[str, Any]] = []

#: Exp 2.1's deliberately tiny completed-run holder. Slots contain normalized
#: copies, not runners or model objects, and disappear with this process.
_COMPARISON_SLOTS: dict[str, dict[str, Any] | None] = {"a": None, "b": None}

_HEADLINE_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("operational_return", "Operational return", "signed operational return", "higher"),
    ("delivery", "Mean delivery", "percent", "higher"),
    ("sla_risk", "Peak SLA risk", "violating demands", "lower"),
    ("peak_utilization", "Peak utilization", "percent", "lower"),
    ("mean_utilization", "Mean utilization", "percent", "lower"),
    ("reroutes", "Accepted TE changes", "changes", "lower"),
    ("reversals", "TE reversals", "reversals", "lower"),
    ("flaps", "Route flaps", "flaps", "lower"),
    ("moved_bandwidth", "Moved bandwidth", "Mbps", "lower"),
)

RECORD_CLASSES: dict[str, dict[str, str]] = {
    "live_demonstration": {
        "label": "Live run",
        "grain": "One scenario, one seed, one pass, still running.",
        "reason": "A live demonstration is behaviour you can watch, not a result "
                  "you can cite.",
    },
    "retained_demonstration": {
        "label": "Earlier runs kept in this session",
        "grain": "One scenario, one seed, one pass each, finished.",
        "reason": "Archived by Reset run inside this server process. Unaudited, "
                  "not selected for anything, and dropped on restart.",
    },
    "governed_evidence": {
        "label": "Closed V2 study",
        "grain": "Five holdout seeds per scenario, three training roots, "
                 "evaluated once after the study closed to selection.",
        "reason": "The only class of number in this product that supports a "
                  "conclusion. Read from the frozen artifacts through /api/v2/*.",
    },
}

SEPARATION_RULE = (
    "These three classes are reported separately and are never averaged, ranked "
    "against each other or placed in one table. A demonstration and a holdout "
    "result answer different questions, and combining them would make the "
    "weaker one look like the stronger one.")


# ------------------------------------------------------------------ summaries
def _movement_totals(history: list[dict[str, Any]]) -> dict[str, int]:
    """Cumulative movement counters, each kept under its own name."""
    keys = ("accepted_te_changes", "rejected_te_requests", "te_reversals",
            "frr_changes", "frr_disconnections", "recovery_restorations",
            "reroutes", "flaps", "frr_events")
    totals: dict[str, int] = {}
    for record in history:
        for key in keys:
            value = record["metrics"].get(key)
            if value is None:
                continue
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _mean(history: list[dict[str, Any]], key: str) -> float | None:
    values = [record["metrics"][key] for record in history
              if record["metrics"].get(key) is not None]
    return round(sum(float(v) for v in values) / len(values), 6) if values else None


def _peak(history: list[dict[str, Any]], key: str) -> float | None:
    values = [record["metrics"][key] for record in history
              if record["metrics"].get(key) is not None]
    return round(max(float(v) for v in values), 6) if values else None


def run_row(algorithm: str, checkpoint_id: str | None,
            cumulative_reward: float, history: list[dict[str, Any]],
            record_class: str) -> dict[str, Any]:
    """One controller's pass, summarized from its own recorded history.

    Every number here comes from the same per-interval records the exports and
    the scoreboard read, so a displayed value and an exported value cannot
    disagree.
    """
    return {
        "record_class": record_class,
        "is_evidence": False,
        "evidence_reason": RECORD_CLASSES[record_class]["reason"],
        "algorithm": algorithm,
        "checkpoint_id": checkpoint_id,
        "steps": len(history),
        "operational_return": round(float(cumulative_reward), 4),
        "return_unit": "signed operational return",
        "max_util_peak": _peak(history, "max_util"),
        "max_util_mean": _mean(history, "max_util"),
        "delivered_ratio_mean": _mean(history, "delivered_ratio"),
        "mean_delay_ms": _mean(history, "mean_delay_ms"),
        "loss_ratio_mean": _mean(history, "loss_ratio"),
        "sla_violations_peak": _peak(history, "sla_violations"),
        "disconnected_peak": _peak(history, "disconnected_demands"),
        "movement": _movement_totals(history),
    }


def _archive_rows(archive: dict[str, Any], record_class: str) -> dict[str, Any]:
    return {
        "generation": archive.get("generation"),
        "environment": archive.get("environment"),
        "scenario": archive.get("scenario"),
        "seed": archive.get("seed"),
        "training_root": archive.get("training_root"),
        "steps": archive.get("steps"),
        "runs": [run_row(run["algorithm"], run.get("checkpoint_id"),
                         run["cumulative_reward"], run.get("history", []),
                         record_class)
                 for run in archive.get("runs", [])],
    }


# -------------------------------------------------------------------- process
def hand_over(session: Any) -> int:
    """Move a session's archive into the process store on full reset.

    Returns how many run groups were handed over. The live run at the moment of
    the reset is archived too, so pressing Full reset does not silently discard
    what is on screen.
    """
    handed = list(session.previous_runs)
    final = session.archive()
    if final is not None:
        handed.append(final)
    _PROCESS_RETAINED.extend(handed)
    return len(handed)


def process_retained() -> list[dict[str, Any]]:
    return list(_PROCESS_RETAINED)


def clear_process_retained() -> None:
    """Drop the process store. Used by tests and by nothing else."""
    _PROCESS_RETAINED.clear()


# ---------------------------------------------------------- Exp 2.1 A/B store
def clear_comparison_runs() -> None:
    """Clear both completed-run slots. Full Reset calls this explicitly."""
    _COMPARISON_SLOTS.update(a=None, b=None)


def clear_comparison_slot(slot: str) -> None:
    slot = _slot_name(slot)
    _COMPARISON_SLOTS[slot] = None


def swap_comparison_slots() -> None:
    _COMPARISON_SLOTS["a"], _COMPARISON_SLOTS["b"] = (
        _COMPARISON_SLOTS["b"], _COMPARISON_SLOTS["a"])


def _slot_name(slot: str) -> str:
    normalized = str(slot).lower()
    if normalized not in _COMPARISON_SLOTS:
        raise ValueError("comparison slot must be 'a' or 'b'")
    return normalized


def _archive_inventory(session: Any | None) -> list[dict[str, Any]]:
    archives = list(session.previous_runs if session else []) + list(_PROCESS_RETAINED)
    if session is not None:
        current = session.archive()
        if current is not None:
            archives.append(current)
    return archives


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metric_values(history: list[dict[str, Any]], key: str) -> list[float | None]:
    return [_number(row.get("metrics", {}).get(key)) for row in history]


def _series(history: list[dict[str, Any]], key: str, unit: str,
            *, source: str = "metrics") -> dict[str, Any]:
    values = []
    missing = 0
    for row in history:
        raw = row.get(key) if source == "record" else row.get("metrics", {}).get(key)
        if raw is None:
            missing += 1
            continue
        values.append({"step": int(row["step"]), "t_min": float(row["t_min"]),
                       "value": float(raw)})
    return {
        "available": bool(values) and missing == 0,
        "reason": (None if bool(values) and missing == 0 else
                   f"{key} was not recorded for every completed interval."),
        "unit": unit,
        "values": values,
    }


def _reward_series(history: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    cumulative = 0.0
    for row in history:
        reward = float(row["reward"])
        cumulative += reward
        values.append({"step": int(row["step"]), "t_min": float(row["t_min"]),
                       "value": reward, "cumulative": cumulative})
    return {"available": bool(values), "reason": None if values else
            "No interval reward was recorded.",
            "unit": "signed operational return", "values": values}


def _component_totals(history: list[dict[str, Any]], environment: str) -> dict[str, Any]:
    names = REWARD_COMPONENTS if environment == "v2" else tuple(dict.fromkeys(
        name for row in history for name in row.get("components", {})))
    out: dict[str, Any] = {}
    for name in names:
        values = []
        missing = 0
        for row in history:
            if name not in row.get("components", {}):
                missing += 1
                continue
            values.append({"step": int(row["step"]), "t_min": float(row["t_min"]),
                           "value": float(row["components"][name])})
        out[name] = {
            "available": bool(values) and missing == 0,
            "reason": (None if bool(values) and missing == 0 else
                       f"{name} was not recorded for every interval."),
            "unit": "reward contribution",
            "total": sum(row["value"] for row in values) if values else None,
            "values": values,
        }
    return out


def _decision_series(history: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in history:
        decision = row.get("decision")
        if decision is None:
            continue
        decoded = decision.get("decoded") or {}
        moves = decision.get("moves") or []
        action = decision.get("action")
        accepted = bool(decoded.get("accepted")) or any(
            move.get("accepted") for move in moves)
        is_noop = action == 0 or (action is None and not moves)
        object_id = decoded.get("demand")
        if object_id is None:
            object_id = next((move.get("demand") for move in moves
                              if move.get("accepted")), None)
        values.append({
            "step": int(row["step"]), "t_min": float(row["t_min"]),
            "action": action, "accepted": accepted, "is_noop": is_noop,
            "object_type": "demand" if object_id else None,
            "object_id": object_id,
            "moved_mbps": _number(row.get("moved_mbps")),
        })
    return {
        "available": len(values) == len(history) and bool(history),
        "reason": (None if len(values) == len(history) and history else
                   "Per-interval actions were not recorded for this completed run."),
        "unit": "decision per interval", "values": values,
    }


def _timeline(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    failed_before: set[str] = set()
    congestion_before = 0
    sla_before = 0
    for row in history:
        step, t_min = int(row["step"]), float(row["t_min"])
        decision = row.get("decision") or {}
        decoded = decision.get("decoded") or {}
        moves = decision.get("moves") or []
        action = decision.get("action")
        accepted_moves = [move for move in moves if move.get("accepted")]
        if (action not in (None, 0) and decoded.get("accepted")) or accepted_moves:
            object_id = decoded.get("demand") or accepted_moves[0].get("demand")
            events.append({"id": f"action:{step}:{object_id or '-'}", "kind": "action",
                           "step": step, "t_min": t_min,
                           "title": f"Accepted TE action{f' for {object_id}' if object_id else ''}",
                           "object_type": "demand" if object_id else None,
                           "object_id": object_id})
        metrics = row.get("metrics", {})
        for kind, key, label in (("reversal", "te_reversals", "TE reversal"),
                                 ("flap", "flaps", "Route flap")):
            if int(metrics.get(key, 0) or 0) > 0:
                events.append({"id": f"{kind}:{step}:-", "kind": kind,
                               "step": step, "t_min": t_min, "title": label,
                               "object_type": None, "object_id": None})
        failed_now = set(row.get("failed_links") or [])
        for link_id in sorted(failed_now - failed_before):
            events.append({"id": f"failure:{step}:{link_id}", "kind": "failure",
                           "step": step, "t_min": t_min,
                           "title": f"Link failure: {link_id}",
                           "object_type": "link", "object_id": link_id})
        for link_id in sorted(failed_before - failed_now):
            events.append({"id": f"recovery:{step}:{link_id}", "kind": "recovery",
                           "step": step, "t_min": t_min,
                           "title": f"Link recovery: {link_id}",
                           "object_type": "link", "object_id": link_id})
        congestion_now = int(metrics.get("congested_links", 0) or 0)
        if congestion_now and not congestion_before:
            events.append({"id": f"congestion:{step}:-", "kind": "congestion",
                           "step": step, "t_min": t_min,
                           "title": "Congestion threshold crossed",
                           "object_type": None, "object_id": None})
        sla_now = int(metrics.get("sla_violations", 0) or 0)
        if sla_now and not sla_before:
            events.append({"id": f"sla_risk:{step}:-", "kind": "sla_risk",
                           "step": step, "t_min": t_min,
                           "title": "SLA risk recorded",
                           "object_type": None, "object_id": None})
        failed_before, congestion_before, sla_before = failed_now, congestion_now, sla_now
    return events


def _sum_metric(history: list[dict[str, Any]], *keys: str) -> float | None:
    values: list[float] = []
    for row in history:
        metrics = row.get("metrics", {})
        value = next((metrics.get(key) for key in keys if metrics.get(key) is not None), None)
        if value is not None:
            values.append(float(value))
    return sum(values) if values else None


def _mean_metric(history: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in _metric_values(history, key) if value is not None]
    return sum(values) / len(values) if values else None


def _max_metric(history: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in _metric_values(history, key) if value is not None]
    return max(values) if values else None


def _normalized_run(archive: dict[str, Any], run: dict[str, Any],
                    run_index: int, archive_index: int) -> dict[str, Any]:
    environment = str(archive.get("environment") or "unknown")
    history = deepcopy(run.get("history", []))
    checkpoint = deepcopy(run.get("checkpoint_provenance") or {})
    session_id = archive.get("session_id") or f"retained-{archive_index}"
    run_id = (f"{session_id}:{archive.get('generation', 0)}:{run_index}:"
              f"{environment}:{archive.get('scenario')}:{archive.get('seed')}:"
              f"{run.get('algorithm')}")
    moved_values = [_number(row.get("moved_mbps")) for row in history]
    moved_total = (sum(value for value in moved_values if value is not None)
                   if moved_values and all(value is not None for value in moved_values)
                   else None)
    identity = {
        "environment": environment,
        "scenario": archive.get("scenario"),
        "seed": archive.get("seed"),
        "controller": run.get("algorithm"),
        "training_root": archive.get("training_root") if environment == "v2" else None,
        "checkpoint_id": run.get("checkpoint_id"),
        "checkpoint_sha256": checkpoint.get("payload_sha256"),
    }
    series = {
        "reward": _reward_series(history),
        "utilization": {
            "unit": "percent",
            "max": _series(history, "max_util", "percent"),
            "mean": _series(history, "mean_util", "percent"),
        },
        "delivery": _series(history, "delivered_ratio", "percent"),
        "sla_risk": _series(history, "sla_violations", "violating demands per interval"),
        "loss": _series(history, "loss_ratio", "percent"),
        "moved_bandwidth": _series(history, "moved_mbps", "Mbps", source="record"),
        "decisions": _decision_series(history),
    }
    return {
        "run_id": run_id,
        "identity": identity,
        "label": (f"{run.get('algorithm')} · {environment.upper()} · "
                  f"{scenario_label(str(archive.get('scenario')))} · seed {archive.get('seed')}"),
        "provenance": {
            "source_kind": "live_session",
            "record_class": "retained_demonstration",
            "state": "completed",
            "is_evidence": False,
            "session_id": session_id,
            "generation": archive.get("generation"),
            "checkpoint": checkpoint or None,
            "synchronization_fields": {
                "environment": environment, "scenario": archive.get("scenario"),
                "seed": archive.get("seed"),
                "steps": [int(row["step"]) for row in history],
                "t_min": [float(row["t_min"]) for row in history],
            },
        },
        "steps": len(history),
        "summary": {
            "operational_return": _number(run.get("cumulative_reward")),
            "delivery": _mean_metric(history, "delivered_ratio"),
            "sla_risk": _max_metric(history, "sla_violations"),
            "peak_utilization": _max_metric(history, "max_util"),
            "mean_utilization": _mean_metric(history, "mean_util"),
            "reroutes": _sum_metric(history, "accepted_te_changes", "reroutes"),
            "reversals": _sum_metric(history, "te_reversals"),
            "flaps": _sum_metric(history, "flaps"),
            "moved_bandwidth": moved_total,
        },
        "series": series,
        "reward_components": _component_totals(history, environment),
        "timeline": _timeline(history),
        "history": history,
    }


def _completed_candidates(session: Any | None) -> list[dict[str, Any]]:
    candidates = []
    for archive_index, archive in enumerate(_archive_inventory(session)):
        if archive.get("completed") is not True:
            continue
        for run_index, run in enumerate(archive.get("runs", [])):
            candidates.append(_normalized_run(archive, run, run_index, archive_index))
    return candidates


def assign_comparison_slot(session: Any | None, slot: str, run_id: str) -> None:
    slot = _slot_name(slot)
    candidates = {run["run_id"]: run for run in _completed_candidates(session)}
    if run_id not in candidates:
        raise KeyError(f"completed run {run_id!r} is not available")
    other = "b" if slot == "a" else "a"
    other_run = _COMPARISON_SLOTS[other]
    if other_run is not None and other_run.get("run_id") == run_id:
        raise ValueError("the same completed run cannot occupy both A and B")
    _COMPARISON_SLOTS[slot] = deepcopy(candidates[run_id])


def _pairing(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    if not a or not b:
        return {"available": False, "synchronized": False,
                "paired_conclusions": False,
                "reason": "Choose one completed run for A and one for B.",
                "mismatched_fields": []}
    fields_a = a["provenance"]["synchronization_fields"]
    fields_b = b["provenance"]["synchronization_fields"]
    mismatched = [key for key in ("environment", "scenario", "seed", "steps", "t_min")
                  if fields_a.get(key) != fields_b.get(key)]
    synchronized = not mismatched
    return {
        "available": True, "synchronized": synchronized,
        "paired_conclusions": synchronized,
        "reason": ("The completed runs share environment, scenario, seed and interval grid."
                   if synchronized else
                   "These are two completed-run results, but their intervals are not paired. "
                   "Interval-by-interval causal conclusions are disabled."),
        "mismatched_fields": mismatched,
    }


def _leader(a: float | None, b: float | None, direction: str) -> str | None:
    if a is None or b is None:
        return None
    if a == b:
        return "equal"
    a_better = a > b if direction == "higher" else a < b
    return "a" if a_better else "b"


def _headline(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not a or not b:
        return []
    rows = []
    for key, label, unit, direction in _HEADLINE_METRICS:
        value_a = a["summary"].get(key)
        value_b = b["summary"].get(key)
        leader = _leader(value_a, value_b, direction)
        rows.append({
            "id": key, "label": label, "unit": unit, "direction": direction,
            "a": value_a, "b": value_b,
            "delta": value_a - value_b if value_a is not None and value_b is not None else None,
            "leader": leader,
            "a_outcome": ("unavailable" if leader is None else
                          "equal" if leader == "equal" else
                          "better" if leader == "a" else "worse"),
            "b_outcome": ("unavailable" if leader is None else
                          "equal" if leader == "equal" else
                          "better" if leader == "b" else "worse"),
        })
    return rows


def comparative_runs(session: Any | None) -> dict[str, Any]:
    candidates = _completed_candidates(session)
    a, b = _COMPARISON_SLOTS["a"], _COMPARISON_SLOTS["b"]
    return {
        "slots": {"a": deepcopy(a), "b": deepcopy(b)},
        "candidates": [{key: deepcopy(run[key]) for key in
                        ("run_id", "identity", "label", "provenance", "steps")}
                       for run in candidates],
        "pairing": _pairing(a, b),
        "headline": _headline(a, b),
        "lifetime": ("A and B are bounded process-memory slots. They disappear on Full Reset "
                     "or server restart and are never written to evidence."),
    }


# -------------------------------------------------------------------- payload
def retained_runs(session: Any | None) -> dict[str, Any]:
    """`GET /api/simulation/retained-runs`, session store plus process store."""
    session_runs = [_archive_rows(a, "retained_demonstration")
                    for a in (session.previous_runs if session else [])]
    process_runs = [_archive_rows(a, "retained_demonstration")
                    for a in _PROCESS_RETAINED]
    return {
        "count": len(session_runs) + len(process_runs),
        "session_count": len(session_runs),
        "process_count": len(process_runs),
        "runs": session_runs + process_runs,
        "session_runs": session_runs,
        "process_runs": process_runs,
        "lifetime": ("Reset run archives to the session. Full reset hands the "
                     "session's archive to this server process. A restart drops "
                     "everything: a demonstration number is never persisted."),
        "record_class": "retained_demonstration",
        "is_evidence": False,
        "evidence_reason": RECORD_CLASSES["retained_demonstration"]["reason"],
    }


def _live_section(session: Any | None) -> dict[str, Any]:
    if session is None:
        return {"available": False,
                "reason": "No run is loaded. Start one from the control panel.",
                "runs": []}
    if not any(runner.history for runner in session.runners):
        return {"available": False,
                "reason": "The loaded run has not completed an interval yet.",
                "runs": []}
    status = session.status()
    return {
        "available": True,
        "session_id": status["session_id"],
        "generation": status["generation"],
        "environment": status["environment"],
        "scenario": status["scenario"],
        "scenario_label": scenario_label(status["scenario"]),
        "seed": status["seed"],
        "training_root": status.get("training_root"),
        "execution": status["execution"],
        "steps": status["step"],
        "runs": [run_row(r.algorithm, getattr(r, "checkpoint_id", None),
                         r.cumulative_reward, r.history, "live_demonstration")
                 for r in session.runners],
    }


def results(session: Any | None) -> dict[str, Any]:
    """`GET /api/product/results` — three classes, three sections, no merge."""
    return {
        "record_classes": RECORD_CLASSES,
        "separation_rule": SEPARATION_RULE,
        "live": {**_live_section(session),
                 "record_class": "live_demonstration",
                 "is_evidence": False,
                 "evidence_reason": RECORD_CLASSES["live_demonstration"]["reason"]},
        "retained": retained_runs(session),
        "study": {
            "record_class": "governed_evidence",
            "is_evidence": True,
            # The frozen numbers are not copied here. A literal in this file
            # would drift from the artifacts silently, and this module has no
            # authority over the study's record.
            "loaded_here": False,
            "read_from": [
                "/api/v2/final-holdout", "/api/v2/final-scenarios",
                "/api/v2/final-reward-components", "/api/v2/final-actions",
                "/api/v2/final-integrity", "/api/v2/final-provenance",
                "/api/v2/development-continuity", "/api/v2/development-seed42",
                "/api/v2/disclosures",
            ],
            "reason": RECORD_CLASSES["governed_evidence"]["reason"],
            "grain": RECORD_CLASSES["governed_evidence"]["grain"],
        },
        "comparable": False,
        "comparable_reason": SEPARATION_RULE,
    }


__all__ = ["RECORD_CLASSES", "SEPARATION_RULE", "assign_comparison_slot",
           "clear_comparison_runs", "clear_comparison_slot", "clear_process_retained",
           "comparative_runs", "hand_over", "process_retained", "results",
           "retained_runs", "run_row", "swap_comparison_slots"]
