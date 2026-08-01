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

from typing import Any

from mplssim.display import scenario_label

#: Process-scoped archive of runs whose session has been fully reset. Cleared on
#: restart by construction: it is a list in memory and nothing writes it to disk.
_PROCESS_RETAINED: list[dict[str, Any]] = []

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


__all__ = ["RECORD_CLASSES", "SEPARATION_RULE", "clear_process_retained",
           "hand_over", "process_retained", "results", "retained_runs", "run_row"]
