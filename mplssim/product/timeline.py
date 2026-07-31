"""A typed incident timeline derived from what the session actually recorded.

Nothing here invents an event. Every entry traces to an engine action-log
record, a metric crossing the engine's own congestion threshold, a scenario
event that really fired, or an operator decision that really happened.

FRR is labelled as built-in protection, separately from TE actions a controller
chose. Conflating the two would credit a learner with the engine's own local
repair.
"""

from __future__ import annotations

from typing import Any

from mplssim.display import CITY_NAMES, link_label
from mplssim.sim import models as m

#: Event kinds the product understands. A component that meets an unknown kind
#: renders it as a generic event rather than dropping it.
EVENT_KINDS: tuple[str, ...] = (
    "congestion", "sla_risk", "failure", "frr", "recommendation",
    "action", "reversal", "flap", "recovery", "stabilization",
)

_SOURCE_LABELS = {
    "frr": "Built-in FRR protection",
    "manual": "Operator intervention",
    "rl": "Policy TE action",
    "greedy": "Greedy controller TE action",
    "cspf": "CSPF controller TE action",
    "static": "Static controller TE action",
}


def _clock(hour: float) -> str:
    total = int(round(hour * 60)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _event(kind: str, step: int, t_min: float, hour: float, title: str,
           detail: str, object_type: str | None, object_id: str | None,
           **extra: Any) -> dict[str, Any]:
    return {
        "id": f"{kind}:{step}:{object_id or '-'}",
        "kind": kind,
        "step": int(step),
        "t_min": round(float(t_min), 3),
        "clock": _clock(hour),
        "title": title,
        "detail": detail,
        "object_type": object_type,
        "object_id": object_id,
        **extra,
    }


def runner_events(runner: Any) -> list[dict[str, Any]]:
    engine = runner.eng
    start_hour = engine.scenario.start_hour
    events: list[dict[str, Any]] = []

    def hour_at(t_min: float) -> float:
        return (start_hour + t_min / 60.0) % 24.0

    # --- controller and protection actions, straight from the engine log
    for record in engine.action_log:
        source = record.source
        is_frr = source == "frr"
        kind = "frr" if is_frr else "action"
        demand = engine.demand_by_id.get(record.demand_id)
        endpoints = (f"{CITY_NAMES.get(demand.src, demand.src)} → "
                     f"{CITY_NAMES.get(demand.dst, demand.dst)}") if demand else record.demand_id
        events.append(_event(
            kind, record.step, record.t_min, hour_at(record.t_min),
            title=(f"{_SOURCE_LABELS.get(source, source)}: {record.demand_id}"),
            detail=(f"{endpoints} moved from path {record.from_path} to "
                    f"path {record.to_path} — {record.reason}"),
            object_type="demand", object_id=record.demand_id,
            actor=source, actor_label=_SOURCE_LABELS.get(source, source),
            accepted=bool(record.accepted), is_protection=is_frr,
            from_path=record.from_path, to_path=record.to_path))
        if record.accepted and record.is_flap:
            events.append(_event(
                "flap", record.step, record.t_min, hour_at(record.t_min),
                title=f"Route flap: {record.demand_id}",
                detail=f"{endpoints} returned to a recently used path.",
                object_type="demand", object_id=record.demand_id, actor=source))

    # --- congestion and SLA pressure, from recorded interval metrics
    previous: dict[str, Any] | None = None
    for interval in engine.metrics_history:
        step = int(interval["step"])
        t_min = float(interval["t_min"])
        hour = float(interval.get("hour", hour_at(t_min)))
        congested_now = int(interval.get("congested_links", 0))
        congested_before = int(previous.get("congested_links", 0)) if previous else 0
        if congested_now > 0 and congested_before == 0:
            events.append(_event(
                "congestion", step, t_min, hour,
                title="Congestion threshold crossed",
                detail=(f"{congested_now} directed link(s) reached "
                        f"{m.CONGESTION_UTIL:.0%} utilization; busiest link at "
                        f"{float(interval.get('max_util', 0)):.0%}."),
                object_type=None, object_id=None,
                congested_links=congested_now))
        sla_now = int(interval.get("sla_violations", 0))
        sla_before = int(previous.get("sla_violations", 0)) if previous else 0
        if sla_now > 0 and sla_before == 0:
            events.append(_event(
                "sla_risk", step, t_min, hour, title="First SLA violation",
                detail=f"{sla_now} demand-interval SLA violation(s) recorded.",
                object_type=None, object_id=None, sla_violations=sla_now))
        failed_now = set(interval.get("failed_links", ()))
        failed_before = set(previous.get("failed_links", ())) if previous else set()
        for link_id in sorted(failed_now - failed_before):
            link = engine.topo.link_defs.get(link_id)
            events.append(_event(
                "failure", step, t_min, hour,
                title=f"Link failure: {link_label(link.a, link.z) if link else link_id}",
                detail=f"{link_id} went down.",
                object_type="link", object_id=link_id))
        for link_id in sorted(failed_before - failed_now):
            link = engine.topo.link_defs.get(link_id)
            events.append(_event(
                "recovery", step, t_min, hour,
                title=f"Link repaired: {link_label(link.a, link.z) if link else link_id}",
                detail=f"{link_id} came back up.",
                object_type="link", object_id=link_id))
        if (previous and congested_before > 0 and congested_now == 0
                and sla_now == 0 and not failed_now):
            events.append(_event(
                "stabilization", step, t_min, hour, title="Network stabilized",
                detail="No congested link, no SLA violation and no failed link.",
                object_type=None, object_id=None))
        previous = interval

    events.sort(key=lambda e: (e["step"], EVENT_KINDS.index(e["kind"])
                              if e["kind"] in EVENT_KINDS else len(EVENT_KINDS)))
    return events


def _advisor_events(session: Any) -> list[dict[str, Any]]:
    engine = session.runners[0].eng
    start_hour = engine.scenario.start_hour
    events = []
    for record in session.advisor_history:
        hour = (start_hour + float(record["t_min"]) / 60.0) % 24.0
        decoded = record.get("decoded") or {}
        target = decoded.get("demand") or "no-op"
        events.append(_event(
            "recommendation", record["step"], record["t_min"], hour,
            title=f"Policy recommendation: {target}",
            detail=("Approved by the operator." if record.get("approved")
                    else "Rejected by the operator; no TE change was applied."),
            object_type="demand" if decoded else None,
            object_id=decoded.get("demand"),
            approved=bool(record.get("approved")),
            operator_rejection=not record.get("approved"),
            action=record.get("action")))
    if session.pending_proposal:
        pending = session.pending_proposal
        hour = (start_hour + float(pending["t_min"]) / 60.0) % 24.0
        decoded = pending.get("decoded") or {}
        events.append(_event(
            "recommendation", pending["step"], pending["t_min"], hour,
            title="Policy recommendation awaiting a decision",
            detail=("The recommendation is a preview. No TE change has been applied."),
            object_type="demand" if decoded else None,
            object_id=decoded.get("demand"), pending=True,
            action=pending.get("action")))
    return events


def session_timeline(session: Any) -> dict[str, Any]:
    primary = session.runners[0]
    events = runner_events(primary) + _advisor_events(session)
    events.sort(key=lambda e: (e["step"], e["kind"]))
    scenario = primary.eng.scenario
    scripted = [{
        "t_min": ev.get("t_min"),
        "clock": _clock((scenario.start_hour + float(ev.get("t_min", 0)) / 60.0) % 24.0),
        "type": ev.get("type"),
        "target": ev.get("link") or ev.get("demand"),
        "description": str(ev),
    } for ev in scenario.events]
    return {
        "provenance": {"source_kind": "live_session", "session_id": session.id,
                       "generation": int(session.generation),
                       "algorithm": primary.algorithm},
        "scenario": scenario.name,
        "start_hour": scenario.start_hour,
        "duration_min": scenario.duration_min,
        "current_step": int(primary.eng.step_count),
        "current_t_min": round(float(primary.eng.t_min), 3),
        "kinds": list(EVENT_KINDS),
        "events": events,
        "scripted_events": scripted,
        "frr_note": ("FRR entries are the engine's built-in local repair, not a "
                     "controller decision. They are never counted as TE actions."),
    }
