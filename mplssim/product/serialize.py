"""Typed product snapshots of a live session.

The shell renders one shape regardless of mode, so the derivation that used to
happen three times in three frontends happens once, here, against the engine
that owns the numbers.

Rules this module keeps:

- A value the engine does not have is **absent with a reason**, never zero.
- An undirected link carries both directions. The map may summarize with the
  worse one, and it says so.
- Delay and loss are modeled analytic values and are labelled as such.
- Nothing here writes to the engine. Every helper reads.
"""

from __future__ import annotations

from typing import Any

from mplssim.display import CITY_NAMES, CLASS_NAMES, scenario_label
from mplssim.product import pairing
from mplssim.product.contracts import SourceKind, source_profile
from mplssim.product.display_map import (
    ROLE_LABELS, ROLE_TOKENS, capacity_class, utilization_band,
)

#: Utilization at or above this counts as congested (mplssim/sim/models.py).
from mplssim.sim import models as m


# --------------------------------------------------------------- provenance
def provenance(session: Any, runner: Any) -> dict[str, Any]:
    profile = source_profile(SourceKind.LIVE_SESSION)
    engine = runner.eng
    version = getattr(runner, "environment_version", "v1")
    checkpoint = getattr(runner, "checkpoint", None)
    return {
        "source_kind": SourceKind.LIVE_SESSION.value,
        "label": profile.label,
        "pattern": profile.pattern,
        "icon": profile.icon,
        "live": True,
        "session_id": session.id,
        "generation": int(session.generation),
        "sequence": int(session.sequence),
        "step": int(engine.step_count),
        "environment_version": version,
        "environment_label": version.upper(),
        "environment_class": ("mplssim.rl.env_v2.MplsTeEnvV2" if version == "v2"
                              else "mplssim.rl.env.MplsTeEnv"),
        "scenario": session.config.scenario,
        "scenario_label": scenario_label(session.config.scenario),
        "seed": int(session.config.seed),
        "policy_id": runner.algorithm,
        "policy_family": ("learner" if checkpoint is not None or
                          runner.algorithm == "rl" else "baseline"),
        "output_semantics": getattr(runner, "output_semantics", "none"),
        "checkpoint_id": getattr(runner, "checkpoint_id", None),
        "training_root": (int(session.config.training_root)
                          if version == "v2" else None),
        # A frozen governed checkpoint driving a live demonstration is still a
        # live record. It is never rendered as study evidence.
        "checkpoint_provenance": (checkpoint.provenance()
                                  if checkpoint is not None else None),
        "safety_filter": bool(session.config.safety_filter),
    }


# ------------------------------------------------------------------- nodes
def _nodes(engine: Any, links_by_router: dict[str, list[dict[str, Any]]],
           demand_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lsps_through: dict[str, int] = {r: 0 for r in engine.topo.routers}
    affected: dict[str, int] = {r: 0 for r in engine.topo.routers}
    for demand in demand_rows:
        for router in demand["current_path"]:
            lsps_through[router] = lsps_through.get(router, 0) + 1
            if not demand["sla_ok"] or demand["disconnected"]:
                affected[router] = affected.get(router, 0) + 1
    nodes = []
    for router_id, router in engine.topo.routers.items():
        adjacent = links_by_router.get(router_id, [])
        worst = max((l["worst_utilization"] for l in adjacent), default=None)
        nodes.append({
            "id": router_id,
            "city": CITY_NAMES.get(router_id, router_id),
            "role": router.role,
            "role_label": ROLE_LABELS.get(router.role, router.role),
            "role_token": ROLE_TOKENS.get(router.role, router.role),
            "title": f"{CITY_NAMES.get(router_id, router_id).upper()} · "
                     f"{ROLE_TOKENS.get(router.role, router.role)}",
            "neighbors": engine.topo.neighbors(router_id),
            "n_links": len(adjacent),
            "n_lsps": lsps_through.get(router_id, 0),
            "affected_demands": affected.get(router_id, 0),
            "worst_adjacent_utilization": worst,
            "has_failed_link": any(not l["up"] for l in adjacent),
        })
    return sorted(nodes, key=lambda n: n["id"])


# ------------------------------------------------------------------- links
def _links(raw_links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the engine's directed rows into one row per physical link."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_links:
        grouped.setdefault(row["link"], []).append(row)

    out = []
    for link_id, rows in grouped.items():
        directions = [{
            "directed_id": r["id"],
            "src": r["src"],
            "dst": r["dst"],
            "src_city": CITY_NAMES.get(r["src"], r["src"]),
            "dst_city": CITY_NAMES.get(r["dst"], r["dst"]),
            "load_mbps": r["load_mbps"],
            "utilization": r["utilization"],
            "available_mbps": r["available_mbps"],
            "queue_delay_ms": r["queue_delay_ms"],
            "loss_fraction": r["loss_fraction"],
            "congested": r["congested"],
            "n_lsps": r["n_lsps"],
            "band": utilization_band(r["utilization"])["id"],
        } for r in rows]
        worst = max(directions, key=lambda d: d["utilization"])
        up = bool(rows[0]["up"])
        band = utilization_band(worst["utilization"])
        capacity = rows[0]["capacity_mbps"]
        state = "failed" if not up else ("congested" if worst["congested"] else "normal")
        out.append({
            "id": link_id,
            "a": rows[0]["src"],
            "z": rows[0]["dst"],
            "a_city": CITY_NAMES.get(rows[0]["src"], rows[0]["src"]),
            "z_city": CITY_NAMES.get(rows[0]["dst"], rows[0]["dst"]),
            "label": f"{CITY_NAMES.get(rows[0]['src'])}–{CITY_NAMES.get(rows[0]['dst'])}",
            "technical": f"{rows[0]['src']}–{rows[0]['dst']}, {link_id}",
            "capacity_mbps": capacity,
            "capacity_class": capacity_class(capacity)["id"],
            "prop_delay_ms": rows[0]["prop_delay_ms"],
            "weight": rows[0]["weight"],
            "up": up,
            "state": state,
            "worst_utilization": worst["utilization"],
            "worst_direction": f"{worst['src']}→{worst['dst']}",
            "worst_direction_rule": "The map summarizes with the busier direction; "
                                    "both directions are listed in inspection.",
            "band": band["id"],
            "band_label": band["label"],
            "pressure_ticks": band["ticks"],
            "n_lsps": sum(d["n_lsps"] for d in directions),
            "directions": directions,
        })
    return sorted(out, key=lambda l: int(l["id"][1:]))


# ----------------------------------------------------------------- demands
def _demands(raw_demands: list[dict[str, Any]], engine: Any) -> list[dict[str, Any]]:
    # V1 holds a reroute cooldown; V2 holds a minimum TE dwell. Different rules,
    # so the label follows the engine rather than being fixed to V1's wording.
    cooldown_label = ("V2 minimum TE dwell"
                      if type(engine).__name__ == "EngineV2View"
                      else "V1 reroute cooldown")
    out = []
    for row in raw_demands:
        cls = row["class"]
        candidates = [{
            "path_idx": c["path_idx"],
            "routers": list(c["routers"]),
            "path_label": " → ".join(CITY_NAMES.get(r, r) for r in c["routers"]),
            "hops": c["hops"],
            "admin_cost": c["admin_cost"],
            "available": c["available"],
            "bottleneck_util": c["bottleneck_util"],
            "projected_bottleneck_util": c["projected_bottleneck_util"],
            "available_bandwidth_mbps": c["available_bandwidth_mbps"],
            "is_current": c["is_current"],
        } for c in row.get("candidates", [])]
        risk = _sla_risk(row)
        out.append({
            "id": row["id"],
            "src": row["src"],
            "dst": row["dst"],
            "src_city": CITY_NAMES.get(row["src"], row["src"]),
            "dst_city": CITY_NAMES.get(row["dst"], row["dst"]),
            "class": cls,
            "class_label": CLASS_NAMES.get(cls, cls),
            "label": (f"{CITY_NAMES.get(row['src'], row['src'])} → "
                      f"{CITY_NAMES.get(row['dst'], row['dst'])} "
                      f"{CLASS_NAMES.get(cls, cls)}"),
            "priority": row["priority"],
            "protected": row["protected"],
            "base_mbps": row["base_mbps"],
            "offered_mbps": row["volume_mbps"],
            "carried_mbps": row["carried_mbps"],
            "current_path_idx": row["current_path_idx"],
            "current_path": row["current_path"],
            "current_path_label": " → ".join(
                CITY_NAMES.get(r, r) for r in row["current_path"]),
            "delay_ms": row["delay_ms"],
            "loss_pct": row["loss_pct"],
            "sla_ok": row["sla_ok"],
            "sla_max_latency_ms": row["sla_max_latency_ms"],
            "sla_max_loss_pct": row["sla_max_loss_pct"],
            "disconnected": row["disconnected"],
            "bottleneck_util": row["bottleneck_util"],
            "path_changes": row["path_changes"],
            "last_reroute_step": row["last_reroute_step"],
            "cooldown_until_step": row["cooldown_until_step"],
            "cooldown_label": cooldown_label,
            "risk_rank": risk[0],
            "risk_label": risk[1],
            "candidates": candidates,
        })
    return out


def _sla_risk(row: dict[str, Any]) -> tuple[int, str]:
    """Default ordering for the demand and SLA-risk table (design §6.7)."""
    if row["disconnected"] and row["protected"]:
        return (0, "Disconnected · protected")
    if row["disconnected"]:
        return (1, "Disconnected")
    if not row["sla_ok"]:
        return (2, "SLA violated")
    if row["bottleneck_util"] is not None and row["bottleneck_util"] >= m.CONGESTION_UTIL:
        return (3, "Crosses a congested link")
    return (4, "Within SLA")


# ------------------------------------------------------------------ metrics
_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "max_util": ("Busiest link", "share"),
    "mean_util": ("Mean link load", "share"),
    "delivered_ratio": ("Delivered traffic", "share"),
    "mean_delay_ms": ("Mean demand delay", "ms"),
    "max_delay_ms": ("Worst demand delay", "ms"),
    "loss_ratio": ("Loss ratio", "share"),
    "sla_violations": ("Demands violating SLA now", "count"),
    "sla_violation_fraction": ("Share of demands violating SLA", "share"),
    "congested_links": ("Congested directed links", "count"),
    "disconnected_demands": ("Disconnected demands", "count"),
    "reroutes": ("TE reroutes this interval", "count"),
    "flaps": ("Flaps this interval", "count"),
    "frr_events": ("FRR protection moves this interval", "count"),
    # V2 keeps controller, protection and recovery moves in separate counters.
    # They are never summed into one "reroutes" number.
    "accepted_te_changes": ("Controller TE changes this interval", "count"),
    "rejected_te_requests": ("TE requests rejected this interval", "count"),
    "te_reversals": ("TE reversals this interval", "count"),
    "frr_changes": ("FRR protection moves this interval", "count"),
    "frr_disconnections": ("FRR disconnections this interval", "count"),
    "recovery_restorations": ("Restorations this interval", "count"),
    "overload_ratio": ("Offered load above capacity", "share"),
    "gross_max_util": ("Busiest link, gross offered", "share"),
    "protected_disconnected_demands": ("Protected demands disconnected", "count"),
}


def _metrics(engine: Any) -> dict[str, Any]:
    history = engine.metrics_history
    if not history:
        return {"available": False,
                "reason": "No interval has completed yet, so there is nothing to report."}
    current = history[-1]
    previous = history[-2] if len(history) > 1 else None
    rows = {}
    for key, (label, unit) in _METRIC_LABELS.items():
        if key not in current:
            continue
        value = current[key]
        prior = previous.get(key) if previous else None
        rows[key] = {
            "label": label, "unit": unit, "value": value, "previous": prior,
            "delta": (round(float(value) - float(prior), 6)
                      if prior is not None else None),
        }
    return {"available": True, "step": current.get("step"),
            "t_min": current.get("t_min"), "values": rows,
            "has_previous": previous is not None}


# ----------------------------------------------------------------- incident
def _phase(links: list[dict[str, Any]], demands: list[dict[str, Any]],
           engine: Any) -> dict[str, Any]:
    failed = [l for l in links if not l["up"]]
    congested = [l for l in links if l["state"] == "congested"]
    at_risk = [d for d in demands if not d["sla_ok"] or d["disconnected"]]
    recent_frr = [a for a in engine.action_log[-24:] if a.source == "frr"]
    if failed:
        phase, label = "failure", "Link failure in progress"
    elif recent_frr and congested:
        phase, label = "recovery", "Restoring after protection moves"
    elif congested:
        phase, label = "pressure", "Congestion pressure"
    elif at_risk:
        phase, label = "rising", "SLA pressure without congestion"
    else:
        phase, label = "normal", "Normal operation"
    return {
        "phase": phase,
        "label": label,
        "failed_links": [l["id"] for l in failed],
        "failed_link_labels": [l["label"] for l in failed],
        "congested_links": [l["id"] for l in congested],
        "demands_at_risk": [d["id"] for d in at_risk],
        "active_incident": (
            f"{failed[0]['label']} link down" if failed else
            (f"{congested[0]['label']} at {congested[0]['worst_utilization']:.0%}"
             if congested else None)),
    }


# ---------------------------------------------------------------- snapshot
def session_snapshot(session: Any, runner: Any) -> dict[str, Any]:
    engine = runner.eng
    raw = engine.snapshot()
    links = _links(raw["links"])
    demands = _demands(raw["demands"], engine)
    links_by_router: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_router.setdefault(link["a"], []).append(link)
        links_by_router.setdefault(link["z"], []).append(link)
    nodes = _nodes(engine, links_by_router, demands)
    return {
        "provenance": provenance(session, runner),
        "session": session.status(),
        "time": {
            "step": raw["step"],
            "t_min": raw["t_min"],
            "hour": raw["hour"],
            "clock": _clock(raw["hour"]),
            "duration_min": engine.scenario.duration_min,
            "done": raw["done"],
        },
        "nodes": nodes,
        "links": links,
        "demands": demands,
        "metrics": _metrics(engine),
        "incident": _phase(links, demands, engine),
        "comparison": pairing.synchronization(session),
        "availability": {
            "link_telemetry": {"available": True, "reason": None},
            "observations": {
                "available": getattr(runner, "_obs", None) is not None,
                "reason": (None if getattr(runner, "_obs", None) is not None else
                           f"{runner.algorithm} reads engine state directly and "
                           f"never builds an observation vector."),
            },
            "expected_telemetry": {
                "available": True,
                "reason": (
                    f"The live {getattr(runner, 'environment_version', 'v1').upper()} "
                    f"engine can be cloned, so a one-interval estimate is "
                    f"computable without touching the session."),
            },
        },
    }


def _clock(hour: float) -> str:
    total = int(round(hour * 60)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


# ------------------------------------------------------------ focused object
def focused_object(runner: Any, kind: str, object_id: str) -> dict[str, Any]:
    engine = runner.eng
    raw = engine.snapshot()
    if kind == "link":
        for link in _links(raw["links"]):
            if link["id"] == object_id:
                crossing = [d["id"] for d in _demands(raw["demands"], engine)
                            if _crosses(d, link)]
                return {"kind": "link", "object": link, "demands_crossing": crossing}
        raise KeyError(f"unknown link {object_id}")
    if kind == "demand":
        for demand in _demands(raw["demands"], engine):
            if demand["id"] == object_id:
                return {"kind": "demand", "object": demand}
        raise KeyError(f"unknown demand {object_id}")
    if kind == "router":
        links = _links(raw["links"])
        adjacent = [l for l in links if object_id in (l["a"], l["z"])]
        if not adjacent and object_id not in engine.topo.routers:
            raise KeyError(f"unknown router {object_id}")
        demands = _demands(raw["demands"], engine)
        return {
            "kind": "router",
            "object": {
                "id": object_id,
                "city": CITY_NAMES.get(object_id, object_id),
                "role": engine.topo.routers[object_id].role,
                "role_label": ROLE_LABELS.get(engine.topo.routers[object_id].role, ""),
                "neighbors": engine.topo.neighbors(object_id),
            },
            "links": adjacent,
            "demands_transiting": [d["id"] for d in demands
                                   if object_id in d["current_path"]],
            "not_modeled": ["CPU", "memory", "label table", "BGP", "RSVP-TE",
                            "interface counters"],
        }
    raise ValueError(f"unknown object kind {kind!r}; expected router, link or demand")


def _crosses(demand: dict[str, Any], link: dict[str, Any]) -> bool:
    path = demand["current_path"]
    hops = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    return (link["a"], link["z"]) in hops or (link["z"], link["a"]) in hops


def comparison_state(session: Any) -> dict[str, Any]:
    """The paired comparison, built in `mplssim.product.comparison`.

    Kept here as the product layer's single entry point so existing callers do
    not need to know where the derivation moved to.
    """
    from mplssim.product import comparison as comparison_mod

    return comparison_mod.comparison_state(session)
