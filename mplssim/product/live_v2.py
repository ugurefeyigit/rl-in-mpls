"""A read-only product view over the frozen V2 simulation engine.

`mplssim/sim/engine_v2.py` is a frozen scientific definition and is not edited
here. This module wraps it instead, exposing the same *shape* the product layer
already reads from the V1 engine — `snapshot()`, `action_log`, `validate_action`
— while every number underneath is the V2 engine's own.

What this view deliberately does **not** do:

- it never invents a value V2 does not compute. V1's reroute cooldown becomes
  V2's TE dwell and says so; V1's manual traffic multiplier and burst injection
  have no V2 counterpart and fail closed rather than being emulated;
- it never renames a V2 quantity into a V1 quantity that means something else.
  Accepted TE changes, FRR protection moves and operator interventions stay
  three separate counters, exactly as the study's accounting requires;
- it never writes. Every method here reads engine state or returns a clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mplssim.sim import models as m

#: V2's TE dwell replaces V1's reroute cooldown. Different rule, different name.
COOLDOWN_LABEL = "V2 minimum TE dwell"

#: Manual traffic shaping exists only in the V1 engine.
UNSUPPORTED_INTERVENTION = (
    "The frozen V2 engine has no manual traffic multiplier or burst injector. "
    "Its offered traffic comes from the governed traffic model and scenario "
    "events only, and this product will not fabricate a V2 traffic override.")


@dataclass(frozen=True)
class ActionRecordV2:
    """One route change, tagged with which subsystem made it.

    `source` is `te` for an accepted controller action, `frr` for a protection
    move or protection disconnection, and `restore` for a post-recovery
    restoration. Nothing collapses these into a single "reroute" count.
    """

    step: int
    t_min: float
    source: str
    demand_id: str
    from_path: int
    to_path: int
    accepted: bool
    reason: str
    is_flap: bool = False


class EngineV2View:
    """Product-shaped read view over one :class:`SimulationEngineV2`.

    Attribute access falls through to the engine, so anything already correct in
    V2 is used unchanged; only the names the product layer reads under a V1
    spelling are translated here.
    """

    def __init__(self, engine: Any) -> None:
        object.__setattr__(self, "_engine", engine)

    # ---------------------------------------------------------- delegation
    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    @property
    def engine(self) -> Any:
        return self._engine

    # ------------------------------------------------------------ aliases
    @property
    def demand_volumes(self) -> np.ndarray:
        """V2 calls this offered traffic; the product reads it as volume."""
        return self._engine.demand_offered

    @property
    def link_load(self) -> np.ndarray:
        return self._engine.link_input_load

    @property
    def _path_links(self) -> list[list[np.ndarray]]:
        """Read alias the repository's existing baseline controllers expect."""
        return self._engine._cand_links

    @property
    def _path_costs(self) -> list[list[float]]:
        return self._engine._cand_cost

    @property
    def cooldown_until(self) -> np.ndarray:
        """Step at which each demand's TE dwell expires."""
        return self._engine.step_count + self._engine.te_dwell_remaining

    @property
    def last_reroute_step(self) -> np.ndarray:
        return self._engine.last_te_step

    # -------------------------------------------------------------- paths
    def path_bottleneck_util(self, d_idx: int, p_idx: int) -> float:
        links = self._engine._cand_links[d_idx][p_idx]
        return float(np.max(self._engine.link_util[links])) if len(links) else 0.0

    def path_available_bandwidth(self, d_idx: int, p_idx: int) -> float:
        engine = self._engine
        links = engine._cand_links[d_idx][p_idx]
        if not len(links):
            return 0.0
        return float(np.min(engine.capacity[links] - engine.link_input_load[links]))

    def projected_bottleneck_after_move(self, d_idx: int, p_idx: int) -> float:
        """V2's projected *gross* bottleneck — the quantity its validator uses."""
        return float(self._engine.projected_gross_bottleneck(d_idx, p_idx))

    # ---------------------------------------------------------- validation
    def validate_action(self, d_idx: int, p_idx: int,
                        source: str = "rl") -> tuple[bool, str]:
        """The authoritative V2 validator, with its own rejection reason."""
        return self._engine.validate_te_action(int(d_idx), int(p_idx))

    def apply_action(self, d_idx: int, p_idx: int,
                     source: str = "rl") -> tuple[bool, str]:
        record = self._engine.apply_te_action(int(d_idx), int(p_idx))
        return bool(record["accepted"]), str(record["reason"])

    # -------------------------------------------------------- interventions
    def inject_failure(self, link_id: str) -> None:
        self._engine.set_link_state(link_id, up=False)

    def recover_link(self, link_id: str) -> None:
        self._engine.set_link_state(link_id, up=True)

    def inject_burst(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(UNSUPPORTED_INTERVENTION)

    # ------------------------------------------------------------- cloning
    def clone(self) -> "EngineV2View":
        return EngineV2View(self._engine.clone())

    def fast_clone(self) -> "EngineV2View":
        return EngineV2View(self._engine.fast_clone())

    # ---------------------------------------------------------- action log
    @property
    def action_log(self) -> list[ActionRecordV2]:
        """Accepted TE changes, FRR moves and restorations, in time order.

        Rejections stay out: `apply_te_action` records them in its counters, and
        putting a refused request in a *route change* log would overstate churn.
        """
        rows: list[tuple[int, int, ActionRecordV2]] = []
        for order, record in enumerate(self._engine.te_history):
            rows.append((0, order, ActionRecordV2(
                step=int(record["step"]), t_min=float(record["t_min"]),
                source="te", demand_id=str(record["demand_id"]),
                from_path=int(record["from_path"] or 0),
                to_path=int(record["to_path"]), accepted=True,
                reason=str(record["reason"]),
                is_flap=bool(record.get("reversal")))))
        for order, record in enumerate(self._engine.frr_history):
            disconnected = record["event"] == "disconnected"
            rows.append((1, order, ActionRecordV2(
                step=int(record["step"]), t_min=float(record["t_min"]),
                source="frr", demand_id=str(record["demand_id"]),
                from_path=int(record["from_path"]),
                to_path=-1 if disconnected else int(record["to_path"]),
                accepted=not disconnected,
                reason=("no live candidate remained, demand disconnected"
                        if disconnected else "fast reroute onto the cheapest "
                                             "live candidate"))))
        for order, record in enumerate(self._engine.restoration_history):
            rows.append((2, order, ActionRecordV2(
                step=int(record["step"]), t_min=float(record["t_min"]),
                source="restore", demand_id=str(record["demand_id"]),
                from_path=int(record["from_path"]),
                to_path=int(record["to_path"]), accepted=True,
                reason="restored after link recovery")))
        rows.sort(key=lambda row: (row[2].t_min, row[0], row[1]))
        return [row[2] for row in rows]

    # ------------------------------------------------------------ snapshot
    def _lsp_counts(self) -> np.ndarray:
        engine = self._engine
        counts = np.zeros(engine.topo.n_dlinks, dtype=np.int64)
        for d_idx in range(engine.n_demands):
            if engine.disconnected[d_idx]:
                continue
            counts[engine._cand_links[d_idx][int(engine.current_path[d_idx])]] += 1
        return counts

    def _path_change_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._engine.te_history:
            counts[record["demand_id"]] = counts.get(record["demand_id"], 0) + 1
        for record in self._engine.frr_history:
            if record["event"] == "reroute":
                counts[record["demand_id"]] = counts.get(record["demand_id"], 0) + 1
        for record in self._engine.restoration_history:
            counts[record["demand_id"]] = counts.get(record["demand_id"], 0) + 1
        return counts

    def _candidates(self, d_idx: int) -> list[dict[str, Any]]:
        engine = self._engine
        demand = engine.demands[d_idx]
        current = int(engine.current_path[d_idx])
        out = []
        for p_idx, routers in enumerate(demand.candidate_paths):
            out.append({
                "path_idx": p_idx,
                "routers": list(routers),
                "hops": len(routers) - 1,
                "admin_cost": engine._cand_cost[d_idx][p_idx],
                "available": bool(engine.path_available(d_idx, p_idx)),
                "bottleneck_util": round(self.path_bottleneck_util(d_idx, p_idx), 4),
                "projected_bottleneck_util": round(
                    self.projected_bottleneck_after_move(d_idx, p_idx), 4),
                "available_bandwidth_mbps": round(
                    self.path_available_bandwidth(d_idx, p_idx), 1),
                "is_current": current == p_idx,
            })
        return out

    def snapshot(self) -> dict[str, Any]:
        """Full JSON-serializable V2 state, in the shape the product reads."""
        engine = self._engine
        n_lsps = self._lsp_counts()
        changes = self._path_change_counts()
        util, load = engine.link_util, engine.link_input_load
        qdelay, loss = engine.link_qdelay, engine.link_loss
        available = engine.capacity - load
        congested = util >= m.CONGESTION_UTIL
        ups = engine._dlink_up.tolist()

        links = []
        for i, dl in enumerate(engine.topo.dlinks):
            links.append({
                "id": dl.id, "link": dl.undirected_id, "src": dl.src, "dst": dl.dst,
                "capacity_mbps": dl.capacity_mbps,
                "load_mbps": round(float(load[i]), 2),
                "utilization": round(float(util[i]), 4),
                "prop_delay_ms": dl.delay_ms,
                "queue_delay_ms": round(float(qdelay[i]), 3),
                "loss_fraction": round(float(loss[i]), 5),
                "weight": dl.weight,
                "up": ups[i],
                "congested": bool(congested[i]),
                "available_mbps": round(float(available[i]), 2),
                "n_lsps": int(n_lsps[i]),
            })

        demands = []
        for d_idx, demand in enumerate(engine.demands):
            current = int(engine.current_path[d_idx])
            demands.append({
                "id": demand.id, "src": demand.src, "dst": demand.dst,
                "class": demand.cls.name, "priority": demand.cls.priority,
                "protected": demand.cls.protected,
                "base_mbps": demand.base_mbps,
                "volume_mbps": round(float(engine.demand_offered[d_idx]), 2),
                "carried_mbps": round(float(engine.demand_delivered[d_idx]), 2),
                "current_path_idx": current,
                "current_path": list(demand.candidate_paths[current]),
                "delay_ms": round(float(engine.demand_delay[d_idx]), 2),
                "loss_pct": round(float(engine.demand_loss_fraction[d_idx]) * 100.0, 3),
                "sla_ok": bool(engine.demand_sla_ok[d_idx]),
                "sla_max_latency_ms": demand.cls.max_latency_ms,
                "sla_max_loss_pct": demand.cls.max_loss_pct,
                "disconnected": bool(engine.disconnected[d_idx]),
                "bottleneck_util": round(self.path_bottleneck_util(d_idx, current), 4),
                "path_changes": int(changes.get(demand.id, 0)),
                "last_reroute_step": int(engine.last_te_step[d_idx]),
                "cooldown_until_step": int(
                    engine.step_count + engine.te_dwell_remaining[d_idx]),
                "te_dwell_remaining": int(engine.te_dwell_remaining[d_idx]),
                "path_age_steps": int(engine.path_age_steps[d_idx]),
                "candidates": self._candidates(d_idx),
            })

        return {
            "scenario": engine.scenario_name,
            "seed": engine.episode_seed,
            "t_min": engine.t_min,
            "hour": (engine.scenario.start_hour + engine.t_min / 60.0) % 24.0,
            "step": engine.step_count,
            "done": engine.done,
            "routers": [{"id": r.id, "role": r.role, "x": r.x, "y": r.y,
                         "neighbors": engine.topo.neighbors(r.id)}
                        for r in engine.topo.routers.values()],
            "links": links,
            "demands": demands,
            "failed_links": [lid for lid, up in engine.link_up.items() if not up],
            "metrics": engine.metrics_history[-1] if engine.metrics_history else None,
            "recent_actions": [vars(a) for a in self.action_log[-12:]],
        }


__all__ = ["ActionRecordV2", "COOLDOWN_LABEL", "EngineV2View",
           "UNSUPPORTED_INTERVENTION"]
