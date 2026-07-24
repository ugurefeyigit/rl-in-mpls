"""Flow-level MPLS-TE simulation engine.

The engine owns all dynamic state: current LSP placement per demand, link
operational status, the simulated clock, telemetry history, and the action
log. Controllers (RL agent or baselines) call :meth:`apply_action` between
control intervals; :meth:`step_interval` advances time by one control
interval composed of several one-minute micro-ticks.

Determinism: given (topology, traffic config, scenario, seed) the offered
traffic and scripted events are fully reproducible. Routing decisions are the
only degree of freedom, so identical scenarios provide paired comparisons.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mplssim.core.model import Demand
from mplssim.core.topology import Topology
from mplssim.paths.candidates import generate_candidate_paths, path_admin_cost
from mplssim.sim import models as m
from mplssim.traffic.model import ScenarioSpec, TrafficConfig, TrafficModel


# candidate-path cache shared across engine instances (paths depend only on
# topology and k/hop-factor, never on dynamic state)
_CANDIDATE_CACHE: dict[tuple, dict[str, tuple]] = {}


@dataclass
class EngineConfig:
    control_interval_min: int = 5
    micro_ticks_per_interval: int = 5
    k_paths: int = 4
    max_hop_factor: float = 2.5
    reroute_cooldown_steps: int = 3
    flap_window_steps: int = 6


@dataclass
class ActionRecord:
    step: int
    t_min: float
    source: str            # "rl" | "greedy" | "cspf" | "static" | "frr" | "manual"
    demand_id: str
    from_path: int
    to_path: int
    accepted: bool
    reason: str
    is_flap: bool = False


class SimulationEngine:
    def __init__(
        self,
        topo: Topology,
        traffic_cfg: TrafficConfig,
        scenario: ScenarioSpec,
        seed: int,
        cfg: EngineConfig | None = None,
    ) -> None:
        self.topo = topo
        self.cfg = cfg or EngineConfig()
        self.scenario_name = scenario.name
        self.seed = seed
        self.traffic = TrafficModel(config=traffic_cfg, scenario=scenario, seed=seed)
        self.scenario = self.traffic.scenario  # materialized (randomized events drawn)

        # Candidate paths per demand (stable ordering: ascending admin cost).
        # Cached per (topology id, k, hop factor): identical for every episode.
        self.demands: list[Demand] = []
        self._path_links: list[list[np.ndarray]] = []  # [d][p] -> dlink index array
        self._path_costs: list[list[float]] = []
        cache_key = (id(topo), self.cfg.k_paths, self.cfg.max_hop_factor)
        cand_cache = _CANDIDATE_CACHE.setdefault(cache_key, {})
        for d in traffic_cfg.demands:
            cands = cand_cache.get(d.id)
            if cands is None:
                cands = generate_candidate_paths(
                    topo, d.src, d.dst, k=self.cfg.k_paths, max_hop_factor=self.cfg.max_hop_factor
                )
                cand_cache[d.id] = cands
            d2 = Demand(id=d.id, src=d.src, dst=d.dst, cls=d.cls,
                        base_mbps=d.base_mbps, index=d.index, candidate_paths=cands)
            self.demands.append(d2)
            self._path_links.append([
                np.array(topo.path_dlink_indices(p), dtype=np.int64) for p in cands
            ])
            self._path_costs.append([path_admin_cost(topo, p) for p in cands])

        self.n_demands = len(self.demands)
        self.demand_by_id = {d.id: d for d in self.demands}

        # numpy constants
        self.capacity = np.array([dl.capacity_mbps for dl in topo.dlinks])
        self.prop_delay = np.array([dl.delay_ms for dl in topo.dlinks])

        # ---- dynamic state ----
        self.t_min: float = 0.0
        self.step_count: int = 0
        self.current_path: np.ndarray = np.zeros(self.n_demands, dtype=np.int64)
        self.disconnected: np.ndarray = np.zeros(self.n_demands, dtype=bool)
        self.link_up: dict[str, bool] = {lid: True for lid in topo.link_defs}
        self.cooldown_until: np.ndarray = np.zeros(self.n_demands, dtype=np.int64)
        self.path_change_count: np.ndarray = np.zeros(self.n_demands, dtype=np.int64)
        self.last_reroute_step: np.ndarray = np.full(self.n_demands, -10**6, dtype=np.int64)
        self.prev_path_hist: list[deque[int]] = [deque(maxlen=4) for _ in range(self.n_demands)]
        self.action_log: list[ActionRecord] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.reroutes_this_interval: int = 0
        self.flaps_this_interval: int = 0
        self.frr_events_this_interval: int = 0
        # last-tick per-link / per-demand telemetry (filled by _compute_tick)
        self.link_util = np.zeros(topo.n_dlinks)
        self.link_load = np.zeros(topo.n_dlinks)
        self.link_qdelay = np.zeros(topo.n_dlinks)
        self.link_loss = np.zeros(topo.n_dlinks)
        self.demand_volumes = np.zeros(self.n_demands)
        self.demand_delay = np.zeros(self.n_demands)
        self.demand_loss = np.zeros(self.n_demands)
        self.demand_carried = np.zeros(self.n_demands)
        self.demand_sla_ok = np.ones(self.n_demands, dtype=bool)
        self.util_ewma = np.zeros(topo.n_dlinks)  # recent-utilization trend feature
        # manual interventions injected at runtime from the UI (on top of the scenario)
        self.manual_multiplier: float = 1.0
        self.manual_bursts: list[dict[str, float]] = []  # {demand_idx, factor, until_min}
        self._prime_tick()

    # ------------------------------------------------------------------ paths
    def dlink_up(self, dlink_index: int) -> bool:
        return self.link_up[self.topo.dlinks[dlink_index].undirected_id]

    def path_available(self, d_idx: int, p_idx: int) -> bool:
        return all(self.dlink_up(int(li)) for li in self._path_links[d_idx][p_idx])

    def path_bottleneck_util(self, d_idx: int, p_idx: int) -> float:
        links = self._path_links[d_idx][p_idx]
        return float(np.max(self.link_util[links])) if links.size else 0.0

    def path_available_bandwidth(self, d_idx: int, p_idx: int) -> float:
        links = self._path_links[d_idx][p_idx]
        return float(np.min(self.capacity[links] - self.link_load[links]))

    def projected_link_loads_after_move(self, d_idx: int, p_idx: int) -> np.ndarray:
        """Link loads as they WOULD be after moving demand d to candidate p.

        Removes the demand's traffic from its current path first, then adds it
        to the proposed path. This matters when old and new paths share links:
        checking raw current loads there double-counts the demand's own
        traffic. Used by action validation, masking, candidate info and the
        heuristics so all consumers agree.
        """
        loads = self.link_load.copy()
        vol = float(self.demand_volumes[d_idx])
        if not self.disconnected[d_idx]:
            loads[self._path_links[d_idx][int(self.current_path[d_idx])]] -= vol
        loads[self._path_links[d_idx][p_idx]] += vol
        return loads

    def projected_path_headroom(self, d_idx: int, p_idx: int) -> float:
        """Min (capacity - projected load) along candidate p after the move."""
        links = self._path_links[d_idx][p_idx]
        loads = self.projected_link_loads_after_move(d_idx, p_idx)
        return float(np.min(self.capacity[links] - loads[links]))

    def projected_bottleneck_after_move(self, d_idx: int, p_idx: int) -> float:
        """Max utilization along candidate p after the move (projected)."""
        links = self._path_links[d_idx][p_idx]
        loads = self.projected_link_loads_after_move(d_idx, p_idx)
        return float(np.max(loads[links] / self.capacity[links]))

    def candidate_info(self, d_idx: int) -> list[dict[str, Any]]:
        d = self.demands[d_idx]
        out = []
        for p, routers in enumerate(d.candidate_paths):
            out.append({
                "path_idx": p,
                "routers": list(routers),
                "hops": len(routers) - 1,
                "admin_cost": self._path_costs[d_idx][p],
                "available": self.path_available(d_idx, p),
                "bottleneck_util": round(self.path_bottleneck_util(d_idx, p), 4),
                "projected_bottleneck_util": round(self.projected_bottleneck_after_move(d_idx, p), 4),
                "available_bandwidth_mbps": round(self.path_available_bandwidth(d_idx, p), 1),
                "is_current": int(self.current_path[d_idx]) == p,
            })
        return out

    # ---------------------------------------------------------------- actions
    def validate_action(self, d_idx: int, p_idx: int, source: str = "rl") -> tuple[bool, str]:
        """Constraint checker (also used as the safe-RL filter)."""
        if not (0 <= d_idx < self.n_demands):
            return False, "unknown demand"
        d = self.demands[d_idx]
        if not (0 <= p_idx < len(d.candidate_paths)):
            return False, "unknown candidate path"
        if int(self.current_path[d_idx]) == p_idx and not self.disconnected[d_idx]:
            return False, "already on this path"
        if not self.path_available(d_idx, p_idx):
            return False, "path traverses a failed link"
        if source in ("rl", "greedy") and self.step_count < int(self.cooldown_until[d_idx]):
            return False, f"cooldown until step {int(self.cooldown_until[d_idx])}"
        if d.cls.protected:
            # projected check: the demand's own traffic is first removed from
            # its current path, so shared links are not double-counted
            if self.projected_path_headroom(d_idx, p_idx) < 0.0:
                return False, "insufficient bandwidth for protected class"
        return True, "ok"

    def apply_action(self, d_idx: int, p_idx: int, source: str = "rl",
                     forced: bool = False) -> tuple[bool, str]:
        """Move demand d to candidate path p. Returns (accepted, reason)."""
        if forced:
            ok, reason = (True, "forced") if self.path_available(d_idx, p_idx) else (False, "path down")
        else:
            ok, reason = self.validate_action(d_idx, p_idx, source)
        old = int(self.current_path[d_idx])
        is_flap = False
        if ok:
            hist = self.prev_path_hist[d_idx]
            recent = self.step_count - int(self.last_reroute_step[d_idx]) <= self.cfg.flap_window_steps
            is_flap = recent and len(hist) > 0 and hist[-1] == p_idx
            self.current_path[d_idx] = p_idx
            self.disconnected[d_idx] = False
            hist.append(old)
            self.path_change_count[d_idx] += 1
            self.last_reroute_step[d_idx] = self.step_count
            if source in ("rl", "greedy", "cspf", "manual"):
                self.cooldown_until[d_idx] = self.step_count + self.cfg.reroute_cooldown_steps
            self.reroutes_this_interval += 1
            if is_flap:
                self.flaps_this_interval += 1
            if source == "frr":
                self.frr_events_this_interval += 1
        self.action_log.append(ActionRecord(
            step=self.step_count, t_min=self.t_min, source=source,
            demand_id=self.demands[d_idx].id, from_path=old, to_path=p_idx,
            accepted=ok, reason=reason, is_flap=is_flap,
        ))
        return ok, reason

    # --------------------------------------------------------------- failures
    def set_link_state(self, link_id: str, up: bool) -> None:
        if link_id not in self.link_up:
            raise KeyError(f"unknown link {link_id}")
        if self.link_up[link_id] == up:
            return
        self.link_up[link_id] = up
        if not up:
            self._fast_reroute_around_failures()
        else:
            self._restore_disconnected()

    def _fast_reroute_around_failures(self) -> None:
        """Local repair: demands on a failed path move to the best surviving
        candidate (lowest admin cost). Models pre-signalled backup LSP / FRR."""
        for d_idx in range(self.n_demands):
            cur = int(self.current_path[d_idx])
            if self.path_available(d_idx, cur) and not self.disconnected[d_idx]:
                continue
            placed = False
            for p_idx in np.argsort(self._path_costs[d_idx]):
                if self.path_available(d_idx, int(p_idx)):
                    self.apply_action(d_idx, int(p_idx), source="frr", forced=True)
                    placed = True
                    break
            if not placed:
                self.disconnected[d_idx] = True

    def _restore_disconnected(self) -> None:
        for d_idx in np.where(self.disconnected)[0]:
            for p_idx in np.argsort(self._path_costs[int(d_idx)]):
                if self.path_available(int(d_idx), int(p_idx)):
                    self.apply_action(int(d_idx), int(p_idx), source="frr", forced=True)
                    break

    def inject_failure(self, link_id: str) -> None:
        self.set_link_state(link_id, False)

    def recover_link(self, link_id: str) -> None:
        self.set_link_state(link_id, True)

    def inject_burst(self, demand_id: str, factor: float, duration_min: float) -> None:
        d = self.demand_by_id[demand_id]
        self.manual_bursts.append({
            "demand_idx": float(d.index), "factor": factor,
            "until_min": self.t_min + duration_min,
        })

    # ------------------------------------------------------------------ steps
    def _compute_tick(self) -> dict[str, float]:
        """Compute link loads and all derived telemetry for the current minute."""
        vols = self.traffic.volumes(self.t_min) * self.manual_multiplier
        self.manual_bursts = [b for b in self.manual_bursts if b["until_min"] > self.t_min]
        for b in self.manual_bursts:
            vols[int(b["demand_idx"])] *= b["factor"]
        self.demand_volumes = vols
        loads = np.zeros(self.topo.n_dlinks)
        for d_idx in range(self.n_demands):
            if self.disconnected[d_idx]:
                continue
            loads[self._path_links[d_idx][int(self.current_path[d_idx])]] += vols[d_idx]
        self.link_load = loads
        self.link_util = loads / self.capacity
        self.link_qdelay = m.queue_delay_ms(self.link_util)
        self.link_loss = m.loss_fraction(self.link_util)
        self.util_ewma = 0.8 * self.util_ewma + 0.2 * self.link_util

        for d_idx in range(self.n_demands):
            if self.disconnected[d_idx]:
                self.demand_delay[d_idx] = 0.0
                self.demand_loss[d_idx] = 1.0
                self.demand_carried[d_idx] = 0.0
                self.demand_sla_ok[d_idx] = False
                continue
            links = self._path_links[d_idx][int(self.current_path[d_idx])]
            delay = float(np.sum(self.prop_delay[links] + self.link_qdelay[links])
                          + m.PROC_DELAY_MS * links.size)
            loss = float(1.0 - np.prod(1.0 - self.link_loss[links]))
            self.demand_delay[d_idx] = delay
            self.demand_loss[d_idx] = loss
            self.demand_carried[d_idx] = vols[d_idx] * (1.0 - loss)
            cls = self.demands[d_idx].cls
            self.demand_sla_ok[d_idx] = (
                delay <= cls.max_latency_ms and loss * 100.0 <= cls.max_loss_pct
            )

        offered = float(np.sum(vols))
        carried = float(np.sum(self.demand_carried))
        prios = np.array([d.cls.priority for d in self.demands], dtype=float)
        pw = prios * vols
        pw_success = float(np.sum(pw * self.demand_sla_ok) / np.sum(pw)) if np.sum(pw) > 0 else 1.0
        active = vols > 1e-9
        w_delay = (float(np.sum(self.demand_delay[active] * vols[active]) / np.sum(vols[active]))
                   if np.any(active) else 0.0)
        return {
            "max_util": float(np.max(self.link_util)),
            "mean_util": float(np.mean(self.link_util)),
            "util_std": float(np.std(self.link_util)),
            "jain_fairness": m.jain_fairness(self.link_util),
            "mean_delay_ms": w_delay,
            "p95_delay_ms": float(np.percentile(self.demand_delay[active], 95)) if np.any(active) else 0.0,
            "max_delay_ms": float(np.max(self.demand_delay)),
            "loss_ratio": (offered - carried) / offered if offered > 0 else 0.0,
            "delivered_ratio": carried / offered if offered > 0 else 1.0,
            "offered_mbps": offered,
            "carried_mbps": carried,
            "sla_violations": int(np.sum(~self.demand_sla_ok)),
            "sla_violation_fraction": float(np.mean(~self.demand_sla_ok)),
            "priority_sla_success": pw_success,
            "congested_links": int(np.sum(self.link_util >= m.CONGESTION_UTIL)),
            "overload_ratio": float(np.sum(np.maximum(loads - self.capacity, 0.0)) / np.sum(self.capacity)),
            "disconnected_demands": int(np.sum(self.disconnected)),
        }

    def _prime_tick(self) -> None:
        """Compute telemetry at t=0 so observations exist before the first step."""
        self._process_link_events(-1.0, 0.5)
        self._compute_tick()

    def _process_link_events(self, t_from: float, t_to: float) -> None:
        for ev in self.traffic.link_events_at(t_from, t_to):
            self.set_link_state(ev["link"], ev["type"] == "link_up")

    def step_interval(self) -> dict[str, Any]:
        """Advance one control interval (micro_ticks x 1-minute ticks).

        Returns aggregated interval metrics; continuous quantities are averaged
        over ticks, worst-case quantities (max_util, sla) take the interval max.
        """
        n = self.cfg.micro_ticks_per_interval
        dt = self.cfg.control_interval_min / n
        ticks: list[dict[str, float]] = []
        for _ in range(n):
            prev_t = self.t_min
            self.t_min += dt
            self._process_link_events(prev_t, self.t_min)
            self.traffic.advance_noise()
            ticks.append(self._compute_tick())
        self.step_count += 1

        agg: dict[str, Any] = {}
        for key in ticks[0]:
            vals = [t[key] for t in ticks]
            if key in ("max_util", "sla_violations", "sla_violation_fraction",
                       "congested_links", "disconnected_demands", "max_delay_ms"):
                agg[key] = max(vals)
            else:
                agg[key] = float(np.mean(vals))
        agg["t_min"] = self.t_min
        agg["hour"] = (self.scenario.start_hour + self.t_min / 60.0) % 24.0
        agg["step"] = self.step_count
        agg["reroutes"] = self.reroutes_this_interval
        agg["flaps"] = self.flaps_this_interval
        agg["frr_events"] = self.frr_events_this_interval
        agg["n_demands"] = self.n_demands
        agg["failed_links"] = [lid for lid, up in self.link_up.items() if not up]
        self.metrics_history.append(agg)
        # Counters accumulate from the moment a controller acts (between
        # intervals) through this interval's ticks; reset only after recording.
        self.reroutes_this_interval = 0
        self.flaps_this_interval = 0
        self.frr_events_this_interval = 0
        return agg

    @property
    def done(self) -> bool:
        return self.t_min >= self.scenario.duration_min - 1e-9

    @property
    def all_disconnected(self) -> bool:
        return bool(np.all(self.disconnected))

    def clone(self) -> "SimulationEngine":
        """Deep copy for one-step counterfactual analysis (post-hoc only)."""
        return copy.deepcopy(self)

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        """Full JSON-serializable state for the API / frontend."""
        links = []
        for dl in self.topo.dlinks:
            i = dl.index
            links.append({
                "id": dl.id, "link": dl.undirected_id, "src": dl.src, "dst": dl.dst,
                "capacity_mbps": dl.capacity_mbps,
                "load_mbps": round(float(self.link_load[i]), 2),
                "utilization": round(float(self.link_util[i]), 4),
                "prop_delay_ms": dl.delay_ms,
                "queue_delay_ms": round(float(self.link_qdelay[i]), 3),
                "loss_fraction": round(float(self.link_loss[i]), 5),
                "weight": dl.weight,
                "up": self.dlink_up(i),
                "congested": bool(self.link_util[i] >= m.CONGESTION_UTIL),
                "available_mbps": round(float(self.capacity[i] - self.link_load[i]), 2),
                "n_lsps": int(sum(
                    1 for d_idx in range(self.n_demands)
                    if not self.disconnected[d_idx]
                    and i in self._path_links[d_idx][int(self.current_path[d_idx])]
                )),
            })
        demands = []
        for d_idx, d in enumerate(self.demands):
            cur = int(self.current_path[d_idx])
            demands.append({
                "id": d.id, "src": d.src, "dst": d.dst, "class": d.cls.name,
                "priority": d.cls.priority, "protected": d.cls.protected,
                "base_mbps": d.base_mbps,
                "volume_mbps": round(float(self.demand_volumes[d_idx]), 2),
                "carried_mbps": round(float(self.demand_carried[d_idx]), 2),
                "current_path_idx": cur,
                "current_path": list(d.candidate_paths[cur]),
                "delay_ms": round(float(self.demand_delay[d_idx]), 2),
                "loss_pct": round(float(self.demand_loss[d_idx]) * 100.0, 3),
                "sla_ok": bool(self.demand_sla_ok[d_idx]),
                "sla_max_latency_ms": d.cls.max_latency_ms,
                "sla_max_loss_pct": d.cls.max_loss_pct,
                "disconnected": bool(self.disconnected[d_idx]),
                "bottleneck_util": round(self.path_bottleneck_util(d_idx, cur), 4),
                "path_changes": int(self.path_change_count[d_idx]),
                "last_reroute_step": int(self.last_reroute_step[d_idx]),
                "cooldown_until_step": int(self.cooldown_until[d_idx]),
                "candidates": self.candidate_info(d_idx),
            })
        return {
            "scenario": self.scenario_name,
            "seed": self.seed,
            "t_min": self.t_min,
            "hour": (self.scenario.start_hour + self.t_min / 60.0) % 24.0,
            "step": self.step_count,
            "done": self.done,
            "routers": [
                {"id": r.id, "role": r.role, "x": r.x, "y": r.y,
                 "neighbors": self.topo.neighbors(r.id)}
                for r in self.topo.routers.values()
            ],
            "links": links,
            "demands": demands,
            "failed_links": [lid for lid, up in self.link_up.items() if not up],
            "metrics": self.metrics_history[-1] if self.metrics_history else None,
            "recent_actions": [vars(a) for a in self.action_log[-12:]],
        }
