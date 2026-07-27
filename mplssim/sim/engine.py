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

# NumPy sums runs shorter than this sequentially and longer ones with pairwise
# blocking. The vectorized per-demand delay sum pads rows with zeros, which is
# only bit-identical to the unpadded per-path sum while both stay sequential.
_MAX_SEQUENTIAL_SUM = 8


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
        # per-demand class constants, precomputed once (used every micro-tick)
        self._priorities = np.array([d.cls.priority for d in self.demands], dtype=float)
        self._n_cands = np.array([len(d.candidate_paths) for d in self.demands],
                                 dtype=np.int64)

        # Padded candidate->directed-link index matrix, shape
        # (n_demands, k_paths, max_hops). Short paths are padded by REPEATING
        # their own first link: max/min/all are idempotent, so a duplicated
        # entry cannot change their result, which keeps the vectorized
        # candidate reductions bit-identical to the per-path loops they
        # replace. Rows for candidates that do not exist are filled with link 0
        # and masked out via ``_cand_exists``.
        k = self.cfg.k_paths
        max_hops = max(len(links) for per_d in self._path_links for links in per_d)
        self._cand_pad = np.zeros((self.n_demands, k, max_hops), dtype=np.int64)
        self._cand_exists = np.zeros((self.n_demands, k), dtype=bool)
        for d_idx, per_d in enumerate(self._path_links):
            for p_idx, links in enumerate(per_d[:k]):
                if links.size == 0:
                    raise ValueError(f"demand {d_idx} candidate {p_idx} has no links")
                self._cand_pad[d_idx, p_idx, :links.size] = links
                self._cand_pad[d_idx, p_idx, links.size:] = links[0]
                self._cand_exists[d_idx, p_idx] = True
        # Second padding scheme, for the reductions where repeating a link
        # would NOT be neutral: sum and product. Padding points at a sentinel
        # slot (index n_dlinks) that holds the identity element - 0.0 for the
        # delay sum, 1.0 for the delivered-fraction product.
        self._cand_pad_sum = np.full((self.n_demands, k, max_hops), topo.n_dlinks,
                                     dtype=np.int64)
        self._cand_hops = np.zeros((self.n_demands, k), dtype=float)
        for d_idx, per_d in enumerate(self._path_links):
            for p_idx, links in enumerate(per_d[:k]):
                self._cand_pad_sum[d_idx, p_idx, :links.size] = links
                self._cand_hops[d_idx, p_idx] = links.size
        self._cand_pad_sum.flags.writeable = False
        self._cand_hops.flags.writeable = False

        self._cand_pad.flags.writeable = False
        self._cand_exists.flags.writeable = False
        self._n_cands.flags.writeable = False

        # Padding a sum with zeros is only bit-exact while NumPy sums the row
        # sequentially; above _MAX_SEQUENTIAL_SUM it switches to pairwise
        # blocking, which regroups the real terms and can move the last ULP.
        # Below that threshold the vectorized path is used; at or above it the
        # engine falls back to the per-demand loop. Both branches are asserted
        # to agree in tests/test_runtime_equivalence.py.
        self._vectorize_demand_metrics = max_hops < _MAX_SEQUENTIAL_SUM
        self._link_delay_ext = np.zeros(topo.n_dlinks + 1)
        self._link_delivered_ext = np.ones(topo.n_dlinks + 1)
        self._max_latency_ms = np.array([d.cls.max_latency_ms for d in self.demands])
        self._max_loss_pct = np.array([d.cls.max_loss_pct for d in self.demands])

        # Scratch buffers for projected-load arithmetic. Two of them, so a
        # candidate/mask sweep holding a base cannot have it overwritten by an
        # unrelated single-shot projection call. Neither ever escapes: the
        # public helpers return freshly allocated arrays.
        self._proj_buf = np.zeros(topo.n_dlinks)    # single-shot public calls
        self._sweep_buf = np.zeros(topo.n_dlinks)   # per-demand candidate sweeps
        # lazily built topology-derived snapshot constants (see _static_snapshot_data)
        self._static_snapshot_cache: tuple[list[tuple], list[dict[str, Any]]] | None = None

        # ---- dynamic state ----
        self.t_min: float = 0.0
        self.step_count: int = 0
        self.current_path: np.ndarray = np.zeros(self.n_demands, dtype=np.int64)
        self.disconnected: np.ndarray = np.zeros(self.n_demands, dtype=bool)
        self.link_up: dict[str, bool] = {lid: True for lid in topo.link_defs}
        # Per-directed-link mirror of ``link_up``, rebuilt by set_link_state.
        # ``link_up`` stays the authoritative, API-visible representation;
        # this array exists so availability checks are vectorized instead of
        # doing a dict lookup per hop. set_link_state is the ONLY supported
        # mutator of link state - writing into ``link_up`` directly would
        # desynchronize the two (tests/test_runtime_equivalence.py asserts they
        # agree across a whole episode).
        self._dlink_up = np.ones(topo.n_dlinks, dtype=bool)
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
        """Availability of one candidate path.

        Deliberately uses dict lookups rather than the ``_dlink_up`` array:
        measured over 68 candidates, dict lookups take 43us against 57us for
        per-path NumPy indexing, because a path is only 4-6 hops and two ufunc
        calls cost more than the lookups they replace. The array pays off only
        when every candidate is reduced at once - see
        :meth:`candidate_available_matrix`, which does all 68 in 1.9us. The two
        representations are asserted to agree across a full episode in
        tests/test_runtime_equivalence.py.
        """
        link_up, dlinks = self.link_up, self.topo.dlinks
        return all(link_up[dlinks[int(li)].undirected_id]
                   for li in self._path_links[d_idx][p_idx])

    def candidate_available_matrix(self) -> np.ndarray:
        """(n_demands, k_paths) availability for every candidate at once.

        Equivalent to calling :meth:`path_available` for each (demand,
        candidate); the padded matrix lets one vectorized reduction replace the
        per-hop Python loop. Non-existent candidates are False.
        """
        return self._dlink_up[self._cand_pad].all(axis=2) & self._cand_exists

    def candidate_bottleneck_matrix(self) -> np.ndarray:
        """(n_demands, k_paths) bottleneck utilization for every candidate.

        Bit-identical to :meth:`path_bottleneck_util` per entry: padding
        repeats a path's own first link and ``max`` is idempotent. Entries for
        non-existent candidates are meaningless and must be masked by the
        caller via ``_cand_exists``.
        """
        return self.link_util[self._cand_pad].max(axis=2)

    def path_bottleneck_util(self, d_idx: int, p_idx: int) -> float:
        links = self._path_links[d_idx][p_idx]
        return float(np.max(self.link_util[links])) if links.size else 0.0

    def path_available_bandwidth(self, d_idx: int, p_idx: int) -> float:
        links = self._path_links[d_idx][p_idx]
        return float(np.min(self.capacity[links] - self.link_load[links]))

    def _projected_base_loads(self, d_idx: int, out: np.ndarray) -> np.ndarray:
        """Link loads with demand ``d_idx`` removed from its current path.

        Adding the demand's volume back on a candidate's links reproduces
        :meth:`projected_link_loads_after_move` exactly, because a link shared
        by the old and new path goes through the same ``(x - vol) + vol``
        sequence either way. Computing this once per demand instead of once per
        candidate is what makes candidate sweeps cheap.

        ``out`` is filled in place and returned. It is always caller-supplied so
        that a sweep holding a base cannot have it clobbered by an unrelated
        projection call made in between.
        """
        np.copyto(out, self.link_load)
        if not self.disconnected[d_idx]:
            out[self._path_links[d_idx][int(self.current_path[d_idx])]] -= \
                float(self.demand_volumes[d_idx])
        return out

    def projected_link_loads_after_move(self, d_idx: int, p_idx: int) -> np.ndarray:
        """Link loads as they WOULD be after moving demand d to candidate p.

        Removes the demand's traffic from its current path first, then adds it
        to the proposed path. This matters when old and new paths share links:
        checking raw current loads there double-counts the demand's own
        traffic. Used by action validation, masking, candidate info and the
        heuristics so all consumers agree.
        """
        loads = self._projected_base_loads(d_idx, np.empty_like(self.link_load))
        loads[self._path_links[d_idx][p_idx]] += float(self.demand_volumes[d_idx])
        return loads

    def projected_path_headroom(self, d_idx: int, p_idx: int) -> float:
        """Min (capacity - projected load) along candidate p after the move."""
        base = self._projected_base_loads(d_idx, self._proj_buf)
        return self._headroom_from_base(base, d_idx, p_idx)

    def _headroom_from_base(self, base: np.ndarray, d_idx: int, p_idx: int) -> float:
        links = self._path_links[d_idx][p_idx]
        return float(np.min(self.capacity[links] - (base[links] + float(self.demand_volumes[d_idx]))))

    def projected_bottleneck_after_move(self, d_idx: int, p_idx: int) -> float:
        """Max utilization along candidate p after the move (projected)."""
        base = self._projected_base_loads(d_idx, self._proj_buf)
        return self._bottleneck_from_base(base, d_idx, p_idx)

    def _bottleneck_from_base(self, base: np.ndarray, d_idx: int, p_idx: int) -> float:
        links = self._path_links[d_idx][p_idx]
        return float(np.max((base[links] + float(self.demand_volumes[d_idx]))
                            / self.capacity[links]))

    def candidate_matrices(self) -> dict[str, np.ndarray]:
        """Every per-candidate quantity for every demand, as (n_demands, k) arrays.

        Replaces four NumPy reductions per candidate (272 tiny calls for a
        17-demand, 4-candidate topology) with four whole-array reductions. Each
        entry is bit-identical to its scalar counterpart: padding repeats a
        path's own first link and max/min/all are idempotent, and the projected
        loads go through the same ``(load - vol) + vol`` sequence per link.
        """
        pad = self._cand_pad
        vols = self.demand_volumes

        # projected loads per demand: its own traffic removed from its current path
        bases = np.broadcast_to(self.link_load,
                                (self.n_demands, self.topo.n_dlinks)).copy()
        for d_idx in range(self.n_demands):
            if not self.disconnected[d_idx]:
                bases[d_idx, self._path_links[d_idx][int(self.current_path[d_idx])]] -= \
                    float(vols[d_idx])
        rows = np.arange(self.n_demands)[:, None, None]
        projected = (bases[rows, pad] + vols[:, None, None]) / self.capacity[pad]

        return {
            "available": self._dlink_up[pad].all(axis=2) & self._cand_exists,
            "bottleneck_util": self.link_util[pad].max(axis=2),
            "projected_bottleneck_util": projected.max(axis=2),
            "available_bandwidth_mbps": (self.capacity - self.link_load)[pad].min(axis=2),
        }

    def candidate_row(self, d_idx: int) -> dict[str, np.ndarray]:
        """:meth:`candidate_matrices` restricted to one demand, as (k,) arrays.

        Same values, but it reduces over that demand's rows only - so a
        single-demand query does not pay for all the others.
        """
        pad = self._cand_pad[d_idx]
        vol = float(self.demand_volumes[d_idx])
        base = self._projected_base_loads(d_idx, self._sweep_buf)
        return {
            "available": self._dlink_up[pad].all(axis=1) & self._cand_exists[d_idx],
            "bottleneck_util": self.link_util[pad].max(axis=1),
            "projected_bottleneck_util": ((base[pad] + vol) / self.capacity[pad]).max(axis=1),
            "available_bandwidth_mbps": (self.capacity - self.link_load)[pad].min(axis=1),
        }

    def _candidate_info_from(self, mats: dict[str, np.ndarray], d_idx: int,
                             row: bool = False) -> list[dict[str, Any]]:
        d = self.demands[d_idx]
        cur = int(self.current_path[d_idx])
        costs = self._path_costs[d_idx]
        sel = (lambda key: mats[key]) if row else (lambda key: mats[key][d_idx])
        avail = sel("available")
        bott = sel("bottleneck_util")
        proj = sel("projected_bottleneck_util")
        bw = sel("available_bandwidth_mbps")
        return [
            {
                "path_idx": p,
                "routers": list(routers),
                "hops": len(routers) - 1,
                "admin_cost": costs[p],
                "available": bool(avail[p]),
                "bottleneck_util": round(float(bott[p]), 4),
                "projected_bottleneck_util": round(float(proj[p]), 4),
                "available_bandwidth_mbps": round(float(bw[p]), 1),
                "is_current": cur == p,
            }
            for p, routers in enumerate(d.candidate_paths)
        ]

    def candidate_info(self, d_idx: int) -> list[dict[str, Any]]:
        return self._candidate_info_from(self.candidate_row(d_idx), d_idx, row=True)

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
        # Keep the per-directed-link mirror in step. Rebuilt wholesale rather
        # than patched: link changes are rare (a handful per episode) and a
        # full rebuild cannot drift out of sync with the dict.
        for dl in self.topo.dlinks:
            self._dlink_up[dl.index] = self.link_up[dl.undirected_id]
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

        self._compute_demand_metrics(vols)

        offered = float(np.sum(vols))
        carried = float(np.sum(self.demand_carried))
        pw = self._priorities * vols
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

    def _compute_demand_metrics(self, vols: np.ndarray) -> None:
        """End-to-end delay, loss, carried volume and SLA state per demand.

        Two implementations that must agree bit-for-bit. The vectorized one
        gathers every demand's current path through a sentinel-padded index
        matrix so all 17 paths reduce in one call; the scalar one walks each
        path individually. See ``_vectorize_demand_metrics`` for when each is
        used and why.
        """
        if not self._vectorize_demand_metrics:
            self._compute_demand_metrics_scalar(vols)
            return

        # elementwise combinations, with the identity element in the sentinel slot
        n = self.topo.n_dlinks
        np.add(self.prop_delay, self.link_qdelay, out=self._link_delay_ext[:n])
        np.subtract(1.0, self.link_loss, out=self._link_delivered_ext[:n])

        rows = np.arange(self.n_demands)
        pad = self._cand_pad_sum[rows, self.current_path]      # (n_demands, max_hops)
        hops = self._cand_hops[rows, self.current_path]

        delay = self._link_delay_ext[pad].sum(axis=1) + m.PROC_DELAY_MS * hops
        loss = 1.0 - self._link_delivered_ext[pad].prod(axis=1)

        disc = self.disconnected
        self.demand_delay[:] = np.where(disc, 0.0, delay)
        self.demand_loss[:] = np.where(disc, 1.0, loss)
        self.demand_carried[:] = np.where(disc, 0.0, vols * (1.0 - loss))
        self.demand_sla_ok[:] = (
            (delay <= self._max_latency_ms)
            & (loss * 100.0 <= self._max_loss_pct)
            & ~disc
        )

    def _compute_demand_metrics_scalar(self, vols: np.ndarray) -> None:
        """Reference per-demand implementation (see _compute_demand_metrics)."""
        link_delay = self.prop_delay + self.link_qdelay
        link_delivered = 1.0 - self.link_loss
        for d_idx in range(self.n_demands):
            if self.disconnected[d_idx]:
                self.demand_delay[d_idx] = 0.0
                self.demand_loss[d_idx] = 1.0
                self.demand_carried[d_idx] = 0.0
                self.demand_sla_ok[d_idx] = False
                continue
            links = self._path_links[d_idx][int(self.current_path[d_idx])]
            delay = float(np.sum(link_delay[links]) + m.PROC_DELAY_MS * links.size)
            loss = float(1.0 - np.prod(link_delivered[links]))
            self.demand_delay[d_idx] = delay
            self.demand_loss[d_idx] = loss
            self.demand_carried[d_idx] = vols[d_idx] * (1.0 - loss)
            cls = self.demands[d_idx].cls
            self.demand_sla_ok[d_idx] = (
                delay <= cls.max_latency_ms and loss * 100.0 <= cls.max_loss_pct
            )

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
        """Deep copy for one-step counterfactual analysis (post-hoc only).

        Fully isolated but expensive: it also deep-copies the topology, the
        networkx graph and the traffic configuration, none of which are ever
        mutated. :meth:`fast_clone` is the cheap equivalent; this method is kept
        as-is so existing callers keep the exact semantics they were written
        against.
        """
        return copy.deepcopy(self)

    #: Attributes that are constant for an engine's lifetime and may therefore
    #: be shared with a lightweight clone. Everything NOT listed here is copied
    #: by :meth:`fast_clone`; ``tests/test_runtime_equivalence.py`` enforces
    #: that no mutable container escapes this list.
    _SHAREABLE_ATTRS = frozenset({
        "topo", "cfg", "scenario_name", "seed", "scenario", "n_demands",
        "demands", "demand_by_id", "_path_links", "_path_costs",
        "capacity", "prop_delay", "_priorities", "_n_cands",
        "_cand_pad", "_cand_exists", "_cand_pad_sum", "_cand_hops",
        "_max_latency_ms", "_max_loss_pct", "_static_snapshot_cache",
    })

    def fast_clone(self) -> "SimulationEngine":
        """Lightweight clone for counterfactual analysis.

        Shares immutable topology/configuration data with the original and
        copies every piece of mutable state, including the traffic RNG, so the
        two engines evolve completely independently. Observationally equivalent
        to :meth:`clone` but far cheaper, because the deep copy of the topology
        graph and traffic configuration dominates ``copy.deepcopy``.

        One documented difference from :meth:`clone`: ``action_log`` and
        ``metrics_history`` are copied as lists, so the two engines share the
        individual (append-only, never mutated) record objects. Appending to
        one engine's history never affects the other; mutating an already
        recorded entry in place - which no code does - would.
        """
        cl = object.__new__(SimulationEngine)
        state = dict(self.__dict__)

        # independent per-demand and per-link mutable arrays
        for attr in ("current_path", "disconnected", "cooldown_until",
                     "path_change_count", "last_reroute_step", "_dlink_up",
                     "link_util", "link_load", "link_qdelay", "link_loss",
                     "util_ewma", "demand_volumes", "demand_delay",
                     "demand_loss", "demand_carried", "demand_sla_ok"):
            state[attr] = self.__dict__[attr].copy()

        # scratch buffers must not be shared: concurrent use would interleave
        state["_proj_buf"] = np.empty_like(self._proj_buf)
        state["_sweep_buf"] = np.empty_like(self._sweep_buf)
        state["_link_delay_ext"] = self._link_delay_ext.copy()
        state["_link_delivered_ext"] = self._link_delivered_ext.copy()

        state["link_up"] = dict(self.link_up)
        state["prev_path_hist"] = [deque(h, maxlen=h.maxlen) for h in self.prev_path_hist]
        state["action_log"] = list(self.action_log)
        state["metrics_history"] = list(self.metrics_history)
        state["manual_bursts"] = [dict(b) for b in self.manual_bursts]

        # traffic: new model object with its own RNG and AR(1) noise state,
        # sharing the immutable config and the already-materialized scenario
        traffic = copy.copy(self.traffic)
        traffic._noise = self.traffic._noise.copy()
        traffic._rng = copy.deepcopy(self.traffic._rng)
        state["traffic"] = traffic

        cl.__dict__.update(state)
        return cl

    # --------------------------------------------------------------- snapshot
    def _lsp_counts(self) -> np.ndarray:
        """Number of live LSPs traversing each directed link.

        One pass over demands accumulating into a per-link counter, instead of
        an ``i in path`` membership scan for every (link, demand) pair.
        """
        counts = np.zeros(self.topo.n_dlinks, dtype=np.int64)
        for d_idx in range(self.n_demands):
            if self.disconnected[d_idx]:
                continue
            counts[self._path_links[d_idx][int(self.current_path[d_idx])]] += 1
        return counts

    def _static_snapshot_data(self) -> tuple[list[tuple], list[dict[str, Any]]]:
        """Topology-derived snapshot constants, built once per engine.

        Returns (per-link constant tuples, router payloads). The router dicts
        are never handed out directly - :meth:`snapshot` copies them - so the
        cache cannot be mutated by a consumer.
        """
        if self._static_snapshot_cache is None:
            link_consts = [
                (dl.id, dl.undirected_id, dl.src, dl.dst, dl.capacity_mbps,
                 dl.delay_ms, dl.weight)
                for dl in self.topo.dlinks
            ]
            routers = [
                {"id": r.id, "role": r.role, "x": r.x, "y": r.y,
                 "neighbors": self.topo.neighbors(r.id)}
                for r in self.topo.routers.values()
            ]
            self._static_snapshot_cache = (link_consts, routers)
        return self._static_snapshot_cache

    def snapshot(self) -> dict[str, Any]:
        """Full JSON-serializable state for the API / frontend."""
        link_consts, routers = self._static_snapshot_data()
        n_lsps = self._lsp_counts()
        cand_mats = self.candidate_matrices()   # once for all demands
        load, util = self.link_load, self.link_util
        qdelay, loss = self.link_qdelay, self.link_loss
        avail = self.capacity - load
        congested = util >= m.CONGESTION_UTIL
        # one conversion to Python bools beats 64 numpy scalar extractions
        ups = self._dlink_up.tolist()
        links = []
        # key order is preserved exactly as the frontend has always seen it
        for i, (lid, uid, src, dst, cap, prop_ms, weight) in enumerate(link_consts):
            links.append({
                "id": lid, "link": uid, "src": src, "dst": dst,
                "capacity_mbps": cap,
                "load_mbps": round(float(load[i]), 2),
                "utilization": round(float(util[i]), 4),
                "prop_delay_ms": prop_ms,
                "queue_delay_ms": round(float(qdelay[i]), 3),
                "loss_fraction": round(float(loss[i]), 5),
                "weight": weight,
                "up": ups[i],
                "congested": bool(congested[i]),
                "available_mbps": round(float(avail[i]), 2),
                "n_lsps": int(n_lsps[i]),
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
                "candidates": self._candidate_info_from(cand_mats, d_idx),
            })
        return {
            "scenario": self.scenario_name,
            "seed": self.seed,
            "t_min": self.t_min,
            "hour": (self.scenario.start_hour + self.t_min / 60.0) % 24.0,
            "step": self.step_count,
            "done": self.done,
            "routers": [dict(r) for r in routers],
            "links": links,
            "demands": demands,
            "failed_links": [lid for lid, up in self.link_up.items() if not up],
            "metrics": self.metrics_history[-1] if self.metrics_history else None,
            "recent_actions": [vars(a) for a in self.action_log[-12:]],
        }
