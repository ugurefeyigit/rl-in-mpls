"""Flow-level MPLS-TE simulation engine, version 2.

Governing documents: docs/RL_ENVIRONMENT_V2_SPEC.md (normative),
docs/MPLS_SIMULATION_REALISM_AUDIT.md (rationale and equations).

This module is additive. :class:`mplssim.sim.engine.SimulationEngine` and every
V1 semantic remain untouched; nothing here is reachable from a V1 code path.

What V2 changes relative to V1, and why
--------------------------------------

**P0-1 right-closed event boundaries.** V1 processes link events in
``[old_t, new_t)`` after advancing the clock, so a failure scheduled at minute
60 is invisible in the observation returned at minute 60 and only lands during
the 60-to-61 tick. A controller could therefore act at minute 60 on a topology
it was told was healthy. V2 uses ``old_t < event_time <= new_t`` and processes
events at ``t == 0`` explicitly during construction. The invariant is: *before
an observation at boundary t, every event with time <= t has been applied
exactly once and the telemetry is for t.*

**P0-2 collision-free episode seeds.** Handled by the caller
(:mod:`mplssim.experiments.v2_factory`); this engine takes a single already
derived ``episode_seed`` and splits it into two independent child streams,
``SeedSequence([episode_seed, 1])`` for scenario materialization and
``SeedSequence([episode_seed, 2])`` for AR(1) traffic noise. No routing
decision consumes either stream, which is what keeps controller comparisons
paired.

**P0-3 role-valid candidates.** See :mod:`mplssim.paths.candidates_v2`.

**P1 carried-flow accounting.** V1 loads every hop of a path with the *full*
offered demand even after upstream links have already dropped traffic, so the
same lost traffic congests the network repeatedly. V2 keeps two ledgers:

``gross_link_load``
    full offered rate on every selected path. Conservative, used only for
    candidate projections and protected-class safety, so a route is never
    declared safe merely because upstream packets are already being dropped.
``link_input_load``
    surviving flow entering each link. Drives utilization, delay and loss.

Because two demands can each be upstream of the other on different links, the
carried-flow equations are implicit. They are solved by the damped fixed-point
iteration specified in the realism audit, and a tick that fails to converge
raises rather than returning stale telemetry.

**P1 separated accounting.** Accepted TE changes, rejected requests, TE
reversals, FRR moves, FRR disconnections and recovery restorations are counted
and recorded independently. FRR never sets TE dwell, never counts as policy
churn and never incurs TE cost.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mplssim.core.model import Demand
from mplssim.core.topology import CONFIG_DIR, Topology
from mplssim.paths.candidates import path_admin_cost
from mplssim.paths.candidates_v2 import (
    build_candidate_table,
    path_directed_edges,
    path_propagation_ms,
)
from mplssim.sim import models as m
from mplssim.traffic.model import ScenarioSpec, TrafficConfig, TrafficModel

#: Version identities emitted and validated by V2. See the spec's
#: "Version identity" table.
ENVIRONMENT_VERSION = "mpls-te-v2.0.0"
OBSERVATION_VERSION = "obs-v2.0-notime-604"
OBSERVATION_VERSION_TIME = "obs-v2.0-time-606"
ACTION_VERSION = "action-v2.0-discrete69"
REWARD_VERSION = "reward-v2.0-operational"
TRANSITION_VERSION = "transition-v2.0-boundary-right-closed"
CONFIG_VERSION = "config-v2.0"
SEED_VERSION = "seed-v2.0-stride1024"

#: Child-stream selectors (spec, "Seed protocol").
SEED_STREAM_SCENARIO = 1
SEED_STREAM_AR = 2

V2_CONFIG_DIR = CONFIG_DIR / "experiments"
ENV_CONFIG_PATH = V2_CONFIG_DIR / "rl_env_v2.yaml"

_EPS = 1e-12


class FlowSolverError(RuntimeError):
    """The carried-flow fixed point did not converge within the iteration cap.

    Raised instead of returning the last iterate: a non-converged tick has no
    defensible utilization, delay, loss or delivery, and silently publishing
    stale telemetry would corrupt every downstream metric and the reward.
    """


class EngineConfigError(ValueError):
    """V2 configuration is missing, malformed, or carries the wrong version."""


@dataclass(frozen=True)
class FlowSolverConfig:
    damping: float = 0.5          # weight retained on the previous loss estimate
    tolerance: float = 1.0e-10
    max_iterations: int = 32


@dataclass(frozen=True)
class EngineConfigV2:
    """Immutable V2 engine configuration (mirrors configs/experiments/rl_env_v2.yaml)."""

    version: str = CONFIG_VERSION
    control_interval_min: int = 5
    micro_ticks_per_interval: int = 5
    k_paths: int = 4
    max_hop_factor: float = 2.5
    minimum_te_dwell_steps: int = 3
    reversal_window_steps: int = 6
    max_te_changes_per_interval: int = 1
    candidate_delay_factor: float = 1.75
    candidate_delay_additive_ms: float = 10.0
    protected_projected_max_util: float = 1.0
    flow_solver: FlowSolverConfig = field(default_factory=FlowSolverConfig)
    worker_stride: int = 1024


def load_engine_config_v2(path: Path | None = None) -> EngineConfigV2:
    """Load and version-check ``configs/experiments/rl_env_v2.yaml``.

    Fails closed on a wrong ``version`` so a V1 or edited config can never be
    silently accepted as V2.
    """
    path = path or ENV_CONFIG_PATH
    if not path.exists():
        raise EngineConfigError(f"missing V2 environment config {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = str(raw.get("version", ""))
    if version != CONFIG_VERSION:
        raise EngineConfigError(
            f"{path}: config version {version!r} != required {CONFIG_VERSION!r}")
    fs = dict(raw.get("flow_solver", {}))
    return EngineConfigV2(
        version=version,
        control_interval_min=int(raw["control_interval_min"]),
        micro_ticks_per_interval=int(raw["micro_ticks_per_interval"]),
        k_paths=int(raw["k_paths"]),
        max_hop_factor=float(raw["max_hop_factor"]),
        minimum_te_dwell_steps=int(raw["minimum_te_dwell_steps"]),
        reversal_window_steps=int(raw["reversal_window_steps"]),
        max_te_changes_per_interval=int(raw["max_te_changes_per_interval"]),
        candidate_delay_factor=float(raw["candidate_delay_factor"]),
        candidate_delay_additive_ms=float(raw["candidate_delay_additive_ms"]),
        protected_projected_max_util=float(raw["protected_projected_max_util"]),
        flow_solver=FlowSolverConfig(
            damping=float(fs["damping"]),
            tolerance=float(fs["tolerance"]),
            max_iterations=int(fs["max_iterations"]),
        ),
        worker_stride=int(raw["seed"]["worker_stride"]),
    )


def loss_curve(util: np.ndarray) -> np.ndarray:
    """:func:`mplssim.sim.models.loss_fraction` with the documented float coercion.

    V1's ``loss_fraction`` allocates its output with ``zeros_like(util)``, so an
    integer-typed input silently truncates every fractional loss to zero. Engine
    utilization is always floating point so V1 execution is unaffected, but V2
    coerces explicitly rather than relying on that (realism audit, "Loss model").
    """
    return m.loss_fraction(np.asarray(util, dtype=float))


@dataclass
class TrafficModelV2(TrafficModel):
    """:class:`TrafficModel` with the two independent V2 child seed streams.

    V1 derives scenario materialization from ``seed + 7919`` and AR noise from
    ``seed``. V2 derives both from ``SeedSequence([episode_seed, n])`` so the
    streams are independent by construction rather than by an offset that
    happens not to collide. Nothing else about traffic generation changes:
    offered volume remains exogenous and identical for every controller facing
    the same episode seed.
    """

    scenario_rng: np.random.Generator | None = None
    ar_rng: np.random.Generator | None = None

    def __post_init__(self) -> None:  # noqa: D105 - replaces the V1 derivation
        if self.scenario_rng is None or self.ar_rng is None:
            raise ValueError("TrafficModelV2 requires explicit child generators")
        self._rng = self.ar_rng
        demand_ids = [d.id for d in self.config.demands]
        egress_ids = sorted({d.dst for d in self.config.demands})
        self.scenario = self.scenario.materialize(
            self.scenario_rng, demand_ids, egress_ids)
        self._noise = np.zeros(len(self.config.demands))
        self._precompute()


@dataclass(frozen=True)
class FlowSolution:
    """Self-consistent carried-flow state for one micro-tick.

    ``link_loss`` is the loss vector actually used to propagate traffic, not
    ``loss_curve(link_input/capacity)`` recomputed afterwards. The two agree to
    within twice the solver tolerance (``residual``), and using the propagating
    vector makes every conservation identity exact rather than approximate:
    link output == input*(1-loss), a demand's next-hop input == its previous-hop
    output, and delivered == final-hop output all hold bit-for-bit.
    """

    gross_link_load: np.ndarray
    link_input_load: np.ndarray
    link_loss: np.ndarray
    hop_input: np.ndarray        # (n_demands, max_hops) rate entering each hop
    delivered: np.ndarray        # (n_demands,) final-hop output
    survival: np.ndarray         # (n_demands,) product of (1 - loss) along the path
    iterations: int
    residual: float              # max |link_loss - loss_curve(link_input/capacity)|


class SimulationEngineV2:
    """Deterministic flow-level MPLS-TE engine with V2 semantics."""

    def __init__(
        self,
        topo: Topology,
        traffic_cfg: TrafficConfig,
        scenario: ScenarioSpec,
        episode_seed: int,
        cfg: EngineConfigV2 | None = None,
    ) -> None:
        self.topo = topo
        self.cfg = cfg or load_engine_config_v2()
        if self.cfg.version != CONFIG_VERSION:
            raise EngineConfigError(
                f"engine config version {self.cfg.version!r} != {CONFIG_VERSION!r}")
        self.scenario_name = scenario.name
        self.episode_seed = int(episode_seed)
        self.seed = self.episode_seed  # alias kept for trace/reporting parity with V1

        # ---- independent child RNG streams (P0-2) ----
        scenario_rng = np.random.default_rng(
            np.random.SeedSequence([self.episode_seed, SEED_STREAM_SCENARIO]))
        ar_rng = np.random.default_rng(
            np.random.SeedSequence([self.episode_seed, SEED_STREAM_AR]))
        self.traffic = TrafficModelV2(
            config=traffic_cfg, scenario=scenario, seed=self.episode_seed,
            scenario_rng=scenario_rng, ar_rng=ar_rng)
        self.scenario = self.traffic.scenario  # materialized

        # ---- role-valid candidate table (P0-3) ----
        self.candidate_table = build_candidate_table(
            topo, traffic_cfg.demands, k=self.cfg.k_paths,
            max_hop_factor=self.cfg.max_hop_factor,
            delay_factor=self.cfg.candidate_delay_factor,
            delay_additive_ms=self.cfg.candidate_delay_additive_ms,
        )
        self.demands: list[Demand] = []
        self._cand_links: list[list[np.ndarray]] = []
        self._cand_edges: list[list[frozenset[int]]] = []
        self._cand_cost: list[list[float]] = []
        for d in traffic_cfg.demands:
            cands = self.candidate_table[d.id]
            self.demands.append(Demand(id=d.id, src=d.src, dst=d.dst, cls=d.cls,
                                       base_mbps=d.base_mbps, index=d.index,
                                       candidate_paths=cands))
            self._cand_links.append(
                [np.array(topo.path_dlink_indices(p), dtype=np.int64) for p in cands])
            self._cand_edges.append([path_directed_edges(topo, p) for p in cands])
            self._cand_cost.append([path_admin_cost(topo, p) for p in cands])

        self.n_demands = len(self.demands)
        self.demand_by_id = {d.id: d for d in self.demands}
        self.k = self.cfg.k_paths

        # ---- per-link and per-demand constants ----
        self.capacity = np.array([dl.capacity_mbps for dl in topo.dlinks], dtype=float)
        self.prop_delay = np.array([dl.delay_ms for dl in topo.dlinks], dtype=float)
        self.total_capacity = float(np.sum(self.capacity))
        self._priorities = np.array([d.cls.priority for d in self.demands], dtype=float)
        self._q = self._priorities / 6.0
        self._protected = np.array([d.cls.protected for d in self.demands], dtype=bool)
        self._protected_idx = np.flatnonzero(self._protected)
        self._delay_sla = np.array([d.cls.max_latency_ms for d in self.demands], dtype=float)
        self._loss_sla = np.array([d.cls.max_loss_pct / 100.0 for d in self.demands],
                                  dtype=float)
        self._base_mbps = np.array([d.base_mbps for d in self.demands], dtype=float)
        self._q_protected_sum = float(np.sum(self._q[self._protected]))
        self._q_unprotected_sum = float(np.sum(self._q[~self._protected]))
        self._q_sum = float(np.sum(self._q))
        self._demand_arange = np.arange(self.n_demands)

        # Padded (demand, candidate, hop) link matrix plus its validity mask.
        # Padding slots are never read: every reduction is masked, so unlike V1
        # there is no reliance on an idempotent-reduction trick.
        self._max_hops = max(len(links) for per_d in self._cand_links for links in per_d)
        self._cand_pad = np.zeros((self.n_demands, self.k, self._max_hops), dtype=np.int64)
        self._cand_hopmask = np.zeros((self.n_demands, self.k, self._max_hops), dtype=bool)
        self._cand_hops = np.zeros((self.n_demands, self.k), dtype=np.int64)
        self._cand_prop_ms = np.zeros((self.n_demands, self.k), dtype=float)
        self._cand_exists = np.zeros((self.n_demands, self.k), dtype=bool)
        for d_idx, per_d in enumerate(self._cand_links):
            for p_idx, links in enumerate(per_d):
                if links.size == 0:
                    raise ValueError(f"demand {d_idx} candidate {p_idx} has no links")
                self._cand_pad[d_idx, p_idx, :links.size] = links
                self._cand_hopmask[d_idx, p_idx, :links.size] = True
                self._cand_hops[d_idx, p_idx] = links.size
                self._cand_prop_ms[d_idx, p_idx] = path_propagation_ms(
                    topo, self.demands[d_idx].candidate_paths[p_idx])
                self._cand_exists[d_idx, p_idx] = True

        # ---- dynamic state ----
        self.t_min: float = 0.0
        self.step_count: int = 0
        self.current_path = np.zeros(self.n_demands, dtype=np.int64)
        self.disconnected = np.zeros(self.n_demands, dtype=bool)
        self.link_up: dict[str, bool] = {lid: True for lid in topo.link_defs}
        self._dlink_up = np.ones(topo.n_dlinks, dtype=bool)

        # route history (spec, "Route history")
        self.path_age_steps = np.zeros(self.n_demands, dtype=np.int64)
        self.te_dwell_remaining = np.zeros(self.n_demands, dtype=np.int64)
        self.previous_te_path = np.full(self.n_demands, -1, dtype=np.int64)
        self.last_te_step = np.full(self.n_demands, -10**9, dtype=np.int64)

        # strictly separated accounting (spec, "V2 accounting")
        self.accepted_te_changes = 0
        self.rejected_te_requests = 0
        self.te_reversals = 0
        self.frr_changes = 0
        self.frr_disconnections = 0
        self.recovery_restorations = 0
        self.episode_totals = {
            "accepted_te_changes": 0, "rejected_te_requests": 0, "te_reversals": 0,
            "frr_changes": 0, "frr_disconnections": 0, "recovery_restorations": 0,
        }
        self.te_history: list[dict[str, Any]] = []
        self.frr_history: list[dict[str, Any]] = []
        self.restoration_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []

        # ---- telemetry (filled by _compute_tick) ----
        n_dl = topo.n_dlinks
        self.gross_link_load = np.zeros(n_dl)
        self.link_input_load = np.zeros(n_dl)
        self.link_util = np.zeros(n_dl)
        self.link_loss = np.zeros(n_dl)
        self.link_qdelay = np.zeros(n_dl)
        self.demand_offered = np.zeros(self.n_demands)
        self.demand_delivered = np.zeros(self.n_demands)
        self.demand_survival = np.ones(self.n_demands)
        self.demand_loss_fraction = np.zeros(self.n_demands)
        self.demand_delay = np.zeros(self.n_demands)
        self.demand_sla_ok = np.ones(self.n_demands, dtype=bool)
        self.hop_input = np.zeros((self.n_demands, self._max_hops))
        self.flow_solver_iterations = 0
        self.flow_solver_residual = 0.0

        # ---- reset-time transition (spec, "Exact transition order") ----
        self._process_events_at_reset()
        self._compute_tick()
        # Reset-time FRR belongs to the episode totals but not to the first
        # control interval, whose counters start clean at the first decision.
        self.accepted_te_changes = 0
        self.rejected_te_requests = 0
        self.te_reversals = 0
        self.frr_changes = 0
        self.frr_disconnections = 0
        self.recovery_restorations = 0

    # ------------------------------------------------------------------ paths
    def path_available(self, d_idx: int, p_idx: int) -> bool:
        """True when every physical link on a candidate is operational."""
        if not (0 <= d_idx < self.n_demands and 0 <= p_idx < self.k):
            return False
        if not self._cand_exists[d_idx, p_idx]:
            return False
        return bool(self._dlink_up[self._cand_links[d_idx][p_idx]].all())

    def candidate_available_matrix(self) -> np.ndarray:
        """(n_demands, k) availability for every candidate at once."""
        avail = np.where(self._cand_hopmask, self._dlink_up[self._cand_pad], True)
        return avail.all(axis=2) & self._cand_exists

    def _current_pad_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Padded hop-link matrix and mask for the demands' *current* paths.

        The mask additionally excludes disconnected demands, which carry no
        traffic and contribute to neither ledger.
        """
        rows = self._demand_arange
        pad = self._cand_pad[rows, self.current_path]
        mask = self._cand_hopmask[rows, self.current_path] & ~self.disconnected[:, None]
        return pad, mask

    # ------------------------------------------------------- gross projections
    def _gross_projection_base(self, d_idx: int) -> np.ndarray:
        """Gross link loads with demand ``d_idx`` removed from its current path.

        Adding the demand's full offered volume back onto a candidate then
        reproduces ``gross - demand_on_old_path + demand_on_candidate`` exactly,
        including for links shared by the old and new path, which go through the
        same ``(x - vol) + vol`` sequence either way.
        """
        base = self.gross_link_load.copy()
        if not self.disconnected[d_idx]:
            base[self._cand_links[d_idx][int(self.current_path[d_idx])]] -= \
                float(self.demand_offered[d_idx])
        return base

    def projected_gross_loads(self, d_idx: int, p_idx: int) -> np.ndarray:
        """Gross link loads as they would be after moving demand d to candidate p."""
        base = self._gross_projection_base(d_idx)
        base[self._cand_links[d_idx][p_idx]] += float(self.demand_offered[d_idx])
        return base

    def projected_gross_bottleneck(self, d_idx: int, p_idx: int) -> float:
        """Max projected gross utilization along candidate p after the move."""
        if not self._cand_exists[d_idx, p_idx]:
            return float("inf")
        base = self._gross_projection_base(d_idx)
        links = self._cand_links[d_idx][p_idx]
        return float(np.max((base[links] + float(self.demand_offered[d_idx]))
                            / self.capacity[links]))

    def projected_gross_bottleneck_matrix(self) -> np.ndarray:
        """(n_demands, k) projected gross bottleneck utilization for all candidates.

        Entry-for-entry identical to :meth:`projected_gross_bottleneck`; absent
        candidates read as ``inf`` and are masked by the caller.
        """
        bases = np.broadcast_to(self.gross_link_load,
                                (self.n_demands, self.topo.n_dlinks)).copy()
        for d_idx in range(self.n_demands):
            if not self.disconnected[d_idx]:
                bases[d_idx, self._cand_links[d_idx][int(self.current_path[d_idx])]] -= \
                    float(self.demand_offered[d_idx])
        rows = self._demand_arange[:, None, None]
        proj = ((bases[rows, self._cand_pad] + self.demand_offered[:, None, None])
                / self.capacity[self._cand_pad])
        proj = np.where(self._cand_hopmask, proj, -np.inf).max(axis=2)
        return np.where(self._cand_exists, proj, np.inf)

    # ---------------------------------------------------------------- actions
    def validate_te_action(self, d_idx: int, p_idx: int) -> tuple[bool, str]:
        """The single pure predicate behind both the mask and the validator.

        Conditions, in the spec's order (see "Validation and mask"):
        indices exist; candidate exists; every physical link is up; the
        candidate is not the live current path; TE dwell has expired; and for
        protected traffic the projected *gross* bottleneck is at most 1.0.
        """
        if not (0 <= d_idx < self.n_demands):
            return False, "unknown demand"
        if not (0 <= p_idx < self.k) or not self._cand_exists[d_idx, p_idx]:
            return False, "unknown candidate path"
        if int(self.current_path[d_idx]) == p_idx and not self.disconnected[d_idx]:
            return False, "already on this path"
        if not self.path_available(d_idx, p_idx):
            return False, "path traverses a failed link"
        if int(self.te_dwell_remaining[d_idx]) > 0:
            return False, f"te dwell {int(self.te_dwell_remaining[d_idx])} step(s) remaining"
        if self._protected[d_idx]:
            proj = self.projected_gross_bottleneck(d_idx, p_idx)
            if proj > self.cfg.protected_projected_max_util:
                return False, (f"projected gross utilization {proj:.6f} exceeds "
                               f"{self.cfg.protected_projected_max_util} for protected class")
        return True, "ok"

    def te_action_matrix(self) -> np.ndarray:
        """(n_demands, k) legality for every TE request, vectorized.

        Proved equivalent element-for-element to :meth:`validate_te_action` in
        tests/test_env_v2.py across reset, dwell, failure, disconnected and
        recovered states.
        """
        allowed = self.candidate_available_matrix()
        live = ~self.disconnected
        allowed[self._demand_arange[live], self.current_path[live]] = False
        allowed[self.te_dwell_remaining > 0] = False
        if self._protected_idx.size:
            proj = self.projected_gross_bottleneck_matrix()
            over = proj > self.cfg.protected_projected_max_util
            allowed[self._protected_idx] &= ~over[self._protected_idx]
        return allowed

    def _move(self, d_idx: int, p_idx: int, source: str) -> bool:
        """Place demand d on candidate p. Returns whether the router sequence changed."""
        old = int(self.current_path[d_idx])
        changed = (self.demands[d_idx].candidate_paths[old]
                   != self.demands[d_idx].candidate_paths[p_idx])
        self.current_path[d_idx] = p_idx
        self.disconnected[d_idx] = False
        if changed:
            self.path_age_steps[d_idx] = 0
        return changed

    def apply_te_action(self, d_idx: int, p_idx: int) -> dict[str, Any]:
        """Validate and, if legal, apply one agent TE request at this boundary.

        A rejected request changes no route, path age, dwell, previous-TE path,
        counter or RNG state — it only records the rejection.
        """
        ok, reason = self.validate_te_action(d_idx, p_idx)
        record: dict[str, Any] = {
            "step": self.step_count, "t_min": self.t_min,
            "demand_id": self.demands[d_idx].id if 0 <= d_idx < self.n_demands else None,
            "demand_idx": d_idx, "from_path": None, "to_path": p_idx,
            "accepted": ok, "reason": reason, "reversal": False,
            "volume_share": 0.0, "edge_divergence": 0.0,
        }
        if not ok:
            self.rejected_te_requests += 1
            self.episode_totals["rejected_te_requests"] += 1
            return record

        old = int(self.current_path[d_idx])
        record["from_path"] = old
        total_offered = float(np.sum(self.demand_offered))
        record["volume_share"] = float(
            self.demand_offered[d_idx] / max(total_offered, _EPS))
        e_old = self._cand_edges[d_idx][old]
        e_new = self._cand_edges[d_idx][p_idx]
        union = e_old | e_new
        record["edge_divergence"] = (
            len(e_old ^ e_new) / len(union) if union else 0.0)

        # A reversal is a move back onto the stored previous TE path within the
        # reversal window of the preceding accepted TE change. FRR clears that
        # slot, so a legitimate post-failure move is never labelled a reversal.
        reversal = bool(
            int(self.previous_te_path[d_idx]) == p_idx
            and self.step_count - int(self.last_te_step[d_idx])
            <= self.cfg.reversal_window_steps
        )
        record["reversal"] = reversal

        self._move(d_idx, p_idx, source="te")
        self.previous_te_path[d_idx] = old
        self.last_te_step[d_idx] = self.step_count
        self.te_dwell_remaining[d_idx] = self.cfg.minimum_te_dwell_steps

        self.accepted_te_changes += 1
        self.episode_totals["accepted_te_changes"] += 1
        if reversal:
            self.te_reversals += 1
            self.episode_totals["te_reversals"] += 1
        self.te_history.append(dict(record))
        return record

    # --------------------------------------------------------------- failures
    def set_link_state(self, link_id: str, up: bool) -> None:
        if link_id not in self.link_up:
            raise KeyError(f"unknown link {link_id}")
        if self.link_up[link_id] == up:
            return
        self.link_up[link_id] = up
        for dl in self.topo.dlinks:
            self._dlink_up[dl.index] = self.link_up[dl.undirected_id]
        if not up:
            self._fast_reroute()
        else:
            self._restore_disconnected()

    def _cheapest_live_candidate(self, d_idx: int) -> int | None:
        """Lowest-cost operational candidate, or None.

        Candidate indices are already ascending in
        ``(admin_cost, propagation_delay, router_tuple)``, so the first
        available index *is* the cheapest live candidate — deterministically,
        with no tie ambiguity.
        """
        for p_idx in range(self.k):
            if self.path_available(d_idx, p_idx):
                return p_idx
        return None

    def _fast_reroute(self) -> None:
        """Immediate deterministic local repair after a link goes down.

        Bypasses TE dwell and the protected-bandwidth filter on purpose:
        restoring connectivity outranks preserving a reservation in an already
        failed topology. Costs nothing, counts as no policy churn, and clears
        the previous-TE-path slot for every demand it touches.
        """
        for d_idx in range(self.n_demands):
            cur = int(self.current_path[d_idx])
            if not self.disconnected[d_idx] and self.path_available(d_idx, cur):
                continue
            p_idx = self._cheapest_live_candidate(d_idx)
            if p_idx is None:
                if not self.disconnected[d_idx]:
                    self.frr_disconnections += 1
                    self.episode_totals["frr_disconnections"] += 1
                    self.frr_history.append({
                        "step": self.step_count, "t_min": self.t_min,
                        "demand_id": self.demands[d_idx].id, "demand_idx": d_idx,
                        "from_path": cur, "to_path": None, "event": "disconnected",
                    })
                self.disconnected[d_idx] = True
                self.previous_te_path[d_idx] = -1
                continue
            self._move(d_idx, p_idx, source="frr")
            self.previous_te_path[d_idx] = -1
            self.frr_changes += 1
            self.episode_totals["frr_changes"] += 1
            self.frr_history.append({
                "step": self.step_count, "t_min": self.t_min,
                "demand_id": self.demands[d_idx].id, "demand_idx": d_idx,
                "from_path": cur, "to_path": p_idx, "event": "reroute",
            })

    def _restore_disconnected(self) -> None:
        """After a link recovers, restore only demands that are still down.

        A connected demand is never moved just because a cheaper link came
        back; that is a central-reoptimization decision for the controller, not
        a protection event.
        """
        for d_idx in np.flatnonzero(self.disconnected):
            d_idx = int(d_idx)
            p_idx = self._cheapest_live_candidate(d_idx)
            if p_idx is None:
                continue
            cur = int(self.current_path[d_idx])
            self._move(d_idx, p_idx, source="restore")
            self.previous_te_path[d_idx] = -1
            self.recovery_restorations += 1
            self.episode_totals["recovery_restorations"] += 1
            self.restoration_history.append({
                "step": self.step_count, "t_min": self.t_min,
                "demand_id": self.demands[d_idx].id, "demand_idx": d_idx,
                "from_path": cur, "to_path": p_idx, "event": "restored",
            })

    # ----------------------------------------------------------------- events
    def _selected_link_events(self, t_from: float, t_to: float) -> list[dict[str, Any]]:
        """Link events in the right-closed window ``(t_from, t_to]`` (P0-1).

        Ordered by ``(t_min, position in the scenario event list)`` so several
        events at the same instant always apply in one documented order.
        """
        picked = [
            (ev["t_min"], i, ev) for i, ev in enumerate(self.scenario.events)
            if ev["type"] in ("link_down", "link_up") and t_from < ev["t_min"] <= t_to
        ]
        picked.sort(key=lambda x: (x[0], x[1]))
        return [ev for _, _, ev in picked]

    def _process_link_events(self, t_from: float, t_to: float) -> None:
        for ev in self._selected_link_events(t_from, t_to):
            self.set_link_state(ev["link"], ev["type"] == "link_up")

    def _process_events_at_reset(self) -> None:
        """Apply every event scheduled at or before t=0, then run FRR.

        The right-closed window ``(old_t, new_t]`` used by every micro-tick can
        never contain t=0, so reset owns it explicitly. Together they make each
        event fire exactly once.
        """
        picked = [
            (ev["t_min"], i, ev) for i, ev in enumerate(self.scenario.events)
            if ev["type"] in ("link_down", "link_up") and ev["t_min"] <= 0.0
        ]
        picked.sort(key=lambda x: (x[0], x[1]))
        for _, _, ev in picked:
            self.set_link_state(ev["link"], ev["type"] == "link_up")

    # ------------------------------------------------------------ flow solver
    def _accumulate_gross(self, offered_routed: np.ndarray, pad: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        """Full offered rate on every hop of every selected path."""
        gross = np.zeros(self.topo.n_dlinks)
        for h in range(pad.shape[1]):
            sel = mask[:, h]
            if not sel.any():
                break
            gross += np.bincount(pad[sel, h], weights=offered_routed[sel],
                                 minlength=self.topo.n_dlinks)
        return gross

    def _forward_propagate(self, ell: np.ndarray, offered_routed: np.ndarray,
                           pad: np.ndarray, mask: np.ndarray):
        """One sequential pass of ``x[d,h+1] = x[d,h]*(1 - ell[e(d,h)])``.

        Returns the aggregated link input load, per-hop input rates, delivered
        rates and path survival fractions. Survival is accumulated separately
        from the rate so it stays meaningful when a demand offers zero traffic.
        """
        n_dl = self.topo.n_dlinks
        x = offered_routed.copy()
        survival = np.ones(self.n_demands)
        hop_in = np.zeros_like(pad, dtype=float)
        link_in = np.zeros(n_dl)
        for h in range(pad.shape[1]):
            sel = mask[:, h]
            if not sel.any():
                break
            li = pad[sel, h]
            xr = x[sel]
            hop_in[sel, h] = xr
            link_in += np.bincount(li, weights=xr, minlength=n_dl)
            keep = 1.0 - ell[li]
            x[sel] = xr * keep
            survival[sel] = survival[sel] * keep
        return link_in, hop_in, x, survival

    def solve_flow(self, offered_routed: np.ndarray, pad: np.ndarray,
                   mask: np.ndarray) -> FlowSolution:
        """Damped fixed-point solution of the carried-flow equations.

        Exactly the iteration in docs/MPLS_SIMULATION_REALISM_AUDIT.md: start
        from the conservative gross-load loss, forward-propagate, aggregate,
        recompute a candidate loss, and damp. The returned state is the
        *pre-update* iterate together with the link loads it produced, so it is
        internally consistent; the stopping rule bounds
        ``|loss_curve(L/cap) - ell|`` by twice the tolerance.
        """
        fs = self.cfg.flow_solver
        cap = self.capacity
        gross = self._accumulate_gross(offered_routed, pad, mask)
        ell = loss_curve(gross / cap)
        for it in range(1, fs.max_iterations + 1):
            link_in, hop_in, delivered, survival = self._forward_propagate(
                ell, offered_routed, pad, mask)
            ell_candidate = loss_curve(link_in / cap)
            ell_new = fs.damping * ell + (1.0 - fs.damping) * ell_candidate
            if float(np.max(np.abs(ell_new - ell))) <= fs.tolerance:
                return FlowSolution(
                    gross_link_load=gross, link_input_load=link_in, link_loss=ell,
                    hop_input=hop_in, delivered=delivered, survival=survival,
                    iterations=it,
                    residual=float(np.max(np.abs(ell_candidate - ell))),
                )
            ell = ell_new
        raise FlowSolverError(
            f"carried-flow fixed point did not converge in {fs.max_iterations} "
            f"iterations at t={self.t_min} min (scenario {self.scenario_name}, "
            f"seed {self.episode_seed}); refusing to publish stale telemetry")

    # ------------------------------------------------------------------ ticks
    def _compute_tick(self) -> dict[str, Any]:
        """Offered traffic, carried flow and all derived telemetry for minute t."""
        offered = self.traffic.volumes(self.t_min)
        self.demand_offered = offered
        offered_routed = np.where(self.disconnected, 0.0, offered)
        pad, mask = self._current_pad_mask()

        sol = self.solve_flow(offered_routed, pad, mask)
        self.gross_link_load = sol.gross_link_load
        self.link_input_load = sol.link_input_load
        self.link_util = sol.link_input_load / self.capacity
        self.link_loss = sol.link_loss
        self.link_qdelay = m.queue_delay_ms(self.link_util)
        self.hop_input = sol.hop_input
        self.flow_solver_iterations = sol.iterations
        self.flow_solver_residual = sol.residual

        self.demand_delivered = np.where(self.disconnected, 0.0, sol.delivered)
        self.demand_survival = sol.survival
        self.demand_loss_fraction = np.where(self.disconnected, 1.0, 1.0 - sol.survival)

        link_delay = self.prop_delay + self.link_qdelay
        delay = np.zeros(self.n_demands)
        for h in range(pad.shape[1]):
            sel = mask[:, h]
            if not sel.any():
                break
            delay[sel] += link_delay[pad[sel, h]]
        delay += m.PROC_DELAY_MS * mask.sum(axis=1)
        self.demand_delay = np.where(self.disconnected, 0.0, delay)
        self.demand_sla_ok = (
            ~self.disconnected
            & (self.demand_delay <= self._delay_sla)
            & (self.demand_loss_fraction <= self._loss_sla)
        )
        return self.tick_metrics()

    def tick_metrics(self) -> dict[str, Any]:
        """Metrics for the current telemetry snapshot (one micro-tick or boundary).

        The reward-bearing quantities are the first block; the rest are
        reporting aggregates that no reward term consumes.
        """
        offered = self.demand_offered
        delivered = self.demand_delivered
        disc = self.disconnected
        connected = ~disc

        delay_excess = np.where(
            connected,
            np.maximum(0.0, self.demand_delay - self._delay_sla) / self._delay_sla,
            0.0)
        loss_excess = np.where(
            connected,
            np.maximum(0.0, self.demand_loss_fraction - self._loss_sla)
            / np.maximum(self._loss_sla, 1e-6),
            0.0)
        h_delay = delay_excess / (1.0 + delay_excess)
        h_loss = loss_excess / (1.0 + loss_excess)
        sla_sum = float(np.sum(self._q * connected * (h_delay + 2.0 * h_loss) / 3.0))

        prot_disc = (float(np.sum(self._q[self._protected] * disc[self._protected]))
                     / self._q_protected_sum) if self._q_protected_sum > 0 else 0.0
        unprot_disc = (float(np.sum(self._q[~self._protected] * disc[~self._protected]))
                       / self._q_unprotected_sum) if self._q_unprotected_sum > 0 else 0.0
        overload = float(
            np.sum(np.maximum(self.link_input_load - self.capacity, 0.0))
            / self.total_capacity)

        offered_sum = float(np.sum(offered))
        delivered_sum = float(np.sum(delivered))
        active = offered > 1e-9
        return {
            # reward-bearing
            "protected_disconnect": prot_disc,
            "unprotected_disconnect": unprot_disc,
            "sla_severity_sum": sla_sum,
            "offered_mbps": offered_sum,
            "delivered_mbps": delivered_sum,
            "max_util": float(np.max(self.link_util)),
            "overload_ratio": overload,
            # reporting only
            "t_min": self.t_min,
            "mean_util": float(np.mean(self.link_util)),
            "util_std": float(np.std(self.link_util)),
            "gross_max_util": float(np.max(self.gross_link_load / self.capacity)),
            "mean_delay_ms": (float(np.sum(self.demand_delay[active] * offered[active])
                                    / np.sum(offered[active])) if np.any(active) else 0.0),
            "max_delay_ms": float(np.max(self.demand_delay)),
            "loss_ratio": ((offered_sum - delivered_sum) / offered_sum
                           if offered_sum > 0 else 0.0),
            "delivered_ratio": (delivered_sum / offered_sum if offered_sum > 0 else 1.0),
            "sla_violations": int(np.sum(~self.demand_sla_ok)),
            "sla_violation_fraction": float(np.mean(~self.demand_sla_ok)),
            "congested_links": int(np.sum(self.link_util >= m.CONGESTION_UTIL)),
            "disconnected_demands": int(np.sum(disc)),
            "protected_disconnected_demands": int(np.sum(disc[self._protected])),
            "flow_solver_iterations": self.flow_solver_iterations,
            "flow_solver_residual": self.flow_solver_residual,
        }

    def boundary_metrics(self) -> dict[str, Any]:
        """Current-boundary metrics in the shape the utility function consumes.

        Used for the potential term, which evaluates the same utility form on a
        single boundary snapshot rather than on an aggregated interval.
        """
        tick = self.tick_metrics()
        return {
            "protected_disconnect": tick["protected_disconnect"],
            "unprotected_disconnect": tick["unprotected_disconnect"],
            "sla_severity": tick["sla_severity_sum"] / self._q_sum,
            "delivered_ratio": tick["delivered_ratio"],
            "max_util": tick["max_util"],
            "overload_ratio": tick["overload_ratio"],
        }

    # ------------------------------------------------------------------ steps
    def step_interval(self) -> dict[str, Any]:
        """Advance one control interval of ``micro_ticks`` one-minute ticks.

        Transition order per micro-tick (spec, "Exact transition order"):
        advance the clock, advance AR noise once, apply link events in
        ``(old_t, new_t]`` including immediate FRR/restoration, compute offered
        traffic at the new time, solve the carried flow, derive telemetry.
        """
        n = self.cfg.micro_ticks_per_interval
        dt = self.cfg.control_interval_min / n
        ticks: list[dict[str, Any]] = []
        for _ in range(n):
            old_t = self.t_min
            self.t_min = old_t + dt
            self.traffic.advance_noise()
            self._process_link_events(old_t, self.t_min)
            ticks.append(self._compute_tick())
        self.step_count += 1

        agg = self.aggregate_interval(ticks)
        self.metrics_history.append(agg)

        # Route bookkeeping happens after the interval is scored: dwell counts
        # down one step per completed interval and path age counts up, so a
        # move accepted at step s is next legal at step s+dwell.
        self.te_dwell_remaining = np.maximum(0, self.te_dwell_remaining - 1)
        self.path_age_steps += 1

        self.accepted_te_changes = 0
        self.rejected_te_requests = 0
        self.te_reversals = 0
        self.frr_changes = 0
        self.frr_disconnections = 0
        self.recovery_restorations = 0
        return agg

    def aggregate_interval(self, ticks: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate micro-ticks exactly as the reward definition requires.

        Connectivity fractions take the worst tick so a short protected outage
        cannot average away; delivered traffic is a ratio of sums, not a mean of
        ratios; overload is the mean tick overload; max utilization is the worst
        tick. Everything else is a reporting mean.
        """
        n = len(ticks)
        offered = sum(t["offered_mbps"] for t in ticks)
        delivered = sum(t["delivered_mbps"] for t in ticks)
        agg: dict[str, Any] = {
            "protected_disconnect": max(t["protected_disconnect"] for t in ticks),
            "unprotected_disconnect": max(t["unprotected_disconnect"] for t in ticks),
            "sla_severity": sum(t["sla_severity_sum"] for t in ticks) / (n * self._q_sum),
            "delivered_ratio": (delivered / offered) if offered > 0 else 1.0,
            "max_util": max(t["max_util"] for t in ticks),
            "overload_ratio": sum(t["overload_ratio"] for t in ticks) / n,
        }
        for key in ("mean_util", "util_std", "mean_delay_ms", "loss_ratio",
                    "sla_violation_fraction", "flow_solver_residual"):
            agg[key] = float(np.mean([t[key] for t in ticks]))
        for key in ("max_delay_ms", "gross_max_util", "sla_violations",
                    "congested_links", "disconnected_demands",
                    "protected_disconnected_demands"):
            agg[key] = max(t[key] for t in ticks)
        agg["offered_mbps"] = offered / n
        agg["delivered_mbps"] = delivered / n
        agg["flow_solver_iterations_max"] = max(t["flow_solver_iterations"] for t in ticks)
        agg["t_min"] = self.t_min
        agg["hour"] = (self.scenario.start_hour + self.t_min / 60.0) % 24.0
        agg["step"] = self.step_count
        agg["n_demands"] = self.n_demands
        agg["accepted_te_changes"] = self.accepted_te_changes
        agg["rejected_te_requests"] = self.rejected_te_requests
        agg["te_reversals"] = self.te_reversals
        agg["frr_changes"] = self.frr_changes
        agg["frr_disconnections"] = self.frr_disconnections
        agg["recovery_restorations"] = self.recovery_restorations
        agg["failed_links"] = [lid for lid, up in self.link_up.items() if not up]
        return agg

    @property
    def done(self) -> bool:
        return self.t_min >= self.scenario.duration_min - 1e-9

    @property
    def all_disconnected(self) -> bool:
        """Reported for diagnostics only.

        V2 never terminates on this: episodes run to scenario duration so
        recovery stays observable and paired controllers keep equal horizons.
        """
        return bool(np.all(self.disconnected))

    # ------------------------------------------------------------------ clone
    def clone(self) -> "SimulationEngineV2":
        """Fully isolated deep copy (counterfactual analysis)."""
        return copy.deepcopy(self)

    #: Lifetime-constant attributes a light clone may share. Anything not listed
    #: is copied by :meth:`fast_clone`; tests/test_env_v2.py asserts that no
    #: mutable container escapes this list.
    _SHAREABLE_ATTRS = frozenset({
        "topo", "cfg", "scenario_name", "seed", "episode_seed", "scenario",
        "n_demands", "k", "demands", "demand_by_id", "candidate_table",
        "_cand_links", "_cand_edges", "_cand_cost", "capacity", "prop_delay",
        "total_capacity", "_priorities", "_q", "_protected", "_protected_idx",
        "_delay_sla", "_loss_sla", "_base_mbps", "_q_protected_sum",
        "_q_unprotected_sum", "_q_sum", "_demand_arange", "_max_hops",
        "_cand_pad", "_cand_hopmask", "_cand_hops", "_cand_prop_ms",
        "_cand_exists",
    })

    def fast_clone(self) -> "SimulationEngineV2":
        """Cheap clone that shares immutable data and copies all mutable state."""
        cl = object.__new__(SimulationEngineV2)
        state = dict(self.__dict__)
        for attr in ("current_path", "disconnected", "_dlink_up", "path_age_steps",
                     "te_dwell_remaining", "previous_te_path", "last_te_step",
                     "gross_link_load", "link_input_load", "link_util", "link_loss",
                     "link_qdelay", "demand_offered", "demand_delivered",
                     "demand_survival", "demand_loss_fraction", "demand_delay",
                     "demand_sla_ok", "hop_input"):
            state[attr] = self.__dict__[attr].copy()
        state["link_up"] = dict(self.link_up)
        state["episode_totals"] = dict(self.episode_totals)
        state["te_history"] = [dict(r) for r in self.te_history]
        state["frr_history"] = [dict(r) for r in self.frr_history]
        state["restoration_history"] = [dict(r) for r in self.restoration_history]
        state["metrics_history"] = list(self.metrics_history)

        traffic = copy.copy(self.traffic)
        traffic._noise = self.traffic._noise.copy()
        traffic._rng = copy.deepcopy(self.traffic._rng)
        traffic.ar_rng = traffic._rng
        traffic.scenario_rng = copy.deepcopy(self.traffic.scenario_rng)
        state["traffic"] = traffic

        cl.__dict__.update(state)
        return cl


def engine_config_v2_with(**overrides) -> EngineConfigV2:
    """Config copy with fields replaced. Used by ablations and synthetic tests."""
    return replace(load_engine_config_v2(), **overrides)
