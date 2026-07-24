"""Conventional traffic-engineering baselines.

All controllers share one interface: ``decide(engine) -> list[(demand_idx,
path_idx)]`` called once per control interval, BEFORE the engine advances.
They only read current telemetry (no future information) and the engine
enforces the same validity rules applied to the RL agent.

Baselines:
  * StaticShortestPathController — every demand pinned to its lowest-admin-cost
    candidate (index 0). Reacts to failures only via the engine's FRR.
  * GreedyUtilizationController — utilization-aware local heuristic: when a
    link exceeds a trigger threshold, move the largest demand crossing the
    most-utilized link to the candidate with the lowest bottleneck
    utilization, subject to an improvement margin and per-demand cooldown.
  * CspfController — constraint-based periodic reoptimization: every N
    intervals, demands are (re)placed in priority order onto the feasible
    candidate with sufficient available bandwidth and minimal admin cost,
    with a hysteresis margin so paths do not churn.
  * RandomController — uniformly random valid action (RL sanity floor).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mplssim.factory import get_baseline_config
from mplssim.sim.engine import SimulationEngine


class Controller(Protocol):
    name: str

    def decide(self, eng: SimulationEngine) -> list[tuple[int, int]]: ...

    def reset(self) -> None: ...


class StaticShortestPathController:
    name = "static"

    def decide(self, eng: SimulationEngine) -> list[tuple[int, int]]:
        # Move demands back to their shortest path once it is available again
        # (models a plain IGP: always prefer the lowest-metric route).
        out = []
        for d_idx in range(eng.n_demands):
            if int(eng.current_path[d_idx]) != 0 and eng.path_available(d_idx, 0):
                out.append((d_idx, 0))
        return out

    def reset(self) -> None:
        pass


class GreedyUtilizationController:
    name = "greedy"

    def __init__(self) -> None:
        cfg = get_baseline_config()["greedy_utilization"]
        self.trigger = float(cfg["trigger_threshold"])
        self.margin = float(cfg["improvement_margin"])
        self.max_moves = int(cfg["max_reroutes_per_interval"])

    def decide(self, eng: SimulationEngine) -> list[tuple[int, int]]:
        worst_link = int(np.argmax(eng.link_util))
        if float(eng.link_util[worst_link]) < self.trigger:
            return []
        # demands crossing the worst link, largest volume first
        crossing = [
            d for d in range(eng.n_demands)
            if not eng.disconnected[d]
            and worst_link in eng._path_links[d][int(eng.current_path[d])]
        ]
        crossing.sort(key=lambda d: -float(eng.demand_volumes[d]))
        moves: list[tuple[int, int]] = []
        for d in crossing:
            cur_bu = eng.path_bottleneck_util(d, int(eng.current_path[d]))
            best_p, best_bu = -1, cur_bu - self.margin
            for p in range(len(eng.demands[d].candidate_paths)):
                if p == int(eng.current_path[d]) or not eng.path_available(d, p):
                    continue
                bu = eng.path_bottleneck_util(d, p)
                if bu < best_bu:
                    best_p, best_bu = p, bu
            if best_p >= 0:
                ok, _ = eng.validate_action(d, best_p, source="greedy")
                if ok:
                    moves.append((d, best_p))
                    if len(moves) >= self.max_moves:
                        break
        return moves

    def reset(self) -> None:
        pass


class CspfController:
    name = "cspf"

    def __init__(self) -> None:
        cfg = get_baseline_config()["cspf"]
        self.period = int(cfg["reopt_period_steps"])
        self.headroom = float(cfg["reservation_headroom"])
        self.hysteresis = float(cfg["hysteresis_margin"])
        self.max_moves = int(cfg["max_reroutes_per_reopt"])

    def decide(self, eng: SimulationEngine) -> list[tuple[int, int]]:
        if eng.step_count % self.period != 0:
            return []
        # Build a reservation ledger from scratch: place demands in priority
        # order (then volume) onto the cheapest candidate whose bottleneck
        # reservation stays under headroom * capacity.
        order = sorted(
            range(eng.n_demands),
            key=lambda d: (-eng.demands[d].cls.priority, -float(eng.demand_volumes[d])),
        )
        reserved = np.zeros(eng.topo.n_dlinks)
        placement: dict[int, int] = {}
        for d in order:
            vol = float(eng.demand_volumes[d])
            cur = int(eng.current_path[d])
            chosen, chosen_cost = None, float("inf")
            for p in np.argsort(eng._path_costs[d]):
                p = int(p)
                if not eng.path_available(d, p):
                    continue
                links = eng._path_links[d][p]
                if np.all(reserved[links] + vol <= self.headroom * eng.capacity[links]):
                    chosen, chosen_cost = p, eng._path_costs[d][p]
                    break
            if chosen is None:
                # no candidate satisfies the constraint: keep current if alive,
                # otherwise least-loaded available candidate (best effort)
                avail = [p for p in range(len(eng.demands[d].candidate_paths))
                         if eng.path_available(d, p)]
                chosen = cur if eng.path_available(d, cur) else (avail[0] if avail else cur)
            placement[d] = chosen
            reserved[eng._path_links[d][chosen]] += vol

        moves: list[tuple[int, int]] = []
        for d, p in placement.items():
            cur = int(eng.current_path[d])
            if p == cur:
                continue
            # hysteresis: only move if it meaningfully improves the bottleneck
            if eng.path_available(d, cur):
                gain = eng.path_bottleneck_util(d, cur) - eng.path_bottleneck_util(d, p)
                if gain < self.hysteresis:
                    continue
            moves.append((d, p))
            if len(moves) >= self.max_moves:
                break
        return moves

    def reset(self) -> None:
        pass


class RandomController:
    """Random-policy sanity floor for RL.

    Exact rule (documented so code and report agree): with probability
    ``noop_prob`` (0.5) the controller takes no action; otherwise it samples
    uniformly from the set of CURRENTLY VALID reroute actions — the same
    validity mask the RL policy sees (failed paths, cooldowns, same-path and
    protected-bandwidth checks all excluded). If no reroute is valid, it
    falls back to no-op.
    """

    name = "random"

    def __init__(self, seed: int = 0, noop_prob: float = 0.5) -> None:
        self.rng = np.random.default_rng(seed)
        self.noop_prob = noop_prob

    def valid_actions(self, eng: SimulationEngine) -> list[tuple[int, int]]:
        return [
            (d, p)
            for d in range(eng.n_demands)
            for p in range(len(eng.demands[d].candidate_paths))
            if eng.validate_action(d, p, source="rl")[0]
        ]

    def decide(self, eng: SimulationEngine) -> list[tuple[int, int]]:
        if self.rng.random() < self.noop_prob:
            return []
        valid = self.valid_actions(eng)
        if not valid:
            return []
        return [valid[int(self.rng.integers(0, len(valid)))]]

    def reset(self) -> None:
        pass


def make_baseline(name: str, seed: int = 0) -> Controller:
    if name == "static":
        return StaticShortestPathController()
    if name == "greedy":
        return GreedyUtilizationController()
    if name == "cspf":
        return CspfController()
    if name == "random":
        return RandomController(seed)
    raise KeyError(f"unknown baseline '{name}'")
