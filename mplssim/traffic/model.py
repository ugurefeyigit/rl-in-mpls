"""Traffic demand model: classes, diurnal profiles, scenarios, stochastic volumes.

Offered traffic is fully exogenous: it depends only on (scenario, seed, time),
never on routing decisions. This is what makes paired algorithm comparisons
valid — every controller faces byte-identical offered demand.

Volume of demand d at simulated wall-clock hour h:

    v_d(t) = base_mbps
             * profile_{class(d)}(h)          # diurnal curve, linear interp
             * scenario.demand_multiplier
             * event_factor_d(t)              # bursts / flash crowds / surges
             * noise_d(t)                     # per-demand AR(1), seeded

AR(1) noise: n_t = phi * n_{t-1} + sigma_d * eps,  eps ~ N(0,1)
             noise multiplier = clip(1 + n_t, 0.4, 2.0)
             sigma_d = noise_sigma(scenario) * burstiness(class) .
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mplssim.core.model import Demand, TrafficClass
from mplssim.core.topology import CONFIG_DIR


@dataclass
class ScenarioSpec:
    name: str
    description: str
    start_hour: float
    duration_min: int
    demand_multiplier: float
    noise_sigma: float
    events: list[dict[str, Any]]
    randomize: dict[str, Any] | None = None

    def materialize(self, rng: np.random.Generator, demand_ids: list[str],
                    egress_ids: list[str]) -> "ScenarioSpec":
        """Return a concrete copy: if this spec has a `randomize` block, draw
        the actual bursts/failures from ``rng`` (used by random_day training).

        Demand and egress candidates come from the loaded configuration —
        nothing about the demand count or PE naming is assumed here.
        """
        if not self.randomize:
            return self
        spec = copy.deepcopy(self)
        spec.randomize = None
        r = self.randomize
        events: list[dict[str, Any]] = []
        n_bursts = int(rng.integers(r["n_bursts"][0], r["n_bursts"][1] + 1))
        for _ in range(n_bursts):
            events.append({
                "t_min": int(rng.integers(30, self.duration_min - 60)),
                "type": "burst",
                "demands": [demand_ids[int(rng.integers(0, len(demand_ids)))]],
                "factor": float(rng.uniform(*r["burst_factor"])),
                "duration_min": int(rng.uniform(*r["burst_duration_min"])),
            })
        if rng.random() < r["failure_prob"]:
            link = str(rng.choice(r["failure_candidates"]))
            t0 = int(rng.integers(60, self.duration_min - 200))
            dur = int(rng.uniform(*r["failure_duration_min"]))
            events.append({"t_min": t0, "type": "link_down", "link": link})
            events.append({"t_min": t0 + dur, "type": "link_up", "link": link})
        if rng.random() < r.get("flash_crowd_prob", 0.0):
            events.append({
                "t_min": int(rng.integers(60, self.duration_min - 120)),
                "type": "flash_crowd",
                "dst": str(rng.choice(egress_ids)),
                "factor": float(rng.uniform(*r["flash_crowd_factor"])),
                "duration_min": int(rng.integers(45, 100)),
            })
        spec.events = sorted(events, key=lambda e: e["t_min"])
        return spec


def load_scenarios(path: Path | None = None) -> dict[str, ScenarioSpec]:
    raw = yaml.safe_load((path or CONFIG_DIR / "scenarios.yaml").read_text(encoding="utf-8"))
    out: dict[str, ScenarioSpec] = {}
    for name, s in raw["scenarios"].items():
        out[name] = ScenarioSpec(
            name=name,
            description=s.get("description", ""),
            start_hour=float(s["start_hour"]),
            duration_min=int(s["duration_min"]),
            demand_multiplier=float(s["demand_multiplier"]),
            noise_sigma=float(s["noise_sigma"]),
            events=list(s.get("events", [])),
            randomize=s.get("randomize"),
        )
    return out


@dataclass
class TrafficConfig:
    classes: dict[str, TrafficClass]
    profiles: dict[str, list[tuple[float, float]]]
    demands: list[Demand]


def load_traffic_config(path: Path | None = None) -> TrafficConfig:
    raw = yaml.safe_load((path or CONFIG_DIR / "traffic_classes.yaml").read_text(encoding="utf-8"))
    classes = {
        name: TrafficClass(name=name, **{k: v for k, v in c.items()})
        for name, c in raw["classes"].items()
    }
    profiles = {name: [(float(h), float(m)) for h, m in pts] for name, pts in raw["profiles"].items()}
    demands = [
        Demand(
            id=d["id"], src=d["src"], dst=d["dst"],
            cls=classes[d["class"]], base_mbps=float(d["base_mbps"]), index=i,
        )
        for i, d in enumerate(raw["demands"])
    ]
    return TrafficConfig(classes=classes, profiles=profiles, demands=demands)


def profile_value(points: list[tuple[float, float]], hour: float) -> float:
    """Linear interpolation over (hour, multiplier) control points, wrap at 24 h."""
    h = hour % 24.0
    for (h0, m0), (h1, m1) in zip(points, points[1:]):
        if h0 <= h <= h1:
            if h1 == h0:
                return m1
            return m0 + (m1 - m0) * (h - h0) / (h1 - h0)
    return points[-1][1]


@dataclass
class TrafficModel:
    """Seeded stochastic traffic generator for one episode of one scenario."""

    config: TrafficConfig
    scenario: ScenarioSpec
    seed: int
    _noise: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False)
    phi: float = 0.9  # AR(1) persistence

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        demand_ids = [d.id for d in self.config.demands]
        egress_ids = sorted({d.dst for d in self.config.demands})
        self.scenario = self.scenario.materialize(
            np.random.default_rng(self.seed + 7919), demand_ids, egress_ids)
        self._noise = np.zeros(len(self.config.demands))
        self._precompute()

    def _precompute(self) -> None:
        """Cache the per-demand constants that ``volumes`` would otherwise
        rebuild on every micro-tick.

        Everything here is derived from the (already materialized) scenario and
        the traffic configuration, both of which are fixed for the lifetime of
        the model, so caching cannot change any produced value.
        """
        demands = self.config.demands
        self._base_mbps = np.array([d.base_mbps for d in demands], dtype=float)
        self._sigmas = np.array(
            [self.scenario.noise_sigma * d.cls.burstiness for d in demands], dtype=float)

        # distinct diurnal profiles: 17 demands share a handful of curves, so
        # interpolate once per curve per tick instead of once per demand
        names: list[str] = []
        for d in demands:
            if d.cls.profile not in names:
                names.append(d.cls.profile)
        self._profile_points = [self.config.profiles[n] for n in names]
        self._profile_index = np.array([names.index(d.cls.profile) for d in demands])

        # per-event demand masks, in scenario event order (order matters: the
        # factors are multiplied in sequence, exactly as the scalar loop did)
        self._volume_events: list[tuple[dict, np.ndarray]] = []
        for ev in self.scenario.events:
            if ev["type"] not in ("burst", "flash_crowd", "multiplier"):
                continue
            if ev["type"] == "burst":
                sel = np.array([d.id in ev["demands"] for d in demands])
            elif ev["type"] == "flash_crowd":
                sel = np.array([d.dst == ev["dst"] for d in demands])
            else:
                sel = np.ones(len(demands), dtype=bool)
            self._volume_events.append((ev, sel))

    def advance_noise(self) -> None:
        """One AR(1) step for every demand (call once per micro-tick)."""
        eps = self._rng.standard_normal(len(self.config.demands))
        self._noise = self.phi * self._noise + self._sigmas * eps

    def event_factor(self, demand: Demand, t_min: float) -> float:
        f = 1.0
        for ev in self.scenario.events:
            if ev["type"] in ("burst", "flash_crowd", "multiplier"):
                if not (ev["t_min"] <= t_min < ev["t_min"] + ev["duration_min"]):
                    continue
                if ev["type"] == "burst" and demand.id in ev["demands"]:
                    f *= ev["factor"]
                elif ev["type"] == "flash_crowd" and demand.dst == ev["dst"]:
                    f *= ev["factor"]
                elif ev["type"] == "multiplier":
                    f *= ev["factor"]
        return f

    def event_factors(self, t_min: float) -> np.ndarray:
        """Vectorized :meth:`event_factor` for every demand at once.

        Factors are applied in scenario event order, so each demand sees the
        same product in the same sequence as the per-demand scalar version.
        """
        out = np.ones(len(self.config.demands))
        for ev, sel in self._volume_events:
            if ev["t_min"] <= t_min < ev["t_min"] + ev["duration_min"]:
                out[sel] *= ev["factor"]
        return out

    def volumes(self, t_min: float) -> np.ndarray:
        """Offered Mbps per demand at simulated minute t (uses current noise state).

        Same arithmetic, same operand order as the original per-demand loop:
        ((((base * profile) * demand_multiplier) * event) * noise).
        """
        hour = self.scenario.start_hour + t_min / 60.0
        profs = np.array([profile_value(pts, hour) for pts in self._profile_points]
                         )[self._profile_index]
        noise_mult = np.clip(1.0 + self._noise, 0.4, 2.0)
        return (self._base_mbps * profs * self.scenario.demand_multiplier
                * self.event_factors(t_min) * noise_mult)

    def link_events_at(self, t_min_from: float, t_min_to: float) -> list[dict[str, Any]]:
        """Scripted link_down/link_up events with t in [from, to)."""
        return [
            ev for ev in self.scenario.events
            if ev["type"] in ("link_down", "link_up") and t_min_from <= ev["t_min"] < t_min_to
        ]
