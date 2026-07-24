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

    def materialize(self, rng: np.random.Generator) -> "ScenarioSpec":
        """Return a concrete copy: if this spec has a `randomize` block, draw
        the actual bursts/failures from ``rng`` (used by random_day training)."""
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
                "demands": [f"D{int(rng.integers(1, 18))}"],
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
                "dst": str(rng.choice(["PE5", "PE6", "PE7", "PE8"])),
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
        self.scenario = self.scenario.materialize(np.random.default_rng(self.seed + 7919))
        self._noise = np.zeros(len(self.config.demands))

    def advance_noise(self) -> None:
        """One AR(1) step for every demand (call once per micro-tick)."""
        sigmas = np.array([
            self.scenario.noise_sigma * d.cls.burstiness for d in self.config.demands
        ])
        eps = self._rng.standard_normal(len(self.config.demands))
        self._noise = self.phi * self._noise + sigmas * eps

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

    def volumes(self, t_min: float) -> np.ndarray:
        """Offered Mbps per demand at simulated minute t (uses current noise state)."""
        hour = self.scenario.start_hour + t_min / 60.0
        out = np.empty(len(self.config.demands))
        for i, d in enumerate(self.config.demands):
            prof = profile_value(self.config.profiles[d.cls.profile], hour)
            noise_mult = float(np.clip(1.0 + self._noise[i], 0.4, 2.0))
            out[i] = d.base_mbps * prof * self.scenario.demand_multiplier \
                * self.event_factor(d, t_min) * noise_mult
        return out

    def link_events_at(self, t_min_from: float, t_min_to: float) -> list[dict[str, Any]]:
        """Scripted link_down/link_up events with t in [from, to)."""
        return [
            ev for ev in self.scenario.events
            if ev["type"] in ("link_down", "link_up") and t_min_from <= ev["t_min"] < t_min_to
        ]
