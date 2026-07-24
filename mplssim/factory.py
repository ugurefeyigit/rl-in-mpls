"""Convenience factory: build engines and controllers from the YAML configs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from mplssim.core.topology import CONFIG_DIR, Topology, load_topology
from mplssim.sim.engine import EngineConfig, SimulationEngine
from mplssim.traffic.model import ScenarioSpec, TrafficConfig, load_scenarios, load_traffic_config


@lru_cache(maxsize=1)
def get_topology() -> Topology:
    return load_topology()


@lru_cache(maxsize=1)
def get_traffic_config() -> TrafficConfig:
    return load_traffic_config()


@lru_cache(maxsize=1)
def get_scenarios() -> dict[str, ScenarioSpec]:
    return load_scenarios()


@lru_cache(maxsize=1)
def get_training_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "training.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_baseline_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "baselines.yaml").read_text(encoding="utf-8"))


def engine_config_from_training(overrides: dict | None = None) -> EngineConfig:
    env = dict(get_training_config()["env"])
    if overrides:
        env.update(overrides)
    return EngineConfig(
        control_interval_min=int(env["control_interval_min"]),
        micro_ticks_per_interval=int(env["micro_ticks_per_interval"]),
        k_paths=int(env["k_paths"]),
        max_hop_factor=float(env["max_hop_factor"]),
        reroute_cooldown_steps=int(env["reroute_cooldown_steps"]),
    )


def make_engine(scenario: str, seed: int, cfg: EngineConfig | None = None) -> SimulationEngine:
    scenarios = get_scenarios()
    if scenario not in scenarios:
        raise KeyError(f"unknown scenario '{scenario}' (have: {sorted(scenarios)})")
    return SimulationEngine(
        topo=get_topology(),
        traffic_cfg=get_traffic_config(),
        scenario=scenarios[scenario],
        seed=seed,
        cfg=cfg or engine_config_from_training(),
    )
