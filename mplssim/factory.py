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


def make_engine(scenario: str, seed: int, cfg: EngineConfig | None = None,
                version: str = "v1"):
    """Build a simulation engine. ``version`` selects V1 (default) or V2.

    V1 behaviour and signature are unchanged; ``version`` is keyword-friendly
    and defaults to ``"v1"``, so every existing caller keeps V1 exactly.

    For ``version="v2"``, ``seed`` is the explicit *episode* seed —
    :class:`~mplssim.sim.engine_v2.SimulationEngineV2` derives its scenario and
    AR child streams from it. Use
    :func:`mplssim.rl.env_v2.episode_seed_for` to obtain a collision-free
    episode seed from ``(root_seed, worker_rank, episode_index)``.
    """
    if version == "v1":
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
    if version == "v2":
        # Imported lazily: mplssim.experiments.v2_factory imports this module.
        from mplssim.experiments.v2_factory import make_engine_v2
        return make_engine_v2(scenario, episode_seed=seed, cfg=cfg)
    raise ValueError(f"unknown engine version {version!r} (expected 'v1' or 'v2')")


def make_env(version: str = "v1", **kwargs):
    """Build an RL environment. ``version`` selects V1 (default) or V2.

    Selection is always explicit — a version is never inferred from a
    checkpoint's shapes. V1 keeps 586 observations, the V1 reward and the V1
    candidate paths; V2 is a different problem definition (604 observations, the
    operational reward, role-valid candidates) and needs its own checkpoints.
    """
    if version == "v1":
        from mplssim.rl.env import MplsTeEnv
        return MplsTeEnv(**kwargs)
    if version == "v2":
        from mplssim.experiments.v2_factory import make_env_v2
        return make_env_v2(**kwargs)
    raise ValueError(f"unknown environment version {version!r} (expected 'v1' or 'v2')")
