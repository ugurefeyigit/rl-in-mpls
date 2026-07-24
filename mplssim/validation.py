"""Startup validation for YAML configuration and model compatibility.

Fail fast with actionable messages instead of deep, cryptic errors later.
`validate_configs()` runs once at server startup and at the top of the
training/evaluation scripts; `check_model_compatibility()` runs whenever a
checkpoint is loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mplssim.core.topology import Topology
from mplssim.traffic.model import ScenarioSpec, TrafficConfig


class ConfigError(ValueError):
    """A configuration file is internally inconsistent."""


def validate_topology(topo: Topology) -> list[str]:
    problems: list[str] = []
    seen_links: set[frozenset[str]] = set()
    for ld in topo.link_defs.values():
        for end in (ld.a, ld.z):
            if end not in topo.routers:
                problems.append(f"link {ld.id}: endpoint '{end}' is not a defined router")
        if ld.capacity_mbps <= 0:
            problems.append(f"link {ld.id}: capacity must be positive, got {ld.capacity_mbps}")
        if ld.delay_ms < 0 or ld.weight <= 0:
            problems.append(f"link {ld.id}: delay/weight must be non-negative/positive")
        pair = frozenset((ld.a, ld.z))
        if pair in seen_links:
            problems.append(f"link {ld.id}: duplicate physical link between {ld.a} and {ld.z}")
        seen_links.add(pair)
    return problems


def validate_traffic(cfg: TrafficConfig, topo: Topology) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for d in cfg.demands:
        if d.id in seen:
            problems.append(f"demand {d.id}: duplicate demand id")
        seen.add(d.id)
        for end, label in ((d.src, "src"), (d.dst, "dst")):
            if end not in topo.routers:
                problems.append(f"demand {d.id}: {label} '{end}' is not a defined router")
        if d.base_mbps <= 0:
            problems.append(f"demand {d.id}: base_mbps must be positive")
        if d.cls.profile not in cfg.profiles:
            problems.append(f"demand {d.id}: class profile '{d.cls.profile}' not defined")
    for name, c in cfg.classes.items():
        if c.max_latency_ms <= 0 or c.max_loss_pct < 0:
            problems.append(f"class {name}: SLA thresholds malformed")
    return problems


def validate_scenarios(scenarios: dict[str, ScenarioSpec], cfg: TrafficConfig,
                       topo: Topology) -> list[str]:
    problems: list[str] = []
    demand_ids = {d.id for d in cfg.demands}
    egress = {d.dst for d in cfg.demands}
    for name, s in scenarios.items():
        if s.duration_min <= 0:
            problems.append(f"scenario {name}: non-positive duration")
        for ev in s.events:
            t = ev.get("t_min", -1)
            if not 0 <= t <= s.duration_min:
                problems.append(f"scenario {name}: event at t={t} outside 0..{s.duration_min}")
            if ev["type"] in ("link_down", "link_up") and ev["link"] not in topo.link_defs:
                problems.append(f"scenario {name}: unknown link '{ev['link']}'")
            if ev["type"] == "burst":
                for did in ev.get("demands", []):
                    if did not in demand_ids:
                        problems.append(f"scenario {name}: unknown demand '{did}'")
            if ev["type"] == "flash_crowd" and ev["dst"] not in egress:
                problems.append(f"scenario {name}: flash_crowd dst '{ev['dst']}' is not an egress PE")
        if s.randomize:
            for link in s.randomize.get("failure_candidates", []):
                if link not in topo.link_defs:
                    problems.append(f"scenario {name}: randomize failure candidate '{link}' unknown")
    return problems


def validate_reward(weights: dict[str, float], params: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for k, v in weights.items():
        if v < 0:
            problems.append(f"reward weight '{k}' must be >= 0 (sign is applied in code), got {v}")
    if not 0 < params.get("util_free_threshold", 0.6) < 1:
        problems.append("reward util_free_threshold must be in (0, 1)")
    if params.get("delay_norm_ms", 1) <= 0 or params.get("loss_norm", 1) <= 0:
        problems.append("reward delay_norm_ms and loss_norm must be positive")
    return problems


def validate_configs() -> None:
    """Validate every YAML config; raise ConfigError listing all problems."""
    import yaml
    from mplssim.core.topology import CONFIG_DIR
    from mplssim.factory import get_scenarios, get_topology, get_traffic_config

    topo = get_topology()
    traffic = get_traffic_config()
    scenarios = get_scenarios()
    raw_reward = yaml.safe_load((CONFIG_DIR / "reward.yaml").read_text(encoding="utf-8"))

    problems = (
        validate_topology(topo)
        + validate_traffic(traffic, topo)
        + validate_scenarios(scenarios, traffic, topo)
        + validate_reward(raw_reward["weights"], raw_reward["params"])
    )
    if problems:
        raise ConfigError("configuration invalid:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------- model metadata
def expected_shapes() -> dict[str, int]:
    """Observation/action dims implied by the CURRENT configuration."""
    from mplssim.factory import engine_config_from_training, get_topology, get_traffic_config
    from mplssim.rl.env import GLOBAL_FEATURES, LINK_FEATURES

    topo = get_topology()
    n_demands = len(get_traffic_config().demands)
    k = engine_config_from_training().k_paths
    obs = LINK_FEATURES * topo.n_dlinks + (7 + 2 * k) * n_demands + GLOBAL_FEATURES
    return {
        "n_dlinks": topo.n_dlinks,
        "n_demands": n_demands,
        "k_paths": k,
        "observation_dim": obs,
        "action_dim": 1 + n_demands * k,
    }


def write_model_metadata(tag: str, extra: dict[str, Any] | None = None) -> Path:
    """Record the shapes/config hashes a model directory was built against."""
    from mplssim.core.topology import CONFIG_DIR

    root = Path(__file__).resolve().parents[1]
    meta = {
        "tag": tag,
        **expected_shapes(),
        "config_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CONFIG_DIR.glob("*.yaml"))
        },
    }
    if extra:
        meta.update(extra)
    path = root / "models" / tag / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path


def check_model_compatibility(model: Any, tag: str) -> None:
    """Raise ConfigError with a clear message if a loaded SB3 model's spaces
    do not match what the current configuration produces."""
    exp = expected_shapes()
    got_obs = int(model.observation_space.shape[0])
    got_act = int(model.action_space.n)
    if got_obs != exp["observation_dim"] or got_act != exp["action_dim"]:
        raise ConfigError(
            f"model '{tag}' is incompatible with the current configuration: "
            f"model expects obs={got_obs}, actions={got_act}; current config "
            f"produces obs={exp['observation_dim']}, actions={exp['action_dim']} "
            f"(n_demands={exp['n_demands']}, k_paths={exp['k_paths']}, "
            f"n_dlinks={exp['n_dlinks']}). Either restore the configuration the "
            f"model was trained with (see models/{tag}/metadata.json) or train "
            f"a new model."
        )
