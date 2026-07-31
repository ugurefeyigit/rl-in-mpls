"""Observation, action and reward schemas, read from the real definitions.

Every offset, group length, transform and semantic label below is derived from
the authoritative source — `configs/experiments/rl_observation_v2.yaml` for V2,
`mplssim/rl/env.py`'s documented layout for V1, the topology's directed-link
order and the traffic config's demand order. Nothing is transcribed by hand into
a display constant that could drift.

The RL Information observation inspector needs a *semantic* label for each of
604 (or 586) positions. That mapping lives here so a single test can assert the
groups tile the vector exactly, with no gap and no overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mplssim.display import CITY_NAMES, CLASS_NAMES
from mplssim.factory import get_topology, get_traffic_config
from mplssim.product.contracts import (
    ACTION_COUNT, ENVIRONMENTS, K_PATHS, V1_REWARD_COMPONENTS,
    V2_REWARD_COMPONENTS,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
OBS_V2_PATH = CONFIG_DIR / "experiments" / "rl_observation_v2.yaml"

#: Plain-language meaning for each V2 observation feature name. The text explains
#: what the number *is*; it never claims why a policy responded to it.
V2_FEATURE_MEANING: dict[str, str] = {
    "link_input_utilization": "Offered load entering this direction divided by its capacity.",
    "link_up": "1 when this direction is operational, 0 when the link is failed.",
    "offered_over_base": "Current offered traffic relative to the demand's base rate.",
    "priority": "Traffic-class priority, 1 (lowest) to 6 (highest).",
    "protected": "1 when the class is protected and must keep positive headroom.",
    "measured_delay_over_sla": "Measured end-to-end delay against the class latency limit.",
    "measured_loss_over_sla": "Measured loss against the class loss limit.",
    "current_path_age_steps": "Control intervals the demand has held its current path.",
    "te_dwell_remaining": "Intervals before this demand may be moved again.",
    "disconnected": "1 when no live candidate path carries this demand.",
    "current_path": "One-hot marker of the candidate path the demand uses now.",
    "previous_te_path": "One-hot marker of the path the last TE action moved it from.",
    "candidate_live": "1 when every hop of this candidate path is operational.",
    "candidate_propagation_ms": "One-way propagation delay of this candidate path.",
    "candidate_projected_gross": "Projected worst-hop load if the demand moved here.",
}

#: Human-readable transforms, matching the YAML transform tokens exactly.
_TRANSFORM_TEXT: dict[str, str] = {
    "sat": "sat(x) = x / (1 + x); bounded, monotone, 0.5 at x = 1",
    "identity": "identity; already 0 or 1",
}


def _transform_text(token: str) -> str:
    if token in _TRANSFORM_TEXT:
        return _TRANSFORM_TEXT[token]
    if token.startswith("divide:"):
        return f"x / {token.split(':', 1)[1]}"
    if token.startswith("min1_divide:"):
        return f"min(x, 1) after dividing by {token.split(':', 1)[1]}"
    return token


def _base_feature(name: str) -> str:
    """`candidate_2_projected_gross` -> `candidate_projected_gross`."""
    for prefix, generic in (
        ("current_path_", "current_path"),
        ("previous_te_path_", "previous_te_path"),
    ):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return generic
    if name.startswith("candidate_"):
        rest = name.split("_", 2)
        if len(rest) == 3 and rest[1].isdigit():
            return f"candidate_{rest[2]}"
    return name


def axis_labels() -> dict[str, list[dict[str, Any]]]:
    """Stable display labels for the two observation axes.

    Directed links follow `Topology.dlinks` order (topology.yaml order, A->Z
    first). Demands follow traffic_classes.yaml order. Both orders are the
    scientific contract; this only attaches city names to them.
    """
    topo = get_topology()
    traffic = get_traffic_config()
    dlinks = [{
        "index": dl.index,
        "id": dl.id,
        "link_id": dl.undirected_id,
        "src": dl.src,
        "dst": dl.dst,
        "label": f"{CITY_NAMES.get(dl.src, dl.src)} → {CITY_NAMES.get(dl.dst, dl.dst)}",
        "capacity_mbps": dl.capacity_mbps,
    } for dl in topo.dlinks]
    demands = [{
        "index": d.index,
        "id": d.id,
        "src": d.src,
        "dst": d.dst,
        "class": d.cls.name,
        "class_label": CLASS_NAMES.get(d.cls.name, d.cls.name),
        "label": (f"{CITY_NAMES.get(d.src, d.src)} → {CITY_NAMES.get(d.dst, d.dst)} "
                  f"{CLASS_NAMES.get(d.cls.name, d.cls.name)}"),
        "base_mbps": d.base_mbps,
        "priority": d.cls.priority,
        "protected": d.cls.protected,
    } for d in traffic.demands]
    return {"dlink": dlinks, "demand": demands}


def observation_schema_v2() -> dict[str, Any]:
    raw = yaml.safe_load(OBS_V2_PATH.read_text(encoding="utf-8"))
    groups = []
    for block in raw["blocks"]:
        feature = block["feature"]
        base = _base_feature(feature)
        groups.append({
            "feature": feature,
            "group": base,
            "axis": block["axis"],
            "start": block["start"],
            "end": block["end"],
            "length": block["end"] - block["start"],
            "transform": block["transform"],
            "transform_text": _transform_text(str(block["transform"])),
            "meaning": V2_FEATURE_MEANING.get(base, ""),
        })
    return {
        "environment_version": "v2",
        "version": raw["version"],
        "dim": raw["dim"],
        "dtype": raw["dtype"],
        "low": raw["low"],
        "high": raw["high"],
        "feature_major": raw["feature_major"],
        "groups": groups,
        "source": "configs/experiments/rl_observation_v2.yaml",
    }


#: The V1 layout as `mplssim/rl/env.py` documents and builds it: 5 link features
#: x 64 directed links, then 15 demand features x 17 demands, then 11 globals.
_V1_LINK_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("link_utilization", "utilization / 2, clipped to [0, 1]",
     "Directed-link load against capacity; 1.0 means 200% or more."),
    ("link_queue_delay", "queue delay / Q_MAX_MS", "Modeled queueing delay on this direction."),
    ("link_loss", "loss fraction", "Modeled loss fraction on this direction."),
    ("link_up", "identity", "1 when this direction is operational."),
    ("link_util_ewma", "EWMA utilization / 2", "Recent utilization trend on this direction."),
)

_V1_DEMAND_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("offered_over_two_base", "offered / (2 * base)", "Offered traffic against twice the base rate."),
    ("priority", "priority / 6", "Traffic-class priority."),
    ("sla_max_latency", "max latency / 400 ms", "The class latency limit."),
    ("sla_max_loss", "max loss / 5%", "The class loss limit."),
    ("protected", "identity", "1 when the class is protected."),
    ("current_path_0", "identity", "One-hot marker of the path in use."),
    ("current_path_1", "identity", "One-hot marker of the path in use."),
    ("current_path_2", "identity", "One-hot marker of the path in use."),
    ("current_path_3", "identity", "One-hot marker of the path in use."),
    ("candidate_0_bottleneck", "bottleneck / 2, 1.0 if absent", "Worst-hop load of this candidate."),
    ("candidate_1_bottleneck", "bottleneck / 2, 1.0 if absent", "Worst-hop load of this candidate."),
    ("candidate_2_bottleneck", "bottleneck / 2, 1.0 if absent", "Worst-hop load of this candidate."),
    ("candidate_3_bottleneck", "bottleneck / 2, 1.0 if absent", "Worst-hop load of this candidate."),
    ("cooldown_remaining", "cooldown / cooldown_steps", "Intervals before this demand may move again."),
    ("disconnected", "identity", "1 when no live candidate carries this demand."),
)

_V1_GLOBAL_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("time_sin", "0.5 + 0.5 sin(2*pi*hour/24)", "Clock position, sine component."),
    ("time_cos", "0.5 + 0.5 cos(2*pi*hour/24)", "Clock position, cosine component."),
    ("max_util", "max utilization / 2", "Busiest directed link."),
    ("mean_util", "mean utilization", "Mean directed-link utilization."),
    ("util_std", "utilization std / 0.5", "Spread of directed-link utilization."),
    ("mean_delay", "mean delay / 60 ms", "Mean demand delay last interval."),
    ("loss_ratio", "loss ratio / 5%", "Network loss ratio last interval."),
    ("sla_violation_fraction", "identity", "Fraction of demands violating SLA."),
    ("delivered_ratio", "identity", "Delivered over offered traffic."),
    ("reroutes", "reroutes / 5", "Reroutes in the last interval."),
    ("episode_progress", "step / total steps", "Position within the scenario."),
)


def observation_schema_v1() -> dict[str, Any]:
    topo = get_topology()
    traffic = get_traffic_config()
    n_dlinks, n_demands = topo.n_dlinks, len(traffic.demands)
    groups, offset = [], 0
    for name, transform, meaning in _V1_LINK_FEATURES:
        groups.append({"feature": name, "group": name, "axis": "dlink",
                       "start": offset, "end": offset + n_dlinks, "length": n_dlinks,
                       "transform": transform, "transform_text": transform,
                       "meaning": meaning})
        offset += n_dlinks
    for name, transform, meaning in _V1_DEMAND_FEATURES:
        groups.append({"feature": name, "group": _base_feature(name), "axis": "demand",
                       "start": offset, "end": offset + n_demands, "length": n_demands,
                       "transform": transform, "transform_text": transform,
                       "meaning": meaning})
        offset += n_demands
    for name, transform, meaning in _V1_GLOBAL_FEATURES:
        groups.append({"feature": name, "group": "global", "axis": "global",
                       "start": offset, "end": offset + 1, "length": 1,
                       "transform": transform, "transform_text": transform,
                       "meaning": meaning})
        offset += 1
    return {
        "environment_version": "v1",
        "version": "obs-v1-586",
        "dim": offset,
        "dtype": "float32",
        "low": 0.0,
        "high": 1.0,
        "feature_major": True,
        "groups": groups,
        "source": "mplssim/rl/env.py",
    }


def action_schema() -> dict[str, Any]:
    """The 69-action space, decoded once for the whole product."""
    traffic = get_traffic_config()
    actions = [{
        "action": 0, "type": "noop", "demand_idx": None, "path_idx": None,
        "demand_id": None, "label": "No TE change",
    }]
    for d in traffic.demands:
        for p_idx in range(K_PATHS):
            actions.append({
                "action": 1 + K_PATHS * d.index + p_idx,
                "type": "reroute",
                "demand_idx": d.index,
                "path_idx": p_idx,
                "demand_id": d.id,
                "label": (f"{CITY_NAMES.get(d.src, d.src)} → "
                          f"{CITY_NAMES.get(d.dst, d.dst)} "
                          f"{CLASS_NAMES.get(d.cls.name, d.cls.name)} → path {p_idx}"),
            })
    return {
        "count": ACTION_COUNT,
        "formula": "0 = no TE change; 1 + 4*d + p moves demand d to candidate path p",
        "n_demands": len(traffic.demands),
        "k_paths": K_PATHS,
        "noop_action": 0,
        "actions": actions,
    }


def reward_schema(environment_version: str) -> dict[str, Any]:
    if environment_version == "v2":
        return {
            "environment_version": "v2",
            "components": list(V2_REWARD_COMPONENTS),
            "exact_sum": True,
            "note": "The 12 components sum exactly to the scalar interval reward "
                    "on every step. Order is the authoritative emission order.",
            "source": "mplssim/rl/reward_v2.py, configs/experiments/rl_reward_v2.yaml",
        }
    return {
        "environment_version": "v1",
        "components": list(V1_REWARD_COMPONENTS),
        "exact_sum": True,
        "note": "V1 reward terms. These are not the governed study's 12 V2 "
                "components and are never padded to look like them.",
        "source": "mplssim/rl/reward.py, configs/reward.yaml",
    }


def rl_schema(environment_version: str) -> dict[str, Any]:
    """`GET /api/rl/schema?environment=v1|v2`."""
    version = str(environment_version).lower()
    if version not in ENVIRONMENTS:
        raise ValueError(f"unknown environment version {environment_version!r}")
    observation = (observation_schema_v2() if version == "v2"
                   else observation_schema_v1())
    env = ENVIRONMENTS[version]
    return {
        "environment_version": version,
        "environment_label": env.label,
        "environment_class": env.env_class,
        "observation": observation,
        "action": action_schema(),
        "reward": reward_schema(version),
        "axes": axis_labels(),
    }
