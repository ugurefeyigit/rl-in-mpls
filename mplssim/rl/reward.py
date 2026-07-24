"""Multi-objective reward. Formula and weights documented in configs/reward.yaml.

Every component is returned individually so the trainer logs them to
TensorBoard and the frontend displays a live breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from mplssim.core.topology import CONFIG_DIR


@dataclass(frozen=True)
class RewardConfig:
    weights: dict[str, float]
    util_free_threshold: float
    delay_norm_ms: float
    loss_norm: float
    flap_window_steps: int


@lru_cache(maxsize=1)
def load_reward_config() -> RewardConfig:
    raw = yaml.safe_load((CONFIG_DIR / "reward.yaml").read_text(encoding="utf-8"))
    p = raw["params"]
    return RewardConfig(
        weights={k: float(v) for k, v in raw["weights"].items()},
        util_free_threshold=float(p["util_free_threshold"]),
        delay_norm_ms=float(p["delay_norm_ms"]),
        loss_norm=float(p["loss_norm"]),
        flap_window_steps=int(p["flap_window_steps"]),
    )


def with_overrides(cfg: RewardConfig, overrides: dict[str, float]) -> RewardConfig:
    """Copy of ``cfg`` with some weights replaced (used by reward ablations)."""
    unknown = set(overrides) - set(cfg.weights)
    if unknown:
        raise KeyError(f"unknown reward weights: {sorted(unknown)}")
    w = dict(cfg.weights)
    w.update(overrides)
    return RewardConfig(weights=w, util_free_threshold=cfg.util_free_threshold,
                        delay_norm_ms=cfg.delay_norm_ms, loss_norm=cfg.loss_norm,
                        flap_window_steps=cfg.flap_window_steps)


def clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def compute_reward(
    interval: dict[str, Any],
    rerouted: bool,
    flapped: bool,
    invalid: bool,
    cfg: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Reward for one control interval. Returns (total, per-component dict).

    Components are signed contributions (positive terms positive, penalties
    negative) so they sum exactly to the total.
    """
    cfg = cfg or load_reward_config()
    w = cfg.weights
    free = cfg.util_free_threshold
    comp = {
        "delivered": w["delivered"] * clip01(interval["delivered_ratio"]),
        "priority_sla": w["priority_sla"] * clip01(interval["priority_sla_success"]),
        "max_util": -w["max_util"] * clip01((interval["max_util"] - free) / (1.0 - free)),
        "util_spread": -w["util_spread"] * clip01(interval["util_std"] / 0.5),
        "delay": -w["delay"] * clip01(interval["mean_delay_ms"] / cfg.delay_norm_ms),
        "loss": -w["loss"] * clip01(interval["loss_ratio"] / cfg.loss_norm),
        "sla": -w["sla"] * clip01(interval["sla_violation_fraction"]),
        "overload": -w["overload"] * clip01(interval["overload_ratio"] * 20.0),
        "reroute": -w["reroute"] * (1.0 if rerouted else 0.0),
        "flap": -w["flap"] * (1.0 if flapped else 0.0),
        "invalid": -w["invalid"] * (1.0 if invalid else 0.0),
        "disconnected": -w["disconnected"] * clip01(
            interval["disconnected_demands"] / max(1, 17)
        ),
    }
    return sum(comp.values()), comp
