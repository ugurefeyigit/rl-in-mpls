"""Operational reward for MPLS-TE Environment V2 (reward-v2.0-operational).

Governing document: docs/RL_ENVIRONMENT_V2_SPEC.md, "V2 reward".
Formula and coefficients are mirrored in configs/experiments/rl_reward_v2.yaml.

Everything here is a pure function of already-aggregated metrics, so the reward
can be unit-tested on synthetic states with no engine at all — which is how the
calibration and ordering gates in the test plan are checked.

Why V2 replaces the V1 reward
-----------------------------

V1 clips almost every term. Over 13,200 audited intervals its max-utilization
term was saturated 60.2% of the time, loss 41.1% and delay 26.2%: a link at 400%
scored exactly the same as one at 100%, so severity carried no gradient. It also
double-counted congestion (max-utilization *and* overload; delivered ratio,
priority SLA *and* a binary SLA fraction), and it priced every reroute at a flat
0.08 regardless of how much traffic moved or how far. A 0.0267 reduction in
unsaturated max utilization paid for a reroute, which is why the published PPO
made an accepted TE move in every single one of 3,300 evaluation intervals, 91.7%
of them flagged reversals on ``full_day``.

V2 is four layers:

1. **Absolute operational utility** ``U``: unsaturated, severity-ordered, with
   protected connectivity dominating everything else (coefficient 30 against 8
   for unprotected and 2 for congestion).
2. **Bounded potential shaping** ``F``: a genuine potential-based term, so it
   cannot change the optimal policy, and bounded to roughly +-0.399 so a
   temporary improvement can never make a persistently bad network look healthy.
3. **Operational move cost** ``C_TE``: fixed + moved-volume share + edge
   divergence + reversal, so churn is priced by what it actually disturbs.
4. **Rejected-request cost** ``C_invalid``.

``g(u) = log(1 + max(0,(u-0.70)/0.30))`` is zero at or below 70% utilization,
continuous, monotone and unsaturated — 0.288 at 80%, 0.693 at 100%, 1.674 at
200%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from mplssim.core.topology import CONFIG_DIR

REWARD_VERSION = "reward-v2.0-operational"
REWARD_CONFIG_PATH = CONFIG_DIR / "experiments" / "rl_reward_v2.yaml"

#: Signed component order. Their left-to-right sum in exactly this order is the
#: scalar reward, bit-for-bit (spec, "Metrics and info contract").
COMPONENT_ORDER: tuple[str, ...] = (
    "delivery",
    "protected_disconnect",
    "unprotected_disconnect",
    "sla_severity",
    "max_util",
    "overload",
    "potential",
    "move_fixed",
    "move_volume",
    "move_divergence",
    "reversal",
    "invalid",
)

#: Metric keys the utility function consumes.
UTILITY_KEYS: tuple[str, ...] = (
    "delivered_ratio",
    "protected_disconnect",
    "unprotected_disconnect",
    "sla_severity",
    "max_util",
    "overload_ratio",
)


class RewardConfigError(ValueError):
    """V2 reward configuration is missing, malformed or wrongly versioned."""


@dataclass(frozen=True)
class RewardConfigV2:
    version: str = REWARD_VERSION
    # utility coefficients (all positive; sign is applied by the component)
    delivered: float = 2.0
    protected_disconnect: float = 30.0
    unprotected_disconnect: float = 8.0
    sla_severity: float = 6.0
    max_util: float = 2.0
    overload: float = 6.0
    # g(u) shape
    util_free_threshold: float = 0.70
    util_span: float = 0.30
    # potential shaping
    potential_coefficient: float = 0.20
    potential_gamma: float = 0.995
    potential_scale: float = 10.0
    # accepted-move cost
    move_fixed: float = 0.08
    move_volume_share: float = 0.30
    move_edge_divergence: float = 0.12
    move_reversal: float = 0.30
    # rejected-request cost
    invalid: float = 0.05


@lru_cache(maxsize=1)
def load_reward_config_v2(path: Path | None = None) -> RewardConfigV2:
    """Load and version-check ``configs/experiments/rl_reward_v2.yaml``."""
    path = path or REWARD_CONFIG_PATH
    if not path.exists():
        raise RewardConfigError(f"missing V2 reward config {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = str(raw.get("version", ""))
    if version != REWARD_VERSION:
        raise RewardConfigError(
            f"{path}: reward version {version!r} != required {REWARD_VERSION!r}")
    order = tuple(raw.get("component_order", ()))
    if order != COMPONENT_ORDER:
        raise RewardConfigError(
            f"{path}: component_order {order} != required {COMPONENT_ORDER}")
    u, c, p, mv = raw["utility"], raw["max_util_curve"], raw["potential"], raw["move"]
    return RewardConfigV2(
        version=version,
        delivered=float(u["delivered"]),
        protected_disconnect=float(u["protected_disconnect"]),
        unprotected_disconnect=float(u["unprotected_disconnect"]),
        sla_severity=float(u["sla_severity"]),
        max_util=float(u["max_util"]),
        overload=float(u["overload"]),
        util_free_threshold=float(c["free_threshold"]),
        util_span=float(c["span"]),
        potential_coefficient=float(p["coefficient"]),
        potential_gamma=float(p["gamma"]),
        potential_scale=float(p["scale"]),
        move_fixed=float(mv["fixed"]),
        move_volume_share=float(mv["volume_share"]),
        move_edge_divergence=float(mv["edge_divergence"]),
        move_reversal=float(mv["reversal"]),
        invalid=float(raw["invalid"]),
    )


def sat(x: float) -> float:
    """``x/(1+x)``: monotone, bounded, non-clipping, exactly 0.5 at ``x == 1``."""
    return x / (1.0 + x)


#: The SLA-severity squashing function is the same map, named as the spec does.
h = sat


def g(u: float, free_threshold: float = 0.70, span: float = 0.30) -> float:
    """Max-utilization penalty shape ``log(1 + max(0,(u-free)/span))``."""
    return math.log1p(max(0.0, (u - free_threshold) / span))


def normalized_delay_excess(delay_ms: float, delay_sla_ms: float) -> float:
    """``max(0, delay - SLA)/SLA``: zero at or below the SLA, monotone above."""
    return max(0.0, delay_ms - delay_sla_ms) / delay_sla_ms


def normalized_loss_excess(loss_fraction: float, loss_sla_fraction: float) -> float:
    """``max(0, loss - SLA)/max(SLA, 1e-6)``."""
    return max(0.0, loss_fraction - loss_sla_fraction) / max(loss_sla_fraction, 1e-6)


def utility(metrics: Mapping[str, float], cfg: RewardConfigV2 | None = None) -> float:
    """Absolute operational utility ``U`` of an interval or a single boundary.

    The same functional form is used for both: the interval version consumes
    aggregated metrics, the potential term consumes a one-tick boundary
    snapshot. That is what makes ``Phi`` a genuine state potential.
    """
    cfg = cfg or load_reward_config_v2()
    missing = [k for k in UTILITY_KEYS if k not in metrics]
    if missing:
        raise KeyError(f"utility metrics missing {missing}")
    return (
        cfg.delivered * float(metrics["delivered_ratio"])
        - cfg.protected_disconnect * float(metrics["protected_disconnect"])
        - cfg.unprotected_disconnect * float(metrics["unprotected_disconnect"])
        - cfg.sla_severity * float(metrics["sla_severity"])
        - cfg.max_util * g(float(metrics["max_util"]),
                           cfg.util_free_threshold, cfg.util_span)
        - cfg.overload * float(metrics["overload_ratio"])
    )


def potential(u_state: float, cfg: RewardConfigV2 | None = None) -> float:
    """``Phi(s) = tanh(U_state(s)/scale)``."""
    cfg = cfg or load_reward_config_v2()
    return math.tanh(u_state / cfg.potential_scale)


def shaping(phi_current: float, phi_next: float,
            cfg: RewardConfigV2 | None = None) -> float:
    """``F = coefficient * (gamma*Phi(s_next) - Phi(s_current))``."""
    cfg = cfg or load_reward_config_v2()
    return cfg.potential_coefficient * (cfg.potential_gamma * phi_next - phi_current)


def move_cost(volume_share: float, edge_divergence: float, reversal: bool,
              cfg: RewardConfigV2 | None = None) -> float:
    """Total ``C_TE`` for one accepted TE change (positive magnitude)."""
    cfg = cfg or load_reward_config_v2()
    return (cfg.move_fixed
            + cfg.move_volume_share * volume_share
            + cfg.move_edge_divergence * edge_divergence
            + cfg.move_reversal * (1.0 if reversal else 0.0))


def compute_reward_v2(
    interval: Mapping[str, float],
    phi_current: float,
    phi_next: float,
    accepted: bool = False,
    volume_share: float = 0.0,
    edge_divergence: float = 0.0,
    reversal: bool = False,
    rejected: bool = False,
    cfg: RewardConfigV2 | None = None,
) -> tuple[float, dict[str, float]]:
    """Scalar reward and its 12 signed components for one control interval.

    ``reward = U_interval + F - C_TE - C_invalid``, returned as the
    left-to-right sum of the components in :data:`COMPONENT_ORDER` so the
    identity ``sum(components) == reward`` holds bit-for-bit rather than merely
    to within rounding.

    An accepted TE change and a rejected request are mutually exclusive: a
    request is either applied or refused. FRR and recovery restoration never
    reach this function's cost terms at all — they are environment transitions,
    not policy actions, and always cost zero.
    """
    cfg = cfg or load_reward_config_v2()
    comp: dict[str, float] = {
        "delivery": cfg.delivered * float(interval["delivered_ratio"]),
        "protected_disconnect": -cfg.protected_disconnect
        * float(interval["protected_disconnect"]),
        "unprotected_disconnect": -cfg.unprotected_disconnect
        * float(interval["unprotected_disconnect"]),
        "sla_severity": -cfg.sla_severity * float(interval["sla_severity"]),
        "max_util": -cfg.max_util * g(float(interval["max_util"]),
                                      cfg.util_free_threshold, cfg.util_span),
        "overload": -cfg.overload * float(interval["overload_ratio"]),
        "potential": shaping(phi_current, phi_next, cfg),
        "move_fixed": -cfg.move_fixed if accepted else 0.0,
        "move_volume": -cfg.move_volume_share * volume_share if accepted else 0.0,
        "move_divergence": -cfg.move_edge_divergence * edge_divergence if accepted else 0.0,
        "reversal": -cfg.move_reversal if (accepted and reversal) else 0.0,
        "invalid": -cfg.invalid if rejected else 0.0,
    }
    total = 0.0
    for name in COMPONENT_ORDER:
        total += comp[name]
    return total, comp


def components_sum(components: Mapping[str, float]) -> float:
    """The defined-order sum. Equals the scalar reward bit-for-bit."""
    total = 0.0
    for name in COMPONENT_ORDER:
        total += float(components[name])
    return total


def interval_metrics_from_ticks(
    ticks: list[Mapping[str, Any]], micro_ticks: int, q_sum: float,
) -> dict[str, float]:
    """Reference (non-vectorized) interval aggregation used by tests.

    Mirrors :meth:`mplssim.sim.engine_v2.SimulationEngineV2.aggregate_interval`
    for the six reward-bearing quantities; the two are asserted to agree.
    """
    offered = sum(float(t["offered_mbps"]) for t in ticks)
    delivered = sum(float(t["delivered_mbps"]) for t in ticks)
    return {
        "protected_disconnect": max(float(t["protected_disconnect"]) for t in ticks),
        "unprotected_disconnect": max(float(t["unprotected_disconnect"]) for t in ticks),
        "sla_severity": sum(float(t["sla_severity_sum"]) for t in ticks)
        / (micro_ticks * q_sum),
        "delivered_ratio": (delivered / offered) if offered > 0 else 1.0,
        "max_util": max(float(t["max_util"]) for t in ticks),
        "overload_ratio": sum(float(t["overload_ratio"]) for t in ticks) / len(ticks),
    }
