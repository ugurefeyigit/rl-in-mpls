"""Synchronization fingerprints for paired comparison.

A comparison between two controllers is only honest when both ran the *same*
experiment: same scenario, same seed, same starting state and the same exogenous
inputs at every step. "Both were constructed with seed 42" is a claim; a
fingerprint is a proof, and it keeps being a proof after a failure injection, a
burst or a multiplier change that could have reached only one engine.

Two fingerprints are computed:

`full`
    Everything, including routing. Two runners must agree on this *before the
    first decision*. Afterwards they legitimately diverge — that divergence is
    the thing being measured.

`exogenous`
    Time, offered traffic, link availability and manual interventions: the
    inputs neither controller chooses. These must agree at **every** step. If
    they stop agreeing, the comparison is no longer a comparison and the product
    disables it rather than showing a verdict.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _round(values: Any, places: int = 6) -> list[float]:
    return [round(float(v), places) for v in np.asarray(values).ravel()]


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def exogenous_state(engine: Any) -> dict[str, Any]:
    """Inputs no controller chooses."""
    return {
        "scenario": engine.scenario_name,
        "seed": int(engine.seed),
        "step": int(engine.step_count),
        "t_min": round(float(engine.t_min), 6),
        "demand_volumes": _round(engine.demand_volumes, 3),
        "link_up": {k: bool(v) for k, v in sorted(engine.link_up.items())},
        "manual_multiplier": round(float(getattr(engine, "manual_multiplier", 1.0)), 6),
    }


def full_state(engine: Any) -> dict[str, Any]:
    state = exogenous_state(engine)
    state.update({
        "current_path": [int(p) for p in np.asarray(engine.current_path).ravel()],
        "link_load": _round(engine.link_load, 3),
        "cooldown_until": [int(c) for c in np.asarray(engine.cooldown_until).ravel()],
    })
    return state


def exogenous_fingerprint(engine: Any) -> str:
    return _digest(exogenous_state(engine))


def full_fingerprint(engine: Any) -> str:
    return _digest(full_state(engine))


def mismatched_fields(a: Any, b: Any, *, full: bool) -> list[str]:
    """Which named parts of the two engines disagree. Empty means synchronized."""
    left = full_state(a) if full else exogenous_state(a)
    right = full_state(b) if full else exogenous_state(b)
    return sorted(k for k in left if left[k] != right.get(k))
