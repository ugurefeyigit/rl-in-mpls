"""Clone-only one-interval estimates.

Decision Lens may show "what this move would do" only when the engine can be
copied and stepped without the running session noticing. That is exactly the
guarantee this module enforces: it fingerprints the live engine, evaluates on
deep clones, fingerprints again, and refuses to return a result if the two
fingerprints differ.

The result is labelled a **simulated estimate**. It is not an observed outcome,
it is not a forecast of what will happen, and it is never final evidence. After
the real step runs, the observed outcome is shown *beside* the estimate — the
difference stays visible.
"""

from __future__ import annotations

from typing import Any

from mplssim.product import fingerprint
from mplssim.product.contracts import ACTION_COUNT, decode_action

#: Metrics an estimate reports. Anything not in this list is not estimated.
_KEYS = ("max_util", "mean_delay_ms", "loss_ratio", "sla_violations",
         "delivered_ratio")

LABEL = ("Simulated one-interval estimate from cloned state. It is not an "
         "observed outcome and not final evidence.")


def _metrics(interval: dict[str, Any]) -> dict[str, float]:
    return {k: round(float(interval[k]), 4) for k in _KEYS if k in interval}


def estimate(session: Any, runner: Any, action: int,
             generation: int | None = None,
             step: int | None = None) -> dict[str, Any]:
    """Compare `action` against no-op on two deep clones of the current state."""
    action = int(action)
    if not 0 <= action < ACTION_COUNT:
        return {"kind": "unavailable", "http_status": 400,
                "reason": f"action {action} is outside 0..{ACTION_COUNT - 1}."}

    if generation is not None and int(generation) != int(session.generation):
        return {"kind": "unavailable", "http_status": 409,
                "reason": ("The session was reset since this request was prepared. "
                           "Re-read the snapshot before asking again."),
                "expected_generation": int(session.generation)}

    engine = runner.eng
    if step is not None and int(step) != int(engine.step_count):
        return {"kind": "unavailable", "http_status": 409,
                "reason": ("The session advanced since this request was prepared."),
                "expected_step": int(engine.step_count)}

    if not hasattr(engine, "clone"):
        return {"kind": "unavailable", "http_status": 409,
                "reason": "This engine cannot be cloned, so no estimate is computable."}

    k_paths = getattr(getattr(runner, "env", None), "k", 4)
    before = fingerprint.full_fingerprint(engine)

    noop_engine = engine.clone()
    noop_metrics = _metrics(noop_engine.step_interval())

    result: dict[str, Any] = {
        "kind": "simulated_estimate",
        "label": LABEL,
        "source_kind": "live_session",
        "session_id": session.id,
        "generation": int(session.generation),
        "step": int(engine.step_count),
        "policy_id": runner.algorithm,
        "action": action,
        "metrics_reported": list(_KEYS),
        "noop": noop_metrics,
    }

    if action == 0:
        result["action_metrics"] = None
        result["action_reason"] = "The requested action is no-op; there is nothing to compare."
    else:
        d_idx, p_idx = decode_action(action)
        act_engine = engine.clone()
        accepted, reason = act_engine.apply_action(d_idx, p_idx, source="rl")
        action_metrics = _metrics(act_engine.step_interval())
        result["action_metrics"] = action_metrics
        result["action_applied"] = bool(accepted)
        result["action_reason"] = reason
        result["demand_idx"] = d_idx
        result["path_idx"] = p_idx
        result["delta"] = {
            key: round(action_metrics[key] - noop_metrics[key], 4)
            for key in action_metrics if key in noop_metrics
        }

    after = fingerprint.full_fingerprint(engine)
    result["session_fingerprint_before"] = before
    result["session_fingerprint_after"] = after
    result["session_unchanged"] = before == after
    if before != after:
        return {"kind": "unavailable", "http_status": 500,
                "reason": ("The live session changed while the estimate was computed. "
                           "No estimate is reported."),
                "session_fingerprint_before": before,
                "session_fingerprint_after": after}
    return result
