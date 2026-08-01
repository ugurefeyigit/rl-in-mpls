"""Proof that a paired session is a fair comparison — or a refusal to claim one.

The comparison lane in Presentation mode shows two controllers side by side. It
may only do so while it can *prove* both runners share one experiment. This
module produces that proof, names the fields that broke it when it fails, and
never repairs a mismatch by averaging, resampling or hiding a lane.

A live session with a single runner is not a comparison and reports itself as
such, rather than as a failed one.
"""

from __future__ import annotations

from typing import Any

from mplssim.product import fingerprint


def _runner_view(runner: Any) -> dict[str, Any]:
    engine = runner.eng
    return {
        "algorithm": runner.algorithm,
        "environment_version": getattr(runner, "environment_version", "v1"),
        "checkpoint_id": getattr(runner, "checkpoint_id", None),
        "output_semantics": getattr(runner, "output_semantics", "none"),
        "step": int(engine.step_count),
        "exogenous_fingerprint": fingerprint.exogenous_fingerprint(engine),
        "full_fingerprint": fingerprint.full_fingerprint(engine),
        "cumulative_reward": round(float(runner.cumulative_reward), 4),
    }


def synchronization(session: Any) -> dict[str, Any]:
    """Is this session's comparison lane allowed to render?

    Before the first decision both runners must match completely. From then on
    they must still agree on time, offered traffic, link availability and manual
    interventions; the routing difference is the measurement.
    """
    runners = list(session.runners)
    versions = {getattr(r, "environment_version", "v1") for r in runners}
    # Two environment versions are two different problems: the same action
    # number addresses a different candidate path. There is no fair comparison
    # to render, and none is claimed. Checked before any engine is read, because
    # a mismatched pair may not even share an engine shape.
    identity = {
        "scenario": session.config.scenario,
        "seed": session.config.seed,
        "environment_version": session.config.environment,
        "training_root": (int(session.config.training_root)
                          if session.config.environment == "v2" else None),
    }
    if len(versions) > 1:
        return {**identity,
                "lanes": [{"algorithm": r.algorithm,
                           "environment_version": getattr(
                               r, "environment_version", "v1")}
                          for r in runners],
                "comparison": True, "matched": False,
                "reason": ("The two lanes run different environment versions, so "
                           "the same action number does not mean the same move. "
                           "No comparative verdict is shown."),
                "mismatched_fields": ["environment_version"],
                "proof": "environment identity"}

    common = {**identity, "lanes": [_runner_view(r) for r in runners]}
    if len(runners) < 2:
        return {**common, "comparison": False, "matched": None,
                "reason": "This session runs one controller. There is nothing to "
                          "compare against.",
                "mismatched_fields": []}

    a, b = runners[0].eng, runners[1].eng
    before_first_decision = int(a.step_count) == 0 and int(b.step_count) == 0
    mismatched = fingerprint.mismatched_fields(a, b, full=before_first_decision)
    if mismatched:
        return {
            **common, "comparison": True, "matched": False,
            "reason": ("The two runners are not running the same experiment, so no "
                       "comparative verdict is shown."),
            "mismatched_fields": mismatched,
            "proof": "full state" if before_first_decision else "exogenous inputs",
        }
    return {
        **common, "comparison": True, "matched": True,
        "reason": ("Both runners share scenario, seed, starting state and every "
                   "exogenous input recorded so far."),
        "mismatched_fields": [],
        "proof": "full state" if before_first_decision else "exogenous inputs",
    }
