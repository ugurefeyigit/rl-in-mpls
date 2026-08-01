"""The decision-observatory payload: observation → mask → output → action → reward.

Every stage is either present with real values or explicitly unavailable with
the reason it is unavailable. The three ways this module could lie are the three
things it refuses to do:

1. It never labels a bandit score a probability, and never labels a PPO
   probability a score. The label comes from the controller's declared output
   semantics, not from the shape of the numbers.
2. It never infers a mask *reason*. A boolean mask says a move is illegal; only
   the engine's own validator says why, so the reason is read from
   `validate_action`.
3. It never ranks observation changes as causal importance. A changed feature is
   a changed feature, and the payload says so in the field name and in the note.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mplssim.display import CITY_NAMES, CLASS_NAMES
from mplssim.product import serialize
from mplssim.product.catalog import live_policies
from mplssim.product.contracts import (
    ACTION_COUNT, K_PATHS, OutputSemantics, V1_REWARD_COMPONENTS, decode_action,
)


def _semantics_for(runner: Any) -> OutputSemantics:
    """The controller's *declared* output semantics, never inferred from shape."""
    declared = getattr(runner, "output_semantics", None)
    if declared:
        for semantics in OutputSemantics:
            if semantics.value == declared:
                return semantics
    version = getattr(runner, "environment_version", "v1")
    for policy in live_policies():
        if policy.id == runner.algorithm and policy.environment_version == version:
            return policy.output_semantics
    return OutputSemantics.NONE


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **extra}


# ------------------------------------------------------------- observation
def observation_state(runner: Any) -> dict[str, Any]:
    obs = getattr(runner, "_obs", None)
    if obs is None:
        return _unavailable(
            f"{runner.algorithm} is a rule-based controller. It consumes engine "
            f"state directly and never builds an observation vector.")
    prior = getattr(runner, "_prior_obs", None)
    current = np.asarray(obs, dtype=float)
    payload: dict[str, Any] = {
        "available": True,
        "environment_version": getattr(runner, "environment_version", "v1"),
        "dim": int(current.size),
        "values": [round(float(v), 6) for v in current],
    }
    if prior is not None and np.asarray(prior).size == current.size:
        prior = np.asarray(prior, dtype=float)
        delta = current - prior
        changed = np.flatnonzero(np.abs(delta) > 1e-9)
        ranking = sorted(
            ({"index": int(i), "prior": round(float(prior[i]), 6),
              "current": round(float(current[i]), 6),
              "delta": round(float(delta[i]), 6)} for i in changed),
            key=lambda r: -abs(r["delta"]))[:40]
        payload["prior_values"] = [round(float(v), 6) for v in prior]
        payload["changed_feature_ranking"] = ranking
        payload["changed_count"] = int(changed.size)
    else:
        payload["prior_values"] = None
        payload["changed_feature_ranking"] = []
        payload["changed_count"] = None
        payload["prior_reason"] = "No prior observation exists yet in this generation."
    payload["ranking_note"] = (
        "Changed features are sorted by absolute change between the prior and "
        "current observation. This is descriptive change, not causal importance, "
        "and it is not the policy's internal reasoning.")
    return payload


# --------------------------------------------------------------- action mask
def action_grid(runner: Any) -> dict[str, Any]:
    env = getattr(runner, "env", None)
    engine = runner.eng
    selected = None
    if runner.last_decision and "action" in runner.last_decision:
        selected = int(runner.last_decision["action"])

    if env is None:
        return {
            "available": False,
            "reason": (f"{runner.algorithm} is a rule-based controller. It does not "
                       f"evaluate the 69-action space, so no mask exists for it."),
            "count": ACTION_COUNT,
            "actions": [],
            "selected_action": None,
            "valid_count": None,
        }

    mask = np.asarray(env.action_masks(), dtype=bool)
    rows = [{
        "action": 0,
        "type": "noop",
        "label": "No TE change",
        "demand_id": None,
        "demand_idx": None,
        "path_idx": None,
        "path_label": None,
        "valid": bool(mask[0]),
        "reason": "no-op is always legal",
        "is_current_path": False,
        "selected": selected == 0,
    }]
    for action in range(1, ACTION_COUNT):
        d_idx, p_idx = decode_action(action)
        demand = engine.demands[d_idx]
        valid = bool(mask[action])
        # The authoritative validator owns the reason. A boolean mask alone
        # cannot say *why* a move is illegal, and the UI must not guess.
        ok, reason = engine.validate_action(d_idx, p_idx, source="rl")
        routers = demand.candidate_paths[p_idx] if p_idx < len(demand.candidate_paths) else ()
        rows.append({
            "action": action,
            "type": "reroute",
            "label": (f"{CITY_NAMES.get(demand.src, demand.src)} → "
                      f"{CITY_NAMES.get(demand.dst, demand.dst)} "
                      f"{CLASS_NAMES.get(demand.cls.name, demand.cls.name)}"),
            "demand_id": demand.id,
            "demand_idx": d_idx,
            "path_idx": p_idx,
            "path_label": " → ".join(CITY_NAMES.get(r, r) for r in routers),
            "valid": valid,
            "reason": reason if not valid else "ok",
            "is_current_path": int(engine.current_path[d_idx]) == p_idx,
            "selected": selected == action,
        })
    return {
        "available": True,
        "count": ACTION_COUNT,
        "formula": "0 = no TE change; 1 + 4*d + p moves demand d to candidate path p",
        "n_demands": len(engine.demands),
        "k_paths": K_PATHS,
        "valid_count": int(mask.sum()),
        "selected_action": selected,
        "actions": rows,
        "reason_source": (
            "mplssim.sim.engine_v2.SimulationEngineV2.validate_te_action"
            if getattr(runner, "environment_version", "v1") == "v2"
            else "mplssim.sim.engine.SimulationEngine.validate_action"),
    }


# ------------------------------------------------------------- policy output
def policy_output(runner: Any) -> dict[str, Any]:
    semantics = _semantics_for(runner)
    base = {
        "policy_id": runner.algorithm,
        "semantics": semantics.value,
        "label": semantics.label,
        "description": semantics.description,
        "is_percentage": semantics.percent,
    }
    if semantics is OutputSemantics.NONE:
        return {**base, "available": False,
                "reason": f"{runner.algorithm} exposes no per-action numbers.",
                "top": [], "selected": None, "runner_up": None, "noop": None,
                "entropy": None, "value": None}

    decision = runner.last_decision or {}
    top = decision.get("top_actions")
    if not top:
        return {**base, "available": False,
                "reason": ("No policy distribution has been captured yet. It is "
                           "extracted at decision time and appears after the first step."),
                "top": [], "selected": None, "runner_up": None, "noop": None,
                "entropy": None, "value": None}

    ordered = sorted(top, key=lambda r: -r["value"])
    selected_action = decision.get("action")
    by_action = {r["action"]: r["value"] for r in ordered}
    runner_up = next((r for r in ordered if r["action"] != selected_action), None)
    return {
        **base,
        "available": True,
        "top": ordered,
        "selected": {"action": selected_action,
                     "value": decision.get("output_value")},
        "runner_up": ({"action": runner_up["action"], "value": runner_up["value"]}
                      if runner_up else None),
        "noop": {"action": 0, "value": by_action.get(0)} if 0 in by_action else
                {"action": 0, "value": None,
                 "reason": "No-op fell outside the reported distribution head."},
        "entropy": None,
        "entropy_reason": "The live runner does not expose distribution entropy.",
        "value": None,
        "value_reason": "The live runner does not expose the value estimate.",
        "distribution_note": (
            "Immediate-reward estimates for each valid action. They are not "
            "probabilities, not confidence and do not sum to one."
            if semantics is OutputSemantics.SCORES else
            "Probabilities cover the valid masked distribution only. "
            "Invalid actions carry no probability bar."),
    }


# -------------------------------------------------------- selected & safety
def selected_action(runner: Any) -> dict[str, Any]:
    decision = runner.last_decision
    if not decision:
        return _unavailable("No decision has been taken yet in this generation.")
    action = decision.get("action")
    if action is None:
        moves = decision.get("moves", [])
        return {
            "available": True,
            "kind": "baseline_moves",
            "policy_id": runner.algorithm,
            "moves": moves,
            "n_moves": len(moves),
            "note": (f"{runner.algorithm} proposes zero or more moves per interval "
                     f"rather than one action from the 69-action space."),
        }
    decoded = decision.get("decoded") or {}
    d_idx, p_idx = decode_action(int(action))
    return {
        "available": True,
        "kind": "single_action",
        "policy_id": runner.algorithm,
        "action": int(action),
        "is_noop": int(action) == 0,
        "demand_idx": d_idx,
        "path_idx": p_idx,
        "decoded": decoded,
        "accepted": decoded.get("accepted") if decoded.get("type") == "reroute" else None,
        "validator_reason": decoded.get("reason"),
        "rejection_source": ("environment" if decoded.get("accepted") is False
                             else None),
        "operator_override": False,
        "valid_action_count": decision.get("mask_valid_actions"),
        "explanation": decision.get("explanation"),
        "explanation_note": ("An engineering interpretation computed from measured "
                             "values. It is not the policy's internal reasoning."),
    }


# ------------------------------------------------------------------- reward
def reward_state(runner: Any, environment_version: str = "v1") -> dict[str, Any]:
    decision = runner.last_decision
    if not decision:
        return _unavailable("No interval has been rewarded yet.")
    components = decision.get("components") or {}
    order = (V1_REWARD_COMPONENTS if environment_version == "v1"
             else tuple(decision.get("component_order") or components))
    ordered = [{"name": name, "value": components.get(name)}
               for name in order if name in components]
    ordered += [{"name": name, "value": value}
                for name, value in components.items()
                if name not in {row["name"] for row in ordered}]
    total = sum(float(row["value"]) for row in ordered if row["value"] is not None)
    reward = float(decision.get("reward", 0.0))
    residual = round(reward - total, 6)
    return {
        "available": True,
        "environment_version": environment_version,
        "component_count": len(ordered),
        "components": ordered,
        "component_sum": round(total, 6),
        "interval_reward": round(reward, 6),
        "residual": residual,
        "exact_sum": abs(residual) <= 5e-4,
        "cumulative_reward": decision.get("cumulative_reward"),
        "note": ("The governed study's exact 12 V2 reward components."
                 if environment_version == "v2" else
                 "V1 reward terms. These are not the governed study's 12 V2 "
                 "components and are never padded to look like them."),
    }


# ------------------------------------------------------------------ payload
_PIPELINE_STAGES = ("observation", "mask", "policy_output", "selected_action",
                    "safety", "transition", "reward", "next_observation")


def decision_payload(session: Any, runner: Any) -> dict[str, Any]:
    grid = action_grid(runner)
    output = policy_output(runner)
    selected = selected_action(runner)
    version = getattr(runner, "environment_version", "v1")
    reward = reward_state(runner, version)
    return {
        "provenance": serialize.provenance(session, runner),
        "pipeline": list(_PIPELINE_STAGES),
        "current_stage": ("reward" if reward.get("available") else "observation"),
        "observation": observation_state(runner),
        "mask": grid,
        "policy_output": output,
        "selected_action": selected,
        "execution": {
            "mode": session.config.execution,
            "policy_acted": session.config.execution == "automatic",
            "note": ("The policy acted automatically; this is an explanation of a "
                     "completed decision, not a proposal awaiting approval."
                     if session.config.execution == "automatic" else
                     "Advisor execution holds the proposed action until an "
                     "operator approves or rejects it."),
        },
        "safety": {
            "safety_filter": bool(session.config.safety_filter),
            "validator": ("SimulationEngineV2.validate_te_action" if version == "v2"
                          else "SimulationEngine.validate_action"),
            "environment_rejection": selected.get("rejection_source") == "environment",
            "operator_rejection": False,
            "reason": selected.get("validator_reason"),
        },
        "reward": reward,
        "counterfactual": {
            "available": True,
            "kind": "clone_only",
            "reason": "POST /api/simulation/counterfactual evaluates on deep clones.",
        },
    }
