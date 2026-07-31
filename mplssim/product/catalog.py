"""What this installation can actually do, and why it cannot do the rest.

The product may only offer controllers, environments and sources that really
exist here. A policy whose checkpoint is not bound appears as *unavailable with
a reason* — never as a selectable option that fails at click time, and never
silently swapped for a different policy.

Availability is computed from the filesystem and the evidence loaders at call
time, so a machine with the V2 full artifacts configured and a machine without
them describe themselves differently and correctly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mplssim.evidence import identity, replay
from mplssim.evidence.loader import default_root
from mplssim.product.contracts import (
    ENVIRONMENTS, LIVE_DEMONSTRATION_LABEL, OutputSemantics, SourceKind,
    source_profile,
)

ROOT = Path(__file__).resolve().parents[2]

#: Where a V2 live-demonstration checkpoint binding would be configured. The
#: governed study's checkpoints live outside Git; without this binding the
#: product truthfully reports that no V2 controller can be run live here.
V2_LIVE_CHECKPOINTS_ENV = "V2_LIVE_CHECKPOINTS"


@dataclass(frozen=True)
class PolicyCapability:
    """One selectable controller, in one environment version."""

    id: str
    label: str
    environment_version: str
    family: str                      # learner | baseline
    output_semantics: OutputSemantics
    available: bool
    unavailable_reason: str | None = None
    checkpoint_id: str | None = None
    exposes_entropy: bool = False
    exposes_value: bool = False
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "environment_version": self.environment_version,
            "family": self.family,
            "output_semantics": self.output_semantics.value,
            "output_label": self.output_semantics.label,
            "output_description": self.output_semantics.description,
            "output_is_percentage": self.output_semantics.percent,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "checkpoint_id": self.checkpoint_id,
            "exposes_entropy": self.exposes_entropy,
            "exposes_value": self.exposes_value,
            "description": self.description,
        }


def _v1_model_available(tag: str) -> bool:
    base = ROOT / "models" / tag
    return (base / "best_model.zip").is_file() or (base / "final_model.zip").is_file()


def _v2_checkpoint_root() -> Path | None:
    configured = os.environ.get(V2_LIVE_CHECKPOINTS_ENV, "").strip()
    if not configured:
        return None
    path = Path(configured)
    return path if path.is_dir() else None


def _v2_binding(policy_id: str) -> tuple[bool, str | None, str | None]:
    """Is a frozen V2 checkpoint bound for live *inference only* demonstration?

    Returns `(available, checkpoint_id, reason)`. A bound checkpoint is loaded
    for inference and writes no evidence: using one in a demonstration does not
    create, modify or extend the governed study.
    """
    root = _v2_checkpoint_root()
    if root is None:
        return (False, None, (
            f"No V2 live-demonstration checkpoint is bound. Set "
            f"{V2_LIVE_CHECKPOINTS_ENV} to a directory holding the frozen "
            f"governed checkpoints to enable a {LIVE_DEMONSTRATION_LABEL} run. "
            f"The closed study's results are unaffected either way."))
    candidates = sorted(root.glob(f"{policy_id}*"))
    if not candidates:
        return (False, None,
                f"{V2_LIVE_CHECKPOINTS_ENV} is set but holds no checkpoint for "
                f"{policy_id!r}.")
    return (True, candidates[0].name, None)


def live_policies() -> list[PolicyCapability]:
    """Controllers that can drive a *live* session on this machine."""
    out: list[PolicyCapability] = []

    ppo_available = _v1_model_available("ppo_te")
    out.append(PolicyCapability(
        id="rl", label="MaskablePPO (V1)", environment_version="v1", family="learner",
        output_semantics=OutputSemantics.PROBABILITIES,
        available=ppo_available,
        unavailable_reason=None if ppo_available else
        "models/ppo_te holds no best_model.zip or final_model.zip.",
        checkpoint_id="ppo_te" if ppo_available else None,
        exposes_entropy=False, exposes_value=False,
        description="The installed V1 MaskablePPO checkpoint. Masked action "
                    "probabilities are read from the policy distribution."))

    for pid, label, desc in (
        ("static", "Static shortest path",
         "Fixed shortest-path routing. Never reroutes for congestion."),
        ("greedy", "Utilization-aware greedy",
         "Moves the demand crossing the busiest link onto its least-loaded candidate."),
        ("cspf", "CSPF periodic reoptimization",
         "Constrained shortest-path reoptimization on a fixed period."),
        ("random", "Random sanity floor",
         "Uniform random valid action. Present as a floor, not as a method."),
    ):
        out.append(PolicyCapability(
            id=pid, label=label, environment_version="v1", family="baseline",
            output_semantics=OutputSemantics.NONE, available=True,
            description=desc))

    for pid, label, semantics, desc in (
        ("maskable_ppo", "MaskablePPO (V2)", OutputSemantics.PROBABILITIES,
         "The governed study's PPO learner, run for demonstration only."),
        ("masked_bandit", "Masked contextual bandit", OutputSemantics.SCORES,
         "The governed study's bandit learner. Its per-action numbers are "
         "immediate-reward estimates, not probabilities."),
    ):
        available, checkpoint_id, reason = _v2_binding(pid)
        out.append(PolicyCapability(
            id=pid, label=label, environment_version="v2", family="learner",
            output_semantics=semantics, available=available,
            unavailable_reason=reason, checkpoint_id=checkpoint_id,
            description=desc))
    return out


def evidence_policies() -> list[dict[str, Any]]:
    """The nine frozen policies the closed study reports. Read-only, always."""
    rows = []
    for algo in identity.LEARNER_ALGORITHMS:
        rows.append({
            "id": algo,
            "family": "learner",
            "roots": list(identity.TRAINING_ROOTS),
            "output_semantics": (OutputSemantics.SCORES.value
                                 if algo == "masked_bandit"
                                 else OutputSemantics.PROBABILITIES.value),
        })
    for algo in identity.BASELINE_ALGORITHMS:
        rows.append({"id": algo, "family": "baseline", "roots": [],
                     "output_semantics": OutputSemantics.NONE.value})
    return rows


def _evidence_available() -> tuple[bool, str | None]:
    try:
        root = default_root()
    except Exception as exc:  # pragma: no cover - defensive
        return (False, f"Evidence root unavailable: {exc}")
    if not root.final_holdout.is_dir():
        return (False, f"Frozen evidence directory {root.final_holdout} is missing.")
    return (True, None)


def source_capabilities() -> list[dict[str, Any]]:
    evidence_ok, evidence_reason = _evidence_available()
    replay_ok = replay.replay_available()
    rows = []
    for kind, available, reason in (
        (SourceKind.LIVE_SESSION, True, None),
        (SourceKind.RECORDED_REPLAY, replay_ok and evidence_ok,
         None if (replay_ok and evidence_ok) else (
             evidence_reason or
             "Recorded step traces live outside Git. Set V2_FULL_ARTIFACTS to the "
             "directory named in results/v2_final_holdout/manifest.json "
             "(full_artifact_path) to enable replay.")),
        (SourceKind.DEVELOPMENT_EVIDENCE, evidence_ok, evidence_reason),
        (SourceKind.FINAL_HOLDOUT_EVIDENCE, evidence_ok, evidence_reason),
    ):
        profile = source_profile(kind)
        rows.append({
            "kind": kind.value,
            "label": profile.label,
            "pattern": profile.pattern,
            "icon": profile.icon,
            "description": profile.description,
            "may_execute_policy": profile.may_execute_policy,
            "may_render_link_telemetry": profile.may_render_link_telemetry,
            "may_state_conclusions": profile.may_state_conclusions,
            "link_telemetry_reason": profile.link_telemetry_reason,
            "required_fields": list(profile.required_fields),
            "available": available,
            "unavailable_reason": reason,
        })
    return rows


def environment_capabilities() -> list[dict[str, Any]]:
    policies = live_policies()
    rows = []
    for version, env in ENVIRONMENTS.items():
        runnable = [p for p in policies
                    if p.environment_version == version and p.available]
        rows.append({
            "version": version,
            "label": env.label,
            "env_class": env.env_class,
            "observation_dim": env.observation_dim,
            "action_count": env.action_count,
            "reward_components": list(env.reward_components),
            "summary": env.summary,
            "live_available": bool(runnable),
            "live_unavailable_reason": None if runnable else (
                "No controller with a bound checkpoint is available for this "
                "environment version on this machine."),
            "supports_clone_counterfactual": version == "v1",
            "clone_reason": (
                "The live V1 engine exposes clone(); a counterfactual is evaluated "
                "on deep copies and the running session is left untouched."
                if version == "v1" else
                "No live V2 session can run here, so no V2 state exists to clone."),
        })
    return rows


def capability_catalog() -> dict[str, Any]:
    """`GET /api/product/capabilities`."""
    return {
        "product": "RL-in-MPLS",
        "modes": ["presentation", "network", "rl"],
        "guided_story_mode": "presentation",
        "sources": source_capabilities(),
        "environments": environment_capabilities(),
        "live_policies": [p.as_dict() for p in live_policies()],
        "evidence_policies": evidence_policies(),
        "live_demonstration_label": LIVE_DEMONSTRATION_LABEL,
        "holdout_seeds_blocked_for_live": list(identity.HOLDOUT_SEEDS),
        "comparison": {
            "max_synchronized_policies": 2,
            "requires_fingerprint_match": True,
            "reason": "A comparison is only shown when both runners provably start "
                      "from the same state and receive the same exogenous inputs.",
        },
    }
