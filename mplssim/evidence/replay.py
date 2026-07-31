"""Recorded replay of preserved final-holdout episodes.

The one-shot evaluation wrote a compressed per-step trace for all 315 episodes. This
module reads those traces back. It is a tape player, not an evaluator: it never
constructs an environment, never imports or loads a learner, never selects a
checkpoint, and never writes anything.

The traces are large and live outside Git. Point `V2_FULL_ARTIFACTS` at the directory
recorded in `results/v2_final_holdout/manifest.json` to enable replay; without it the
catalogue still lists every episode and reports each one as unavailable.
"""

from __future__ import annotations

import csv
import gzip
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from mplssim.evidence import identity
from mplssim.evidence.errors import ArtifactMissingError, IdentityError
from mplssim.evidence.loader import EvidenceRoot, FinalHoldout, default_root

#: Numeric per-step fields worth carrying into a replay timeline.
_FLOAT_FIELDS = (
    "reward", "moved_mbps", "dwell_remaining_mean", "delivered_ratio", "max_util",
    "gross_max_util", "overload_ratio", "mean_util", "util_std", "mean_delay_ms",
    "max_delay_ms", "loss_ratio", "sla_violation_fraction", "offered_mbps",
    "delivered_mbps", "t_min", "hour", "protected_disconnect", "unprotected_disconnect",
    "sla_severity",
)
_INT_FIELDS = (
    "step_index", "action", "valid_action_count", "sla_violations", "congested_links",
    "disconnected_demands", "protected_disconnected_demands", "step", "n_demands",
    "accepted_te_changes", "rejected_te_requests", "te_reversals", "frr_changes",
    "frr_disconnections", "recovery_restorations", "n_failed_links",
)
_BOOL_FIELDS = ("action_accepted", "terminated", "truncated")


# ------------------------------------------------------------------ location
def full_artifact_root() -> Path | None:
    """Where the preserved step traces live, or `None` when unavailable.

    Resolution order: `$V2_FULL_ARTIFACTS`, then the path the final-holdout manifest
    recorded. Absence is a normal, reportable state — it never raises.
    """
    override = os.environ.get("V2_FULL_ARTIFACTS", "").strip()
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    try:
        manifest_path = default_root().final_holdout / "manifest.json"
        if not manifest_path.is_file():
            return None
        import json
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")) \
            .get("full_artifact_path")
    except Exception:
        return None
    if not recorded:
        return None
    p = Path(recorded)
    return p if p.is_dir() else None


def replay_available() -> bool:
    return full_artifact_root() is not None


# ------------------------------------------------------------------ catalogue
@lru_cache(maxsize=4)
def _policies(results_dir: str) -> tuple[dict[str, Any], ...]:
    """Policy identities from the committed compact evidence."""
    fh = FinalHoldout.load(EvidenceRoot(Path(results_dir)))
    prov = {(int(r["training_root"]), r["algorithm"]): r
            for _, r in fh.provenance.iterrows()}
    out: list[dict[str, Any]] = []
    for _, r in fh.per_root.iterrows():
        root_raw = str(r["training_root"])
        is_learner = root_raw != "baseline"
        root = int(root_raw) if is_learner else None
        p = prov.get((root, r["algorithm"])) if is_learner else None
        out.append({
            "policy_id": r["policy_id"],
            "algorithm": r["algorithm"],
            "training_root": root,
            "checkpoint_transition": int(p["checkpoint_transition"]) if p is not None
            else None,
            "training_source_sha": p["training_source_sha"] if p is not None else None,
            "is_learner": is_learner,
        })
    return tuple(out)


def episode_index(root: EvidenceRoot | None = None) -> list[dict[str, Any]]:
    """Every recorded holdout episode, with per-episode availability."""
    root = root or default_root()
    base = full_artifact_root()
    out: list[dict[str, Any]] = []
    for policy in _policies(str(root.results_dir)):
        for scenario in identity.SCENARIOS:
            for seed in identity.HOLDOUT_SEEDS:
                path = _trace_path(base, policy, scenario, seed) if base else None
                out.append({
                    **policy,
                    "scenario": scenario,
                    "seed": seed,
                    "stage": identity.STAGE_FINAL_HOLDOUT,
                    "kind": "recorded_replay",
                    "live": False,
                    "available": bool(path and path.is_file()),
                })
    return out


def _trace_path(base: Path, policy: dict[str, Any], scenario: str, seed: int) -> Path:
    algo = policy["algorithm"]
    stem = f"{algo}_{scenario}_seed{seed}_steps.csv.gz"
    if policy["is_learner"]:
        return base / "learners" / policy["policy_id"] / "steps" / stem
    return base / "baselines" / algo / "steps" / stem


# -------------------------------------------------------------------- reading
def _coerce(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _INT_FIELDS:
        if k in row and row[k] != "":
            out[k] = int(float(row[k]))
    for k in _FLOAT_FIELDS:
        if k in row and row[k] != "":
            out[k] = float(row[k])
    for k in _BOOL_FIELDS:
        if k in row:
            out[k] = row[k].strip().lower() == "true"
    out["action_type"] = row.get("action_type", "")
    out["components"] = {
        c: float(row[f"rc_{c}"]) for c in identity.REWARD_COMPONENTS if f"rc_{c}" in row
    }
    return out


def _read_steps(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            yield _coerce(row)


@lru_cache(maxsize=1)
def _episode_summaries(base: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Frozen per-episode facts, keyed by (policy_id, scenario, seed)."""
    path = Path(base) / "final_holdout_episode_summary.csv"
    if not path.is_file():
        return {}
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    with open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["policy_id"], row["scenario"], int(row["seed"]))] = {
                "operational_return": float(row["operational_return"]),
                "episode_seed": int(row["episode_seed"]),
                "episode_length": int(row["episode_length"]),
                "truncated": row["truncated"].strip().lower() == "true",
                "terminated": row["terminated"].strip().lower() == "true",
                "reward_component_sum_exact":
                    row["reward_component_sum_exact"].strip().lower() == "true",
            }
    return out


def load_episode(policy_id: str, scenario: str, seed: int,
                 root: EvidenceRoot | None = None) -> dict[str, Any]:
    """Return one recorded episode: its provenance and its preserved step sequence."""
    root = root or default_root()

    if scenario not in identity.SCENARIOS:
        raise IdentityError(
            f"{scenario!r} is not one of the seven frozen scenarios")
    if seed not in identity.HOLDOUT_SEEDS:
        raise IdentityError(
            f"seed {seed} is not a final-holdout seed {list(identity.HOLDOUT_SEEDS)}; "
            "replay serves holdout evidence only")
    policy = next((p for p in _policies(str(root.results_dir))
                   if p["policy_id"] == policy_id), None)
    if policy is None:
        raise IdentityError(f"{policy_id!r} is not one of the nine frozen policies")

    base = full_artifact_root()
    if base is None:
        raise ArtifactMissingError(
            "recorded step traces are not configured on this machine. Set "
            "V2_FULL_ARTIFACTS to the preserved run directory recorded in "
            "results/v2_final_holdout/manifest.json (full_artifact_path).")

    path = _trace_path(base, policy, scenario, seed)
    if not path.is_file():
        raise ArtifactMissingError(f"recorded trace not found: {path}")

    steps = list(_read_steps(path))
    summary = _episode_summaries(str(base)).get((policy_id, scenario, seed), {})

    return {
        "provenance": {
            **policy,
            "scenario": scenario,
            "seed": seed,
            "stage": identity.STAGE_FINAL_HOLDOUT,
            "kind": "recorded_replay",
            "live": False,
            "evaluation_source_sha": identity.EVALUATION_SOURCE_SHA,
            "environment": identity.ENVIRONMENT,
            "artifact_path": str(path),
            "steps": len(steps),
            **summary,
        },
        "steps": steps,
    }
