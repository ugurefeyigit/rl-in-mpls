"""Read-only HTTP surface over the frozen governed V2 evidence.

Every route is a GET. There is deliberately no route that could train, tune,
evaluate a checkpoint, reselect a checkpoint, run a sweep, or re-open the final
holdout — the study is closed, and this API is an archive reader.

Final-holdout and development/continuity evidence are served from separate routes
and every payload states its `stage`, so no client can accidentally average them.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from mplssim.evidence import claims, errors, identity, replay
from mplssim.evidence.loader import Continuity, FinalHoldout, Seed42, default_root

router = APIRouter(prefix="/api/v2", tags=["v2 study evidence"])


# ------------------------------------------------------------------ plumbing
# Indirected through module-level functions so a failure can be simulated in tests
# and so a broken artifact surfaces as an explicit outage rather than as zeros.
def _final_holdout() -> FinalHoldout:
    return FinalHoldout.load(default_root())


def _continuity() -> Continuity:
    return Continuity.load(default_root())


def _seed42() -> Seed42:
    return Seed42.load(default_root())


def _guard(fn, *a, **kw) -> Any:
    """Turn a fail-closed evidence error into an honest HTTP status."""
    try:
        return fn(*a, **kw)
    except errors.IdentityError as e:
        raise HTTPException(400, {"error": type(e).__name__, "message": str(e)}) from e
    except errors.EvidenceError as e:
        raise HTTPException(503, {"error": type(e).__name__, "message": str(e)}) from e


# -------------------------------------------------------------------- study
@router.get("/study", summary="Status and identity of the closed V2 study")
def study() -> dict:
    return {
        "status": "closed",
        "title": "Governed V2 learner comparison",
        "environment": identity.ENVIRONMENT,
        "observation_dim": identity.OBSERVATION_DIM,
        "action_count": identity.ACTION_COUNT,
        "training_roots": list(identity.TRAINING_ROOTS),
        "holdout_seeds": list(identity.HOLDOUT_SEEDS),
        "continuity_seeds": list(identity.CONTINUITY_SEEDS),
        "scenarios": list(identity.SCENARIOS),
        "learners": list(identity.LEARNER_ALGORITHMS),
        "baselines": list(identity.BASELINE_ALGORITHMS),
        "reward_components": list(identity.REWARD_COMPONENTS),
        "sources": {
            "evaluation": identity.EVALUATION_SOURCE_SHA,
            "seed42_training": identity.SEED42_SOURCE_SHA,
            "continuation_training": identity.CONTINUATION_SOURCE_SHA,
            "environment_pin": identity.SIGNED_OFF_ENV_SHA,
            "approved_ancestor": identity.APPROVED_ANCESTOR_SHA,
            "closeout": identity.CLOSEOUT_SHA,
        },
        "conclusions": list(claims.CONCLUSIONS),
        "stages": {
            identity.STAGE_DEVELOPMENT: (
                "Seed-42 pilot and three-root continuity, evaluated on seeds 101-105. "
                "Checkpoint selection happened here."),
            identity.STAGE_FINAL_HOLDOUT: (
                "One-shot evaluation on untouched seeds 1001-1005. No selection, "
                "tuning or redesign used its results."),
        },
    }


# ------------------------------------------------------------- final holdout
@router.get("/final-holdout", summary="One-shot final-holdout result")
def final_holdout() -> dict:
    fh = _guard(_final_holdout)
    return _guard(claims.holdout_summary, fh)


@router.get("/final-holdout/scenarios", summary="Seven-scenario comparison")
def final_holdout_scenarios() -> dict:
    fh = _guard(_final_holdout)
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "source_sha": identity.EVALUATION_SOURCE_SHA,
        "artifact_path": str(fh.directory),
        "grain": "root-averaged: each learner value is the mean of three training-root "
                 "means, each over five holdout seeds",
        "scenarios": _guard(claims.scenario_comparison, fh),
    }


@router.get("/final-holdout/reward-components", summary="12-component reward breakdown")
def final_holdout_reward_components() -> dict:
    fh = _guard(_final_holdout)
    return _guard(claims.reward_reconciliation, fh)


@router.get("/final-holdout/actions", summary="Action and no-op distributions")
def final_holdout_actions() -> dict:
    fh = _guard(_final_holdout)
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "source_sha": identity.EVALUATION_SOURCE_SHA,
        "distribution": _guard(claims.action_distribution, fh),
        "noop": _guard(claims.noop_shares, fh),
    }


@router.get("/final-holdout/integrity", summary="Safety and integrity status")
def final_holdout_integrity() -> dict:
    fh = _guard(_final_holdout)
    return _guard(claims.safety_summary, fh)


@router.get("/final-holdout/provenance", summary="Checkpoint provenance and runtime")
def final_holdout_provenance() -> dict:
    fh = _guard(_final_holdout)
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "source_sha": identity.EVALUATION_SOURCE_SHA,
        "artifact_path": str(fh.directory),
        "full_artifact_path": fh.manifest.get("full_artifact_path"),
        "checkpoints": _guard(claims.provenance_table, fh),
        "runtime": _guard(claims.runtime_summary, fh),
    }


# ------------------------------------------------------- development evidence
@router.get("/development/continuity", summary="Three-root continuity (development)")
def development_continuity() -> dict:
    cont = _guard(_continuity)
    return {
        "stage": identity.STAGE_DEVELOPMENT,
        "source_sha": identity.CONTINUATION_SOURCE_SHA,
        "artifact_path": str(cont.directory),
        "summary": _guard(claims.development_summary, cont),
        "learning_curves": _guard(claims.learning_curves, cont),
    }


@router.get("/development/seed42", summary="Seed-42 pilot (development)")
def development_seed42() -> dict:
    pilot = _guard(_seed42)
    return _guard(claims.pilot_summary, pilot)


# ------------------------------------------------------------------ disclosure
@router.get("/disclosures", summary="Invalidated, superseded and repaired runs")
def disclosures() -> dict:
    fh = _guard(_final_holdout)
    cont = _guard(_continuity)
    pilot = _guard(_seed42)
    return {
        "kinds": {
            "invalidated": "scientifically void; excluded from every reported result",
            "superseded": "scientifically valid, replaced so training and evaluation "
                          "share one exact source identity",
            "failed": "did not complete; preserved where a run directory existed",
            "repaired": "tooling defect found and fixed before the holdout was accessed",
        },
        "disclosures": _guard(claims.disclosures, fh, cont, pilot),
    }


# ---------------------------------------------------------------------- replay
@router.get("/replay/index", summary="Catalogue of recorded holdout episodes")
def replay_index() -> dict:
    root = default_root()
    episodes = _guard(replay.episode_index, root)
    base = replay.full_artifact_root()
    return {
        "stage": identity.STAGE_FINAL_HOLDOUT,
        "kind": "recorded_replay",
        "live": False,
        "available": base is not None,
        "artifact_root": str(base) if base else None,
        "configure_hint": (
            "Recorded step traces live outside Git. Set V2_FULL_ARTIFACTS to the "
            "directory named in results/v2_final_holdout/manifest.json "
            "(full_artifact_path) to enable replay."),
        "scenario_steps": dict(identity.SCENARIO_STEPS),
        "episodes": episodes,
    }


@router.get("/replay/episode", summary="One recorded holdout episode")
def replay_episode(
    policy_id: str = Query(..., description="One of the nine frozen policy ids"),
    scenario: str = Query(..., description="One of the seven frozen scenarios"),
    seed: int = Query(..., description="One of the five holdout seeds 1001-1005"),
) -> dict:
    return _guard(replay.load_episode, policy_id, scenario, seed)
