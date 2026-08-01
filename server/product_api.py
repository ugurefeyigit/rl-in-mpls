"""Additive product-layer HTTP surface for the unified three-mode shell.

Everything here is a read of state that already exists: the live engine, the
topology config, the observation schema and the frozen evidence. No route in
this module trains, tunes, evaluates a checkpoint, touches a holdout seed, or
writes under `results/` or `runs/`.

The existing `/api/*` and `/api/v2/*` surfaces are untouched; this router only
adds paths under `/api/product`, `/api/rl/schema` and `/api/simulation/` that
did not exist before.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from mplssim.factory import get_topology
from mplssim.product import catalog, contracts, counterfactual, decision as decision_mod
from mplssim.product import display_map as display_map_mod
from mplssim.product import results as results_mod
from mplssim.product import schemas, serialize, timeline as timeline_mod

router = APIRouter(tags=["product"])

#: Injected by `server.main` so this router never imports the app module back.
_session_provider: Callable[[], Any] | None = None


def bind_session_provider(provider: Callable[[], Any]) -> None:
    global _session_provider
    _session_provider = provider


def _session() -> Any:
    if _session_provider is None:  # pragma: no cover - wired at import time
        raise HTTPException(503, "product API is not bound to a session provider")
    session = _session_provider()
    if session is None:
        raise HTTPException(404, {
            "error": "NoLiveSession",
            "message": "No live session is running. Start one from the product "
                       "shell or POST /api/simulation/start.",
        })
    return session


def _runner(session: Any, algorithm: str | None):
    if algorithm is None:
        return session.runners[0]
    for runner in session.runners:
        if runner.algorithm == algorithm:
            return runner
    raise HTTPException(404, f"no runner for algorithm {algorithm!r} in this session")


# --------------------------------------------------------------- capabilities
@router.get("/api/product/capabilities",
            summary="What this installation can run, and why not the rest")
def capabilities() -> dict:
    return catalog.capability_catalog()


@router.get("/api/product/contracts",
            summary="Modes, routes, source kinds and product vocabulary")
def product_contracts() -> dict:
    return {
        "modes": [{"id": m.id, "label": m.label, "shortcut": m.shortcut,
                   "summary": m.summary} for m in contracts.PRIMARY_MODES],
        "workflows": [{"id": w.id, "mode": w.mode, "label": w.label,
                       "summary": w.summary} for w in contracts.WORKFLOWS],
        "rl_views": list(contracts.RL_VIEWS),
        "routes": {path: {"mode": ctx.mode, "source_kind": ctx.source_kind.value,
                          "rl_view": ctx.rl_view, "workflow": ctx.workflow,
                          "note": ctx.note}
                   for path, ctx in contracts.ROUTES.items()},
        "sources": {k.value: {
            "label": contracts.source_profile(k).label,
            "pattern": contracts.source_profile(k).pattern,
            "icon": contracts.source_profile(k).icon,
            "may_execute_policy": contracts.source_profile(k).may_execute_policy,
            "may_render_link_telemetry":
                contracts.source_profile(k).may_render_link_telemetry,
            "may_state_conclusions": contracts.source_profile(k).may_state_conclusions,
            "link_telemetry_reason": contracts.source_profile(k).link_telemetry_reason,
            "description": contracts.source_profile(k).description,
        } for k in contracts.SourceKind},
        "noop_metrics": {k: {"id": v.id, "label": v.label,
                             "denominator": v.denominator,
                             "description": v.description}
                         for k, v in contracts.NOOP_METRICS.items()},
        "output_semantics": {s.value: {"label": s.label, "description": s.description,
                                       "percent": s.percent}
                             for s in contracts.OutputSemantics},
        "disclaimer": contracts.TOPOLOGY_DISCLAIMER,
    }


@router.get("/api/product/display-map",
            summary="Display-only fixed engineering layout for the topology stage")
def display_map() -> dict:
    return display_map_mod.display_map(get_topology())


@router.get("/api/rl/schema", summary="Observation, action and reward schema")
def rl_schema(environment: str = Query("v2", pattern="^(v1|v2)$")) -> dict:
    try:
        return schemas.rl_schema(environment)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ------------------------------------------------------------------ snapshot
@router.get("/api/simulation/snapshot",
            summary="Typed product snapshot of the running session")
def snapshot(algorithm: str | None = None) -> dict:
    session = _session()
    return serialize.session_snapshot(session, _runner(session, algorithm))


@router.get("/api/simulation/moment",
            summary="Atomic product snapshot, decision, timeline and comparison")
async def moment(algorithm: str | None = None) -> dict:
    """Read every surface of one displayed moment under the session lock.

    The individual endpoints remain compatible, but the unified shell consumes
    this composite so a fast runner cannot advance between its snapshot and
    decision reads.
    """
    session = _session()
    async with session._lock:
        runner = _runner(session, algorithm)
        snapshot_payload = serialize.session_snapshot(session, runner)
        decision_payload = decision_mod.decision_payload(session, runner)
        timeline_payload = timeline_mod.session_timeline(session)
        comparison_payload = serialize.comparison_state(session)
        advisor_payload = session.advisor_status()
        provenance = snapshot_payload["provenance"]
        decision_provenance = decision_payload["provenance"]
        identity_fields = ("session_id", "generation", "step")
        if any(provenance[key] != decision_provenance[key] for key in identity_fields):
            raise HTTPException(409, "atomic moment provenance did not reconcile")
        if timeline_payload["current_step"] != provenance["step"]:
            raise HTTPException(409, "atomic moment timeline did not reconcile")
        return {
            "provenance": provenance,
            "snapshot": snapshot_payload,
            "decision": decision_payload,
            "timeline": timeline_payload,
            "comparison": comparison_payload,
            "advisor": advisor_payload,
        }


@router.get("/api/simulation/object/{kind}/{object_id}",
            summary="Focused router, link, demand or path detail")
def focused_object(kind: str, object_id: str, algorithm: str | None = None) -> dict:
    session = _session()
    try:
        return serialize.focused_object(_runner(session, algorithm), kind, object_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ------------------------------------------------------------------ decision
@router.get("/api/simulation/decision",
            summary="Observation → mask → policy output → action → reward")
def decision(algorithm: str | None = None) -> dict:
    session = _session()
    return decision_mod.decision_payload(session, _runner(session, algorithm))


@router.get("/api/simulation/timeline", summary="Typed session event timeline")
def timeline() -> dict:
    session = _session()
    return timeline_mod.session_timeline(session)


@router.get("/api/simulation/comparison",
            summary="Synchronization proof and the paired decision comparison")
def comparison() -> dict:
    return serialize.comparison_state(_session())


# ------------------------------------------------------------------- results
@router.get("/api/product/results",
            summary="Live run, retained runs and the study record, kept apart")
def product_results() -> dict:
    """Three record classes in three sections.

    This route never 404s on a missing session: retained runs and the pointer to
    the governed study outlive any one session, and a full reset is exactly when
    an operator wants to read them.
    """
    session = _session_provider() if _session_provider else None
    return results_mod.results(session)


# ------------------------------------------------------------ counterfactual
class CounterfactualRequest(BaseModel):
    action: int = Field(..., ge=0, lt=contracts.ACTION_COUNT)
    generation: int | None = None
    step: int | None = None
    algorithm: str | None = None


@router.post("/api/simulation/counterfactual",
             summary="Clone-only simulated one-interval estimate")
def post_counterfactual(req: CounterfactualRequest) -> dict:
    session = _session()
    runner = _runner(session, req.algorithm)
    result = counterfactual.estimate(session, runner, req.action,
                                     generation=req.generation, step=req.step)
    if result.get("kind") == "unavailable" and result.get("http_status"):
        raise HTTPException(result.pop("http_status"), result)
    return result
