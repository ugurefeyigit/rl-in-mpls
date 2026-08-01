"""FastAPI backend: simulation control, telemetry streaming, advisor mode,
training control, presentation support.

Run:  python -m uvicorn server.main:app --port 8000
Docs: http://127.0.0.1:8000/docs        (OpenAPI, always current)
UI:   http://127.0.0.1:8000/            (Advanced engineering console)
      http://127.0.0.1:8000/present     (Presentation Mode)
WS:   ws://127.0.0.1:8000/ws/telemetry  (tick stream, JSON)

Set ALLOW_TRAINING=false (the demo default) to disable the training endpoint.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mplssim.display import display_bundle, scenario_label
from mplssim.evidence import identity
from mplssim.factory import get_scenarios, get_topology, get_traffic_config
from mplssim.product import checkpoints_v2, results, run_summary
from mplssim.validation import ConfigError, validate_configs
from server import db
from server.events import log_event, recent_events
from server.evidence_api import router as evidence_router
from server.product_api import bind_session_provider
from server.product_api import router as product_router
from server.session import (
    DEFAULT_ENVIRONMENT, SessionConfig, SessionError, SessionState, SimSession,
    algorithms_for, list_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

STATE: dict[str, Any] = {"session": None, "training": None}


def training_allowed() -> bool:
    return os.environ.get("ALLOW_TRAINING", "true").strip().lower() not in (
        "false", "0", "no", "off")


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        validate_configs()
        log_event("startup", training_allowed=training_allowed())
    except ConfigError as e:
        # Fail loudly in the log; endpoints will re-raise on use.
        logging.getLogger("mplssim.server").error("CONFIG INVALID: %s", e)
    yield


app = FastAPI(
    title="RL-in-MPLS Traffic Engineering API",
    version="1.1.0",
    description="Flow-level MPLS-TE simulation with RL and heuristic controllers.",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ schemas
class StartRequest(BaseModel):
    scenario: str = "demo_evening"
    #: V2 is the truthful default: it is the governed study environment and the
    #: only one with continuity-selected checkpoints. V1 must be asked for.
    environment: str = Field(default=DEFAULT_ENVIRONMENT, pattern="^(v1|v2)$")
    algorithms: list[str] = Field(default=["masked_bandit"], min_length=1,
                                  max_length=2)
    seed: int = 42
    model_tag: str | None = "ppo_te"       # V1 checkpoint tag only
    training_root: int = checkpoints_v2.DEFAULT_ROOT   # V2 checkpoint root only
    safety_filter: bool = True
    speed: str = "1x"
    autostart: bool = True
    #: "automatic" runs the policy; "advisor" holds each proposed action for an
    #: operator decision. `advisor` remains accepted for existing clients.
    execution: str | None = Field(default=None, pattern="^(automatic|advisor)$")
    advisor: bool = False
    interface_mode: str = "advanced"


class SpeedRequest(BaseModel):
    speed: str


class FailureRequest(BaseModel):
    link: str


class BurstRequest(BaseModel):
    demand: str
    factor: float = 2.0
    duration_min: float = 60.0


class MultiplierRequest(BaseModel):
    factor: float = 1.0


class RunUntilRequest(BaseModel):
    condition: str = "next_event"   # next_event | congestion | failure | recovery | end
    max_steps: int = 300
    util_threshold: float = 0.9
    #: Advisor execution only. A fast-forward applies the controller's own
    #: actions for a stretch of intervals without individual approval, so the
    #: caller must say it is delegating them. Refused with the reason otherwise.
    delegate: bool = False


class TrainRequest(BaseModel):
    timesteps: int = 100000
    tag: str = "ppo_custom"
    seed: int = 42
    confirm: bool = False


def current_session() -> SimSession:
    s: SimSession | None = STATE["session"]
    if s is None:
        raise HTTPException(404, "no active session — POST /api/simulation/start first")
    return s


def _handle(coro):
    """Await a session coroutine, mapping SessionError to HTTP 409."""
    async def run():
        try:
            return await coro
        except SessionError as e:
            raise HTTPException(409, str(e)) from e
    return run()


# ------------------------------------------------------------- static info
@app.get("/api/topology")
def topology() -> dict:
    topo = get_topology()
    return {
        "routers": [vars(r) for r in topo.routers.values()],
        "links": [vars(l) for l in topo.link_defs.values()],
    }


@app.get("/api/display")
def display() -> dict:
    return display_bundle(get_topology())


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {
        name: {"description": s.description, "display_name": scenario_label(name),
               "start_hour": s.start_hour, "duration_min": s.duration_min,
               "events": s.events, "randomized": s.randomize is not None}
        for name, s in get_scenarios().items()
    }


@app.get("/api/traffic-classes")
def traffic_classes() -> dict:
    cfg = get_traffic_config()
    return {
        "classes": {n: vars(c) for n, c in cfg.classes.items()},
        "demands": [
            {"id": d.id, "src": d.src, "dst": d.dst, "class": d.cls.name,
             "base_mbps": d.base_mbps} for d in cfg.demands
        ],
    }


@app.get("/api/checkpoints")
def checkpoints() -> list[dict]:
    return list_checkpoints()


@app.get("/api/events")
def events(limit: int = 100) -> list[dict]:
    return recent_events(limit)


# ---------------------------------------------------------------- benchmark
@app.get("/api/benchmark")
def benchmark() -> dict:
    """Published V1 multi-seed evaluation, read from the committed CSV —
    never hardcoded. Marks the winner per scenario by mean reward."""
    path = ROOT / "results" / "eval_stats.csv"
    if not path.exists():
        raise HTTPException(404, "results/eval_stats.csv not found")
    df = pd.read_csv(path)
    out: dict[str, Any] = {"source": "results/eval_stats.csv (Version 1, 5 seeds, paired)",
                           "scenarios": {}}
    for scen, g in df.groupby("scenario"):
        rows = {}
        for _, r in g.iterrows():
            rows[r["algorithm"]] = {
                "reward_mean": round(float(r["reward_sum_mean"]), 1),
                "reward_ci95": round(float(r["reward_sum_ci95"]), 1),
                "max_util_mean": round(float(r["max_util_mean_mean"]), 3),
                "sla_violations_mean": round(float(r["sla_violations_total_mean"]), 1),
                "reroutes_mean": round(float(r["reroutes_total_mean"]), 1),
                "n_seeds": int(r["n_seeds"]),
            }
        winner = max(rows, key=lambda a: rows[a]["reward_mean"])
        out["scenarios"][scen] = {
            "display_name": scenario_label(scen),
            "algorithms": rows,
            "winner": winner,
        }
    return out


# ------------------------------------------------------ simulation control
@app.post("/api/simulation/start")
async def sim_start(req: StartRequest) -> dict:
    if req.seed in identity.HOLDOUT_SEEDS:
        raise HTTPException(400, "frozen final-holdout seeds are blocked for live sessions")
    if req.seed < 0:
        raise HTTPException(400, "seed must be a non-negative integer")
    if req.scenario not in get_scenarios():
        raise HTTPException(400, f"unknown scenario {req.scenario}")
    allowed = algorithms_for(req.environment)
    for a in req.algorithms:
        if a not in allowed:
            raise HTTPException(400, (
                f"{a!r} cannot run in the {req.environment.upper()} environment. "
                f"Available: {', '.join(allowed)}."))
    if req.environment == "v2" and req.training_root not in checkpoints_v2.TRAINING_ROOTS:
        raise HTTPException(400, (
            f"training root {req.training_root} is not one of the study's "
            f"continuity roots {list(checkpoints_v2.TRAINING_ROOTS)}."))
    advisor = req.advisor if req.execution is None else req.execution == "advisor"
    try:
        session = SimSession(SessionConfig(
            scenario=req.scenario, algorithms=tuple(req.algorithms),
            seed=req.seed, model_tag=req.model_tag,
            safety_filter=req.safety_filter, speed=req.speed,
            interface_mode=req.interface_mode, advisor=advisor,
            environment=req.environment, training_root=req.training_root,
        ))
    # A missing or incompatible V2 checkpoint fails closed here, with the
    # verification reason, instead of degrading the session to V1.
    except checkpoints_v2.CheckpointUnavailable as e:
        raise HTTPException(409, str(e)) from e
    except (FileNotFoundError, ConfigError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    old: SimSession | None = STATE["session"]
    if old is not None:
        await old.pause()
    STATE["session"] = session
    session.subscribers.append(WS_HUB.send)
    if req.autostart and not advisor:
        await session.resume()
    return session.status()


@app.post("/api/simulation/pause")
async def sim_pause() -> dict:
    return await _handle(current_session().pause())


@app.post("/api/simulation/resume")
async def sim_resume() -> dict:
    return await _handle(current_session().resume())


@app.post("/api/simulation/step")
async def sim_step() -> dict:
    return await _handle(current_session().step_manual())


@app.post("/api/simulation/reset")
async def sim_reset() -> dict:
    """Reset run: same experiment at step zero; the replaced run is retained."""
    return await _handle(current_session().reset())


@app.post("/api/simulation/stop")
async def sim_stop() -> dict:
    """Full reset: stop the runners and return to initial configuration.

    Clears the active session only. No model, checkpoint or evidence artifact is
    read, written or modified.
    """
    session: SimSession | None = STATE["session"]
    if session is None:
        return {"stopped": False, "reason": "no active session",
                "retained_runs": len(results.process_retained())}
    # The session's archive — and the run that was on screen — are handed to the
    # process store, so returning to the configuration form does not silently
    # discard what the operator just watched. A restart still drops everything;
    # see docs/ADR-003 for why a demonstration number is never persisted.
    handed = results.hand_over(session)
    await session.shutdown()
    STATE["session"] = None
    log_event("session_full_reset", scenario=session.config.scenario,
              handed_over=handed)
    return {"stopped": True, "session_id": session.id,
            "handed_to_process_store": handed,
            "retained_runs": len(results.process_retained())}


@app.get("/api/simulation/retained-runs")
def sim_retained_runs() -> dict:
    """Runs archived by a reset run or handed over by a full reset.

    Never 404s on a missing session: a full reset is exactly when the operator
    wants to look at what was kept.
    """
    return results.retained_runs(STATE["session"])


@app.post("/api/simulation/run-until")
async def sim_run_until(req: RunUntilRequest) -> dict:
    return await _handle(current_session().run_until(
        req.condition, req.max_steps, req.util_threshold, req.delegate))


@app.post("/api/simulation/speed")
async def sim_speed(req: SpeedRequest) -> dict:
    try:
        return await current_session().set_speed(req.speed)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/simulation/status")
def sim_status() -> dict:
    s: SimSession | None = STATE["session"]
    return s.status() if s else {"state": "idle", "running": False, "session": None}


@app.get("/api/telemetry/current")
def telemetry_current() -> dict:
    return current_session().payload()


# ----------------------------------------------------------- interventions
@app.post("/api/failure/inject")
async def failure_inject(req: FailureRequest) -> dict:
    if req.link not in get_topology().link_defs:
        raise HTTPException(400, f"unknown link {req.link}")
    return await _handle(current_session().inject_failure(req.link))


@app.post("/api/failure/recover")
async def failure_recover(req: FailureRequest) -> dict:
    if req.link not in get_topology().link_defs:
        raise HTTPException(400, f"unknown link {req.link}")
    return await _handle(current_session().recover_link(req.link))


@app.post("/api/traffic/burst")
async def traffic_burst(req: BurstRequest) -> dict:
    s = current_session()
    if req.demand not in s.runners[0].eng.demand_by_id:
        raise HTTPException(400, f"unknown demand {req.demand}")
    return await _handle(s.inject_burst(req.demand, req.factor, req.duration_min))


@app.post("/api/traffic/multiplier")
async def traffic_multiplier(req: MultiplierRequest) -> dict:
    return await _handle(current_session().set_multiplier(req.factor))


# ---------------------------------------------------------------- advisor
@app.post("/api/advisor/propose")
async def advisor_propose() -> dict:
    return await _handle(current_session().advisor_propose())


@app.post("/api/advisor/approve")
async def advisor_approve() -> dict:
    return await _handle(current_session().advisor_approve())


@app.post("/api/advisor/reject")
async def advisor_reject() -> dict:
    return await _handle(current_session().advisor_reject())


@app.get("/api/advisor/status")
def advisor_status() -> dict:
    return current_session().advisor_status()


# ----------------------------------------------------------------- metrics
@app.get("/api/metrics/history")
def metrics_history() -> dict:
    s = current_session()
    return {
        "runs": [
            {"algorithm": r.algorithm,
             "history": [
                 {**h["metrics"], "reward": h["reward"],
                  "n_failed_links": h["n_failed_links"]}
                 for h in r.history
             ]}
            for r in s.runners
        ]
    }


@app.get("/api/lsps")
def lsps() -> dict:
    s = current_session()
    return {"runs": [
        {"algorithm": r.algorithm, "demands": r.eng.snapshot()["demands"]}
        for r in s.runners
    ]}


@app.get("/api/links")
def links() -> dict:
    s = current_session()
    return {"runs": [
        {"algorithm": r.algorithm, "links": r.eng.snapshot()["links"]}
        for r in s.runners
    ]}


@app.get("/api/agent/status")
def agent_status() -> dict:
    s = current_session()
    return {
        "runs": [
            {"algorithm": r.algorithm, "last_decision": r.last_decision,
             "cumulative_reward": r.cumulative_reward}
            for r in s.runners
        ]
    }


# ------------------------------------------------------------------ export
def _session_rows(s: SimSession) -> list[dict]:
    rows: list[dict] = []
    for r in s.runners:
        for h in r.history:
            row = dict(h["metrics"])
            row["n_failed_links"] = h["n_failed_links"]
            row["reward"] = h["reward"]
            row.update({f"rc_{k}": v for k, v in h["components"].items()})
            row["algorithm"] = r.algorithm
            row["scenario"] = s.config.scenario
            row["seed"] = s.config.seed
            rows.append(row)
    return rows


@app.get("/api/export/results")
def export_results(fmt: str = "csv") -> Response:
    rows = _session_rows(current_session())
    if not rows:
        raise HTTPException(409, "no metrics recorded yet")
    if fmt == "json":
        return Response(json.dumps(rows, indent=1), media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=results.json"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=results.csv"})


@app.post("/api/export/save-run")
def save_run() -> dict:
    """Save each lane's episode summary, summarized for its own environment.

    V1 and V2 record different interval columns, so each gets its own
    summarizer. A V2 row is never padded into the shape of a V1 row.
    """
    s = current_session()
    if not any(r.history for r in s.runners):
        raise HTTPException(409, "no interval has completed yet, so there is "
                                 "nothing to save")
    ids = []
    for r in s.runners:
        if not r.history:
            continue
        summary = run_summary.summarize_session_runner(
            r, scenario=s.config.scenario, seed=s.config.seed,
            environment=s.config.environment,
            training_root=(s.config.training_root
                           if s.config.environment == "v2" else None))
        ids.append(db.save_run("live", s.config.scenario, r.algorithm,
                               s.config.seed, summary))
    log_event("run_saved", ids=ids, scenario=s.config.scenario,
              environment=s.config.environment)
    return {"saved_run_ids": ids, "environment": s.config.environment}


@app.get("/api/runs")
def runs(limit: int = 50) -> list[dict]:
    return db.list_runs(limit)


# ---------------------------------------------------------------- training
@app.post("/api/agent/train")
def train_start(req: TrainRequest) -> dict:
    if not training_allowed():
        raise HTTPException(403, "Training is disabled during presentation mode "
                                 "(ALLOW_TRAINING=false).")
    if not req.confirm:
        raise HTTPException(400, "training requires confirm=true (the UI shows "
                                 "a confirmation dialog first)")
    job = STATE.get("training")
    if job and job["proc"].poll() is None:
        raise HTTPException(409, "a training job is already running")
    log: deque[str] = deque(maxlen=200)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "train.py"),
         "--timesteps", str(req.timesteps), "--tag", req.tag, "--seed", str(req.seed)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )

    def _pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            log.append(line.rstrip())

    import threading
    threading.Thread(target=_pump, daemon=True).start()
    STATE["training"] = {"proc": proc, "log": log, "tag": req.tag,
                         "timesteps": req.timesteps}
    log_event("training_started", model_tag=req.tag, timesteps=req.timesteps)
    return {"started": True, "tag": req.tag, "timesteps": req.timesteps}


@app.get("/api/training/progress")
def training_progress() -> dict:
    job = STATE.get("training")
    if not job:
        return {"active": False, "allowed": training_allowed(), "log": []}
    running = job["proc"].poll() is None
    return {
        "active": running,
        "allowed": training_allowed(),
        "exit_code": None if running else job["proc"].returncode,
        "tag": job["tag"],
        "timesteps": job["timesteps"],
        "log": list(job["log"])[-60:],
    }


# --------------------------------------------------------------- websocket
class WsHub:
    """Single fan-out point: sessions broadcast to `send`, which relays to all sockets."""

    def __init__(self) -> None:
        self.sockets: list[WebSocket] = []

    async def send(self, payload: dict) -> None:
        text = json.dumps(payload)
        dead = []
        for ws in self.sockets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.sockets:
                self.sockets.remove(ws)


WS_HUB = WsHub()


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    WS_HUB.sockets.append(ws)
    log_event("ws_connected", clients=len(WS_HUB.sockets))
    s: SimSession | None = STATE["session"]
    if s is not None:
        await ws.send_text(json.dumps(s.payload()))
    try:
        while True:
            await ws.receive_text()  # keepalive pings from the client
    except WebSocketDisconnect:
        if ws in WS_HUB.sockets:
            WS_HUB.sockets.remove(ws)
        log_event("ws_disconnected", clients=len(WS_HUB.sockets))


# ---------------------------------------------------------------- frontend
@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "app.html")


@app.get("/advanced")
def advanced() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "app.html")


@app.get("/present")
def present() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "app.html")


@app.get("/study")
def study_page() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "app.html")


# Read-only V2 study evidence. GET-only by construction — see server/evidence_api.py.
app.include_router(evidence_router)

# Additive product-layer surface for the unified three-mode shell. It reads the
# live session through this provider rather than importing this module back.
bind_session_provider(lambda: STATE["session"])
app.include_router(product_router)

app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
