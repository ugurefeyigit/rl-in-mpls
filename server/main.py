"""FastAPI backend: simulation control, telemetry streaming, training control.

Run:  python -m uvicorn server.main:app --port 8000
Docs: http://127.0.0.1:8000/docs        (OpenAPI, always current)
UI:   http://127.0.0.1:8000/            (dashboard, served statically)
WS:   ws://127.0.0.1:8000/ws/telemetry  (tick stream, JSON)
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mplssim.factory import get_scenarios, get_topology, get_traffic_config
from mplssim.experiments.runner import summarize_records
from server import db
from server.session import SimSession, list_checkpoints

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="RL-in-MPLS Traffic Engineering API",
    version="1.0.0",
    description="Flow-level MPLS-TE simulation with RL and heuristic controllers.",
)

STATE: dict[str, Any] = {"session": None, "training": None}


# ------------------------------------------------------------------ schemas
class StartRequest(BaseModel):
    scenario: str = "demo_evening"
    algorithms: list[str] = Field(default=["rl"], min_length=1, max_length=2)
    seed: int = 42
    model_tag: str | None = "ppo_te"
    safety_filter: bool = True
    speed: str = "1x"
    autostart: bool = True


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


class TrainRequest(BaseModel):
    timesteps: int = 100000
    tag: str = "ppo_custom"
    seed: int = 42


def current_session() -> SimSession:
    s: SimSession | None = STATE["session"]
    if s is None:
        raise HTTPException(404, "no active session — POST /api/simulation/start first")
    return s


# ------------------------------------------------------------- static info
@app.get("/api/topology")
def topology() -> dict:
    topo = get_topology()
    return {
        "routers": [vars(r) for r in topo.routers.values()],
        "links": [vars(l) for l in topo.link_defs.values()],
    }


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {
        name: {"description": s.description, "start_hour": s.start_hour,
               "duration_min": s.duration_min, "events": s.events,
               "randomized": s.randomize is not None}
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


# ------------------------------------------------------ simulation control
@app.post("/api/simulation/start")
async def sim_start(req: StartRequest) -> dict:
    old: SimSession | None = STATE["session"]
    if old is not None:
        old.pause()
    if req.scenario not in get_scenarios():
        raise HTTPException(400, f"unknown scenario {req.scenario}")
    try:
        session = SimSession(
            scenario=req.scenario, algorithms=req.algorithms, seed=req.seed,
            model_tag=req.model_tag, safety_filter=req.safety_filter, speed=req.speed,
        )
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    STATE["session"] = session
    session.subscribers.append(WS_HUB.send)
    if req.autostart:
        session.start()
    return session.status()


@app.post("/api/simulation/pause")
def sim_pause() -> dict:
    s = current_session()
    s.pause()
    return s.status()


@app.post("/api/simulation/resume")
def sim_resume() -> dict:
    s = current_session()
    s.start()
    return s.status()


@app.post("/api/simulation/step")
async def sim_step() -> dict:
    s = current_session()
    if s.running:
        raise HTTPException(409, "pause the simulation before stepping manually")
    if s.done:
        raise HTTPException(409, "scenario finished — reset to run again")
    payload = await asyncio.to_thread(s.step_once)
    await s._broadcast(payload)
    return payload


@app.post("/api/simulation/reset")
async def sim_reset() -> dict:
    s = current_session()
    s.pause()
    session = SimSession(scenario=s.scenario, algorithms=s.algorithms, seed=s.seed,
                         speed=s.speed)
    STATE["session"] = session
    session.subscribers.append(WS_HUB.send)
    return session.status()


@app.post("/api/simulation/speed")
def sim_speed(req: SpeedRequest) -> dict:
    s = current_session()
    try:
        s.set_speed(req.speed)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return s.status()


@app.get("/api/simulation/status")
def sim_status() -> dict:
    s: SimSession | None = STATE["session"]
    return s.status() if s else {"running": False, "session": None}


@app.get("/api/telemetry/current")
def telemetry_current() -> dict:
    return current_session().payload()


# ----------------------------------------------------------- interventions
@app.post("/api/failure/inject")
def failure_inject(req: FailureRequest) -> dict:
    s = current_session()
    try:
        s.inject_failure(req.link)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "failed_links": s.runners[0].eng.snapshot()["failed_links"]}


@app.post("/api/failure/recover")
def failure_recover(req: FailureRequest) -> dict:
    s = current_session()
    try:
        s.recover_link(req.link)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "failed_links": s.runners[0].eng.snapshot()["failed_links"]}


@app.post("/api/traffic/burst")
def traffic_burst(req: BurstRequest) -> dict:
    s = current_session()
    try:
        s.inject_burst(req.demand, req.factor, req.duration_min)
    except KeyError as e:
        raise HTTPException(400, f"unknown demand {req.demand}") from e
    return {"ok": True}


@app.post("/api/traffic/multiplier")
def traffic_multiplier(req: MultiplierRequest) -> dict:
    s = current_session()
    s.set_multiplier(req.factor)
    return {"ok": True, "factor": req.factor}


# ----------------------------------------------------------------- metrics
@app.get("/api/metrics/history")
def metrics_history() -> dict:
    s = current_session()
    return {
        "runs": [
            {"algorithm": r.algorithm, "history": r.eng.metrics_history}
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
@app.get("/api/export/results")
def export_results(fmt: str = "csv") -> Response:
    s = current_session()
    rows: list[dict] = []
    for r in s.runners:
        for h in r.eng.metrics_history:
            row = {k: v for k, v in h.items() if k != "failed_links"}
            row["n_failed_links"] = len(h["failed_links"])
            row["algorithm"] = r.algorithm
            row["scenario"] = s.scenario
            row["seed"] = s.seed
            rows.append(row)
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
    import pandas as pd
    s = current_session()
    ids = []
    for r in s.runners:
        hist = [
            {**{k: v for k, v in h.items() if k != "failed_links"},
             "n_failed_links": len(h["failed_links"]),
             "reward": 0.0}
            for h in r.eng.metrics_history
        ]
        if not hist:
            continue
        df = pd.DataFrame(hist)
        df["reward"] = 0.0  # live sessions track reward on the decision stream
        summary = summarize_records(df, r.algorithm, s.scenario, s.seed, engine=r.eng)
        summary["cumulative_reward"] = r.cumulative_reward
        ids.append(db.save_run("live", s.scenario, r.algorithm, s.seed, summary))
    return {"saved_run_ids": ids}


@app.get("/api/runs")
def runs(limit: int = 50) -> list[dict]:
    return db.list_runs(limit)


# ---------------------------------------------------------------- training
@app.post("/api/agent/train")
def train_start(req: TrainRequest) -> dict:
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
    return {"started": True, "tag": req.tag, "timesteps": req.timesteps}


@app.get("/api/training/progress")
def training_progress() -> dict:
    job = STATE.get("training")
    if not job:
        return {"active": False, "log": []}
    running = job["proc"].poll() is None
    return {
        "active": running,
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
    s: SimSession | None = STATE["session"]
    if s is not None:
        await ws.send_text(json.dumps(s.payload()))
    try:
        while True:
            await ws.receive_text()  # keepalive pings from the client
    except WebSocketDisconnect:
        if ws in WS_HUB.sockets:
            WS_HUB.sockets.remove(ws)


# ---------------------------------------------------------------- frontend
@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
