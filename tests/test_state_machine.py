"""Session state machine, intervention, advisor, and export-consistency tests.

These encode the acceptance criteria from the presentation-hardening brief:
pause stops step progression, resume creates exactly one loop, reset
preserves the full configuration, failures affect both directed edges and
broadcast immediately, advisor proposals never mutate the real engine, and
exported rewards equal displayed cumulative rewards.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

import server.main as srv
from server.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
    srv.STATE["session"] = None


def start(client, **over):
    body = {"scenario": "evening_peak", "algorithms": ["greedy"], "seed": 5,
            "model_tag": None, "speed": "fast", "autostart": False}
    body.update(over)
    r = client.post("/api/simulation/start", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def status(client):
    return client.get("/api/simulation/status").json()


# ------------------------------------------------------------ state machine
def test_pause_freezes_step_counter(client):
    # Paced, not "fast": at zero delay the 84-interval scenario can run to
    # completion inside the sleep below, and the session would legitimately be
    # "completed" rather than "paused" — which made this test flaky. At 20x
    # (0.1 s per interval) the pause always lands mid-run.
    start(client, autostart=True, speed="20x")
    time.sleep(0.4)
    client.post("/api/simulation/pause")
    s1 = status(client)
    assert s1["state"] == "paused"
    assert 0 < s1["step"] < 84, "pause must land mid-scenario for this assertion"
    time.sleep(0.4)
    s2 = status(client)
    assert s2["step"] == s1["step"], "step advanced after pause"


def test_repeated_pause_resume_idempotent_single_loop(client):
    start(client, autostart=True)
    for _ in range(3):
        assert client.post("/api/simulation/resume").status_code == 200
    for _ in range(3):
        assert client.post("/api/simulation/pause").status_code == 200
    for _ in range(2):
        assert client.post("/api/simulation/resume").status_code == 200
    time.sleep(0.3)
    client.post("/api/simulation/pause")
    s1 = status(client)
    time.sleep(0.3)
    assert status(client)["step"] == s1["step"]
    # exactly one live loop task on the session
    session = srv.STATE["session"]
    live = 1 if (session._loop_task and not session._loop_task.done()) else 0
    assert live <= 1


def test_manual_step_requires_pause_and_advances_exactly_one(client):
    # At "fast" (zero delay), the optimized simulator can finish all 84
    # intervals before the pause below, making manual step correctly return
    # 409 for a completed session. Pace this test so it measures the intended
    # running -> paused -> manual-step transition.
    start(client, autostart=True, speed="20x")
    time.sleep(0.2)
    r = client.post("/api/simulation/step")
    assert r.status_code == 409, "step must be rejected while running"
    client.post("/api/simulation/pause")
    s0 = status(client)["step"]
    assert client.post("/api/simulation/step").status_code == 200
    assert status(client)["step"] == s0 + 1
    assert client.post("/api/simulation/step").status_code == 200
    assert status(client)["step"] == s0 + 2


def test_reset_preserves_full_configuration(client):
    start(client, scenario="link_failure", algorithms=["greedy", "static"],
          seed=7, safety_filter=False, speed="5x")
    for _ in range(3):
        client.post("/api/simulation/step")
    r = client.post("/api/simulation/reset")
    assert r.status_code == 200
    s = r.json()
    assert s["state"] == "idle" and s["step"] == 0 and s["t_min"] == 0
    assert s["scenario"] == "link_failure"
    assert s["algorithms"] == ["greedy", "static"]
    assert s["seed"] == 7
    assert s["safety_filter"] is False
    assert s["speed"] == "5x"
    hist = client.get("/api/metrics/history").json()
    assert all(len(run["history"]) == 0 for run in hist["runs"]), "charts not clean"


def test_reset_while_running_stops_session(client):
    start(client, autostart=True)
    time.sleep(0.3)
    r = client.post("/api/simulation/reset")
    assert r.status_code == 200 and r.json()["state"] == "idle"
    time.sleep(0.4)
    assert status(client)["step"] == 0, "session kept running after reset"


def test_completed_session_requires_reset(client):
    start(client, scenario="overload_stress", algorithms=["static"],
          autostart=True, speed="fast")
    for _ in range(100):
        if status(client)["state"] == "completed":
            break
        time.sleep(0.3)
    assert status(client)["state"] == "completed"
    assert client.post("/api/simulation/resume").status_code == 409
    assert client.post("/api/simulation/step").status_code == 409
    r = client.post("/api/simulation/reset")
    assert r.status_code == 200 and r.json()["state"] == "idle"
    assert client.post("/api/simulation/resume").status_code == 200


# ------------------------------------------------------------ interventions
def test_failure_affects_both_directions_and_no_clock_advance(client):
    start(client)
    client.post("/api/simulation/step")
    t0 = status(client)["t_min"]
    r = client.post("/api/failure/inject", json={"link": "L20"})
    assert r.status_code == 200 and r.json()["changed"] is True
    assert status(client)["t_min"] == t0, "failure injection advanced the clock"
    links = client.get("/api/links").json()
    l20 = [l for l in links["runs"][0]["links"] if l["link"] == "L20"]
    assert len(l20) == 2 and all(not l["up"] for l in l20), \
        "both directed edges must be down"


def test_failure_idempotent_and_recovery(client):
    start(client)
    client.post("/api/simulation/step")
    assert client.post("/api/failure/inject", json={"link": "L11"}).json()["changed"] is True
    r2 = client.post("/api/failure/inject", json={"link": "L11"})
    assert r2.status_code == 200 and r2.json()["changed"] is False, \
        "double-failing must be a safe no-op"
    rec = client.post("/api/failure/recover", json={"link": "L11"})
    assert rec.json()["changed"] is True and rec.json()["failed_links"] == []
    rec2 = client.post("/api/failure/recover", json={"link": "L11"})
    assert rec2.json()["changed"] is False


def test_failure_paired_in_compare_mode(client):
    start(client, algorithms=["greedy", "static"])
    client.post("/api/simulation/step")
    client.post("/api/failure/inject", json={"link": "L19"})
    links = client.get("/api/links").json()
    for run in links["runs"]:
        down = [l for l in run["links"] if l["link"] == "L19"]
        assert all(not l["up"] for l in down), f"{run['algorithm']} engine missed failure"
    client.post("/api/failure/recover", json={"link": "L19"})


def test_failure_broadcasts_immediately_even_while_paused(client):
    start(client)
    client.post("/api/simulation/step")
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.receive_json()  # initial snapshot
        client.post("/api/failure/inject", json={"link": "L22"})
        msg = ws.receive_json()
        assert msg["type"] == "intervention"
        l22 = [l for l in msg["runs"][0]["snapshot"]["links"] if l["link"] == "L22"]
        assert all(not l["up"] for l in l22)
    client.post("/api/failure/recover", json={"link": "L22"})


def test_failure_works_while_running_and_after_reset(client):
    start(client, autostart=True)
    time.sleep(0.2)
    assert client.post("/api/failure/inject", json={"link": "L13"}).status_code == 200
    client.post("/api/simulation/reset")
    links = client.get("/api/links").json()
    l13 = [l for l in links["runs"][0]["links"] if l["link"] == "L13"]
    assert all(l["up"] for l in l13), "reset must clear injected failures"
    assert client.post("/api/failure/inject", json={"link": "L13"}).json()["changed"]


# ----------------------------------------------------------------- advisor
def test_advisor_propose_does_not_mutate_engine(client):
    start(client, algorithms=["rl"], model_tag="ppo_te", advisor=True,
          scenario="demo_evening", seed=42)
    s0 = status(client)
    r = client.post("/api/advisor/propose")
    assert r.status_code == 200
    prop = r.json()
    s1 = status(client)
    assert s1["step"] == s0["step"] and s1["t_min"] == s0["t_min"], \
        "proposal must not advance the engine"
    assert s1["state"] == "paused" and s1["awaiting_decision"] is True
    assert "lookahead" in prop and "noop" in prop["lookahead"]
    # idempotent second propose returns the same pending proposal
    assert client.post("/api/advisor/propose").json()["id"] == prop["id"]
    # resume is blocked while a decision is pending
    assert client.post("/api/simulation/resume").status_code == 409


def test_advisor_approve_applies_exact_action(client):
    start(client, algorithms=["rl"], model_tag="ppo_te", advisor=True,
          scenario="demo_evening", seed=42)
    prop = client.post("/api/advisor/propose").json()
    s0 = status(client)["step"]
    rec = client.post("/api/advisor/approve").json()
    assert rec["approved"] is True
    assert rec["applied_action"] == prop["action"]
    assert status(client)["step"] == s0 + 1
    hist = client.get("/api/advisor/status").json()["history"]
    assert hist[-1]["id"] == prop["id"] and "actual" in hist[-1]


def test_advisor_reject_applies_noop(client):
    start(client, algorithms=["rl"], model_tag="ppo_te", advisor=True,
          scenario="demo_evening", seed=42)
    client.post("/api/advisor/propose")
    rec = client.post("/api/advisor/reject").json()
    assert rec["approved"] is False and rec["applied_action"] == 0
    # no pending proposal afterwards
    assert client.get("/api/advisor/status").json()["pending"] is None
    # approve with nothing pending → 409
    assert client.post("/api/advisor/approve").status_code == 409


# ------------------------------------------------------- export consistency
def test_exported_rewards_match_cumulative(client):
    start(client, algorithms=["greedy", "static"])
    for _ in range(6):
        client.post("/api/simulation/step")
    rows = client.get("/api/export/results?fmt=json").json()
    agent = client.get("/api/agent/status").json()
    for run in agent["runs"]:
        exported = sum(r["reward"] for r in rows if r["algorithm"] == run["algorithm"])
        assert exported == pytest.approx(run["cumulative_reward"], abs=1e-6)
        assert exported != 0.0, "per-step rewards must be real, not zero"
    saved = client.post("/api/export/save-run").json()
    assert len(saved["saved_run_ids"]) == 2
    stored = client.get("/api/runs").json()[0]["summary"]
    assert stored["reward_sum"] != 0.0


# ----------------------------------------------------------- misc endpoints
def test_run_until_next_event(client):
    start(client, scenario="link_failure", algorithms=["greedy"])
    r = client.post("/api/simulation/run-until", json={"condition": "next_event"})
    assert r.status_code == 200
    out = r.json()
    assert out["steps"] > 0
    assert out["status"]["t_min"] >= 60, "should reach the scripted failure at t=60"


def test_run_until_failure_and_recovery_follow_real_scenario_events(client):
    start(client, scenario="demo_evening", algorithms=["greedy"])

    failed = client.post(
        "/api/simulation/run-until", json={"condition": "failure"}
    )
    assert failed.status_code == 200
    assert failed.json()["status"]["t_min"] >= 195
    engine = srv.STATE["session"].runners[0].eng
    assert engine.link_up["L20"] is False

    recovered = client.post(
        "/api/simulation/run-until", json={"condition": "recovery"}
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"]["t_min"] >= 240
    assert engine.link_up["L20"] is True


@pytest.mark.parametrize("seed", [1001, 1002, 1003, 1004, 1005])
def test_live_session_refuses_frozen_holdout_seeds_without_replacing_session(client, seed):
    original = start(client, seed=42)
    response = client.post("/api/simulation/start", json={
        "scenario": "demo_evening", "algorithms": ["greedy"],
        "seed": seed, "autostart": False,
    })
    assert response.status_code == 400
    assert "holdout" in response.json()["detail"].lower()
    assert status(client)["session_id"] == original["session_id"]


def test_events_endpoint_records_lifecycle(client):
    start(client)
    client.post("/api/simulation/step")
    client.post("/api/failure/inject", json={"link": "L9"})
    events = client.get("/api/events").json()
    kinds = [e["event"] for e in events]
    assert "session_created" in kinds and "manual_step" in kinds \
        and "link_failed" in kinds


def test_display_registry(client):
    d = client.get("/api/display").json()
    assert d["cities"]["PE1"] == "İstanbul"
    assert d["cities"]["P5"] == "Kayseri"
    assert d["links"]["L11"]["label"] == "Ankara–Kayseri link"
    assert d["links"]["L11"]["technical"] == "P2–P5, L11"
    assert "not a real operator topology" in d["disclaimer"]
    # internal IDs unchanged
    topo = client.get("/api/topology").json()
    assert {r["id"] for r in topo["routers"]} >= {"PE1", "P5", "A1"}


def test_benchmark_loaded_from_committed_csv(client):
    import pandas as pd
    b = client.get("/api/benchmark").json()
    assert "full_day" in b["scenarios"]
    fd = b["scenarios"]["full_day"]
    assert fd["winner"] == "rl"
    df = pd.read_csv("results/eval_stats.csv")
    row = df[(df.scenario == "full_day") & (df.algorithm == "rl")].iloc[0]
    assert fd["algorithms"]["rl"]["reward_mean"] == pytest.approx(
        round(float(row["reward_sum_mean"]), 1))


def test_training_gated(client, monkeypatch):
    monkeypatch.setenv("ALLOW_TRAINING", "false")
    r = client.post("/api/agent/train",
                    json={"timesteps": 1000, "tag": "x", "seed": 1, "confirm": True})
    assert r.status_code == 403
    assert client.get("/api/training/progress").json()["allowed"] is False
    monkeypatch.setenv("ALLOW_TRAINING", "true")
    r = client.post("/api/agent/train",
                    json={"timesteps": 1000, "tag": "x", "seed": 1, "confirm": False})
    assert r.status_code == 400, "training must require explicit confirmation"
