"""End-to-end API test: start a session, step it, inject a failure, export results."""

import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_full_session_lifecycle(client: TestClient):
    # static info endpoints
    topo = client.get("/api/topology").json()
    assert len(topo["routers"]) == 18 and len(topo["links"]) == 32
    assert "demo_evening" in client.get("/api/scenarios").json()
    assert "voice" in client.get("/api/traffic-classes").json()["classes"]

    # start a paused comparison session with two baselines (no model needed)
    r = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["static", "greedy"],
        "seed": 7, "autostart": False, "model_tag": None,
    })
    assert r.status_code == 200, r.text
    assert r.json()["algorithms"] == ["static", "greedy"]

    # manual stepping
    for _ in range(3):
        r = client.post("/api/simulation/step")
        assert r.status_code == 200
        payload = r.json()
        assert payload["type"] == "tick" and len(payload["runs"]) == 2
        snap = payload["runs"][0]["snapshot"]
        assert snap["metrics"]["offered_mbps"] > 0

    # paired manual failure injection
    r = client.post("/api/failure/inject", json={"link": "L20"})
    assert r.status_code == 200 and "L20" in r.json()["failed_links"]
    client.post("/api/simulation/step")
    links = client.get("/api/links").json()
    for run in links["runs"]:
        l20 = [l for l in run["links"] if l["link"] == "L20"]
        assert all(not l["up"] for l in l20)
    r = client.post("/api/failure/recover", json={"link": "L20"})
    assert r.status_code == 200 and r.json()["failed_links"] == []

    # burst + multiplier
    assert client.post("/api/traffic/burst",
                       json={"demand": "D5", "factor": 2.0, "duration_min": 30}).status_code == 200
    assert client.post("/api/traffic/multiplier", json={"factor": 1.2}).status_code == 200

    # history, lsps, agent status
    hist = client.get("/api/metrics/history").json()
    assert len(hist["runs"][0]["history"]) >= 4
    lsps = client.get("/api/lsps").json()
    assert len(lsps["runs"][0]["demands"]) == 17
    status = client.get("/api/agent/status").json()
    assert status["runs"][0]["algorithm"] == "static"

    # exports
    csv_resp = client.get("/api/export/results?fmt=csv")
    assert csv_resp.status_code == 200 and "max_util" in csv_resp.text.splitlines()[0]
    json_resp = client.get("/api/export/results?fmt=json")
    assert json_resp.status_code == 200 and isinstance(json_resp.json(), list)
    saved = client.post("/api/export/save-run").json()
    assert len(saved["saved_run_ids"]) == 2
    assert client.get("/api/runs").json()[0]["scenario"] == "link_failure"


def test_error_paths(client: TestClient):
    r = client.post("/api/simulation/start", json={
        "scenario": "nope", "algorithms": ["static"], "seed": 1})
    assert r.status_code == 400
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["static"], "seed": 1, "autostart": False})
    assert client.post("/api/failure/inject", json={"link": "L99"}).status_code == 400
    assert client.post("/api/simulation/speed", json={"speed": "warp"}).status_code == 400
    assert client.post("/api/simulation/speed", json={"speed": "5x"}).status_code == 200


def test_websocket_receives_ticks(client: TestClient):
    client.post("/api/simulation/start", json={
        "scenario": "evening_peak", "algorithms": ["greedy"], "seed": 3,
        "autostart": False})
    with client.websocket_connect("/ws/telemetry") as ws:
        first = ws.receive_json()  # initial snapshot on connect
        assert first["type"] == "tick"
        client.post("/api/simulation/step")
        msg = ws.receive_json()
        assert msg["type"] == "tick"
        assert msg["runs"][0]["snapshot"]["step"] >= 1
