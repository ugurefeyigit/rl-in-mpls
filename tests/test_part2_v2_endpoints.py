"""Endpoints Part 1 left unverified under V2, and the delegated fast-forward.

Two of these were open questions in the Part 1 handoff:

- `/api/export/save-run` was written against V1's interval record. It reads
  `jain_fairness`, `p95_delay_ms`, `priority_sla_success`, `carried_mbps`,
  `reroutes`, `flaps`, `frr_events` and `engine.path_change_count`, and the
  frozen V2 record has none of them. It raised under V2. It now has a V2
  summarizer over V2's own columns, and the missing quantities are declared
  absent rather than padded with zeros.
- `/api/lsps` reads `snapshot()["demands"]`, which `EngineV2View` supplies. It
  works, and this file pins that so a future change to the view cannot break it
  silently.

The third group covers the asymmetry the handoff asked Part 2 to decide: a
fast-forward under advisor execution now requires explicit delegation and enters
the approval ledger as one delegated batch (docs/ADR-003).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mplssim.product import run_summary
from server.main import STATE, app
from server.session import SessionConfig, SimSession

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
    STATE["session"] = None


@pytest.fixture()
def v2_live(client):
    """A paused V2 session driven by baselines only, so no checkpoint is needed."""
    response = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v2",
        "algorithms": ["greedy", "static"], "seed": 42, "autostart": False,
        "model_tag": None})
    assert response.status_code == 200, response.text
    for _ in range(3):
        client.post("/api/simulation/step")
    yield client


@pytest.fixture()
def v1_live(client):
    response = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v1",
        "algorithms": ["greedy"], "seed": 42, "autostart": False,
        "model_tag": None})
    assert response.status_code == 200
    for _ in range(3):
        client.post("/api/simulation/step")
    yield client


# ================================================================== save-run
def test_save_run_succeeds_under_v2(v2_live):
    response = v2_live.post("/api/export/save-run")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["environment"] == "v2"
    assert len(body["saved_run_ids"]) == 2


def test_a_saved_v2_row_uses_v2_columns_and_declares_what_it_cannot_measure(v2_live):
    v2_live.post("/api/export/save-run")
    runs = v2_live.get("/api/runs").json()
    summary = runs[0]["summary"] if isinstance(runs[0].get("summary"), dict) else None
    if summary is None:                       # stored as JSON text
        import json
        summary = json.loads(runs[0]["summary"])
    assert summary["environment_version"] == "v2"
    assert "accepted_te_changes_total" in summary
    assert "frr_changes_total" in summary
    # V1-only quantities are absent with a reason, never zero-padded.
    assert "jain_fairness_mean" not in summary
    assert "jain_fairness" in summary["not_measured"]
    assert "padded" in summary["not_measured_reason"]


def test_a_saved_run_is_labelled_a_demonstration_not_evidence(v2_live):
    v2_live.post("/api/export/save-run")
    import json
    runs = v2_live.get("/api/runs").json()
    summary = runs[0]["summary"]
    summary = summary if isinstance(summary, dict) else json.loads(summary)
    assert summary["record_class"] == "live_demonstration"
    assert summary["is_evidence"] is False


def test_save_run_still_uses_the_v1_summarizer_under_v1(v1_live):
    v1_live.post("/api/export/save-run")
    import json
    runs = v1_live.get("/api/runs").json()
    summary = runs[0]["summary"]
    summary = summary if isinstance(summary, dict) else json.loads(summary)
    assert summary["environment_version"] == "v1"
    assert "jain_fairness_mean" in summary


def test_save_run_refuses_before_any_interval_completes(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v2",
        "algorithms": ["greedy"], "seed": 42, "autostart": False,
        "model_tag": None})
    response = client.post("/api/export/save-run")
    assert response.status_code == 409
    assert "nothing to save" in response.json()["detail"]


def test_the_v2_summarizer_never_invents_a_v1_column():
    frame = pd.DataFrame([{
        "step": 1, "t_min": 5.0, "reward": -0.5, "n_failed_links": 0,
        "max_util": 0.7, "mean_util": 0.3, "util_std": 0.1,
        "mean_delay_ms": 12.0, "max_delay_ms": 30.0, "loss_ratio": 0.0,
        "delivered_ratio": 1.0, "sla_violations": 0, "congested_links": 0,
        "disconnected_demands": 0, "accepted_te_changes": 1,
        "rejected_te_requests": 0, "te_reversals": 0, "frr_changes": 0,
        "frr_disconnections": 0, "recovery_restorations": 0,
    }])
    summary = run_summary.summarize_v2_records(frame, "greedy", "link_failure", 42)
    assert summary["environment_version"] == "v2"
    for v1_only in ("jain_fairness_mean", "p95_delay_ms",
                    "priority_sla_success_mean", "path_changes_per_demand"):
        assert v1_only not in summary


# ====================================================================== lsps
def test_lsps_answers_under_v2_with_the_engines_own_demands(v2_live):
    body = v2_live.get("/api/lsps").json()
    assert {run["algorithm"] for run in body["runs"]} == {"greedy", "static"}
    for run in body["runs"]:
        assert run["demands"], "a V2 snapshot must carry its demands"
        first = run["demands"][0]
        for field in ("id", "current_path", "sla_ok", "candidates"):
            assert field in first


def test_links_and_metrics_history_answer_under_v2(v2_live):
    links = v2_live.get("/api/links").json()
    assert links["runs"][0]["links"]
    history = v2_live.get("/api/metrics/history").json()
    assert len(history["runs"][0]["history"]) == 3


def test_export_results_answers_under_v2(v2_live):
    response = v2_live.get("/api/export/results?fmt=json")
    assert response.status_code == 200
    rows = response.json()
    assert rows and rows[0]["scenario"] == "link_failure"


# ======================================================= delegated fast-forward
def test_advisor_execution_refuses_an_undelegated_fast_forward(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v1", "algorithms": ["greedy"],
        "seed": 42, "autostart": False, "model_tag": None, "execution": "advisor"})
    response = client.post("/api/simulation/run-until",
                           json={"condition": "next_event"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "delegate=true" in detail
    assert "approved individually" in detail


def test_a_delegated_fast_forward_is_recorded_as_one_batch(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v1", "algorithms": ["rl"],
        "seed": 42, "autostart": False, "model_tag": "ppo_te",
        "execution": "advisor"})
    response = client.post("/api/simulation/run-until",
                           json={"condition": "next_event", "delegate": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["delegated"] is True
    assert body["steps"] >= 1
    assert "delegated batch" in body["note"]

    advisor = client.get("/api/advisor/status").json()
    batches = advisor["delegated_batches"]
    assert len(batches) == 1
    assert batches[0]["steps"] == body["steps"]
    assert batches[0]["delegated"] is True
    # A delegated batch is not a proposal, and never becomes one.
    assert advisor["proposals"] == []
    assert advisor["delegated_intervals"] == body["steps"]


def test_a_delegated_batch_is_its_own_timeline_event_not_a_recommendation(client):
    """Regression: the ledger holds two record shapes, and the timeline reads
    every record in it. A delegated batch has no proposal and no single
    interval, so forcing it into the recommendation shape raised KeyError and
    took the whole moment endpoint down with it."""
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v1", "algorithms": ["rl"],
        "seed": 42, "autostart": False, "model_tag": "ppo_te",
        "execution": "advisor"})
    client.post("/api/simulation/run-until",
                json={"condition": "next_event", "delegate": True})

    moment = client.get("/api/simulation/moment")
    assert moment.status_code == 200, moment.text
    events = moment.json()["timeline"]["events"]
    delegations = [e for e in events if e["kind"] == "delegation"]
    assert len(delegations) == 1
    event = delegations[0]
    assert event["delegated"] is True
    assert event["approved"] is None
    assert event["steps"] == event["to_step"] - event["from_step"]
    assert "delegated" in event["title"]
    # It never appears as a recommendation the operator answered.
    assert not [e for e in events if e["kind"] == "recommendation"]


def test_automatic_execution_still_fast_forwards_without_delegation(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "environment": "v1", "algorithms": ["greedy"],
        "seed": 42, "autostart": False, "model_tag": None,
        "execution": "automatic"})
    response = client.post("/api/simulation/run-until",
                           json={"condition": "next_event"})
    assert response.status_code == 200
    body = response.json()
    assert body["delegated"] is False
    assert body["approval_bypassed"] is False


def test_the_session_states_the_delegation_rule_in_one_place():
    assert "delegate=true" in SimSession.DELEGATION_REQUIRED
    config = SessionConfig(scenario="link_failure", algorithms=("greedy",),
                           seed=42, model_tag=None, safety_filter=True,
                           speed="1x", advisor=True, environment="v1")
    assert config.execution == "advisor"


def test_the_ui_asks_before_delegating_and_discloses_afterwards():
    main = (ROOT / "frontend" / "js" / "product" / "main.js").read_text(
        encoding="utf-8")
    forward = main.split("async function fastForward", 1)[1].split("\n}", 1)[0]
    assert "window.confirm" in forward
    assert "NOT approved individually" in forward
    panel = (ROOT / "frontend" / "js" / "product" / "control-panel.js").read_text(
        encoding="utf-8")
    assert "delegated_intervals" in panel
