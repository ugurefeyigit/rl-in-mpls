"""The paired decision comparison: what it shows, and what it refuses to show.

These tests are about the refusals. A comparison surface is easy to make look
good and easy to make lie, and the three lies available to it are:

1. rendering a verdict when the two lanes are not running the same experiment;
2. turning a signed operational return into a percentage difference;
3. summing controller TE changes with FRR protection moves so the policy gets
   credit for what protection did.

Every test below fails if one of those becomes possible.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.product import comparison
from server.main import STATE, app

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "frontend" / "js" / "product" / "comparison-lane.js"


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def paired(client):
    """A paused two-baseline V1 session: paired, fast, no checkpoint load."""
    response = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["static", "greedy"],
        "seed": 42, "autostart": False, "environment": "v1", "model_tag": None})
    assert response.status_code == 200
    for _ in range(3):
        client.post("/api/simulation/step")
    yield client


@pytest.fixture()
def solo(client):
    response = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["static"], "seed": 42,
        "autostart": False, "environment": "v1", "model_tag": None})
    assert response.status_code == 200
    client.post("/api/simulation/step")
    yield client


# ============================================================ shape and lanes
def test_a_matched_pair_reports_both_lanes_with_their_own_identity(paired):
    body = paired.get("/api/simulation/comparison").json()
    assert body["matched"] is True
    detail = body["detail"]
    assert detail["available"] is True
    assert [lane["token"] for lane in detail["lanes"]] == ["A", "B"]
    assert [lane["position"] for lane in detail["lanes"]] == ["a", "b"]
    for lane in detail["lanes"]:
        assert lane["algorithm"] in {"static", "greedy"}
        assert lane["family"] in {"learner", "baseline"}
        assert lane["steps_recorded"] == 3
        assert lane["action"]["available"] is True


def test_each_lane_reports_the_action_it_actually_took(paired):
    detail = paired.get("/api/simulation/comparison").json()["detail"]
    for lane in detail["lanes"]:
        action = lane["action"]
        assert action["kind"] in {"single_action", "baseline_moves"}
        assert isinstance(action["text"], str) and action["text"]


def test_metric_rows_carry_both_values_a_gap_and_a_direction(paired):
    rows = paired.get("/api/simulation/comparison").json()["detail"]["metric_rows"]
    assert rows, "a matched pair with completed intervals must compare metrics"
    for row in rows:
        assert row["gap"] == pytest.approx(row["a"] - row["b"], abs=1e-6)
        assert row["better"] in {"lower", "higher", None}
        # A metric with no better direction never names a leader.
        if row["better"] is None:
            assert row["leader"] is None
        assert row["leader"] in {"a", "b", None}


# ================================================================== refusals
def test_a_single_controller_session_is_not_a_failed_comparison(solo):
    body = solo.get("/api/simulation/comparison").json()
    assert body["comparison"] is False
    assert body["matched"] is None
    assert body["detail"]["available"] is False
    assert "nothing to compare" in body["reason"]
    assert "verdict" not in body["detail"]
    assert "metric_rows" not in body["detail"]


def test_an_unmatched_pair_shows_no_metric_no_gap_and_no_verdict():
    """A broken proof produces a refusal, not a caveated verdict."""

    class _Engine:
        step_count = 4
        metrics_history = [{"max_util": 0.9}]

    class _Runner:
        def __init__(self, algorithm, version):
            self.algorithm = algorithm
            self.environment_version = version
            self.eng = _Engine()
            self.cumulative_reward = 1.0
            self.history = []
            self.last_decision = None
            self.checkpoint = None

    class _Config:
        scenario = "link_failure"
        seed = 42
        environment = "v2"
        training_root = 42

    class _Session:
        config = _Config()
        runners = [_Runner("masked_bandit", "v2"), _Runner("rl", "v1")]

    state = comparison.comparison_state(_Session())
    assert state["matched"] is False
    detail = state["detail"]
    assert detail["available"] is False
    assert detail["mismatched_fields"] == ["environment_version"]
    for forbidden in ("verdict", "metric_rows", "divergence"):
        assert forbidden not in detail
    for lane in detail["lanes"]:
        for forbidden in ("metrics", "movement", "reward_components"):
            assert forbidden not in lane


# ================================================================== verdict
def test_the_verdict_reports_a_signed_gap_and_never_a_percentage(paired):
    verdict = paired.get("/api/simulation/comparison").json()["detail"]["verdict"]
    assert verdict["percentage"] is None
    assert "percentage" in verdict["percentage_reason"].lower()
    assert verdict["unit"] == "signed operational return"
    assert verdict["gap"] == pytest.approx(verdict["a"] - verdict["b"], abs=1e-6)


def test_the_verdict_never_claims_to_be_evidence(paired):
    verdict = paired.get("/api/simulation/comparison").json()["detail"]["verdict"]
    assert verdict["is_evidence"] is False
    assert "holdout" in verdict["evidence_reason"]


def test_no_module_level_percentage_of_a_return_can_be_computed():
    """There is no division of one cumulative return by another, anywhere."""
    source = inspect.getsource(comparison)
    assert "cumulative_reward /" not in source
    assert "a_total /" not in source and "b_total /" not in source
    # The gap is a subtraction. If a ratio ever appears here, this assertion is
    # the thing that has to be argued with first.
    assert "gap = round(a_total - b_total, 4)" in source


# ================================================================= movement
def test_controller_protection_and_recovery_counters_stay_separate():
    attributions = {attribution for _, _, attribution in comparison.MOVEMENT_COUNTERS}
    assert attributions == {"controller", "protection", "recovery"}
    keys = [key for key, _, _ in comparison.MOVEMENT_COUNTERS]
    # No counter is a sum of the others.
    assert "reroutes_total" not in keys
    assert "total_movement" not in keys


def test_movement_is_reported_per_counter_not_as_one_number(paired):
    detail = paired.get("/api/simulation/comparison").json()["detail"]
    for lane in detail["lanes"]:
        for key, row in lane["movement"].items():
            assert row["attribution"] in {"controller", "protection", "recovery"}
            assert isinstance(row["total"], int)


# =============================================================== divergence
def test_divergence_is_measured_from_recorded_history_or_declines(paired):
    divergence = paired.get("/api/simulation/comparison").json()["detail"]["divergence"]
    assert set(divergence) >= {"available"}
    if divergence["available"]:
        assert isinstance(divergence["step"], int)
        assert divergence["a_moved"] != divergence["b_moved"]
    else:
        assert divergence["reason"]


# ================================================================ the surface
def test_the_lane_renders_nothing_comparative_while_the_proof_is_broken():
    source = LANE.read_text(encoding="utf-8")
    broken = source.split("if (!comparison.matched)", 1)[1].split("\n  }", 1)[0]
    for forbidden in ("verdictBlock", "metricTable", "movementTable",
                      "divergenceBlock"):
        assert forbidden not in broken, forbidden


def test_the_lane_never_formats_a_return_as_a_percentage():
    source = LANE.read_text(encoding="utf-8")
    verdict = source.split("function verdictBlock", 1)[1].split("\n}", 1)[0]
    assert "percent(" not in verdict
    assert "signed(" in verdict


def test_lanes_are_distinguishable_without_colour():
    """A letter token and a border style, not colour alone."""
    source = LANE.read_text(encoding="utf-8")
    assert 'text: lane.token' in source
    css = (ROOT / "frontend" / "css" / "presentation-mode.css").read_text(
        encoding="utf-8")
    assert 'border-left-style: dashed' in css
    assert '.cmp__lane[data-lane="a"]' in css and '.cmp__lane[data-lane="b"]' in css


def test_the_comparison_route_never_writes_a_governed_path(paired):
    before = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    paired.get("/api/simulation/comparison")
    paired.get("/api/simulation/moment")
    assert {p for p in (ROOT / "results").rglob("*") if p.is_file()} == before


def teardown_module(module):  # noqa: ARG001 - pytest hook
    STATE["session"] = None
