"""The cross-mode results surface, and the merge it must never perform.

Three record classes appear in this product: a live demonstration, a retained
demonstration, and the closed study's governed evidence. The failure mode these
tests exist to prevent is a single number that mixes them — an "average return
across all runs", a leaderboard with the study in it, a copy of a frozen holdout
figure living in a second renderer where it can drift.

They also pin the retention decision recorded in docs/ADR-003: reset run keeps a
run in the session, full reset keeps it in the process, and a restart keeps
nothing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.product import results
from server.main import STATE, app

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "frontend" / "js" / "product" / "results.js"


@pytest.fixture(autouse=True)
def clean_process_store():
    results.clear_process_retained()
    yield
    results.clear_process_retained()
    STATE["session"] = None


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def live(client):
    response = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["static", "greedy"],
        "seed": 42, "autostart": False, "environment": "v1", "model_tag": None})
    assert response.status_code == 200
    for _ in range(3):
        client.post("/api/simulation/step")
    yield client


# ============================================================== three classes
def test_the_payload_has_exactly_three_named_record_classes(client):
    body = client.get("/api/product/results").json()
    assert set(body["record_classes"]) == {
        "live_demonstration", "retained_demonstration", "governed_evidence"}
    assert set(body) >= {"live", "retained", "study", "separation_rule"}


def test_only_the_study_class_is_marked_as_evidence(live):
    body = live.get("/api/product/results").json()
    assert body["live"]["is_evidence"] is False
    assert body["retained"]["is_evidence"] is False
    assert body["study"]["is_evidence"] is True
    for run in body["live"]["runs"]:
        assert run["is_evidence"] is False
        assert run["evidence_reason"]


def test_the_payload_refuses_to_be_comparable_across_classes(client):
    body = client.get("/api/product/results").json()
    assert body["comparable"] is False
    assert "never averaged" in body["separation_rule"]


def test_the_study_numbers_are_never_loaded_into_this_module(client):
    body = client.get("/api/product/results").json()
    study = body["study"]
    assert study["loaded_here"] is False
    assert study["read_from"], "the surface must point at the governed routes"
    # No frozen figure is transcribed anywhere in the module. A literal here
    # would be a second copy of the study's record, free to drift from it.
    source = inspect.getsource(results)
    for frozen_number in ("18.221", "9.036", "-2.327", "1.107"):
        assert frozen_number not in source


def test_no_function_in_the_module_can_aggregate_across_classes():
    """There is no code path that takes two record classes and returns one row."""
    source = inspect.getsource(results)
    for forbidden in ("def combined", "def overall", "def leaderboard",
                      "def all_runs", "def rank"):
        assert forbidden not in source, forbidden
    # `run_row` takes exactly one record class and stamps it onto the row.
    signature = inspect.signature(results.run_row)
    assert "record_class" in signature.parameters


# ================================================================== live rows
def test_a_live_row_is_derived_from_the_runners_own_history(live):
    body = live.get("/api/product/results").json()
    section = body["live"]
    assert section["available"] is True
    assert section["steps"] == 3
    assert {run["algorithm"] for run in section["runs"]} == {"static", "greedy"}
    for run in section["runs"]:
        assert run["steps"] == 3
        assert run["record_class"] == "live_demonstration"
        assert run["return_unit"] == "signed operational return"


def test_a_live_row_reports_movement_counters_separately(live):
    body = live.get("/api/product/results").json()
    for run in body["live"]["runs"]:
        assert "movement" in run
        assert "total_movement" not in run


def test_the_results_route_answers_without_a_session(client):
    """A full reset is exactly when an operator wants to read what was kept."""
    response = client.get("/api/product/results")
    assert response.status_code == 200
    assert response.json()["live"]["available"] is False


# ================================================================= retention
def test_reset_run_archives_the_replaced_run_into_the_session(live):
    live.post("/api/simulation/reset")
    body = live.get("/api/simulation/retained-runs").json()
    assert body["session_count"] == 1
    assert body["process_count"] == 0
    archive = body["session_runs"][0]
    assert archive["scenario"] == "link_failure"
    assert {run["algorithm"] for run in archive["runs"]} == {"static", "greedy"}


def test_full_reset_hands_the_archive_and_the_live_run_to_the_process(live):
    live.post("/api/simulation/reset")     # one archived run
    for _ in range(2):
        live.post("/api/simulation/step")  # the replacement run has history too
    stopped = live.post("/api/simulation/stop").json()
    assert stopped["stopped"] is True
    assert stopped["handed_to_process_store"] == 2

    body = live.get("/api/simulation/retained-runs").json()
    assert body["session_count"] == 0
    assert body["process_count"] == 2
    assert body["count"] == 2


def test_retained_runs_are_never_promoted_to_evidence(live):
    live.post("/api/simulation/reset")
    body = live.get("/api/simulation/retained-runs").json()
    assert body["is_evidence"] is False
    for archive in body["runs"]:
        for run in archive["runs"]:
            assert run["record_class"] == "retained_demonstration"
            assert run["is_evidence"] is False


def test_nothing_persists_retained_runs_to_disk():
    """A restart drops everything, because nothing writes it down."""
    source = inspect.getsource(results)
    for forbidden in ("open(", "write_text", "to_csv", "json.dump", "Path("):
        assert forbidden not in source, forbidden


def test_retained_runs_report_their_own_lifetime(client):
    body = client.get("/api/simulation/retained-runs").json()
    assert "restart" in body["lifetime"]
    assert "never persisted" in body["lifetime"]


# =============================================================== the surface
def test_the_surface_renders_three_sections_and_no_combined_table():
    source = SURFACE.read_text(encoding="utf-8")
    for section in ("liveSection", "retainedSection", "studySection"):
        assert f"function {section}" in source
    for forbidden in ("combinedTable", "leaderboard", "allRunsTable"):
        assert forbidden not in source


def test_the_surface_does_not_render_a_frozen_study_number():
    source = SURFACE.read_text(encoding="utf-8")
    study = source.split("function studySection", 1)[1].split("\n}", 1)[0]
    for frozen_number in ("18.221", "9.036", "2.327", "1.107"):
        assert frozen_number not in study
    assert "governed-study.js" in source or "Governed Study" in study


def test_the_surface_never_formats_a_return_as_a_percentage():
    source = SURFACE.read_text(encoding="utf-8")
    cell = source.split("function cell(", 1)[1]
    assert 'unit === "return"' in cell
    assert cell.index('unit === "return"') < cell.index('unit === "share"')


def test_the_results_route_writes_nothing_under_results(live):
    before = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    live.get("/api/product/results")
    live.get("/api/simulation/retained-runs")
    assert {p for p in (ROOT / "results").rglob("*") if p.is_file()} == before
