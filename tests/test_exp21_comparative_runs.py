"""Exp 2.1 completed-run comparison lifecycle and analytical contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.evidence.identity import REWARD_COMPONENTS
from mplssim.product import results
from server.main import STATE, app


ROOT = Path(__file__).resolve().parents[1]


def _history(*, reward: float, seed_shift: float = 0.0) -> list[dict]:
    components_1 = {name: 0.0 for name in REWARD_COMPONENTS}
    components_1.update({"delivery": reward + 0.2, "max_util": -0.2})
    components_2 = {name: 0.0 for name in REWARD_COMPONENTS}
    components_2.update({"delivery": reward + 0.1, "sla_severity": -0.1})
    return [
        {
            "step": 1, "t_min": 5.0, "reward": reward,
            "components": components_1,
            "metrics": {
                "max_util": 0.82 + seed_shift, "mean_util": 0.43,
                "delivered_ratio": 0.96 - seed_shift,
                "sla_violations": 1, "loss_ratio": 0.01,
                "accepted_te_changes": 1, "rejected_te_requests": 0,
                "te_reversals": 0, "frr_changes": 0,
                "frr_disconnections": 0, "recovery_restorations": 0,
                "flaps": 0,
            },
            "n_failed_links": 0, "failed_links": [], "moved_mbps": 125.0,
            "decision": {
                "action": 1,
                "decoded": {"type": "te_request", "demand": "D1",
                            "path_idx": 0, "accepted": True},
            },
        },
        {
            "step": 2, "t_min": 10.0, "reward": reward / 2,
            "components": components_2,
            "metrics": {
                "max_util": 0.74 + seed_shift, "mean_util": 0.38,
                "delivered_ratio": 0.99 - seed_shift,
                "sla_violations": 0, "loss_ratio": 0.002,
                "accepted_te_changes": 0, "rejected_te_requests": 0,
                "te_reversals": 0, "frr_changes": 0,
                "frr_disconnections": 0, "recovery_restorations": 1,
                "flaps": 0,
            },
            "n_failed_links": 1, "failed_links": ["L11"], "moved_mbps": 0.0,
            "decision": {"action": 0, "decoded": {"type": "noop", "accepted": False}},
        },
    ]


def _archive(session_id: str, algorithm: str, *, seed: int = 42,
             reward: float = 1.0, root: int = 314159) -> dict:
    history = _history(reward=reward, seed_shift=0.01 if seed != 42 else 0.0)
    return {
        "session_id": session_id,
        "generation": 0,
        "completed": True,
        "environment": "v2",
        "scenario": "demo_evening",
        "seed": seed,
        "training_root": root,
        "steps": len(history),
        "runs": [{
            "algorithm": algorithm,
            "checkpoint_id": f"{algorithm}-root{root}-300000",
            "checkpoint_provenance": {
                "payload_sha256": f"hash-{algorithm}-{root}",
                "training_source_sha": "6a8a4068b98bf9a71dead6e547595b4bbd755689",
            },
            "output_semantics": "scores" if algorithm == "masked_bandit" else "none",
            "cumulative_reward": sum(row["reward"] for row in history),
            "history": history,
        }],
    }


class _Session:
    def __init__(self, archives: list[dict]):
        self.previous_runs = archives

    def archive(self):
        return None

    async def reset(self):
        return {"state": "idle", "generation": 1}


@pytest.fixture()
def client():
    previous = STATE["session"]
    results.clear_process_retained()
    if hasattr(results, "clear_comparison_runs"):
        results.clear_comparison_runs()
    with TestClient(app) as test_client:
        yield test_client
    STATE["session"] = previous
    results.clear_process_retained()
    if hasattr(results, "clear_comparison_runs"):
        results.clear_comparison_runs()


def _candidate_ids(client: TestClient) -> list[str]:
    response = client.get("/api/product/comparative-runs")
    assert response.status_code == 200
    return [row["run_id"] for row in response.json()["candidates"]]


def test_two_completed_runs_can_be_captured_replaced_swapped_and_cleared(client):
    STATE["session"] = _Session([
        _archive("session-a", "masked_bandit", reward=1.2),
        _archive("session-b", "greedy", reward=0.8),
        _archive("session-c", "cspf", reward=0.4),
    ])
    first, second, third = _candidate_ids(client)

    assert client.put("/api/product/comparative-runs/a", json={"run_id": first}).status_code == 200
    assert client.put("/api/product/comparative-runs/b", json={"run_id": second}).status_code == 200
    replaced = client.put("/api/product/comparative-runs/a", json={"run_id": third}).json()
    assert replaced["slots"]["a"]["run_id"] == third
    assert replaced["slots"]["b"]["run_id"] == second

    swapped = client.post("/api/product/comparative-runs/swap").json()
    assert swapped["slots"]["a"]["run_id"] == second
    assert swapped["slots"]["b"]["run_id"] == third

    cleared_a = client.delete("/api/product/comparative-runs/a").json()
    assert cleared_a["slots"]["a"] is None
    assert cleared_a["slots"]["b"]["run_id"] == third
    cleared_all = client.delete("/api/product/comparative-runs").json()
    assert cleared_all["slots"] == {"a": None, "b": None}


def test_slot_payload_keeps_real_identity_history_components_and_units(client):
    STATE["session"] = _Session([_archive("session-a", "masked_bandit", reward=1.2)])
    run_id = _candidate_ids(client)[0]
    payload = client.put(
        "/api/product/comparative-runs/a", json={"run_id": run_id}).json()
    run = payload["slots"]["a"]

    assert run["identity"] == {
        "environment": "v2", "scenario": "demo_evening", "seed": 42,
        "controller": "masked_bandit", "training_root": 314159,
        "checkpoint_id": "masked_bandit-root314159-300000",
        "checkpoint_sha256": "hash-masked_bandit-314159",
    }
    assert run["provenance"]["record_class"] == "retained_demonstration"
    assert run["provenance"]["state"] == "completed"
    assert run["series"]["reward"]["unit"] == "signed operational return"
    assert run["series"]["utilization"]["unit"] == "percent"
    assert run["series"]["sla_risk"]["unit"] == "violating demands per interval"
    assert run["series"]["moved_bandwidth"]["unit"] == "Mbps"
    assert len(run["series"]["reward"]["values"]) == 2
    assert run["series"]["moved_bandwidth"]["values"][0]["value"] == 125.0
    assert set(run["reward_components"]) == set(REWARD_COMPONENTS)
    assert any(event["kind"] == "action" for event in run["timeline"])
    assert any(event["kind"] == "failure" for event in run["timeline"])


def test_pairing_integrity_controls_paired_conclusions_and_metric_direction(client):
    STATE["session"] = _Session([
        _archive("session-a", "masked_bandit", reward=1.2),
        _archive("session-b", "greedy", reward=0.8),
    ])
    first, second = _candidate_ids(client)
    client.put("/api/product/comparative-runs/a", json={"run_id": first})
    paired = client.put("/api/product/comparative-runs/b", json={"run_id": second}).json()

    assert paired["pairing"]["synchronized"] is True
    assert paired["pairing"]["paired_conclusions"] is True
    directions = {row["id"]: row["direction"] for row in paired["headline"]}
    assert directions["operational_return"] == "higher"
    assert directions["delivery"] == "higher"
    for key in ("sla_risk", "peak_utilization", "reroutes", "flaps", "moved_bandwidth"):
        assert directions[key] == "lower"
    for row in paired["headline"]:
        assert "delta" in row and "unit" in row and "leader" in row

    results.clear_comparison_runs()
    STATE["session"] = _Session([
        _archive("session-a", "masked_bandit", reward=1.2),
        _archive("session-x", "greedy", seed=7, reward=0.8),
    ])
    first, second = _candidate_ids(client)
    client.put("/api/product/comparative-runs/a", json={"run_id": first})
    unpaired = client.put("/api/product/comparative-runs/b", json={"run_id": second}).json()
    assert unpaired["pairing"]["synchronized"] is False
    assert unpaired["pairing"]["paired_conclusions"] is False
    assert "seed" in unpaired["pairing"]["mismatched_fields"]


def test_unfinished_runs_never_become_candidates_and_reset_keeps_completed_slots(client):
    complete = _archive("session-a", "masked_bandit")
    unfinished = deepcopy(_archive("session-u", "greedy"))
    unfinished["completed"] = False
    STATE["session"] = _Session([complete, unfinished])
    candidates = client.get("/api/product/comparative-runs").json()["candidates"]
    assert [row["identity"]["controller"] for row in candidates] == ["masked_bandit"]

    client.put("/api/product/comparative-runs/a", json={"run_id": candidates[0]["run_id"]})
    assert client.post("/api/simulation/reset").status_code == 200
    assert client.get("/api/product/comparative-runs").json()["slots"]["a"] is not None


def test_full_reset_deletes_both_slots_even_without_an_active_session(client):
    STATE["session"] = _Session([_archive("session-a", "masked_bandit")])
    run_id = _candidate_ids(client)[0]
    client.put("/api/product/comparative-runs/a", json={"run_id": run_id})
    STATE["session"] = None

    assert client.post("/api/simulation/stop").status_code == 200
    assert client.get("/api/product/comparative-runs").json()["slots"] == {
        "a": None, "b": None,
    }


def test_comparative_run_api_never_writes_governed_paths(client):
    STATE["session"] = _Session([_archive("session-a", "masked_bandit")])
    before = {p: p.stat().st_mtime_ns for p in (ROOT / "results").rglob("*") if p.is_file()}
    run_id = _candidate_ids(client)[0]
    client.put("/api/product/comparative-runs/a", json={"run_id": run_id})
    client.get("/api/product/comparative-runs")
    client.delete("/api/product/comparative-runs")
    after = {p: p.stat().st_mtime_ns for p in (ROOT / "results").rglob("*") if p.is_file()}
    assert after == before


def test_compare_is_a_fourth_primary_mode_and_existing_routes_stay_compatible(client):
    contracts = client.get("/api/product/contracts").json()
    assert [mode["id"] for mode in contracts["modes"]] == [
        "presentation", "network", "rl", "compare",
    ]
    assert contracts["routes"]["/compare"]["mode"] == "compare"

    for path in ("/", "/advanced", "/present", "/study", "/compare"):
        response = client.get(path)
        assert response.status_code == 200
        assert 'id="mode-compare"' in response.text
        assert 'id="panel-compare"' in response.text
        assert "/static/css/comparison-mode.css" in response.text


def test_presentation_gateway_exposes_the_complete_ab_lifecycle_and_truth_labels(client):
    picker_path = ROOT / "frontend/js/product/comparison-picker.js"
    assert picker_path.is_file(), "the Presentation A/B gateway module is missing"
    picker = picker_path.read_text(encoding="utf-8")
    presentation = (ROOT / "frontend/js/product/modes/presentation.js").read_text(encoding="utf-8")
    adapter = (ROOT / "frontend/js/product/adapters/live-v1.js").read_text(encoding="utf-8")

    for label in ("Run A", "Run B", "Swap A/B", "Clear A", "Clear B", "Clear All",
                  "View Full Results", "Synchronization", "COMPLETED LIVE DEMONSTRATION"):
        assert label in picker
    STATE["session"] = _Session([
        _archive("session-a", "masked_bandit"), _archive("session-b", "greedy")])
    first, second = _candidate_ids(client)
    client.put("/api/product/comparative-runs/a", json={"run_id": first})
    payload = client.put("/api/product/comparative-runs/b", json={"run_id": second}).json()
    labels = {row["label"] for row in payload["headline"]}
    for metric in ("Operational return", "Mean delivery", "Peak SLA risk",
                   "Peak utilization", "Accepted TE changes", "Route flaps",
                   "Moved bandwidth"):
        assert metric in labels
    assert "renderComparisonPicker" in presentation
    for endpoint in ("comparative-runs", "comparative-runs/swap"):
        assert endpoint in adapter


def test_full_compare_surface_names_each_question_unit_and_accessible_alternative():
    charts_path = ROOT / "frontend/js/product/comparison-charts.js"
    assert charts_path.is_file(), "the authored SVG comparison instruments are missing"
    charts = charts_path.read_text(encoding="utf-8")
    surface = (ROOT / "frontend/js/product/modes/compare.js").read_text(encoding="utf-8")
    css = (ROOT / "frontend/css/comparison-mode.css").read_text(encoding="utf-8")

    for question in ("How did interval reward evolve?", "How did cumulative reward evolve?",
                     "Where did link pressure differ?",
                     "How much traffic was delivered?", "When did SLA risk appear?",
                     "When did controllers act and incidents occur?",
                     "Which reward terms created the return?"):
        assert question in surface
    for contract in ("Simulation time (minutes)", "table", "caption", "tabindex",
                     "aria-label", "keydown", "mouseenter", "focus", "selectedStep"):
        assert contract in charts
    for reference in ("70% pressure", "100% capacity", "Zero reward"):
        assert reference in charts + surface
    assert "Cumulative reward" in surface
    assert "Interval pairing unavailable" in surface
    assert "Reset view" in surface
    assert "Network Information" in surface and "RL Information" in surface
    assert "prefers-reduced-motion" in css
    assert "overflow-x: hidden" not in css
