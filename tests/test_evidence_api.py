"""The `/api/v2/*` surface serves frozen evidence and nothing else.

These tests treat the API as the product's honesty boundary: every number it emits
must reconcile with the committed files, development and holdout stages must stay
apart, and no route may exist that could train, evaluate or select anything.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mplssim.evidence import identity
from server.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ the study
def test_study_endpoint_states_the_closed_status(client):
    d = client.get("/api/v2/study").json()
    assert d["status"] == "closed"
    assert d["environment"] == identity.ENVIRONMENT
    assert d["observation_dim"] == 604
    assert d["action_count"] == 69
    assert d["training_roots"] == list(identity.TRAINING_ROOTS)
    assert d["holdout_seeds"] == list(identity.HOLDOUT_SEEDS)
    assert d["sources"]["evaluation"] == identity.EVALUATION_SOURCE_SHA


def test_study_endpoint_carries_both_halves_of_the_planning_claim(client):
    joined = " ".join(client.get("/api/v2/study").json()["conclusions"]).lower()
    assert "does not positively support" in joined
    assert "temporal planning" in joined
    assert "not evidence that planning is generally irrelevant" in joined


# ---------------------------------------------------------------- the holdout
def test_final_holdout_headline_numbers(client):
    d = client.get("/api/v2/final-holdout").json()
    assert d["stage"] == identity.STAGE_FINAL_HOLDOUT
    assert d["episodes"]["total"] == 315
    assert d["episodes"]["per_policy"] == 35
    assert d["episodes"]["ran_once"] is True
    c = d["comparison"]
    assert round(c["bandit_return"], 3) == 18.221
    assert round(c["ppo_return"], 3) == 9.036
    assert round(c["advantage"], 3) == 9.185
    assert c["roots_won"] == 3
    by_algo = {r["algorithm"]: r for r in d["aggregate"]}
    assert round(by_algo["greedy"]["operational_return_mean"], 3) == -2.327
    assert len(by_algo) == 5


def test_final_holdout_reports_root_grain_not_pooled_episodes(client):
    d = client.get("/api/v2/final-holdout").json()
    for algo in identity.LEARNER_ALGORITHMS:
        row = next(r for r in d["aggregate"] if r["algorithm"] == algo)
        assert row["root_count"] == 3
        assert row["episodes"] == 105
        assert row["episodes_per_root"] == 35
    for algo in identity.BASELINE_ALGORITHMS:
        row = next(r for r in d["aggregate"] if r["algorithm"] == algo)
        assert row["root_count"] == 1
        assert row["root_mean_std"] == 0.0


def test_scenarios_endpoint_reports_the_one_ppo_win(client):
    d = client.get("/api/v2/final-holdout/scenarios").json()
    assert d["stage"] == identity.STAGE_FINAL_HOLDOUT
    rows = d["scenarios"]
    assert len(rows) == 7
    ppo = [r for r in rows if r["winner"] == "maskable_ppo"]
    assert [r["scenario"] for r in ppo] == ["deceptive_local_optimum"]
    assert round(-ppo[0]["advantage"], 3) == 1.107
    assert sum(r["winner"] == "masked_bandit" for r in rows) == 6


def test_reward_components_endpoint_proves_the_exact_sum(client):
    d = client.get("/api/v2/final-holdout/reward-components").json()
    assert d["exact"] is True
    assert d["max_residual"] < 1e-9
    assert len(d["component_names"]) == 12
    assert len(d["rows"]) == 9


def test_actions_endpoint_exposes_both_noop_grains(client):
    d = client.get("/api/v2/final-holdout/actions").json()
    assert len(d["distribution"]) == 9 * 69
    n = d["noop"]
    assert round(n["pooled_step_share"]["masked_bandit"] * 100, 2) == 87.09
    assert round(n["episode_mean_share"]["masked_bandit"] * 100, 2) == 82.10
    assert n["pooled_grain"] and n["episode_grain"]


def test_integrity_endpoint_reports_all_checks_passed(client):
    d = client.get("/api/v2/final-holdout/integrity").json()
    assert d["all_checks_passed"] is True
    assert all(v == 0 for v in d["counters"].values())
    assert d["protected_disconnection_identical_across_methods"] is True


def test_provenance_endpoint_binds_six_checkpoints_to_approved_sources(client):
    d = client.get("/api/v2/final-holdout/provenance").json()
    assert len(d["checkpoints"]) == 6
    assert {c["evaluation_source_sha"] for c in d["checkpoints"]} == \
        {identity.EVALUATION_SOURCE_SHA}
    assert {c["training_source_sha"] for c in d["checkpoints"]} <= \
        set(identity.TRAINING_SOURCE_SHAS)
    r = d["runtime"]
    assert round(r["total_runner_wall_seconds"], 3) == 152.093
    assert round(r["checkpoint_wall_seconds_sum"], 3) == 115.213


# ------------------------------------------------------------ the development
def test_development_stage_is_labelled_and_carries_learning_curves(client):
    d = client.get("/api/v2/development/continuity").json()
    assert d["stage"] == identity.STAGE_DEVELOPMENT
    assert d["summary"]["holdout_accessed"] is False
    assert len(d["learning_curves"]["series"]) == 6
    assert "not" in d["learning_curves"]["caption"].lower()
    assert "holdout" in d["learning_curves"]["caption"].lower()


def test_seed42_pilot_is_development_stage(client):
    d = client.get("/api/v2/development/seed42").json()
    assert d["stage"] == identity.STAGE_DEVELOPMENT
    assert d["source_sha"] == identity.SEED42_SOURCE_SHA
    assert len(d["methods"]) == 5


def test_development_and_holdout_never_arrive_from_the_same_endpoint(client):
    fh = client.get("/api/v2/final-holdout").json()
    dev = client.get("/api/v2/development/continuity").json()
    assert fh["stage"] != dev["stage"]
    assert "learning_curves" not in fh
    assert "comparison" not in dev


# ------------------------------------------------------------- the disclosures
def test_disclosures_separate_invalidated_superseded_and_repaired(client):
    d = client.get("/api/v2/disclosures").json()
    kinds = {x["kind"] for x in d["disclosures"]}
    assert {"invalidated", "superseded", "repaired"} <= kinds
    text = " ".join(x["summary"] for x in d["disclosures"]).lower()
    assert "sb3" in text
    assert all(x["used_in_reported_results"] is False for x in d["disclosures"])
    invalid = [x for x in d["disclosures"] if x["kind"] == "invalidated"]
    assert invalid and all(x["preserved"] for x in invalid)


# ------------------------------------------------------------------- replay
def test_replay_index_is_complete(client):
    d = client.get("/api/v2/replay/index").json()
    assert len(d["episodes"]) == 315
    assert d["stage"] == identity.STAGE_FINAL_HOLDOUT
    assert "available" in d
    assert d["configure_hint"]


def test_replay_rejects_non_holdout_seed(client):
    r = client.get("/api/v2/replay/episode",
                   params={"policy_id": "root42_masked_bandit",
                           "scenario": "link_failure", "seed": 101})
    assert r.status_code == 400
    assert "holdout" in r.json()["detail"]["message"].lower()


def test_replay_rejects_unknown_scenario(client):
    r = client.get("/api/v2/replay/episode",
                   params={"policy_id": "root42_masked_bandit",
                           "scenario": "nope", "seed": 1001})
    assert r.status_code == 400


@pytest.mark.skipif(
    not __import__("mplssim.evidence.replay", fromlist=["x"]).replay_available(),
    reason="full holdout artifacts not configured on this machine")
def test_replay_episode_is_labelled_recorded_not_live(client):
    d = client.get("/api/v2/replay/episode",
                   params={"policy_id": "root42_masked_bandit",
                           "scenario": "link_failure", "seed": 1001}).json()
    assert d["provenance"]["kind"] == "recorded_replay"
    assert d["provenance"]["live"] is False
    assert len(d["steps"]) == identity.SCENARIO_STEPS["link_failure"]


# ------------------------------------------------------------------ read only
def test_evidence_api_exposes_only_get_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    v2 = [p for p in paths if p.startswith("/api/v2")]
    assert len(v2) >= 10
    for p in v2:
        methods = set(paths[p])
        assert methods == {"get"}, f"{p} exposes {methods - {'get'}}"


def test_evidence_api_has_no_training_evaluation_or_selection_route(client):
    paths = [p for p in client.get("/openapi.json").json()["paths"]
             if p.startswith("/api/v2")]
    for p in paths:
        low = p.lower()
        for banned in ("train", "tune", "evaluate", "select", "sweep", "rerun"):
            assert banned not in low, f"{p} looks like a {banned} route"


def test_evidence_endpoints_never_write_into_governed_paths(client, monkeypatch):
    import builtins
    from pathlib import Path

    from mplssim.evidence.loader import default_root

    real_open = builtins.open
    guarded = [default_root().results_dir.resolve(),
               (default_root().results_dir.parent / "runs").resolve()]

    def guard(file, mode="r", *a, **kw):
        if not hasattr(file, "read") and any(m in mode for m in "wxa+"):
            p = Path(file).resolve()
            for g in guarded:
                assert g not in p.parents, f"write attempted inside {g}: {p}"
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guard)
    for path in ("/api/v2/study", "/api/v2/final-holdout",
                 "/api/v2/final-holdout/scenarios",
                 "/api/v2/final-holdout/reward-components",
                 "/api/v2/final-holdout/actions", "/api/v2/final-holdout/integrity",
                 "/api/v2/final-holdout/provenance",
                 "/api/v2/development/continuity", "/api/v2/development/seed42",
                 "/api/v2/disclosures", "/api/v2/replay/index"):
        assert client.get(path).status_code == 200, path


def test_evidence_failure_is_reported_as_unavailable_not_as_data(client, monkeypatch):
    """A broken artifact must surface as 503 with a named error, never as zeros."""
    from mplssim.evidence import errors
    from server import evidence_api

    def boom(*a, **kw):
        raise errors.IntegrityError("synthetic integrity failure")

    monkeypatch.setattr(evidence_api, "_final_holdout", boom)
    r = client.get("/api/v2/final-holdout")
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "IntegrityError"
    assert "synthetic" in r.json()["detail"]["message"]


# ------------------------------------------------------- the live app is intact
def test_mounting_the_evidence_router_did_not_disturb_the_live_api(client):
    assert client.get("/api/scenarios").status_code == 200
    assert client.get("/api/benchmark").status_code == 200
    assert client.get("/api/simulation/status").json()["state"] == "idle"
