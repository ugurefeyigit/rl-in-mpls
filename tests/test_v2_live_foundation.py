"""The live V2 foundation: default, provenance, pairing, execution, resets.

These tests pin the behaviours Part 1 exists to establish:

- V2 is the live default, and an unavailable V2 artifact fails closed with a
  reason rather than degrading into a V1 run;
- the six live checkpoints are exactly the pre-holdout continuity selection,
  identified by the SHA-256 hashes the governed provenance record carries;
- a paired V2 session proves synchronization or refuses to claim a comparison;
- automatic execution has no proposal to approve, and advisor execution holds
  one until the operator answers;
- reset run rebuilds the same experiment and keeps the run it replaced, while
  full reset clears the session; neither touches a model or any evidence.

Nothing here trains, tunes, evaluates, reads a holdout environment or writes
under `results/` or `runs/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.product import checkpoints_v2
from mplssim.product import pairing
from mplssim.product.contracts import SourceKind, source_profile
from server.main import STATE, app
from server.session import (
    DEFAULT_ENVIRONMENT, V1_ALGORITHMS, V2_ALGORITHMS, algorithms_for,
)

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = ROOT.parents[2] if (ROOT / ".claude").exists() else ROOT
PROVENANCE_CSV = ROOT / "results" / "v2_final_holdout" / "checkpoint_provenance.csv"
SELECTION_CSV = (ROOT / "results" / "v2_three_root_continuity"
                 / "checkpoint_selection.csv")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client
    STATE["session"] = None


def start(client, **over):
    body = {"scenario": "demo_evening", "algorithms": ["masked_bandit"],
            "seed": 42, "speed": "fast", "autostart": False}
    body.update(over)
    return client.post("/api/simulation/start", json=body)


# ==================================================== registry and provenance
def test_the_registry_is_exactly_the_six_continuity_checkpoints():
    assert len(checkpoints_v2.REGISTRY) == 6
    pairs = {(e.training_root, e.algorithm) for e in checkpoints_v2.REGISTRY}
    assert pairs == {(root, algo) for root in (42, 314159, 271828)
                     for algo in ("masked_bandit", "maskable_ppo")}


def test_every_registry_hash_matches_the_governed_provenance_record():
    """The live registry is transcribed from the study's own record, not guessed."""
    rows = list(csv.DictReader(PROVENANCE_CSV.read_text(encoding="utf-8").splitlines()))
    recorded = {(int(r["training_root"]), r["algorithm"]): r for r in rows}
    for entry in checkpoints_v2.REGISTRY:
        row = recorded[(entry.training_root, entry.algorithm)]
        assert entry.payload_sha256 == row["payload_sha256"]
        assert entry.sidecar_sha256 == row["sidecar_sha256"]
        assert entry.transition == int(row["checkpoint_transition"])
        assert entry.training_source_sha == row["training_source_sha"]
        assert row["checkpoint_path"].endswith(entry.payload_name)


def test_the_transitions_are_the_pre_holdout_continuity_selection():
    rows = list(csv.DictReader(SELECTION_CSV.read_text(encoding="utf-8").splitlines()))
    selected = {(int(r["training_root"]), r["algorithm"]):
                int(r["checkpoint_transition"]) for r in rows}
    for entry in checkpoints_v2.REGISTRY:
        assert entry.transition == selected[(entry.training_root, entry.algorithm)]


def test_the_default_root_rule_is_neutral_to_holdout_performance():
    assert checkpoints_v2.DEFAULT_ROOT == 42
    assert checkpoints_v2.DEFAULT_ROOT == checkpoints_v2.TRAINING_ROOTS[0]
    rule = checkpoints_v2.DEFAULT_ROOT_RULE.lower()
    assert "not chosen from final-holdout performance" in rule


def test_an_unknown_algorithm_or_root_fails_closed():
    with pytest.raises(checkpoints_v2.CheckpointUnavailable):
        checkpoints_v2.entry_for("masked_bandit", 999)
    with pytest.raises(checkpoints_v2.CheckpointUnavailable):
        checkpoints_v2.load("ppo_te")


def test_a_tampered_payload_hash_refuses_to_load(monkeypatch):
    entry = checkpoints_v2.entry_for("masked_bandit", 42)
    if not checkpoints_v2.availability(entry)[0]:
        pytest.skip("frozen V2 artifacts are not present on this machine")
    monkeypatch.setattr(checkpoints_v2, "_sha256_file", lambda path: "0" * 64)
    with pytest.raises(checkpoints_v2.CheckpointUnavailable) as excinfo:
        checkpoints_v2.verify(entry)
    assert "SHA-256" in str(excinfo.value)


def test_a_missing_artifact_root_names_the_override_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(checkpoints_v2, "artifact_root", lambda: None)
    entry = checkpoints_v2.entry_for("maskable_ppo", 42)
    available, reason = checkpoints_v2.availability(entry)
    assert available is False
    assert checkpoints_v2.ARTIFACT_ROOT_ENV in reason
    assert "v1" not in reason.lower().split()


def test_the_verified_sidecar_declares_v2_and_the_expected_identity():
    entry = checkpoints_v2.entry_for("masked_bandit", 42)
    if not checkpoints_v2.availability(entry)[0]:
        pytest.skip("frozen V2 artifacts are not present on this machine")
    metadata = checkpoints_v2.verify(entry)
    assert metadata["run_config"]["environment_version"] == "v2"
    environment = metadata["environment_record"]["environment"]
    assert environment["observation_dim"] == 604
    assert environment["action_dim"] == 69
    assert environment["environment_class"] == "mplssim.rl.env_v2.MplsTeEnvV2"


# ============================================================ live V2 default
def test_v2_is_the_declared_live_default(client):
    assert DEFAULT_ENVIRONMENT == "v2"
    body = client.get("/api/product/capabilities").json()
    assert body["default_environment"] == "v2"
    v2 = next(e for e in body["environments"] if e["version"] == "v2")
    assert v2["is_default"] is True
    assert v2["observation_dim"] == 604


def test_starting_without_an_environment_starts_v2(client):
    response = start(client)
    assert response.status_code == 200, response.text
    status = response.json()
    assert status["environment"] == "v2"
    assert status["training_root"] == 42
    assert status["controllers"][0]["environment_version"] == "v2"
    assert status["controllers"][0]["checkpoint_id"] == "masked_bandit-root42-250000"


def test_only_genuinely_compatible_controllers_are_offered(client):
    assert set(algorithms_for("v2")) == set(V2_ALGORITHMS)
    assert "rl" in V1_ALGORITHMS and "rl" not in V2_ALGORITHMS
    response = start(client, algorithms=["rl"])
    assert response.status_code == 400
    assert "V2 environment" in response.json()["detail"]


def test_an_unavailable_v2_checkpoint_fails_closed_rather_than_using_v1(
        client, monkeypatch):
    monkeypatch.setattr(checkpoints_v2, "artifact_root", lambda: None)
    checkpoints_v2._load_cached.cache_clear()
    response = start(client, algorithms=["maskable_ppo"])
    checkpoints_v2._load_cached.cache_clear()
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert checkpoints_v2.ARTIFACT_ROOT_ENV in detail


def test_v1_still_runs_when_it_is_asked_for_by_name(client):
    response = start(client, environment="v1", algorithms=["rl"],
                     model_tag="ppo_te")
    assert response.status_code == 200, response.text
    status = response.json()
    assert status["environment"] == "v1"
    assert status["controllers"][0]["checkpoint_id"] == "ppo_te"
    snapshot = client.get("/api/simulation/snapshot").json()
    assert snapshot["provenance"]["environment_version"] == "v1"


def test_a_holdout_seed_and_a_negative_seed_are_both_refused(client):
    assert start(client, seed=1001).status_code == 400
    assert start(client, seed=-1).status_code == 400


def test_an_unknown_training_root_is_refused(client):
    response = start(client, training_root=7)
    assert response.status_code == 400
    assert "continuity roots" in response.json()["detail"]


# ===================================================== live V2 payload truth
def test_the_v2_snapshot_carries_v2_identity_and_checkpoint_provenance(client):
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    snapshot = client.get("/api/simulation/snapshot").json()
    provenance = snapshot["provenance"]
    assert provenance["environment_version"] == "v2"
    assert provenance["environment_class"] == "mplssim.rl.env_v2.MplsTeEnvV2"
    assert provenance["output_semantics"] == "scores"
    assert provenance["training_root"] == 42
    record = provenance["checkpoint_provenance"]
    assert record["inference_only"] is True
    assert record["writes_evidence"] is False
    assert record["selection"] == "pre-holdout continuity selection"
    # A frozen checkpoint driving a demonstration is still a live record.
    assert provenance["source_kind"] == SourceKind.LIVE_SESSION.value


def test_the_v2_decision_reports_twelve_components_and_the_v2_validator(client):
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    decision = client.get("/api/simulation/decision").json()
    assert decision["reward"]["environment_version"] == "v2"
    assert decision["reward"]["component_count"] == 12
    assert decision["reward"]["exact_sum"] is True
    assert decision["mask"]["reason_source"].endswith("validate_te_action")
    assert decision["safety"]["validator"] == "SimulationEngineV2.validate_te_action"


def test_a_bandit_score_is_never_called_a_probability(client):
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    output = client.get("/api/simulation/decision").json()["policy_output"]
    assert output["semantics"] == "scores"
    assert output["is_percentage"] is False
    note = output["distribution_note"].lower()
    assert "immediate-reward estimates" in note
    # The note may deny being a probability; it may never claim to be one.
    assert "not probabilities" in note
    assert "do not sum to one" in note


def test_ppo_probabilities_are_named_only_when_ppo_is_running(client):
    assert start(client, algorithms=["maskable_ppo"]).status_code == 200
    client.post("/api/simulation/step")
    output = client.get("/api/simulation/decision").json()["policy_output"]
    assert output["semantics"] == "probabilities"
    assert output["is_percentage"] is True


def test_a_v2_baseline_reports_no_observation_and_no_per_action_numbers(client):
    assert start(client, algorithms=["greedy"]).status_code == 200
    client.post("/api/simulation/step")
    decision = client.get("/api/simulation/decision").json()
    assert decision["observation"]["available"] is False
    assert decision["policy_output"]["available"] is False
    # The mask is still authoritative: a baseline submits through it.
    assert decision["mask"]["available"] is True


def test_v2_has_no_manual_traffic_override_and_says_so(client):
    assert start(client).status_code == 200
    response = client.post("/api/traffic/multiplier", json={"factor": 2.0})
    assert response.status_code == 409
    assert "will not fabricate" in response.json()["detail"]


# ================================================================== pairing
def test_two_v2_controllers_start_from_one_proved_synchronized_state(client):
    assert start(client, algorithms=["masked_bandit", "greedy"]).status_code == 200
    comparison = client.get("/api/simulation/comparison").json()
    assert comparison["environment_version"] == "v2"
    assert comparison["comparison"] is True
    assert comparison["matched"] is True
    assert comparison["mismatched_fields"] == []
    assert {lane["environment_version"] for lane in comparison["lanes"]} == {"v2"}


def test_the_pair_stays_synchronized_on_exogenous_inputs_after_stepping(client):
    assert start(client, algorithms=["masked_bandit", "maskable_ppo"]).status_code == 200
    for _ in range(3):
        client.post("/api/simulation/step")
    comparison = client.get("/api/simulation/comparison").json()
    assert comparison["matched"] is True
    assert comparison["proof"] == "exogenous inputs"


def test_a_mixed_environment_pair_is_refused_as_a_comparison():
    """Two versions are two problems; no fair comparison is claimed."""
    class Lane:
        def __init__(self, version):
            self.algorithm = "x"
            self.environment_version = version
            self.checkpoint_id = None
            self.output_semantics = "none"
            self.cumulative_reward = 0.0
            self.eng = None

    class Config:
        scenario, seed, environment, training_root = "demo_evening", 42, "v2", 42

    class Session:
        config = Config()
        runners = [Lane("v1"), Lane("v2")]

    state = pairing.synchronization(Session())
    assert state["matched"] is False
    assert state["mismatched_fields"] == ["environment_version"]


# ============================================== automatic vs advisor execution
def test_automatic_execution_has_no_proposal_to_approve(client):
    assert start(client, execution="automatic").status_code == 200
    advisor = client.get("/api/advisor/status").json()
    assert advisor["execution"] == "automatic"
    assert advisor["explanation_only"] is True
    assert advisor["pending"] is None
    response = client.post("/api/advisor/propose")
    assert response.status_code == 409
    assert "runs the policy automatically" in response.json()["detail"]


def test_advisor_execution_holds_the_action_until_the_operator_answers(client):
    assert start(client, execution="advisor").status_code == 200
    before = client.get("/api/simulation/status").json()["step"]
    proposal = client.post("/api/advisor/propose").json()
    assert proposal["output_semantics"] == "scores"
    assert client.get("/api/simulation/status").json()["awaiting_decision"] is True
    # Nothing advanced while the proposal was held.
    assert client.get("/api/simulation/status").json()["step"] == before
    assert client.post("/api/simulation/step").status_code == 409
    record = client.post("/api/advisor/reject").json()
    assert record["approved"] is False
    assert record["applied_action"] == 0
    assert client.get("/api/simulation/status").json()["step"] == before + 1


def test_approving_applies_the_proposed_action_and_rejecting_applies_none(client):
    assert start(client, execution="advisor").status_code == 200
    proposal = client.post("/api/advisor/propose").json()
    record = client.post("/api/advisor/approve").json()
    assert record["approved"] is True
    assert record["applied_action"] == proposal["action"]


def test_a_baseline_only_session_says_there_is_nothing_to_approve(client):
    assert start(client, algorithms=["greedy"], execution="advisor").status_code == 200
    response = client.post("/api/advisor/propose")
    assert response.status_code == 409
    assert "fixed rules" in response.json()["detail"]


# =================================================================== resets
def test_reset_run_rebuilds_the_same_experiment_and_keeps_the_previous_run(client):
    assert start(client, algorithms=["masked_bandit", "greedy"]).status_code == 200
    for _ in range(2):
        client.post("/api/simulation/step")
    before = client.get("/api/simulation/status").json()
    status = client.post("/api/simulation/reset").json()
    assert status["step"] == 0
    assert status["scenario"] == before["scenario"]
    assert status["seed"] == before["seed"]
    assert status["environment"] == before["environment"]
    assert status["algorithms"] == before["algorithms"]
    assert status["training_root"] == before["training_root"]
    assert status["retained_runs"] == 1
    retained = client.get("/api/simulation/retained-runs").json()
    assert retained["count"] == 1
    assert retained["runs"][0]["steps"] == 2
    assert {r["algorithm"] for r in retained["runs"][0]["runs"]} == {
        "masked_bandit", "greedy"}


def test_full_reset_stops_the_runners_and_clears_the_session(client):
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    stopped = client.post("/api/simulation/stop").json()
    assert stopped["stopped"] is True
    assert STATE["session"] is None
    assert client.get("/api/simulation/status").json()["state"] == "idle"
    assert client.get("/api/simulation/snapshot").status_code == 404


def test_neither_reset_touches_a_model_a_checkpoint_or_any_evidence(client):
    def stamp(path: Path):
        if not path.exists():
            return None
        return sorted((p.relative_to(path).as_posix(), p.stat().st_mtime,
                       p.stat().st_size)
                      for p in path.rglob("*") if p.is_file())

    watched = {name: stamp(ROOT / name) for name in ("results", "models")}
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    client.post("/api/simulation/reset")
    client.post("/api/simulation/stop")
    assert {name: stamp(ROOT / name) for name in ("results", "models")} == watched


# ========================================================= evidence separation
def test_study_evidence_is_grouped_away_from_the_live_setup_path(client):
    sources = client.get("/api/product/capabilities").json()["sources"]
    by_kind = {source["kind"]: source for source in sources}
    assert by_kind["live_session"]["group"] == "live"
    for kind in ("recorded_replay", "development_evidence",
                 "final_holdout_evidence"):
        assert by_kind[kind]["group"] == "study_evidence"


def test_evidence_is_described_in_plain_language_not_a_bare_label(client):
    sources = {s["kind"]: s for s in
               client.get("/api/product/capabilities").json()["sources"]}
    development = sources["development_evidence"]
    final = sources["final_holdout_evidence"]
    assert "before the holdout" in development["plain_label"].lower()
    assert "pilot" in development["plain_summary"].lower()
    assert "frozen" in final["plain_label"].lower()
    assert "read-only" in final["plain_label"].lower()
    assert "never a live comparison" in final["plain_summary"].lower()
    assert "never a model you" in final["plain_summary"].lower()


def test_no_evidence_source_may_execute_a_policy():
    for kind in (SourceKind.DEVELOPMENT_EVIDENCE,
                 SourceKind.FINAL_HOLDOUT_EVIDENCE,
                 SourceKind.RECORDED_REPLAY):
        assert source_profile(kind).may_execute_policy is False
    assert source_profile(SourceKind.LIVE_SESSION).may_execute_policy is True


def test_the_frozen_study_numbers_are_still_only_read_from_evidence(client):
    holdout = client.get("/api/v2/final-holdout").json()
    returns = {row["algorithm"]: row["operational_return_mean"]
               for row in holdout["aggregate"]}
    assert returns["masked_bandit"] == pytest.approx(18.221, abs=5e-3)
    assert returns["maskable_ppo"] == pytest.approx(9.036, abs=5e-3)
    assert returns["greedy"] == pytest.approx(-2.327, abs=5e-3)


# =========================================== definition freeze and no writing
def test_the_live_layer_never_moves_a_v2_definition():
    from mplssim.experiments.v2_factory import frozen_definition_drift
    assert frozen_definition_drift() == {}


def test_reading_the_v2_snapshot_does_not_advance_or_mutate_the_engine(client):
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    session = STATE["session"]
    engine = session.runners[0].eng
    before = (int(engine.step_count), float(engine.t_min),
              [int(p) for p in engine.current_path],
              dict(engine.episode_totals))
    for _ in range(3):
        engine.snapshot()
        client.get("/api/simulation/snapshot")
        client.get("/api/simulation/timeline")
    after = (int(engine.step_count), float(engine.t_min),
             [int(p) for p in engine.current_path],
             dict(engine.episode_totals))
    assert before == after


def test_the_counterfactual_runs_on_a_clone_and_leaves_the_session_unchanged(client):
    assert start(client).status_code == 200
    client.post("/api/simulation/step")
    status = client.get("/api/simulation/status").json()
    result = client.post("/api/simulation/counterfactual", json={
        "action": 0, "generation": status["generation"], "step": status["step"],
    }).json()
    assert result["kind"] == "simulated_estimate"
    assert result["session_unchanged"] is True
    assert client.get("/api/simulation/status").json()["step"] == status["step"]


def test_the_checkpoint_loader_never_saves_or_trains():
    source = (ROOT / "mplssim" / "product" / "checkpoints_v2.py").read_text(
        encoding="utf-8")
    for banned in (".learn(", ".save(", "optimizer.step", "total_timesteps"):
        assert banned not in source
