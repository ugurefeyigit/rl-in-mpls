"""Contract tests for the additive product API.

The point of these is not that the routes return 200. It is that they cannot
return a *plausible* answer where the engine has none: an unbound checkpoint is
unavailable with a reason, a bandit output is never a probability, a mask reason
comes from the validator, a counterfactual runs on clones only, and nothing here
writes a byte under `results/` or `runs/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.evidence import identity
from mplssim.factory import get_topology, get_traffic_config
from mplssim.product import catalog, contracts, display_map, fingerprint, schemas
from server.main import STATE, app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def live(client):
    """A paused two-runner V1 session; no model load, so it stays fast."""
    r = client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["static", "greedy"],
        "seed": 42, "autostart": False, "model_tag": None})
    assert r.status_code == 200
    client.post("/api/simulation/step")
    yield client


# =========================================================== capabilities
def test_capability_catalog_lists_only_real_controllers(client):
    body = client.get("/api/product/capabilities").json()
    ids = {p["id"] for p in body["live_policies"]}
    assert ids == {"rl", "static", "greedy", "cspf", "random",
                   "maskable_ppo", "masked_bandit"}
    assert body["modes"] == ["presentation", "network", "rl"]
    assert body["guided_story_mode"] == "presentation"


def test_unavailable_policies_carry_a_reason_and_are_not_silently_swapped(client):
    body = client.get("/api/product/capabilities").json()
    for policy in body["live_policies"]:
        if not policy["available"]:
            assert policy["unavailable_reason"], policy["id"]
            assert policy["checkpoint_id"] is None


def test_v2_live_demonstration_is_unavailable_without_a_bound_checkpoint(
        client, monkeypatch):
    monkeypatch.delenv(catalog.V2_LIVE_CHECKPOINTS_ENV, raising=False)
    body = client.get("/api/product/capabilities").json()
    v2 = [p for p in body["live_policies"] if p["environment_version"] == "v2"]
    assert v2, "the V2 learners must still be described, as unavailable"
    for policy in v2:
        assert policy["available"] is False
        assert catalog.V2_LIVE_CHECKPOINTS_ENV in policy["unavailable_reason"]
    v2_env = next(e for e in body["environments"] if e["version"] == "v2")
    assert v2_env["live_available"] is False


def test_no_catalog_entry_calls_a_bandit_score_a_probability(client):
    body = client.get("/api/product/capabilities").json()
    bandit = next(p for p in body["live_policies"] if p["id"] == "masked_bandit")
    assert bandit["output_semantics"] == "scores"
    assert bandit["output_is_percentage"] is False
    # The label and the semantic description may *deny* being a probability;
    # neither may claim to be one.
    for field in ("output_label", "label"):
        assert "probabilit" not in bandit[field].lower()
        assert "confidence" not in bandit[field].lower()
    assert "immediate-reward estimate" in bandit["output_description"].lower()
    assert "confidence" not in json.dumps(bandit).lower()


def test_ppo_output_is_declared_as_probabilities(client):
    body = client.get("/api/product/capabilities").json()
    ppo = next(p for p in body["live_policies"] if p["id"] == "rl")
    assert ppo["output_semantics"] == "probabilities"
    assert ppo["output_is_percentage"] is True


def test_holdout_seeds_are_declared_blocked_for_live_use(client):
    body = client.get("/api/product/capabilities").json()
    assert body["holdout_seeds_blocked_for_live"] == list(identity.HOLDOUT_SEEDS)


def test_recorded_source_reports_its_own_availability(client):
    body = client.get("/api/product/capabilities").json()
    recorded = next(s for s in body["sources"] if s["kind"] == "recorded_replay")
    assert recorded["may_render_link_telemetry"] is False
    assert "per-link" in recorded["link_telemetry_reason"].lower()
    if not recorded["available"]:
        assert "V2_FULL_ARTIFACTS" in recorded["unavailable_reason"]


def test_contracts_endpoint_publishes_both_noop_grains(client):
    body = client.get("/api/product/contracts").json()
    grains = body["noop_metrics"]
    assert set(grains) == {"step_pooled_noop_share", "episode_mean_noop_frequency"}
    assert grains["step_pooled_noop_share"]["denominator"] != \
        grains["episode_mean_noop_frequency"]["denominator"]


def test_contracts_endpoint_carries_both_planning_conclusion_halves(client):
    findings = " ".join(client.get("/api/product/contracts").json()["final_findings"])
    assert "did not positively establish" in findings
    assert "does not establish that planning is generally irrelevant" in findings


# ============================================================= display map
def test_display_map_covers_every_router_and_link(client):
    body = client.get("/api/product/display-map").json()
    topo = get_topology()
    assert {n["id"] for n in body["nodes"]} == set(topo.routers)
    assert {l["id"] for l in body["links"]} == set(topo.link_defs)
    assert body["geographic_precision"] == "curated_not_gis"
    assert "not exact GIS" in body["layout_note"]


def test_display_map_leads_with_city_and_role_and_keeps_the_internal_id(client):
    body = client.get("/api/product/display-map").json()
    ankara = next(n for n in body["nodes"] if n["id"] == "P2")
    assert ankara["city"] == "Ankara"
    assert ankara["title"] == "ANKARA · LSR"
    assert ankara["id"] == "P2"


def test_every_display_city_is_the_established_registry_name(client):
    from mplssim.display import CITY_NAMES
    body = client.get("/api/product/display-map").json()
    assert {n["id"]: n["city"] for n in body["nodes"]} == CITY_NAMES


def test_display_coordinates_are_fixed_and_never_overlap(client):
    body = client.get("/api/product/display-map").json()
    points = [(n["x"], n["y"]) for n in body["nodes"]]
    assert len(set(points)) == len(points)
    for x, y in points:
        assert 0 <= x <= 100 and 0 <= y <= 100
    # a second read returns identical positions: the layout is not generated
    again = client.get("/api/product/display-map").json()
    assert {n["id"]: (n["x"], n["y"]) for n in again["nodes"]} == \
        {n["id"]: (n["x"], n["y"]) for n in body["nodes"]}


def test_node_plates_do_not_collide_with_link_geometry():
    """Every node keeps clearance from every link it is not an endpoint of."""
    body = display_map.display_map(get_topology())
    nodes = {n["id"]: (n["x"], n["y"]) for n in body["nodes"]}

    def distance_to_segment(p, a, b):
        (px, py), (ax, ay), (bx, by) = p, a, b
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        cx, cy = ax + t * dx, ay + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    tight = []
    for link in body["links"]:
        points = [nodes[link["a"]], *[tuple(p) for p in link["bends"]], nodes[link["z"]]]
        for node_id, point in nodes.items():
            if node_id in (link["a"], link["z"]):
                continue
            gap = min(distance_to_segment(point, points[i], points[i + 1])
                      for i in range(len(points) - 1))
            if gap < 4.0:
                tight.append((link["id"], node_id, round(gap, 2)))
    assert not tight, f"links pass too close to node plates: {tight}"


def test_the_two_signature_links_stay_separable(client):
    body = client.get("/api/product/display-map").json()
    signature = {l["id"]: l for l in body["links"] if l["signature"]}
    assert set(signature) == {"L11", "L20"}
    assert "Ankara–Kayseri" in signature["L11"]["signature"]
    assert "Kayseri–Samsun" in signature["L20"]["signature"]
    assert signature["L11"]["a"] == "P2" and signature["L11"]["z"] == "P5"
    assert signature["L20"]["a"] == "P5" and signature["L20"]["z"] == "P8"


def test_display_map_does_not_mutate_the_scientific_topology():
    before = {r: (t.x, t.y) for r, t in get_topology().routers.items()}
    display_map.display_map(get_topology())
    after = {r: (t.x, t.y) for r, t in get_topology().routers.items()}
    assert before == after


def test_utilization_bands_are_discrete_and_labelled():
    bands = display_map.UTILIZATION_BANDS
    assert [b["id"] for b in bands] == ["quiet", "working", "loaded",
                                        "congested", "overloaded"]
    assert bands[-1]["max"] is None
    assert display_map.utilization_band(0.95)["id"] == "congested"
    assert display_map.utilization_band(1.4)["state"] == "failure"
    # every band carries a non-colour marker
    assert all("ticks" in b for b in bands)


# ================================================================= schemas
@pytest.mark.parametrize("version,dim", [("v1", 586), ("v2", 604)])
def test_observation_groups_tile_the_vector_exactly(client, version, dim):
    body = client.get(f"/api/rl/schema?environment={version}").json()
    observation = body["observation"]
    assert observation["dim"] == dim
    covered = []
    for group in observation["groups"]:
        assert group["end"] - group["start"] == group["length"]
        covered.extend(range(group["start"], group["end"]))
    assert sorted(covered) == list(range(dim)), "groups overlap or leave a gap"


def test_v2_schema_comes_from_the_yaml_definition(client):
    body = client.get("/api/rl/schema?environment=v2").json()
    assert body["observation"]["source"].endswith("rl_observation_v2.yaml")
    assert body["observation"]["version"] == "obs-v2.0-notime-604"
    assert body["environment_class"] == identity.ENVIRONMENT


def test_schema_rejects_an_unknown_environment(client):
    assert client.get("/api/rl/schema?environment=v3").status_code == 422


def test_action_schema_is_noop_plus_seventeen_by_four(client):
    action = client.get("/api/rl/schema?environment=v2").json()["action"]
    assert action["count"] == 69
    assert action["n_demands"] == 17 and action["k_paths"] == 4
    assert len(action["actions"]) == 69
    assert action["actions"][0]["type"] == "noop"
    reroutes = [a for a in action["actions"] if a["type"] == "reroute"]
    assert len(reroutes) == 68
    for row in reroutes:
        assert row["action"] == 1 + 4 * row["demand_idx"] + row["path_idx"]


def test_reward_schema_keeps_v1_and_v2_apart(client):
    v2 = client.get("/api/rl/schema?environment=v2").json()["reward"]
    v1 = client.get("/api/rl/schema?environment=v1").json()["reward"]
    assert v2["components"] == list(identity.REWARD_COMPONENTS)
    assert len(v2["components"]) == 12
    assert v1["components"] != v2["components"]
    assert "never padded" in v1["note"]


def test_schema_axes_match_the_engine_ordering(client):
    axes = client.get("/api/rl/schema?environment=v2").json()["axes"]
    topo, traffic = get_topology(), get_traffic_config()
    assert [d["index"] for d in axes["dlink"]] == list(range(topo.n_dlinks))
    assert [d["id"] for d in axes["demand"]] == [d.id for d in traffic.demands]
    assert axes["dlink"][0]["label"].count("→") == 1


# ================================================================ snapshot
def test_snapshot_requires_a_live_session(client):
    STATE["session"] = None
    r = client.get("/api/simulation/snapshot")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "NoLiveSession"


def test_snapshot_carries_full_live_provenance(live):
    body = live.get("/api/simulation/snapshot").json()
    provenance = body["provenance"]
    assert provenance["source_kind"] == "live_session"
    assert provenance["label"] == "LIVE"
    assert provenance["live"] is True
    required = contracts.source_profile(
        contracts.SourceKind.LIVE_SESSION).required_fields
    for field in required:
        assert provenance.get(field) is not None, field


def test_snapshot_folds_both_directions_into_one_physical_link(live):
    body = live.get("/api/simulation/snapshot").json()
    assert len(body["links"]) == 32
    for link in body["links"]:
        assert len(link["directions"]) == 2
        assert link["worst_utilization"] == max(d["utilization"]
                                                for d in link["directions"])
        assert "busier direction" in link["worst_direction_rule"]


def test_snapshot_names_city_role_and_keeps_the_internal_id(live):
    body = live.get("/api/simulation/snapshot").json()
    node = next(n for n in body["nodes"] if n["id"] == "P5")
    assert node["city"] == "Kayseri" and node["role_token"] == "LSR"
    assert node["title"] == "KAYSERI · LSR"


def test_snapshot_metrics_separate_current_from_previous(live):
    live.post("/api/simulation/step")
    body = live.get("/api/simulation/snapshot").json()
    metrics = body["metrics"]
    assert metrics["available"] is True and metrics["has_previous"] is True
    row = metrics["values"]["max_util"]
    assert row["previous"] is not None and row["delta"] is not None


def test_snapshot_reports_no_metrics_rather_than_zero_before_the_first_step(client):
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["static"], "seed": 7,
        "autostart": False, "model_tag": None})
    metrics = client.get("/api/simulation/snapshot").json()["metrics"]
    assert metrics["available"] is False
    assert "nothing to report" in metrics["reason"]


def test_focused_object_routes_resolve_and_refuse_unknown_ids(live):
    assert live.get("/api/simulation/object/link/L11").json()["kind"] == "link"
    assert live.get("/api/simulation/object/demand/D1").json()["kind"] == "demand"
    router = live.get("/api/simulation/object/router/P2").json()
    assert router["object"]["city"] == "Ankara"
    assert "RSVP-TE" in router["not_modeled"]
    assert live.get("/api/simulation/object/link/L999").status_code == 404
    assert live.get("/api/simulation/object/planet/P2").status_code == 400


# ================================================================ decision
def test_decision_pipeline_lists_every_stage(live):
    body = live.get("/api/simulation/decision").json()
    assert body["pipeline"] == ["observation", "mask", "policy_output",
                                "selected_action", "safety", "transition",
                                "reward", "next_observation"]


def test_mask_reasons_come_from_the_validator_not_from_the_boolean(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["rl"], "seed": 42,
        "autostart": False, "model_tag": "ppo_te"})
    client.post("/api/simulation/step")
    grid = client.get("/api/simulation/decision").json()["mask"]
    assert grid["available"] is True
    assert grid["count"] == 69
    assert grid["reason_source"].endswith("validate_action")
    invalid = [a for a in grid["actions"] if not a["valid"] and a["action"] != 0]
    assert invalid, "at least one action is masked in a failure scenario"
    for row in invalid:
        assert row["reason"] not in ("", "ok")


def test_action_grid_covers_the_whole_space_with_no_op_separated(client):
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["rl"], "seed": 42,
        "autostart": False, "model_tag": "ppo_te"})
    grid = client.get("/api/simulation/decision").json()["mask"]
    assert grid["actions"][0]["type"] == "noop"
    assert grid["actions"][0]["valid"] is True
    assert len([a for a in grid["actions"] if a["type"] == "reroute"]) == 68


def test_a_baseline_runner_reports_no_action_space_rather_than_a_fake_one(live):
    body = live.get("/api/simulation/decision?algorithm=static").json()
    assert body["mask"]["available"] is False
    assert "rule-based" in body["mask"]["reason"]
    assert body["observation"]["available"] is False


def test_policy_output_labels_match_the_declared_semantics(client):
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["rl"], "seed": 42,
        "autostart": False, "model_tag": "ppo_te"})
    client.post("/api/simulation/step")
    output = client.get("/api/simulation/decision").json()["policy_output"]
    assert output["semantics"] == "probabilities"
    assert output["label"] == "Action probability"
    if output["available"]:
        assert output["entropy"] is None and output["entropy_reason"]
        assert output["value"] is None and output["value_reason"]


def test_reward_reports_its_own_exact_sum_state(live):
    reward = live.get("/api/simulation/decision").json()["reward"]
    assert reward["available"] is True
    assert reward["exact_sum"] is True
    assert abs(reward["residual"]) <= 5e-4
    assert reward["cumulative_reward"] is not None
    assert reward["environment_version"] == "v1"


def test_changed_feature_ranking_is_never_called_causal(client):
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["rl"], "seed": 42,
        "autostart": False, "model_tag": "ppo_te"})
    client.post("/api/simulation/step")
    observation = client.get("/api/simulation/decision").json()["observation"]
    assert observation["available"] is True
    assert observation["dim"] == 586
    note = observation["ranking_note"].lower()
    assert "not causal importance" in note
    assert "internal reasoning" in note


# ================================================================ timeline
def test_timeline_separates_frr_protection_from_te_actions(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["greedy"], "seed": 42,
        "autostart": False, "model_tag": None})
    client.post("/api/simulation/run-until",
                json={"condition": "end", "max_steps": 40})
    body = client.get("/api/simulation/timeline").json()
    kinds = {e["kind"] for e in body["events"]}
    assert "failure" in kinds
    assert "built-in local repair" in body["frr_note"]
    for event in body["events"]:
        if event["kind"] == "frr":
            assert event["is_protection"] is True
        if event["kind"] == "action":
            assert event.get("is_protection") is False


def test_timeline_event_ids_are_stable_across_reads(client):
    client.post("/api/simulation/start", json={
        "scenario": "link_failure", "algorithms": ["greedy"], "seed": 42,
        "autostart": False, "model_tag": None})
    client.post("/api/simulation/run-until",
                json={"condition": "end", "max_steps": 20})
    first = [e["id"] for e in client.get("/api/simulation/timeline").json()["events"]]
    second = [e["id"] for e in client.get("/api/simulation/timeline").json()["events"]]
    assert first == second and first


# ============================================================== comparison
def test_paired_runners_prove_they_share_one_experiment(live):
    body = live.get("/api/simulation/comparison").json()
    assert body["comparison"] is True
    assert body["matched"] is True
    assert body["mismatched_fields"] == []
    assert len(body["lanes"]) == 2


def test_a_single_runner_session_is_not_a_failed_comparison(client):
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["static"], "seed": 3,
        "autostart": False, "model_tag": None})
    body = client.get("/api/simulation/comparison").json()
    assert body["comparison"] is False
    assert body["matched"] is None
    assert "nothing to compare" in body["reason"]


def test_a_desynchronized_pair_disables_the_verdict(live):
    session = STATE["session"]
    # Reach into one engine only — exactly the failure mode the proof exists for.
    session.runners[1].eng.link_up["L11"] = False
    body = live.get("/api/simulation/comparison").json()
    assert body["matched"] is False
    assert "link_up" in body["mismatched_fields"]
    assert "no comparative verdict" in body["reason"]


def test_interventions_reach_every_paired_engine(live):
    live.post("/api/failure/inject", json={"link": "L11"})
    body = live.get("/api/simulation/comparison").json()
    assert body["matched"] is True, body["mismatched_fields"]


# ========================================================== counterfactual
def test_counterfactual_leaves_the_running_session_untouched(live):
    session = STATE["session"]
    engine = session.runners[0].eng
    before = fingerprint.full_fingerprint(engine)
    step_before, history_before = engine.step_count, len(engine.metrics_history)

    body = live.post("/api/simulation/counterfactual", json={"action": 5}).json()

    assert body["kind"] == "simulated_estimate"
    assert body["session_unchanged"] is True
    assert fingerprint.full_fingerprint(engine) == before
    assert engine.step_count == step_before
    assert len(engine.metrics_history) == history_before


def test_counterfactual_is_labelled_a_simulated_estimate(live):
    body = live.post("/api/simulation/counterfactual", json={"action": 5}).json()
    assert "Simulated one-interval estimate" in body["label"]
    assert "not an observed outcome" in body["label"]
    assert "not final evidence" in body["label"]


def test_counterfactual_refuses_a_stale_generation(live):
    body = live.post("/api/simulation/counterfactual",
                     json={"action": 5, "generation": 999})
    assert body.status_code == 409
    assert "reset" in body.json()["detail"]["reason"]


def test_counterfactual_refuses_a_stale_step(live):
    body = live.post("/api/simulation/counterfactual",
                     json={"action": 5, "step": 999})
    assert body.status_code == 409


def test_counterfactual_rejects_an_out_of_range_action(live):
    assert live.post("/api/simulation/counterfactual",
                     json={"action": 69}).status_code == 422


def test_counterfactual_of_noop_says_there_is_nothing_to_compare(live):
    body = live.post("/api/simulation/counterfactual", json={"action": 0}).json()
    assert body["action_metrics"] is None
    assert "nothing to compare" in body["action_reason"]


# ============================================== no writes to governed paths
def test_the_product_layer_writes_nothing_under_results_or_runs(live):
    def stamp(directory: Path) -> dict[str, float]:
        if not directory.exists():
            return {}
        return {str(p): p.stat().st_mtime_ns
                for p in directory.rglob("*") if p.is_file()}

    before = {name: stamp(ROOT / name) for name in ("results", "runs", "models")}
    live.get("/api/product/capabilities")
    live.get("/api/product/display-map")
    live.get("/api/rl/schema?environment=v2")
    live.get("/api/simulation/snapshot")
    live.get("/api/simulation/decision")
    live.get("/api/simulation/timeline")
    live.post("/api/simulation/counterfactual", json={"action": 5})
    after = {name: stamp(ROOT / name) for name in ("results", "runs", "models")}
    assert before == after


def test_the_product_modules_never_import_training_code():
    banned = ("trainers_v2", "evaluation_v2", "sb3_contrib", "stable_baselines",
              "masked_bandit", "learning_common")
    for module in (ROOT / "mplssim" / "product").glob("*.py"):
        text = module.read_text(encoding="utf-8")
        for name in banned:
            assert f"import {name}" not in text, f"{module.name} imports {name}"
            assert f"from mplssim.experiments.{name}" not in text, module.name


def test_the_evidence_api_stays_get_only_and_unchanged(client):
    assert client.get("/api/v2/study").status_code == 200
    for method, path in (("post", "/api/v2/study"), ("put", "/api/v2/final-holdout"),
                         ("delete", "/api/v2/replay/index")):
        assert getattr(client, method)(path).status_code == 405


# ================================================================ contracts
def test_the_serialized_snapshot_matches_the_engine_arrays(live):
    session = STATE["session"]
    engine = session.runners[0].eng
    body = live.get("/api/simulation/snapshot").json()
    raw = engine.snapshot()
    assert body["time"]["step"] == raw["step"]
    for link in body["links"]:
        source = [r for r in raw["links"] if r["link"] == link["id"]]
        assert link["capacity_mbps"] == source[0]["capacity_mbps"]
        assert link["worst_utilization"] == max(r["utilization"] for r in source)
    assert len(body["demands"]) == len(raw["demands"])
