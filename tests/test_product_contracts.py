"""Executable product contracts for the unified four-mode product.

These freeze the Exp 2.1 migration — exactly four primary modes, Guided Story
nested inside Presentation, four *typed* source
kinds that cannot be coerced into one another, backward-compatible routes, and
the scientific constants any product surface is allowed to state.

Reference: docs/superpowers/specs/2026-07-31-unified-rl-in-mpls-product-design.md
"""

from __future__ import annotations

import pytest

from mplssim.evidence import identity
from mplssim.product import contracts


# ------------------------------------------------------------------ modes
def test_exactly_four_primary_modes():
    assert [m.id for m in contracts.PRIMARY_MODES] == [
        "presentation", "network", "rl", "compare"]


def test_guided_story_is_a_presentation_workflow_not_a_mode():
    assert contracts.GUIDED_STORY.mode == "presentation"
    assert contracts.GUIDED_STORY.id not in {m.id for m in contracts.PRIMARY_MODES}
    for workflow in contracts.WORKFLOWS:
        assert workflow.mode in {m.id for m in contracts.PRIMARY_MODES}


def test_mode_ids_are_stable_and_labelled():
    labels = {m.id: m.label for m in contracts.PRIMARY_MODES}
    assert labels == {
        "presentation": "Presentation",
        "network": "Network Information",
        "rl": "RL Information",
        "compare": "Comparative Run Results",
    }


# ----------------------------------------------------------------- routes
@pytest.mark.parametrize("path,mode", [
    ("/", "network"),
    ("/advanced", "network"),
    ("/present", "presentation"),
    ("/study", "rl"),
    ("/compare", "compare"),
])
def test_backward_compatible_routes_map_to_the_approved_mode(path, mode):
    assert contracts.route_context(path).mode == mode


def test_study_route_opens_governed_study_on_final_evidence():
    ctx = contracts.route_context("/study")
    assert ctx.rl_view == "study"
    assert ctx.source_kind == contracts.SourceKind.FINAL_HOLDOUT_EVIDENCE


def test_present_route_opens_a_live_session_context():
    ctx = contracts.route_context("/present")
    assert ctx.source_kind == contracts.SourceKind.LIVE_SESSION
    assert ctx.workflow is None


def test_every_documented_route_is_registered():
    assert set(contracts.ROUTES) == {"/", "/advanced", "/present", "/study", "/compare"}


# ----------------------------------------------------------- source kinds
def test_four_source_kinds_exist_with_distinct_persistent_labels():
    labels = {k: contracts.source_profile(k).label for k in contracts.SourceKind}
    assert labels == {
        contracts.SourceKind.LIVE_SESSION: "LIVE",
        contracts.SourceKind.RECORDED_REPLAY: "RECORDED",
        contracts.SourceKind.DEVELOPMENT_EVIDENCE: "DEVELOPMENT",
        contracts.SourceKind.FINAL_HOLDOUT_EVIDENCE: "FINAL EVIDENCE",
    }
    assert len(set(labels.values())) == 4


def test_only_live_sessions_may_execute_a_policy():
    executes = {k for k in contracts.SourceKind
                if contracts.source_profile(k).may_execute_policy}
    assert executes == {contracts.SourceKind.LIVE_SESSION}


def test_only_live_sessions_may_render_link_level_topology_telemetry():
    renders = {k for k in contracts.SourceKind
               if contracts.source_profile(k).may_render_link_telemetry}
    assert renders == {contracts.SourceKind.LIVE_SESSION}


def test_recorded_traces_declare_link_telemetry_unavailable_with_a_reason():
    profile = contracts.source_profile(contracts.SourceKind.RECORDED_REPLAY)
    assert profile.may_render_link_telemetry is False
    assert "per-link" in profile.link_telemetry_reason.lower()


def test_source_kinds_cannot_be_coerced_into_one_another():
    with pytest.raises(ValueError):
        contracts.SourceKind("live")           # not the wire value
    with pytest.raises(ValueError):
        contracts.SourceKind("final_evidence")  # not the wire value
    assert contracts.SourceKind.LIVE_SESSION.value == "live_session"
    assert contracts.SourceKind.FINAL_HOLDOUT_EVIDENCE.value == "final_holdout_evidence"


def test_required_provenance_fields_differ_by_source_kind():
    live = contracts.source_profile(contracts.SourceKind.LIVE_SESSION).required_fields
    final = contracts.source_profile(
        contracts.SourceKind.FINAL_HOLDOUT_EVIDENCE).required_fields
    assert "session_id" in live and "session_id" not in final
    assert "source_sha" in final and "source_sha" not in live
    assert live != final


def test_evidence_stages_never_share_a_region():
    dev = contracts.SourceKind.DEVELOPMENT_EVIDENCE
    final = contracts.SourceKind.FINAL_HOLDOUT_EVIDENCE
    assert not contracts.may_share_region(dev, final)
    assert not contracts.may_share_region(final, dev)
    assert contracts.may_share_region(final, final)


def test_live_demonstration_is_live_not_final_evidence():
    profile = contracts.source_profile(contracts.SourceKind.LIVE_SESSION)
    assert profile.label == "LIVE"
    assert contracts.LIVE_DEMONSTRATION_LABEL == "LIVE DEMONSTRATION"
    assert contracts.LIVE_DEMONSTRATION_KIND == contracts.SourceKind.LIVE_SESSION


# ------------------------------------------------------- scientific pins
def test_environment_versions_pin_observation_sizes():
    assert contracts.ENVIRONMENTS["v1"].observation_dim == 586
    assert contracts.ENVIRONMENTS["v2"].observation_dim == identity.OBSERVATION_DIM == 604
    assert contracts.ENVIRONMENTS["v1"].action_count == 69
    assert contracts.ENVIRONMENTS["v2"].action_count == identity.ACTION_COUNT == 69


def test_action_space_is_noop_plus_seventeen_demands_by_four_paths():
    assert contracts.ACTION_COUNT == 1 + 17 * 4 == 69
    assert contracts.decode_action(0) == (None, None)
    assert contracts.decode_action(1) == (0, 0)
    assert contracts.decode_action(68) == (16, 3)
    seen = {contracts.decode_action(a) for a in range(1, 69)}
    assert len(seen) == 68
    with pytest.raises(ValueError):
        contracts.decode_action(69)


def test_v2_reward_components_keep_the_authoritative_order():
    assert contracts.V2_REWARD_COMPONENTS == identity.REWARD_COMPONENTS
    assert len(contracts.V2_REWARD_COMPONENTS) == 12
    assert contracts.V2_REWARD_COMPONENTS[0] == "delivery"
    assert contracts.V2_REWARD_COMPONENTS[-1] == "invalid"


# ------------------------------------------------------------ no-op grains
def test_the_two_noop_grains_never_collapse_into_one_metric():
    pooled = contracts.NOOP_METRICS["step_pooled_noop_share"]
    episode = contracts.NOOP_METRICS["episode_mean_noop_frequency"]
    assert pooled.id != episode.id
    assert pooled.label != episode.label
    assert pooled.denominator != episode.denominator
    assert pooled.description != episode.description
    assert "no-op rate" not in {pooled.label.lower(), episode.label.lower()}


# ---------------------------------------------------- policy output types
def test_policy_output_semantics_cannot_cross_label():
    assert contracts.OutputSemantics.PROBABILITIES.label == "Action probability"
    assert contracts.OutputSemantics.SCORES.label == "Action score"
    for semantics in contracts.OutputSemantics:
        if semantics is contracts.OutputSemantics.SCORES:
            assert "probabilit" not in semantics.label.lower()
            assert "confidence" not in semantics.label.lower()
    assert contracts.OutputSemantics.SCORES.percent is False
    assert contracts.OutputSemantics.PROBABILITIES.percent is True


def test_bandit_output_is_never_described_as_a_probability():
    text = contracts.OutputSemantics.SCORES.description.lower()
    assert "immediate-reward estimate" in text
    assert "probability" not in text and "confidence" not in text


# ------------------------------------------------------- banned vocabulary
def test_forbidden_product_vocabulary_is_enumerated():
    banned = {b.lower() for b in contracts.FORBIDDEN_PRODUCT_PHRASES}
    assert "ai advisor" in banned
    assert "causal importance" in banned


def test_product_contracts_do_not_duplicate_frozen_scientific_conclusions():
    assert not hasattr(contracts, "FINAL_FINDINGS")
