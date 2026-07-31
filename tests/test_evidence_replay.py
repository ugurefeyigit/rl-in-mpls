"""Recorded replay: serve preserved holdout episodes, never re-evaluate anything.

Replay reads compressed step traces that the one-shot evaluation already wrote. It
never constructs an environment, never loads a checkpoint, and never writes. When the
full artifacts are not configured, it degrades to a clearly-labelled unavailable state
rather than inventing a substitute.
"""

from __future__ import annotations

import builtins

import pytest

from mplssim.evidence import errors, identity, loader, replay

ROOT = loader.default_root()
HAS_ARTIFACTS = replay.replay_available()
needs_artifacts = pytest.mark.skipif(
    not HAS_ARTIFACTS, reason="full holdout artifacts not configured on this machine")


# ------------------------------------------------------------------- index
def test_index_is_complete_and_needs_no_full_artifacts():
    """The catalogue comes from the committed compact evidence, so the UI can list
    every episode and explain what is missing even with no large artifacts."""
    idx = replay.episode_index(ROOT)
    assert len(idx) == identity.TOTAL_HOLDOUT_EPISODES
    assert {e["seed"] for e in idx} == set(identity.HOLDOUT_SEEDS)
    assert {e["scenario"] for e in idx} == set(identity.SCENARIOS)
    assert len({e["policy_id"] for e in idx}) == identity.POLICY_COUNT
    assert all(e["stage"] == identity.STAGE_FINAL_HOLDOUT for e in idx)
    assert all(e["kind"] == "recorded_replay" for e in idx)


def test_index_marks_availability_rather_than_pretending(monkeypatch):
    monkeypatch.setattr(replay, "full_artifact_root", lambda: None)
    idx = replay.episode_index(ROOT)
    assert len(idx) == identity.TOTAL_HOLDOUT_EPISODES
    assert all(e["available"] is False for e in idx)


def test_unavailable_replay_raises_rather_than_fabricating(monkeypatch):
    monkeypatch.setattr(replay, "full_artifact_root", lambda: None)
    with pytest.raises(errors.ArtifactMissingError) as e:
        replay.load_episode("root42_masked_bandit", "link_failure", 1001)
    assert "V2_FULL_ARTIFACTS" in str(e.value)


# ------------------------------------------------------------ identity gates
def test_rejects_seeds_that_are_not_holdout_seeds():
    """Continuity seeds are development evidence; replay is a holdout surface."""
    with pytest.raises(errors.IdentityError):
        replay.load_episode("root42_masked_bandit", "link_failure", 101)


def test_rejects_unknown_scenarios():
    with pytest.raises(errors.IdentityError):
        replay.load_episode("root42_masked_bandit", "not_a_scenario", 1001)


def test_rejects_unknown_policies():
    with pytest.raises(errors.IdentityError):
        replay.load_episode("root99_masked_bandit", "link_failure", 1001)


# --------------------------------------------------------------- the record
@needs_artifacts
def test_recorded_episode_is_labelled_with_its_full_provenance():
    ep = replay.load_episode("root42_masked_bandit", "link_failure", 1001)
    p = ep["provenance"]
    assert p["kind"] == "recorded_replay"
    assert p["live"] is False
    assert p["stage"] == identity.STAGE_FINAL_HOLDOUT
    assert p["policy_id"] == "root42_masked_bandit"
    assert p["algorithm"] == "masked_bandit"
    assert p["training_root"] == 42
    assert p["scenario"] == "link_failure"
    assert p["seed"] == 1001
    assert p["evaluation_source_sha"] == identity.EVALUATION_SOURCE_SHA
    assert p["training_source_sha"] == identity.SEED42_SOURCE_SHA
    assert p["checkpoint_transition"] == 250000
    assert p["artifact_path"].endswith(".csv.gz")


@needs_artifacts
def test_recorded_steps_preserve_the_recorded_sequence():
    for scenario, horizon in identity.SCENARIO_STEPS.items():
        ep = replay.load_episode("root42_masked_bandit", scenario, 1001)
        steps = ep["steps"]
        assert len(steps) == horizon, scenario
        assert [s["step_index"] for s in steps] == list(range(horizon))
        assert all(0 <= s["action"] < identity.ACTION_COUNT for s in steps)


@needs_artifacts
def test_recorded_rewards_reproduce_the_episode_return():
    """The replay must reconcile with the frozen episode summary, not approximate it."""
    ep = replay.load_episode("root42_masked_bandit", "link_failure", 1001)
    total = sum(s["reward"] for s in ep["steps"])
    assert total == pytest.approx(ep["provenance"]["operational_return"], rel=1e-9)


@needs_artifacts
def test_recorded_reward_components_sum_to_each_step_reward():
    ep = replay.load_episode("root42_masked_bandit", "link_failure", 1001)
    for s in ep["steps"][:20]:
        assert set(s["components"]) == set(identity.REWARD_COMPONENTS)
        assert sum(s["components"].values()) == pytest.approx(s["reward"], abs=1e-9)


@needs_artifacts
def test_baseline_episodes_replay_too_so_comparison_is_possible():
    ep = replay.load_episode("baseline_greedy", "link_failure", 1001)
    assert ep["provenance"]["algorithm"] == "greedy"
    assert ep["provenance"]["training_root"] is None
    assert ep["provenance"]["checkpoint_transition"] is None
    assert len(ep["steps"]) == identity.SCENARIO_STEPS["link_failure"]


@needs_artifacts
def test_replay_opens_nothing_for_writing(monkeypatch):
    real_open = builtins.open

    def guard(file, mode="r", *a, **kw):
        if not hasattr(file, "read") and any(m in mode for m in "wxa+"):
            raise AssertionError(f"replay attempted a write: {file} ({mode})")
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guard)
    replay.load_episode("root42_masked_bandit", "link_failure", 1001)
    replay.load_episode("baseline_cspf", "full_day", 1005)


@needs_artifacts
def test_replay_never_imports_a_learner():
    """Loading a checkpoint would be new evaluation. The module must not be able to."""
    import sys
    for banned in ("stable_baselines3", "sb3_contrib", "mplssim.rl.train_v2"):
        sys.modules.pop(banned, None)
    replay.load_episode("root42_maskable_ppo", "overload_stress", 1003)
    assert "stable_baselines3" not in sys.modules
    assert "sb3_contrib" not in sys.modules
