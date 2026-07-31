"""Every scientific number the product shows must reconcile with the frozen files.

These tests are the reconciliation gate. They recompute the study's published claims
from the committed CSV/JSON and refuse any drift, including drift that would round
into a different conclusion.
"""

from __future__ import annotations

import pytest

from mplssim.evidence import claims, identity, loader

ROOT = loader.default_root()


@pytest.fixture(scope="module")
def fh():
    return loader.FinalHoldout.load(ROOT)


@pytest.fixture(scope="module")
def cont():
    return loader.Continuity.load(ROOT)


# ------------------------------------------------------------ headline claims
def test_headline_numbers_reproduce_from_frozen_evidence(fh):
    c = claims.learner_comparison(fh)
    assert round(c["bandit_return"], 3) == 18.221
    assert round(c["ppo_return"], 3) == 9.036
    assert round(c["advantage"], 3) == 9.185
    assert c["roots_won"] == 3
    assert c["roots_total"] == 3
    assert len(c["roots"]) == 3


def test_greedy_is_the_strongest_baseline_at_minus_2_327(fh):
    rows = claims.aggregate_table(fh)
    by_algo = {r["algorithm"]: r for r in rows}
    assert round(by_algo["greedy"]["operational_return_mean"], 3) == -2.327
    baselines = [by_algo[a]["operational_return_mean"]
                 for a in identity.BASELINE_ALGORITHMS]
    assert by_algo["greedy"]["operational_return_mean"] == max(baselines)


def test_every_policy_contributed_exactly_thirty_five_episodes(fh):
    counts = claims.episode_accounting(fh)
    assert counts["total"] == 315
    assert counts["per_policy"] == 35
    assert counts["policies"] == 9
    assert counts["learner_checkpoints"] == 6
    assert counts["baselines"] == 3
    assert counts["ran_once"] is True


# --------------------------------------------------- root-aware, not pooled
def test_aggregate_is_root_aware_not_episode_pooled(fh):
    """The aggregate must equal the mean of the three ROOT means. Pooling the 105
    episodes directly treats episodes as independent training roots, which they
    are not."""
    for algo, expected in (("masked_bandit", 18.220918), ("maskable_ppo", 9.035842)):
        rows = fh.per_root[(fh.per_root.algorithm == algo)
                           & (fh.per_root.training_root.astype(str) != "baseline")]
        assert len(rows) == 3
        agg = claims.root_aggregate(fh.per_root, algo)
        assert agg["operational_return_mean"] == pytest.approx(
            rows.operational_return_mean.mean(), abs=1e-12)
        assert agg["operational_return_mean"] == pytest.approx(expected, abs=1e-6)
        assert agg["root_mean_std"] == pytest.approx(
            rows.operational_return_mean.std(ddof=1), abs=1e-12)
        assert agg["root_count"] == 3


def test_baselines_are_evaluated_once_and_carry_no_root_spread(fh):
    for algo in identity.BASELINE_ALGORITHMS:
        agg = claims.root_aggregate(fh.per_root, algo)
        assert agg["root_count"] == 1
        assert agg["root_mean_std"] == 0.0
        assert agg["episodes"] == 35


def test_scenario_grain_rolls_up_to_the_root_grain(fh):
    """Each root return must be the mean of its seven scenario means."""
    for pid in fh.per_root["policy_id"]:
        rows = fh.scenario[fh.scenario.policy_id == pid]
        assert len(rows) == 7
        expected = float(fh.per_root[fh.per_root.policy_id == pid]
                         .operational_return_mean.iloc[0])
        assert rows.operational_return_mean.mean() == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------- scenario comparison
def test_scenario_comparison_is_six_of_seven_with_the_one_ppo_win(fh):
    rows = claims.scenario_comparison(fh)
    assert len(rows) == 7
    assert sum(r["winner"] == "masked_bandit" for r in rows) == 6
    ppo_wins = [r for r in rows if r["winner"] == "maskable_ppo"]
    assert [r["scenario"] for r in ppo_wins] == ["deceptive_local_optimum"]
    assert round(-ppo_wins[0]["advantage"], 3) == 1.107
    assert round(max(r["advantage"] for r in rows), 3) == 20.183
    assert max(rows, key=lambda r: r["advantage"])["scenario"] == "link_failure"


def test_scenario_comparison_averages_roots_before_comparing(fh):
    """A scenario winner decided by pooling 15 episodes would be a different
    statistic from one decided by averaging three root means."""
    rows = {r["scenario"]: r for r in claims.scenario_comparison(fh)}
    for scen, row in rows.items():
        per_root = fh.scenario[(fh.scenario.scenario == scen)
                               & (fh.scenario.algorithm == "masked_bandit")]
        assert len(per_root) == 3
        assert row["bandit"] == pytest.approx(
            per_root.operational_return_mean.mean(), abs=1e-12)
        assert row["root_count"] == 3


# ------------------------------------------------------------ reward integrity
def test_reward_components_sum_exactly_to_the_operational_return(fh):
    rec = claims.reward_reconciliation(fh)
    assert len(rec["rows"]) == 9
    assert all(len(r["components"]) == 12 for r in rec["rows"])
    assert set(rec["rows"][0]["components"]) == set(identity.REWARD_COMPONENTS)
    assert rec["max_residual"] < 1e-9
    for r in rec["rows"]:
        assert r["sum"] == pytest.approx(r["operational_return_mean"], abs=1e-9)


def test_reported_max_aggregation_residual_is_preserved(fh):
    rec = claims.reward_reconciliation(fh)
    assert f"{rec['reported_max_abs_residual']:.4e}" == "1.7053e-13"


# ---------------------------------------------- two grains that look alike
def test_noop_shares_expose_both_grains_separately(fh):
    """`action_distribution.csv` pools 3,300 steps; `aggregate_metrics.csv` averages
    per-episode frequencies. Both are correct; presenting one as the other is not."""
    n = claims.noop_shares(fh)
    assert round(n["pooled_step_share"]["masked_bandit"] * 100, 2) == 87.09
    assert round(n["pooled_step_share"]["maskable_ppo"] * 100, 2) == 87.31
    assert round(n["episode_mean_share"]["masked_bandit"] * 100, 2) == 82.10
    assert round(n["episode_mean_share"]["maskable_ppo"] * 100, 2) == 82.10
    assert n["pooled_step_share"] != n["episode_mean_share"]
    assert n["pooled_grain"] and n["episode_grain"]


def test_runtime_keeps_runner_total_separate_from_checkpoint_sum(fh):
    """152.093 s is the whole-runner wall time including baselines and setup. The six
    learner evaluations sum to 115.213 s. Neither may stand in for the other."""
    r = claims.runtime_summary(fh)
    assert round(r["total_runner_wall_seconds"], 3) == 152.093
    assert round(r["checkpoint_wall_seconds_sum"], 3) == 115.213
    assert r["total_runner_wall_seconds"] != r["checkpoint_wall_seconds_sum"]
    assert r["device"] == "cuda:0"
    assert r["peak_gpu_memory_bytes_min"] == 13386752
    assert r["peak_gpu_memory_bytes_max"] == 16926720


# --------------------------------------------------------- churn and safety
def test_churn_summary_preserves_the_full_uncomfortable_picture(fh):
    c = claims.churn_summary(fh)
    b, p, g = c["masked_bandit"], c["maskable_ppo"], c["greedy"]
    assert round(b["reroutes_per_hour"], 3) == 2.148
    assert round(p["reroutes_per_hour"], 3) == 2.148
    assert b["te_reversals"] < p["te_reversals"]
    assert b["flaps_per_demand"] < p["flaps_per_demand"]
    # the bandit's cost: it moves MORE bandwidth than PPO, and this must not be hidden
    assert b["moved_mbps_total"] > p["moved_mbps_total"]
    assert b["moved_mbps_total"] < g["moved_mbps_total"]


def test_safety_summary_reports_all_checks_passed(fh):
    s = claims.safety_summary(fh)
    assert s["all_checks_passed"] is True
    assert s["policies"] == 9
    assert all(v == 0 for v in s["counters"].values())
    assert s["protected_disconnection_identical_across_methods"] is True
    assert s["unprotected_disconnection_identical_across_methods"] is True
    assert s["rejected_te_requests_total"] == 0


# ----------------------------------------------------------- stage separation
def test_development_and_holdout_summaries_are_distinct_stages(fh, cont):
    dev = claims.development_summary(cont)
    hold = claims.holdout_summary(fh)
    assert dev["stage"] == identity.STAGE_DEVELOPMENT
    assert hold["stage"] == identity.STAGE_FINAL_HOLDOUT
    assert dev["holdout_accessed"] is False
    # the development numbers are genuinely different figures, not the holdout's
    assert round(dev["bandit_return"], 3) != round(hold["comparison"]["bandit_return"], 3)


def test_learning_curves_are_development_only(cont):
    curves = claims.learning_curves(cont)
    assert curves["stage"] == identity.STAGE_DEVELOPMENT
    assert len(curves["series"]) == 6          # 3 roots x 2 learners
    assert all(len(s["points"]) == 8 for s in curves["series"])
    assert all(s["selected_transition"] for s in curves["series"])
    assert "not" in curves["caption"].lower() and "holdout" in curves["caption"].lower()


# ---------------------------------------------------------------- conclusions
def test_conclusions_state_both_halves_of_the_planning_claim():
    joined = " ".join(claims.CONCLUSIONS).lower()
    assert "does not positively support" in joined
    assert "temporal planning" in joined
    assert "not evidence that planning is generally irrelevant" in joined
    assert "exactly once" in joined
    assert "315" in joined and "35" in joined


def test_conclusions_deny_any_holdout_driven_selection():
    joined = " ".join(claims.CONCLUSIONS).lower()
    for word in ("training", "tuning", "sweep", "reselection", "redesign"):
        assert word in joined
