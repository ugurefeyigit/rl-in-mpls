"""V2 delay/loss/SLA severity and the operational reward.

Covers layers 7 and 10 of docs/RL_ENVIRONMENT_V2_TEST_PLAN.md.

Calibration note
----------------
The spec publishes six calibration rewards but only publishes the *input state*
for the first two ("healthy"). Rows 1-2 and the whole route-cost table are
therefore asserted as exact golden values. For rows 3-6 the spec's "Preferred"
column — which is what the pre-training gate actually requires — is asserted
against explicitly constructed controlled states, and those states are written
out to results/environment_v2_validation/reward_calibration.csv so the owner can
compare them against whatever states produced the published numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mplssim.experiments.v2_factory import make_engine_v2, make_env_v2
from mplssim.rl.reward_v2 import (
    COMPONENT_ORDER,
    RewardConfigError,
    compute_reward_v2,
    components_sum,
    g,
    h,
    interval_metrics_from_ticks,
    load_reward_config_v2,
    move_cost,
    normalized_delay_excess,
    normalized_loss_excess,
    potential,
    sat,
    shaping,
    utility,
)
from mplssim.sim import models as m
from mplssim.sim.engine_v2 import loss_curve

CFG = load_reward_config_v2()
N_DEMANDS = 17


def state(delivered=1.0, prot=0.0, unprot=0.0, sla=0.0, util=0.0, overload=0.0):
    return {
        "delivered_ratio": delivered,
        "protected_disconnect": prot,
        "unprotected_disconnect": unprot,
        "sla_severity": sla,
        "max_util": util,
        "overload_ratio": overload,
    }


def reward_for(interval, before=None, **move):
    """Scalar reward with the potential evaluated from `before` to `interval`."""
    before = before if before is not None else interval
    return compute_reward_v2(interval,
                             potential(utility(before, CFG), CFG),
                             potential(utility(interval, CFG), CFG),
                             cfg=CFG, **move)[0]


# ================================================ 7. delay, loss, SLA, severity
DOCUMENTED_CURVES = [
    # utilization, queue delay ms, per-link loss fraction
    (0.00, 0.0, 0.000000),
    (0.50, 1.5, 0.000000),
    (0.80, 6.0, 0.000000),
    (0.90, 13.5, 0.000000),
    (0.95, 28.5, 0.005000),
    (1.00, 60.0, 0.020000),
    (1.10, 60.0, 0.109091),
    (1.50, 60.0, 0.346667),
    (2.00, 60.0, 0.510000),
]


@pytest.mark.parametrize("util,qdelay,loss", DOCUMENTED_CURVES)
def test_documented_delay_and_loss_table(util, qdelay, loss):
    arr = np.array([util])
    assert float(m.queue_delay_ms(arr)[0]) == pytest.approx(qdelay, abs=1e-9)
    assert float(loss_curve(arr)[0]) == pytest.approx(loss, abs=1e-6)


def test_loss_curve_returns_floats_for_integer_like_inputs():
    """V1's loss_fraction allocates via zeros_like, so an int array truncates."""
    ints = np.array([1, 2], dtype=np.int64)
    out = loss_curve(ints)
    assert out.dtype.kind == "f"
    assert float(out[0]) == pytest.approx(0.02, abs=1e-9)
    assert float(out[1]) == pytest.approx(0.51, abs=1e-9)
    assert np.all(m.loss_fraction(ints) == 0)      # the documented V1 caveat


def test_delay_and_loss_are_monotone_on_a_dense_grid():
    grid = np.linspace(0.0, 3.0, 3001)
    q = m.queue_delay_ms(grid)
    l = loss_curve(grid)
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(np.diff(l) >= -1e-12)


def test_loss_is_continuous_at_the_onset_and_at_capacity():
    eps = 1e-9
    for point in (0.90, 1.00):
        lo = float(loss_curve(np.array([point - eps]))[0])
        hi = float(loss_curve(np.array([point + eps]))[0])
        assert abs(hi - lo) < 1e-7


def test_end_to_end_delay_sums_propagation_queue_and_processing_exactly():
    eng = make_engine_v2("evening_peak", episode_seed=101)
    eng.step_interval()
    pad, mask = eng._current_pad_mask()
    for d in range(eng.n_demands):
        hops = int(mask[d].sum())
        if hops == 0:
            continue
        links = pad[d, :hops]
        expected = float(np.sum(eng.prop_delay[links] + eng.link_qdelay[links])
                         + m.PROC_DELAY_MS * hops)
        assert eng.demand_delay[d] == pytest.approx(expected, rel=1e-12)


def test_end_to_end_survival_equals_sequential_hop_survival():
    eng = make_engine_v2("overload_stress", episode_seed=103)
    for _ in range(10):
        eng.step_interval()
    pad, mask = eng._current_pad_mask()
    for d in range(eng.n_demands):
        hops = int(mask[d].sum())
        if hops == 0:
            continue
        survival = 1.0
        for hop in range(hops):
            survival *= (1.0 - eng.link_loss[int(pad[d, hop])])
        assert eng.demand_survival[d] == pytest.approx(survival, rel=1e-12)
        assert eng.demand_loss_fraction[d] == pytest.approx(1.0 - survival, abs=1e-12)


def test_sla_equality_is_success_and_any_positive_excess_is_a_violation():
    eng = make_engine_v2("full_day", episode_seed=101)
    eng.disconnected[:] = False
    eng.demand_delay = eng._delay_sla.copy()
    eng.demand_loss_fraction = eng._loss_sla.copy()
    eng.demand_sla_ok = (~eng.disconnected
                         & (eng.demand_delay <= eng._delay_sla)
                         & (eng.demand_loss_fraction <= eng._loss_sla))
    assert np.all(eng.demand_sla_ok)
    assert eng.tick_metrics()["sla_severity_sum"] == 0.0
    eng.demand_delay = eng._delay_sla * 1.0000001
    eng.demand_sla_ok = (eng.demand_delay <= eng._delay_sla)
    assert not np.any(eng.demand_sla_ok)
    assert eng.tick_metrics()["sla_severity_sum"] > 0.0


def test_normalized_excess_and_h_are_zero_at_or_below_sla_and_monotone_above():
    assert normalized_delay_excess(50.0, 60.0) == 0.0
    assert normalized_delay_excess(60.0, 60.0) == 0.0
    assert normalized_delay_excess(90.0, 60.0) == pytest.approx(0.5)
    assert normalized_loss_excess(0.001, 0.005) == 0.0
    assert normalized_loss_excess(0.005, 0.005) == 0.0
    assert normalized_loss_excess(0.010, 0.005) == pytest.approx(1.0)
    assert h(0.0) == 0.0
    xs = np.linspace(0, 50, 500)
    hs = [h(float(x)) for x in xs]
    assert all(b >= a for a, b in zip(hs, hs[1:]))
    assert h(1.0) == pytest.approx(0.5)
    assert h(1e9) < 1.0


def test_sat_is_half_at_the_reference_boundary_and_never_clips():
    assert sat(1.0) == pytest.approx(0.5)
    assert sat(2.0) < sat(4.0) < sat(8.0) < 1.0


def test_a_severe_violation_costs_more_than_a_mild_one():
    mild = state(sla=0.05)
    severe = state(sla=0.50)
    assert utility(severe, CFG) < utility(mild, CFG)
    assert reward_for(severe) < reward_for(mild)


def test_disconnected_traffic_is_excluded_from_sla_severity():
    """Connectivity owns the penalty; it must not also inflate SLA severity."""
    eng = make_engine_v2("full_day", episode_seed=101)
    eng.disconnected[:] = False
    eng.demand_delay = eng._delay_sla * 10.0
    eng.demand_loss_fraction = np.minimum(eng._loss_sla * 10.0, 1.0)
    connected_sum = eng.tick_metrics()["sla_severity_sum"]
    assert connected_sum > 0.0
    eng.disconnected[:] = True
    metrics = eng.tick_metrics()
    assert metrics["sla_severity_sum"] == 0.0
    assert metrics["protected_disconnect"] == pytest.approx(1.0)
    assert metrics["unprotected_disconnect"] == pytest.approx(1.0)


# ================================================ 10. reward arithmetic/ordering
def test_all_twelve_components_are_present_and_sum_exactly():
    e = make_env_v2(scenario="link_failure", root_seed=101)
    e.reset(options={"episode_seed": 101})
    rng = np.random.default_rng(1)
    for _ in range(60):
        legal = np.flatnonzero(e.action_masks())
        _, reward, _, truncated, info = e.step(int(rng.choice(legal)))
        comp = info["reward_components"]
        assert tuple(comp) == COMPONENT_ORDER
        assert len(comp) == 12
        assert components_sum(comp) == reward       # bit-for-bit, not approx
        if truncated:
            break


def test_component_sum_is_bit_exact_on_synthetic_states():
    rng = np.random.default_rng(9)
    for _ in range(200):
        interval = state(delivered=float(rng.uniform(0, 1)),
                         prot=float(rng.uniform(0, 1)),
                         unprot=float(rng.uniform(0, 1)),
                         sla=float(rng.uniform(0, 1)),
                         util=float(rng.uniform(0, 4)),
                         overload=float(rng.uniform(0, 1)))
        reward, comp = compute_reward_v2(
            interval, float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
            accepted=bool(rng.integers(2)), volume_share=float(rng.uniform(0, 1)),
            edge_divergence=float(rng.uniform(0, 1)),
            reversal=bool(rng.integers(2)), rejected=False, cfg=CFG)
        assert components_sum(comp) == reward


def test_noop_has_zero_route_cost():
    _, comp = compute_reward_v2(state(), 0.0, 0.0, cfg=CFG)
    for key in ("move_fixed", "move_volume", "move_divergence", "reversal", "invalid"):
        assert comp[key] == 0.0


def test_rejected_action_costs_only_the_invalid_charge():
    _, comp = compute_reward_v2(state(), 0.0, 0.0, rejected=True, cfg=CFG)
    assert comp["invalid"] == pytest.approx(-0.05)
    for key in ("move_fixed", "move_volume", "move_divergence", "reversal"):
        assert comp[key] == 0.0


def test_frr_and_recovery_never_incur_te_cost():
    """A failure interval with FRR but no agent action must show zero move cost."""
    from tests.test_transition_v2 import synth_scenario
    from mplssim.factory import get_scenarios
    e = make_env_v2(scenario="link_failure", root_seed=101)
    e.reset(options={"episode_seed": 101})
    saw_frr = False
    while True:
        _, _, _, truncated, info = e.step(0)
        if info["frr_changes"] > 0 or info["recovery_restorations"] > 0:
            saw_frr = True
            comp = info["reward_components"]
            assert comp["move_fixed"] == 0.0
            assert comp["move_volume"] == 0.0
            assert comp["move_divergence"] == 0.0
            assert comp["reversal"] == 0.0
            assert comp["invalid"] == 0.0
            assert info["accepted_te_changes"] == 0
        if truncated:
            break
    assert saw_frr


def test_max_util_penalty_is_free_to_70pct_then_monotone_and_unsaturated():
    assert g(0.0) == 0.0
    assert g(0.50) == 0.0
    assert g(0.70) == 0.0
    for a, b in zip([0.70, 0.80, 0.90, 1.00, 1.10, 1.50, 2.00],
                    [0.80, 0.90, 1.00, 1.10, 1.50, 2.00, 4.00]):
        assert g(b) > g(a)
    # documented values
    for util, want in [(0.80, 0.288), (0.90, 0.511), (1.00, 0.693),
                       (1.10, 0.847), (1.50, 1.299), (2.00, 1.674)]:
        assert g(util) == pytest.approx(want, abs=5e-4)


def test_200pct_utilization_is_worse_than_150_110_100_and_90():
    rewards = [reward_for(state(util=u)) for u in (0.90, 1.00, 1.10, 1.50, 2.00)]
    assert rewards == sorted(rewards, reverse=True)
    assert rewards[-1] < rewards[-2]        # V1 clipped these to the same value


def test_overload_penalty_is_linear_and_unsaturated():
    _, a = compute_reward_v2(state(overload=0.10), 0.0, 0.0, cfg=CFG)
    _, b = compute_reward_v2(state(overload=0.20), 0.0, 0.0, cfg=CFG)
    _, c = compute_reward_v2(state(overload=0.30), 0.0, 0.0, cfg=CFG)
    # equal increments in overload produce equal increments in penalty
    assert b["overload"] - a["overload"] == pytest.approx(c["overload"] - b["overload"],
                                                          rel=1e-12)
    assert c["overload"] == pytest.approx(-6.0 * 0.30)
    # and it never saturates
    _, far = compute_reward_v2(state(overload=3.0), 0.0, 0.0, cfg=CFG)
    assert far["overload"] == pytest.approx(-6.0 * 3.0)


def test_larger_volume_divergence_and_reversal_all_cost_more():
    base = move_cost(0.01, 0.2, False, CFG)
    assert move_cost(0.25, 0.2, False, CFG) > base           # more volume
    assert move_cost(0.01, 1.0, False, CFG) > base           # more divergence
    assert move_cost(0.01, 0.2, True, CFG) > base            # reversal
    assert move_cost(0.01, 0.2, True, CFG) == pytest.approx(base + 0.30)


def test_the_published_route_cost_table_is_reproduced_exactly():
    assert move_cost(0.01, 0.2, False, CFG) == pytest.approx(0.107, abs=1e-9)
    assert move_cost(0.25, 0.2, False, CFG) == pytest.approx(0.179, abs=1e-9)
    assert move_cost(0.01, 1.0, False, CFG) == pytest.approx(0.203, abs=1e-9)
    assert move_cost(0.01, 0.2, True, CFG) == pytest.approx(0.407, abs=1e-9)
    assert CFG.invalid == pytest.approx(0.050, abs=1e-9)


def test_calibration_row_1_healthy_no_action():
    healthy = state()
    assert utility(healthy, CFG) == pytest.approx(2.0, abs=1e-12)
    assert reward_for(healthy) == pytest.approx(1.9998, abs=5e-5)


def test_calibration_row_2_healthy_unnecessary_move():
    healthy = state()
    noop = reward_for(healthy)
    action = reward_for(healthy, accepted=True, volume_share=0.05,
                        edge_divergence=0.5)
    assert noop == pytest.approx(1.9998, abs=5e-5)
    assert action == pytest.approx(1.8448, abs=5e-5)
    assert action < noop, "an unnecessary reroute must be worse than a healthy no-op"


def test_healthy_low_load_state_favours_no_op():
    healthy = state(util=0.3)
    noop = reward_for(healthy)
    for share in (0.01, 0.10, 0.25):
        for div in (0.0, 0.5, 1.0):
            assert reward_for(healthy, accepted=True, volume_share=share,
                              edge_divergence=div) < noop


def test_mild_congestion_materially_improved_favours_the_reroute():
    bad = state(delivered=0.99, sla=0.05, util=0.95)
    improved = state(delivered=1.0, util=0.75)
    noop = reward_for(bad)
    action = reward_for(improved, before=bad, accepted=True,
                        volume_share=0.05, edge_divergence=0.3)
    assert action > noop


def test_severe_overload_materially_improved_favours_the_reroute():
    bad = state(delivered=0.60, sla=0.60, util=2.00, overload=0.25)
    improved = state(delivered=0.95, sla=0.15, util=1.05, overload=0.02)
    noop = reward_for(bad)
    action = reward_for(improved, before=bad, accepted=True,
                        volume_share=0.25, edge_divergence=1.0)
    assert action > noop


def test_utilization_gain_with_critical_sla_worsening_favours_no_op():
    bad = state(sla=0.02, util=0.95)
    traded = state(sla=0.30, util=0.80)
    assert reward_for(traded, before=bad, accepted=True, volume_share=0.05,
                      edge_divergence=0.3) < reward_for(bad)


def test_protected_disconnection_dominates_an_ordinary_congestion_gain():
    """Disconnecting one critical protected demand must never pay."""
    congested = state(delivered=0.90, sla=0.30, util=1.30, overload=0.10)
    # protected q-sum is 3*5/6 + 1 = 3.5; dropping the critical demand gives 1/3.5
    gain_by_disconnect = state(delivered=0.996, prot=1.0 / 3.5, util=0.70)
    assert reward_for(gain_by_disconnect, before=congested, accepted=True,
                      volume_share=0.01, edge_divergence=0.2) < reward_for(congested)


def test_the_protected_weight_dominates_every_other_single_term():
    assert CFG.protected_disconnect > (CFG.delivered + CFG.unprotected_disconnect
                                       + CFG.sla_severity + CFG.overload)


def test_immediate_reversal_costs_more_than_stable_routing():
    s = state(util=0.5)
    stable = reward_for(s, accepted=True, volume_share=0.01, edge_divergence=0.2)
    reverted = reward_for(s, accepted=True, volume_share=0.01, edge_divergence=0.2,
                          reversal=True)
    assert reverted == pytest.approx(stable - 0.30, abs=1e-12)
    assert reverted < stable < reward_for(s)


def test_unavoidable_overload_remains_bad_under_no_op():
    overloaded = state(delivered=0.80, sla=0.40, util=1.60, overload=0.20)
    assert reward_for(overloaded) < reward_for(state(util=0.5))
    assert reward_for(overloaded) < 0.0


def _congested_engine():
    eng = make_engine_v2("overload_stress", episode_seed=103)
    for _ in range(20):
        eng.step_interval()
    assert eng.tick_metrics()["max_util"] > 1.0, "expected a congested tick"
    return eng


def _utility_if_disconnected(eng, victim):
    """Utility of the same tick with one demand forced offline.

    ``_compute_tick`` is re-run on the clone rather than patching telemetry by
    hand, so offered traffic, the carried-flow solution, per-demand delay and
    SLA severity are all mutually consistent. The AR state is copied, so the
    offered vector is identical and the only difference is the disconnection.
    """
    shed = eng.fast_clone()
    shed.disconnected[victim] = True
    shed._compute_tick()
    return utility(shed.boundary_metrics(), CFG), shed


def test_disconnecting_protected_traffic_never_pays_even_under_severe_overload():
    """The safety property: protected connectivity outranks any congestion gain."""
    eng = _congested_engine()
    carried = utility(eng.boundary_metrics(), CFG)
    for victim in eng._protected_idx:
        dropped, _ = _utility_if_disconnected(eng, int(victim))
        assert dropped < carried, eng.demands[int(victim)].id


def test_disconnection_cost_is_priority_ordered():
    """Protected beats unprotected, and within a set, higher priority costs more."""
    eng = _congested_engine()
    penalties = {}
    for d_idx in range(eng.n_demands):
        shed = eng.fast_clone()
        shed.disconnected[:] = False
        shed.disconnected[d_idx] = True
        tick = shed.tick_metrics()
        penalties[d_idx] = (CFG.protected_disconnect * tick["protected_disconnect"]
                            + CFG.unprotected_disconnect * tick["unprotected_disconnect"])
    worst_unprotected = max(penalties[i] for i in np.flatnonzero(~eng._protected))
    best_protected = min(penalties[int(i)] for i in eng._protected_idx)
    assert best_protected > worst_unprotected, (
        "dropping any protected demand must cost more than dropping any "
        "unprotected one")
    for pool in (np.flatnonzero(~eng._protected), eng._protected_idx):
        ranked = sorted((eng._priorities[i], penalties[int(i)]) for i in pool)
        assert [p for _, p in ranked] == sorted(p for _, p in ranked)


def test_disconnection_is_never_free():
    eng = _congested_engine()
    for d_idx in range(eng.n_demands):
        shed = eng.fast_clone()
        shed.disconnected[:] = False
        shed.disconnected[d_idx] = True
        tick = shed.tick_metrics()
        assert (tick["protected_disconnect"] + tick["unprotected_disconnect"]) > 0.0


def test_no_agent_action_can_disconnect_a_demand():
    """Closes the 'shed traffic to relieve congestion' loophole structurally.

    An accepted TE move requires every link on the target candidate to be up,
    so it can never strand a demand; disconnection is reachable only through a
    topology failure, which is an environment transition and not an action.
    """
    e = make_env_v2(scenario="overload_stress", root_seed=103)
    e.reset(options={"episode_seed": 103})
    rng = np.random.default_rng(6)
    while True:
        before = e.eng.disconnected.copy()
        legal = np.flatnonzero(e.action_masks())
        _, _, _, truncated, info = e.step(int(rng.choice(legal)))
        if info["frr_disconnections"] == 0:
            assert int(e.eng.disconnected.sum()) <= int(before.sum())
        if truncated:
            break
    assert e.eng.episode_totals["frr_disconnections"] == 0   # no failures here


def test_potential_term_stays_within_its_analytical_bound():
    bound = CFG.potential_coefficient * (CFG.potential_gamma * 1.0 + 1.0)
    assert bound == pytest.approx(0.399)
    rng = np.random.default_rng(4)
    for _ in range(1000):
        f = shaping(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), CFG)
        assert -bound <= f <= bound
    for _ in range(200):
        u = float(rng.uniform(-60, 10))
        assert -1.0 <= potential(u, CFG) <= 1.0


def test_potential_is_a_true_state_function():
    """Phi depends only on the state, so shaping cannot change the optimum."""
    s = state(delivered=0.9, util=1.2)
    assert potential(utility(s, CFG), CFG) == potential(utility(dict(s), CFG), CFG)


def test_potential_coefficient_zero_is_an_ablation_not_the_primary_reward():
    from dataclasses import replace
    assert CFG.potential_coefficient == 0.20
    ablation = replace(CFG, potential_coefficient=0.0)
    assert shaping(0.5, -0.5, ablation) == 0.0
    assert shaping(0.5, -0.5, CFG) != 0.0


def test_engine_and_reference_interval_aggregation_agree():
    eng = make_engine_v2("overload_stress", episode_seed=103)
    n = eng.cfg.micro_ticks_per_interval
    ticks = []
    for _ in range(n):
        old_t = eng.t_min
        eng.t_min = old_t + 1.0
        eng.traffic.advance_noise()
        eng._process_link_events(old_t, eng.t_min)
        ticks.append(eng._compute_tick())
    agg = eng.aggregate_interval(ticks)
    ref = interval_metrics_from_ticks(ticks, n, eng._q_sum)
    for key, value in ref.items():
        assert agg[key] == pytest.approx(value, rel=1e-15), key


def test_utility_rejects_incomplete_metrics():
    with pytest.raises(KeyError):
        utility({"delivered_ratio": 1.0}, CFG)


def test_reward_config_is_the_required_version_and_order():
    assert CFG.version == "reward-v2.0-operational"
    assert COMPONENT_ORDER == (
        "delivery", "protected_disconnect", "unprotected_disconnect",
        "sla_severity", "max_util", "overload", "potential", "move_fixed",
        "move_volume", "move_divergence", "reversal", "invalid")
    assert (CFG.delivered, CFG.protected_disconnect, CFG.unprotected_disconnect,
            CFG.sla_severity, CFG.max_util, CFG.overload) == (2.0, 30.0, 8.0, 6.0,
                                                              2.0, 6.0)
    assert (CFG.move_fixed, CFG.move_volume_share, CFG.move_edge_divergence,
            CFG.move_reversal, CFG.invalid) == (0.08, 0.30, 0.12, 0.30, 0.05)
    assert (CFG.potential_coefficient, CFG.potential_gamma,
            CFG.potential_scale) == (0.20, 0.995, 10.0)


def test_reward_config_fails_closed_on_a_wrong_version(tmp_path):
    import yaml

    from mplssim.rl.reward_v2 import REWARD_CONFIG_PATH
    raw = yaml.safe_load(REWARD_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["version"] = "reward-v1"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    load_reward_config_v2.cache_clear()
    try:
        with pytest.raises(RewardConfigError, match="reward version"):
            load_reward_config_v2(bad)
    finally:
        load_reward_config_v2.cache_clear()


def test_protected_and_unprotected_weights_use_the_real_demand_mix():
    """q-sums used by the calibration cases match the shipped traffic classes."""
    eng = make_engine_v2("full_day", episode_seed=101)
    assert eng._q_protected_sum == pytest.approx(3 * (5 / 6) + 1.0)
    assert [eng.demands[i].id for i in eng._protected_idx] == ["D1", "D11", "D14", "D17"]
    assert eng._q_sum == pytest.approx(float(np.sum(eng._priorities / 6.0)))
