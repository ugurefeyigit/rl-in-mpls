"""V2 transition: event timing, flow conservation, TE/FRR history, aggregation.

Covers layers 4, 6, 9 and 11 of docs/RL_ENVIRONMENT_V2_TEST_PLAN.md.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mplssim.experiments.v2_factory import make_engine_v2, make_env_v2
from mplssim.factory import get_topology, get_traffic_config
from mplssim.sim.engine_v2 import (
    FlowSolverConfig,
    FlowSolverError,
    SimulationEngineV2,
    load_engine_config_v2,
    loss_curve,
)
from mplssim.traffic.model import ScenarioSpec

K = 4
N_DEMANDS = 17
N_DLINKS = 64

#: Every candidate of D1, D4, D9 and D16 terminates on L27 (P8->PE5), so
#: failing L27 disconnects exactly those four demands and nothing else. Used to
#: exercise FRR disconnection and recovery restoration deterministically.
PE5_DEMANDS = ("D1", "D4", "D9", "D16")
PE5_LAST_HOP = "L27"


def synth_scenario(events, duration_min=120, start_hour=12.0,
                   name="synthetic", multiplier=1.0, sigma=0.0) -> ScenarioSpec:
    return ScenarioSpec(name=name, description="synthetic test scenario",
                        start_hour=start_hour, duration_min=duration_min,
                        demand_multiplier=multiplier, noise_sigma=sigma,
                        events=list(events))


def synth_engine(events, seed=101, cfg=None, **kw) -> SimulationEngineV2:
    return SimulationEngineV2(get_topology(), get_traffic_config(),
                              synth_scenario(events, **kw), seed,
                              cfg or load_engine_config_v2())


# ================================================== 4. event timing / leakage
def test_event_at_t0_is_reflected_in_the_reset_observation():
    eng = synth_engine([{"t_min": 0, "type": "link_down", "link": "L11"}])
    assert eng.t_min == 0.0
    assert eng.link_up["L11"] is False
    assert eng.episode_totals["frr_changes"] + eng.episode_totals["frr_disconnections"] > 0
    assert eng.link_input_load[eng.topo.dlink_by_pair[("P2", "P5")].index] == 0.0


def test_failure_at_60_is_invisible_at_55_and_visible_at_60():
    """The P0-1 correction. V1 showed L11 up at 60 and down only at 61."""
    eng = synth_engine([{"t_min": 60, "type": "link_down", "link": "L11"}],
                       duration_min=120)
    seen = {}
    while not eng.done:
        eng.step_interval()
        seen[eng.t_min] = eng.link_up["L11"]
    assert seen[55.0] is True, "a failure at 60 must not be visible at the 55 boundary"
    assert seen[60.0] is False, "a failure at 60 must be visible at the 60 boundary"
    assert seen[65.0] is False


def test_the_event_is_processed_in_the_59_to_60_micro_tick():
    eng = synth_engine([{"t_min": 60, "type": "link_down", "link": "L11"}])
    while eng.t_min < 55.0:
        eng.step_interval()
    assert eng.t_min == 55.0 and eng.link_up["L11"] is True
    states = []
    for _ in range(eng.cfg.micro_ticks_per_interval):
        old_t = eng.t_min
        eng.t_min = old_t + 1.0
        eng.traffic.advance_noise()
        eng._process_link_events(old_t, eng.t_min)
        states.append((eng.t_min, eng.link_up["L11"]))
    assert states == [(56.0, True), (57.0, True), (58.0, True),
                      (59.0, True), (60.0, False)]


def test_an_event_fires_exactly_once():
    eng = synth_engine([{"t_min": 60, "type": "link_down", "link": "L11"}])
    fired = 0
    while not eng.done:
        for _ in range(eng.cfg.micro_ticks_per_interval):
            old_t = eng.t_min
            eng.t_min = old_t + 1.0
            fired += len(eng._selected_link_events(old_t, eng.t_min))
        eng.step_count += 1
    assert fired == 1


def test_the_windows_tile_the_timeline_without_gap_or_overlap():
    """(old,new] over ticks plus the t<=0 reset window covers every instant once."""
    events = [{"t_min": t, "type": "link_down", "link": "L11"} for t in (0, 1, 5, 60)]
    events.append({"t_min": 120, "type": "link_up", "link": "L11"})
    eng = synth_engine(events, duration_min=120)
    fired = len([e for e in eng.scenario.events if e["t_min"] <= 0])
    t = 0.0
    while t < 120.0:
        old_t, t = t, t + 1.0
        fired += len(eng._selected_link_events(old_t, t))
    assert fired == len(events)


def test_recovery_uses_the_same_right_closed_boundary():
    eng = synth_engine([{"t_min": 10, "type": "link_down", "link": "L11"},
                        {"t_min": 60, "type": "link_up", "link": "L11"}])
    seen = {}
    while not eng.done:
        eng.step_interval()
        seen[eng.t_min] = eng.link_up["L11"]
    assert seen[55.0] is False
    assert seen[60.0] is True


def test_link_and_volume_events_at_the_same_instant_use_the_documented_order():
    """Link state (and its FRR) is applied first; offered traffic is then
    computed at the new time including any volume event active there."""
    events = [{"t_min": 60, "type": "link_down", "link": "L11"},
              {"t_min": 60, "type": "burst", "demands": ["D2"],
               "factor": 3.0, "duration_min": 30}]
    eng = synth_engine(events, duration_min=120)
    plain = synth_engine([], duration_min=120)
    while eng.t_min < 60.0:
        eng.step_interval()
        plain.step_interval()
    assert eng.t_min == 60.0
    assert eng.link_up["L11"] is False              # link event applied
    d2 = eng.demand_by_id["D2"].index
    assert eng.demand_offered[d2] == pytest.approx(
        3.0 * plain.demand_offered[d2], rel=1e-12)  # burst active at 60 too


def test_adding_a_purely_future_event_changes_nothing_observable_now():
    a = make_env_v2(scenario="full_day", root_seed=101)
    a.reset(options={"episode_seed": 101})
    b_eng = synth_engine([{"t_min": 600, "type": "link_down", "link": "L11"}],
                         duration_min=1440, start_hour=0.0, sigma=1.0, seed=101)
    c_eng = synth_engine([], duration_min=1440, start_hour=0.0, sigma=1.0, seed=101)
    for _ in range(12):                             # up to t = 60, long before 600
        b_eng.step_interval()
        c_eng.step_interval()
    np.testing.assert_array_equal(b_eng.link_input_load, c_eng.link_input_load)
    np.testing.assert_array_equal(b_eng.demand_offered, c_eng.demand_offered)
    np.testing.assert_array_equal(b_eng.current_path, c_eng.current_path)
    np.testing.assert_array_equal(b_eng.te_action_matrix(), c_eng.te_action_matrix())


def test_identical_present_state_with_different_futures_gives_identical_obs_and_masks():
    def env_for(events):
        e = make_env_v2(scenario="full_day", root_seed=101)
        e.reset(options={"episode_seed": 101})
        e.eng = SimulationEngineV2(
            get_topology(), get_traffic_config(),
            synth_scenario(events, duration_min=1440, start_hour=0.0, sigma=1.0),
            101, e.engine_cfg)
        return e

    a = env_for([{"t_min": 900, "type": "link_down", "link": "L20"}])
    b = env_for([{"t_min": 1200, "type": "link_down", "link": "L13"}])
    for _ in range(10):
        a.eng.step_interval()
        b.eng.step_interval()
    np.testing.assert_array_equal(a._obs(), b._obs())
    np.testing.assert_array_equal(a.action_masks(), b.action_masks())


# =========================================== 6. flow propagation / conservation
def _assert_conservation(eng, atol=0.0):
    """Every identity the test plan requires of a solved tick."""
    pad, mask = eng._current_pad_mask()
    loss, hop_in = eng.link_loss, eng.hop_input

    assert np.all(eng.gross_link_load >= 0.0)
    assert np.all(eng.link_input_load >= 0.0)
    assert np.all(eng.demand_offered >= 0.0)
    assert np.all(eng.demand_delivered >= 0.0)
    assert np.all(hop_in >= 0.0)

    # failed links carry nothing
    assert np.all(eng.link_input_load[~eng._dlink_up] == 0.0)
    assert np.all(eng.gross_link_load[~eng._dlink_up] == 0.0)

    # disconnected demands deliver nothing and load nothing
    for d in np.flatnonzero(eng.disconnected):
        assert eng.demand_delivered[d] == 0.0
        assert np.all(hop_in[d] == 0.0)

    rebuilt = np.zeros(N_DLINKS)
    for d in range(eng.n_demands):
        hops = int(mask[d].sum())
        if hops == 0:
            continue
        assert hop_in[d, 0] == eng.demand_offered[d]          # x[d,0] = offered
        for h in range(hops):
            e = int(pad[d, h])
            rebuilt[e] += hop_in[d, h]
            out = hop_in[d, h] * (1.0 - loss[e])              # link output
            nxt = hop_in[d, h + 1] if h + 1 < hops else eng.demand_delivered[d]
            assert out == pytest.approx(nxt, abs=1e-15, rel=1e-15)
            assert nxt <= hop_in[d, h] + 1e-12                # non-increasing
    np.testing.assert_allclose(rebuilt, eng.link_input_load, rtol=1e-12, atol=1e-9)

    assert float(np.sum(eng.demand_delivered)) <= float(np.sum(eng.demand_offered)) + 1e-9
    # gross is the conservative ledger: it never under-states link input
    assert np.all(eng.gross_link_load >= eng.link_input_load - 1e-9)


@pytest.mark.parametrize("scenario,seed", [
    ("full_day", 101), ("link_failure", 101),
    ("ood_double_failure", 101), ("overload_stress", 103),
])
def test_conservation_holds_at_every_boundary_of_a_shipped_scenario(scenario, seed):
    eng = make_engine_v2(scenario, episode_seed=seed)
    _assert_conservation(eng)
    while not eng.done:
        eng.step_interval()
        _assert_conservation(eng)


def test_zero_loss_regime_reduces_v2_to_the_v1_accounting_rule():
    """With no link above the loss onset, surviving input == full offered load."""
    eng = make_engine_v2("night_consolidation", episode_seed=101)
    checked = 0
    while not eng.done:
        eng.step_interval()
        if np.max(eng.link_util) <= 0.90:
            np.testing.assert_allclose(eng.link_input_load, eng.gross_link_load,
                                       rtol=0, atol=0)
            np.testing.assert_allclose(eng.demand_delivered,
                                       np.where(eng.disconnected, 0.0,
                                                eng.demand_offered), rtol=0, atol=0)
            assert np.all(eng.link_loss == 0.0)
            assert eng.flow_solver_iterations == 1
            checked += 1
    assert checked > 0


def test_downstream_links_only_receive_surviving_traffic():
    """The P1 correction: V1 loaded every hop with the full offered volume."""
    eng = make_engine_v2("overload_stress", episode_seed=103)
    found = False
    while not eng.done and not found:
        eng.step_interval()
        pad, mask = eng._current_pad_mask()
        for d in range(eng.n_demands):
            hops = int(mask[d].sum())
            if hops < 2:
                continue
            if eng.link_loss[int(pad[d, 0])] > 1e-6:
                assert eng.hop_input[d, 1] < eng.hop_input[d, 0]
                # V1 would have put the full offered volume on the second hop
                assert eng.hop_input[d, 1] < eng.demand_offered[d]
                found = True
                break
    assert found, "overload_stress should exercise a lossy first hop"


def test_solver_converges_within_the_configured_cap_on_every_shipped_scenario():
    cfg = load_engine_config_v2()
    for scenario in ("full_day", "evening_peak", "flash_crowd", "link_failure",
                     "deceptive_local_optimum", "ood_double_failure",
                     "overload_stress"):
        for seed in (101, 103):
            eng = make_engine_v2(scenario, episode_seed=seed)
            worst = 0
            while not eng.done:
                agg = eng.step_interval()
                worst = max(worst, agg["flow_solver_iterations_max"])
            assert worst <= cfg.flow_solver.max_iterations, (scenario, seed, worst)


def test_converged_loss_matches_the_curve_within_twice_the_tolerance():
    """The returned link_loss is the propagating vector; its residual against
    loss_curve(util) is bounded by 2*tolerance from the stopping rule."""
    cfg = load_engine_config_v2()
    eng = make_engine_v2("overload_stress", episode_seed=103)
    while not eng.done:
        agg = eng.step_interval()
        residual = float(np.max(np.abs(eng.link_loss - loss_curve(eng.link_util))))
        assert residual <= 2.0 * cfg.flow_solver.tolerance
        assert agg["flow_solver_residual"] <= 2.0 * cfg.flow_solver.tolerance


def test_a_nonconvergent_configuration_fails_the_tick_instead_of_returning_stale():
    """Nonconvergence must raise, and must not publish a partial iterate."""
    eng = make_engine_v2("overload_stress", episode_seed=103)
    eng.step_interval()
    last_good = {
        "link_util": eng.link_util.copy(),
        "link_input_load": eng.link_input_load.copy(),
        "link_loss": eng.link_loss.copy(),
        "demand_delivered": eng.demand_delivered.copy(),
    }
    eng.cfg = replace(eng.cfg, flow_solver=FlowSolverConfig(
        damping=0.5, tolerance=1e-10, max_iterations=2))
    with pytest.raises(FlowSolverError, match="did not converge"):
        eng.step_interval()
    for name, value in last_good.items():
        np.testing.assert_array_equal(getattr(eng, name), value, err_msg=name)


def test_a_nonconvergent_configuration_fails_at_construction_too():
    cfg = replace(load_engine_config_v2(),
                  flow_solver=FlowSolverConfig(damping=0.5, tolerance=1e-10,
                                               max_iterations=2))
    with pytest.raises(FlowSolverError, match="did not converge"):
        make_engine_v2("overload_stress", episode_seed=103, cfg=cfg)


def test_solver_results_are_deterministic_across_repeats_and_clones():
    a = make_engine_v2("flash_crowd", episode_seed=101)
    b = make_engine_v2("flash_crowd", episode_seed=101)
    for _ in range(15):
        a.step_interval()
        b.step_interval()
    np.testing.assert_array_equal(a.link_input_load, b.link_input_load)
    np.testing.assert_array_equal(a.link_loss, b.link_loss)
    np.testing.assert_array_equal(a.demand_delivered, b.demand_delivered)
    c = a.fast_clone()
    for _ in range(5):
        a.step_interval()
        c.step_interval()
    np.testing.assert_array_equal(a.link_input_load, c.link_input_load)


def test_synthetic_shared_merge_and_diverge_cases_conserve_flow():
    """Drive the solver directly on hand-built placements."""
    eng = make_engine_v2("evening_peak", episode_seed=101)
    rng = np.random.default_rng(5)
    for trial in range(25):
        eng.current_path = rng.integers(0, K, size=N_DEMANDS).astype(np.int64)
        eng.disconnected = rng.random(N_DEMANDS) < 0.15
        pad, mask = eng._current_pad_mask()
        offered = np.where(eng.disconnected, 0.0,
                           rng.uniform(50.0, 900.0, size=N_DEMANDS))
        sol = eng.solve_flow(offered, pad, mask)
        assert sol.iterations <= eng.cfg.flow_solver.max_iterations
        rebuilt = np.zeros(N_DLINKS)
        for d in range(N_DEMANDS):
            hops = int(mask[d].sum())
            x = offered[d]
            for h in range(hops):
                e = int(pad[d, h])
                assert sol.hop_input[d, h] == pytest.approx(x, abs=1e-12)
                rebuilt[e] += sol.hop_input[d, h]
                x = x * (1.0 - sol.link_loss[e])
            assert sol.delivered[d] == pytest.approx(x, abs=1e-12)
        np.testing.assert_allclose(rebuilt, sol.link_input_load, rtol=1e-12, atol=1e-9)
        assert np.all(sol.link_input_load >= 0.0)
        assert np.all(sol.gross_link_load + 1e-9 >= sol.link_input_load)


def test_gross_projection_leaves_shared_links_unchanged():
    eng = make_engine_v2("evening_peak", episode_seed=101)
    for _ in range(4):
        eng.step_interval()
    for d_idx in range(N_DEMANDS):
        cur = int(eng.current_path[d_idx])
        cur_links = set(eng._cand_links[d_idx][cur].tolist())
        for p_idx in range(K):
            if p_idx == cur:
                continue
            projected = eng.projected_gross_loads(d_idx, p_idx)
            shared = cur_links & set(eng._cand_links[d_idx][p_idx].tolist())
            for e in shared:
                assert projected[e] == pytest.approx(eng.gross_link_load[e], abs=1e-9)


def test_protected_projection_uses_gross_not_already_dropped_traffic():
    """Gross >= carried input, so the safety check never benefits from upstream drops."""
    eng = make_engine_v2("overload_stress", episode_seed=103)
    used_gross = False
    while not eng.done:
        eng.step_interval()
        if np.any(eng.gross_link_load > eng.link_input_load + 1e-6):
            used_gross = True
            d_idx = int(eng._protected_idx[0])
            for p_idx in range(K):
                gross_proj = eng.projected_gross_bottleneck(d_idx, p_idx)
                base = eng.link_input_load.copy()
                if not eng.disconnected[d_idx]:
                    base[eng._cand_links[d_idx][int(eng.current_path[d_idx])]] -= \
                        eng.demand_offered[d_idx]
                links = eng._cand_links[d_idx][p_idx]
                carried_proj = float(np.max(
                    (base[links] + eng.demand_offered[d_idx]) / eng.capacity[links]))
                assert gross_proj >= carried_proj - 1e-9
            break
    assert used_gross


# ================================= 9. TE history, reversals, FRR and recovery
def test_accepted_te_stores_the_old_path_and_sets_dwell():
    eng = make_engine_v2("full_day", episode_seed=101)
    d_idx, old = 0, int(eng.current_path[0])
    target = next(p for p in range(K) if eng.validate_te_action(d_idx, p)[0])
    rec = eng.apply_te_action(d_idx, target)
    assert rec["accepted"] and not rec["reversal"]
    assert int(eng.previous_te_path[d_idx]) == old
    assert int(eng.te_dwell_remaining[d_idx]) == eng.cfg.minimum_te_dwell_steps
    assert int(eng.current_path[d_idx]) == target
    assert int(eng.path_age_steps[d_idx]) == 0
    assert eng.episode_totals["accepted_te_changes"] == 1
    assert eng.episode_totals["te_reversals"] == 0


def test_a_to_b_to_a_inside_the_window_is_a_reversal():
    eng = make_engine_v2("full_day", episode_seed=101)
    d_idx, a = 0, int(eng.current_path[0])
    b = next(p for p in range(K) if eng.validate_te_action(d_idx, p)[0])
    eng.apply_te_action(d_idx, b)
    for _ in range(eng.cfg.minimum_te_dwell_steps):
        eng.step_interval()
    rec = eng.apply_te_action(d_idx, a)
    assert rec["accepted"] and rec["reversal"]
    assert eng.episode_totals["te_reversals"] == 1


def test_the_same_return_outside_the_window_is_not_a_reversal():
    eng = make_engine_v2("full_day", episode_seed=101)
    d_idx, a = 0, int(eng.current_path[0])
    b = next(p for p in range(K) if eng.validate_te_action(d_idx, p)[0])
    eng.apply_te_action(d_idx, b)
    for _ in range(eng.cfg.reversal_window_steps + 1):
        eng.step_interval()
    rec = eng.apply_te_action(d_idx, a)
    assert rec["accepted"] and not rec["reversal"]
    assert eng.episode_totals["te_reversals"] == 0


def test_a_to_b_to_c_to_a_follows_the_single_slot_previous_path_rule():
    """previous_te_path holds only the immediately preceding path, so the
    return to A after an intervening move to C is not a reversal."""
    eng = make_engine_v2("full_day", episode_seed=101)
    d_idx, a = 0, int(eng.current_path[0])
    others = [p for p in range(K) if p != a]
    b, c = others[0], others[1]
    assert eng.apply_te_action(d_idx, b)["accepted"]
    for _ in range(eng.cfg.minimum_te_dwell_steps):
        eng.step_interval()
    rec_c = eng.apply_te_action(d_idx, c)
    assert rec_c["accepted"] and rec_c["reversal"] is False
    assert int(eng.previous_te_path[d_idx]) == b
    for _ in range(eng.cfg.minimum_te_dwell_steps):
        eng.step_interval()
    rec_a = eng.apply_te_action(d_idx, a)
    assert rec_a["accepted"] and rec_a["reversal"] is False


def test_path_age_resets_only_on_an_actual_router_sequence_change():
    eng = make_engine_v2("full_day", episode_seed=101)
    for _ in range(5):
        eng.step_interval()
    d_idx = 0
    assert int(eng.path_age_steps[d_idx]) == 5
    cur = int(eng.current_path[d_idx])
    eng.apply_te_action(d_idx, cur)              # rejected: same live path
    assert int(eng.path_age_steps[d_idx]) == 5
    target = next(p for p in range(K) if eng.validate_te_action(d_idx, p)[0])
    eng.apply_te_action(d_idx, target)
    assert int(eng.path_age_steps[d_idx]) == 0


def test_path_age_increments_once_per_completed_interval():
    eng = make_engine_v2("full_day", episode_seed=101)
    for expected in range(1, 6):
        eng.step_interval()
        assert np.all(eng.path_age_steps == expected)


def test_frr_moves_to_the_cheapest_live_candidate_and_is_never_policy_churn():
    eng = synth_engine([{"t_min": 60, "type": "link_down", "link": "L11"}])
    while eng.t_min < 60.0:
        eng.step_interval()
    assert eng.episode_totals["frr_changes"] > 0
    assert eng.episode_totals["accepted_te_changes"] == 0
    assert eng.episode_totals["te_reversals"] == 0
    assert eng.episode_totals["rejected_te_requests"] == 0
    for rec in eng.frr_history:
        if rec["event"] != "reroute":
            continue
        d_idx = rec["demand_idx"]
        chosen = rec["to_path"]
        for cheaper in range(chosen):
            assert not eng.path_available(d_idx, cheaper)


def test_frr_clears_the_previous_te_path_and_does_not_set_dwell():
    eng = synth_engine([{"t_min": 30, "type": "link_down", "link": "L11"}])
    d_idx = eng.demand_by_id["D4"].index
    target = next(p for p in range(K) if eng.validate_te_action(d_idx, p)[0])
    eng.apply_te_action(d_idx, target)
    assert int(eng.previous_te_path[d_idx]) >= 0
    while eng.t_min < 30.0:
        eng.step_interval()
    moved = [r for r in eng.frr_history if r["demand_idx"] == d_idx]
    if moved:
        assert int(eng.previous_te_path[d_idx]) == -1
    assert eng.episode_totals["frr_changes"] > 0
    # FRR never contributes to the TE ledger
    assert eng.episode_totals["accepted_te_changes"] == 1


def test_frr_bypasses_te_dwell():
    eng = synth_engine([{"t_min": 30, "type": "link_down", "link": PE5_LAST_HOP}])
    d_idx = eng.demand_by_id["D1"].index
    eng.te_dwell_remaining[d_idx] = 3
    assert not eng.validate_te_action(d_idx, (int(eng.current_path[d_idx]) + 1) % K)[0]
    while eng.t_min < 30.0:
        eng.step_interval()
    assert bool(eng.disconnected[d_idx])          # no live candidate at all


def test_no_live_candidate_yields_disconnection_counted_separately():
    eng = synth_engine([{"t_min": 30, "type": "link_down", "link": PE5_LAST_HOP}])
    while eng.t_min < 30.0:
        eng.step_interval()
    down = {eng.demands[i].id for i in np.flatnonzero(eng.disconnected)}
    assert down == set(PE5_DEMANDS)
    assert eng.episode_totals["frr_disconnections"] == len(PE5_DEMANDS)
    assert eng.episode_totals["accepted_te_changes"] == 0
    for d in np.flatnonzero(eng.disconnected):
        assert eng.demand_delivered[d] == 0.0


def test_recovery_restores_disconnected_demands_and_is_classified_separately():
    eng = synth_engine([{"t_min": 30, "type": "link_down", "link": PE5_LAST_HOP},
                        {"t_min": 60, "type": "link_up", "link": PE5_LAST_HOP}])
    while eng.t_min < 60.0:
        eng.step_interval()
    assert not np.any(eng.disconnected)
    assert eng.episode_totals["recovery_restorations"] == len(PE5_DEMANDS)
    assert eng.episode_totals["accepted_te_changes"] == 0
    assert {r["demand_id"] for r in eng.restoration_history} == set(PE5_DEMANDS)


def test_connected_traffic_is_not_moved_merely_because_a_link_recovered():
    eng = synth_engine([{"t_min": 30, "type": "link_down", "link": "L11"},
                        {"t_min": 60, "type": "link_up", "link": "L11"}])
    while eng.t_min < 30.0:
        eng.step_interval()
    after_failure = eng.current_path.copy()
    assert not np.any(eng.disconnected)
    while eng.t_min < 60.0:
        eng.step_interval()
    np.testing.assert_array_equal(eng.current_path, after_failure)
    assert eng.episode_totals["recovery_restorations"] == 0


def test_full_temporary_disconnection_does_not_terminate_the_episode():
    e = make_env_v2(scenario="full_day", root_seed=101)
    e.reset(options={"episode_seed": 101})
    e.eng.disconnected[:] = True
    assert e.eng.all_disconnected
    _, _, terminated, truncated, info = e.step(0)
    assert terminated is False and truncated is False
    assert info["metrics"]["disconnected_demands"] == N_DEMANDS


def test_recovery_can_occur_later_in_the_same_scenario():
    eng = synth_engine([{"t_min": 30, "type": "link_down", "link": PE5_LAST_HOP},
                        {"t_min": 60, "type": "link_up", "link": PE5_LAST_HOP}],
                       duration_min=120)
    disconnected_seen = restored_seen = False
    while not eng.done:
        eng.step_interval()
        if np.any(eng.disconnected):
            disconnected_seen = True
        elif disconnected_seen:
            restored_seen = True
    assert disconnected_seen and restored_seen


# ================================================ 11. aggregation / termination
def test_exactly_five_one_minute_ticks_per_interval():
    eng = make_engine_v2("full_day", episode_seed=101)
    assert eng.cfg.micro_ticks_per_interval == 5
    assert eng.cfg.control_interval_min == 5
    t0 = eng.t_min
    eng.step_interval()
    assert eng.t_min == pytest.approx(t0 + 5.0)
    assert eng.step_count == 1


def test_aggregation_uses_the_defined_reducers():
    eng = make_engine_v2("overload_stress", episode_seed=103)
    ticks = []
    n = eng.cfg.micro_ticks_per_interval
    for _ in range(n):
        old_t = eng.t_min
        eng.t_min = old_t + 1.0
        eng.traffic.advance_noise()
        eng._process_link_events(old_t, eng.t_min)
        ticks.append(eng._compute_tick())
    agg = eng.aggregate_interval(ticks)

    assert agg["protected_disconnect"] == max(t["protected_disconnect"] for t in ticks)
    assert agg["unprotected_disconnect"] == max(t["unprotected_disconnect"] for t in ticks)
    assert agg["max_util"] == max(t["max_util"] for t in ticks)
    assert agg["overload_ratio"] == pytest.approx(
        float(np.mean([t["overload_ratio"] for t in ticks])))
    # ratio of sums, not mean of ratios
    assert agg["delivered_ratio"] == pytest.approx(
        sum(t["delivered_mbps"] for t in ticks) / sum(t["offered_mbps"] for t in ticks))
    assert agg["sla_severity"] == pytest.approx(
        sum(t["sla_severity_sum"] for t in ticks) / (n * eng._q_sum))


def test_delivered_ratio_is_a_ratio_of_sums_not_a_mean_of_ratios():
    eng = make_engine_v2("overload_stress", episode_seed=103)
    ticks = [
        {"protected_disconnect": 0.0, "unprotected_disconnect": 0.0,
         "sla_severity_sum": 0.0, "offered_mbps": 100.0, "delivered_mbps": 100.0,
         "max_util": 0.1, "overload_ratio": 0.0, "mean_util": 0.0, "util_std": 0.0,
         "mean_delay_ms": 0.0, "loss_ratio": 0.0, "sla_violation_fraction": 0.0,
         "flow_solver_residual": 0.0, "max_delay_ms": 0.0, "gross_max_util": 0.0,
         "sla_violations": 0, "congested_links": 0, "disconnected_demands": 0,
         "protected_disconnected_demands": 0, "flow_solver_iterations": 1},
        {"protected_disconnect": 0.0, "unprotected_disconnect": 0.0,
         "sla_severity_sum": 0.0, "offered_mbps": 900.0, "delivered_mbps": 450.0,
         "max_util": 0.1, "overload_ratio": 0.0, "mean_util": 0.0, "util_std": 0.0,
         "mean_delay_ms": 0.0, "loss_ratio": 0.0, "sla_violation_fraction": 0.0,
         "flow_solver_residual": 0.0, "max_delay_ms": 0.0, "gross_max_util": 0.0,
         "sla_violations": 0, "congested_links": 0, "disconnected_demands": 0,
         "protected_disconnected_demands": 0, "flow_solver_iterations": 1},
    ]
    agg = eng.aggregate_interval(ticks)
    assert agg["delivered_ratio"] == pytest.approx(550.0 / 1000.0)     # not 0.75


def test_scenario_end_is_truncation_never_termination():
    e = make_env_v2(scenario="overload_stress", root_seed=103)
    e.reset(options={"episode_seed": 103})
    steps = 0
    while True:
        _, _, terminated, truncated, _ = e.step(0)
        steps += 1
        assert terminated is False
        if truncated:
            break
    assert steps == 240 // 5
    assert e.eng.t_min == pytest.approx(240.0)


@pytest.mark.parametrize("scenario,expected", [
    ("full_day", 288), ("evening_peak", 84), ("link_failure", 60),
    ("ood_double_failure", 60), ("overload_stress", 48),
])
def test_every_paired_controller_gets_the_same_horizon(scenario, expected):
    """Equal horizons are what make the paired comparison valid."""
    lengths = []
    for policy in ("noop", "random", "roundrobin"):
        e = make_env_v2(scenario=scenario, root_seed=101)
        e.reset(options={"episode_seed": 101})
        rng = np.random.default_rng(7)
        n = 0
        while True:
            if policy == "noop":
                a = 0
            elif policy == "random":
                a = int(rng.choice(np.flatnonzero(e.action_masks())))
            else:
                legal = np.flatnonzero(e.action_masks())
                a = int(legal[n % len(legal)])
            _, _, _, truncated, _ = e.step(a)
            n += 1
            if truncated:
                break
        lengths.append((n, e.eng.t_min))
    assert lengths[0][0] == expected
    assert len(set(lengths)) == 1, lengths


def test_route_cost_is_applied_at_most_once_per_interval():
    e = make_env_v2(scenario="full_day", root_seed=101)
    e.reset(options={"episode_seed": 101})
    rng = np.random.default_rng(2)
    for _ in range(30):
        legal = np.flatnonzero(e.action_masks())
        _, _, _, truncated, info = e.step(int(rng.choice(legal)))
        c = info["reward_components"]
        assert c["move_fixed"] in (0.0, pytest.approx(-0.08))
        assert c["invalid"] in (0.0, pytest.approx(-0.05))
        assert not (c["move_fixed"] != 0.0 and c["invalid"] != 0.0)
        if truncated:
            break
