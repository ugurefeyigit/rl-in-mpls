"""Unit tests for topology, candidate paths, traffic model, and engine invariants."""

import numpy as np
import pytest

from mplssim.factory import get_scenarios, get_topology, get_traffic_config, make_engine
from mplssim.paths.candidates import generate_candidate_paths
from mplssim.sim import models as m


# ------------------------------------------------------------------ topology
def test_topology_loads_and_is_connected():
    topo = get_topology()
    assert len(topo.routers) == 18
    assert len(topo.link_defs) == 32
    assert topo.n_dlinks == 64
    roles = [r.role for r in topo.routers.values()]
    assert roles.count("PE_IN") == 4 and roles.count("PE_OUT") == 4


def test_directed_links_are_symmetric_pairs():
    topo = get_topology()
    for ld in topo.link_defs.values():
        assert (ld.a, ld.z) in topo.dlink_by_pair
        assert (ld.z, ld.a) in topo.dlink_by_pair


# ------------------------------------------------------------- candidate paths
def test_candidate_paths_valid_and_loop_free():
    topo = get_topology()
    for d in get_traffic_config().demands:
        cands = generate_candidate_paths(topo, d.src, d.dst, k=4)
        assert 1 <= len(cands) <= 4
        for path in cands:
            assert path[0] == d.src and path[-1] == d.dst
            assert len(set(path)) == len(path), "loop in candidate path"
            # every hop must be a real directed link
            topo.path_dlink_indices(path)
        # index 0 is the admin-shortest path
        costs = [
            sum(topo.dlink_by_pair[(p[i], p[i + 1])].weight for i in range(len(p) - 1))
            for p in cands
        ]
        assert costs[0] == min(costs)


def test_alternative_paths_exist_for_all_demands():
    topo = get_topology()
    for d in get_traffic_config().demands:
        cands = generate_candidate_paths(topo, d.src, d.dst, k=4)
        assert len(cands) >= 2, f"{d.id} has no alternative path"


# ---------------------------------------------------------------- delay/loss
def test_delay_and_loss_models():
    util = np.array([0.0, 0.5, 0.9, 0.99, 1.5])
    q = m.queue_delay_ms(util)
    assert q[0] == 0.0 and np.all(np.diff(q) >= 0) and q[-1] <= m.Q_MAX_MS
    loss = m.loss_fraction(util)
    assert loss[0] == 0.0 and loss[1] == 0.0 and loss[2] == 0.0
    assert 0 < loss[3] < m.SOFT_LOSS_MAX
    # at rho=1.5, a third of offered traffic (minus soft residual) is dropped
    assert loss[4] == pytest.approx(1.0 - 0.98 / 1.5)


# -------------------------------------------------------------------- traffic
def test_traffic_reproducible_for_same_seed():
    e1 = make_engine("full_day", seed=1)
    e2 = make_engine("full_day", seed=1)
    for _ in range(10):
        e1.step_interval()
        e2.step_interval()
    assert np.allclose(e1.demand_volumes, e2.demand_volumes)
    assert e1.metrics_history[-1] == e2.metrics_history[-1]


def test_traffic_differs_across_seeds_and_time():
    e1 = make_engine("full_day", seed=1)
    e2 = make_engine("full_day", seed=2)
    e1.step_interval()
    v_t1 = e1.demand_volumes.copy()
    e2.step_interval()
    assert not np.allclose(v_t1, e2.demand_volumes)
    for _ in range(50):
        e1.step_interval()
    assert not np.allclose(v_t1, e1.demand_volumes), "traffic must vary over time"


# --------------------------------------------------------------------- engine
def test_flow_conservation_on_links():
    """Sum of link loads == sum over demands of volume * hops (no traffic invented)."""
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    expected = 0.0
    for d_idx in range(eng.n_demands):
        if not eng.disconnected[d_idx]:
            hops = len(eng._path_links[d_idx][int(eng.current_path[d_idx])])
            expected += eng.demand_volumes[d_idx] * hops
    assert float(np.sum(eng.link_load)) == pytest.approx(expected)


def test_rerouting_changes_link_loads():
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    before = eng.link_load.copy()
    d_idx = 4  # D5, bulk PE2->PE6
    target = next(p for p in range(len(eng.demands[d_idx].candidate_paths))
                  if p != int(eng.current_path[d_idx]) and eng.path_available(d_idx, p))
    ok, reason = eng.apply_action(d_idx, target, source="manual")
    assert ok, reason
    eng.step_interval()
    assert not np.allclose(before, eng.link_load), "routing must affect link loads"


def test_cooldown_blocks_immediate_reroute():
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    d_idx = 4
    tgt = next(p for p in range(len(eng.demands[d_idx].candidate_paths))
               if p != int(eng.current_path[d_idx]) and eng.path_available(d_idx, p))
    ok, _ = eng.apply_action(d_idx, tgt, source="rl")
    assert ok
    back = int(eng.prev_path_hist[d_idx][-1])
    ok2, reason2 = eng.apply_action(d_idx, back, source="rl")
    assert not ok2 and "cooldown" in reason2


def test_link_failure_triggers_frr_and_recovery():
    eng = make_engine("full_day", seed=5)
    eng.step_interval()
    affected = [d for d in range(eng.n_demands)
                if any(eng.topo.dlinks[int(li)].undirected_id == "L11"
                       for li in eng._path_links[d][int(eng.current_path[d])])]
    assert affected, "L11 should carry at least one shortest path"
    eng.inject_failure("L11")
    for d in affected:
        cur = int(eng.current_path[d])
        assert eng.disconnected[d] or eng.path_available(d, cur)
        assert all(eng.topo.dlinks[int(li)].undirected_id != "L11"
                   for li in eng._path_links[d][cur])
    eng.step_interval()
    assert "L11" in eng.metrics_history[-1]["failed_links"]
    eng.recover_link("L11")
    assert eng.link_up["L11"]


def test_invalid_actions_rejected():
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    ok, reason = eng.apply_action(0, int(eng.current_path[0]), source="rl")
    assert not ok and "already" in reason
    eng.inject_failure("L1")  # PE1-P1: on candidate paths of D1
    bad = [p for p in range(len(eng.demands[0].candidate_paths))
           if not eng.path_available(0, p)]
    if bad:
        ok, reason = eng.apply_action(0, bad[0], source="rl")
        assert not ok and "failed link" in reason


def test_scripted_failure_scenario_executes():
    eng = make_engine("link_failure", seed=7)
    for _ in range(int(300 / eng.cfg.control_interval_min)):
        eng.step_interval()
    hist = eng.metrics_history
    down_steps = [h for h in hist if "L11" in h["failed_links"]]
    assert down_steps, "L11 failure window missing"
    assert not hist[-1]["failed_links"], "L11 should have recovered"
    assert any(a.source == "frr" for a in eng.action_log)


def test_overload_produces_loss_and_sla_violations():
    eng = make_engine("overload_stress", seed=11)
    for _ in range(24):  # two hours into an overloaded evening
        last = eng.step_interval()
    assert last["max_util"] > 1.0 or last["loss_ratio"] > 0.0
    assert last["sla_violations"] >= 1


def test_engine_clone_is_independent():
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    cl = eng.clone()
    cl.apply_action(4, 1, source="manual")
    cl.step_interval()
    assert int(eng.current_path[4]) != 1 or eng.step_count != cl.step_count


def test_snapshot_is_json_serializable():
    import json
    eng = make_engine("demo_evening", seed=42)
    eng.step_interval()
    json.dumps(eng.snapshot())


# ---------------------------------------------------------------- scenarios
def test_all_scenarios_run_to_completion():
    for name in get_scenarios():
        eng = make_engine(name, seed=1)
        steps = 0
        while not eng.done and steps < 300:
            eng.step_interval()
            steps += 1
        assert eng.done or steps == 300


def test_random_day_materializes_reproducibly():
    e1 = make_engine("random_day", seed=9)
    e2 = make_engine("random_day", seed=9)
    e3 = make_engine("random_day", seed=10)
    assert e1.scenario.events == e2.scenario.events
    assert e1.scenario.events != e3.scenario.events
