"""Tests for the presentation-hardening correctness fixes (no retraining)."""

import numpy as np
import pandas as pd
import pytest

from mplssim.baselines.controllers import RandomController
from mplssim.experiments.runner import summarize_records
from mplssim.factory import engine_config_from_training, make_engine
from mplssim.rl.env import GLOBAL_FEATURES, LINK_FEATURES, MplsTeEnv
from mplssim.sim.engine import EngineConfig


# --------------------------------------------------- observation dimension
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_observation_dim_formula_for_any_k_paths(k):
    cfg = engine_config_from_training()
    ecfg = EngineConfig(
        control_interval_min=cfg.control_interval_min,
        micro_ticks_per_interval=cfg.micro_ticks_per_interval,
        k_paths=k, max_hop_factor=cfg.max_hop_factor,
        reroute_cooldown_steps=cfg.reroute_cooldown_steps,
    )
    env = MplsTeEnv(scenario="evening_peak", base_seed=1, engine_cfg=ecfg)
    obs, _ = env.reset(options={"episode_seed": 1})
    n_demands, n_dlinks = env.n_demands, env.n_dlinks
    expected = LINK_FEATURES * n_dlinks + (7 + 2 * k) * n_demands + GLOBAL_FEATURES
    assert env.observation_space.shape == (expected,)
    assert obs.shape == (expected,)
    assert env.action_space.n == 1 + n_demands * k
    # run a few steps with valid random actions — no indexing errors
    rng = np.random.default_rng(0)
    for _ in range(5):
        legal = np.flatnonzero(env.action_masks())
        obs, _, te, tr, _ = env.step(int(rng.choice(legal)))
        assert obs.shape == (expected,)
        if te or tr:
            break


def test_gym_checker_passes_for_nondefault_k():
    from gymnasium.utils.env_checker import check_env
    cfg = engine_config_from_training()
    ecfg = EngineConfig(control_interval_min=cfg.control_interval_min,
                        micro_ticks_per_interval=cfg.micro_ticks_per_interval,
                        k_paths=3, max_hop_factor=cfg.max_hop_factor,
                        reroute_cooldown_steps=cfg.reroute_cooldown_steps)
    check_env(MplsTeEnv(scenario="evening_peak", base_seed=2, engine_cfg=ecfg),
              skip_render_check=True)


def test_pretrained_model_still_compatible():
    """k=4 config must still produce the exact shapes ppo_te was trained with."""
    from mplssim.validation import expected_shapes
    exp = expected_shapes()
    assert exp["observation_dim"] == 586 and exp["action_dim"] == 69


# ------------------------------------------------------ projected-load check
def test_projected_loads_conserve_volume_and_avoid_double_count():
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    for d_idx in range(eng.n_demands):
        cur = int(eng.current_path[d_idx])
        for p_idx in range(len(eng.demands[d_idx].candidate_paths)):
            proj = eng.projected_link_loads_after_move(d_idx, p_idx)
            if p_idx == cur:
                # moving to the same path must change nothing (no double count)
                assert np.allclose(proj, eng.link_load)
            # links shared by old and new path keep their load exactly
            shared = set(eng._path_links[d_idx][cur]) & set(eng._path_links[d_idx][p_idx])
            for li in shared:
                assert proj[li] == pytest.approx(eng.link_load[li])


def test_protected_check_uses_projected_load():
    """Synthetic double-count case: a protected demand's candidate shares a
    link with its current path. The shared link has headroom below the
    demand's volume ONLY because the demand's own traffic is already on it.
    The corrected (projected) check must accept the move; the old raw-load
    check would have rejected it."""
    eng = make_engine("full_day", seed=3)
    eng.step_interval()
    # find a protected demand and a candidate sharing >=1 link with current
    found = None
    for d_idx, d in enumerate(eng.demands):
        if not d.cls.protected:
            continue
        cur = int(eng.current_path[d_idx])
        for p_idx in range(len(d.candidate_paths)):
            if p_idx == cur:
                continue
            shared = set(map(int, eng._path_links[d_idx][cur])) & \
                set(map(int, eng._path_links[d_idx][p_idx]))
            if shared:
                found = (d_idx, cur, p_idx, min(shared))
                break
        if found:
            break
    assert found, "topology should give protected demands overlapping candidates"
    d_idx, cur, p_idx, s = found

    # craft the state: only this demand's traffic in the network, plus filler
    # on the shared link so raw headroom < vol but projected headroom >= 0
    vol = 100.0
    eng.disconnected[:] = False
    eng.demand_volumes[:] = 0.0
    eng.demand_volumes[d_idx] = vol
    eng.link_load[:] = 0.0
    eng.link_load[eng._path_links[d_idx][cur]] += vol       # its own traffic
    cap_s = eng.capacity[s]
    eng.link_load[s] = cap_s - vol / 2                      # raw headroom = vol/2 < vol

    raw_headroom = eng.path_available_bandwidth(d_idx, p_idx)
    assert raw_headroom < vol, "old raw check would reject this move"
    ok, reason = eng.validate_action(d_idx, p_idx, source="rl")
    assert ok, f"projected check must accept (own traffic cancels): {reason}"


# --------------------------------------------------------- random baseline
def test_random_baseline_uniform_over_valid_actions():
    eng = make_engine("full_day", seed=9)
    eng.step_interval()
    ctl = RandomController(seed=1)
    valid = ctl.valid_actions(eng)
    assert len(valid) > 5
    counts = {v: 0 for v in valid}
    noop = 0
    n = 4000
    for _ in range(n):
        moves = ctl.decide(eng)
        if not moves:
            noop += 1
        else:
            counts[moves[0]] += 1
    assert noop / n == pytest.approx(0.5, abs=0.05), "documented no-op prob is 0.5"
    # chi-square goodness-of-fit against uniform at alpha = 0.001
    from scipy.stats import chisquare
    observed = [counts[v] for v in valid]
    stat, pvalue = chisquare(observed)
    assert pvalue > 0.001, f"sampling not uniform over the mask (p={pvalue:.2g})"


# ------------------------------------------------------------ unit conversion
def test_dropped_gbit_conversion():
    """100 Mbps dropped for two 5-minute intervals = 100*300*2 Mbit = 60 Gbit."""
    base = {
        "max_util": 0.5, "mean_util": 0.2, "util_std": 0.1, "jain_fairness": 0.9,
        "mean_delay_ms": 10.0, "p95_delay_ms": 20.0, "max_delay_ms": 30.0,
        "loss_ratio": 0.1, "delivered_ratio": 0.9, "sla_violations": 0,
        "sla_violation_fraction": 0.0, "priority_sla_success": 1.0,
        "congested_links": 0, "overload_ratio": 0.0, "disconnected_demands": 0,
        "t_min": 5.0, "hour": 1.0, "step": 1, "reroutes": 0, "flaps": 0,
        "frr_events": 0, "n_failed_links": 0, "reward": 0.0, "n_demands": 17,
    }
    rows = [dict(base, offered_mbps=1000.0, carried_mbps=900.0),
            dict(base, offered_mbps=1000.0, carried_mbps=900.0)]
    s = summarize_records(pd.DataFrame(rows), "static", "test", 1)
    assert s["dropped_gbit_total"] == pytest.approx(60.0)


# --------------------------------------------------------- config validation
def test_config_validation_passes_on_shipped_configs():
    from mplssim.validation import validate_configs
    validate_configs()


def test_model_mismatch_message_is_clear():
    from mplssim.validation import ConfigError, check_model_compatibility

    class FakeSpace:
        def __init__(self, shape=None, n=None):
            self.shape, self.n = shape, n

    class FakeModel:
        observation_space = FakeSpace(shape=(100,))
        action_space = FakeSpace(n=10)

    with pytest.raises(ConfigError) as exc:
        check_model_compatibility(FakeModel(), "fake")
    msg = str(exc.value)
    assert "obs=100" in msg and "586" in msg and "metadata.json" in msg


# ----------------------------------------------- randomized scenario derivation
def test_random_day_uses_configured_demands_and_egress():
    eng = make_engine("random_day", seed=77)
    demand_ids = {d.id for d in eng.demands}
    egress = {d.dst for d in eng.demands}
    for ev in eng.scenario.events:
        if ev["type"] == "burst":
            assert set(ev["demands"]) <= demand_ids
        if ev["type"] == "flash_crowd":
            assert ev["dst"] in egress


# ------------------------------------------------------------ display scaling
def test_display_scale_keeps_utilization_invariant():
    from mplssim.display import scale_mbps
    for load, cap in [(150.0, 1000.0), (1800.0, 2000.0), (90.0, 250.0)]:
        s_load, s_cap = scale_mbps(load, 10), scale_mbps(cap, 10)
        assert s_load / s_cap == pytest.approx(load / cap)
        assert s_load == load * 10 and s_cap == cap * 10
