"""Tests for the Gymnasium environment, action masking, and baseline controllers."""

import numpy as np

from mplssim.baselines import make_baseline
from mplssim.factory import make_engine
from mplssim.rl.env import MplsTeEnv


def test_env_passes_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    env = MplsTeEnv(scenario="evening_peak", base_seed=1)
    check_env(env, skip_render_check=True)


def test_env_reset_is_reproducible():
    e1 = MplsTeEnv(scenario="full_day", base_seed=5)
    e2 = MplsTeEnv(scenario="full_day", base_seed=5)
    o1, _ = e1.reset(options={"episode_seed": 5})
    o2, _ = e2.reset(options={"episode_seed": 5})
    assert np.allclose(o1, o2)
    for _ in range(5):
        r1 = e1.step(0)
        r2 = e2.step(0)
        assert np.allclose(r1[0], r2[0]) and r1[1] == r2[1]


def test_action_mask_shape_and_noop_always_valid():
    env = MplsTeEnv(scenario="full_day", base_seed=1)
    env.reset(options={"episode_seed": 1})
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,)
    assert bool(mask[0])
    assert mask.sum() > 1, "some reroute should be legal at start"


def test_masked_actions_are_rejected_if_forced():
    env = MplsTeEnv(scenario="full_day", base_seed=1, safety_filter=True)
    env.reset(options={"episode_seed": 1})
    mask = env.action_masks()
    bad = int(np.argmin(mask))  # a masked (invalid) action
    if not mask[bad]:
        _, _, _, _, info = env.step(bad)
        assert info["decoded_action"]["accepted"] is False
        assert info["reward_components"]["invalid"] < 0


def test_mask_blocks_paths_over_failed_links():
    env = MplsTeEnv(scenario="full_day", base_seed=1)
    env.reset(options={"episode_seed": 1})
    env.eng.inject_failure("L11")
    mask = env.action_masks()
    for d_idx, d in enumerate(env.eng.demands):
        for p_idx in range(len(d.candidate_paths)):
            if not env.eng.path_available(d_idx, p_idx):
                assert not mask[1 + d_idx * env.k + p_idx]


def test_env_episode_runs_to_truncation():
    env = MplsTeEnv(scenario="evening_peak", base_seed=3)
    env.reset(options={"episode_seed": 3})
    rng = np.random.default_rng(0)
    truncated = False
    total_r = 0.0
    for _ in range(200):
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        a = int(rng.choice(legal))
        _, r, terminated, truncated, info = env.step(a)
        total_r += r
        assert set(info["reward_components"]) >= {"delivered", "loss", "reroute"}
        if terminated or truncated:
            break
    assert truncated, "evening_peak (84 steps) must end within 200 steps"
    assert np.isfinite(total_r)


def test_actions_change_outcomes_vs_noop():
    """The same scenario with different actions must yield different metrics."""
    e1 = MplsTeEnv(scenario="deceptive_local_optimum", base_seed=2)
    e2 = MplsTeEnv(scenario="deceptive_local_optimum", base_seed=2)
    e1.reset(options={"episode_seed": 2})
    e2.reset(options={"episode_seed": 2})
    for _ in range(30):
        e1.step(0)
        mask = e2.action_masks()
        legal = np.flatnonzero(mask)
        e2.step(int(legal[-1]))
    m1 = e1.eng.metrics_history[-1]
    m2 = e2.eng.metrics_history[-1]
    assert m1["max_util"] != m2["max_util"] or m1["mean_delay_ms"] != m2["mean_delay_ms"]


# ----------------------------------------------------------------- baselines
def _run_controller(name: str, scenario: str, seed: int, steps: int = 60):
    eng = make_engine(scenario, seed=seed)
    ctl = make_baseline(name, seed=seed)
    for _ in range(steps):
        if eng.done:
            break
        for d_idx, p_idx in ctl.decide(eng):
            eng.apply_action(d_idx, p_idx, source=ctl.name if ctl.name != "random" else "rl")
        eng.step_interval()
    return eng


def test_static_baseline_stays_on_shortest_paths():
    eng = _run_controller("static", "full_day", seed=4)
    assert np.all(eng.current_path == 0)


def test_greedy_reacts_to_congestion():
    eng = _run_controller("greedy", "overload_stress", seed=4)
    moves = [a for a in eng.action_log if a.source == "greedy" and a.accepted]
    assert moves, "greedy must reroute under overload"


def test_cspf_runs_and_respects_period():
    eng = _run_controller("cspf", "evening_peak", seed=4)
    steps_with_moves = {a.step for a in eng.action_log if a.source == "cspf"}
    assert all(s % 6 == 0 for s in steps_with_moves)


def test_baselines_beat_nothing_catastrophic():
    """All baselines finish overload_stress without total disconnection."""
    for name in ("static", "greedy", "cspf", "random"):
        eng = _run_controller(name, "overload_stress", seed=6, steps=48)
        assert not eng.all_disconnected
