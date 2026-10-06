"""Tests for the post-V2 research study package (mplssim/study)."""

from __future__ import annotations

import numpy as np
import pytest

from mplssim.experiments.evaluation_v2 import run_evaluation_episode
from mplssim.experiments.v2_factory import make_env_v2
from mplssim.study.episode import BaselinePolicy, NoopPolicy, run_episode
from mplssim.study.oracles import MyopicOracle, simulate


SHARED_KEYS = (
    "operational_return", "episode_length", "delivered_ratio_mean",
    "sla_violations_demand_intervals", "max_utilization_mean",
    "accepted_te_changes", "te_reversals", "moved_mbps_total",
    "reward_component_sum_exact", "loss_ratio_mean", "delay_ms_mean",
)


@pytest.mark.parametrize("algorithm", ["greedy", "cspf"])
def test_study_evaluator_matches_governed_evaluator(algorithm):
    _, governed = run_evaluation_episode(
        algorithm=algorithm, scenario="link_failure", seed=101, policy=None)
    mine, _ = run_episode(BaselinePolicy(algorithm, seed=101), "link_failure", 101)
    for key in SHARED_KEYS:
        assert mine[key] == governed[key], key
    assert mine["reward_components"] == governed["reward_components"]


def test_simulate_restores_exact_state_and_rng():
    a = make_env_v2("flash_crowd", root_seed=7)
    b = make_env_v2("flash_crowd", root_seed=7)
    oa, _ = a.reset(options={"episode_seed": 7})
    ob, _ = b.reset(options={"episode_seed": 7})
    rng = np.random.default_rng(0)
    for _ in range(30):
        mask = a.action_masks()
        assert np.array_equal(mask, b.action_masks())
        # interleave clairvoyant rollouts on ``a`` only
        for act in rng.choice(np.flatnonzero(mask), size=3):
            simulate(a, [int(act)], horizon=4)
        act = int(rng.choice(np.flatnonzero(mask)))
        oa, ra, _, ta, _ = a.step(act)
        ob, rb, _, tb, _ = b.step(act)
        assert ra == rb and ta == tb
        assert np.array_equal(oa, ob)


def test_simulated_first_step_equals_real_step():
    env = make_env_v2("deceptive_local_optimum", root_seed=3)
    env.reset(options={"episode_seed": 3})
    for _ in range(10):
        env.step(0)
    mask = env.action_masks()
    act = int(np.flatnonzero(mask)[5])
    predicted, rewards = simulate(env, [act], horizon=1)
    _, real, _, _, _ = env.step(act)
    assert predicted == rewards[0] == real


def test_myopic_oracle_maximizes_exact_immediate_reward():
    env = make_env_v2("evening_peak", root_seed=11)
    env.reset(options={"episode_seed": 11})
    for _ in range(12):
        env.step(0)
    mask = env.action_masks()
    chosen = MyopicOracle().act(None, mask, env)
    values = {int(a): simulate(env, [int(a)], horizon=1)[0] for a in np.flatnonzero(mask)}
    assert values[chosen] == max(values.values())


def test_noop_policy_never_moves():
    summary, _ = run_episode(NoopPolicy(), "overload_stress", 5)
    assert summary["accepted_te_changes"] == 0
    assert summary["noop_frequency"] == 1.0


# ---------------------------------------------------------------- learners
def test_masked_q_with_gamma_zero_is_the_bandit_exactly():
    import torch
    from mplssim.experiments.learning_common import load_learning_config
    from mplssim.experiments.masked_bandit import MaskedContextualBandit
    from mplssim.study.learners import MaskedQLearner

    cfg = dict(load_learning_config()["masked_bandit"])
    cfg.update(batch_size=32, warmup_transitions=64, replay_capacity=500)
    a = MaskedContextualBandit(10, 5, torch.device("cpu"), 3, cfg)
    b = MaskedQLearner(10, 5, torch.device("cpu"), 3, {**cfg, "gamma": 0.0})
    rng = np.random.default_rng(1)
    for _ in range(40):
        obs = rng.random((4, 10), dtype=np.float32)
        masks = rng.random((4, 5)) < 0.7
        masks[:, 0] = True
        acts_a = a.predict(obs, masks, deterministic=False)
        acts_b = b.predict(obs, masks, deterministic=False)
        assert np.array_equal(acts_a, acts_b)
        r = rng.normal(size=4).astype(np.float32)
        a.observe(obs, acts_a, masks, r)
        b.observe_transitions(obs, acts_b, masks, r, obs, masks, np.zeros(4, bool))
        a.update(); b.update()
    for pa, pb in zip(a.network.parameters(), b.network.parameters()):
        assert torch.equal(pa, pb)


def test_masked_q_bootstraps_only_legal_successor_actions():
    import torch
    from mplssim.experiments.learning_common import load_learning_config
    from mplssim.study.learners import MaskedQLearner

    cfg = dict(load_learning_config()["masked_bandit"])
    cfg.update(batch_size=8, warmup_transitions=8, replay_capacity=64, gamma=0.9)
    q = MaskedQLearner(3, 4, torch.device("cpu"), 0, cfg)
    with torch.no_grad():  # make the illegal action 3 look extremely valuable
        q.network.output.bias[:] = torch.tensor([0.0, 0.0, 0.0, 1e6])
        q.target.load_state_dict(q.network.state_dict())
    obs = np.zeros((8, 3), np.float32)
    m = np.ones((8, 4), bool)
    nm = m.copy(); nm[:, 3] = False
    q.observe_transitions(obs, np.zeros(8, int), m, np.zeros(8, np.float32), obs, nm,
                          np.zeros(8, bool))
    out = q.update()
    assert out is not None and out["loss"] < 1e3  # 1e6 never entered the target


# ---------------------------------------------------------------- variants
def test_delayed_variant_with_zero_delay_reproduces_frozen_env():
    from mplssim.study.variants import DelayedTeEnvV2

    base = make_env_v2("link_failure", root_seed=9)
    var = DelayedTeEnvV2(scenario="link_failure", root_seed=9, delay_steps=0)
    ob, _ = base.reset(options={"episode_seed": 9})
    ov, _ = var.reset(options={"episode_seed": 9})
    assert ov.shape[0] == 604 + 5 * 17
    rng = np.random.default_rng(4)
    trunc = False
    while not trunc:
        mask = base.action_masks()
        assert np.array_equal(mask, var.action_masks())
        act = 0 if rng.random() < 0.6 else int(rng.choice(np.flatnonzero(mask)))
        ob, rb, _, trunc, ib = base.step(act)
        ov, rv, _, _, iv = var.step(act)
        assert rb == rv
        assert np.array_equal(ob, ov[:604]) and not ov[604:].any()


def test_delayed_variant_activates_after_delay_and_blocks_pending_demand():
    from mplssim.study.variants import DelayedTeEnvV2

    env = DelayedTeEnvV2(scenario="full_day", root_seed=2, delay_steps=2)
    env.reset(options={"episode_seed": 2})
    mask = env.action_masks()
    act = int(np.flatnonzero(mask[1:])[0]) + 1
    d, p = divmod(act - 1, env.k)
    before = int(env.eng.current_path[d])
    _, r0, _, _, i0 = env.step(act)
    assert i0["reward_components"]["move_fixed"] < 0       # cost charged now
    assert int(env.eng.current_path[d]) == before          # but no effect yet
    assert not env.action_masks()[1 + d * env.k: 1 + (d + 1) * env.k].any()
    env.step(0)
    assert int(env.eng.current_path[d]) == before
    _, _, _, _, i2 = env.step(0)                            # due at this boundary
    assert int(env.eng.current_path[d]) == p
    assert i2["activated_requests"] == 1
    assert i2["reward_components"]["move_fixed"] == 0.0


def test_simulate_restores_delayed_variant_state():
    from mplssim.study.variants import DelayedTeEnvV2

    env = DelayedTeEnvV2(scenario="evening_peak", root_seed=5, delay_steps=1)
    env.reset(options={"episode_seed": 5})
    mask = env.action_masks()
    act = int(np.flatnonzero(mask[1:])[3]) + 1
    pend = env.pending_path.copy()
    simulate(env, [act], horizon=3)
    assert np.array_equal(env.pending_path, pend)
    assert np.array_equal(env.action_masks(), mask)


# ---------------------------------------------------------------- statistics
def test_stratified_bootstrap_contains_true_mean_and_is_degenerate_for_constants():
    from mplssim.study.stats import stratified_bootstrap_mean
    rng = np.random.default_rng(0)
    strata = np.repeat(np.arange(7), 20)
    x = rng.normal(1.0, 1.0, size=140) + strata  # stratum offsets
    ci = stratified_bootstrap_mean(x - strata, strata, n_boot=2000)
    assert ci.low < 1.0 < ci.high
    c = stratified_bootstrap_mean(np.full(10, 3.0), np.zeros(10), n_boot=200)
    assert c.low == c.high == c.estimate == 3.0


def test_paired_frame_and_learner_comparison():
    import pandas as pd
    from mplssim.study.stats import compare_learners, paired_frame
    rows = []
    for root in (1, 2, 3):
        for s in ("a", "b"):
            for seed in range(5):
                base = 10 * (s == "a") + seed
                rows.append({"policy": "x", "root": root, "scenario": s, "seed": seed,
                             "operational_return": base + 2.0})
                rows.append({"policy": "y", "root": root, "scenario": s, "seed": seed,
                             "operational_return": base})
    pairs = paired_frame(pd.DataFrame(rows), "x", "y")
    assert len(pairs) == 30 and np.allclose(pairs["diff"], 2.0)
    out = compare_learners(pairs, seed=1)
    assert out["boot_est"] == 2.0 and out["boot_lo"] == out["boot_hi"] == 2.0
    assert out["roots_positive"] == 3


def test_frozen_rollout_holds_traffic_and_restores_state():
    from mplssim.study.oracles import simulate
    env = make_env_v2("flash_crowd", root_seed=13)
    env.reset(options={"episode_seed": 13})
    for _ in range(15):
        env.step(0)
    offered = env.eng.demand_offered.copy()
    _, frozen_r = simulate(env, [0], horizon=8, frozen=True)
    assert np.array_equal(env.eng.demand_offered, offered)        # restored
    assert len(set(np.round(frozen_r, 12))) == 1                   # stationary
    _, live_r = simulate(env, [0], horizon=8)
    _, real, _, _, _ = env.step(0)
    assert live_r[0] == real                                        # live path untouched


def test_milp_target_never_worse_than_current_and_action_is_legal():
    from mplssim.study.milp import MilpTargetPolicy
    env = make_env_v2("deceptive_local_optimum", root_seed=21)
    env.reset(options={"episode_seed": 21})
    pol = MilpTargetPolicy()
    for _ in range(20):
        eng = env.eng
        target, u_star = pol.target(eng)
        u_now = float(np.max(eng.gross_link_load / eng.capacity))
        assert u_star <= u_now + 1e-6
        mask = env.action_masks()
        a = pol.act(None, mask, env)
        assert mask[a]
        env.step(a)
    assert pol.failures == 0
