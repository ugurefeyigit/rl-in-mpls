"""Gymnasium environment for MPLS traffic engineering.

Observation (flat float32 vector, all features normalized; exact ordering):

  Per directed link, for each of the 64 directed links in topology order
  (see Topology.dlinks):                                     [5 x 64 = 320]
      0: utilization / 2 (clipped to [0, 1]; 1.0 means >=200%)
      1: queue delay / Q_MAX_MS
      2: loss fraction
      3: operational state (1 = up)
      4: EWMA utilization / 2 (recent trend)

  Per demand, for each of the 17 demands in config order:   [15 x 17 = 255]
      0: offered volume / (2 * base_mbps)
      1: class priority / 6
      2: SLA max latency / 400 ms
      3: SLA max loss / 5 %
      4: protected flag
      5-8:  current path one-hot (k = 4 candidates)
      9-12: per-candidate bottleneck utilization / 2 (1.0 if candidate absent)
      13: cooldown remaining / cooldown_steps
      14: disconnected flag

  Global:                                                              [11]
      sin(hour), cos(hour), max_util/2, mean_util, util_std,
      mean_delay/60ms, loss_ratio/5% (clipped), SLA violation fraction,
      delivered ratio, reroutes in last interval / 5, episode progress

  Total: 586 floats. No future information is included; the time-of-day
  features are clock data an operator would also have.

Action space: Discrete(1 + n_demands * k_paths) = Discrete(69)
      0                     -> no-op
      1 + d * k + p         -> move demand d to candidate path p

Invalid actions (failed path, cooldown, same path, bandwidth-infeasible for
protected classes, nonexistent candidate) are masked via action_masks().

Reward: see configs/reward.yaml and mplssim/rl/reward.py.

Episode: one scenario run. `terminated` = catastrophic full disconnection,
`truncated` = scenario duration reached (time limit).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mplssim.factory import engine_config_from_training, make_engine
from mplssim.rl.reward import compute_reward, load_reward_config
from mplssim.sim import models as m
from mplssim.sim.engine import EngineConfig, SimulationEngine

LINK_FEATURES = 5
GLOBAL_FEATURES = 11


class MplsTeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "random_day",
        base_seed: int = 0,
        engine_cfg: EngineConfig | None = None,
        safety_filter: bool = True,
        reward_overrides: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.base_seed = base_seed
        self.engine_cfg = engine_cfg or engine_config_from_training()
        self.safety_filter = safety_filter
        from mplssim.rl.reward import with_overrides
        self.reward_cfg = with_overrides(load_reward_config(), reward_overrides or {})
        self._episode = 0

        self.eng: SimulationEngine = make_engine(scenario, seed=base_seed, cfg=self.engine_cfg)
        self.k = self.engine_cfg.k_paths
        self.n_demands = self.eng.n_demands
        self.n_dlinks = self.eng.topo.n_dlinks
        # per-demand layout: 5 scalars + current-path one-hot (k) +
        # per-candidate bottleneck (k) + cooldown + disconnected = 7 + 2k
        # (with k=4 this is 15, matching the pretrained ppo_te model)
        self.demand_features = 7 + 2 * self.k
        obs_dim = (LINK_FEATURES * self.n_dlinks
                   + self.demand_features * self.n_demands
                   + GLOBAL_FEATURES)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(1 + self.n_demands * self.k)
        self._steps_total = self.eng.scenario.duration_min // self.engine_cfg.control_interval_min
        self._precompute_demand_constants()

    def _precompute_demand_constants(self) -> None:
        """Per-demand observation constants that never change during an episode.

        Values and normalizations are identical to the original per-demand
        loop; they are simply computed once instead of on every observation.
        """
        demands = self.eng.demands
        self._two_base_mbps = np.array([2.0 * d.base_mbps for d in demands])
        self._obs_priority = np.array([d.cls.priority / 6.0 for d in demands])
        self._obs_max_latency = np.array(
            [min(d.cls.max_latency_ms / 400.0, 1.0) for d in demands])
        self._obs_max_loss = np.array([min(d.cls.max_loss_pct / 5.0, 1.0) for d in demands])
        self._obs_protected = np.array(
            [1.0 if d.cls.protected else 0.0 for d in demands])
        self._protected_idx = np.flatnonzero([d.cls.protected for d in demands])
        self._demand_arange = np.arange(self.n_demands)

    # ------------------------------------------------------------------- gym
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            # explicit seed => fully deterministic episode (gym API contract)
            self.base_seed = seed
            self._episode = 0
        episode_seed = self.base_seed + self._episode * 1000
        if options and "episode_seed" in options:
            episode_seed = int(options["episode_seed"])
        scenario = (options or {}).get("scenario", self.scenario)
        self._episode += 1
        self.eng = make_engine(scenario, seed=episode_seed, cfg=self.engine_cfg)
        self._steps_total = self.eng.scenario.duration_min // self.engine_cfg.control_interval_min
        return self._obs(), {"episode_seed": episode_seed, "scenario": scenario}

    def step(self, action: int):
        action = int(action)
        rerouted = flapped = invalid = False
        decoded: dict[str, Any] = {"action": action, "type": "noop"}
        if action > 0:
            d_idx, p_idx = divmod(action - 1, self.k)
            decoded = {"action": action, "type": "reroute",
                       "demand": self.eng.demands[d_idx].id,
                       "demand_idx": d_idx, "path_idx": p_idx,
                       "from_path": int(self.eng.current_path[d_idx])}
            if self.safety_filter:
                ok, reason = self.eng.validate_action(d_idx, p_idx, source="rl")
                if ok:
                    ok, reason = self.eng.apply_action(d_idx, p_idx, source="rl")
            else:
                ok, reason = self.eng.apply_action(d_idx, p_idx, source="rl")
            decoded["accepted"] = ok
            decoded["reason"] = reason
            rerouted = ok
            invalid = not ok
            flapped = ok and self.eng.action_log[-1].is_flap

        interval = self.eng.step_interval()
        reward, components = compute_reward(interval, rerouted, flapped, invalid, self.reward_cfg)
        terminated = self.eng.all_disconnected
        truncated = self.eng.done
        info = {
            "metrics": interval,
            "reward_components": components,
            "decoded_action": decoded,
            "action_mask": self.action_masks(),
        }
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ masks
    def action_masks(self) -> np.ndarray:
        """Boolean mask over the discrete action space (True = allowed).

        Semantically identical to calling ``validate_action(d, p, "rl")`` for
        every action - and verified as such against a per-action sweep in
        tests/test_runtime_equivalence.py - but the three constraints that can
        be evaluated for all candidates at once (candidate exists, path is up,
        demand is not on that path already, demand is not cooling down) are
        applied as array operations. Only the protected-class bandwidth check,
        which needs a projected-load computation, still runs per candidate, and
        only for candidates that survived the cheap checks.
        """
        eng = self.eng
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True  # no-op is always legal

        # candidate exists AND every hop is operational
        allowed = eng.candidate_available_matrix()

        # "already on this path" (a disconnected demand may re-select its path)
        live = ~eng.disconnected
        allowed[self._demand_arange[live], eng.current_path[live]] = False

        # per-demand reroute cooldown (applies to source "rl")
        allowed[eng.step_count < eng.cooldown_until] = False

        # protected classes must keep non-negative projected headroom
        for d_idx in self._protected_idx:
            cands = np.flatnonzero(allowed[d_idx])
            if cands.size == 0:
                continue
            base = eng._projected_base_loads(int(d_idx), eng._sweep_buf)
            for p_idx in cands:
                if eng._headroom_from_base(base, int(d_idx), int(p_idx)) < 0.0:
                    allowed[d_idx, p_idx] = False

        mask[1:1 + self.n_demands * self.k] = allowed.ravel()
        return mask

    # -------------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        eng = self.eng
        link = np.empty((LINK_FEATURES, self.n_dlinks), dtype=np.float32)
        link[0] = np.clip(eng.link_util / 2.0, 0, 1)
        link[1] = np.clip(eng.link_qdelay / m.Q_MAX_MS, 0, 1)
        link[2] = np.clip(eng.link_loss, 0, 1)
        link[3] = eng._dlink_up
        link[4] = np.clip(eng.util_ewma / 2.0, 0, 1)

        # Same feature ordering and normalizations as the original per-demand
        # loop, written as whole-row assignments. dm is (features, demands) and
        # is raveled feature-major, so row order defines the layout the
        # pretrained model expects - do not reorder these rows.
        k = self.k
        dm = np.zeros((self.demand_features, self.n_demands), dtype=np.float32)
        dm[0] = np.minimum(eng.demand_volumes / self._two_base_mbps, 1.0)
        dm[1] = self._obs_priority
        dm[2] = self._obs_max_latency
        dm[3] = self._obs_max_loss
        dm[4] = self._obs_protected

        cur = eng.current_path
        in_range = cur < k
        dm[5 + cur[in_range], self._demand_arange[in_range]] = 1.0

        # per-candidate bottleneck utilization; absent candidates read as 1.0
        bottleneck = np.minimum(eng.candidate_bottleneck_matrix() / 2.0, 1.0)
        dm[5 + k:5 + 2 * k] = np.where(eng._cand_exists, bottleneck, 1.0).T

        cd = np.maximum(0, eng.cooldown_until - eng.step_count)
        dm[5 + 2 * k] = np.minimum(
            cd / max(1, self.engine_cfg.reroute_cooldown_steps), 1.0)
        dm[6 + 2 * k] = eng.disconnected

        mtr = eng.metrics_history[-1] if eng.metrics_history else None
        hour = (eng.scenario.start_hour + eng.t_min / 60.0) % 24.0
        theta = 2.0 * np.pi * hour / 24.0
        glob = np.array([
            0.5 + 0.5 * np.sin(theta),
            0.5 + 0.5 * np.cos(theta),
            min(float(np.max(eng.link_util)) / 2.0, 1.0),
            min(float(np.mean(eng.link_util)), 1.0),
            min(float(np.std(eng.link_util)) / 0.5, 1.0),
            min((mtr["mean_delay_ms"] if mtr else 0.0) / 60.0, 1.0),
            min((mtr["loss_ratio"] if mtr else 0.0) / 0.05, 1.0),
            (mtr["sla_violation_fraction"] if mtr else 0.0),
            (mtr["delivered_ratio"] if mtr else 1.0),
            min((mtr["reroutes"] if mtr else 0) / 5.0, 1.0),
            min(eng.step_count / max(1, self._steps_total), 1.0),
        ], dtype=np.float32)
        return np.concatenate([link.ravel(), dm.ravel(), glob])
