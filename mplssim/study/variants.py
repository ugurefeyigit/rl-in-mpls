"""Controlled-coupling environment variant: delayed TE activation.

Identity ``mpls-te-v2.0.0+delay-L`` (never confused with the frozen V2 env).

Motivation
----------
In the frozen V2 environment a TE change requested at boundary ``t`` carries
traffic during interval ``t``, so the immediate reward already contains the
first interval of the change's effect. That is what lets a myopic learner
work. Real controllers have an actuation latency between decision and
effect -- path computation, signalling, make-before-break, or simply a
periodic commit cycle. With a 5-minute control interval, ``L = 1`` models a
decision committed at the next cycle.

Semantics with delay ``L >= 1``
-------------------------------
* A request is validated against the frozen V2 mask *and* the demand must not
  already have a pending request. If accepted, its operational move cost
  (fixed + moved-volume share + edge divergence + predicted reversal) is
  charged **at request time**: the operator commits to the change now.
* The route change is activated at the boundary ``L`` intervals later, before
  that interval is simulated, through the frozen engine's own
  ``apply_te_action``. If the change is no longer legal then (failed link,
  FRR already placed the demand there, or the protected-class projection now
  exceeds 100 %), it is cancelled and counted; no further cost is charged.
* The observation appends, per demand, a one-hot of the pending target path
  (``k`` blocks) and the fraction of the delay remaining (one block), so the
  state remains Markov: ``604 + (k + 1) * n_demands = 689`` features.

With ``L = 0`` no request is ever pending, the appended blocks are all zero,
and ``tests/test_study.py`` asserts rewards, masks and the base observation
are bit-identical to :class:`MplsTeEnvV2` along random legal trajectories.

The immediate reward of a request under ``L >= 1`` contains only its cost, so
a learner that regresses immediate reward cannot, by construction, credit the
change with its effect. The variant therefore defines a regime in which
sequential credit assignment is necessary, and asks whether the sequential
learners actually recover the lost performance under the same budget.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from gymnasium import spaces

from mplssim.rl.env_v2 import MplsTeEnvV2
from mplssim.rl.reward_v2 import compute_reward_v2, potential, utility


class DelayedTeEnvV2(MplsTeEnvV2):
    """Frozen V2 environment plus a fixed TE activation delay of ``L`` intervals."""

    def __init__(self, *args: Any, delay_steps: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        self.delay_steps = int(delay_steps)
        self._base_dim = int(self.observation_space.shape[0])
        dim = self._base_dim + (self.k + 1) * self.n_demands
        self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)
        self._reset_pending()
        self.cancelled_activations = 0

    # ------------------------------------------------------------ identity
    def environment_versions(self) -> dict[str, str]:
        v = super().environment_versions()
        v["environment"] = f"{v['environment']}+delay-{self.delay_steps}"
        v["observation"] = f"{v['observation']}+pending-{self.observation_space.shape[0]}"
        return v

    # --------------------------------------------------------------- state
    def _reset_pending(self) -> None:
        self.pending_path = np.full(self.n_demands, -1, dtype=np.int64)
        self.pending_due = np.full(self.n_demands, -1, dtype=np.int64)

    def snapshot_state(self) -> tuple[Any, np.ndarray, np.ndarray, int]:
        return (self.eng.fast_clone(), self.pending_path.copy(),
                self.pending_due.copy(), self.cancelled_activations)

    def restore_state(self, snap: tuple[Any, np.ndarray, np.ndarray, int]) -> None:
        eng, path, due, cancelled = snap
        self.eng, self.pending_path, self.pending_due = eng, path.copy(), due.copy()
        self.cancelled_activations = cancelled

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_pending()
        self.cancelled_activations = 0
        info["environment_versions"] = self.environment_versions()
        return self._full_obs(obs), info

    # --------------------------------------------------------------- masks
    def action_masks(self) -> np.ndarray:
        mask = super().action_masks()
        busy = np.flatnonzero(self.pending_path >= 0)
        for d in busy:
            mask[1 + d * self.k: 1 + (d + 1) * self.k] = False
        return mask

    # --------------------------------------------------------- observation
    def _full_obs(self, base: np.ndarray) -> np.ndarray:
        extra = np.zeros((self.k + 1, self.n_demands), dtype=np.float32)
        has = self.pending_path >= 0
        rows = np.flatnonzero(has)
        extra[self.pending_path[has], rows] = 1.0
        if self.delay_steps > 0:
            remaining = np.where(has, self.pending_due - self.eng.step_count, 0)
            extra[self.k] = np.clip(remaining / self.delay_steps, 0.0, 1.0)
        return np.concatenate([base, extra.ravel()]).astype(np.float32, copy=False)

    # ---------------------------------------------------------------- step
    def step(self, action: int):
        if self.delay_steps == 0:
            obs, reward, term, trunc, info = super().step(action)
            info["environment_versions"] = self.environment_versions()
            return self._full_obs(obs), reward, term, trunc, info

        eng = self.eng
        cfg = self.reward_cfg
        phi_current = potential(utility(eng.boundary_metrics(), cfg), cfg)
        action = int(action)
        mask = self.action_masks()
        accepted = rejected = reversal = False
        volume_share = edge_divergence = 0.0
        decoded: dict[str, Any] = {"action": action, "type": "noop", "accepted": False}
        if action < 0 or action >= self.action_space.n:
            rejected = True
            decoded = {"action": action, "type": "out_of_range", "accepted": False}
        elif action > 0:
            d, p = divmod(action - 1, self.k)
            if bool(mask[action]):
                accepted = True
                cur = int(eng.current_path[d])
                total = float(np.sum(eng.demand_offered))
                volume_share = float(eng.demand_offered[d] / max(total, 1e-12))
                e_old, e_new = eng._cand_edges[d][cur], eng._cand_edges[d][p]
                union = e_old | e_new
                edge_divergence = len(e_old ^ e_new) / len(union) if union else 0.0
                # Predicted reversal at activation time (same rule as the engine).
                reversal = bool(int(eng.previous_te_path[d]) == p and
                                eng.step_count + self.delay_steps - int(eng.last_te_step[d])
                                <= eng.cfg.reversal_window_steps)
                self.pending_path[d] = p
                self.pending_due[d] = eng.step_count + self.delay_steps
            else:
                rejected = True
            decoded = {"action": action, "type": "te_request", "demand_idx": d,
                       "path_idx": p, "accepted": accepted, "pending": accepted,
                       "reversal": reversal, "volume_share": volume_share,
                       "edge_divergence": edge_divergence}

        # Activate every request that is due at this boundary.
        activated = cancelled = 0
        for d in np.flatnonzero((self.pending_path >= 0) & (self.pending_due <= eng.step_count)):
            rec = eng.apply_te_action(int(d), int(self.pending_path[d]))
            if rec["accepted"]:
                activated += 1
            else:
                # apply_te_action counted a rejected request; an expired request is
                # a cancellation, not a policy rejection, so undo that counter.
                eng.rejected_te_requests -= 1
                eng.episode_totals["rejected_te_requests"] -= 1
                cancelled += 1
            self.pending_path[d] = -1
            self.pending_due[d] = -1
        self.cancelled_activations += cancelled

        interval = eng.step_interval()
        phi_next = potential(utility(eng.boundary_metrics(), cfg), cfg)
        reward, components = compute_reward_v2(
            interval, phi_current, phi_next, accepted=accepted,
            volume_share=volume_share, edge_divergence=edge_divergence,
            reversal=reversal, rejected=rejected, cfg=cfg)
        info = {
            "environment_versions": self.environment_versions(),
            "episode_seed": self.episode_seed, "metrics": interval,
            "reward_components": components, "decoded_action": decoded,
            "action_mask": self.action_masks(),
            "accepted_te_changes": interval["accepted_te_changes"],
            "rejected_te_requests": interval["rejected_te_requests"],
            "te_reversals": interval["te_reversals"],
            "flow_solver_iterations_max": interval["flow_solver_iterations_max"],
            "episode_totals": dict(eng.episode_totals),
            "activated_requests": activated, "cancelled_activations": cancelled,
            "phi_current": phi_current, "phi_next": phi_next,
        }
        return self._full_obs(super()._obs()), float(reward), False, bool(eng.done), info


def make_variant_factory(spec: dict[str, Any]):
    """Environment factory for an experiment registry ``env`` block."""
    from mplssim.experiments.v2_factory import make_env_v2
    variant = spec.get("variant", "base")
    if variant == "base":
        return make_env_v2
    if variant == "noshaping":
        # Training-only reward ablation: potential-shaping coefficient 0. Never used
        # for evaluation, which always scores policies with the primary reward.
        from dataclasses import replace
        from mplssim.rl.reward_v2 import load_reward_config_v2
        cfg = replace(load_reward_config_v2(), potential_coefficient=0.0)

        def factory(scenario: str, root_seed: int = 0, worker_rank: int = 0):
            return make_env_v2(scenario=scenario, root_seed=root_seed,
                               worker_rank=worker_rank, reward_cfg=cfg)
        return factory
    if variant == "delayed":
        L = int(spec["delay_steps"])

        def factory(scenario: str, root_seed: int = 0, worker_rank: int = 0):
            return DelayedTeEnvV2(scenario=scenario, root_seed=root_seed,
                                  worker_rank=worker_rank, delay_steps=L)
        return factory
    raise KeyError(f"unknown environment variant {variant!r}")
