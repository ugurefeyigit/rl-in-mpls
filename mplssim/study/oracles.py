"""Clairvoyant lookahead reference policies built on exact engine cloning.

These are *reference points*, not deployable controllers. Cloning the engine
clones the traffic model's random generator, so a rollout from a clone sees
the exact future offered traffic (AR noise, bursts, failures) of the episode.
They answer a scientific question rather than an engineering one: *how much
reward is available to a controller that optimizes over a horizon of H
control intervals, compared with one that optimizes only the next interval?*

``H = 1`` (:class:`MyopicOracle`) is the clairvoyant version of exactly the
target the masked contextual bandit regresses on: the immediate reward of
each legal action. The bandit approximates it from the observation alone.

:class:`RolloutOracle` scores each first action by the H-interval return of
taking it and then following a fixed continuation policy (no-op by default).
Because routes persist until changed, no-op continuation measures the
*persistent* value of a configuration change over the next H intervals.
None of these is an optimal policy for the full episode; that would need a
search over 69^288 action sequences. They are labelled "clairvoyant H-step
greedy" everywhere they are reported.
"""

from __future__ import annotations

from typing import Any

import copy

import numpy as np


def snapshot(env: Any) -> Any:
    """Copy of every piece of mutable decision-relevant state of ``env``."""
    if hasattr(env, "snapshot_state"):
        return env.snapshot_state()
    return env.eng.fast_clone()


def restore(env: Any, snap: Any) -> None:
    if hasattr(env, "restore_state"):
        env.restore_state(snap)
    else:
        env.eng = snap


def freeze_exogenous(env: Any) -> None:
    """Hold offered traffic and link state at their current values (in place).

    Used only on a state that :func:`simulate` restores afterwards. The frozen
    rollout answers: *if nothing exogenous changed, what would this move be
    worth over H intervals?* -- i.e. the value of the move that is predictable
    from the current observation, without clairvoyance about future traffic,
    bursts or failures. The AR noise generator still advances but its output
    is ignored, so the restored state's RNG is untouched either way.
    """
    eng = env.eng
    frozen = eng.demand_offered.copy()
    traffic = copy.copy(eng.traffic)
    traffic.volumes = lambda t_min: frozen.copy()
    eng.traffic = traffic
    eng._process_link_events = lambda t_from, t_to: None


def simulate(env: Any, actions: list[int], continuation: str = "noop",
             horizon: int | None = None, gamma: float = 1.0,
             frozen: bool = False) -> tuple[float, list[float]]:
    """Return of an action sequence from the current state, then restore.

    ``actions`` are applied first; if ``horizon`` exceeds ``len(actions)`` the
    remaining steps follow ``continuation`` (only "noop" is supported, which is
    always legal). Stops early at scenario end. The environment is restored to
    its exact prior state afterwards, including RNG state.
    """
    saved = snapshot(env)  # restored in ``finally``; the live state is consumed
    rewards: list[float] = []
    try:
        if frozen:
            freeze_exogenous(env)
        steps = horizon if horizon is not None else len(actions)
        for i in range(steps):
            a = actions[i] if i < len(actions) else 0
            if i < len(actions) and a != 0 and not bool(env.action_masks()[a]):
                raise ValueError(f"simulated action {a} is not legal at rollout step {i}")
            _, r, _, truncated, _ = env.step(int(a))
            rewards.append(float(r))
            if truncated:
                break
    finally:
        restore(env, saved)
    ret = float(sum((gamma ** i) * r for i, r in enumerate(rewards)))
    return ret, rewards


def action_values(env: Any, mask: np.ndarray, horizon: int,
                  gamma: float = 1.0) -> dict[int, tuple[float, list[float]]]:
    """H-step no-op-continuation return of every legal first action."""
    out: dict[int, tuple[float, list[float]]] = {}
    for a in np.flatnonzero(mask):
        out[int(a)] = simulate(env, [int(a)], horizon=horizon, gamma=gamma)
    return out


def _argmax_prefer_noop(values: dict[int, float], tol: float = 1e-12) -> int:
    """Highest value; exact ties (within ``tol``) resolve to no-op, then lowest index."""
    best_a, best_v = 0, values.get(0, -np.inf)
    for a in sorted(values):
        if values[a] > best_v + tol:
            best_a, best_v = a, values[a]
    return best_a


class RolloutOracle:
    """Clairvoyant H-step greedy policy with no-op continuation."""

    def __init__(self, horizon: int, gamma: float = 1.0, name: str | None = None) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.name = name or f"oracle_h{self.horizon}"

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int:
        vals = action_values(env, mask, self.horizon, self.gamma)
        return _argmax_prefer_noop({a: v for a, (v, _) in vals.items()})


class MyopicOracle(RolloutOracle):
    """Clairvoyant one-interval greedy: argmax of the exact immediate reward."""

    def __init__(self) -> None:
        super().__init__(horizon=1, name="oracle_h1")
