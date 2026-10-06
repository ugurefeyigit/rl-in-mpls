"""Masked value learners with a configurable horizon.

:class:`MaskedQLearner` generalizes the closed study's
:class:`~mplssim.experiments.masked_bandit.MaskedContextualBandit` by one
parameter, the discount ``gamma``. Everything else -- network, optimizer,
replay capacity, batch size, update cadence, masked epsilon-greedy schedule,
Huber loss, gradient clipping -- is inherited unchanged, so a sweep over
``gamma`` isolates the effect of *bootstrapping future value* from every
other design decision.

* ``gamma == 0``: the update delegates to the parent class and is the masked
  contextual bandit exactly (asserted in ``tests/test_study.py``).
* ``gamma > 0``: masked double-DQN. The regression target is the normalized
  discounted return

      y = (1 - gamma) * r + gamma * Q_target(s', argmax_{a' legal in s'} Q(s', a'))

  The ``(1 - gamma)`` normalization keeps target magnitudes comparable across
  ``gamma`` (the Huber transition point therefore means the same thing) and
  does not change the greedy policy, because it rescales every action value of
  a fixed ``gamma`` by the same positive constant.

Episode ends in V2 are time-limit truncations at scenario end. The terminal
state's action mask is not available after the vector environment's
auto-reset, so the final transition of each episode is treated as terminal
(no bootstrap). This affects 1 in 288 training transitions on ``random_day``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mplssim.experiments.masked_bandit import MaskedContextualBandit


class TransitionReplay:
    """Replay with next observation, next mask and terminal flag."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = int(capacity)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_masks = np.empty((capacity, act_dim), dtype=bool)
        self.terminal = np.empty(capacity, dtype=bool)
        self.position = 0
        self.size = 0

    def add(self, next_obs: np.ndarray, next_mask: np.ndarray, terminal: bool) -> None:
        self.next_obs[self.position] = next_obs
        self.next_masks[self.position] = next_mask
        self.terminal[self.position] = bool(terminal)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class MaskedQLearner(MaskedContextualBandit):
    """Masked epsilon-greedy value learner; ``gamma = 0`` is the bandit."""

    def __init__(self, observation_dim: int, action_dim: int, device: torch.device,
                 seed: int, config: Mapping[str, Any]) -> None:
        super().__init__(observation_dim, action_dim, device, seed, config)
        self.gamma = float(config.get("gamma", 0.0))
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")
        self.target_update_every = int(config.get("target_update_every", 250))
        self.next_replay = TransitionReplay(
            int(config["replay_capacity"]), self.observation_dim, self.action_dim)
        self.target = copy.deepcopy(self.network).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

    def observe_transitions(self, observations, actions, masks, rewards,
                            next_observations, next_masks, terminals) -> None:
        """Bandit feedback plus the successor information bootstrapping needs."""
        self.observe(observations, actions, masks, rewards)
        for row in range(len(actions)):
            self.next_replay.add(next_observations[row], next_masks[row], terminals[row])
        if self.next_replay.position != self.replay.position:
            raise RuntimeError("replay buffers fell out of alignment")

    def update(self) -> dict[str, float] | None:
        if self.gamma == 0.0:
            return super().update()
        batch_size = int(self.config["batch_size"])
        warmup = int(self.config["warmup_transitions"])
        if self.transitions < warmup or len(self.replay) < batch_size:
            return None
        idx = self.rng.choice(len(self.replay), size=batch_size, replace=False)
        dev = self.device
        obs = torch.as_tensor(self.replay.observations[idx], device=dev)
        act = torch.as_tensor(self.replay.actions[idx], device=dev)
        rew = torch.as_tensor(self.replay.rewards[idx], device=dev)
        nobs = torch.as_tensor(self.next_replay.next_obs[idx], device=dev)
        nmask = torch.as_tensor(self.next_replay.next_masks[idx], device=dev)
        term = torch.as_tensor(self.next_replay.terminal[idx], device=dev)
        with torch.no_grad():
            self.network.eval()
            online_next = self.network(nobs).masked_fill(~nmask, -torch.inf)
            a_star = online_next.argmax(dim=1, keepdim=True)
            q_next = self.target(nobs).gather(1, a_star).squeeze(1)
            q_next = torch.where(term, torch.zeros_like(q_next), q_next)
            target = (1.0 - self.gamma) * rew + self.gamma * q_next
        self.network.train()
        pred = self.network(obs).gather(1, act.long().unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Q-learner loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.network.parameters(), float(self.config["gradient_clip_norm"]))
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_update_every == 0:
            self.target.load_state_dict(self.network.state_dict())
        self.last_loss = float(loss.detach().cpu())
        return {"loss": self.last_loss, "gradient_norm": float(grad_norm),
                "epsilon": self.epsilon(), "updates": float(self.updates)}

    def save_policy(self, path: Path) -> None:
        """Inference-only checkpoint (network + config); no replay buffer."""
        torch.save({"format": "masked-q-policy-v1", "observation_dim": self.observation_dim,
                    "action_dim": self.action_dim, "seed": self.seed,
                    "config": self.config, "network_state": self.network.state_dict(),
                    "transitions": self.transitions, "updates": self.updates}, path)

    @classmethod
    def load_policy(cls, path: Path, device: torch.device | str = "cpu") -> "MaskedQLearner":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format") != "masked-q-policy-v1":
            raise ValueError(f"{path}: not a masked-q policy checkpoint")
        cfg = dict(payload["config"])
        cfg["replay_capacity"] = 1  # inference only; no replay needed
        learner = cls(payload["observation_dim"], payload["action_dim"],
                      torch.device(device), payload["seed"], cfg)
        learner.network.load_state_dict(payload["network_state"])
        learner.transitions = int(payload["transitions"])
        learner.updates = int(payload["updates"])
        return learner
