"""Masked neural contextual bandit for the frozen MPLS-TE V2 environment.

The learner predicts an immediate reward for each action.  Training gathers
only the selected action head and regresses it onto the reward observed for
that transition.  Deliberately absent: next observations, discounts, terminal
flags, target networks, Bellman targets, and future-value bootstrapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    rewards: np.ndarray


class ImmediateRewardReplay:
    """Fixed-capacity replay containing contextual-bandit feedback only."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.observations = np.empty(
            (capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.masks = np.empty((capacity, action_dim), dtype=bool)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, observation: np.ndarray, action: int,
            mask: np.ndarray, immediate_reward: float) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = int(action)
        self.masks[self.position] = mask
        self.rewards[self.position] = float(immediate_reward)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError(
                f"replay contains {self.size} entries, need {batch_size}")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return self._batch(indices)

    def sample_all(self) -> ReplayBatch:
        return self._batch(np.arange(self.size))

    def _batch(self, indices: np.ndarray) -> ReplayBatch:
        return ReplayBatch(
            observations=self.observations[indices].copy(),
            actions=self.actions[indices].copy(),
            masks=self.masks[indices].copy(),
            rewards=self.rewards[indices].copy(),
        )

    def __len__(self) -> int:
        return self.size


class ImmediateRewardNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, int(hidden)), nn.ReLU()))
            width = int(hidden)
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(width, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(observations))


def selected_action_huber_loss(
    predictions: torch.Tensor,
    actions: torch.Tensor,
    immediate_rewards: torch.Tensor,
) -> torch.Tensor:
    """Huber loss on selected action heads and immediate rewards only."""
    selected = predictions.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    return F.smooth_l1_loss(selected, immediate_rewards.float())


class MaskedContextualBandit:
    """Neural immediate-reward regressor with masked epsilon-greedy actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        seed: int,
        config: Mapping[str, Any],
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.config = dict(config)
        hidden = [int(x) for x in config["network"]]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = ImmediateRewardNetwork(
                self.observation_dim, self.action_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=float(config["learning_rate"]))
        self.replay = ImmediateRewardReplay(
            int(config["replay_capacity"]), self.observation_dim, self.action_dim)
        self.rng = np.random.default_rng(self.seed)
        self.transitions = 0
        self.updates = 0
        self.last_loss: float | None = None

    def epsilon(self) -> float:
        exploration = self.config["exploration"]
        initial = float(exploration["initial_epsilon"])
        final = float(exploration["final_epsilon"])
        decay = max(1, int(exploration["decay_transitions"]))
        fraction = min(1.0, self.transitions / decay)
        return initial + fraction * (final - initial)

    def predict(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        valid = np.asarray(masks, dtype=bool)
        if obs.ndim == 1:
            obs = obs[None, :]
        if valid.ndim == 1:
            valid = valid[None, :]
        if obs.shape != (len(obs), self.observation_dim):
            raise ValueError(
                f"observation shape {obs.shape} != (*,{self.observation_dim})")
        if valid.shape != (len(obs), self.action_dim):
            raise ValueError(f"mask shape {valid.shape} != (*,{self.action_dim})")
        if np.any(~valid.any(axis=1)):
            raise ValueError("every action mask must contain at least one valid action")

        self.network.eval()
        with torch.no_grad():
            scores = self.network(
                torch.as_tensor(obs, device=self.device)).cpu().numpy()
        scores[~valid] = -np.inf
        actions = np.argmax(scores, axis=1).astype(np.int64)
        if deterministic:
            return actions
        epsilon = self.epsilon()
        for row in range(len(actions)):
            if self.rng.random() < epsilon:
                choices = np.flatnonzero(valid[row])
                actions[row] = choices[int(self.rng.integers(len(choices)))]
        return actions

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        immediate_rewards: np.ndarray,
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32)
        acts = np.asarray(actions, dtype=np.int64).reshape(-1)
        valid = np.asarray(masks, dtype=bool)
        rewards = np.asarray(immediate_rewards, dtype=np.float32).reshape(-1)
        if not (len(obs) == len(acts) == len(valid) == len(rewards)):
            raise ValueError("bandit feedback batch lengths disagree")
        if not np.all(valid[np.arange(len(acts)), acts]):
            raise ValueError("bandit feedback contains an invalid selected action")
        if not np.all(np.isfinite(rewards)):
            raise FloatingPointError("bandit feedback contains non-finite rewards")
        for row in range(len(acts)):
            self.replay.add(obs[row], int(acts[row]), valid[row], float(rewards[row]))
        self.transitions += len(acts)

    def update(self) -> dict[str, float] | None:
        batch_size = int(self.config["batch_size"])
        warmup = int(self.config["warmup_transitions"])
        if self.transitions < warmup or len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size, self.rng)
        observations = torch.as_tensor(
            batch.observations, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        self.network.train()
        predictions = self.network(observations)
        loss = selected_action_huber_loss(predictions, actions, rewards)
        if not torch.isfinite(loss):
            raise FloatingPointError("bandit loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.network.parameters(), float(self.config["gradient_clip_norm"]))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("bandit gradient norm is non-finite")
        self.optimizer.step()
        self.updates += 1
        self.last_loss = float(loss.detach().cpu())
        return {
            "loss": self.last_loss,
            "gradient_norm": float(grad_norm.detach().cpu()),
            "epsilon": self.epsilon(),
            "updates": float(self.updates),
        }

    def save(self, path: Path) -> None:
        """Save resumable learner state; integrity hashes live in its sidecar."""
        replay = self.replay.sample_all()
        torch.save({
            "format": "masked-contextual-bandit-v1",
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "seed": self.seed,
            "config": self.config,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "transitions": self.transitions,
            "updates": self.updates,
            "last_loss": self.last_loss,
            "rng_state": self.rng.bit_generator.state,
            "replay": {
                "observations": replay.observations,
                "actions": replay.actions,
                "masks": replay.masks,
                "rewards": replay.rewards,
            },
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "MaskedContextualBandit":
        """Load a checkpoint after the caller validates its hash sidecar."""
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format") != "masked-contextual-bandit-v1":
            raise ValueError(f"{path}: unsupported bandit checkpoint format")
        learner = cls(
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            device=device,
            seed=int(payload["seed"]),
            config=payload["config"],
        )
        learner.network.load_state_dict(payload["network_state"])
        learner.optimizer.load_state_dict(payload["optimizer_state"])
        replay = payload["replay"]
        for index in range(len(replay["actions"])):
            learner.replay.add(
                replay["observations"][index],
                int(replay["actions"][index]),
                replay["masks"][index],
                float(replay["rewards"][index]),
            )
        learner.transitions = int(payload["transitions"])
        learner.updates = int(payload["updates"])
        learner.last_loss = payload["last_loss"]
        learner.rng.bit_generator.state = payload["rng_state"]
        return learner
