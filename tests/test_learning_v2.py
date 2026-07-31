"""Focused tests for the governed V2 learning comparison tooling."""

from __future__ import annotations

import importlib.util
import inspect
import ast
import gzip
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

def test_learning_common_module_is_available() -> None:
    """Deleting the shared V2 experiment boundary must break this test."""
    assert importlib.util.find_spec("mplssim.experiments.learning_common") is not None


def test_governed_config_registers_only_the_two_v2_learners() -> None:
    """An accidental algorithm alias or V1 fallback must fail this contract."""
    from mplssim.experiments.learning_common import load_learning_config

    cfg = load_learning_config()
    assert cfg["version"] == "learning-v2.0"
    assert tuple(cfg["algorithms"]) == ("maskable_ppo", "masked_bandit")
    assert cfg["environment_version"] == "v2"
    assert cfg["training_roots"] == [42, 314159, 271828]
    assert cfg["active_training_roots"] == [42, 314159, 271828]
    assert cfg["continuity_seeds"] == [101, 102, 103, 104, 105]
    assert cfg["holdout_seeds"] == [1001, 1002, 1003, 1004, 1005]


def test_task_seed_policy_allows_preregistered_roots_and_continuity_only() -> None:
    """Rejecting a preregistered root or accepting an unknown root is a bug."""
    from mplssim.experiments.learning_common import (
        validate_evaluation_seeds,
        validate_training_root,
    )

    for root_seed in (42, 314159, 271828):
        validate_training_root(root_seed)
    validate_evaluation_seeds([101, 102, 103, 104, 105])
    with pytest.raises(ValueError, match="active preregistered training roots"):
        validate_training_root(7)
    with pytest.raises(ValueError, match="forbidden holdout"):
        validate_evaluation_seeds([1001])
    with pytest.raises(ValueError, match="continuity"):
        validate_evaluation_seeds([999])


def test_final_holdout_seed_gate_requires_explicit_complete_workflow() -> None:
    """Holdout seeds open only for the complete, explicitly named final matrix."""
    from mplssim.experiments.learning_common import validate_evaluation_seeds

    holdout = [1001, 1002, 1003, 1004, 1005]
    validate_evaluation_seeds(
        holdout, evaluation_mode="final_holdout", require_complete=True)
    with pytest.raises(ValueError, match="forbidden holdout"):
        validate_evaluation_seeds(holdout)
    with pytest.raises(ValueError, match="complete final holdout"):
        validate_evaluation_seeds(
            holdout[:-1], evaluation_mode="final_holdout", require_complete=True)
    with pytest.raises(ValueError, match="final holdout seeds"):
        validate_evaluation_seeds(
            [101], evaluation_mode="final_holdout", require_complete=False)
    with pytest.raises(ValueError, match="evaluation mode"):
        validate_evaluation_seeds(holdout, evaluation_mode="checkpoint_selection")


def test_run_directory_must_be_new(tmp_path: Path) -> None:
    """Reusing a run directory could silently mix checkpoints and metrics."""
    from mplssim.experiments.learning_common import create_run_directory

    fresh = tmp_path / "new-run"
    assert create_run_directory(fresh) == fresh.resolve()
    with pytest.raises(FileExistsError, match="already exists"):
        create_run_directory(fresh)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        create_run_directory(occupied)


def test_vector_transition_accounting_is_aggregate_and_exact() -> None:
    """Counting vector calls instead of aggregate transitions changes budgets."""
    from mplssim.experiments.learning_common import (
        require_exact_vector_budget,
        vector_step_transition_count,
    )

    assert vector_step_transition_count(vector_steps=7, n_envs=8) == 56
    require_exact_vector_budget(400_000, 8)
    require_exact_vector_budget(400_000, 16)
    with pytest.raises(ValueError, match="not divisible"):
        require_exact_vector_budget(400_000, 12)


def test_ppo_exact_rollout_boundary_allows_the_gradient_update() -> None:
    """Stopping at the final rollout callback must not skip SB3's train phase."""
    from mplssim.experiments.trainers_v2 import should_continue_ppo_rollout

    assert should_continue_ppo_rollout(
        num_timesteps=8_192,
        target_transitions=8_192,
        rollout_transitions=8_192,
    )
    assert not should_continue_ppo_rollout(
        num_timesteps=400_000,
        target_transitions=400_000,
        rollout_transitions=8_192,
    )
    assert should_continue_ppo_rollout(
        num_timesteps=391_216,
        target_transitions=400_000,
        rollout_transitions=8_192,
    )


def test_device_selection_is_truthful() -> None:
    """Resolved metadata must never claim CUDA for CPU tensors."""
    from mplssim.experiments.learning_common import resolve_device

    cpu = resolve_device("cpu")
    assert cpu.requested == "cpu"
    assert cpu.resolved == "cpu"
    assert cpu.torch_device == torch.device("cpu")

    auto = resolve_device("auto")
    assert auto.resolved == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            resolve_device("cuda")
    with pytest.raises(ValueError, match="auto, cuda, or cpu"):
        resolve_device("tpu")


def _tiny_bandit(*, epsilon: float = 0.0):
    from mplssim.experiments.masked_bandit import MaskedContextualBandit

    return MaskedContextualBandit(
        observation_dim=3,
        action_dim=4,
        device=torch.device("cpu"),
        seed=7,
        config={
            "network": [5],
            "learning_rate": 1e-3,
            "batch_size": 2,
            "replay_capacity": 8,
            "warmup_transitions": 2,
            "update_every_vector_steps": 1,
            "exploration": {
                "initial_epsilon": epsilon,
                "final_epsilon": epsilon,
                "decay_transitions": 1,
            },
            "gradient_clip_norm": 1.0,
        },
    )


def test_bandit_excludes_invalid_actions_from_greedy_and_exploration() -> None:
    """An invalid action with the highest score must never be selected."""
    bandit = _tiny_bandit(epsilon=1.0)
    with torch.no_grad():
        for parameter in bandit.network.parameters():
            parameter.zero_()
        bandit.network.output.bias.copy_(torch.tensor([0.0, 1.0, 99.0, 2.0]))

    obs = np.zeros((32, 3), dtype=np.float32)
    masks = np.tile(np.array([True, True, False, True]), (32, 1))
    exploratory = bandit.predict(obs, masks, deterministic=False)
    assert set(exploratory.tolist()) <= {0, 1, 3}
    greedy = bandit.predict(obs[:1], masks[:1], deterministic=True)
    assert greedy.tolist() == [3]


def test_selected_action_loss_backpropagates_only_selected_outputs() -> None:
    """Updating unselected action heads would turn partial feedback into fiction."""
    from mplssim.experiments.masked_bandit import selected_action_huber_loss

    predictions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    actions = torch.tensor([1, 0])
    immediate_rewards = torch.tensor([0.0, 10.0])
    loss = selected_action_huber_loss(predictions, actions, immediate_rewards)
    loss.backward()

    assert predictions.grad is not None
    assert predictions.grad[0, 1] != 0
    assert predictions.grad[1, 0] != 0
    assert torch.count_nonzero(predictions.grad).item() == 2


def test_bandit_replay_target_is_exact_immediate_reward() -> None:
    """The replay target must not contain a return or a future-value estimate."""
    bandit = _tiny_bandit()
    obs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    actions = np.array([0, 2])
    masks = np.array([[True, False, False, False],
                      [True, True, True, False]])
    rewards = np.array([1.25, -3.5], dtype=np.float32)

    bandit.observe(obs, actions, masks, rewards)
    batch = bandit.replay.sample_all()
    np.testing.assert_array_equal(batch.rewards, rewards)
    assert not hasattr(batch, "next_observations")
    assert not hasattr(batch, "discounts")
    assert not hasattr(batch, "dones")


def test_bandit_observe_api_has_no_bootstrap_inputs() -> None:
    """Adding next-state inputs would violate the contextual-bandit contract."""
    from mplssim.experiments.masked_bandit import MaskedContextualBandit

    assert tuple(inspect.signature(MaskedContextualBandit.observe).parameters) == (
        "self", "observations", "actions", "masks", "immediate_rewards")


def test_bandit_deterministic_prediction_is_repeatable() -> None:
    """Deterministic evaluation must disable epsilon exploration."""
    bandit = _tiny_bandit(epsilon=1.0)
    obs = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    mask = np.array([[True, False, True, True]])
    first = bandit.predict(obs, mask, deterministic=True)
    for _ in range(10):
        np.testing.assert_array_equal(
            bandit.predict(obs, mask, deterministic=True), first)


def test_bandit_checkpoint_round_trip_preserves_policy_and_feedback(
    tmp_path: Path,
) -> None:
    """A reloaded checkpoint must represent the same learner state."""
    from mplssim.experiments.masked_bandit import MaskedContextualBandit

    bandit = _tiny_bandit(epsilon=0.0)
    obs = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32)
    masks = np.array([[True, True, False, True],
                      [True, False, True, True]])
    actions = np.array([1, 2])
    rewards = np.array([0.5, -1.25], dtype=np.float32)
    bandit.observe(obs, actions, masks, rewards)
    assert bandit.update() is not None
    before = bandit.predict(obs, masks, deterministic=True)

    path = tmp_path / "bandit.pt"
    bandit.save(path)
    restored = MaskedContextualBandit.load(path, device=torch.device("cpu"))

    np.testing.assert_array_equal(
        restored.predict(obs, masks, deterministic=True), before)
    np.testing.assert_array_equal(
        restored.replay.sample_all().rewards, rewards)
    assert restored.transitions == 2
    assert restored.updates == 1


def test_audited_factory_constructs_v2_and_records_derived_episode_seeds() -> None:
    """A V1 fallback or unrecorded reset seed invalidates the experiment."""
    from mplssim.experiments.learning_common import SeedLedger, make_audited_env
    from mplssim.rl.env_v2 import MplsTeEnvV2

    ledger = SeedLedger()
    env = make_audited_env(
        scenario="full_day", root_seed=42, worker_rank=3, seed_ledger=ledger)
    assert isinstance(env.unwrapped, MplsTeEnvV2)
    _, first = env.reset()
    _, second = env.reset()

    assert first["episode_seed"] == 45
    assert second["episode_seed"] == 1069
    assert [row["episode_seed"] for row in ledger.records] == [45, 1069]
    assert all(row["root_seed"] == 42 for row in ledger.records)
    assert all(row["worker_rank"] == 3 for row in ledger.records)


def test_ppo_vector_seeding_cannot_replace_the_governed_root() -> None:
    """SB3's seed+rank reset must not double-count rank in episode seeds."""
    from mplssim.experiments.learning_common import SeedLedger
    from mplssim.experiments.trainers_v2 import build_training_vec

    ledger = SeedLedger()
    vec, _ = build_training_vec(
        scenario="random_day",
        root_seed=42,
        n_envs=2,
        seed_ledger=ledger,
    )
    try:
        vec.seed(42)
        vec.reset()
    finally:
        vec.close()

    assert [row["root_seed"] for row in ledger.records] == [42, 42]
    assert [row["worker_rank"] for row in ledger.records] == [0, 1]
    assert [row["episode_seed"] for row in ledger.records] == [42, 43]


def test_audited_env_rejects_an_invalid_action_before_transition() -> None:
    """Rollout code must not rely on the environment's rejected-action penalty."""
    from mplssim.experiments.learning_common import SeedLedger, make_audited_env

    env = make_audited_env(
        scenario="full_day", root_seed=42, worker_rank=0,
        seed_ledger=SeedLedger())
    env.reset()
    mask = env.action_masks()
    invalid = int(np.flatnonzero(~mask)[0])
    step_before = env.unwrapped.eng.step_count
    with pytest.raises(RuntimeError, match="invalid action"):
        env.step(invalid)
    assert env.unwrapped.eng.step_count == step_before
    assert env.integrity.invalid_action_attempts == 1


def test_audited_env_uses_authoritative_mask_and_reward_decomposition() -> None:
    """Mask disagreement or an inexact component sum must stop a run."""
    from mplssim.experiments.learning_common import SeedLedger, make_audited_env
    from mplssim.rl.reward_v2 import components_sum

    env = make_audited_env(
        scenario="full_day", root_seed=42, worker_rank=0,
        seed_ledger=SeedLedger())
    env.reset()
    before = env.action_masks()
    _, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    np.testing.assert_array_equal(info["action_mask"], env.action_masks())
    assert components_sum(info["reward_components"]) == reward
    assert env.integrity.mask_disagreements == 0
    assert env.integrity.reward_mismatches == 0
    assert env.integrity.protected_safety_failures == 0
    assert env.integrity.aggregate_transitions == 1
    assert before[0]


def test_seed_ledger_fails_on_a_derived_seed_collision() -> None:
    """The same derived seed may not identify two distinct worker episodes."""
    from mplssim.experiments.learning_common import SeedLedger

    ledger = SeedLedger()
    ledger.record({
        "root_seed": 42, "worker_rank": 0, "episode_seed": 42,
        "scenario": "full_day",
    })
    with pytest.raises(RuntimeError, match="seed collision"):
        ledger.record({
            "root_seed": 42, "worker_rank": 1, "episode_seed": 42,
            "scenario": "full_day",
        })


def test_current_definition_pin_is_embedded_and_verified() -> None:
    """Training may not start if the signed-off definition content drifts."""
    from mplssim.experiments.learning_common import verified_environment_record

    record = verified_environment_record(root_seed=42)
    assert record["environment"]["observation_dim"] == 604
    assert record["environment"]["action_dim"] == 69
    assert record["training_pin"]["pinned_environment_commit"] == (
        "dca533b5c6fa9953307d01470c23cac512eb2961")
    assert record["training_pin"]["frozen_definitions_verified"] is True


def test_metrics_writer_persists_machine_readable_steps_and_episodes(
    tmp_path: Path,
) -> None:
    """Dropping either metric stream would make a run unauditable."""
    from mplssim.experiments.learning_common import MetricsWriter

    with MetricsWriter(tmp_path) as writer:
        writer.write_step({"transition": 8, "reward": 1.25})
        writer.write_episode({"episode_seed": 42, "return": 7.5})

    with gzip.open(tmp_path / "training_steps.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert json.loads(fh.readline()) == {"transition": 8, "reward": 1.25}
    episode_lines = (tmp_path / "training_episodes.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert [json.loads(line) for line in episode_lines] == [
        {"episode_seed": 42, "return": 7.5}]


@pytest.mark.parametrize("root_seed", [42, 314159, 271828])
def test_checkpoint_sidecar_detects_payload_corruption(
    tmp_path: Path,
    root_seed: int,
) -> None:
    """Checkpoint bytes may not be loaded after their recorded hash changes."""
    from mplssim.experiments.learning_common import (
        validate_checkpoint_sidecar,
        verified_environment_record,
        write_checkpoint_sidecar,
    )

    payload = tmp_path / "checkpoint_000050000.pt"
    payload.write_bytes(b"trusted checkpoint")
    environment_record = verified_environment_record(root_seed=root_seed)
    sidecar = write_checkpoint_sidecar(
        payload,
        algorithm="masked_bandit",
        aggregate_transitions=50_000,
        environment_record=environment_record,
        run_config={
            "algorithm": "masked_bandit",
            "environment_version": "v2",
            "root_seed": root_seed,
            "n_envs": 8,
        },
        device={"requested": "auto", "resolved": "cpu"},
    )
    validated = validate_checkpoint_sidecar(
        payload, sidecar, expected_algorithm="masked_bandit",
        require_clean_source=False)
    assert validated["aggregate_transitions"] == 50_000

    payload.write_bytes(b"corrupted checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_checkpoint_sidecar(
            payload, sidecar, expected_algorithm="masked_bandit",
            require_clean_source=False)


def test_checkpoint_sidecar_binds_filename_transition_and_source(
    tmp_path: Path,
) -> None:
    """Renaming a smoke checkpoint must not make it a governed 50k checkpoint."""
    from mplssim.experiments.learning_common import (
        validate_checkpoint_sidecar,
        validate_source_identity,
        verified_environment_record,
        write_checkpoint_sidecar,
    )

    environment_record = verified_environment_record(
        root_seed=42, require_clean=False)
    validate_source_identity(
        environment_record["source"], require_clean=False)
    payload = tmp_path / "checkpoint_000000008.pt"
    payload.write_bytes(b"smoke")
    sidecar = write_checkpoint_sidecar(
        payload,
        algorithm="masked_bandit",
        aggregate_transitions=8,
        environment_record=environment_record,
        run_config={
            "algorithm": "masked_bandit",
            "environment_version": "v2",
            "root_seed": 42,
            "aggregate_transitions": 8,
            "checkpoint_interval": 4,
            "purpose": "smoke",
        },
        device={"requested": "cpu", "resolved": "cpu"},
    )
    renamed = tmp_path / "checkpoint_000050000.pt"
    payload.rename(renamed)
    with pytest.raises(ValueError, match="payload name|transition"):
        validate_checkpoint_sidecar(
            renamed, sidecar, expected_algorithm="masked_bandit",
            require_clean_source=False)


def test_checkpoint_sidecar_cross_source_mode_is_explicit_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the final workflow may load an ancestor-bound checkpoint by hash."""
    from mplssim.experiments.learning_common import (
        sha256_file,
        validate_checkpoint_sidecar,
        verified_environment_record,
        write_checkpoint_sidecar,
    )

    training_source = "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
    payload = tmp_path / "checkpoint_000050000.pt"
    payload.write_bytes(b"cross-source final holdout checkpoint")
    sidecar = write_checkpoint_sidecar(
        payload,
        algorithm="masked_bandit",
        aggregate_transitions=50_000,
        environment_record=verified_environment_record(42, require_clean=False),
        run_config={
            "algorithm": "masked_bandit",
            "environment_version": "v2",
            "root_seed": 42,
            "aggregate_transitions": 400_000,
            "checkpoint_interval": 50_000,
            "purpose": "meaningful",
        },
        device={"requested": "cpu", "resolved": "cpu"},
    )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["source"]["git_commit"] = training_source
    metadata["source"]["git_dirty"] = False
    metadata["environment_record"]["source"] = copy.deepcopy(metadata["source"])
    sidecar.write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    # Exercise the historical one-shot workflow at its actual evaluation SHA.
    # Post-study presentation commits must remain outside that loader allowlist.
    from mplssim.experiments import v2_factory
    monkeypatch.setattr(v2_factory, "git_metadata", lambda: {
        "git_commit": "f7ed0f407c50c5472ecff89f977bc656439a8c49",
        "git_branch": "feat/rl-environment-v2",
        "git_dirty": False,
    })

    with pytest.raises(ValueError, match="does not match the current checkout"):
        validate_checkpoint_sidecar(
            payload,
            sidecar,
            expected_algorithm="masked_bandit",
            require_clean_source=False,
        )
    validated = validate_checkpoint_sidecar(
        payload,
        sidecar,
        expected_algorithm="masked_bandit",
        require_clean_source=False,
        final_holdout_checkpoint_source_sha=training_source,
        expected_payload_sha256=sha256_file(payload),
    )
    assert validated["source"]["git_commit"] == training_source
    with pytest.raises(ValueError, match="expected final-holdout payload"):
        validate_checkpoint_sidecar(
            payload,
            sidecar,
            expected_algorithm="masked_bandit",
            require_clean_source=False,
            final_holdout_checkpoint_source_sha=training_source,
            expected_payload_sha256="0" * 64,
        )


def test_meaningful_checkpoint_metadata_rejects_smoke_or_unknown_root() -> None:
    """Checkpoint selection accepts every active preregistered 400k root."""
    from scripts.compare_v2 import validate_meaningful_checkpoint_metadata

    metadata = {
        "algorithm": "masked_bandit",
        "aggregate_transitions": 50_000,
        "run_config": {
            "algorithm": "masked_bandit",
            "environment_version": "v2",
            "root_seed": 42,
            "aggregate_transitions": 400_000,
            "checkpoint_interval": 50_000,
            "purpose": "meaningful",
        },
    }
    validate_meaningful_checkpoint_metadata(
        Path("checkpoint_000050000.pt"), metadata, "masked_bandit")
    for root_seed in (314159, 271828):
        validate_meaningful_checkpoint_metadata(
            Path("checkpoint_000050000.pt"),
            {
                **metadata,
                "run_config": {
                    **metadata["run_config"],
                    "root_seed": root_seed,
                },
            },
            "masked_bandit",
        )
    with pytest.raises(ValueError, match="400000"):
        validate_meaningful_checkpoint_metadata(
            Path("checkpoint_000050000.pt"),
            {
                **metadata,
                "run_config": {
                    **metadata["run_config"],
                    "aggregate_transitions": 8,
                    "purpose": "smoke",
                },
            },
            "masked_bandit",
        )
    with pytest.raises(ValueError, match="active preregistered training roots"):
        validate_meaningful_checkpoint_metadata(
            Path("checkpoint_000050000.pt"),
            {
                **metadata,
                "run_config": {**metadata["run_config"], "root_seed": 7},
            },
            "masked_bandit",
        )


def test_algorithm_registry_rejects_unknown_learners() -> None:
    """A typo must not fall through to another learner or to V1."""
    from mplssim.experiments.trainers_v2 import require_algorithm

    assert require_algorithm("maskable_ppo") == "maskable_ppo"
    assert require_algorithm("masked_bandit") == "masked_bandit"
    with pytest.raises(ValueError, match="maskable_ppo, masked_bandit"):
        require_algorithm("ppo")


def test_maskable_ppo_propagates_masks_through_rollout_and_reload(
    tmp_path: Path,
) -> None:
    """Rollout and both inference modes must exclude every invalid action."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    from mplssim.experiments.trainers_v2 import MaskablePpoLearner

    class StrictMaskEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                -1.0, 1.0, shape=(3,), dtype=np.float32)
            self.action_space = spaces.Discrete(4)
            self.mask_calls = 0
            self.invalid_attempts = 0
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {}

        def action_masks(self):
            self.mask_calls += 1
            return np.array([False, False, True, False])

        def step(self, action):
            if int(action) != 2:
                self.invalid_attempts += 1
                raise RuntimeError("invalid action reached toy environment")
            self.steps += 1
            return (
                np.zeros(3, dtype=np.float32),
                1.0,
                False,
                self.steps >= 4,
                {},
            )

    strict = StrictMaskEnv()
    vec = DummyVecEnv([lambda: strict])
    learner = MaskablePpoLearner(
        env=vec,
        device=torch.device("cpu"),
        seed=7,
        ppo_config={
            "learning_rate": 3e-4,
            "n_steps": 4,
            "batch_size": 4,
            "n_epochs": 1,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": [8]},
        },
        tensorboard_log=None,
    )
    learner.learn(total_timesteps=8)
    assert strict.mask_calls >= 8
    assert strict.invalid_attempts == 0

    obs = np.zeros((1, 3), dtype=np.float32)
    mask = np.array([[False, False, True, False]])
    for deterministic in (False, True):
        for _ in range(10):
            assert learner.predict(obs, mask, deterministic).tolist() == [2]

    path = tmp_path / "ppo.zip"
    learner.save(path)
    restored = MaskablePpoLearner.load(
        path, device=torch.device("cpu"), env=vec)
    for deterministic in (False, True):
        assert restored.predict(obs, mask, deterministic).tolist() == [2]


def test_v2_training_cli_has_explicit_shared_algorithm_and_device_options() -> None:
    """The shared CLI must not contain an environment switch that can select V1."""
    from scripts.train_v2 import parse_args

    args = parse_args([
        "--algorithm", "masked_bandit",
        "--run-dir", "runs/v2/example",
        "--purpose", "smoke",
        "--transitions", "8",
        "--n-envs", "2",
        "--device", "cpu",
    ])
    assert args.algorithm == "masked_bandit"
    assert args.device == "cpu"
    assert args.purpose == "smoke"
    assert not hasattr(args, "env_version")
    with pytest.raises(SystemExit):
        parse_args([
            "--algorithm", "unknown",
            "--run-dir", "runs/v2/example",
        ])


def test_tiny_bandit_training_persists_exact_metrics_seeds_and_checkpoints(
    tmp_path: Path,
) -> None:
    """A run must account for every vector transition and produce reloadable checkpoints."""
    from mplssim.experiments.learning_common import load_learning_config
    from mplssim.experiments.trainers_v2 import train_experiment

    cfg = copy.deepcopy(load_learning_config())
    cfg["masked_bandit"].update({
        "network": [8],
        "batch_size": 2,
        "replay_capacity": 32,
        "warmup_transitions": 2,
        "update_every_vector_steps": 1,
    })
    run_dir = tmp_path / "tiny-bandit"
    result = train_experiment(
        algorithm="masked_bandit",
        run_directory=run_dir,
        root_seed=42,
        scenario="evening_peak",
        aggregate_transitions=8,
        n_envs=2,
        checkpoint_interval=4,
        requested_device="cpu",
        learning_config=cfg,
        purpose="smoke",
        require_clean_checkout=False,
        command=["py", "scripts/train_v2.py", "--algorithm", "masked_bandit"],
    )

    assert result["status"] == "completed"
    assert result["aggregate_transitions"] == 8
    checkpoint_paths = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    assert [path.stem for path in checkpoint_paths] == [
        "checkpoint_000000004", "checkpoint_000000008"]
    assert all(Path(f"{path}.metadata.json").is_file() for path in checkpoint_paths)
    with gzip.open(run_dir / "training_steps.jsonl.gz", "rt", encoding="utf-8") as fh:
        step_rows = [json.loads(line) for line in fh]
    assert len(step_rows) == 8
    from mplssim.rl.reward_v2 import COMPONENT_ORDER
    first_step = step_rows[0]
    exact = 0.0
    for component in COMPONENT_ORDER:
        exact += first_step[f"rc_{component}"]
    assert first_step["reward_component_sum_exact"] is True
    assert exact == first_step["reward"]
    assert first_step["terminated"] is False
    assert first_step["truncated"] is False
    seeds = json.loads((run_dir / "episode_seeds.json").read_text(encoding="utf-8"))
    assert {row["episode_seed"] for row in seeds} == {42, 43}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["integrity"]["invalid_action_attempts"] == 0
    assert manifest["algorithm_diagnostics"]["updates"] >= 1
    assert manifest["run_config"]["command"][:2] == [
        "py", "scripts/train_v2.py"]


@pytest.mark.parametrize("name", ["static", "greedy", "cspf"])
def test_existing_baseline_adapter_submits_only_legal_v2_actions(name: str) -> None:
    """The existing controller proposals must be filtered through the V2 mask."""
    from mplssim.baselines import make_baseline
    from mplssim.experiments.evaluation_v2 import choose_baseline_action
    from mplssim.experiments.learning_common import SeedLedger, make_audited_env

    env = make_audited_env(
        scenario="evening_peak", root_seed=101, worker_rank=0,
        seed_ledger=SeedLedger())
    env.reset(options={"episode_seed": 101})
    controller = make_baseline(name, seed=101)
    for _ in range(8):
        mask = env.action_masks()
        action = choose_baseline_action(controller, env.unwrapped.eng, mask)
        assert mask[action]
        env.step(action)


def test_deterministic_evaluation_runs_the_governed_full_horizon() -> None:
    """Evaluation must disable exploration and must not shorten V2 episodes."""
    from mplssim.experiments.evaluation_v2 import run_evaluation_episode

    class NoopPolicy:
        def __init__(self) -> None:
            self.deterministic_flags: list[bool] = []

        def predict(self, observations, masks, deterministic):
            self.deterministic_flags.append(bool(deterministic))
            return np.zeros(len(observations), dtype=np.int64)

    policy = NoopPolicy()
    frame, summary = run_evaluation_episode(
        algorithm="masked_bandit",
        scenario="evening_peak",
        seed=101,
        policy=policy,
    )
    assert len(frame) == 84
    assert summary["episode_length"] == 84
    assert summary["truncated"] is True
    assert all(policy.deterministic_flags)
    assert summary["invalid_action_attempts"] == 0
    assert summary["reward_mismatches"] == 0
    assert summary["nonfinite_values"] == 0
    assert summary["reward_component_sum_exact"] is True


def test_checkpoint_selection_uses_return_then_earlier_exact_tie() -> None:
    """Training return or a later tie must not select the checkpoint."""
    from mplssim.experiments.evaluation_v2 import select_checkpoint

    rows = [
        {"checkpoint_transition": 50_000, "mean_operational_return": 4.0,
         "valid": True},
        {"checkpoint_transition": 100_000, "mean_operational_return": 7.0,
         "valid": True},
        {"checkpoint_transition": 150_000, "mean_operational_return": 7.0,
         "valid": True},
        {"checkpoint_transition": 200_000, "mean_operational_return": 99.0,
         "valid": False},
    ]
    assert select_checkpoint(rows)["checkpoint_transition"] == 100_000
    with pytest.raises(ValueError, match="no valid checkpoints"):
        select_checkpoint([{**rows[0], "valid": False}])


def test_evaluation_refuses_holdout_before_environment_construction() -> None:
    """The final holdout must remain untouched even by a malformed command."""
    from mplssim.experiments.evaluation_v2 import run_evaluation_episode

    with pytest.raises(ValueError, match="forbidden holdout"):
        run_evaluation_episode(
            algorithm="static", scenario="full_day", seed=1001, policy=None)


def test_final_holdout_matrix_rejects_partial_seed_set_before_output(
    tmp_path: Path,
) -> None:
    """A partial final matrix must fail before an output directory or env exists."""
    from mplssim.experiments.evaluation_v2 import evaluate_algorithm_matrix

    output = tmp_path / "partial-final-holdout"
    with pytest.raises(ValueError, match="complete final holdout"):
        evaluate_algorithm_matrix(
            algorithm="static",
            policy=None,
            scenarios=["full_day"],
            seeds=[1001, 1002, 1003, 1004],
            output_directory=output,
            write_steps=False,
            evaluation_mode="final_holdout",
        )
    assert not output.exists()


def test_final_holdout_registry_is_fixed_and_exposes_no_selection_inputs() -> None:
    """The one-shot workflow must evaluate only the six preregistered choices."""
    import scripts.final_holdout_v2 as final_holdout

    expected = [
        (42, "maskable_ppo", 250_000,
         "d34cc77ded05b064fa2a39dbe5c5ccc3126c9e6cf85e36c1b507127c987f5676",
         "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"),
        (42, "masked_bandit", 250_000,
         "c15097700eac518ee259cba67e34e4fba1716881ab3dd912188b55da0c79bf49",
         "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"),
        (314159, "maskable_ppo", 350_000,
         "0af41be78102617b103c3e21ebb0ba26ae251f2626ff50b30c0887fdb1320489",
         "6a8a4068b98bf9a71dead6e547595b4bbd755689"),
        (314159, "masked_bandit", 300_000,
         "fd474430e9f5ed60d09d82e3d08390151f54c8c0ca10b5abd98fe11d5d2c8433",
         "6a8a4068b98bf9a71dead6e547595b4bbd755689"),
        (271828, "maskable_ppo", 150_000,
         "40d0f9b7fe92449e6e8bfe2bcb44604ac2a5002c0f2a662dbad6cf70c219fb79",
         "6a8a4068b98bf9a71dead6e547595b4bbd755689"),
        (271828, "masked_bandit", 400_000,
         "d9c31430ad4320ae238f6d3aa833614edc120f7411c5a3e99372c85707116e73",
         "6a8a4068b98bf9a71dead6e547595b4bbd755689"),
    ]
    actual = [
        (spec.training_root, spec.algorithm, spec.checkpoint_transition,
         spec.payload_sha256, spec.training_source_sha)
        for spec in final_holdout.FINAL_HOLDOUT_CHECKPOINTS
    ]
    assert actual == expected

    option_strings = {
        option
        for action in final_holdout.build_parser()._actions
        for option in action.option_strings
    }
    assert not option_strings & {"--checkpoint", "--seeds", "--scenarios"}
    parsed = ast.parse(inspect.getsource(final_holdout))
    referenced_names = {
        node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}
    assert "select_checkpoint" not in referenced_names


def test_final_holdout_cross_source_gate_rejects_scientific_changes() -> None:
    """Cross-source loading may span only enumerated governance/evaluation files."""
    from mplssim.experiments.learning_common import (
        validate_final_holdout_changed_paths,
    )

    validate_final_holdout_changed_paths([
        "NEXT_STAGE_HANDOFF.md",
        "configs/experiments/learning_v2.yaml",
        "mplssim/experiments/learning_common.py",
        "mplssim/experiments/evaluation_v2.py",
        "scripts/compare_v2.py",
        "scripts/final_holdout_v2.py",
        "tests/test_learning_v2.py",
        "results/v2_three_root_continuity/manifest.json",
    ])
    for forbidden in (
        "CURRENT_SYSTEM_BASELINE.md",
        "frontend/study.html",
        "mplssim/rl/reward_v2.py",
        "mplssim/rl/env_v2.py",
        "mplssim/experiments/masked_bandit.py",
        "mplssim/experiments/trainers_v2.py",
    ):
        with pytest.raises(RuntimeError, match="scientific source change"):
            validate_final_holdout_changed_paths([forbidden])


def test_final_holdout_source_compatibility_requires_ancestor_and_exact_shas() -> None:
    """An unrelated checkout cannot claim comparability with a training source."""
    from mplssim.experiments.learning_common import (
        validate_final_holdout_source_compatibility,
    )

    training = "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
    evaluation = "6f5d4337431223b91a0dcaf6140ead07e306eaf1"
    validate_final_holdout_source_compatibility(
        training,
        evaluation,
        checkpoint_is_ancestor=True,
        changed_paths=["scripts/final_holdout_v2.py"],
    )
    with pytest.raises(RuntimeError, match="not an ancestor"):
        validate_final_holdout_source_compatibility(
            training,
            evaluation,
            checkpoint_is_ancestor=False,
            changed_paths=[],
        )
    with pytest.raises(RuntimeError, match="full SHA"):
        validate_final_holdout_source_compatibility(
            "ca64b62",
            evaluation,
            checkpoint_is_ancestor=True,
            changed_paths=[],
        )


def test_final_holdout_checkpoint_metadata_is_bound_to_registry() -> None:
    """Hash, source, root, algorithm, and transition drift must fail closed."""
    from scripts.final_holdout_v2 import (
        FINAL_HOLDOUT_CHECKPOINTS,
        validate_final_holdout_checkpoint_metadata,
    )

    spec = FINAL_HOLDOUT_CHECKPOINTS[0]
    metadata = {
        "algorithm": spec.algorithm,
        "aggregate_transitions": spec.checkpoint_transition,
        "payload_sha256": spec.payload_sha256,
        "source": {"git_commit": spec.training_source_sha},
        "run_config": {
            "algorithm": spec.algorithm,
            "environment_version": "v2",
            "root_seed": spec.training_root,
            "aggregate_transitions": 400_000,
            "checkpoint_interval": 50_000,
            "purpose": "meaningful",
        },
    }
    validate_final_holdout_checkpoint_metadata(
        metadata, spec, actual_payload_sha256=spec.payload_sha256)
    mutations = (
        ("payload_sha256", "0" * 64),
        ("aggregate_transitions", spec.checkpoint_transition + 50_000),
        ("algorithm", "masked_bandit"),
    )
    for field, value in mutations:
        changed = copy.deepcopy(metadata)
        changed[field] = value
        with pytest.raises(ValueError, match="final-holdout checkpoint"):
            validate_final_holdout_checkpoint_metadata(
                changed, spec, actual_payload_sha256=spec.payload_sha256)
    changed = copy.deepcopy(metadata)
    changed["source"]["git_commit"] = "0" * 40
    with pytest.raises(ValueError, match="final-holdout checkpoint"):
        validate_final_holdout_checkpoint_metadata(
            changed, spec, actual_payload_sha256=spec.payload_sha256)
    changed = copy.deepcopy(metadata)
    changed["run_config"]["root_seed"] = 7
    with pytest.raises(ValueError, match="final-holdout checkpoint"):
        validate_final_holdout_checkpoint_metadata(
            changed, spec, actual_payload_sha256=spec.payload_sha256)


def test_final_holdout_compact_tables_include_rewards_actions_and_integrity() -> None:
    """Compact evidence must retain the required operational and audit views."""
    from scripts.final_holdout_v2 import build_compact_tables

    row = {
        "policy_id": "root42_maskable_ppo",
        "algorithm": "maskable_ppo",
        "training_root": 42,
        "scenario": "full_day",
        "seed": 1001,
        "operational_return": 3.0,
        "reward_components": {"delivery": 4.0, "max_util": -1.0},
        "action_distribution": {"0": 2, "3": 1},
        "episode_length": 3,
        "truncated": True,
        "terminated": False,
        "reward_component_sum_exact": True,
        "invalid_action_attempts": 0,
        "mask_disagreements": 0,
        "reward_mismatches": 0,
        "nonfinite_values": 0,
        "solver_convergence_failures": 0,
        "protected_safety_failures": 0,
    }
    for column in (
        "offered_gbit_total", "delivered_gbit_total", "delivered_ratio_mean",
        "sla_violations_demand_intervals",
        "protected_disconnection_demand_intervals",
        "unprotected_disconnection_demand_intervals", "max_utilization_peak",
        "max_utilization_mean", "link_utilization_mean",
        "congested_link_intervals", "overload_ratio_mean", "delay_ms_mean",
        "delay_ms_max", "loss_ratio_mean", "accepted_te_changes",
        "reroutes_per_hour", "te_reversals", "flaps_per_demand",
        "moved_mbps_total", "dwell_active_demand_intervals",
        "dwell_remaining_mean", "rejected_te_requests", "frr_changes",
        "frr_disconnections", "recovery_restorations", "noop_frequency",
        "solver_iterations_mean", "solver_iterations_max",
        "mean_decision_time_ms", "mean_mask_time_ms", "wall_time_seconds",
    ):
        row[column] = 0.0
    tables = build_compact_tables(pd.DataFrame([row]))
    assert set(tables) == {
        "per_root_metrics", "aggregate_metrics", "scenario_metrics",
        "reward_components", "action_distribution", "evaluation_integrity",
    }
    assert int(tables["per_root_metrics"].iloc[0]["episodes"]) == 1
    assert int(tables["action_distribution"].query("action == 0").iloc[0]["count"]) == 2
    assert int(tables["action_distribution"].query("action == 3").iloc[0]["count"]) == 1
    assert bool(tables["evaluation_integrity"].iloc[0]["all_checks_passed"])
    reward = tables["reward_components"].iloc[0]
    assert reward["delivery_mean"] == 4.0
    assert reward["max_util_mean"] == -1.0
    assert reward["max_abs_reward_residual"] == 0.0


def test_final_holdout_evidence_requires_exactly_315_paired_episodes() -> None:
    """No learner, baseline, scenario, or holdout seed may be omitted or repeated."""
    from scripts.final_holdout_v2 import (
        FINAL_HOLDOUT_CHECKPOINTS,
        validate_complete_final_evidence,
    )

    scenarios = [
        "full_day", "evening_peak", "flash_crowd", "link_failure",
        "deceptive_local_optimum", "ood_double_failure", "overload_stress",
    ]
    seeds = [1001, 1002, 1003, 1004, 1005]
    policies = [
        f"root{spec.training_root}_{spec.algorithm}"
        for spec in FINAL_HOLDOUT_CHECKPOINTS
    ] + ["baseline_static", "baseline_greedy", "baseline_cspf"]
    rows = [
        {"policy_id": policy, "scenario": scenario, "seed": seed}
        for policy in policies
        for scenario in scenarios
        for seed in seeds
    ]
    evidence = pd.DataFrame(rows)
    validate_complete_final_evidence(evidence, scenarios=scenarios, seeds=seeds)
    assert len(evidence) == 315
    with pytest.raises(RuntimeError, match="315"):
        validate_complete_final_evidence(
            evidence.iloc[:-1], scenarios=scenarios, seeds=seeds)


def test_policy_loader_validates_sidecar_before_bandit_inference(
    tmp_path: Path,
) -> None:
    """Evaluation must use a hash- and identity-validated checkpoint."""
    from mplssim.experiments.evaluation_v2 import load_policy_checkpoint
    from mplssim.experiments.learning_common import (
        verified_environment_record,
        write_checkpoint_sidecar,
    )

    bandit = _tiny_bandit(epsilon=0.0)
    payload = tmp_path / "bandit.pt"
    bandit.save(payload)
    write_checkpoint_sidecar(
        payload,
        algorithm="masked_bandit",
        aggregate_transitions=50_000,
        environment_record=verified_environment_record(42),
        run_config={
            "algorithm": "masked_bandit",
            "environment_version": "v2",
            "root_seed": 42,
            "n_envs": 8,
        },
        device={"requested": "auto", "resolved": "cpu"},
    )
    restored, metadata = load_policy_checkpoint(
        payload, algorithm="masked_bandit", requested_device="cpu",
        require_clean_source=False)
    action = restored.predict(
        np.zeros((1, 3), dtype=np.float32),
        np.array([[True, False, True, False]]),
        deterministic=True,
    )
    assert action.shape == (1,)
    assert metadata["aggregate_transitions"] == 50_000


def test_policy_loader_accepts_only_explicit_final_holdout_source_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-source policy construction must carry the approved source and hash."""
    from mplssim.experiments.evaluation_v2 import load_policy_checkpoint
    from mplssim.experiments.learning_common import (
        sha256_file,
        verified_environment_record,
        write_checkpoint_sidecar,
    )

    training_source = "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
    bandit = _tiny_bandit(epsilon=0.0)
    payload = tmp_path / "checkpoint_000050000.pt"
    bandit.save(payload)
    sidecar = write_checkpoint_sidecar(
        payload,
        algorithm="masked_bandit",
        aggregate_transitions=50_000,
        environment_record=verified_environment_record(42, require_clean=False),
        run_config={
            "algorithm": "masked_bandit",
            "environment_version": "v2",
            "root_seed": 42,
            "aggregate_transitions": 400_000,
            "checkpoint_interval": 50_000,
            "purpose": "meaningful",
        },
        device={"requested": "cpu", "resolved": "cpu"},
    )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["source"]["git_commit"] = training_source
    metadata["source"]["git_dirty"] = False
    metadata["environment_record"]["source"] = copy.deepcopy(metadata["source"])
    sidecar.write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    # Keep this regression bound to the pushed one-shot evaluation source. The
    # current post-study checkout is intentionally not authorized to load it.
    from mplssim.experiments import v2_factory
    monkeypatch.setattr(v2_factory, "git_metadata", lambda: {
        "git_commit": "f7ed0f407c50c5472ecff89f977bc656439a8c49",
        "git_branch": "feat/rl-environment-v2",
        "git_dirty": False,
    })

    restored, loaded = load_policy_checkpoint(
        payload,
        algorithm="masked_bandit",
        requested_device="cpu",
        require_clean_source=False,
        final_holdout_checkpoint_source_sha=training_source,
        expected_payload_sha256=sha256_file(payload),
    )
    assert restored.predict(
        np.zeros((1, 3), dtype=np.float32),
        np.array([[True, False, True, False]]),
        deterministic=True,
    ).shape == (1,)
    assert loaded["source"]["git_commit"] == training_source


def test_evaluation_matrix_persists_steps_and_episode_summary(
    tmp_path: Path,
) -> None:
    """The shared evaluator must emit both granular and compact evidence."""
    from mplssim.experiments.evaluation_v2 import evaluate_algorithm_matrix

    output = tmp_path / "evaluation"
    result = evaluate_algorithm_matrix(
        algorithm="static",
        policy=None,
        scenarios=["evening_peak"],
        seeds=[101],
        output_directory=output,
        write_steps=True,
    )
    assert len(result) == 1
    assert result.iloc[0]["episode_length"] == 84
    assert (output / "episode_summary.csv").is_file()
    assert (output / "episode_summary.json").is_file()
    assert len(list((output / "steps").glob("*.csv.gz"))) == 1
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_algorithm_matrix(
            algorithm="static",
            policy=None,
            scenarios=["evening_peak"],
            seeds=[101],
            output_directory=output,
            write_steps=False,
        )


def test_benchmark_selection_requires_both_learners_and_exact_budget() -> None:
    """A faster but scientifically ineligible vector count must not win."""
    from scripts.benchmark_v2 import select_benchmark_configuration

    rows = [
        {"algorithm": "maskable_ppo", "n_envs": 8, "device": "cpu",
         "transitions_per_second": 90.0, "status": "completed"},
        {"algorithm": "masked_bandit", "n_envs": 8, "device": "cpu",
         "transitions_per_second": 110.0, "status": "completed"},
        {"algorithm": "maskable_ppo", "n_envs": 12, "device": "cpu",
         "transitions_per_second": 200.0, "status": "completed"},
        {"algorithm": "masked_bandit", "n_envs": 12, "device": "cpu",
         "transitions_per_second": 220.0, "status": "completed"},
        {"algorithm": "maskable_ppo", "n_envs": 16, "device": "cpu",
         "transitions_per_second": 120.0, "status": "completed"},
        {"algorithm": "masked_bandit", "n_envs": 16, "device": "cpu",
         "transitions_per_second": 130.0, "status": "completed"},
    ]
    selected = select_benchmark_configuration(rows)
    assert selected["n_envs"] == 16
    assert selected["device"] == "cpu"
    assert selected["eligible_counts"] == [8, 16]


def test_checkpoint_sweep_parses_counts_and_disqualifies_integrity_failures() -> None:
    """A checkpoint with any integrity failure cannot win on return."""
    from scripts.compare_v2 import (
        checkpoint_transition_from_path,
        checkpoint_summaries_are_valid,
        validate_checkpoint_schedule,
    )

    assert checkpoint_transition_from_path(
        Path("checkpoint_000150000.zip")) == 150_000
    complete = [
        Path(f"checkpoint_{transition:09d}.zip")
        for transition in range(50_000, 400_001, 50_000)
    ]
    validate_checkpoint_schedule(complete)
    with pytest.raises(ValueError, match="exact eight-checkpoint schedule"):
        validate_checkpoint_schedule(complete[1:])
    healthy = [{
        "reward_component_sum_exact": True,
        "invalid_action_attempts": 0,
        "mask_disagreements": 0,
        "solver_convergence_failures": 0,
        "protected_safety_failures": 0,
        "truncated": True,
        "terminated": False,
    }]
    assert checkpoint_summaries_are_valid(healthy)
    assert not checkpoint_summaries_are_valid([
        {**healthy[0], "mask_disagreements": 1}])


def test_compact_comparison_aggregates_repository_metric_names() -> None:
    """The final comparison must aggregate measured episode fields, not aliases."""
    from scripts.compare_v2 import aggregate_comparison

    episodes = pd.DataFrame([
        {
            "algorithm": "maskable_ppo", "operational_return": 10.0,
            "sla_violations_demand_intervals": 2,
            "reroutes_per_hour": 1.0, "max_utilization_mean": 0.8,
            "delivered_ratio_mean": 0.95, "te_reversals": 1,
                "protected_disconnection_demand_intervals": 0,
                "unprotected_disconnection_demand_intervals": 0,
                "invalid_action_attempts": 0, "mask_disagreements": 0,
                "solver_convergence_failures": 0, "wall_time_seconds": 1.0,
            },
        {
            "algorithm": "maskable_ppo", "operational_return": 14.0,
            "sla_violations_demand_intervals": 4,
            "reroutes_per_hour": 3.0, "max_utilization_mean": 1.0,
            "delivered_ratio_mean": 0.85, "te_reversals": 3,
                "protected_disconnection_demand_intervals": 0,
                "unprotected_disconnection_demand_intervals": 2,
                "invalid_action_attempts": 0, "mask_disagreements": 0,
                "solver_convergence_failures": 0, "wall_time_seconds": 1.5,
            },
    ])
    compact = aggregate_comparison(episodes)
    row = compact.iloc[0]
    assert row["operational_return_mean"] == 12.0
    assert row["sla_violations_demand_intervals_mean"] == 3.0
    assert row["reroutes_per_hour_mean"] == 2.0
    assert row["delivered_ratio_mean"] == pytest.approx(0.9)
