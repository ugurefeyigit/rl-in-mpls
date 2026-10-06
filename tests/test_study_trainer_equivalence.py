"""The study trainer reproduces the governed V2 trainer (slow: ~2 CPU-minutes).

Run with ``pytest -m slow``. Equality of network parameters after an
identical budget shows that runs from both trainers are the same experiment.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("algorithm", ["masked_bandit", "maskable_ppo"])
def test_study_trainer_matches_governed_trainer(tmp_path, algorithm):
    from mplssim.experiments.learning_common import load_learning_config
    from mplssim.experiments.trainers_v2 import train_experiment
    from mplssim.experiments.v2_factory import make_env_v2
    from mplssim.study.evalrun import load_any_policy
    from mplssim.study.registry import base_config
    from mplssim.study.train import train_run

    torch.set_num_threads(1)
    n = 8192
    train_experiment(algorithm=algorithm, run_directory=tmp_path / "gov", root_seed=42,
                     scenario="random_day", aggregate_transitions=n, n_envs=16,
                     checkpoint_interval=n, requested_device="cpu",
                     learning_config=load_learning_config(), purpose="smoke",
                     require_clean_checkout=False)
    train_run(run_dir=tmp_path / "mine", algorithm=algorithm, root=42,
              config=base_config(algorithm), env_factory=make_env_v2,
              env_spec={"variant": "base"}, transitions=n, checkpoint_interval=n)
    suffix = "zip" if algorithm == "maskable_ppo" else "pt"
    a = load_any_policy(tmp_path / "gov" / "checkpoints" / f"checkpoint_{n:09d}.{suffix}", algorithm)
    b = load_any_policy(tmp_path / "mine" / "checkpoints" / f"checkpoint_{n:09d}.{suffix}", algorithm)
    pa = (a.model.policy if algorithm == "maskable_ppo" else a.network).state_dict()
    pb = (b.model.policy if algorithm == "maskable_ppo" else b.network).state_dict()
    assert pa.keys() == pb.keys()
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k
