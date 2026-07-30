# Next-stage handoff: V2 learning comparison

## Repository state

- Branch: `feat/rl-environment-v2`
- Final code commit: `ca64b62fe29e45ab61aa86d642799aec5a4c25e1`
- Approved tooling ancestor: `859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac`
- Signed-off environment pin: `dca533b5c6fa9953307d01470c23cac512eb2961`
- Environment: `MplsTeEnvV2`, observation 604, actions 69
- Remote: `origin/feat/rl-environment-v2` contains the final code commit
- V1 behavior/artifacts: unchanged
- Frozen V2 definition files: unchanged and verified

The concise result/report commit is the next commit after the code SHA above.
Use `git log -2 --oneline` to obtain its full SHA after pulling the branch.

## Changed files since `859fdb2`

- `.gitignore`
- `configs/experiments/learning_v2.yaml`
- `docs/superpowers/plans/2026-07-30-v2-learning-comparison.md`
- `docs/superpowers/specs/2026-07-30-v2-learning-comparison-design.md`
- `mplssim/experiments/evaluation_v2.py`
- `mplssim/experiments/learning_common.py`
- `mplssim/experiments/masked_bandit.py`
- `mplssim/experiments/trainers_v2.py`
- `scripts/benchmark_v2.py`
- `scripts/compare_v2.py`
- `scripts/evaluate_v2.py`
- `scripts/train_v2.py`
- `tests/test_learning_v2.py`
- `tests/test_state_machine.py`
- `tests/test_v1_v2_compatibility.py`
- `results/v2_seed42/PILOT_REPORT.md`
- `results/v2_seed42/manifest.json`
- `results/v2_seed42/benchmark_selection.json`
- `results/v2_seed42/comparison.csv`
- `results/v2_seed42/comparison.json`
- `results/v2_seed42/comparison_episode_summary.csv`
- `results/v2_seed42/comparison_tool_manifest.json`
- `results/v2_seed42/learning_curve.csv`
- `results/v2_seed42/ppo_checkpoint_selection.csv`
- `results/v2_seed42/bandit_checkpoint_selection.csv`
- `NEXT_STAGE_HANDOFF.md`

## Tests and gates

- Focused learning tests: `38 passed`
- Final full suite: `428 passed`
- Relevant V1/V2 compatibility, V2 environment, reward, and transition suite:
  `262 passed` before the final seed fix; the subsequent full suite includes all
  of them.
- `validate_env_v2.py --all`: `12/12` gates passed
- Final definition-freeze/pin recheck: passed
- Every final checkpoint sidecar/hash: passed
- Final checkpoint schedules: exactly 50k, 100k, ..., 400k for both learners
- Final training integrity: all counters zero
- Evaluation integrity: exact reward sums, normal truncation, no termination,
  invalid action, mask disagreement, solver failure, or safety failure

## Exact substantive commands executed

```powershell
git merge-base --is-ancestor 859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac HEAD
py -m pytest tests/test_learning_v2.py -q
py -m pytest tests/test_v1_v2_compatibility.py tests/test_env_v2.py tests/test_reward_v2.py tests/test_transition_v2.py -q
py -m pytest -q
py scripts/validate_env_v2.py --all

py -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

py scripts/benchmark_v2.py --output-dir runs/v2/benchmark_seed42_rerun --vector-counts 8 12 16 --vector-steps 512

py scripts/train_v2.py --purpose smoke --algorithm maskable_ppo --run-dir runs/v2/gpu_benchmark_corrected/maskable_ppo_cuda_8env --root-seed 42 --scenario random_day --transitions 4096 --n-envs 8 --checkpoint-interval 4096 --device cuda
py scripts/train_v2.py --purpose smoke --algorithm masked_bandit --run-dir runs/v2/gpu_benchmark/masked_bandit_cuda_8env --root-seed 42 --scenario random_day --transitions 4096 --n-envs 8 --checkpoint-interval 4096 --device cuda
py scripts/train_v2.py --purpose smoke --algorithm maskable_ppo --run-dir runs/v2/gpu_benchmark_corrected/maskable_ppo_cuda_16env --root-seed 42 --scenario random_day --transitions 8192 --n-envs 16 --checkpoint-interval 8192 --device cuda
py scripts/train_v2.py --purpose smoke --algorithm masked_bandit --run-dir runs/v2/gpu_benchmark_corrected/masked_bandit_cuda_16env --root-seed 42 --scenario random_day --transitions 8192 --n-envs 16 --checkpoint-interval 8192 --device cuda

py scripts/evaluate_v2.py --algorithm maskable_ppo --checkpoint runs/v2/gpu_benchmark_corrected/maskable_ppo_cuda_16env/checkpoints/checkpoint_000008192.zip --output-dir runs/v2/smoke_eval/maskable_ppo --scenarios evening_peak --seeds 101 --device cuda --no-steps
py scripts/evaluate_v2.py --algorithm masked_bandit --checkpoint runs/v2/gpu_benchmark_corrected/masked_bandit_cuda_16env/checkpoints/checkpoint_000008192.pt --output-dir runs/v2/smoke_eval/masked_bandit --scenarios evening_peak --seeds 101 --device cuda --no-steps

# Preserved but invalidated/superseded first attempt
py scripts/train_v2.py --purpose meaningful --algorithm maskable_ppo --run-dir runs/v2/seed42_maskable_ppo --root-seed 42 --scenario random_day --transitions 400000 --n-envs 16 --checkpoint-interval 50000 --device cuda
py scripts/train_v2.py --purpose meaningful --algorithm masked_bandit --run-dir runs/v2/seed42_masked_bandit --root-seed 42 --scenario random_day --transitions 400000 --n-envs 16 --checkpoint-interval 50000 --device cuda
py scripts/compare_v2.py --ppo-run runs/v2/seed42_maskable_ppo --bandit-run runs/v2/seed42_masked_bandit --output-dir runs/v2/seed42_comparison --device cuda

py scripts/train_v2.py --purpose smoke --algorithm maskable_ppo --run-dir runs/v2/seedfix_smoke_ppo --root-seed 42 --scenario random_day --transitions 4096 --n-envs 8 --checkpoint-interval 4096 --device cuda

py scripts/train_v2.py --purpose meaningful --algorithm maskable_ppo --run-dir runs/v2/seed42_maskable_ppo_final --root-seed 42 --scenario random_day --transitions 400000 --n-envs 16 --checkpoint-interval 50000 --device cuda

py scripts/train_v2.py --purpose meaningful --algorithm masked_bandit --run-dir runs/v2/seed42_masked_bandit_final --root-seed 42 --scenario random_day --transitions 400000 --n-envs 16 --checkpoint-interval 50000 --device cuda

py scripts/compare_v2.py --ppo-run runs/v2/seed42_maskable_ppo_final --bandit-run runs/v2/seed42_masked_bandit_final --output-dir runs/v2/seed42_comparison_final --device cuda
```

There were additional targeted test invocations, checkpoint validation
one-liners, process-monitoring reads, Git inspection commands, and smoke
evaluations. The final training/evaluation commands above are also embedded
verbatim in the machine-readable run and result manifests.

## Hardware and selected execution

- Windows 11 Home 10.0.26200, 64 bit
- Intel Core i7-14700HX, 20 physical / 28 logical processors
- RAM: 34,048,245,760 bytes
- NVIDIA RTX 4070 Laptop GPU, 8,188 MiB, driver 610.62
- Python 3.13.4
- PyTorch 2.11.0+cu128, CUDA runtime 12.8
- Gymnasium 1.3.0
- Stable-Baselines3 / SB3-Contrib 2.9.0
- NumPy 2.3.0, pandas 2.3.0, SciPy 1.18.0
- Selected vector environments: 16
- Selected neural device: CUDA
- Mixed precision: disabled

Bounded combined throughput:

- 8 env CPU: 104.50 transitions/s
- 16 env CPU: 96.51 transitions/s
- 8 env CUDA: 200.43 transitions/s
- 16 env CUDA: 223.13 transitions/s (selected)
- 12 env was measured but excluded because it cannot divide both 400,000 and
  50,000 exactly.

## Hyperparameters

MaskablePPO retained the governed configuration:

- Network `[256, 256]`
- Learning rate `3e-4`
- Rollout steps `512`
- Batch `512`
- Epochs `8`
- Gamma `0.995`
- GAE lambda `0.95`
- Clip `0.2`
- Entropy coefficient `0.01`
- Value coefficient `0.5`
- Gradient clip `0.5`

Masked contextual bandit:

- Network `604 -> 256 -> 256 -> 69`
- Adam `3e-4`
- Batch `512`
- Replay capacity `100,000`
- Warm-up `4,096`
- Update every 4 vector steps
- Masked epsilon-greedy `0.20 -> 0.02` over 200,000 transitions
- Selected-action Huber loss against immediate observed reward
- No discount, next observation, Bellman target, target network, or bootstrap
- No reward scaling
- Gradient clip `1.0`

## Final run status and artifacts

All paths are under:

`C:/Users/ugure/OneDrive/Masaüstü/rl_in_mpls/.worktrees/seed42`

- PPO: `runs/v2/seed42_maskable_ppo_final`
  - completed, 400,000 transitions
  - 1,608.43 s, 248.69 transitions/s
  - peak GPU allocation 30,024,192 bytes
- Bandit: `runs/v2/seed42_masked_bandit_final`
  - completed, 400,000 transitions
  - 1,810.32 s, 220.95 transitions/s
  - 6,187 updates, replay 100,000, final epsilon 0.02
  - peak GPU allocation 24,719,360 bytes
- Comparison: `runs/v2/seed42_comparison_final`
  - completed, 392.38 s
  - all 16 checkpoint matrices plus selected learners and three baselines
- Compact committed artifacts: `results/v2_seed42`
- Full per-step training metrics:
  - `runs/v2/seed42_maskable_ppo_final/training_steps.jsonl.gz`
  - `runs/v2/seed42_masked_bandit_final/training_steps.jsonl.gz`
- Per-episode training metrics and complete derived seed ledgers are in each
  run directory.

No model, checkpoint, replay buffer, raw TensorBoard log, or large step dataset
is committed to Git.

## Selected checkpoints and hashes

- PPO 250,000:
  - `runs/v2/seed42_maskable_ppo_final/checkpoints/checkpoint_000250000.zip`
  - SHA-256 `d34cc77ded05b064fa2a39dbe5c5ccc3126c9e6cf85e36c1b507127c987f5676`
- Bandit 250,000:
  - `runs/v2/seed42_masked_bandit_final/checkpoints/checkpoint_000250000.pt`
  - SHA-256 `c15097700eac518ee259cba67e34e4fba1716881ab3dd912188b55da0c79bf49`

All sixteen periodic hashes are in `results/v2_seed42/manifest.json`.

## Seed-42 comparison

Mean operational return over 35 evaluation episodes:

| Rank | Method | Mean | Std |
|---:|---|---:|---:|
| 1 | masked_bandit | 22.654 | 153.580 |
| 2 | maskable_ppo | 13.453 | 150.438 |
| 3 | greedy | 6.467 | 133.610 |
| 4 | cspf | -24.308 | 132.422 |
| 5 | static | -94.510 | 80.449 |

Both learners selected the 250k checkpoint. The bandit showed a useful late
plateau; PPO was non-monotonic and ended weaker than its selected checkpoint.
The bandit had fewer SLA violation intervals, lower mean max utilization, lower
loss/delay, fewer reroutes, and less moved bandwidth than PPO. Both learners
avoided the severe churn of greedy without changing disconnection safety.

The result does not demonstrate a need for temporal planning because the
myopic bandit won this root. Do not generalize that conclusion beyond seed 42.

## Seed and holdout confirmation

- Final training root: 42 only
- Final PPO/bandit seed ledgers are byte-identical
- Ledger SHA-256:
  `4928123efccc4324614943935f63b01b57ebde2ef07cc4a2d885ac2a73275668`
- 1,392 records and 1,392 unique derived seeds per learner
- Workers 0–15; derived seed range 42–88,121
- Evaluation seeds: 101–105 only
- **Holdout seeds 1001–1005 were not constructed, inspected, tuned on, selected
  on, or evaluated.**

## Defects and limitations

1. The initial PPO run was invalidated because SB3 replaced the governed root
   with `42 + worker_rank`; its comparison was invalidated too. The wrapper now
   isolates model seeding from V2 root derivation, tests reproduce the failure,
   and the final run uses root 42 for every worker.
2. The original bandit run was valid, but it was repeated unchanged to bind
   both final learners to `ca64b62…`. Payload hashes reproduced exactly.
3. PPO did not show a stable learning curve.
4. Only root 42 is complete; roots 314159 and 271828 remain.
5. Final holdout remains untouched.
6. Periodic PPO checkpoint transition counts are collected aggregate
   transitions; model updates consume only completed 8,192-transition rollout
   blocks preceding the checkpoint.
7. Pip reports no broken dependencies, but the CUDA installation left stale
   `~orch*` uninstall-backup directories that the environment deletion policy
   would not permit removing. They do not affect imports or execution.

Measured estimate for roots 314159 and 271828: about 2.12 hours sequential for
both learners and both comparisons on this machine, excluding setup/reporting.

## Recommended next prompt inputs

Use the following constraints in the next Codex task:

1. Start from branch `feat/rl-environment-v2` and the latest report commit above
   `ca64b62fe29e45ab61aa86d642799aec5a4c25e1`.
2. Read `NEXT_STAGE_HANDOFF.md`, `results/v2_seed42/PILOT_REPORT.md`, and
   `results/v2_seed42/manifest.json`.
3. Verify the branch descends from `ca64b62…` and the V2 freeze gate still
   passes; do not reset newer work.
4. Run the preregistered unchanged comparison for roots `314159` and `271828`
   only, using the same 16-env CUDA configuration, algorithms, hyperparameters,
   checkpoint rule, continuity scenarios/seeds, metrics, and integrity checks.
5. Preserve root seeds through the wrapper and audit that both episode-seed
   ledgers are byte-identical per root before accepting results.
6. Do not tune on seed-42 results, change V2/V1, add algorithms/UI, or access
   holdout seeds 1001–1005.
7. Stop after the three-root continuity comparison and prepare the decision
   gate for the still-untouched final holdout.
