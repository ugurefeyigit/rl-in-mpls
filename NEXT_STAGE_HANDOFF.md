# Next-stage handoff: V2 final-holdout decision gate

## Decision

The three-root continuity study is complete. The masked contextual bandit beat
MaskablePPO on all three preregistered training roots. Mean selected-policy
return across roots was 25.74 for the bandit and 16.70 for PPO; greedy, the
strongest baseline, remained at 6.47.

Recommendation: authorize the untouched final holdout as the next and only
scientific stage, with the procedure still frozen and no tuning. Do not redesign
the learners before that gate.

## Repository and scientific identity

- Branch: `feat/rl-environment-v2`
- Report parent and continuation training source:
  `6a8a4068b98bf9a71dead6e547595b4bbd755689`
- Seed-42 scientific training source:
  `ca64b62fe29e45ab61aa86d642799aec5a4c25e1`
- Approved tooling ancestor:
  `859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac`
- Signed-off environment pin:
  `dca533b5c6fa9953307d01470c23cac512eb2961`
- Environment: `MplsTeEnvV2`, observation 604, actions 69
- The final pushed report commit is the commit containing this handoff; obtain
  its full SHA with `git log -1 --format=%H` after pulling.

The continuation source differs from the seed-42 scientific source only through
committed reports and two tested governance repairs: activation of the two
remaining preregistered roots and admission of those roots during checkpoint
selection. Environment, learners, hyperparameters, masks, rewards, scenarios,
evaluation, and checkpoint selection semantics did not change.

## Three-root result

| Root | Bandit | PPO | Advantage | Bandit checkpoint | PPO checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 22.654 | 13.453 | 9.201 | 250k | 250k |
| 314159 | 29.800 | 16.159 | 13.641 | 300k | 350k |
| 271828 | 24.769 | 20.484 | 4.285 | 400k | 150k |

The bandit advantage persists 3/3. This provides no positive evidence that
temporal planning is necessary: the myopic learner won every root. Safety was
preserved, and learned-policy churn remained substantially below greedy.

## Integrity and tests

- Focused learning/compatibility suite: 71 passed
- Definition-freeze/pin subset: 14 passed, 110 deselected
- CUDA was exercised through PyTorch parameters, gradients, backward, and Adam
  state on the NVIDIA GeForce RTX 4070 Laptop GPU.
- Every accepted learner completed exactly 400,000 aggregate transitions with
  16 environments and eight 50,000-transition checkpoints.
- All 48 checkpoint payloads and sidecars passed validation.
- Every learner ledger contains 1,392 records and 1,392 unique seeds.
- PPO and bandit ledgers are byte-identical within each root.
- All training integrity counters are zero.
- All 525 final episodes have exact reward sums, normal truncation, no
  termination, and zero invalid actions, mask disagreements, solver failures,
  or protected safety failures.
- Evaluation scenarios are the approved seven and evaluation seeds are
  101–105 only.
- **Final holdout seeds 1001–1005 were not constructed, evaluated, inspected,
  debugged, selected, or tuned with.**

## Compact evidence

Read:

- `results/v2_three_root_continuity/REPORT.md`
- `results/v2_three_root_continuity/manifest.json`
- `results/v2_three_root_continuity/comparison_metrics_by_root.csv`
- `results/v2_three_root_continuity/aggregate_metrics.csv`
- `results/v2_three_root_continuity/scenario_metrics.csv`
- `results/v2_three_root_continuity/reward_components.csv`
- `results/v2_three_root_continuity/action_distribution.csv`
- `results/v2_three_root_continuity/checkpoint_selection.csv`
- `results/v2_three_root_continuity/learning_curves.csv`
- `results/v2_three_root_continuity/training_summary.csv`
- `results/v2_three_root_continuity/training_integrity.csv`
- `results/v2_three_root_continuity/evaluation_integrity.csv`

The continuity manifest contains exact commands, run paths, checkpoint hashes,
ledger hashes, failures, superseded runs, runtimes, source SHAs, and the
explicit holdout confirmation.

## Full artifacts

Seed 42 remains under:

`C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\seed42`

Accepted continuation artifacts remain under:

`C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2`

- `runs/v2/seed314159_maskable_ppo_final_r2`
- `runs/v2/seed314159_masked_bandit_final_r2`
- `runs/v2/seed314159_comparison_final_r2`
- `runs/v2/seed271828_maskable_ppo_final`
- `runs/v2/seed271828_masked_bandit_final`
- `runs/v2/seed271828_comparison_final`

Preserved failed/superseded evidence:

- Seed42 worktree:
  `runs/v2/seed314159_maskable_ppo_final.stdout.log`
- Continuity worktree:
  `runs/v2/seed314159_maskable_ppo_final`
- Continuity worktree:
  `runs/v2/seed314159_masked_bandit_final`
- Continuity worktree:
  `runs/v2/seed314159_comparison_final`

No checkpoint, model binary, replay buffer, TensorBoard data, raw step log, or
large dataset is committed.

## Final-holdout gate

If authorized, the next task must use the already frozen final-holdout
procedure without altering definitions, learners, hyperparameters, checkpoint
selection, or metrics. It must not use the holdout for tuning, debugging,
checkpoint selection, or repeated redesign cycles. Until explicit authorization
is given, the final holdout remains closed.
