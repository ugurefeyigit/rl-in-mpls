# V2 study closeout: final holdout complete

## Final decision

The governed V2 study is complete and closed. The masked contextual bandit
advantage generalized to the single untouched final holdout. Across the three
continuity-selected roots, mean holdout return was **18.221** for the bandit and
**9.036** for MaskablePPO, an advantage of **9.185**. Greedy was the strongest
repository baseline at **-2.327**.

The bandit beat PPO on all three training roots and in six of seven scenarios.
PPO retained a small 1.107-point advantage in `deceptive_local_optimum`; this is
preserved as a negative result against an across-the-board bandit claim.

The final evidence provides no positive support for a need for temporal
planning in this frozen task: the explicitly myopic learner remains stronger.
This does not establish that planning is generally irrelevant. No further
tuning, checkpoint selection, or redesign recommendation is made from holdout
performance.

## Repository and scientific identity

- Branch: `feat/rl-environment-v2`
- Final-holdout evaluation source:
  `f7ed0f407c50c5472ecff89f977bc656439a8c49`
- Seed-42 scientific training source:
  `ca64b62fe29e45ab61aa86d642799aec5a4c25e1`
- Continuation training/evaluation source:
  `6a8a4068b98bf9a71dead6e547595b4bbd755689`
- Approved ancestor:
  `859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac`
- Signed-off environment pin:
  `dca533b5c6fa9953307d01470c23cac512eb2961`
- Environment: `MplsTeEnvV2`, observation 604, actions 69

The pre-existing unstaged generated change to
`results/environment_v2_validation/manifest.json` was preserved and excluded
from all commits.

## Final holdout result

| Training root | Bandit | PPO | Bandit advantage | Bandit checkpoint | PPO checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 16.128 | 6.568 | 9.560 | 250k | 250k |
| 314159 | 20.546 | 8.297 | 12.250 | 300k | 350k |
| 271828 | 17.988 | 12.242 | 5.746 | 400k | 150k |

| Method | Return | Delivery | SLA intervals | Reroutes/hour | Moved Mbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| masked_bandit | 18.221 | 0.9492 | 174.94 | 2.148 | 1,602.86 |
| maskable_ppo | 9.036 | 0.9459 | 208.17 | 2.148 | 1,291.00 |
| greedy | -2.327 | 0.9444 | 200.34 | 4.913 | 9,963.93 |
| cspf | -28.339 | 0.9347 | 249.57 | 0.636 | 683.18 |
| static | -101.851 | 0.8998 | 353.43 | 0.366 | 205.87 |

Bandit utilization, congestion, delay, loss, reversal, and flap metrics were
better than PPO overall. Bandit moved more bandwidth than PPO but far less than
greedy. Bandit and PPO both averaged about 2.148 reroutes/hour; bandit averaged
1.58 reversals and 0.0930 flaps/demand versus PPO's 2.00 and 0.1176. This is
acceptable churn within the frozen comparison.

Protected and unprotected disconnection accounting was identical across every
method. FRR disconnections and restorations were also identical. All methods
had zero rejected TE requests. The gains therefore preserve the governed
safety envelope.

## One-shot authorization and integrity

- Final seeds: 1001, 1002, 1003, 1004, 1005 only.
- Scenarios: the seven frozen scenarios only.
- Six fixed continuity-selected learner checkpoints; no sweep or reselection.
- Static, greedy, and CSPF ran through the repository implementations once.
- Exactly 35 episodes per checkpoint or baseline; 315 total.
- Exactly 315 compressed full-step artifacts and 315 episode summaries.
- Deterministic inference with authoritative masks.
- All episodes reached normal truncation; no abnormal termination.
- Zero invalid actions, mask disagreements, reward mismatches, non-finite
  values, solver convergence failures, or protected safety failures.
- Every step passed the exact 12-component reward sum check. The largest
  residual after separately aggregating episode components was 1.7053e-13.
- All six payload hashes, sidecar hashes, source SHAs, roots, algorithms, and
  transitions were independently revalidated after completion.
- The final evaluation ran exactly once. No episode was retried, omitted, or
  selectively repeated.

No training, tuning, checkpoint selection, checkpoint reselection, policy
debugging, or algorithm redesign used holdout results.

## Authorization-tooling repair

The original gate rejected holdout seeds unconditionally and required the
evaluation checkout SHA to equal each checkpoint's training SHA. That made a
single safe evaluation across the two approved source identities impossible.
Before holdout access, commit `f7ed0f4` added an evaluation-only workflow with:

- an immutable six-checkpoint registry;
- exact payload, sidecar, root, algorithm, transition, and source binding;
- descendant-only cross-source loading;
- an allowlist restricted to evaluation, governance, tests, and compact
  results, with scientific-definition changes rejected;
- an explicit complete final-holdout seed mode;
- no checkpoint, seed, or scenario selection inputs;
- fail-closed new output directories and no retry path.

The repair was committed and pushed before the final evaluation. Verification
before holdout access was 81 focused learning/compatibility tests passed,
14 freeze/pin tests passed (110 deselected), and 440 full-suite tests passed.

## Runtime and artifacts

The one-shot evaluation used CUDA on the NVIDIA GeForce RTX 4070 Laptop GPU.
Total runner wall time was 152.093 seconds. Learner peak allocated GPU memory
ranged from 13,386,752 to 16,926,720 bytes.

Compact evidence:

- `results/v2_final_holdout/FINAL_HOLDOUT_REPORT.md`
- `results/v2_final_holdout/manifest.json`
- `results/v2_final_holdout/per_root_metrics.csv`
- `results/v2_final_holdout/aggregate_metrics.csv`
- `results/v2_final_holdout/scenario_metrics.csv`
- `results/v2_final_holdout/reward_components.csv`
- `results/v2_final_holdout/action_distribution.csv`
- `results/v2_final_holdout/evaluation_integrity.csv`
- `results/v2_final_holdout/checkpoint_provenance.csv`

Full artifacts:

`C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\final_holdout_v2\runs\v2\final_holdout_20260731_f7ed0f4`

This directory contains compressed step evidence and episode summaries. It is
outside Git. No checkpoint, model binary, replay buffer, TensorBoard data, raw
training log, compressed episode log, or large dataset is committed.

## Closeout

There is no next governed V2 experiment stage in this study. The final report
and this handoff are the stop condition. Preserve all worktrees, failed and
superseded earlier runs, selected checkpoints, final-holdout artifacts, and the
unstaged validation-manifest change.
