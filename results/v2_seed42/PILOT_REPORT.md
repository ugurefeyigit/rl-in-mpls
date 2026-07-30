# V2 seed-42 learning comparison

## Outcome

The governed seed-42 pilot completed successfully on `MplsTeEnvV2` at commit
`ca64b62fe29e45ab61aa86d642799aec5a4c25e1`. Both learners collected exactly
400,000 aggregate transitions with root seed 42, 16 vector environments, CUDA
neural execution, and checkpoints every 50,000 transitions. All final training
and evaluation integrity counters are zero.

The masked contextual bandit is the strongest method in this pilot. Its selected
250k checkpoint achieved a mean operational return of **22.65**, ahead of the
selected 250k MaskablePPO checkpoint (**13.45**), utilization-aware greedy
(**6.47**), CSPF (**-24.31**), and static (**-94.51**) over 7 scenarios × 5
continuity seeds.

This is a one-training-root result, not a final generalization claim. Holdout
seeds 1001–1005 were never constructed or evaluated.

## Direct answers

1. **Did MaskablePPO learn a stable improvement?** No. It learned a useful
   checkpoint and beat every baseline on aggregate mean return, but its
   checkpoint curve was non-monotonic: `-8.51, -2.87, 10.33, -16.79, 13.45,
   2.86, -8.75, -17.80` from 50k through 400k. The 250k peak did not persist,
   and cross-scenario return dispersion remained high (`std 150.44`).
2. **Did the contextual bandit learn a useful policy?** Yes. It moved from
   negative early checkpoints to `6.31` at 200k, peaked at `22.65` at 250k,
   and remained positive at 300k–400k (`21.24, 16.57, 18.52`). It ranked first
   overall and used less churn than PPO or greedy.
3. **Which method outperformed which baselines?** On aggregate mean operational
   return: bandit > PPO > greedy > CSPF > static. The bandit beat greedy in
   five of seven scenarios; greedy was better in `link_failure` and
   `ood_double_failure`. PPO beat the bandit only in `link_failure`.
4. **Does temporal planning appear necessary?** This seed provides no positive
   evidence that temporal planning was necessary: the explicitly myopic bandit
   outperformed PPO. That does not prove planning is irrelevant; two training
   roots and the untouched holdout remain.
5. **Were gains achieved without excessive churn or safety degradation?** Yes.
   Bandit/PPO reroutes were `1.86/2.14` per hour versus greedy `4.60`; moved
   bandwidth was `1,123/1,186` Mbps versus greedy `9,091`; TE reversals were
   `1.20/0.71` versus greedy `11.49`. Protected and unprotected disconnection
   counts were unchanged across all methods, and there were zero protected
   safety failures.
6. **What defects or limitations remain?** PPO learning was unstable; this is
   one root; scenario heterogeneity produces large aggregate variance; selected
   periodic PPO checkpoints include only the completed rollout blocks preceding
   the exact checkpoint transition; and no final holdout evidence exists.
   During the pilot, an SB3 seed-propagation bug invalidated the first PPO run
   and comparison. It was fixed, regression-tested, committed, and both final
   learners were rerun under one SHA. The failed artifacts were preserved.
7. **Runtime estimate for the remaining roots?** Final training took 1,608.4 s
   for PPO and 1,810.3 s for the bandit; comparison took 392.4 s. At the
   measured cadence, roots 314159 and 271828 require about **2.12 hours
   sequential** for both learners plus their comparisons, excluding setup and
   report time.

## Aggregate evaluation metrics

All values are means over 35 full scenario/seed episodes unless marked total.

| Metric | PPO | Bandit | Static | Greedy | CSPF |
|---|---:|---:|---:|---:|---:|
| Operational return | 13.453 | **22.654** | -94.510 | 6.467 | -24.308 |
| Return std | 150.438 | 153.580 | 80.449 | 133.610 | 132.422 |
| Offered Gbit total | 50,137.4 | 50,137.4 | 50,137.4 | 50,137.4 | 50,137.4 |
| Delivered Gbit total | 47,939.8 | **48,130.2** | 45,615.4 | 47,845.6 | 47,352.6 |
| Delivered ratio | 0.9495 | **0.9532** | 0.9048 | 0.9497 | 0.9374 |
| SLA violation demand-intervals | 207.5 | **162.0** | 341.5 | 185.1 | 242.2 |
| Protected disconnection intervals | 3.57 | 3.57 | 3.57 | 3.57 | 3.57 |
| Unprotected disconnection intervals | 10.71 | 10.71 | 10.71 | 10.71 | 10.71 |
| Peak maximum utilization | 1.577 | 1.596 | 2.099 | 1.820 | 1.810 |
| Mean maximum utilization | 1.041 | **1.037** | 1.384 | 1.059 | 1.182 |
| Mean link utilization | 0.1756 | 0.1665 | 0.1508 | 0.1677 | 0.1589 |
| Congested link intervals | 154.4 | **122.2** | 134.0 | 128.0 | 131.3 |
| Mean overload ratio | 0.00184 | **0.00171** | 0.00412 | 0.00189 | 0.00252 |
| Mean delay ms | 40.58 | **33.94** | 38.64 | 34.87 | 35.66 |
| Maximum delay ms | 226.16 | 186.58 | **168.10** | 195.56 | 173.57 |
| Mean loss ratio | 0.0501 | **0.0464** | 0.0946 | 0.0498 | 0.0621 |
| Accepted TE changes | 11.91 | 11.51 | 1.86 | 31.43 | 3.57 |
| Reroutes/hour | 2.14 | **1.86** | 0.37 | 4.60 | 0.65 |
| TE reversals | 0.71 | 1.20 | 0.00 | 11.49 | 0.11 |
| Flaps/demand | 0.042 | 0.071 | 0.000 | 0.676 | 0.007 |
| Moved bandwidth Mbps | 1,186 | 1,123 | 204 | 9,091 | 629 |
| Dwell-active demand intervals | 23.8 | 22.8 | 3.7 | 61.5 | 7.1 |
| Dwell remaining mean | 0.0315 | 0.0270 | 0.0055 | 0.0662 | 0.0095 |
| FRR changes | 1.71 | 1.66 | 1.86 | 1.77 | 1.74 |
| FRR disconnections | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| Recovery restorations | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| Solver iterations mean | 14.90 | **12.45** | 15.92 | 12.87 | 13.07 |
| Solver iterations maximum | 34.63 | 30.11 | 33.03 | 31.06 | **29.74** |
| Invalid actions total | 0 | 0 | 0 | 0 | 0 |
| Mask disagreements total | 0 | 0 | 0 | 0 | 0 |
| Solver failures total | 0 | 0 | 0 | 0 | 0 |

No rejected TE requests were observed. Full per-episode metrics, including
decision/mask timing and scenario-level rows, are in
`comparison_episode_summary.csv`.

## Reward decomposition

Mean episodic reward components use the repository’s exact component names.

| Component | PPO | Bandit | Static | Greedy | CSPF |
|---|---:|---:|---:|---:|---:|
| delivery | 181.928 | 182.507 | 174.598 | 181.583 | 180.151 |
| protected_disconnect | -25.510 | -25.510 | -25.510 | -25.510 | -25.510 |
| unprotected_disconnect | -8.403 | -8.403 | -8.403 | -8.403 | -8.403 |
| sla_severity | -22.798 | **-17.435** | -51.999 | -20.714 | -31.419 |
| max_util | -108.804 | **-105.461** | -181.108 | -109.599 | -137.444 |
| overload | -0.693 | **-0.636** | -1.727 | -0.757 | -0.969 |
| potential | -0.0046 | 0.0032 | -0.0139 | 0.0038 | -0.0021 |
| move_fixed | -0.953 | -0.921 | **-0.149** | -2.514 | -0.286 |
| move_volume | -0.200 | -0.188 | **-0.033** | -1.349 | -0.097 |
| move_divergence | -0.894 | -0.943 | **-0.165** | -2.828 | -0.293 |
| reversal | -0.214 | -0.360 | 0.000 | -3.446 | -0.034 |
| invalid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Every episode carried `reward_component_sum_exact=true`; the worst absolute
difference between the component sum and operational return was
`1.7053e-13`.

## Learning curves and checkpoint selection

| Transition | PPO return | Bandit return |
|---:|---:|---:|
| 50,000 | -8.508 | -16.908 |
| 100,000 | -2.868 | -21.726 |
| 150,000 | 10.334 | -13.836 |
| 200,000 | -16.794 | 6.313 |
| 250,000 | **13.453** | **22.654** |
| 300,000 | 2.856 | 21.237 |
| 350,000 | -8.754 | 16.566 |
| 400,000 | -17.800 | 18.524 |

All 16 checkpoints passed hash, metadata, freeze, mask, solver, reward, safety,
and horizon checks. The preregistered highest-valid-return rule selected 250k
for both methods.

## Action and churn behavior

Across 3,300 evaluated actions, PPO used no-op 2,883 times (87.36%) and the
bandit 2,897 times (87.79%). PPO’s most common non-noop actions were 55 (64
uses), 34 (45), and 7/14/18 (40 each). The bandit’s were 14 (49), 18 (40), 42
(38), and 68 (35). The complete per-episode action distributions are retained
in the machine-readable summary.

Both learned policies greatly reduced greedy-controller churn. Their gains were
not paid for with extra protected or unprotected disconnections, invalid
actions, mask disagreement, or safety failures.

## Hardware, device, and runtime

- CPU: Intel Core i7-14700HX, 20 physical cores / 28 logical processors
- RAM: 34,048,245,760 bytes
- GPU: NVIDIA RTX 4070 Laptop GPU, 8,188 MiB, driver 610.62
- Runtime: Python 3.13.4, PyTorch 2.11.0+cu128, CUDA runtime 12.8
- Libraries: Gymnasium 1.3.0, SB3/SB3-Contrib 2.9.0, NumPy 2.3.0,
  pandas 2.3.0, SciPy 1.18.0
- Vector benchmark winner: 16 environments on CUDA, combined 223.13
  transitions/s; mixed precision disabled
- Final PPO: 1,608.4 s, 248.69 transitions/s, 30,024,192 peak GPU bytes
- Final bandit: 1,810.3 s, 220.95 transitions/s, 24,719,360 peak GPU bytes
- Final comparison: 392.4 s

CUDA use was verified with parameters, loss, backward pass, and Adam optimizer
state on `cuda:0`; it was not inferred merely from GPU visibility.

## Integrity and seed audit

The two final episode-seed ledgers are byte-identical with SHA-256
`4928123efccc4324614943935f63b01b57ebde2ef07cc4a2d885ac2a73275668`.
Each contains 1,392 unique records, roots are exclusively 42, workers are
0–15, and derived seeds range from 42 to 88,121 with zero collisions.

The first PPO attempt was invalidated after its ledger exposed roots 42–57.
SB3 had forwarded `model_seed + worker_rank` to a V2 environment that already
derives child seeds from `root + worker_rank`, counting rank twice. The
experiment wrapper now preserves the governed root, a focused regression
reproduces the SB3 behavior, and training fails closed if any recorded root
differs. The failed PPO and comparison artifacts remain under `runs/v2/`; they
were not used in this report.

## Artifacts

- Compact comparison: `comparison.csv` and `comparison.json`
- Per-episode evaluation summary: `comparison_episode_summary.csv`
- Checkpoint curves: `learning_curve.csv`,
  `ppo_checkpoint_selection.csv`, `bandit_checkpoint_selection.csv`
- Full provenance: `manifest.json`
- Raw training/evaluation paths and checkpoint hashes: `manifest.json`
- Handoff: repository-root `NEXT_STAGE_HANDOFF.md`

No checkpoint, replay buffer, raw TensorBoard log, or large per-step dataset is
tracked by Git.
