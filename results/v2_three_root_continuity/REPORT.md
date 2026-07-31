# V2 three-root continuity analysis

## Decision

The masked contextual bandit advantage persists across all three preregistered
training roots. It beat MaskablePPO on every root and achieved a three-root
mean selected-policy return of **25.74**
versus **16.70** for PPO, an average
advantage of **9.04**. Both learners beat all three
repository baselines on every root; greedy remained the strongest baseline at
**6.47**.

This continuity study does **not** provide positive evidence that temporal
planning is necessary. The explicitly myopic bandit won 3/3 roots. That result
does not prove temporal planning is irrelevant, but it makes a planning-based
explanation for the observed gains less likely than a strong masked
state-to-action mapping.

The gains preserve the governed safety envelope. Across all final evaluations
there were zero invalid actions, mask disagreements, solver convergence
failures, protected safety failures, or reward-sum mismatches. Protected and
unprotected disconnection accounting is unchanged across methods. Learned
policy churn remained far below greedy overall: bandit reroutes averaged
2.12/h and PPO
2.17/h versus greedy
4.60/h.

**Recommendation:** authorize the untouched final holdout as the next and only
decision stage, using the already frozen procedure with no tuning. Do not
authorize algorithm redesign before that gate.

## Per-root result and selected checkpoint

| Training root | Bandit return | PPO return | Bandit advantage | Bandit selected | PPO selected |
| --- | --- | --- | --- | --- | --- |
| 42 | 22.654 | 13.453 | 9.201 | 250000 | 250000 |
| 314159 | 29.800 | 16.159 | 13.641 | 300000 | 350000 |
| 271828 | 24.769 | 20.484 | 4.285 | 400000 | 150000 |

Bandit won all three roots. Its advantage varied from
**4.29** to **13.64**, so the direction is
stable while the magnitude is not constant. The selected bandit checkpoints
moved later across the continuation roots (250k, 300k, 400k); PPO remained
non-monotonic and selected 250k, 350k, and 150k. PPO's root-271828 curve peaked
at 150k and then stayed negative through 400k, while the bandit finished at its
best checkpoint on that root.

## Aggregate operational and traffic-engineering metrics

| Method | Return mean | Root-mean SD | Delivered ratio | SLA intervals | Reroutes/hour | Moved Mbps |
| --- | --- | --- | --- | --- | --- | --- |
| masked_bandit | 25.741 | 3.671 | 0.954 | 159.905 | 2.118 | 1599.604 |
| maskable_ppo | 16.699 | 3.547 | 0.951 | 197.571 | 2.166 | 1274.152 |
| greedy | 6.467 | 0.000 | 0.950 | 185.086 | 4.602 | 9091.398 |
| cspf | -24.308 | 0.000 | 0.937 | 242.200 | 0.645 | 629.121 |
| static | -94.510 | 0.000 | 0.905 | 341.457 | 0.371 | 203.611 |

The bandit delivered the best aggregate return, delivery ratio, and SLA count.
It moved more bandwidth than PPO (1,600 versus 1,274 Mbps) but far less than
greedy (9,091 Mbps). PPO had slightly fewer reroutes than the bandit on root
314159; the bandit had fewer on roots 42 and 271828. Root 271828 raised churn
for both learners, but both remained below greedy.

| Method | Mean max util | Congested intervals | Delay ms | Loss ratio | Reversals | Flaps/demand | Dwell-active | FRR changes | FRR disconnects | Restorations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked_bandit | 1.030 | 125.68 | 34.77 | 0.0459 | 1.48 | 0.0868 | 23.91 | 1.80 | 0.57 | 0.57 |
| maskable_ppo | 1.032 | 138.25 | 38.35 | 0.0488 | 2.36 | 0.1389 | 24.50 | 1.74 | 0.57 | 0.57 |
| greedy | 1.059 | 127.97 | 34.87 | 0.0498 | 11.49 | 0.6756 | 61.49 | 1.77 | 0.57 | 0.57 |
| cspf | 1.182 | 131.26 | 35.66 | 0.0621 | 0.11 | 0.0067 | 7.14 | 1.74 | 0.57 | 0.57 |
| static | 1.384 | 134.03 | 38.64 | 0.0946 | 0.00 | 0.0000 | 3.71 | 1.86 | 0.57 | 0.57 |

The bandit has the best mean utilization, loss, and congestion results and
nearly the best delay. Its moved-bandwidth cost is offset by fewer reversals
and flaps than PPO. FRR disconnections and recovery restorations are identical
across methods, confirming that the gains did not change failure accounting.
Complete overload, TE, dwell, and scenario-level fields are in
`comparison_metrics_by_root.csv`, `aggregate_metrics.csv`, and
`scenario_metrics.csv`.

## Stability, variance, and learning behavior

Selected-policy episode return dispersion remains high because the seven
scenarios differ sharply: aggregate episode standard deviations are
154.11 for bandit and
154.50 for PPO. Across training-root
means, standard deviations are 3.67
and 3.55, respectively. The bandit is
directionally stable against PPO, but neither learner eliminates
scenario-driven variance. `learning_curves.csv` preserves every checkpoint
return, validity decision, selected flag, and payload hash.

## Reward, action, and state-change accounting

Every evaluated episode's operational return equals the exact sum of the 12
named reward components. `reward_components.csv` reports component means and
the maximum floating-point residual by root and method.

`action_distribution.csv` contains all action counts and frequencies. Aggregate
no-op frequency is 82.35% for bandit,
81.95% for PPO, and
61.65% for greedy. The detailed tables retain
accepted/rejected TE changes, reversals, flaps, moved bandwidth, dwell,
FRR changes/disconnections, and recovery restorations.

## Runtime and diagnostics

All accepted learner runs used 16 vector environments and CUDA on the NVIDIA
GeForce RTX 4070 Laptop GPU. Mixed precision remained disabled. Training
runtime, throughput, and peak allocated GPU memory:

| Root | Algorithm | Wall seconds | Transitions/s | Peak GPU bytes |
| --- | --- | --- | --- | --- |
| 42 | maskable_ppo | 1608.434 | 248.689 | 30024192 |
| 42 | masked_bandit | 1810.324 | 220.955 | 24719360 |
| 314159 | maskable_ppo | 1435.116 | 278.723 | 30024192 |
| 314159 | masked_bandit | 1196.129 | 334.412 | 24719360 |
| 271828 | maskable_ppo | 1472.238 | 271.695 | 30024192 |
| 271828 | masked_bandit | 1270.822 | 314.757 | 24719360 |

PPO ended each run with 384 updates. The bandit ended each run with 6,187
updates, replay size 100,000, and epsilon 0.02. Full final diagnostics are
machine-readable in `training_summary.csv`. Comparison wall times were 392.4 s
for root 42, 395.5 s for root 314159, and 360.6 s for root 271828.

## Integrity and provenance

- Environment: `MplsTeEnvV2`, observation 604, actions 69.
- Signed-off definition pin: `dca533b5c6fa9953307d01470c23cac512eb2961`; the freeze gate passed.
- Seed-42 scientific source: `ca64b62fe29e45ab61aa86d642799aec5a4c25e1`.
- Continuation source: `6a8a4068b98bf9a71dead6e547595b4bbd755689`. Its only non-report differences
  from the seed-42 scientific source are the tested governance repairs that
  activate preregistered roots and admit them during checkpoint selection.
- Focused learning/compatibility tests: 71 passed.
- Freeze/pin subset: 14 passed, 110 deselected.
- Each accepted learner has exactly 400,000 transitions, eight checkpoints,
  1,392 ledger records, 1,392 unique episode seeds, and zero collisions.
- PPO and bandit ledgers are byte-identical within every root.
- All 48 checkpoint payloads and sidecars were validated; all checkpoint-sweep
  rows were valid.
- All 525 final evaluation episodes use scenarios from the approved seven and
  continuity seeds 101–105 only.
- **Final holdout seeds 1001–1005 were not constructed, evaluated, inspected,
  debugged, selected, or tuned with.**

Exact commands, checkpoint hashes, ledger hashes, run paths, failures, source
SHAs, and compact artifact inventory are in `manifest.json`.

## Preserved failures and limitations

The first root-314159 launch failed before creating a run directory because the
pilot-only root guard still admitted only seed 42. After the first governance
repair, PPO and bandit training completed at `288a980`, but comparison failed
because checkpoint selection still hard-coded seed 42. Those artifacts and the
failed comparison were preserved. The final root-314159 learners were rerun
unchanged at `6a8a4068b98bf9a71dead6e547595b4bbd755689` so training and evaluation share one exact
source identity. The superseded bandit checkpoint payloads reproduced
byte-for-byte; PPO ZIP payload hashes differed because the archive format is
not byte-reproducible, while its seed ledger reproduced exactly.

Limitations remain: only three training roots are measured; continuity
evaluation reuses the same 35 scenario/seed episodes per root; baseline results
are therefore identical across roots; aggregate episode variance is dominated
by scenario heterogeneity; and the final holdout remains completely untouched.

## Full artifact locations

- Seed 42: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\seed42`
- Continuation experiment worktree: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2`
- Root 314159 PPO: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2\runs\v2\seed314159_maskable_ppo_final_r2`
- Root 314159 bandit: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2\runs\v2\seed314159_masked_bandit_final_r2`
- Root 314159 comparison: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2\runs\v2\seed314159_comparison_final_r2`
- Root 271828 PPO: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2\runs\v2\seed271828_maskable_ppo_final`
- Root 271828 bandit: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2\runs\v2\seed271828_masked_bandit_final`
- Root 271828 comparison: `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\continuity_v2\runs\v2\seed271828_comparison_final`

No checkpoint, model binary, replay buffer, TensorBoard data, raw step log, or
large dataset is included in this compact result directory.
