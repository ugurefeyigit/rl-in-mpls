# V2 final holdout report

## Decision

The masked contextual bandit advantage generalizes to the untouched holdout: it beat PPO on 3/3 training roots and by 9.185 return points in the root-aggregated learner comparison. This final evidence does not support a need for temporal planning; the explicitly myopic learner remains stronger. That is a result about these frozen learners and scenarios, not a claim that planning is generally useless.

The strongest repository baseline was greedy at **-2.327**. The
bandit achieved **18.221** and PPO **9.036**. No
checkpoint was selected, reselected, tuned, or redesigned with holdout results.
The study is closed; no further tuning recommendation is made from this holdout.

The advantage is broad but not universal across scenarios. After averaging the
three training roots, the bandit beat PPO in six of seven scenarios. PPO retained
a small **1.107**-point edge in `deceptive_local_optimum`; the largest bandit edge
was **20.183** in `link_failure`.

## Per-root learner performance

| Training root | Bandit return | PPO return | Bandit advantage |
| ---: | ---: | ---: | ---: |
| 42 | 16.128 | 6.568 | 9.560 |
| 314159 | 20.546 | 8.297 | 12.250 |
| 271828 | 17.988 | 12.242 | 5.746 |

## Aggregate operational and traffic-engineering metrics

| Method | Return mean | Episode SD | Delivered ratio | SLA intervals | Reroutes/hour | Moved Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| masked_bandit | 18.221 | 156.021 | 0.9492 | 174.94 | 2.148 | 1602.86 |
| maskable_ppo | 9.036 | 154.544 | 0.9459 | 208.17 | 2.148 | 1291.00 |
| greedy | -2.327 | 135.649 | 0.9444 | 200.34 | 4.913 | 9963.93 |
| cspf | -28.339 | 136.276 | 0.9347 | 249.57 | 0.636 | 683.18 |
| static | -101.851 | 82.118 | 0.8998 | 353.43 | 0.366 | 205.87 |

`aggregate_metrics.csv` and `scenario_metrics.csv` retain utilization,
congestion, overload, delay, loss, delivery, disconnections, reroutes,
reversals, flaps, moved bandwidth, dwell, accepted and rejected TE changes,
FRR changes/disconnections, restorations, decision time, and mask time.

| Method | Mean max util | Congested intervals | Delay ms | Loss ratio | Reversals | Flaps/demand | No-op share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| masked_bandit | 1.051 | 135.40 | 36.41 | 0.0504 | 1.58 | 0.0930 | 87.09% |
| maskable_ppo | 1.056 | 145.16 | 39.80 | 0.0537 | 2.00 | 0.1176 | 87.31% |
| greedy | 1.086 | 140.31 | 36.56 | 0.0551 | 12.89 | 0.7580 | 64.64% |
| cspf | 1.203 | 139.63 | 36.67 | 0.0649 | 0.11 | 0.0067 | 95.97% |
| static | 1.420 | 138.89 | 39.66 | 0.0996 | 0.00 | 0.0000 | 98.06% |

Protected and unprotected disconnection demand-interval means were identical
for every method (3.571 and 10.714). FRR disconnections and restorations were
also identical (0.571 each). The bandit averaged 12.17 accepted TE changes and
zero rejected requests per episode, versus 11.96 and zero for PPO. It had
slightly more dwell-active intervals (24.11 versus 23.79) but essentially the
same mean dwell remaining (0.0313).

## Stability and scenario variance

Aggregate episode return SD is
**156.02** for the
bandit and **154.54**
for PPO. Across root means, SD is
**2.22** and
**2.91**, respectively.
Scenario-level values are reported without selective omission in
`scenario_metrics.csv`; scenario heterogeneity remains the main source of
episode variance.

| Scenario | Bandit | PPO | Bandit advantage |
| --- | ---: | ---: | ---: |
| full_day | 335.551 | 321.714 | 13.838 |
| evening_peak | 31.683 | 20.159 | 11.524 |
| flash_crowd | -42.045 | -45.769 | 3.724 |
| link_failure | -0.547 | -20.730 | 20.183 |
| deceptive_local_optimum | 91.442 | 92.549 | -1.107 |
| ood_double_failure | -191.259 | -192.917 | 1.658 |
| overload_stress | -97.280 | -111.755 | 14.475 |

## Reward, actions, safety, and churn

All named reward components and their maximum aggregation residuals are in
`reward_components.csv`. Every step passed the repository's exact component
sum check; the largest residual after separately aggregating episode components
was **1.7053e-13**. The bandit's advantage is mainly associated with higher
delivery reward, lower utilization and SLA penalties, and a smaller reversal
penalty. Its movement penalties were slightly larger, consistent with its higher
moved bandwidth. `action_distribution.csv` contains all actions 0-68, including
zero-count actions and no-op frequency, for each of the six policies and three
baselines.

Safety and integrity passed: **true**. Every method has
exactly 35 episodes (seven scenarios by five seeds); all 315 episodes reached
normal truncation with no abnormal termination, invalid action, mask
disagreement, reward mismatch, non-finite value, solver convergence failure,
or protected safety failure. Gains therefore preserve the governed safety
envelope. Churn is acceptable relative to the frozen comparators: bandit and PPO
both averaged about **2.148 reroutes/hour**, while bandit had fewer reversals and
flaps. Bandit moved more bandwidth than PPO (**1,603** versus **1,291 Mbps** per
episode) but far less than greedy (**9,964 Mbps**), whose reroute, reversal, and
flap rates were also much higher.

## Validation

Before holdout access, the evaluation-only repair passed 81 focused
learning/compatibility tests, 14 freeze/pin tests (110 deselected), and the
440-test full suite. The repair commit
`f7ed0f407c50c5472ecff89f977bc656439a8c49` was pushed before evaluation.

After completion, independent checks reconciled all 315 raw episode summaries
to the per-root, scenario, and aggregate tables; matched every episode length
to its compressed step file; revalidated every payload and sidecar hash;
confirmed unique keys at every table grain; and matched action counts to step
counts. Fresh final verification repeated the same results: 81 focused tests,
14 freeze/pin tests, and 440 full-suite tests passed.

## Runtime and provenance

| Root | Algorithm | Wall seconds | Device | Peak GPU bytes |
| ---: | --- | ---: | --- | ---: |
| 42 | maskable_ppo | 22.909 | cuda | 15096832 |
| 42 | masked_bandit | 17.376 | cuda | 13386752 |
| 314159 | maskable_ppo | 21.309 | cuda | 16926720 |
| 314159 | masked_bandit | 16.155 | cuda | 13386752 |
| 271828 | maskable_ppo | 19.891 | cuda | 16926720 |
| 271828 | masked_bandit | 17.572 | cuda | 13386752 |

Total one-shot evaluation runtime was **152.093 seconds**.
Evaluation source: `f7ed0f407c50c5472ecff89f977bc656439a8c49`. Checkpoint payload and sidecar
hashes, training-source bindings, exact paths, and worktree heads are in
`checkpoint_provenance.csv`.

Full compressed step evidence and per-episode summaries are preserved at:

`C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees\final_holdout_v2\runs\v2\final_holdout_20260731_f7ed0f4`

## Failures and limitations

Before holdout access, the original source-equality/seed gate blocked the
authorized cross-source workflow and was repaired in the pushed evaluation-only
commit above. A separate cold-context CUDA peak-memory preflight call failed
before device initialization; no environment or policy was constructed, and a
real forward/backward/optimizer CUDA operation then passed. Neither event was a
holdout attempt.

No holdout episode failed, retried, or was omitted. The final comparison covers three
training roots, five holdout seeds, seven fixed scenarios, two frozen learner
families, and three fixed baselines. Baselines are evaluated once because they
have no training root. High scenario variance limits broad generalization, and
the result cannot establish that memory or planning would never help under a
different learner, observation design, or task.

## Final scientific conclusion

The masked contextual bandit advantage generalizes to the untouched holdout: it
beat PPO on 3/3 training roots, six of seven scenarios, and by 9.185 return
points in the root-aggregated learner comparison. This final evidence does not
support a need for temporal planning; the explicitly myopic learner remains
stronger. That is a result about these frozen learners and scenarios, not a
claim that planning is generally useless. The gains preserve safety and
acceptable churn under the frozen comparison, despite higher moved bandwidth
than PPO. The governed V2 study ends here without a holdout-driven tuning
recommendation.
