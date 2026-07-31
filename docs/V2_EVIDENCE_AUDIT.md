# V2 evidence audit

An independent reconciliation of the closed governed V2 study, run before any
release work was allowed to proceed. It recomputes the published claims from the
committed CSV and JSON rather than reading the prose reports, and it checks the
grain of every table.

**No evidence file was modified. No discrepancy was "resolved" by editing an
artifact.** The audit is read-only by construction; the loader it exercises opens
frozen paths for reading only, and a test asserts that no write is attempted
anywhere under `results/` or `runs/`.

**Result: passed.** Nothing material disagreed. Two look-alike pairs are recorded
below because they are easy to mistake for discrepancies and must never be
conflated in a UI or a document.

## Scope

Audited artifacts, all under `results/`:

| Directory | Stage | Files reconciled |
| --- | --- | --- |
| `v2_final_holdout/` | final holdout | `aggregate_metrics.csv`, `per_root_metrics.csv`, `scenario_metrics.csv`, `reward_components.csv`, `action_distribution.csv`, `evaluation_integrity.csv`, `checkpoint_provenance.csv`, `manifest.json`, `FINAL_HOLDOUT_REPORT.md` |
| `v2_three_root_continuity/` | development | `aggregate_metrics.csv`, `comparison_metrics_by_root.csv`, `scenario_metrics.csv`, `learning_curves.csv`, `checkpoint_selection.csv`, `training_summary.csv`, `training_integrity.csv`, `evaluation_integrity.csv`, `manifest.json`, `REPORT.md` |
| `v2_seed42/` | development | `comparison.csv`, `learning_curve.csv`, `ppo_checkpoint_selection.csv`, `bandit_checkpoint_selection.csv`, `manifest.json`, `PILOT_REPORT.md` |

## Grain and coverage

Verified at every table grain, not only the headline:

- **Per policy.** Nine unique `policy_id` values — six learner checkpoints and
  three baselines — each with exactly 35 episodes, summing to 315.
- **Per scenario.** 63 rows (9 × 7), each with 5 episodes, summing to 315. Keys
  unique on `(policy_id, scenario)`. Every scenario name is one of the frozen
  seven.
- **Per seed.** Every policy covers exactly 5 unique seeds and 7 unique
  scenarios, per `evaluation_integrity.csv`.
- **Per action.** 621 rows (9 × 69), covering actions 0–68 for every policy
  including zero-count actions. Frequencies sum to 1 within each policy.
- **Per checkpoint.** Six provenance rows, one per (root, learner).

Episode horizons are per-scenario, not a single global length:
288 + 84 + 60 + 60 + 60 + 60 + 48 = 660 steps per seed, so each policy recorded
3,300 steps. That figure reconciles independently with the action-distribution
totals.

## Root-aware aggregation

The critical check, because pooling would be a scientific error: the aggregate
row must be the unweighted mean of the three training-root means, not a pool of
the 105 episodes.

Verified for **every numeric column**, not just return: for both learners, each
aggregate value equals the mean of its three per-root values to within 1e-9.

| Method | Aggregate return | Mean of root means | `root_mean_std` | Recomputed sample SD (ddof=1) |
| --- | ---: | ---: | ---: | ---: |
| masked_bandit | 18.220918162847 | 18.220918162847 | 2.218363427 | 2.218363427 |
| maskable_ppo | 9.035842086078 | 9.035842086078 | 2.908347635 | 2.908347635 |

Baselines have no training root: each carries `root_count = 1`,
`root_mean_std = 0.0`, and 35 episodes. They were evaluated once and are not
replicated per root.

The scenario grain also rolls up correctly: for all nine policies, the per-root
return equals the mean of its seven scenario means to within 1e-9.

## Headline claims

Every published figure reproduced from the frozen tables:

| Claim | Reported | Recomputed |
| --- | ---: | ---: |
| Bandit mean return | 18.221 | 18.220918 |
| PPO mean return | 9.036 | 9.035842 |
| Bandit advantage | 9.185 | 9.185076 |
| Greedy mean return | -2.327 | -2.326854 |
| Training roots won by bandit | 3 / 3 | 3 / 3 |
| Scenarios won by bandit | 6 / 7 | 6 / 7 |
| PPO lead in `deceptive_local_optimum` | 1.107 | 1.106617 |
| Largest bandit edge (`link_failure`) | 20.183 | 20.183059 |
| Reroutes/hour, both learners | 2.148 | 2.1480 / 2.1478 |

Greedy is confirmed as the strongest of the three baselines. The bandit has fewer
reversals (1.581 vs 2.000) and fewer flaps per demand (0.0930 vs 0.1176) than
PPO, and moves **more** bandwidth (1,602.86 vs 1,291.00 Mbps) — less than greedy
(9,963.93). The higher moved bandwidth is a real cost and is reported, not
smoothed away.

`deceptive_local_optimum` is the only scenario PPO wins. It is preserved as a
negative result against an across-the-board bandit claim, and the study surface
foregrounds rather than buries it.

## Reward-component integrity

Each policy's twelve named components were summed independently and compared to
its operational return.

- Twelve components confirmed: `delivery`, `protected_disconnect`,
  `unprotected_disconnect`, `sla_severity`, `max_util`, `overload`, `potential`,
  `move_fixed`, `move_volume`, `move_divergence`, `reversal`, `invalid`.
- Worst recomputed residual across all nine rows: **5.684e-14**.
- Largest residual the study reported, after separately aggregating episode
  components: **1.7053e-13**. Confirmed as the maximum of
  `max_abs_reward_residual`.

Every reward row reconciles to the matching `per_root_metrics.csv` return exactly.

## Safety and integrity

`evaluation_integrity.csv` was checked rather than trusted:

- `all_checks_passed` true for all nine policies.
- All episodes truncated normally; none terminated abnormally.
- All six counters zero across every policy: invalid actions, mask
  disagreements, reward mismatches, non-finite values, solver convergence
  failures, protected safety failures.
- Protected disconnection demand-intervals identical across every method
  (3.571), as are unprotected (10.714). Rejected TE requests: zero everywhere.

## Provenance

- All six checkpoints cite evaluation source `f7ed0f4…`.
- Training sources are exactly the two approved identities: `ca64b62…` (seed-42)
  and `6a8a406…` (continuation). No third source appears.
- Each row's `artifact_worktree_head` equals its `training_source_sha`.
- All twelve payload and sidecar hashes are distinct, well-formed SHA-256.
- Selected transitions match the closeout table: root 42 → 250k/250k,
  root 314159 → PPO 350k / bandit 300k, root 271828 → PPO 150k / bandit 400k.
- `manifest.json` asserts `training_performed`, `tuning_performed`,
  `checkpoint_selection_performed`, `checkpoint_sweep_performed` and
  `holdout_used_for_debugging` are all **false**. The loader refuses to serve the
  evidence if any of those is not false.

## Two look-alike pairs — not discrepancies

Both were flagged during the audit and both resolved to "two different
statistics that share a name". Neither is an error in the evidence. Both are now
surfaced separately, with their grain stated, everywhere they appear.

### No-op share

| Method | Pooled over steps | Mean over episodes |
| --- | ---: | ---: |
| masked_bandit | 87.09% | 82.10% |
| maskable_ppo | 87.31% | 82.10% |
| greedy | 64.64% | 59.06% |
| cspf | 95.97% | 94.70% |
| static | 98.06% | 96.95% |

The pooled figure counts action 0 across all 3,300 recorded steps
(`action_distribution.csv`). The episode-mean figure averages each episode's own
no-op frequency (`noop_frequency_mean` in the metric tables). Because episode
lengths differ by scenario, the two do not coincide.
`FINAL_HOLDOUT_REPORT.md` quotes the **pooled** figure.

### Wall time

- `manifest.json → runtime.total_wall_seconds` = **152.093 s** — the whole
  one-shot runner, including the three baselines and setup.
- Sum of the six `checkpoint_provenance.csv → evaluation_wall_seconds` =
  **115.213 s** — the learner evaluations only.

An early audit assertion required these to be equal. That assertion was wrong;
the evidence is right. Both figures are now reported side by side, each labelled.

## What the audit does not claim

The audit verifies internal consistency, coverage, grain, and identity of the
frozen artifacts. It does not re-run the evaluation, does not load a checkpoint,
and cannot verify that the recorded numbers were produced by the code that claims
to have produced them beyond the recorded SHA bindings and payload hashes.

The scientific limitations of the study itself are unchanged and are stated in
`results/v2_final_holdout/FINAL_HOLDOUT_REPORT.md` and
[TECHNICAL_DEFENSE.md](TECHNICAL_DEFENSE.md): three training roots, five holdout
seeds, seven scenarios, one topology, one reward design, high scenario variance.

## Reproducing the audit

The reconciliation lives in the tested evidence layer, not in a one-off script:

```bash
python -m pytest tests/test_evidence_loader.py tests/test_evidence_claims.py -q
```

`tests/test_evidence_claims.py` recomputes each headline claim from the frozen
files and fails on any drift, including drift that would round into a different
conclusion. `tests/test_evidence_loader.py` proves the loader rejects a missing
file, a missing column, a wrong episode count, a foreign source SHA, an
unexpected seed set, a failed integrity flag and a non-zero safety counter — and
that loading never opens a governed path for writing.
