# Experiment log

Chronological record of every experiment launched in the post-V2 study,
including failed and superseded attempts. Implementation/process failures are
marked **[process]**; scientifically informative negative or null results are
marked **[science]**. The closed V2 study's own failures (SB3 seed-propagation
bug, root-guard and selection-guard repairs, CUDA preflight) are recorded in
`results/v2_seed42/PILOT_REPORT.md`, `results/v2_three_root_continuity/REPORT.md`
and `results/v2_final_holdout/FINAL_HOLDOUT_REPORT.md` and are not repeated.

Machine for everything below: 4-core Intel Xeon @ 2.10 GHz (cloud VM), 15 GB
RAM, no GPU; Python 3.13.16, PyTorch 2.14.1+cpu, SB3/SB3-contrib 2.9.0,
Gymnasium 1.4.0, NumPy 2.5.3. All learners run on CPU with one thread.

| # | When (UTC, 2026-10-06/07) | What | Outcome |
|---|---|---|---|
| 1 | 22:23 | Full test suite on fresh install | 802 passed, 10 skipped, 16 failed; all failures = missing external checkpoints [process] |
| 2 | 22:27 | 8,192-transition smoke runs of both governed learners (CPU) | ok; 155–157 transitions/s single-process |
| 3 | 22:29 | First launch of 6 governed reproduction runs in the main checkout | **all 6 aborted at start**: governed trainer refuses a dirty checkout (an untracked `logs/` directory) [process]. Relaunched in a clean detached worktree at `1457e9b` (`.worktrees/repro`). |
| 4 | 22:31 | Relaunch: roots 42, 314159 (PPO), 42 (bandit) first, 3 in parallel | running |
| 5 | 22:31 | Sequentiality diagnostic, greedy reference, seeds 4001–4003, H ≤ 24, clairvoyant | see `docs/SEQUENTIALITY_AUDIT.md` |
| 6 | 22:40 | `scripts/study/queue.py` crashed on import: module name shadowed stdlib `queue` (imported by networkx) [process] | renamed to `scheduler.py` |
| 7 | 22:42 | Study-trainer equivalence test (8,192 transitions, both learners) | **bit-identical** network parameters to the governed trainer |
| 8 | 22:44 | Baselines on historical holdout seeds 1001–1005 | all 21 (policy, scenario) means reproduce historical values to ≤ 3e-14 |
| 9 | 22:47 | Mask audit pass 1 (engine-side checks) | 15,480 states, 1,068,120 checks, 0 violations |
| 10 | 23:00 | Early look at diagnostic (full_day): myopic action agrees with 24-step best in 35 % of states. Added frozen-exogenous counterfactual to separate predictable persistence value from clairvoyant anticipation. | frozen run queued after #5 |
| 11 | 23:02 | Mask audit pass 2 (+ independent from-scratch projection, + PPO probability mass) | see `docs/MASK_VALIDATION.md` |

Entries after #11 are appended as experiments complete.
