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
| 12 | 23:00 | Diagnostics launched with `nice` still took ~75 % of CPU: kernel autogroup scheduling ignores niceness across sessions [process] | disabled `kernel.sched_autogroup_enabled`; training throughput restored |
| 13 | 23:17–23:21 | First reproduction trio finished (bandit r42, PPO r42, PPO r314159); second trio started | evaluated automatically under both protocols |
| 14 | 23:35 | Study scheduler had never started (it counted the four low-priority diagnostics as occupying slots); on restart it launched 3 jobs because it matched only commands starting with `python` and not `/usr/bin/python` [process] | fixed process matching; scheduler now also skips runs already executing (a restart would otherwise have renamed a live run directory) |
| 15 | 00:27–00:43 | All 6 reproductions evaluated; E1 bandit r161803, r141421 and PPO r161803 completed | `docs/REPRODUCTION_REPORT.md` |
| 16 | 00:43 | **Worker restart killed every background process** [process]. Lost: E1 PPO r141421 (608 episodes) and E2 qg05 r42 (672 episodes) mid-training, mask-audit pass 2, partial diagnostics. Persisted: all completed runs and every per-episode output. | interrupted run directories kept as `*.interrupted`; runs restarted from scratch (study checkpoints hold no replay buffer, so they are not resumable); diagnostics resumed per file; `scripts/study/relaunch_background.sh` makes relaunch idempotent |
| 17 | 00:50 | MILP-track on all 140 test episodes | statistically indistinguishable from the bandit; above PPO [science] |
| 18 | 01:00–01:15 | Model fidelity (root 42) and deceptive-scenario case study | PPO's actions are worse than no-op over 6 intervals; PPO flips one demand every 3 intervals (hold-down expiry) [science] |
| 19 | 01:15 | Return decomposition over all test episodes | on 3 of 4 PPO roots > 50 % of moves are reversals; PPO's network utility is comparable to the bandit's but move/reversal costs are 5–13× larger; shaping contributes ≤ 0.02 per episode [science] |
| 20 | 01:40–02:00 | Frozen-exogenous sequentiality diagnostic complete (495 states paired with the clairvoyant pass) | myopic = 24-step best in 88 % of states with traffic frozen vs 33 % with true future; non-myopic value is anticipation of exogenous change [science]. Motivated post-hoc E7 (`docs/HYPOTHESES.md` H9) |
| 21 | 01:47 | E1 complete (5 roots per learner) | bandit − PPO positive on 5/5 roots |
| 22 | 02:27 | E2 PPO γ = 0, 3 roots | **23.6 vs bandit 24.2 on the same roots (−0.6 [−5.2, 2.7]); PPO γ = 0.995: −6.4.** Flapping disappears (≤ 2.4 reversals/episode). Supports H2 and H10 [science] |
| 23 | 02:35 | Compute re-planning (≈34 run-equivalents left). Training slots 3 → 4; queue reordered by information value (Q γ0.9 → E4 → E7 → PPO γ0.9 → E3 → rest of E2 → E6 → E5). E3 trimmed from 8 to 6 configurations (`ns128`, `net64` dropped **before running**) [process] | |
| 24 | 02:50 | Oracle-3 complete (5 test seeds) | Oracle-3 − Oracle-1 = +2.9 [1.5, 4.2], versus Oracle-1 − bandit ≈ 22 on the same episodes [science] |
| 25 | 03:13 | E2 Q-learner γ = 0.9, 3 roots | 19.4 vs bandit 24.2 (−4.8 [−7.0, −2.1]), 0/3 roots better; supports H3 [science] |
| 26 | 04:07 | E4 delayed activation (L = 1), bandit and PPO, roots 42/314159 | bandit 22.0 → −30.3; PPO 1.6 → 1.2; PPO − bandit = +31.5 [20.6, 42.6] under delay, 2/2 roots. Supports H6 [science] |
| 27 | 04:10 | Oracle-6 complete (5 test seeds) | Oracle-6 − Oracle-1 = +1.4 [0.5, 2.5]; Oracle-1 − bandit ≈ 22 [science] |
| 28 | 04:25 | E3 extended (post-hoc, before any E3 result): `rewnorm` (reward normalization) and `gae08` (GAE λ = 0.8), variance-reduction knobs that target the proposed mechanism while keeping γ = 0.995 | queued |
| 29 | 04:45 | **E3 selection rule fixed before any E3 result:** each configuration's score is the validation mean (seeds 2001–2005, 35 episodes) of its best checkpoint, exactly as checkpoint selection for every learner. The configuration with the highest score is retrained on roots 314159 and 271828 (E3b) if its score exceeds the default PPO's root-42 validation score; otherwise E3 is reported as "no configuration improved validation return" and E3b is not run. Test returns of E3 runs are reported for completeness only and play no role in selection. Caveat recorded in advance: one-root selection is noisy, given per-root shifts of up to 41 points from numerical noise alone (#15) [process] | |
| 30 | 05:05 | Sequentiality diagnostic on the delayed variant (L = 1), frozen pass first, then clairvoyant; greedy reference, seeds 4001–4003. Post-hoc (H11), recorded before launch; tests whether the diagnostic flags the regime where the bandit fails [science] | running (low priority) |

Entries after #28 are appended as experiments complete.
