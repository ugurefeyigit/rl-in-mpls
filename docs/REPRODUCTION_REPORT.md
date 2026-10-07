# Reproduction report

Every central claim of the closed V2 study was re-executed from the
repository. Numbers below come from `experiments/processed/*.csv` and
`results/tables/reproduction_*.csv` (built by `scripts/study/analyze.py`);
historical numbers come from `results/v2_final_holdout/` and
`results/v2_three_root_continuity/`.

## Machines

| | Historical (closed study) | This reproduction |
|---|---|---|
| CPU | Intel i7-14700HX, 20 cores | Intel Xeon @ 2.1 GHz, 4 vCPU |
| GPU | RTX 4070 Laptop (networks on CUDA) | none (networks on CPU, 1 thread) |
| OS | Windows | Linux 6.18 |
| Python / PyTorch | 3.13.4 / 2.11.0+cu128 | 3.13.16 / 2.14.1+cpu |
| SB3 / SB3-contrib | 2.9.0 / 2.9.0 | 2.9.0 / 2.9.0 |
| NumPy | 2.3.0 | 2.5.3 |
| Code | `ca64b62` (root 42), `6a8a406` (others) | `1457e9b` in a clean detached worktree (descendant; frozen definitions identical) |
| Training wall time per run | 20–30 min | 42–51 min (3 runs concurrently) |
| Throughput | 221–334 transitions/s | 130–158 transitions/s |

## Classification of claims

| Claim (historical) | Status | Evidence |
|---|---|---|
| Environment: 18 routers, 64 directed links, 17 demands, k=4, obs 604, 69 actions | **reproducible** (verified in code) | `docs/REPOSITORY_AUDIT.md` §5 |
| Baselines' holdout results (static −101.85, greedy −2.33, CSPF −28.34) | **reproduced exactly** | all 21 (policy, scenario) means and SDs within 2.8·10⁻¹⁴ (`experiments/processed/baseline_reproduction.csv`) |
| Training traffic of each root | **reproduced exactly** | all 6 episode-seed ledgers match the recorded SHA-256 (after CRLF→LF) |
| Bandit holdout return 18.22 (3-root mean) | **reproduced** | 18.04 (roots: 14.67 / 17.21 / 22.23 vs historical 16.13 / 20.55 / 17.99) |
| PPO holdout return 9.04 | **not reproduced** | −12.04 (roots: −20.13 / 12.99 / −28.97 vs historical 6.57 / 8.30 / 12.24) |
| Bandit beats PPO on 3/3 roots, by 9.19 | **direction reproduced, magnitude not** | 3/3 roots, gap 30.1, two-stage bootstrap [4.4, 50.8] (with 3 roots this is close to the range of the root means, not a population interval); t-interval over 3 roots [−29.2, 89.3]. Historical gap's 3-root t-interval: [1.1, 17.3] |
| PPO best in `deceptive_local_optimum` (+1.11 over bandit) | **not reproduced** | bandit − PPO = +14.7 (two-stage bootstrap over roots then episodes [8.0, 23.3]; positive on 3/3 roots), bandit ahead in 15/15 paired episodes |
| Bandit beats PPO in 6/7 scenarios | **reproduced in direction** | bandit ahead in 6/7; `ood_double_failure` −5.1 [−24.2, 11.9], positive on 1/3 roots (historical +1.7) |
| Zero invalid actions / mask disagreements / solver / safety failures | **reproduced** (and independently audited) | all reproduced runs completed; mask audit in `docs/MASK_VALIDATION.md` |
| Selected checkpoints (bandit 250k/300k/400k; PPO 250k/350k/150k) | **not reproduced** (expected) | bandit 400k/250k/350k; PPO 350k/200k/400k — selection is sensitive to small curve differences |
| PPO learning curve "non-monotonic" | **reproduced qualitatively** | see below |

## Why the bandit reproduces and PPO does not

The two learners of a root saw the same training episodes as historically.
What differs is the device (CUDA vs CPU) and library versions (torch
2.11.0+cu128 → 2.14.1+cpu, numpy 2.3.0 → 2.5.3, Python 3.13.4 → 3.13.16).
That changes floating-point arithmetic for both learners and, for PPO only,
the action-sampling random stream: SB3 samples from the device's torch
generator, whereas the bandit's ε-greedy and replay sampling use a NumPy
generator that is device-independent. *(Corrected after round-2 review; an
earlier version attributed the difference to floating-point arithmetic alone.)* Validation (continuity-seed) curves of
the reproduction, per 50k transitions:

| Run | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit r42 | −23.4 | −14.1 | 2.5 | 6.3 | 15.1 | 19.9 | 19.3 | 23.8 |
| bandit r314159 | 2.2 | 5.0 | 16.8 | 16.8 | 26.5 | 22.2 | 23.2 | 18.2 |
| bandit r271828 | −18.5 | 12.3 | 20.1 | 12.4 | 17.9 | 16.9 | 28.7 | 26.9 |
| PPO r42 | −26.3 | −29.9 | −18.6 | −28.3 | −16.6 | −16.8 | −10.1 | −47.2 |
| PPO r314159 | −60.9 | −23.5 | 1.3 | 21.2 | 11.2 | 9.9 | 14.1 | 16.4 |
| PPO r271828 | −33.1 | −25.4 | −28.0 | −23.5 | −21.7 | −22.7 | −19.5 | −17.1 |

Historical continuity curves had the same shape for the bandit (monotone rise
to ≈20–30 by 250–400k) and the same instability for PPO (e.g. root 42:
−8.5, −2.9, 10.3, −16.8, 13.5, 2.9, −8.8, −17.8). The bandit's outcome is
insensitive to a change of execution (device, versions); PPO's is not: on identical data, two
of three reproduced PPO roots never reached a positive validation return.
We read this as evidence about PPO's optimization in this environment (high
run-to-run variance), not as a defect of either execution. It also means the
historical PPO numbers are one draw from a wide distribution, and that
three roots are too few to estimate PPO's mean performance precisely.

## Post-V2 protocol (independent seeds)

Under the post-V2 protocol (select on validation seeds 2001–2005, report on
test seeds 3001–3020, 140 episodes) the same six reproduced runs give the same
picture (bandit − PPO = +30.6, two-stage 95 % CI [9.1, 50.6], 3/3 roots).
Results with all training roots, PPO tuning and the horizon/delay studies are
in `docs/RESULTS_MANIFEST.md` and the paper.

## Not reproducible from the repository alone

* The historical checkpoints themselves and per-episode holdout data (not
  archived). Historical learner numbers are therefore compared at the
  (root, scenario) aggregate level that was committed.
* Bit-exact neural training across CPU/GPU.
