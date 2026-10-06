# Experiment registry

Machine-readable definitions live in `experiments/registry/*.yaml` and are
loaded by `mplssim/study/registry.py`. A run's stable identifier is
`<family>__<tag>__r<root>` (e.g. `E2_horizon__qg09__r314159`); its fully
resolved configuration is written to its manifest
(`experiments/raw/learner_eval/<run_id>/manifest.json`). `python
scripts/study/job.py --list` prints every registered run.

## Fixed protocol (all families)

| Item | Value | Source |
|---|---|---|
| Environment | frozen `MplsTeEnvV2` (`mpls-te-v2.0.0`, obs 604, Discrete(69)) unless a variant is named | `configs/experiments/rl_env_v2.yaml` |
| Training scenario | `random_day` (randomized bursts, failures, flash crowds per episode seed) | `configs/scenarios.yaml` |
| Vector environments | 16 (`DummyVecEnv`, serial) | as closed study |
| Budget | 400,000 transitions, checkpoints every 50,000 (E5: 1.2 M / 100 k) | registry |
| Evaluation scenarios | full_day, evening_peak, flash_crowd, link_failure, deceptive_local_optimum, ood_double_failure, overload_stress | `mplssim/study/protocol.py` |
| Checkpoint selection | max mean return on VALIDATION seeds 2001–2005 (35 episodes); ties → earlier | `mplssim/study/evalrun.py` |
| Reported test | selected **and** final checkpoint on TEST seeds 3001–3020 (140 episodes) | idem |
| Policy at test | deterministic (masked argmax) | |
| Hardware | 4-core Xeon 2.1 GHz, CPU only, 1 thread per run | `docs/EXPERIMENT_LOG.md` |

Seed namespaces are disjoint (validation, test, diagnostic, historical
continuity, historical holdout); training episode seeds
`root + rank + 1024·episode` never coincide with them for the roots used.

## Families

| Family | Hypothesis tested | Runs | Variable | Roots |
|---|---|---|---|---|
| **E0_repro** | The closed study's learner results reproduce on different hardware. | `bandit`, `ppo` (governed trainer, clean worktree at `1457e9b`) | — | 42, 314159, 271828 |
| **E1_main** | The bandit–PPO ordering holds with more training roots. | `bandit`, `ppo` | — | 161803, 141421 |
| **E2_horizon** | Bootstrapped future value (γ>0) improves a value learner; a shorter horizon improves or does not hurt PPO. | `qg05`, `qg09`, `qg099` (masked Q, γ = 0.5/0.9/0.99); `ppog0`, `ppog09` (PPO γ = 0/0.9) | γ | 42, 314159, 271828 |
| **E3_ppo_tuning** | PPO's deficit is a tuning artefact. | 8 one-factor variations of lr, rollout length, entropy, clip, width | PPO hyper-parameters | 42 (selection), best config on remaining roots |
| **E4_delay** | When an action's effect is delayed by one interval, the myopic learner fails and sequential learners recover the lost value. | `L1_bandit`, `L1_ppo`, `L1_qg09` | TE activation delay L = 1 | 42, 314159 |
| **E5_budget** | PPO catches up with more data; the bandit plateaus. | `bandit_1p2m`, `ppo_1p2m` | budget 1.2 M | 42 |

E0 + E1 together form the primary comparison (5 roots × 2 learners). Pooling is
justified because the study trainer is bit-identical to the governed trainer
(`tests/test_study_trainer_equivalence.py`).

## Non-learning references (evaluated on the same TEST episodes)

`static`, `greedy`, `cspf`, `noop`, `random_valid` (all 140 test episodes);
clairvoyant oracles `oracle_h1` (all 140), `oracle_h3`/`oracle_h6` (subset,
see results). Script: `scripts/study/run_oracles.py`.

## Diagnostics (select nothing)

| Diagnostic | Script | Seeds |
|---|---|---|
| Sequentiality (action-value rollouts, H ≤ 24, clairvoyant and frozen-exogenous) | `scripts/study/run_seqdiag.py` | 4001–4003 |
| Action-mask audit | `scripts/study/run_mask_audit.py` | 5000–5005 (eval scenarios), 6000–6039 (`random_day`) |
