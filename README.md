# When does MPLS traffic engineering need sequential RL?

> **Author:** Uğur Efe Yiğit · **License:** proprietary, all rights reserved
> ([LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
> This repository is publicly readable but not open source.

**Research question.** In a masked, incremental MPLS traffic-engineering
problem — one LSP move per 5-minute interval, persistent routes, move costs,
hold-down timers, protected traffic classes — does a sequential RL agent
(MaskablePPO, γ = 0.995) beat a learner that only predicts the *immediate*
reward of each action (a masked neural contextual bandit, γ = 0)? If so,
why? If not, when would it?

**Principal finding (so far; numbers are generated, see table below).** The
myopic bandit beats PPO on every training root and matches a non-learning
per-interval min-max-utilization MILP controller; PPO is below both and its
outcome is unusually sensitive to numerical perturbation. The closed
predecessor study's bandit result reproduces on different hardware; its PPO
result and its "PPO wins the deceptive scenario" claim do not. Clairvoyant
lookahead diagnostics show that the problem does contain non-myopic structure
(moves whose one-off cost exceeds one interval's gain), so the bandit's win is
not because the problem is trivially myopic — see
[docs/SEQUENTIALITY_AUDIT.md](docs/SEQUENTIALITY_AUDIT.md) and the paper.

**What is here.** A deterministic flow-level MPLS-TE simulator (18 routers,
64 directed links, 17 demands, 4 candidate LSPs each, observation 604,
Discrete(69) with action masks), the frozen "Environment V2" decision problem,
PPO / contextual-bandit / masked-Q(γ) learners, static / greedy / CSPF / MILP
baselines, clairvoyant lookahead oracles, a controlled delayed-effect variant,
an experiment registry, and a paper built from scripts.

**Reproduce:** `python scripts/reproduce.py --suite sanity` (≈10 min, CPU).

---

## Results summary

<!-- RESULTS:BEGIN -->
| Policy | Test return [95 % CI] | Roots | Delivered | SLA viol. | Reroutes/h |
|---|---:|---:|---:|---:|---:|
| Oracle-1 † (exact next-interval reward) (partial: 137 episodes) | 49.0 [47.3, 50.7] | – | 96.05 % | 130 | 3.33 |
| Masked contextual bandit (γ=0) | 25.2 [21.5, 28.1] | 5 | 95.31 % | 164 | 2.30 |
| MILP-track (per-interval min-max-util, no learning) | 24.0 [21.9, 26.3] | – | 95.51 % | 162 | 5.83 |
| Greedy | 4.0 [1.7, 6.3] | – | 94.85 % | 187 | 4.60 |
| MaskablePPO (γ=0.995) | 0.3 [-13.7, 13.6] | 5 | 94.79 % | 190 | 6.38 |
| CSPF | -25.6 [-28.0, -23.2] | – | 93.64 % | 243 | 0.61 |
| No-op | -95.5 [-98.0, -92.9] | – | 90.58 % | 352 | 0.00 |
| Static shortest path | -96.5 [-98.9, -94.0] | – | 90.40 % | 346 | 0.37 |
| Random valid | -107.5 [-111.3, -103.6] | – | 89.44 % | 401 | 11.66 |

Bandit − PPO, paired: **24.9** [12.2, 39.1], positive on 5/5 roots.
<!-- RESULTS:END -->

All numbers: 7 scripted scenarios × 20 test seeds (3001–3020), paired;
learners selected on separate validation seeds; 95 % CIs (two-stage bootstrap
over training roots and episodes for learners; stratified bootstrap over
episodes for fixed policies). † clairvoyant reference, not a controller.

## Documents

| Document | Content |
|---|---|
| [paper/main.pdf](paper/main.pdf) | the research paper (LaTeX source in `paper/`) |
| [report/technical_report.pdf](report/technical_report.pdf) | longer technical report (thesis-chapter style) |
| [slides/talk.pdf](slides/talk.pdf) | 12–15 minute research talk |
| [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md) | the study in two pages |
| [docs/MATHEMATICAL_FORMULATION.md](docs/MATHEMATICAL_FORMULATION.md) | the decision problem, exactly as implemented |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | every policy, with pseudocode mapped to source |
| [docs/HYPOTHESES.md](docs/HYPOTHESES.md) | hypotheses and falsification criteria, committed before the new results |
| [docs/SEQUENTIALITY_AUDIT.md](docs/SEQUENTIALITY_AUDIT.md) | how much does an action affect the future? |
| [docs/MASK_VALIDATION.md](docs/MASK_VALIDATION.md) | independent audit of the action mask |
| [docs/REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md) | what of the closed V2 study reproduces |
| [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md), [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | every experiment, including failures |
| [docs/RESULTS_MANIFEST.md](docs/RESULTS_MANIFEST.md) | provenance of every figure, table and number |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | environment, seeds, commands, runtimes |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | what this work does not show |
| [docs/HOSTILE_REVIEWS.md](docs/HOSTILE_REVIEWS.md) | networking, RL and paper-reviewer critiques and responses |
| [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md) | forensic audit of the inherited repository |
| [literature/LITERATURE_MATRIX.md](literature/LITERATURE_MATRIX.md) | verified related work and novelty audit |
| [docs/PRODUCT_README.md](docs/PRODUCT_README.md) | the interactive dashboard / product (original README) |

## Installation

Python ≥ 3.11 (tested 3.13), Linux/macOS/Windows, CPU only.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/lock-cpu-py313.txt   # exact tested versions
python -m pip install -e . --no-deps
python -m pytest -q -m "not slow"                          # ≈ 850 tests
```

## Reproducing

| Command | Runtime (4 cores) |
|---|---|
| `python scripts/reproduce.py --suite sanity` — tests, exact baseline reproduction of the closed study, tiny training of both learners | ≈ 10 min |
| `python scripts/reproduce.py --suite core` — one more training root, all references, one diagnostic seed, tables, figures | ≈ 3–4 h |
| `python scripts/reproduce.py --suite full` — every registered experiment, diagnostics, paper | ≈ 20–24 h |
| `python scripts/reproduce.py --suite analysis` — rebuild tables, numbers, figures, PDFs from existing raw results | ≈ 2 min |

Single pieces: `python scripts/study/job.py <run_id>` (train + select + test one
registered run; `--list` shows ids), `scripts/study/run_oracles.py`,
`scripts/study/run_seqdiag.py`, `scripts/study/run_mask_audit.py`. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository structure

```
configs/                 topology, traffic classes, scenarios (shared); V2 env/obs/reward/learning (frozen)
mplssim/sim, rl, paths,  simulator and frozen Environment V2 (engine_v2, env_v2, reward_v2, candidates_v2)
  traffic, core
mplssim/experiments/     closed-study learners, governed trainer/evaluator, audit wrapper, freeze pin
mplssim/study/           this study: evaluator, oracles, Q-learner, delay variant, MILP baseline,
                         registry, trainer, statistics, diagnostics, collector
mplssim/baselines/       static / greedy / CSPF controllers
scripts/study/           job runner, scheduler, diagnostics, analysis; scripts/reproduce.py
experiments/registry/    experiment definitions (E1–E6)        experiments/raw/  versioned per-episode outputs
experiments/processed/   tidy tables                           results/          tables, figures, closed-study evidence
paper/, report/, slides/ LaTeX sources (numbers in paper/generated/numbers.tex are generated)
docs/, literature/       documentation listed above
server/, frontend/, mplssim/product, mplssim/evidence   interactive dashboard (see docs/PRODUCT_README.md)
tests/                   unit, integration and equivalence tests
```

## Known limitations (short)

Flow-level model with exogenous, inelastic traffic and an instantaneous
control plane; one hand-built topology and scripted scenarios; five training
roots; budget 400k transitions; one reward specification. Full list:
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Citing

See [CITATION.cff](CITATION.cff).
