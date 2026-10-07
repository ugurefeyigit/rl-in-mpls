# Repository audit

*Forensic audit performed before any scientific change. Audited state: commit
`1457e9b` (tip of `exp2.1`, tag `v2.0.0` + Exp 2.1 UI release), reached from
`main` (`10e6d59`) by fast-forward of the working branch. Nothing was rewritten
or deleted; every pre-existing artifact is preserved at its original path.*

Statements are tagged **[verified]** (checked against code, configs, or a run in
this audit), **[historical]** (reported by a committed artifact, not
re-executed), **[inferred]**, or **[pending → doc]** (resolved by an
experiment documented elsewhere).

---

## 1. History and branches

| Ref | Tip | Content |
|---|---|---|
| `main` | `10e6d59` | V1: simulator, `MplsTeEnv` (586-dim obs), MaskablePPO `ppo_te`, baselines, dashboard, V1 report. 6 commits. |
| `optimize/runtime-evaluation`, `presentation-hardening` | `5e429bc`, `4b8de03` | V1 freeze (tag-like manifest), presentation mode, profiling, evaluation-integrity audit. |
| `feat/rl-environment-v2` | `d7d2b3f` | **The scientific core**: Environment V2, governed learning comparison (seed-42 pilot, three-root continuity, final holdout). |
| `feat/post-study-productization` … `final` | `e82d134` | Read-only evidence API, product UI, release `v2.0.0`. |
| `exp2.1`, `exp-2.1-comparative-run-results` | `1457e9b` | Exp 2.1 UI comparison surface (no scientific change). |

All branches are strict ancestors of `exp2.1` [verified with
`git merge-base --is-ancestor`]; history is linear. The working branch
`claude/adoring-euler-dwj3m9` was fast-forwarded to `1457e9b` before any new
commit, so the full V1 → V2 history is preserved.

Development was AI-assisted (repository-root handoff files `CODEX_HANDOFF.md`,
`OPUS5_PART1_HANDOFF.md`, `OPUS5_PART2_HANDOFF.md`, `NEXT_STAGE_HANDOFF.md`).
They are preserved as historical process records.

## 2. Architecture

```
configs/*.yaml ─────────────┐ (topology, classes/demands, scenarios: shared V1/V2)
configs/experiments/*.yaml ─┤ (V2 env, observation schema, reward, learning contract)
                            ▼
mplssim/core      Topology, Demand, TrafficClass            (graph, directed links)
mplssim/paths     candidates_v2: role-valid k=4 Yen paths   (fixed per demand)
mplssim/traffic   TrafficModel: diurnal × multiplier × events × AR(1)  (exogenous)
mplssim/sim       engine_v2: carried-flow fixed point, FRR, dwell, accounting
mplssim/rl        env_v2 (Gymnasium, 604-dim obs, Discrete(69), masks), reward_v2
mplssim/baselines static / greedy / cspf / random  (V1 controllers, read via adapter)
mplssim/experiments  v2_factory (freeze pin, identity), learning_common (audit
                  wrapper, ledgers, sidecars), masked_bandit, trainers_v2,
                  evaluation_v2 (governed evaluator, checkpoint selection)
mplssim/evidence, mplssim/product, server/, frontend/   read-only evidence API + UI
mplssim/study     (added by this audit) diagnostics, oracles, Q-learner, variants,
                  registry, trainer, evaluator
```

| Area | Lines (approx.) | Role in the science |
|---|---:|---|
| `mplssim/sim`, `rl`, `paths`, `traffic`, `core` | 7,500 | Decision problem (V1 + V2) |
| `mplssim/experiments` | 5,000 | V2 learning comparison |
| `mplssim/baselines` | 450 | Baselines |
| `mplssim/evidence`, `product`, `server`, `frontend` | 26,800 | UI/product; **not** part of the scientific pipeline |
| `scripts` | 7,900 | V1 and V2 entry points |
| `tests` | 24,100 | 828 tests (802 pass offline; see §6) |

The UI/product layer is roughly 3/4 of the code. It consumes frozen evidence and
does not feed back into training or evaluation [verified by import graph:
nothing under `mplssim/experiments` imports `product`, `evidence`, or `server`].

## 3. Experimental pipeline (V2, as executed historically)

1. **Freeze.** `v2_factory.PINNED_ENVIRONMENT_COMMIT = dca533b`; 16 definition
   files (`FROZEN_DEFINITION_PATHS`) must be byte-identical (LF-canonical) to
   that commit or training refuses to start.
2. **Train** (`scripts/train_v2.py` → `trainers_v2.train_experiment`):
   `random_day` scenario, 16 `DummyVecEnv` workers (serial, one process),
   400,000 aggregate transitions, checkpoints every 50,000; per-step JSONL
   telemetry; per-episode seed ledger (`root + rank + 1024·episode`).
3. **Select** (`scripts/compare_v2.py`): every checkpoint evaluated on 7
   scenarios × continuity seeds 101–105; highest mean return wins.
4. **Report continuity** on the *same* 35 episodes used for selection.
5. **Final holdout** (`scripts/final_holdout_v2.py`, CUDA-only CLI): selected
   checkpoints of 3 roots + 3 baselines on 7 scenarios × seeds 1001–1005, once.

Raw per-step/per-episode outputs, checkpoints and replay buffers of steps 2–5
were written to `C:\Users\ugure\...\.worktrees\…\runs\v2\` on the author's
machine and are **not in the repository** [verified]. Only compact CSV/JSON
summaries are committed (`results/v2_seed42`, `results/v2_three_root_continuity`,
`results/v2_final_holdout`).

## 4. Evidence inventory

| Artifact | Location | Granularity | In repo |
|---|---|---|---|
| V1 PPO `ppo_te` + ablations, checkpoints | `models/` | full model zips | yes (84 MB) |
| V1 evaluation | `results/eval_*.csv/json`, `results/figures/*.png` | per episode | yes |
| V2 validation (paths, solver convergence, reward calibration) | `results/environment_v2_validation/` | per case | yes |
| V2 seed-42 pilot | `results/v2_seed42/` | **per episode** (175 rows) | yes |
| V2 three-root continuity | `results/v2_three_root_continuity/` | per (root, algorithm, scenario) mean/SD | yes |
| V2 final holdout | `results/v2_final_holdout/` | per (root, algorithm, scenario) mean/SD | yes |
| V2 checkpoints (48), replay buffers, step logs, holdout episodes | author's machine | full | **no** |
| Normative specs `RL_ENVIRONMENT_V2_SPEC.md`, `MPLS_SIMULATION_REALISM_AUDIT.md`, `RL_ENVIRONMENT_V2_TEST_PLAN.md`, `RL_ENVIRONMENT_V2_MIGRATION_PLAN.md`, `RL_MODEL_TRAINING_CONTRACT.md` | cited in docstrings | — | **never committed** [verified: absent from all refs] |

## 5. Verification of the prior project description

| Claimed fact | Repository evidence | Verified value | Confidence | Notes |
|---|---|---|---|---|
| ≈18 routers | `configs/topology.yaml` | **18** (4 PE-in, 4 PE-out, 8 P, 2 AGG) | high | |
| ≈32 links | `configs/topology.yaml` | **32 undirected = 64 directed**, 100–2000 Mb/s per direction | high | File header still says "28 undirected (56 directed)" — stale comment. |
| ≈17 demands | `configs/traffic_classes.yaml` | **17** in 6 classes; 2 protected classes (voice, critical) | high | |
| ≈4 candidate paths per demand | `rl_env_v2.yaml: k_paths 4`, `candidate_paths_v2.csv` | **exactly 4** role-valid candidates for every demand (68 rows) | high | |
| Observation ≈604 | `rl_observation_v2.yaml`, env schema check | **604** (V2). V1 was 586. | high | Feature-major; no time-of-day by default. |
| Masked action space ≈69 | `env_v2.action_space` | **Discrete(69)** = no-op + 17×4 | high | Typically 48–52 legal actions per state [verified, `full_day` seed 101]. |
| Invalid actions prevented by masking | mask = vectorized `validate_te_action`; `AuditedV2Env` aborts on any illegal action | true | high | Re-validated independently: `docs/MASK_VALIDATION.md`. |
| Learners: MaskablePPO, masked contextual bandit | `trainers_v2.py`, `masked_bandit.py` | true | high | Bandit = neural immediate-reward regressor, masked ε-greedy (not LinUCB/Thompson). |
| Greedy/heuristic baselines | `baselines/controllers.py` | static, greedy, cspf (+random in V1 only) | high | In V2 only the **first** legal proposal of a baseline is submitted per interval (CSPF designed for ≤3 moves per re-optimization). |
| ≈400k transitions per learner | training manifests | **400,000** per run, 3 roots × 2 learners | high | |
| "Hundreds of held-out episodes" | `v2_final_holdout/evaluation_integrity.csv` | **35 episodes per policy** (7 scenarios × 5 seeds); 315 total over 9 policies | high | Only **35 distinct traffic instances**; learner results nest 3 training roots over them. "Hundreds" overstates the independent evidence. |
| Bandit substantially outperformed PPO overall | `FINAL_HOLDOUT_REPORT.md` | holdout mean return **18.22 vs 9.04** (+9.19), bandit ahead on 3/3 roots | high (as historical fact) | No CI was ever computed. **Reproduction (CPU):** bandit 18.0 (reproduced), PPO −12.0 (not reproduced); bandit ahead on 3/3 roots again; see `docs/REPRODUCTION_REPORT.md`. |
| PPO best in the deceptive-local-optimum scenario | `v2_final_holdout/scenario_metrics.csv` | PPO ahead of bandit by **1.11** return points (92.55 vs 91.44, ≈1.2 %) | low | Root-averaged, no CI. **Did not reproduce:** reproduced bandit − PPO = +14.7 in this scenario (two-stage bootstrap [8.0, 23.3]; positive on 3/3 roots). |
| No invalid-action / mask / solver / safety failures | integrity CSVs | zero in all historical runs | high, but weak evidence | Counters are **fail-fast**: any failure aborts the run, so "zero" is implied by completion. Independent large-sample validation added. |
| Substantial parallel simulation | `trainers_v2.build_training_vec` | **16 environments in a `DummyVecEnv`**, i.e. stepped serially in one process; only the networks ran on CUDA | high | Throughput 221–334 transitions/s (historical, GPU machine); 155 transitions/s per core here (CPU). |

## 6. Test suite status (this machine)

`python -m pytest`: **802 passed, 10 skipped, 16 failed**. All 16 failures are in
`tests/test_v2_live_foundation.py` and have one cause: the live V2 UI needs the
external checkpoint directory (`V2_LIVE_CHECKPOINTS`), which is not in the
repository. They are environmental, not code defects; they now skip with an
explicit reason when the directory is absent (see §10, item 9).

## 7. Technical debt

1. Two parallel stacks (V1 and V2) of engine, env, reward, training and
   evaluation coexist; V1 is retained for compatibility. New work targets V2.
2. The governed pipeline hard-codes seed sets, roots and learner families
   (`learning_common.validate_*`); adding an experiment required a new layer
   (`mplssim/study`) rather than configuration.
3. `final_holdout_v2.py` accepts only `--device cuda` and absolute Windows
   worktree paths; it cannot be re-run on a CPU machine as committed.
4. Per-step training telemetry is ~90 MB per run; the summaries actually used
   are per-episode.
5. Large untested-for-science UI layer; product docs and handoff files at the
   repository root obscure the research artifact.
6. No `pyproject.toml`; `requirements.txt` uses lower bounds only.

## 8. Scientific ambiguities

1. **Selection/reporting overlap.** Checkpoints were selected on continuity
   seeds 101–105 and continuity results were reported on the same episodes.
   Continuity tables are therefore optimistically biased; only the holdout
   numbers are unbiased. (Holdout handling itself was exemplary.)
2. **Aggregate return mixes horizons.** `full_day` has 288 decisions, the other
   six scenarios 48–84. The mean episode return weights `full_day` by ≈4×, and
   its absolute level (~+330) dominates the cross-scenario SD (≈155). Method
   differences must be analysed per scenario / paired, not as raw SDs.
3. **No uncertainty for learner comparisons.** Historical V2 reports give SDs
   but no CIs, effect sizes, or paired analysis; with 3 roots the
   root-level uncertainty is large.
4. **"Deceptive local optimum"** is a scripted burst on D2/D4/D5; whether it
   actually requires non-myopic behaviour was never measured.
5. **Training vs evaluation distribution.** Learners train only on
   `random_day` (randomized bursts/failures); all seven evaluation scenarios
   are distinct scripted scenarios, several with features rare in training
   (`overload_stress` 1.6× load, `ood_double_failure`). Evaluation is
   distribution-shifted by design; this is not documented as such.
6. **Potential-based shaping and γ.** The shaping term
   `0.2(0.995 Φ(s') − Φ(s))` is policy-invariant only for a learner whose
   discount equals 0.995 (PPO). For the γ=0 bandit it is a (bounded, ≤0.4)
   reward modification. Measured contribution is ~0.01 per episode
   (`reward_components.csv`), so practically negligible, but the formal
   statement in the reward docstring does not apply to the bandit.
7. **Observation omits time of day**, so the diurnal phase is only partially
   observable (through current offered volumes). Both learners share this.

## 9. Reproducibility blockers

| Blocker | Impact | Mitigation in this work |
|---|---|---|
| V2 checkpoints/raw logs not in repository | historical numbers unverifiable at episode level | Retrained all 6 runs on CPU in a clean worktree at `1457e9b` (`docs/REPRODUCTION_REPORT.md`). |
| Original runs on CUDA (RTX 4070); this machine CPU-only | neural results not bit-reproducible across devices | Baselines (deterministic) compared exactly; learners compared statistically. |
| Governed trainer requires clean checkout | untracked files abort runs | Reproduction in a detached worktree (as the author did). |
| `final_holdout_v2.py` CUDA/Windows-only | cannot rerun as committed | Equivalent holdout evaluation via `mplssim/study/episode.py`, tested equal to the governed evaluator. |
| Normative spec documents never committed | design rationale partly unrecoverable | Mathematical formulation reconstructed from code (`docs/MATHEMATICAL_FORMULATION.md`). |

## 10. Suspected bugs and issues

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | Topology header comment "28 links / 56 directed" contradicts the 32/64 defined. | cosmetic | Not fixed (frozen file); documented. |
| 2 | Integrity counters are fail-fast, so reported zeros carry no information beyond "run completed". | evidence quality | Independent mask/safety validation added. |
| 3 | Continuity results reported on selection episodes (§8.1). | methodological | New protocol separates validation and test seeds. |
| 4 | Baseline adapter submits only the first legal proposal; CSPF loses its multi-move re-optimization. | baseline strength | Documented; clairvoyant oracles added as stronger references. |
| 5 | Greedy/CSPF thresholds tuned for V1, never re-tuned for V2. | baseline strength | Documented as limitation. |
| 6 | V2 SB3 seeding bug (rank counted twice) | was fatal | Fixed historically (`ca64b62`), regression-tested; preserved in history. |
| 7 | Flow-solver cap raised 32 → 64 iterations (authorized deviation) | none | Documented in `rl_env_v2.yaml`; measured max 41. |
| 8 | PPO checkpoints are saved mid-rollout (policy after the preceding complete rollout). | minor | Documented. |
| 9 | 16 live-UI tests fail without external checkpoints. | test hygiene | Now skip with explicit reason when the directory is missing. |

## 11. Undocumented assumptions (made explicit)

* Offered traffic is exogenous and independent of routing (no TCP feedback).
* A TE change takes effect at the same boundary at which it is requested
  (zero actuation latency). This is the single most important structural
  assumption for the sequentiality question (see `docs/MATHEMATICAL_FORMULATION.md` §6).
* At most one TE change per 5-minute interval network-wide.
* Candidate paths are fixed for the episode (no online path computation).
* FRR is instantaneous, free, and deterministic (cheapest live candidate).
* Telemetry is exact and instantaneous.

## 12. Missing metadata

* Per-episode holdout and continuity summaries (only aggregates committed).
* Exact library versions for evaluation runs are in manifests; the CUDA
  determinism settings used for training are not recorded.
* No record of how reward coefficients were chosen beyond the calibration
  table (`reward_calibration.csv`, which shows that 4 of 6 constructed
  calibration cases do not reproduce their published reward values although
  the preferred-action orderings all match).
