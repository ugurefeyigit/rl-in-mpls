# Preserved baselines - what is frozen and how to recover it

Two things are frozen and immutable in this repository: the **Version 1**
baseline described below, and the **governed V2 study**, which is closed.
Neither is edited in place.

## Version 1 baseline

**Frozen on branch point `presentation-hardening` ← `main` @ `10e6d59`,
tagged `v1-original-results`.**

## What Version 1 is

The system as originally built and evaluated:

- Pretrained model `models/ppo_te/best_model.zip` (MaskablePPO, 400,000 steps,
  training seed 42, trained on the `random_day` scenario). SHA-256 recorded in
  [results/v1_manifest.json](results/v1_manifest.json).
- Published paired evaluation: 7 scenarios × 5 seeds (101–105) × 5 algorithms —
  `results/eval_summary.csv`, `results/eval_stats.csv`, `results/eval_summary.json`,
  plus ablation/safety variants (`abl_*`, `nosafety_*`) and 28 figures in
  `results/figures/`.
- The original single-interface dashboard, configs, technical report
  (`docs/REPORT.md`) and 32-test suite.

## Preservation rules for all later work

1. **Nothing under the tag is edited in place.** The published CSV/JSON result
   files and figures are never overwritten; any artifact regenerated after
   bug fixes lands under a new prefix (`results/corrected_runtime_*`).
2. **The pretrained model files are immutable** (`models/ppo_te/*`). No new
   training is run as part of the presentation-hardening work.
3. Runtime fixes may change *live* behavior slightly (e.g. the corrected
   protected-bandwidth check); the report distinguishes published V1 numbers
   from the hardened runtime.

## How to recover Version 1 exactly

```bash
git checkout v1-original-results
pip install -r requirements.txt   # versions recorded in results/v1_manifest.json
python -m uvicorn server.main:app --port 8000
```

The manifest records the commit SHA, Python/package versions, model and
result-file SHA-256 hashes, config hashes, and all training/evaluation seeds
needed to reproduce or audit the published numbers.


---

## Version 2 - the governed study, closed

**Frozen on branch `feat/rl-environment-v2`, sealed at
`d7d2b3f8623ec26ef802dcc07b768978a81c2e19`.**

### What Version 2 is

A preregistered comparison of MaskablePPO against a masked contextual bandit on
`MplsTeEnvV2` (observation 604, actions 69), plus the three repository baselines.
Its final holdout ran exactly once on untouched seeds 1001-1005, over 315
episodes. The result and its limits are in
[docs/TECHNICAL_DEFENSE.md](docs/TECHNICAL_DEFENSE.md); the independent
reconciliation is in [docs/V2_EVIDENCE_AUDIT.md](docs/V2_EVIDENCE_AUDIT.md).

### Scientific identity

| What | SHA |
|---|---|
| Closeout | `d7d2b3f8623ec26ef802dcc07b768978a81c2e19` |
| Final-holdout evaluation source | `f7ed0f407c50c5472ecff89f977bc656439a8c49` |
| Seed-42 training source | `ca64b62fe29e45ab61aa86d642799aec5a4c25e1` |
| Continuation training source | `6a8a4068b98bf9a71dead6e547595b4bbd755689` |
| Signed-off environment pin | `dca533b5c6fa9953307d01470c23cac512eb2961` |
| Approved ancestor | `859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac` |

Compact evidence is committed under `results/v2_final_holdout/`,
`results/v2_three_root_continuity/` and `results/v2_seed42/`. Checkpoints, full
step traces, replay buffers and TensorBoard data are **not** in Git; they live in
the preserved experiment worktrees named in each manifest.

### Preservation rules for all later work

1. **The V2 study is closed.** No training, tuning, checkpoint loading for
   evaluation, reselection, sweep, or holdout re-run - ever. Its holdout was
   consumed by a single evaluation and cannot be reused.
2. **Frozen definitions do not move.** Reward, observation, action, topology,
   scenario, seed, mask, horizon, baseline and evaluation semantics are pinned;
   the freeze/pin tests fail if one changes under a trained checkpoint.
3. **Artifacts are read-only inputs.** Manifests, compact tables, checkpoints,
   sidecars and raw artifacts are never rewritten. `mplssim/evidence/` is the one
   component that reads them, and it never writes.
4. **Preserved failures stay preserved.** The invalidated seed-42 PPO run, the
   superseded root-314159 runs and the failed launches remain on disk and are
   disclosed as three distinct statuses.
5. **New learners are V3.** Any A2C, recurrent learner, planner, new controller,
   reward change, environment change or additional evaluation requires a separate
   preregistration with new development seeds and untouched evaluation seeds. See
   [docs/V3_RESEARCH_BACKLOG.md](docs/V3_RESEARCH_BACKLOG.md) - everything in it
   is unapproved and unevaluated.

### How to inspect Version 2

```bash
git checkout d7d2b3f8623ec26ef802dcc07b768978a81c2e19
python -m uvicorn server.main:app --port 8000
```

Open `http://127.0.0.1:8000/study`. Nothing on that page runs a controller.
