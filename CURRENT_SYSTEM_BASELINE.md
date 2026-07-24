# Version 1 baseline — what is preserved and how to recover it

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
