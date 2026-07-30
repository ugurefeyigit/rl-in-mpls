# V2 Seed-42 Learning Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, and report a reproducible seed-42 comparison between
MaskablePPO and a masked neural contextual bandit on the frozen V2 environment.

**Architecture:** Add a V2-only experiment package and CLI that share
environment construction, masks, seeds, run layout, checkpoints, metrics, and
evaluation. Keep V1 and the sixteen frozen definition files untouched.

**Tech Stack:** Python 3.13, NumPy, PyTorch, Gymnasium, Stable-Baselines3,
SB3-contrib MaskablePPO, pandas, pytest.

## Global Constraints

- V2 environment pin:
  `dca533b5c6fa9953307d01470c23cac512eb2961`.
- Training root in this task: 42 only.
- Evaluation seeds: 101, 102, 103, 104, 105.
- Holdout seeds 1001-1005 must never be accessed.
- Aggregate budget: exactly 400,000 transitions per learner.
- Checkpoint interval: exactly 50,000 aggregate transitions.
- Do not edit the sixteen `FROZEN_DEFINITION_PATHS`, V1 behavior/artifacts, or UI.

---

### Task 1: Governed Configuration and Run Lifecycle

**Files:**
- Create: `configs/experiments/learning_v2.yaml`
- Create: `mplssim/experiments/learning_common.py`
- Create: `tests/test_learning_v2.py`

**Interfaces:**
- Produces `load_learning_config() -> dict`.
- Produces `create_run_directory(path: Path) -> Path`.
- Produces `resolve_device(requested: str) -> torch.device`.
- Produces seed-policy validation, hardware inventory, hashing, and JSON helpers.

- [ ] Write failing tests for registry values, forbidden holdout seeds, new-run
      enforcement, device truthfulness, exact aggregate counts, and pin checks.
- [ ] Run `py -m pytest -q tests/test_learning_v2.py` and confirm failures are
      caused by missing interfaces.
- [ ] Implement the minimal configuration and lifecycle helpers.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Integrity Wrapper and Metric Recorder

**Files:**
- Modify: `mplssim/experiments/learning_common.py`
- Modify: `tests/test_learning_v2.py`

**Interfaces:**
- Produces `AuditedV2Env`, `SeedLedger`, and `MetricsWriter`.
- `AuditedV2Env.action_masks()` is the only mask source.
- `AuditedV2Env.step(action)` fails on invalid selection, mask disagreement,
  non-finite values, solver failure, or reward decomposition mismatch.

- [ ] Write failing tiny-environment tests for seed recording, invalid-action
      rejection, mask propagation, reward sums, and vector transition counts.
- [ ] Run the focused tests and observe the intended failures.
- [ ] Implement the wrapper and compressed JSONL/episode recorders.
- [ ] Re-run the focused tests.

### Task 3: Masked Contextual Bandit

**Files:**
- Create: `mplssim/experiments/masked_bandit.py`
- Modify: `tests/test_learning_v2.py`

**Interfaces:**
- Produces `MaskedContextualBandit.predict(obs, masks, deterministic)`.
- Produces `observe(obs, actions, masks, rewards)` and `update()`.
- Produces `save(path)` and `load(path, device)`.

- [ ] Write failing tests proving invalid-action exclusion, deterministic
      argmax, selected-action-only Huber loss, immediate-reward targets, and
      absence of next-state/bootstrap data.
- [ ] Run focused tests and confirm the expected failures.
- [ ] Implement the configured MLP, replay, exploration, supervised update, and
      checkpoint serialization.
- [ ] Re-run focused tests and refactor only while green.

### Task 4: Shared PPO/Bandit Training CLI and Checkpoints

**Files:**
- Create: `mplssim/experiments/trainers_v2.py`
- Create: `scripts/train_v2.py`
- Modify: `tests/test_learning_v2.py`

**Interfaces:**
- CLI supports `--algorithm maskable_ppo|masked_bandit`,
  `--device auto|cuda|cpu`, root seed, transitions, vectors, checkpoint
  interval, scenario, and a new run directory.
- Produces checkpoint payloads plus validated SHA-256 metadata sidecars.
- PPO uses governed hyperparameters and authoritative masks.

- [ ] Write failing tests for CLI/registry selection, guaranteed V2
      construction, PPO mask propagation, exact budget stopping, checkpoint
      hashes/reload, metadata mismatches, and directory reuse rejection.
- [ ] Run focused tests and confirm the failures.
- [ ] Implement learner adapters, callbacks, checkpoint helpers, and CLI.
- [ ] Re-run focused tests.

### Task 5: Shared V2 Evaluation and Existing Baselines

**Files:**
- Create: `mplssim/experiments/evaluation_v2.py`
- Create: `scripts/evaluate_v2.py`
- Modify: `tests/test_learning_v2.py`

**Interfaces:**
- Produces deterministic learner evaluation and V2 adapters for the existing
  `static`, `greedy`, and `cspf` controllers.
- Produces checkpoint selection, per-step/episode summaries, and integrity
  disqualification records.

- [ ] Write failing tests for deterministic learner inference, baseline action
      legality, equal horizons, checkpoint selection, exact-tie behavior, and
      holdout isolation.
- [ ] Run focused tests and observe intended failures.
- [ ] Implement evaluation, summaries, and selection.
- [ ] Re-run focused tests.

### Task 6: Verify and Publish Training Code

**Files:**
- All code/config/test/design/plan files from Tasks 1-5.

- [ ] Run `py -m pytest -q tests/test_learning_v2.py`.
- [ ] Run the existing relevant V2, reward, transition, compatibility, and
      baseline suites once.
- [ ] Run `py scripts/validate_env_v2.py --all` once and verify 12/12 gates.
- [ ] Confirm no frozen definition path changed and no model/result binary is
      staged.
- [ ] Commit and push implementation; record the full SHA.
- [ ] Create a clean worktree at that SHA for experiments and re-run the freeze
      gate there.

### Task 7: Benchmark and Smoke

**Files:**
- Create runtime artifacts only below ignored `runs/v2/`.

- [ ] Benchmark 8, 12, and 16 DummyVecEnv workers for both learners; compare
      CPU/CUDA only when CUDA is actually available.
- [ ] Select one eligible stable worker count from measured combined throughput.
- [ ] Run disposable smoke training for PPO and bandit through save, reload,
      deterministic evaluation, and clean termination.
- [ ] Preserve benchmark and smoke manifests.

### Task 8: Meaningful Seed-42 Training

**Files:**
- Create runtime artifacts only below ignored `runs/v2/`.

- [ ] Train MaskablePPO for exactly 400,000 aggregate transitions with
      checkpoints every 50,000.
- [ ] Train the masked bandit with the same root, scenario, vector policy,
      budget, and checkpoint interval.
- [ ] Preserve any failed run and resume only after a concrete tooling fix.

### Task 9: Checkpoint Selection and Comparison

**Files:**
- Create runtime evaluation artifacts below ignored `runs/v2/`.

- [ ] Evaluate all 16 learner checkpoints on seven scenarios and seeds 101-105.
- [ ] Select each valid learner checkpoint by the preregistered return rule.
- [ ] Evaluate selected learners and static, greedy, and CSPF on the same matrix.
- [ ] Produce compact comparison CSV/JSON and learning-curve data.

### Task 10: Report and Handoff

**Files:**
- Create: `results/v2_seed42/PILOT_REPORT.md`
- Create: `results/v2_seed42/manifest.json`
- Create: `results/v2_seed42/comparison.csv`
- Create: `results/v2_seed42/comparison.json`
- Create: `NEXT_STAGE_HANDOFF.md`

- [ ] Write the scientific answers, limitations, churn/safety comparison,
      timings, and remaining-root runtime estimate from actual artifacts.
- [ ] Record commands, hashes, seeds, devices, checkpoints, test/gate results,
      artifact paths, and explicit holdout non-access.
- [ ] Run final report/manifest consistency checks and the freeze gate.
- [ ] Confirm only small intended artifacts are staged, then commit and push.

