# Post-Study Productization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a read-only evidence, replay and presentation layer over the closed
governed V2 study, so the frozen result can be inspected, demonstrated and defended
without ever touching a learner.

**Architecture:** A new `mplssim/evidence/` package is the only component that reads the
frozen compact artifacts. It validates schemas and study identity, fails closed, and
centralizes every scientific calculation. A thin `server/evidence_api.py` router exposes
it under `/api/v2/*` and mounts into the existing FastAPI app without altering any
live-session endpoint. A third frontend surface at `/study` consumes that API. Nothing
imports a learner, loads a checkpoint, or writes into `results/` or `runs/`.

**Tech Stack:** Python 3.13, FastAPI, pandas (read-only), pytest; build-free ES modules
with vendored ECharts (no npm, no CDN) for the frontend.

## Global Constraints

- The governed V2 study is **closed**. No training, tuning, checkpoint loading for
  evaluation, checkpoint reselection, sweep, or holdout re-run — ever.
- Frozen artifacts under `results/` are **immutable read-only inputs**. Never write,
  regenerate, reformat or rewrite them.
- `results/environment_v2_validation/manifest.json` is a protected unstaged file. Expected
  SHA-256 `5680610c95cec9551cd22fad2b365b1023485f59edb87d3e568bc908eda086c0`. Never stage,
  commit, alter, or revert it. A mismatch is a stop condition.
- Development/continuity evidence and final-holdout evidence must never be visually or
  statistically blurred together.
- Frozen conclusions to represent exactly: holdout ran **once**; **315** episodes (**35**
  per learner checkpoint or baseline); bandit **18.221**; PPO **9.036**; advantage
  **9.185**; greedy **-2.327**; bandit won **3/3** roots and **6/7** scenarios; PPO led
  `deceptive_local_optimum` by **1.107**; all safety and integrity checks passed; both
  learners ~**2.148** reroutes/hour; bandit had fewer reversals and flaps but moved more
  bandwidth than PPO.
- The evidence gives **no positive support for a need for temporal planning in this
  formulation**. It is **not** evidence that planning is generally irrelevant to MPLS or
  traffic engineering. Both halves must always appear together.
- No holdout result may be used for further V2 selection, tuning or redesign.
- Never commit checkpoints, model binaries, replay buffers, TensorBoard data, compressed
  step logs, raw episode logs, or large datasets.
- Any A2C, recurrent learner, planner, new controller, reward change, environment change,
  or additional evaluation belongs to a separately preregistered **V3** study and must be
  labelled unapproved and unevaluated.
- Frontend stays build-free: vendored assets only, no CDN, no npm. The presentation test
  `tests/test_presentation.py::test_pages_reference_only_vendored_scripts` enforces this.

## Two grains that look alike — never conflate

`aggregate_metrics.csv → noop_frequency_mean` is the **mean over episodes** of each
episode's no-op frequency (bandit 82.10%, PPO 82.10%). `action_distribution.csv` action 0
is the **step-pooled** share over 3,300 steps (bandit 87.09%, PPO 87.31%). Both are
correct. `FINAL_HOLDOUT_REPORT.md` quotes the pooled figure. Always label the grain.

Likewise `manifest.json → runtime.total_wall_seconds` (152.093 s) is the **whole-runner**
wall time including the three baselines and setup; the six per-checkpoint
`evaluation_wall_seconds` sum to 115.213 s. Never present one as the other.

## File Structure

**Created**
- `mplssim/evidence/__init__.py` — public surface re-exports.
- `mplssim/evidence/identity.py` — frozen study constants. No I/O, no logic.
- `mplssim/evidence/errors.py` — `EvidenceError` and subclasses.
- `mplssim/evidence/loader.py` — schema-validated read-only readers for each artifact.
- `mplssim/evidence/claims.py` — every scientific calculation, in one place.
- `mplssim/evidence/replay.py` — recorded-trace index and reader.
- `server/evidence_api.py` — `/api/v2/*` FastAPI router.
- `frontend/study.html`, `frontend/js/study.js`, `frontend/css/study.css` — the surface.
- `tests/test_evidence_loader.py`, `tests/test_evidence_claims.py`,
  `tests/test_evidence_replay.py`, `tests/test_evidence_api.py`,
  `tests/test_study_ui.py` — the test cycles.
- `docs/V2_EVIDENCE_AUDIT.md`, `docs/TECHNICAL_DEFENSE.md`,
  `docs/RELEASE_CHECKLIST.md`, `docs/V3_RESEARCH_BACKLOG.md`.

**Modified**
- `server/main.py` — mount the router, add the `/study` route. Nothing else.
- `README.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/PRESENTATION_MODE.md`,
  `docs/DEMO_SCRIPT.md`, `docs/UI_ACCEPTANCE_TESTS.md`, `CURRENT_SYSTEM_BASELINE.md`.

**Never touched:** anything under `results/`, `runs/`, `models/`, `.worktrees/*` other than
this one, `mplssim/rl/`, `mplssim/sim/`, `configs/`, and every frozen-definition test.

---

### Task 1: Frozen study identity and error types

**Files:**
- Create: `mplssim/evidence/identity.py`
- Create: `mplssim/evidence/errors.py`
- Create: `mplssim/evidence/__init__.py`
- Test: `tests/test_evidence_loader.py`

**Interfaces:**
- Produces: `EVALUATION_SOURCE_SHA`, `SEED42_SOURCE_SHA`, `CONTINUATION_SOURCE_SHA`,
  `SIGNED_OFF_ENV_SHA`, `APPROVED_ANCESTOR_SHA` (all `str`); `TRAINING_ROOTS: tuple[int, ...]`;
  `SCENARIOS: tuple[str, ...]`; `HOLDOUT_SEEDS`, `CONTINUITY_SEEDS: tuple[int, ...]`;
  `EPISODES_PER_POLICY: int`; `TOTAL_HOLDOUT_EPISODES: int`; `REWARD_COMPONENTS: tuple[str, ...]`;
  `ACTION_COUNT: int`; `OBSERVATION_DIM: int`; `LEARNER_ALGORITHMS`, `BASELINE_ALGORITHMS`.
- Produces: `EvidenceError(Exception)`, `ArtifactMissingError`, `SchemaError`,
  `IdentityError`, `IntegrityError` — all subclasses of `EvidenceError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evidence_loader.py
from mplssim.evidence import identity


def test_frozen_identity_matches_the_closed_study():
    assert identity.EVALUATION_SOURCE_SHA == "f7ed0f407c50c5472ecff89f977bc656439a8c49"
    assert identity.SEED42_SOURCE_SHA == "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
    assert identity.CONTINUATION_SOURCE_SHA == "6a8a4068b98bf9a71dead6e547595b4bbd755689"
    assert identity.SIGNED_OFF_ENV_SHA == "dca533b5c6fa9953307d01470c23cac512eb2961"
    assert identity.TRAINING_ROOTS == (42, 314159, 271828)
    assert identity.HOLDOUT_SEEDS == (1001, 1002, 1003, 1004, 1005)
    assert identity.CONTINUITY_SEEDS == (101, 102, 103, 104, 105)
    assert set(identity.HOLDOUT_SEEDS).isdisjoint(identity.CONTINUITY_SEEDS)
    assert identity.EPISODES_PER_POLICY == 35
    assert identity.TOTAL_HOLDOUT_EPISODES == 315
    assert len(identity.SCENARIOS) == 7
    assert len(identity.REWARD_COMPONENTS) == 12
    assert identity.ACTION_COUNT == 69
    assert identity.OBSERVATION_DIM == 604
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evidence_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mplssim.evidence'`

- [ ] **Step 3: Write the implementation**

`mplssim/evidence/errors.py`:

```python
"""Failure modes of the read-only evidence layer. Every one is fail-closed."""


class EvidenceError(Exception):
    """Base class: the frozen evidence could not be served as promised."""


class ArtifactMissingError(EvidenceError):
    """A required frozen artifact is absent or unreadable."""


class SchemaError(EvidenceError):
    """An artifact is present but does not carry the columns/fields promised."""


class IdentityError(EvidenceError):
    """An artifact's study identity (SHA, root, seed, scenario) is not the frozen one."""


class IntegrityError(EvidenceError):
    """An artifact's own integrity flags or counts do not hold."""
```

`mplssim/evidence/identity.py`:

```python
"""Frozen identity of the closed governed V2 study.

These are assertions about evidence that already exists and can never change.
Nothing here is configuration: if a value disagrees with an artifact, the artifact
is not the study this repository closed, and the loader must fail closed.
"""

from __future__ import annotations

# --- source identities -------------------------------------------------------
EVALUATION_SOURCE_SHA = "f7ed0f407c50c5472ecff89f977bc656439a8c49"
SEED42_SOURCE_SHA = "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
CONTINUATION_SOURCE_SHA = "6a8a4068b98bf9a71dead6e547595b4bbd755689"
SIGNED_OFF_ENV_SHA = "dca533b5c6fa9953307d01470c23cac512eb2961"
APPROVED_ANCESTOR_SHA = "859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac"
CLOSEOUT_SHA = "d7d2b3f8623ec26ef802dcc07b768978a81c2e19"

# --- experimental design -----------------------------------------------------
TRAINING_ROOTS: tuple[int, ...] = (42, 314159, 271828)
HOLDOUT_SEEDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)
CONTINUITY_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
SCENARIOS: tuple[str, ...] = (
    "full_day", "evening_peak", "flash_crowd", "link_failure",
    "deceptive_local_optimum", "ood_double_failure", "overload_stress",
)
LEARNER_ALGORITHMS: tuple[str, ...] = ("maskable_ppo", "masked_bandit")
BASELINE_ALGORITHMS: tuple[str, ...] = ("static", "greedy", "cspf")

EPISODES_PER_POLICY = 35
TOTAL_HOLDOUT_EPISODES = 315
POLICY_COUNT = 9

# --- environment -------------------------------------------------------------
ENVIRONMENT = "MplsTeEnvV2"
OBSERVATION_DIM = 604
ACTION_COUNT = 69

REWARD_COMPONENTS: tuple[str, ...] = (
    "delivery", "protected_disconnect", "unprotected_disconnect", "sla_severity",
    "max_util", "overload", "potential", "move_fixed", "move_volume",
    "move_divergence", "reversal", "invalid",
)
```

`mplssim/evidence/__init__.py`:

```python
"""Read-only access to the frozen governed V2 evidence.

This package never trains, never loads a checkpoint for evaluation, and never
writes into results/ or runs/. It reads committed compact artifacts, validates
them against the frozen study identity, and fails closed.
"""

from mplssim.evidence import identity
from mplssim.evidence.errors import (
    ArtifactMissingError, EvidenceError, IdentityError, IntegrityError, SchemaError,
)

__all__ = [
    "identity", "EvidenceError", "ArtifactMissingError", "SchemaError",
    "IdentityError", "IntegrityError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evidence_loader.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add mplssim/evidence tests/test_evidence_loader.py
git commit -m "Pin the frozen V2 study identity for read-only consumers"
```

---

### Task 2: Fail-closed artifact loader

**Files:**
- Create: `mplssim/evidence/loader.py`
- Test: `tests/test_evidence_loader.py` (extend)

**Interfaces:**
- Consumes: `mplssim.evidence.identity`, `mplssim.evidence.errors` from Task 1.
- Produces:
  - `EvidenceRoot(results_dir: Path)` dataclass with `.final_holdout`, `.continuity`,
    `.seed42` properties returning `Path`.
  - `default_root() -> EvidenceRoot` — repository `results/`, honouring `$V2_EVIDENCE_ROOT`.
  - `load_table(path: Path, required: Sequence[str]) -> pandas.DataFrame` — raises
    `ArtifactMissingError` / `SchemaError`.
  - `load_json(path: Path, required: Sequence[str]) -> dict`.
  - `FinalHoldout` dataclass with fields `aggregate`, `per_root`, `scenario`,
    `reward_components`, `actions`, `integrity`, `provenance` (all `DataFrame`) and
    `manifest` (`dict`); classmethod `load(root: EvidenceRoot) -> FinalHoldout`.
  - `Continuity` dataclass with `aggregate`, `per_root`, `scenario`, `reward_components`,
    `actions`, `integrity`, `learning_curves`, `checkpoint_selection`,
    `training_summary`, `training_integrity`, `manifest`; classmethod `load`.
  - `Seed42` dataclass with `comparison`, `learning_curve`, `ppo_selection`,
    `bandit_selection`, `manifest`; classmethod `load`.
  - Each `load` validates identity and integrity and raises on any violation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evidence_loader.py (append)
import json
import shutil
from pathlib import Path

import pytest

from mplssim.evidence import errors, loader


def test_final_holdout_loads_the_committed_evidence():
    fh = loader.FinalHoldout.load(loader.default_root())
    assert len(fh.per_root) == 9
    assert int(fh.per_root["episodes"].sum()) == 315
    assert len(fh.scenario) == 63
    assert len(fh.provenance) == 6
    assert len(fh.actions) == 9 * 69
    assert fh.manifest["status"]


def test_missing_artifact_fails_closed(tmp_path: Path):
    root = loader.EvidenceRoot(tmp_path)
    with pytest.raises(errors.ArtifactMissingError):
        loader.FinalHoldout.load(root)


def test_missing_column_fails_closed(tmp_path: Path):
    src = loader.default_root().final_holdout
    dst = tmp_path / "v2_final_holdout"
    shutil.copytree(src, dst)
    csv = dst / "per_root_metrics.csv"
    text = csv.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    drop = header.index("operational_return_mean")
    csv.write_text("\n".join(
        ",".join(c for i, c in enumerate(line.split(",")) if i != drop) for line in text
    ), encoding="utf-8")
    with pytest.raises(errors.SchemaError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_wrong_episode_count_fails_closed(tmp_path: Path):
    src = loader.default_root().final_holdout
    dst = tmp_path / "v2_final_holdout"
    shutil.copytree(src, dst)
    csv = dst / "per_root_metrics.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    csv.write_text("\n".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(errors.IntegrityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_foreign_evaluation_sha_fails_closed(tmp_path: Path):
    src = loader.default_root().final_holdout
    dst = tmp_path / "v2_final_holdout"
    shutil.copytree(src, dst)
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    man["evaluation_source_sha"] = "0" * 40
    (dst / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(errors.IdentityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_failed_integrity_flag_fails_closed(tmp_path: Path):
    src = loader.default_root().final_holdout
    dst = tmp_path / "v2_final_holdout"
    shutil.copytree(src, dst)
    csv = dst / "evaluation_integrity.csv"
    csv.write_text(csv.read_text(encoding="utf-8").replace("True\n", "False\n", 1),
                   encoding="utf-8")
    with pytest.raises(errors.IntegrityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_loader_never_writes_into_results(tmp_path, monkeypatch):
    """Any attempt to open a results/ path for writing is a release blocker."""
    import builtins
    real_open = builtins.open
    results = loader.default_root().results_dir.resolve()

    def guard(file, mode="r", *a, **kw):
        p = Path(file).resolve() if not hasattr(file, "read") else None
        if p is not None and any(m in mode for m in "wxa+") and results in p.parents:
            raise AssertionError(f"write attempted inside results/: {p}")
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guard)
    loader.FinalHoldout.load(loader.default_root())
    loader.Continuity.load(loader.default_root())
    loader.Seed42.load(loader.default_root())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evidence_loader.py -q`
Expected: FAIL — `AttributeError: module 'mplssim.evidence.loader' has no attribute ...`

- [ ] **Step 3: Write the implementation**

Implement `mplssim/evidence/loader.py` exactly as specified in the Interfaces block.
Rules the implementation must follow:

- Read with `pandas.read_csv(path)` and `json.loads(path.read_text(encoding="utf-8"))`.
  Never open a frozen path in a writing mode.
- `load_table` raises `ArtifactMissingError` when the file does not exist, and
  `SchemaError` listing every missing column when one is absent.
- `FinalHoldout.load` validates, in order: all seven files present with required
  columns; `manifest["evaluation_source_sha"] == identity.EVALUATION_SOURCE_SHA`;
  manifest scenarios == `set(identity.SCENARIOS)`; manifest seeds ==
  `set(identity.HOLDOUT_SEEDS)`; `per_root` has 9 unique `policy_id` each with
  `episodes == 35` summing to 315; `scenario` has 63 rows with 7 unique scenarios all
  drawn from `identity.SCENARIOS`; `actions` has `9 * 69` rows covering actions 0..68 for
  every policy; `integrity["all_checks_passed"]` all true and every `*_total` counter
  zero; `provenance` has 6 rows whose `evaluation_source_sha` all equal
  `identity.EVALUATION_SOURCE_SHA` and whose `training_source_sha` values are a subset of
  `{SEED42_SOURCE_SHA, CONTINUATION_SOURCE_SHA}`.
- Count/coverage violations raise `IntegrityError`; SHA/seed/scenario violations raise
  `IdentityError`.
- `Continuity.load` validates the continuity artifacts and asserts that no row anywhere
  references a holdout seed — continuity evidence must be free of 1001–1005.
- `default_root()` resolves `Path(__file__).resolve().parents[2] / "results"` unless
  `$V2_EVIDENCE_ROOT` is set.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evidence_loader.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add mplssim/evidence/loader.py tests/test_evidence_loader.py
git commit -m "Load the frozen V2 evidence read-only and fail closed"
```

---

### Task 3: Centralized scientific claims

**Files:**
- Create: `mplssim/evidence/claims.py`
- Test: `tests/test_evidence_claims.py`

**Interfaces:**
- Consumes: `loader.FinalHoldout`, `loader.Continuity`, `loader.Seed42`.
- Produces:
  - `root_aggregate(per_root: DataFrame, algorithm: str) -> dict` — unweighted mean over
    root means plus `root_mean_std` (ddof=1) and `root_count`.
  - `learner_comparison(fh: FinalHoldout) -> dict` with keys `bandit_return`,
    `ppo_return`, `advantage`, `roots` (list of per-root dicts), `roots_won`.
  - `scenario_comparison(fh: FinalHoldout) -> list[dict]` — one entry per scenario with
    `scenario`, `bandit`, `ppo`, `advantage`, `winner`; root-averaged, never episode-pooled.
  - `reward_reconciliation(fh: FinalHoldout) -> dict` with `rows` (per policy: component
    means, `sum`, `operational_return_mean`, `residual`) and `max_residual`.
  - `noop_shares(fh: FinalHoldout) -> dict` returning **both** grains explicitly under
    keys `pooled_step_share` and `episode_mean_share`.
  - `runtime_summary(fh: FinalHoldout) -> dict` returning `total_runner_wall_seconds`
    and `checkpoint_wall_seconds_sum` as **separate, differently named** fields.
  - `CONCLUSIONS: tuple[str, ...]` — the frozen prose statements, verbatim.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evidence_claims.py
import math

import pytest

from mplssim.evidence import claims, loader

ROOT = loader.default_root()


@pytest.fixture(scope="module")
def fh():
    return loader.FinalHoldout.load(ROOT)


def test_headline_numbers_reproduce_from_frozen_evidence(fh):
    c = claims.learner_comparison(fh)
    assert round(c["bandit_return"], 3) == 18.221
    assert round(c["ppo_return"], 3) == 9.036
    assert round(c["advantage"], 3) == 9.185
    assert c["roots_won"] == 3
    assert len(c["roots"]) == 3


def test_aggregate_is_root_aware_not_episode_pooled(fh):
    """The aggregate must equal the mean of the three ROOT means. Pooling the 105
    episodes directly would be a different number and a scientific error."""
    for algo, expected in (("masked_bandit", 18.220918), ("maskable_ppo", 9.035842)):
        rows = fh.per_root[(fh.per_root.algorithm == algo)
                           & (fh.per_root.training_root.astype(str) != "baseline")]
        assert len(rows) == 3
        agg = claims.root_aggregate(fh.per_root, algo)
        assert agg["operational_return_mean"] == pytest.approx(
            rows.operational_return_mean.mean(), abs=1e-12)
        assert agg["operational_return_mean"] == pytest.approx(expected, abs=1e-6)
        assert agg["root_mean_std"] == pytest.approx(
            rows.operational_return_mean.std(ddof=1), abs=1e-12)
        assert agg["root_count"] == 3


def test_scenario_comparison_is_six_of_seven_with_the_one_ppo_win(fh):
    rows = claims.scenario_comparison(fh)
    assert len(rows) == 7
    assert sum(r["winner"] == "masked_bandit" for r in rows) == 6
    ppo_wins = [r for r in rows if r["winner"] == "maskable_ppo"]
    assert [r["scenario"] for r in ppo_wins] == ["deceptive_local_optimum"]
    assert round(-ppo_wins[0]["advantage"], 3) == 1.107
    assert round(max(r["advantage"] for r in rows), 3) == 20.183


def test_reward_components_sum_exactly(fh):
    rec = claims.reward_reconciliation(fh)
    assert len(rec["rows"]) == 9
    assert all(len(r["components"]) == 12 for r in rec["rows"])
    assert rec["max_residual"] < 1e-9
    for r in rec["rows"]:
        assert r["sum"] == pytest.approx(r["operational_return_mean"], abs=1e-9)


def test_noop_shares_expose_both_grains_separately(fh):
    n = claims.noop_shares(fh)
    assert round(n["pooled_step_share"]["masked_bandit"] * 100, 2) == 87.09
    assert round(n["pooled_step_share"]["maskable_ppo"] * 100, 2) == 87.31
    assert round(n["episode_mean_share"]["masked_bandit"] * 100, 2) == 82.10
    assert n["pooled_step_share"] != n["episode_mean_share"]


def test_runtime_keeps_runner_total_separate_from_checkpoint_sum(fh):
    r = claims.runtime_summary(fh)
    assert round(r["total_runner_wall_seconds"], 3) == 152.093
    assert round(r["checkpoint_wall_seconds_sum"], 3) == 115.213
    assert r["total_runner_wall_seconds"] != r["checkpoint_wall_seconds_sum"]


def test_conclusions_state_both_halves_of_the_planning_claim():
    joined = " ".join(claims.CONCLUSIONS).lower()
    assert "does not" in joined and "temporal planning" in joined
    assert "not" in joined and "generally" in joined
    assert any("not" in c.lower() and "general" in c.lower() for c in claims.CONCLUSIONS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evidence_claims.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mplssim.evidence.claims'`

- [ ] **Step 3: Write the implementation**

Implement `mplssim/evidence/claims.py` per the Interfaces block. `CONCLUSIONS` must
contain, verbatim, at minimum:

```python
CONCLUSIONS: tuple[str, ...] = (
    "The final holdout ran exactly once, over 315 episodes: 35 per learner "
    "checkpoint or baseline.",
    "The masked contextual bandit reached a mean holdout return of 18.221 against "
    "MaskablePPO's 9.036, an advantage of 9.185. Greedy was the strongest "
    "repository baseline at -2.327.",
    "The bandit won all three training roots and six of seven scenarios. PPO "
    "retained a 1.107-point lead in deceptive_local_optimum.",
    "All safety and integrity checks passed. Bandit and PPO both averaged about "
    "2.148 reroutes/hour; the bandit had fewer reversals and flaps but moved more "
    "bandwidth than PPO.",
    "The frozen evidence does not positively support a need for temporal planning "
    "in this formulation: the explicitly myopic learner remained stronger.",
    "This is not evidence that planning is generally irrelevant to MPLS or traffic "
    "engineering. It is a result about these frozen learners, scenarios and "
    "observation design only.",
    "No training, tuning, sweep, reselection, redesign, retry, or policy debugging "
    "used holdout results.",
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evidence_claims.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add mplssim/evidence/claims.py tests/test_evidence_claims.py
git commit -m "Centralize the V2 scientific claims on one tested surface"
```

---

### Task 4: Recorded-replay index and reader

**Files:**
- Create: `mplssim/evidence/replay.py`
- Test: `tests/test_evidence_replay.py`

**Interfaces:**
- Produces:
  - `full_artifact_root() -> Path | None` — `$V2_FULL_ARTIFACTS` if set, else the path
    recorded in the final-holdout manifest, else `None`. Returns `None` when the path
    does not exist; never raises for absence.
  - `replay_available() -> bool`.
  - `episode_index(root: EvidenceRoot) -> list[dict]` — from the committed compact
    evidence only, so the index works even with no full artifacts: one entry per
    `(policy_id, algorithm, training_root, scenario, seed)` with `available: bool`.
  - `load_episode(policy_id: str, scenario: str, seed: int) -> dict` with keys
    `provenance` (policy, algorithm, root, scenario, seed, source SHA, artifact path,
    `kind: "recorded_replay"`) and `steps` (list of per-step dicts).
  - Raises `ArtifactMissingError` when replay is unavailable, `IdentityError` for a seed
    outside `identity.HOLDOUT_SEEDS` or a scenario outside `identity.SCENARIOS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evidence_replay.py
import pytest

from mplssim.evidence import errors, identity, loader, replay

ROOT = loader.default_root()


def test_index_is_complete_and_works_without_full_artifacts():
    idx = replay.episode_index(ROOT)
    assert len(idx) == 315
    assert {e["seed"] for e in idx} == set(identity.HOLDOUT_SEEDS)
    assert {e["scenario"] for e in idx} == set(identity.SCENARIOS)
    assert len({e["policy_id"] for e in idx}) == 9


def test_rejects_seeds_that_are_not_holdout_seeds():
    with pytest.raises(errors.IdentityError):
        replay.load_episode("root42_masked_bandit", "link_failure", 101)


def test_rejects_unknown_scenarios():
    with pytest.raises(errors.IdentityError):
        replay.load_episode("root42_masked_bandit", "not_a_scenario", 1001)


@pytest.mark.skipif(not replay.replay_available(),
                    reason="full holdout artifacts not configured")
def test_recorded_episode_is_labelled_and_reproduces_its_own_return():
    ep = replay.load_episode("root42_masked_bandit", "link_failure", 1001)
    assert ep["provenance"]["kind"] == "recorded_replay"
    assert ep["provenance"]["seed"] == 1001
    assert ep["provenance"]["evaluation_source_sha"] == identity.EVALUATION_SOURCE_SHA
    assert len(ep["steps"]) == 288
    assert ep["steps"][0]["step_index"] == 0
    total = sum(s["reward"] for s in ep["steps"])
    assert total == pytest.approx(ep["provenance"]["operational_return"], rel=1e-9)


@pytest.mark.skipif(not replay.replay_available(),
                    reason="full holdout artifacts not configured")
def test_replay_opens_nothing_for_writing(monkeypatch):
    import builtins
    from pathlib import Path
    real_open = builtins.open

    def guard(file, mode="r", *a, **kw):
        if not hasattr(file, "read") and any(m in mode for m in "wxa+"):
            raise AssertionError(f"replay attempted a write: {file}")
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guard)
    replay.load_episode("root42_masked_bandit", "link_failure", 1001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evidence_replay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mplssim.evidence.replay'`

- [ ] **Step 3: Write the implementation**

Read step files with `gzip.open(path, "rt", encoding="utf-8")` and `csv.DictReader`.
Learner traces live at
`<root>/learners/<policy_id>/steps/<algorithm>_<scenario>_seed<seed>_steps.csv.gz`;
baseline traces at `<root>/baselines/<algorithm>/steps/...`. Coerce numeric fields;
keep `action`, `action_type`, `action_accepted`, `reward`, all `rc_*` components,
`max_util`, `mean_util`, `mean_delay_ms`, `loss_ratio`, `delivered_ratio`,
`sla_violations`, `congested_links`, `n_failed_links`, `moved_mbps`, `hour`, `t_min`.
Never write, never copy the artifacts into the repository.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evidence_replay.py -q`
Expected: PASS (5 passed, or 3 passed / 2 skipped without full artifacts)

- [ ] **Step 5: Commit**

```bash
git add mplssim/evidence/replay.py tests/test_evidence_replay.py
git commit -m "Serve recorded V2 holdout episodes without re-evaluating anything"
```

---

### Task 5: `/api/v2/*` read-only evidence router

**Files:**
- Create: `server/evidence_api.py`
- Modify: `server/main.py` — import the router, `app.include_router(...)`, add `/study`.
- Test: `tests/test_evidence_api.py`

**Interfaces:**
- Produces `router: APIRouter` with prefix `/api/v2`, exposing:
  `GET /study`, `GET /final-holdout`, `GET /final-holdout/scenarios`,
  `GET /final-holdout/reward-components`, `GET /final-holdout/actions`,
  `GET /final-holdout/integrity`, `GET /final-holdout/provenance`,
  `GET /development/continuity`, `GET /development/seed42`, `GET /disclosures`,
  `GET /replay/index`, `GET /replay/episode`.
- Every payload carries `"stage"`: `"final_holdout"` or `"development"`, plus
  `"source_sha"` and `"artifact_path"`.
- `EvidenceError` maps to HTTP 503 with `{"detail": {"error": <class>, "message": str}}`;
  `IdentityError` from replay maps to HTTP 400.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evidence_api.py
import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_study_endpoint_states_the_closed_status_and_both_halves(client):
    d = client.get("/api/v2/study").json()
    assert d["status"] == "closed"
    assert d["environment"] == "MplsTeEnvV2"
    joined = " ".join(d["conclusions"]).lower()
    assert "does not positively support" in joined
    assert "not evidence that planning is generally irrelevant" in joined


def test_final_holdout_headline_numbers(client):
    d = client.get("/api/v2/final-holdout").json()
    assert d["stage"] == "final_holdout"
    assert d["episodes"]["total"] == 315
    assert d["episodes"]["per_policy"] == 35
    assert round(d["comparison"]["bandit_return"], 3) == 18.221
    assert round(d["comparison"]["ppo_return"], 3) == 9.036
    assert round(d["comparison"]["advantage"], 3) == 9.185
    assert d["comparison"]["roots_won"] == 3
    assert round(next(r["operational_return_mean"] for r in d["aggregate"]
                      if r["algorithm"] == "greedy"), 3) == -2.327


def test_scenarios_endpoint_reports_the_one_ppo_win(client):
    rows = client.get("/api/v2/final-holdout/scenarios").json()["scenarios"]
    assert len(rows) == 7
    ppo = [r for r in rows if r["winner"] == "maskable_ppo"]
    assert [r["scenario"] for r in ppo] == ["deceptive_local_optimum"]


def test_development_stage_is_labelled_and_carries_learning_curves(client):
    d = client.get("/api/v2/development/continuity").json()
    assert d["stage"] == "development"
    assert d["learning_curves"]
    assert d["source_sha"]


def test_development_and_holdout_are_separate_endpoints(client):
    fh = client.get("/api/v2/final-holdout").json()
    dev = client.get("/api/v2/development/continuity").json()
    assert fh["stage"] != dev["stage"]
    assert "learning_curves" not in fh


def test_disclosures_cover_invalidated_and_superseded_runs(client):
    d = client.get("/api/v2/disclosures").json()
    kinds = {x["kind"] for x in d["disclosures"]}
    assert {"invalidated", "superseded"} <= kinds
    text = " ".join(x["summary"] for x in d["disclosures"]).lower()
    assert "sb3" in text and "seed" in text


def test_replay_index_is_complete(client):
    d = client.get("/api/v2/replay/index").json()
    assert len(d["episodes"]) == 315
    assert "available" in d


def test_replay_rejects_non_holdout_seed(client):
    r = client.get("/api/v2/replay/episode",
                   params={"policy_id": "root42_masked_bandit",
                           "scenario": "link_failure", "seed": 101})
    assert r.status_code == 400


def test_evidence_api_never_exposes_a_training_or_evaluation_route(client):
    paths = client.get("/openapi.json").json()["paths"]
    v2 = [p for p in paths if p.startswith("/api/v2")]
    assert v2
    for p in v2:
        assert set(paths[p]) == {"get"}, f"{p} exposes a non-GET method"
        assert not any(w in p for w in ("train", "evaluate", "select", "holdout/run"))


def test_study_page_serves(client):
    r = client.get("/study")
    assert r.status_code == 200
    assert "V2 Study" in r.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evidence_api.py -q`
Expected: FAIL — 404 on every `/api/v2/*` path

- [ ] **Step 3: Write the implementation**

Create the router; in `server/main.py` add exactly:

```python
from server.evidence_api import router as evidence_router
...
app.include_router(evidence_router)


@app.get("/study")
def study() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "study.html")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evidence_api.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add server/evidence_api.py server/main.py tests/test_evidence_api.py frontend/study.html
git commit -m "Expose the frozen V2 evidence over a read-only API"
```

---

### Task 6: The `/study` surface

**Files:**
- Create: `frontend/study.html`, `frontend/js/study.js`, `frontend/css/study.css`
- Test: `tests/test_study_ui.py`

**Interfaces:**
- Consumes: `/api/v2/*` from Task 5.
- Element IDs pinned by the test (mirroring the existing `test_presentation.py` contract):
  `study-header`, `stage-badge`, `nav-verdict`, `nav-holdout`, `nav-scenarios`,
  `nav-churn`, `nav-development`, `nav-provenance`, `nav-replay`,
  `verdict-body`, `holdout-body`, `scenario-body`, `churn-body`, `development-body`,
  `provenance-body`, `replay-body`, `replay-controls`, `replay-timeline`,
  `disclosure-body`, `error-banner`, `empty-state`.

Sections, in order:
1. **Verdict** — closed-study status, the seven `CONCLUSIONS`, the one-shot badge.
2. **Final holdout** — aggregate five-method comparison, per-root learner comparison.
3. **Scenarios** — seven-scenario comparison, PPO's `deceptive_local_optimum` win called
   out rather than buried.
4. **Operations & churn** — delivery, SLA, utilization, congestion, delay, loss;
   reroutes, reversals, flaps, moved bandwidth, dwell, TE changes, FRR, disconnections,
   restoration; reward-component breakdown with the exact-sum residual; action/no-op
   distribution showing **both** no-op grains with their labels.
5. **Development evidence** — visually distinct region, persistent "development /
   continuity — not holdout" marker, learning curves, checkpoint selection, seed-42 pilot.
6. **Provenance & safety** — integrity table, checkpoint provenance, runtime/device/GPU,
   source SHAs, artifact locations, and progressively disclosed invalidated/superseded
   run disclosures.
7. **Recorded replay** — episode picker plus timeline, permanently marked RECORDED.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_study_ui.py
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import app

ROOT = Path(__file__).resolve().parents[1]
STUDY_IDS = [
    "study-header", "stage-badge", "nav-verdict", "nav-holdout", "nav-scenarios",
    "nav-churn", "nav-development", "nav-provenance", "nav-replay", "verdict-body",
    "holdout-body", "scenario-body", "churn-body", "development-body",
    "provenance-body", "replay-body", "replay-controls", "replay-timeline",
    "disclosure-body", "error-banner", "empty-state",
]


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_study_page_has_every_bound_element(client):
    html = client.get("/study").text
    missing = [i for i in STUDY_IDS if f'id="{i}"' not in html]
    assert not missing, f"/study is missing element ids: {missing}"


def test_study_page_uses_only_vendored_assets(client):
    html = client.get("/study").text
    for src in re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html):
        assert src.startswith("/static/") or src.startswith("/api/"), src


def test_study_page_hardcodes_no_scientific_numbers():
    """Every number on the page must arrive from /api/v2. A literal in the markup
    would drift from the frozen evidence silently."""
    html = (ROOT / "frontend" / "study.html").read_text(encoding="utf-8")
    body = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    for forbidden in ("18.221", "9.036", "9.185", "-2.327", "1.107", "315"):
        assert forbidden not in body, f"study.html hardcodes {forbidden}"


def test_study_js_never_calls_a_mutating_endpoint():
    js = (ROOT / "frontend" / "js" / "study.js").read_text(encoding="utf-8")
    assert "method: \"POST\"" not in js and "method: 'POST'" not in js
    for banned in ("/api/agent/train", "/api/simulation/start", "/api/export/save-run"):
        assert banned not in js


def test_study_marks_replay_as_recorded_not_live():
    html = (ROOT / "frontend" / "study.html").read_text(encoding="utf-8")
    js = (ROOT / "frontend" / "js" / "study.js").read_text(encoding="utf-8")
    assert "RECORDED" in html or "RECORDED" in js


def test_development_region_is_labelled_in_the_markup():
    html = (ROOT / "frontend" / "study.html").read_text(encoding="utf-8")
    lowered = html.lower()
    assert "development" in lowered and "continuity" in lowered
    assert "not holdout" in lowered or "not final holdout" in lowered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_study_ui.py -q`
Expected: FAIL — `/study` returns 404 or the file is absent

- [ ] **Step 3: Build the page**

Follow `superpowers:frontend-design`, `design:accessibility-review` and `design:ux-copy`.
Visual direction: an operations-evidence dossier, not a SaaS dashboard. Requirements:
data tables as the primary form; one signature visualization (the seven-scenario
paired comparison, since it is where the negative result lives); a coherent type scale
and spacing system; semantic colour that is never the sole encoding; WCAG 2.1 AA contrast;
visible focus rings; full keyboard navigation; `prefers-reduced-motion` honoured;
explicit loading / empty / error / replay-unavailable states.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_study_ui.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/study.html frontend/js/study.js frontend/css/study.css tests/test_study_ui.py
git commit -m "Add the V2 study evidence surface"
```

---

### Task 7: Documentation, release material and the V3 backlog

**Files:**
- Create: `docs/V2_EVIDENCE_AUDIT.md`, `docs/TECHNICAL_DEFENSE.md`,
  `docs/RELEASE_CHECKLIST.md`, `docs/V3_RESEARCH_BACKLOG.md`
- Modify: `README.md`, `CURRENT_SYSTEM_BASELINE.md`, `docs/ARCHITECTURE.md`,
  `docs/API.md`, `docs/PRESENTATION_MODE.md`, `docs/DEMO_SCRIPT.md`,
  `docs/UI_ACCEPTANCE_TESTS.md`

- [ ] **Step 1: Reconcile every stale tracked claim**

`README.md` currently says "78 tests" (the suite is 440+), describes only MaskablePPO
versus conventional controllers, and its limitations section quotes **V1** numbers only.
Correct the count, add the V2 outcome with the both-halves planning statement, add the
`/study` route and the reproducible demo path, and mark the V1 limitations as V1-scoped
rather than deleting them.

- [ ] **Step 2: Write `docs/V2_EVIDENCE_AUDIT.md`**

Record the independent reconciliation: what was checked at which grain, the root-aware
aggregation proof, the exact reward-sum residuals, the two look-alike grains (no-op share
and wall time) and why neither is a discrepancy, and the explicit statement that no
evidence file was modified to resolve anything.

- [ ] **Step 3: Write `docs/TECHNICAL_DEFENSE.md`**

Cover problem framing; MPLS/TE architecture; the V2 environment and its governance;
PPO versus contextual-bandit methodology; roots, checkpoint selection, continuity
evaluation and the one-shot final holdout; the invalidated PPO provenance and its SB3
seed-propagation root cause; results, safety, churn and limitations; authorship; and
why the conclusion is formulation-specific.

- [ ] **Step 4: Write `docs/V3_RESEARCH_BACKLOG.md`**

Every item labelled **UNAPPROVED — NOT EVALUATED**, with a header stating that nothing
in the file is supported by V2 evidence and that each item requires separate
preregistration with new development and untouched evaluation seeds.

- [ ] **Step 5: Commit**

```bash
git add README.md CURRENT_SYSTEM_BASELINE.md docs/
git commit -m "Document the closed V2 study and the post-study release"
```

---

### Task 8: Visual QA, full verification, push

- [ ] **Step 1: Serve and capture**

Start the server via the Browser pane's preview tools (never `Bash`), then screenshot
`/study` at desktop 1280×800, presentation 1920×1080 and narrow 768×1024. Inspect for
clipping, overlap, overflow, illegible charts, hidden controls, inconsistent spacing and
stale content. Iterate until clean.

- [ ] **Step 2: Accessibility pass**

Keyboard-only traversal of every control; visible focus throughout; contrast spot-checks
on body text, table text and every semantic colour; confirm no colour-only encoding;
confirm reduced-motion.

- [ ] **Step 3: Run every gate**

```bash
python -m pytest tests/test_evidence_loader.py tests/test_evidence_claims.py tests/test_evidence_replay.py tests/test_evidence_api.py tests/test_study_ui.py -q
python -m pytest tests/test_presentation.py tests/test_api_e2e.py -q
python -m pytest tests/ -q -k "freeze or pin or frozen"
python -m pytest tests/test_v1_v2_compatibility.py -q
python -m pytest tests/ -q
```

- [ ] **Step 4: Release gates**

Re-verify the protected manifest SHA-256; `git status` for unintended files;
`git diff --stat` reviewed in full; confirm no artifact over ~1 MB and no
`.zip`/`.pt`/`.gz`/TensorBoard file staged; run `superpowers:verification-before-completion`.

- [ ] **Step 5: Push the dedicated branch only**

```bash
git push -u origin feat/post-study-productization
```

Never push `feat/rl-environment-v2`. Re-verify the protected manifest SHA-256 after the push.
