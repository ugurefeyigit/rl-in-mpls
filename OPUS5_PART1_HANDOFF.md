# Part 1 handoff — live V2 foundation and a simple Presentation Mode

This is the single source of truth for Part 2. Start from the commit named
below, not from `81ac145`.

## 1. Branch, commit, worktree, status

| Field | Value |
|---|---|
| Branch | `claude/rl-mpls-ui-part1-c51fff` |
| Base commit | `81ac1451e83954542af52b8dcdd2a928f1aa58e2` (`feat/three-mode-ui-implementation`) |
| Worktree | `.claude/worktrees/rl-mpls-ui-part1-c51fff` |
| Upstream | `origin/claude/rl-mpls-ui-part1-c51fff` |
| Pushed commit | _see § 12_ |
| Tree status after commit | clean |

No existing worktree was reused, deleted or repurposed. `.worktrees/seed42`,
`.worktrees/continuity_v2`, `.worktrees/final_holdout_v2`,
`.worktrees/post_study_productization` and `.worktrees/three_mode_ui` are
untouched, as are all local changes and artifacts in the primary worktree.

## 2. Protected manifest hashes

`results/environment_v2_validation/manifest.json` was never read for
modification, staged, reverted or normalized.

| Worktree | SHA-256 before | SHA-256 after |
|---|---|---|
| UI worktree (committed version) | `5227C91A1990C52101A510BC7115C049E138D5B34C2CDD649C66C8AA726A979D` | identical |
| Primary worktree (pre-existing unstaged version) | `5680610C95CEC9551CD22FAD2B365B1023485F59EDB87D3E568BC908EDA086C0` | identical |

Prohibited-artifact audit: `git status --porcelain` shows **no** change under
`results/`, `runs/` or `models/`, and no `.zip` or `.pt` file.

## 3. Exact files and APIs changed

### New modules

| File | Purpose |
|---|---|
| `mplssim/product/checkpoints_v2.py` | Immutable six-checkpoint registry, fail-closed verification, inference-only loader |
| `mplssim/product/live_v2.py` | `EngineV2View` — read-only product view over the frozen `SimulationEngineV2` |
| `frontend/js/product/control-panel.js` | The single left control panel |
| `tests/test_v2_live_foundation.py` | 39 tests: default, provenance, pairing, execution, resets, evidence separation |
| `tests/test_presentation_controls.py` | 61 tests: one panel, audience exit, advisor truth, evidence region, responsive |

### Modified

| File | Change |
|---|---|
| `server/session.py` | `environment` + `training_root` on `SessionConfig`; `AlgoRunnerV2`; `make_runner`; `execution` property; advisor-aware `step_manual` and loop; `archive()` / `reset(retain)` / `shutdown()`; `_policy_runner`; V2 traffic-override refusal; semantics-neutral `output_value` / `top_actions[].value` |
| `server/main.py` | `StartRequest` gains `environment`, `training_root`, `execution`; per-environment algorithm validation; `CheckpointUnavailable` → 409; `POST /api/simulation/stop`; `GET /api/simulation/retained-runs` |
| `mplssim/product/catalog.py` | Registry-driven V2 availability, `checkpoint_registry()`, `default_environment`, V2 baselines, plain-language source rows |
| `mplssim/product/contracts.py` | `SourceProfile.plain_label` / `plain_summary` / `group` |
| `mplssim/product/serialize.py` | Environment-aware provenance with checkpoint provenance; V2 metric labels; engine-derived cooldown label; observation availability |
| `mplssim/product/decision.py` | Environment-aware semantics, validator source, reward order and note; `execution` block |
| `mplssim/product/pairing.py` | Per-lane environment identity; mixed-version pairs refused before any engine is read |
| `frontend/app.html` | Control-panel landmark, `.work` column, persistent audience exit; cockpit removed |
| `frontend/js/product/{main,shell,store,provenance,recommendation-card,guided-story,contracts}.js`, `modes/presentation.js`, `adapters/live-v1.js` | Setup state, start/reset actions, control-panel wiring, audience exit + Escape, explanation-vs-proposal card, V2 story contract |
| `frontend/css/{shell,presentation-mode,responsive}.css` | Control column, audience exit, larger topology, responsive collapse |
| `docs/{API,ARCHITECTURE,PRODUCT_UI,PRESENTATION_MODE}.md`, `README.md` | Documented above |
| `tests/test_{state_machine,api_e2e,product_api,presentation,product_ui,product_accessibility}.py` | Legacy V1 suites now declare `"environment": "v1"`; UI id contract migrated to the control panel |

### API surface

```
POST /api/simulation/start   {scenario, environment="v2", algorithms, seed,
                              training_root=42, model_tag, safety_filter,
                              speed, autostart, execution}
POST /api/simulation/reset          reset run  — same experiment at t=0, run retained
POST /api/simulation/stop           full reset — stop runners, clear session
GET  /api/simulation/retained-runs  archived runs (Part 2 comparison input)
```

`GET /api/product/capabilities` now carries `default_environment` and
`checkpoint_registry` (roots, default root + rule, six rows with hashes and
per-row availability), and every source row carries `group`, `plain_label`,
`plain_summary`.

## 4. V2 models, roots, transitions, paths and hashes integrated

Selection is the **pre-holdout continuity selection** recorded in
`results/v2_three_root_continuity/checkpoint_selection.csv`; hashes are
transcribed from `results/v2_final_holdout/checkpoint_provenance.csv`. A test
asserts both, so the registry cannot drift from the study's own record.

Artifact root: `V2_LIVE_CHECKPOINTS` if set, otherwise the main worktree's
`.worktrees/` directory (resolved through `git rev-parse --git-common-dir`, so a
linked worktree still finds it). Resolved on this machine:
`C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls\.worktrees`.

| Root | Algorithm | Transition | Worktree / run directory | Payload SHA-256 | Sidecar SHA-256 |
|---:|---|---:|---|---|---|
| 42 | `masked_bandit` | 250 000 | `seed42` / `seed42_masked_bandit_final` | `c15097700eac518ee259cba67e34e4fba1716881ab3dd912188b55da0c79bf49` | `4c041138f64883c39b5500229cbc55852bac8e164b20eedde0d89a1c9a2e6656` |
| 42 | `maskable_ppo` | 250 000 | `seed42` / `seed42_maskable_ppo_final` | `d34cc77ded05b064fa2a39dbe5c5ccc3126c9e6cf85e36c1b507127c987f5676` | `5b82401d32dca4c1bf3c15301709282a6dd3b8cafac1bc5ee184ef3aece2a67e` |
| 314159 | `masked_bandit` | 300 000 | `continuity_v2` / `seed314159_masked_bandit_final_r2` | `fd474430e9f5ed60d09d82e3d08390151f54c8c0ca10b5abd98fe11d5d2c8433` | `d3b9aaa9561379cc5f201b0cd1f9fc1e281a20029603279ce38b27ffd8b3d9f0` |
| 314159 | `maskable_ppo` | 350 000 | `continuity_v2` / `seed314159_maskable_ppo_final_r2` | `0af41be78102617b103c3e21ebb0ba26ae251f2626ff50b30c0887fdb1320489` | `431d9702c619712ead29f71d2fe4a898dd5fa98220b7d74339e7b8f14cde75c1` |
| 271828 | `masked_bandit` | 400 000 | `continuity_v2` / `seed271828_masked_bandit_final` | `d9c31430ad4320ae238f6d3aa833614edc120f7411c5a3e99372c85707116e73` | `76fd196ed41452c1452bce59afc03c6b474a597203213dc2f37e89e03b1748ff` |
| 271828 | `maskable_ppo` | 150 000 | `continuity_v2` / `seed271828_maskable_ppo_final` | `40d0f9b7fe92449e6e8bfe2bcb44604ac2a5002c0f2a662dbad6cf70c219fb79` | `6352ac6e38d2228a7f2f5bf5118fbcde2f669e950391cf589f99fe84576efdde` |

Training source SHAs: `ca64b62fe29e45ab61aa86d642799aec5a4c25e1` (seed42),
`6a8a4068b98bf9a71dead6e547595b4bbd755689` (continuity_v2). All six verify on
this machine.

**Default root rule (neutral, documented):** root 42 — the study's primary
seed-42 scientific training root and first in the registered root order. Chosen
by fixed identity, never from final-holdout performance. Default policy:
`masked_bandit`, stated as a product choice rather than re-derived at runtime.

**Fail-closed verification order** (`CheckpointUnavailable` on any failure, with
the reason): artifact root present → payload present → sidecar present →
payload SHA-256 == registry → sidecar SHA-256 == registry → sidecar format
`v2-learning-checkpoint-v1` → algorithm, transition, payload hash agreement →
`run_config.environment_version == "v2"` → `run_config.root_seed` == registry
root → source commit == registry training source → the two source records agree
→ `validate_environment_metadata` against the live V2 environment. **V1 is never
substituted, and no observation is padded, truncated or reordered.**

## 5. Live V2 / session / paired-run architecture

- `MplsTeEnvV2` is the live default (`DEFAULT_ENVIRONMENT = "v2"`). Observation
  604, actions 69, the exact 12 V2 reward components.
- `EngineV2View` wraps the frozen engine. It translates only names the product
  layer reads under a V1 spelling — offered traffic → `demand_volumes`, TE dwell
  → cooldown, `validate_te_action` → `validate_action`, and a route-change log
  built from `te_history` + `frr_history` + `restoration_history` with the
  source kept separate (`te` / `frr` / `restore`). `mplssim/sim/engine_v2.py`
  is not edited.
- **Every** V2 controller — learner or baseline — drives a real `MplsTeEnvV2`.
  A baseline only supplies the action integer through
  `choose_baseline_action`, so the authoritative mask, the validator's rejection
  reason and the twelve reward components come from the governed environment in
  every lane.
- Pairing: both lanes are built from the same `SessionConfig` with
  `reset(options={"episode_seed": seed})`, so scenario, seed, initial state,
  traffic, failures and interventions are identical. `pairing.synchronization`
  fingerprints exogenous inputs and reports `matched` with the proof kind, or
  refuses. A pair spanning two environment versions is refused before any engine
  is read.
- Output semantics are declared by the controller, never inferred: PPO exposes
  masked action probabilities from the policy distribution; the bandit exposes
  per-action immediate-reward estimates. The wire fields are semantics-neutral
  (`output_value`, `top_actions[].value`) and only `output_semantics` says which
  a number is.

### Remaining limitations

1. `run-until` in advisor execution applies the controller's own actions for
   that stretch without individual approval. The API returns
   `approval_bypassed: true` with a note and the story copy says so, but it is
   still a real asymmetry with per-step approval.
2. V2 has no manual traffic multiplier or burst injector. Both endpoints return
   409 with the reason rather than emulating one.
3. PPO entropy and value estimates are still not exposed by the live runner;
   both report their reason.
4. Recorded replay still has no per-link utilization, unchanged from before.
5. `/api/export/save-run` and `/api/lsps` were not re-verified against a V2
   session; they read `snapshot()`/history and should work, but Part 2 should
   confirm before relying on them.
6. Retained runs are held in memory on the session object only. They do not
   survive a full reset or a server restart.

## 6. Presentation controls and state-machine behaviour delivered

One persistent left control panel, in this order: environment (V2 default),
scenario, seed (validated), execution style, controller A, optional comparison +
controller B, checkpoint root, speed, Start run; then play/pause, step, skip to
next event, stop, reset run, full reset; then approve/reject (advisor only);
then Guided Story with manual and automatic pacing; then a separate **Study
evidence and results** region. The bottom cockpit is gone. The topology owns the
full work column and is larger.

- **Reset run** — same environment, scenario, seed, controllers and root at step
  zero; the replaced run is archived and readable at
  `/api/simulation/retained-runs`.
- **Full reset** — stops runners, clears the session server-side, clears
  transient UI state, closes Guided Story and the advisor workflow, leaves
  audience view, and returns to the configuration form.
- Neither mutates a model, a checkpoint or any evidence artifact (asserted by a
  file-stamp test over `results/` and `models/`).
- **Audience view** — exit control rendered outside the chrome audience view
  hides, pinned visible at every viewport including fullscreen; `Escape` leaves
  audience view before fullscreen and restores focus to the toggle; nothing
  reloads.
- **Automatic execution** — the policy acts; the card explains a completed
  decision; there is no approval affordance and no fabricated preview. The
  meaningless "Preview recommendation" control is gone everywhere.
- **Advisor execution** — `step` produces a proposal instead of advancing, the
  loop proposes and stops, and only Approve / Reject moves the run on. Reject
  applies action 0.
- **Guided Story** — real `demo_evening`, seed 42, V2, `masked_bandit` +
  `greedy`, advisor execution. Fully step-driven; automatic playback pauses at
  every recommendation until Approve or Reject, then continues. Previous, next,
  automatic, restart and exit are all present.
- **Evidence** — bare "Development" and "Final Evidence" are gone from the setup
  path. Plain wording is used, and all three records live in the Study evidence
  region, never beside the scenario or model pickers.

## 7. Test results

| Run | Command | Result |
|---|---|---|
| Full suite | `py -m pytest -q` | **758 passed, 0 failed** in 117.62 s |
| Baseline required | — | 654 — exceeded by 104 |
| New: live V2 foundation | `py -m pytest tests/test_v2_live_foundation.py -q` | 39 passed |
| New: presentation controls | `py -m pytest tests/test_presentation_controls.py -q` | 61 passed |
| V1/V2 compatibility | `py -m pytest tests/test_v1_v2_compatibility.py -q` | 31 passed |
| JavaScript parse | `node --check` over all 43 modules under `frontend/js` | all pass |
| Python parse | `ast.parse` over `mplssim/`, `server/`, `tests/` | all pass |

Two governance failures appeared on the first full run and were resolved
honestly rather than by weakening a check:
`tests/test_v1_v2_compatibility.py` carries an explicit allowlist that every
prior authorized stage extended in a reviewable commit. Part 1's five new files
plus this handoff were added to `ALLOWED_NEW_FILES`, and `tests/test_api_e2e.py`
to `ALLOWED_MODIFIED_FILES`, each with the reason. The substantive guard —
`test_models_results_figures_and_v1_configs_are_byte_identical_to_the_base` —
passed unchanged on both runs, so V1's models, results, figures, configs and
simulation source are still byte identical to the audited base.

Migrated suites: the pre-existing V1 API/UI suites now declare
`"environment": "v1"` explicitly (V2 is the default), and the UI id contract
moved from the removed cockpit to the control panel. No assertion was relaxed.

## 8. Browser QA

Server: `python -m uvicorn server.main:app --port 8000`, route `/present`.

| Viewport | Page-level horizontal overflow | Clipped panel controls | Notes |
|---|---|---|---|
| 1920×1080 | none (`scrollWidth` 1905 ≤ 1920) | 0 | Control column 312 px |
| 1440×900 | none (1425 = client) | 0 | Column 280 px, atlas 486 px tall |
| 1280×800 | none (1265 = client) | 0 | Column 264 px, atlas 418 px |
| 768×1024 | none (753 = client) | 0 | Column stacks full width; every control ≥ 44 px |
| 390×844 | none (390 = client) | 0 | Audience exit fully in view at (12, 788, 366×44) |

Exercised live: start a V2 run from the panel; step; automatic card reads
"…moved İzmir → Erzurum bulk data" with provenance
`masked_bandit-root42-250000`; advisor run holds at step 0 with "Awaiting your
approval" and "Nothing has been applied"; approve advances to step 1 and the
card reads "Resolved"; reset run returns to step 0 with "1 earlier run(s) kept";
full reset returns to "No run yet"; audience view exit and `Escape` both work
and restore focus; Guided Story reaches beat 2, refuses to advance with "This
beat is waiting for you: approve or reject the recommendation before
continuing", and after approving narrates the real interval (17:05, busiest link
116 %, 3 SLA violations). Network and RL modes still render with no overflow.
Browser console: no errors. Server log: no application errors.

### Newcomer confusion log and what was done

| Observed | Resolution |
|---|---|
| No obvious starting point; configuration was split across the header chip, the bottom cockpit and drawers | One numbered left panel; cockpit removed |
| "Preview recommendation" in automatic mode implied an approval that did not exist | Removed; automatic shows a completed decision explanation |
| Advisor mode did not actually hold anything — `Step` applied the action | `step` and the loop now propose and stop; only Approve/Reject advances |
| Audience view had no visible way out and `Escape` did not leave it | Persistent exit control outside the hidden chrome; `Escape` leaves audience view before fullscreen |
| Guided Story beat 2 read "busiest link is at —" while a proposal was held | Beat copy branches on the pending proposal and on "no completed interval" |
| "Development" / "Final Evidence" read as live model choices | Plain wording, own region, explicitly not runnable |
| `rl` vs `ppo_te` read as two controllers | Labelled "MaskablePPO · V1 checkpoint ppo_te" with the relationship stated |

## 9. Part 2 tasks

1. **Comparison presentation.** The paired V2 foundation and synchronization
   proof exist; the comparison lane still renders the old summary. Build the
   full side-by-side decision comparison on top of `/api/simulation/comparison`
   and `/api/simulation/moment`.
2. **Retained runs.** `/api/simulation/retained-runs` returns archived runs but
   nothing consumes them. Build the cross-run results surface, and decide
   whether they should survive a full reset or a restart.
3. **Cross-mode results system.** Unify live results, retained runs and the
   frozen study record into one results surface without ever averaging
   development and final evidence together.
4. **Final evidence integration.** Refine the Study evidence region: the
   frozen numbers (bandit 18.221, PPO 9.036, greedy −2.327; bandit won six of
   seven scenarios; PPO retained a 1.107-point advantage in
   `deceptive_local_optimum`) are rendered only from governed evidence today,
   which must stay true.
5. **Advisor fast-forward asymmetry.** Decide whether `run-until` in advisor
   execution should be refused, batched into a single approval, or kept with
   the current disclosure.
6. **`/api/export/save-run` and `/api/lsps` under V2** — verify.
7. **Guided Story beats 3–11 under V2 advisor pacing** — only beats 1–2 were
   exercised interactively; the rest pass structurally but deserve a live pass.
8. **`final/` release assembly** — not started, as instructed.
9. **Deferred detail:** the V1 `provenance-word` id is replaced on first render
   (pre-existing); the moment rail still shows eight cells and could be reduced
   for a projector; `mplssim/product/checkpoints_v2.py` is the one product
   module allowed to import the learner classes (inference only), guarded by a
   test that no product module can train or save.

## 10. Commands

```bash
# Serve the application
python -m uvicorn server.main:app --port 8000
# then open http://127.0.0.1:8000/present
```

```bash
# Focused Part 1 suites
py -m pytest tests/test_v2_live_foundation.py tests/test_presentation_controls.py -q
```

```bash
# Full suite
py -m pytest -q
```

```bash
# Resume work
cd .claude/worktrees/rl-mpls-ui-part1-c51fff && git status
```

Optional: set `V2_LIVE_CHECKPOINTS` to a directory holding `seed42` and
`continuity_v2` if the training worktrees are not under the main worktree's
`.worktrees/`.

## 11. Scientific confirmation

No training, tuning, evaluation, checkpoint selection or reselection, holdout
environment access, holdout-informed decision, evidence mutation or
scientific-semantics change was performed. Environment semantics, observations,
actions, masks, rewards, candidate paths, simulator topology, traffic, failures,
algorithms, hyperparameters, governed checkpoints, frozen seeds and internal IDs
are unchanged; `frozen_definition_drift()` is empty and asserted by a test. No
telemetry, probability, explanation, counterfactual, reward or replay link
utilization is fabricated. The frozen study truth is rendered only from governed
evidence.

## 12. Final verification

- Full suite: **758 passed, 0 failed** (baseline 654).
- JavaScript and Python parse checks: clean.
- Protected manifest hashes: unchanged in both worktrees (§ 2).
- `git status --porcelain`: no change under `results/`, `runs/` or `models/`;
  no `.zip` or `.pt` artifact.
- Browser QA at 1920, 1440, 1280, 768 and 390 px: no page-level horizontal
  overflow, no clipped control, no inaccessible exit, no hidden primary action,
  no misleading provenance, topology readable (§ 8).
- V1 byte-identity test passes against the audited base.

Pushed commit: recorded in `git log -1` on
`origin/claude/rl-mpls-ui-part1-c51fff`.
