# Handoff — Unified three-mode UI implementation (Prompt 2)

**Status:** backend complete and verified; frontend partially built. This document
is written so a fresh session can resume without any of the preceding conversation.

**Date:** 2026-07-31

---

## 1. Anchor state

| Item | Value |
|---|---|
| Selected design branch | `feat/post-study-productization` |
| Selected design commit | `3bb791e9a2ec6d7e1a402ecaa70a61f37d11f52a` |
| Base commit of the design work | `c49da5febfc754dcf08cbf8db4823c01d3133f15` |
| Implementation branch | `feat/three-mode-ui-implementation` |
| Implementation worktree | `.worktrees/three_mode_ui` |
| Test baseline before this work | 524 passed |

The design package was verified as the newest internally consistent one: it is
the only descendant of `c49da5f` carrying `PRODUCT.md`, `DESIGN.md`, ADR-002, the
surface brief, the 1272-line specification and the 12-task plan, its worktree is
clean, and its only non-document change is an allowlist addition — no governed
science, frozen evidence or experiment artifact was touched by it.

### Protected manifest

`results/environment_v2_validation/manifest.json`

- SHA-256 in this worktree **before and after** all work so far:
  `5227C91A1990C52101A510BC7115C049E138D5B34C2CDD649C66C8AA726A979D`
- `git diff -- results/environment_v2_validation/manifest.json` is **empty**.
- Note: the primary worktree (`rl_in_mpls/`) carries a **pre-existing uncommitted
  modification** to this same file, hashing to
  `5680610C95CEC9551CD22FAD2B365B1023485F59EDB87D3E568BC908EDA086C0`. That change
  predates this session and was deliberately **not** touched, staged or reverted.
  Do not touch it.

### Read these before resuming

1. `docs/superpowers/specs/2026-07-31-unified-rl-in-mpls-product-design.md` — the authority
2. `docs/superpowers/plans/2026-07-31-unified-rl-in-mpls-ui.md` — the 12-task plan
3. `DESIGN.md` — the "Dispatch Atlas" visual contract (tokens, motion, bans)
4. `PRODUCT.md`, `docs/ADR-002-unified-three-mode-product-shell.md`
5. `docs/UNIFIED_UI_SURFACE_BRIEF.md`

---

## 2. What is complete and verified

### Backend — committed

Commit: `feat(product-api): add typed product contracts and decision APIs`

New package `mplssim/product/`:

| Module | Responsibility |
|---|---|
| `contracts.py` | Three modes; Guided Story nested in Presentation; four `SourceKind`s with per-kind permissions; both no-op grains; `OutputSemantics`; forbidden vocabulary; final findings |
| `display_map.py` | Curated Turkey layout (display-only), link bend points, capacity classes, discrete utilization bands. `configs/topology.yaml` untouched |
| `catalog.py` | Capability catalog computed from the filesystem. V2 learners report **unavailable with a reason** because no `V2_LIVE_CHECKPOINTS` binding exists on this machine |
| `schemas.py` | 586/604 observation groups, 69-action space, V1/V2 reward orders — all derived from the real YAML/Python sources |
| `serialize.py` | Typed live snapshot: nodes, links folded to one row per physical link with both directions, demands, metrics with previous/delta, incident phase |
| `decision.py` | observation → mask → policy output → selected action → safety → reward. Mask reasons read from `validate_action` |
| `timeline.py` | Typed events with stable IDs; FRR labelled as built-in protection, separate from TE actions |
| `counterfactual.py` | Clone-only estimate; fingerprints the live engine before and after and refuses to report if it moved |
| `fingerprint.py` / `pairing.py` | Synchronization proof for the comparison lane; disables the verdict when the proof fails |

New `server/product_api.py` routes (all additive; `/api/*` and `/api/v2/*` unchanged):

```
GET  /api/product/capabilities
GET  /api/product/contracts
GET  /api/product/display-map
GET  /api/rl/schema?environment=v1|v2
GET  /api/simulation/snapshot
GET  /api/simulation/decision
GET  /api/simulation/timeline
GET  /api/simulation/comparison
GET  /api/simulation/object/{kind}/{object_id}
POST /api/simulation/counterfactual
```

`server/session.py` additions (no scientific semantics change): `SimSession.id`,
`SimSession.generation`, monotonic `SimSession.sequence`, and
`AlgoRunner._prior_obs` for the observation inspector.

Tests added: `tests/test_product_contracts.py` (26) and `tests/test_product_api.py`
(54) — **80 passing**. They assert, among other things, that no catalog entry
calls a bandit score a probability, that mask reasons come from the validator,
that a counterfactual leaves the session fingerprint unchanged, that a
desynchronized pair disables the verdict, and that the product layer writes
nothing under `results/`, `runs/` or `models/`.

### Frontend — written, NOT yet committed

Untracked in the worktree:

```
frontend/app.html
frontend/css/tokens.css
frontend/css/shell.css
frontend/css/topology-atlas.css
frontend/js/product/  (see inventory below)
```

Complete and reviewed:

- `dom.js`, `format.js` (probability and score have **separate** formatters by design)
- `contracts.js`, `store.js` (generation/sequence guards), `router.js`
- `adapters/live-v1.js`, `adapters/recorded-v2.js`, `adapters/evidence-v2.js`
- `topology-atlas.js` (fixed-position SVG, keyboard traversal, route overlays),
  `topology-list.js` (accessible twin sharing one selection)
- `provenance.js`, `timeband.js`, `explain.js`, `help.js`
- `recommendation-card.js`, `comparison-lane.js`, `guided-story.js` (11 beats)
- `object-inspector.js`, `demand-risk-table.js`, `network-filters.js`
- `observation-inspector.js`, `action-grid.js`, `policy-outputs.js`,
  `reward-waterfall.js`, `model-provenance.js`
- `governed-study.js`, `recorded-trace.js`
- `modes/presentation.js`, `modes/network.js`

---

## 3. What remains

In priority order. Everything is specified; nothing needs a new design decision.

### 3.1 Finish the frontend

**Missing files** (already pre-authorized in the preservation allowlist):

| File | Notes |
|---|---|
| `frontend/js/product/modes/rl.js` | Compose the pipeline strip, observation inspector, action grid, policy outputs, reward waterfall, provenance, governed study and recorded trace. Three secondary views: `decision`, `study`, `provenance` — **not** primary modes |
| `frontend/js/product/shell.js` | Mode switching, drawers with focus trap/restore, keyboard map (§4.3 of the spec), audience view, fullscreen, cockpit wiring, live region |
| `frontend/js/product/main.js` | Boot: read route → load capabilities/contracts/display-map/schema → build atlas → connect WebSocket → render |
| `frontend/css/presentation-mode.css` | Moment rail (fixed-width cells), recommendation card, story beats, cockpit |
| `frontend/css/network-mode.css` | Filter bar, inspector, disclosure |
| `frontend/css/rl-mode.css` | Pipeline strip, action grid, policy bars, reward waterfall |
| `frontend/css/responsive.css` | 1920 / 1440 / 1280 / 768 / 390 per spec §16 |

CSS class names the JS already emits and the stylesheets must cover:
`.moment-rail`, `.moment-cell`, `.story`, `.story__beats`, `.rec`, `.rec__head`,
`.rec__facts`, `.rec__outcomes`, `.rec__outcome`, `.cmp`, `.cmp__token`,
`.filters`, `.filters__group`, `.field`, `.insp`, `.insp__head`, `.insp__badges`,
`.disclosure`, `.obs`, `.obs__controls`, `.acts`, `.acts__grid`, `.act`,
`.pol__bars`, `.pol__track`, `.pol__fill`, `.rew__bars`, `.rew__row`,
`.rew__track`, `.rew__fill`, `.region`, `.region--final`, `.region--development`,
`.findings`, `.replay`, `.replay__picker`, `.prov`, `.cf`.

`frontend/app.html` already exists with every element ID the modules bind to and
an authored 1.5px-stroke SVG icon set (no emoji, no external font, no CDN).

### 3.2 Route cutover (plan Task 12)

Serve `frontend/app.html` from `/`, `/advanced`, `/present` and `/study` in
`server/main.py`, preserving query strings. Optionally stage it at `/app` first
per migration Phase A.

### 3.3 Migrate the legacy surface tests

`tests/test_presentation.py` (`ADVANCED_IDS`, `PRESENT_IDS`,
`test_existing_frontends_are_untouched_by_the_new_page`) and `tests/test_study_ui.py`
(`STUDY_IDS`) pin element IDs and page titles of the **surfaces being replaced**.
Update the ID lists and title markers to the new shell; **keep every behavioural
test** (websocket reconnect, out-of-band intervention, no-training-on-launch,
training confirmation, benchmark honesty, display-scale agreement, offline asset
check, module-graph resolution). Record this as a documented migration reason.

Add a **recursive** module-graph test — the existing one uses
`js_dir.glob("*.js")` and will not see `frontend/js/product/**`.

### 3.4 New test files (pre-authorized in the allowlist)

- `tests/test_product_ui.py` — three-mode nav, backward-compatible routes,
  provenance labels, live/recorded and development/final distinction, city and
  role labels, stable topology positions, PPO-probability vs bandit-score naming,
  banned vocabulary (`contracts.FORBIDDEN_PRODUCT_PHRASES`) absent from product copy
- `tests/test_product_accessibility.py` — landmarks, skip links, `prefers-reduced-motion`,
  `:focus-visible`, non-colour status encoding, list twin present

### 3.5 Documentation (plan Task 12)

`README.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/PRESENTATION_MODE.md`,
`docs/RELEASE_CHECKLIST.md`, plus new `docs/PRODUCT_UI.md` and `docs/ACCESSIBILITY.md`
(both pre-authorized).

### 3.6 Visual QA and Impeccable pass

Per the brief: exercise 1920×1080, 1440×900, 1280, 768, 390; all three modes;
Guided Story; live, recorded, development and final records. Then run **once**:

```bash
node "C:/Users/ugure/.claude/skills/impeccable/scripts/detect.mjs" --json frontend/app.html frontend/css frontend/js/product
```

followed by exactly one deliberate correction pass. Do not re-loop.

---

## 4. Hard constraints that must not be relaxed

1. **Never** modify, stage, revert or overwrite
   `results/environment_v2_validation/manifest.json`. Verify its hash before and after.
2. Every new or modified path must be added to `ALLOWED_NEW_FILES` /
   `ALLOWED_MODIFIED_FILES` in `tests/test_v1_v2_compatibility.py`. Most of the
   remaining work is already listed there. `models/` and `results/` stay fully protected.
3. No training, tuning, evaluation, checkpoint reselection or holdout access.
   Holdout seeds 1001–1005 are blocked for live demonstration.
4. No fabricated telemetry. Recorded V2 traces have **no** per-link utilization —
   the stage must show a static reference topology labelled
   `REFERENCE TOPOLOGY · NO RECORDED LINK TELEMETRY`.
5. Bandit outputs are **action scores / immediate-reward estimates**, never
   probabilities or confidence. Changed observations are **changed features**,
   never causal importance.
6. Both no-op grains keep their full names and denominators.
7. Guided Story is a Presentation workflow. There is no fourth primary mode.
8. Internal router IDs are unchanged; city and role lead, ID is secondary.
9. No new framework, bundler, CDN or font binary.
10. The final suite must meet or exceed **524 + all new tests**.

---

## 5. Known open items

- **V2 live demonstration is unavailable on this machine.** Only V1 checkpoints
  exist under `models/`. The catalog reports this truthfully via the
  `V2_LIVE_CHECKPOINTS` env binding. Guided Story must therefore offer the
  installed V1 MaskablePPO demo explicitly, not silently substitute it
  (spec §8.1). This is the correct behaviour, not a defect.
- **Recorded replay is unavailable** unless `V2_FULL_ARTIFACTS` is set. The
  catalogue still lists all 315 episodes and reports each as unavailable.
- **PPO entropy and value** are not exposed by the live runner. The decision
  payload returns `null` with an explicit reason; the UI must show the reason.
- **Paired-session cloning:** plan Task 4 asked for one engine cloned into both
  runners. Implemented instead as a *fingerprint proof* over the existing
  independently-constructed same-seed runners (`pairing.synchronization`), which
  verifies full-state equality before the first decision and exogenous-input
  equality at every step, and disables the comparison verdict when either fails.
  This is the smallest truthful mechanism and satisfies spec §5.6. If a future
  session wants literal cloning, it must not disturb `MplsTeEnv.eng` ownership.
- **`tests/test_presentation.py::test_pages_reference_only_vendored_scripts`** and the
  study-page equivalents will need `frontend/app.html` added to their path lists
  once the routes cut over.

---

## 6. Resume checklist

```bash
cd "C:/Users/ugure/OneDrive/Masaüstü/rl_in_mpls/.worktrees/three_mode_ui"
git status
git log --oneline -3
py -m pytest -q
```

Then work through §3 in order. Before every commit:

```bash
git diff -- results/environment_v2_validation/manifest.json
```

That must print nothing.
