# Exp 2.1 Comparative Run Results Implementation Plan

> **For agentic workers:** Execute inline in the isolated Exp 2.1 worktree. Subagents are prohibited for this task.

**Goal:** Add a truthful two-slot completed-run comparison workflow, a compact Presentation gateway, and a fourth `/compare` primary mode without changing frozen science or the incumbent visual language.

**Architecture:** Extend the existing process-scoped results store with bounded A/B slots that accept only completed in-memory run records. Serve one additive API payload containing identities, provenance, pairing integrity, real interval series, metric direction, components, and events. Render it with new framework-free ES modules and a dedicated stylesheet, touching incumbent shell files only for route/nav, actions, and the existing Presentation comparison gateway.

**Tech Stack:** FastAPI, Python in-memory records, plain ES modules, authored SVG, existing Dispatch Atlas CSS tokens, pytest, browser QA.

## Global Constraints

- Do not train, tune, evaluate, access holdout environments, or write under governed `results/`, `runs/`, `models/`, or evidence paths.
- Never modify `results/environment_v2_validation/manifest.json` or frozen V2 definitions.
- Keep A and B identities fixed and non-semantic; apply green/amber/red only after metric direction is known.
- Do not invent missing actions, components, utilization, moved bandwidth, incidents, or provenance.
- Full Reset clears A, B, and comparison UI state; Reset run does not clear completed slots.
- Use no new framework, chart library, font, or external asset.
- Run the complete suite exactly once at completion; QA only at 1440x900 and 390 px.

---

### Task 1: Completed-run lifecycle and truthful comparison payload

**Files:**
- Modify: `server/session.py`
- Modify: `mplssim/product/results.py`
- Modify: `server/product_api.py`
- Modify: `server/main.py`
- Test: `tests/test_exp21_comparative_runs.py`

**Interfaces:**
- Produces: `results.comparative_runs(session)`, `assign_comparison_slot(session, slot, run_id)`, `clear_comparison_slot(slot)`, `clear_comparison_runs()`, and `swap_comparison_slots()`.
- Produces: `GET /api/product/comparative-runs`, `PUT/DELETE /api/product/comparative-runs/{slot}`, `POST /api/product/comparative-runs/swap`, and `DELETE /api/product/comparative-runs`.

- [ ] Write API tests for completed-only candidates; capture, replacement, swap, clear, Clear All, Full Reset, and Reset-run preservation.
- [ ] Run the focused file and verify the new tests fail because the API is absent.
- [ ] Record real per-interval decisions, failed-link sets, and moved Mbps in session history; archive completion and checkpoint provenance.
- [ ] Implement the two-slot store and derived pairing/metric/chart/timeline payload with unavailable reasons.
- [ ] Run the focused file and verify it passes.

### Task 2: Fourth route and compact Presentation gateway

**Files:**
- Modify: `mplssim/product/contracts.py`
- Modify: `frontend/app.html`
- Modify: `frontend/js/product/contracts.js`
- Modify: `frontend/js/product/router.js`
- Modify: `frontend/js/product/store.js`
- Modify: `frontend/js/product/adapters/live-v1.js`
- Modify: `frontend/js/product/main.js`
- Modify: `frontend/js/product/shell.js`
- Modify: `frontend/js/product/modes/presentation.js`
- Create: `frontend/js/product/comparison-picker.js`
- Test: `tests/test_exp21_comparative_runs.py`

**Interfaces:**
- Produces: fourth mode id `compare`, stable route `/compare`, A/B slot actions, selected interval state, and deep-link route state.

- [ ] Add failing route/navigation and A/B control tests.
- [ ] Run the focused file and verify failures name the missing fourth mode and controls.
- [ ] Add the route/nav/panel and minimal shell/store/action integration.
- [ ] Replace only the existing lower comparison lane content with the compact selector/summary gateway.
- [ ] Run the focused file and verify it passes.

### Task 3: Comparative analytical surface

**Files:**
- Create: `frontend/js/product/modes/compare.js`
- Create: `frontend/js/product/comparison-charts.js`
- Create: `frontend/css/comparison-mode.css`
- Modify: `frontend/css/responsive.css`
- Test: `tests/test_exp21_comparative_runs.py`

**Interfaces:**
- Consumes: `state.data.comparativeRuns` and comparison actions.
- Produces: accessible SVG plus table alternatives for reward, utilization, delivery, SLA risk, decisions, churn, and 12 reward components.

- [ ] Add failing tests for chart questions, units, table alternatives, zero/70%/100% references, A/B redundant encoding, deep links, keyboard selection, and reduced motion.
- [ ] Run the focused file and verify failures are caused by the absent surface.
- [ ] Build the compare mode in the inherited Dispatch Atlas world with a shared interval cursor and truthful empty states.
- [ ] Run the focused file and verify it passes.

### Task 4: Governance, documentation, and release verification

**Files:**
- Modify: `PRODUCT.md`
- Modify: `DESIGN.md`
- Create: `docs/EXP_2_1_COMPARATIVE_RUN_RESULTS.md`
- Modify: `tests/test_v1_v2_compatibility.py`

**Interfaces:**
- Produces: an explicit three-to-four-mode migration record and reviewed allowlist widening.

- [ ] Add exact modified/new files to the compatibility allowlist with reasons.
- [ ] Document the product-only Exp 2.1 migration and limitations.
- [ ] Parse every Python/JS module and run the focused Exp 2.1 tests once after the milestone.
- [ ] Run browser QA at 1440x900 and 390 px, apply one batched fix pass, and run the Impeccable detector once.
- [ ] Run `py -m pytest -q` exactly once, verify protected hash and prohibited-path diff, then commit and push the branch.
