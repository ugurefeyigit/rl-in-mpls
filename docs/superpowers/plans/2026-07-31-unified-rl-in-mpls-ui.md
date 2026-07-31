# Unified RL-in-MPLS UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved three-mode Dispatch Atlas product without changing governed science, while preserving truthful distinctions among live simulation, recorded replay, development evidence, and final evidence.

**Architecture:** Keep the build-free frontend and existing APIs operational while introducing additive typed product contracts, a shared application state store, and source adapters. A single shell renders Presentation, Network Information, and RL Information from one context. V1 live, V2 live demonstration, recorded aggregate traces, development evidence, and final evidence remain separate adapters with explicit capability declarations. Route cutover occurs only after parity gates pass.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, NumPy, existing simulation engines and learners, pytest, dependency-free ES modules, SVG, vendored ECharts, Playwright or the repository's existing browser-test harness, axe-core if vendored.

## Global Constraints

- Treat the governed V2 study as closed. Do not train, tune, evaluate, select, retry, sweep, or access untouched holdout seeds.
- Never modify, stage, overwrite, or revert `results/environment_v2_validation/manifest.json`.
- Do not modify frozen reports, result artifacts, checkpoints, learners, reward semantics, scenario semantics, or experiment worktrees.
- Do not write through `/api/v2/*`; retain its fail-closed loaders and claims arithmetic unchanged.
- Keep LIVE, RECORDED, DEVELOPMENT, and FINAL EVIDENCE as distinct source types. `LIVE DEMONSTRATION` is live, not final evidence.
- Never infer or synthesize missing telemetry. Recorded V2 traces remain aggregate; they do not drive a link-level topology replay.
- Preserve both no-op grains and labels: episode-level mean no-op frequency and step-pooled no-op share.
- Preserve internal router IDs. Display city and role first, internal ID second.
- Only expose repository-implemented policies and baselines. Unconfigured checkpoints appear unavailable with a reason.
- The frontend remains build-free, offline-capable, CDN-free, and framework-free unless a separately approved ADR replaces this decision.
- All additions are backward compatible until Task 12. Existing `/api/*` clients and `/present`, `/advanced`, `/study` links must continue to work.
- Before every commit, run `git diff -- results/environment_v2_validation/manifest.json` and confirm no output.
- Reference specification: `docs/superpowers/specs/2026-07-31-unified-rl-in-mpls-product-design.md`.
- Visual contract: `DESIGN.md` and `.impeccable/surfaces/frontend-index-html.md`.
- Product contract: `PRODUCT.md`.

---

### Task 1: Freeze the product contracts in executable tests

**Files:**
- Create: `tests/test_product_contracts.py`
- Create: `tests/frontend/product-contracts.test.mjs`
- Modify: `pyproject.toml` only if the existing test discovery requires it

**Interfaces and invariants:**
- Defines canonical source kinds: `live`, `recorded`, `development_evidence`, `final_evidence`.
- Defines stable mode IDs: `presentation`, `network`, `rl`.
- Defines backward-compatible route expectations.
- Defines frozen scientific strings, values, scenario facts, action count, observation sizes, and reward-component order as regression assertions.

- [ ] Write tests that fail because product contracts do not exist yet.
- [ ] Assert exactly three primary modes and assert Guided Story is nested under Presentation.
- [ ] Assert `/present`, `/advanced`, `/study`, and `/` mappings from the approved specification.
- [ ] Assert source kinds cannot be coerced into one another and required provenance fields differ by kind.
- [ ] Assert V1 observation length 586, V2 length 604, action count 69, and the exact 12-component V2 reward order.
- [ ] Assert the two no-op metrics have different IDs, labels, denominators, and descriptions.
- [ ] Assert recorded traces declare link telemetry unavailable.
- [ ] Assert final findings include the negative temporal-planning conclusion and safety qualification.
- [ ] Run `py -m pytest -q tests/test_product_contracts.py` and the existing frontend contract command; confirm expected missing-contract failures.
- [ ] Commit only the failing contract tests: `test(product): freeze three-mode and provenance contracts`.

### Task 2: Add capability, policy, schema, and display metadata APIs

**Files:**
- Create: `server/product_api.py`
- Create: `mplssim/product/__init__.py`
- Create: `mplssim/product/contracts.py`
- Create: `mplssim/product/catalog.py`
- Create: `mplssim/product/schemas.py`
- Create: `mplssim/product/display_map.py`
- Create: `tests/test_product_api.py`
- Modify: `server/main.py`
- Modify: `mplssim/display.py`
- Modify: `docs/API.md`

**Interfaces:**
- `GET /api/product/capabilities`
- `GET /api/product/display`
- `GET /api/rl/schema?environment=v1|v2`
- Pydantic models `CapabilityCatalog`, `PolicyCapability`, `SourceCapability`, `ObservationSchema`, and `DisplayMap`.

- [ ] Write failing API tests for stable policy IDs, output semantics, clone support, checkpoint availability, environment version, and unavailable reasons.
- [ ] Assert the catalog lists only actual implementations: V1 PPO (`rl`), static, greedy, CSPF, random where supported; V2 MaskablePPO, masked bandit, static, greedy, and CSPF only where configured.
- [ ] Assert no catalog item describes bandit scores as probabilities.
- [ ] Write failing schema tests for exact offsets, group lengths, normalization descriptions, action formula, and reward order derived from actual Python/YAML definitions.
- [ ] Add a display-only Turkey coordinate registry for all 18 routers, link bend points, label offsets, city names, roles, and an explicit `geographic_precision: curated_not_gis` field.
- [ ] Assert scientific topology IDs and links are never mutated by display lookup.
- [ ] Mount the additive router without changing existing routes.
- [ ] Run `py -m pytest -q tests/test_product_contracts.py tests/test_product_api.py tests/test_display_registry.py`.
- [ ] Commit: `feat(product-api): add truthful capability schema and map metadata`.

### Task 3: Introduce typed frontend state and source adapters

**Files:**
- Create: `frontend/js/product/contracts.js`
- Create: `frontend/js/product/store.js`
- Create: `frontend/js/product/router.js`
- Create: `frontend/js/product/availability.js`
- Create: `frontend/js/product/adapters/live-v1.js`
- Create: `frontend/js/product/adapters/live-v2.js`
- Create: `frontend/js/product/adapters/recorded-v2.js`
- Create: `frontend/js/product/adapters/evidence-v2.js`
- Create: `frontend/js/product/explain.js`
- Create: `frontend/js/product/moment-ref.js`
- Create: `tests/frontend/product-store.test.mjs`
- Create: `tests/frontend/source-adapters.test.mjs`

**Interfaces:**
- `ProductContext` contains mode, source descriptor, session, scenario, model, time, phase, selection, moment reference, playback, and panel state.
- Every adapter returns `{ context, capabilities, snapshot, decision, timeline, availability }` without fabricating absent fields.
- `explainMoment(context, depth)` returns deterministic fact templates and direct object references.

- [ ] Write failing tests for reset generation, monotonically increasing sequence, stale-response rejection, delta guards, mode switching, deep-link parsing, and focus restoration.
- [ ] Write a negative fixture proving a recorded aggregate row cannot satisfy the live topology adapter.
- [ ] Write learner-semantics fixtures proving probabilities, scores, and unavailable outputs render through different fields.
- [ ] Implement the smallest immutable store and adapter layer; avoid view-specific network requests.
- [ ] Implement deterministic explanations at Presentation, Network, and RL depth using only payload facts.
- [ ] Assert explanations contain no causal or anthropomorphic claim unless the source field explicitly supports it.
- [ ] Run the frontend unit suite and `py -m pytest -q tests/test_product_contracts.py`.
- [ ] Commit: `feat(frontend): add typed product context and source adapters`.

### Task 4: Build an isolated V2 live-demonstration and exact paired-session service

**Files:**
- Create: `mplssim/product/live_v2.py`
- Create: `mplssim/product/pairing.py`
- Create: `mplssim/product/fingerprint.py`
- Create: `tests/test_live_v2_product.py`
- Create: `tests/test_paired_sessions.py`
- Modify: `server/session.py`
- Modify: `server/product_api.py`

**Interfaces:**
- Versioned session start accepts `environment_version`, stable policy ID, optional configured checkpoint ID, ordinary demonstration seed, and optional comparator.
- `PairedSession.start()` creates one engine state and clones it for two runners.
- `PairedSession.step()` applies the same exogenous interventions and exposes a synchronization fingerprint before and after each step.
- V2 runner loads configured frozen checkpoints for inference only and writes no evidence or experiment artifact.

- [ ] Write failing tests that reject holdout seeds 1001-1005, missing checkpoint bindings, unsupported policy/environment pairs, more than two runners, and any training/evaluation option.
- [ ] Write clone equality and divergence-detection tests over engine state, traffic state, RNG state, scenario step, topology state, and initial routing.
- [ ] Assert comparison disables itself with a visible reason if fingerprints differ.
- [ ] Assert each action is validated against its runner's authoritative mask.
- [ ] Assert reset, step, and close write nowhere under `results/`, `runs/`, or experiment worktrees.
- [ ] Preserve the current V1 `SessionManager` behavior behind the compatibility adapter.
- [ ] Run `py -m pytest -q tests/test_live_v2_product.py tests/test_paired_sessions.py tests/test_server.py`.
- [ ] Commit: `feat(simulation): add isolated V2 demos and exact paired sessions`.

### Task 5: Add versioned snapshot, decision, timeline, and counterfactual contracts

**Files:**
- Create: `mplssim/product/serialize_v1.py`
- Create: `mplssim/product/serialize_v2.py`
- Create: `mplssim/product/decision.py`
- Create: `mplssim/product/timeline.py`
- Create: `mplssim/product/counterfactual.py`
- Create: `tests/test_product_serializers.py`
- Create: `tests/test_decision_contract.py`
- Create: `tests/test_counterfactual.py`
- Modify: `server/product_api.py`
- Modify: `server/session.py`
- Modify: `docs/API.md`

**Interfaces:**
- `GET /api/simulation/snapshot`
- `GET /api/simulation/decision`
- `GET /api/simulation/timeline`
- `GET /api/simulation/object/{kind}/{object_id}`
- `POST /api/simulation/counterfactual`
- Every live payload carries session ID, generation, sequence, step, environment version, and source kind.

- [ ] Write failing golden-contract tests for node, link, demand, path, metrics, incident, and availability fields in V1 and V2.
- [ ] Test each of the 69 actions: no-op plus 17 demands times 4 candidate paths.
- [ ] Test mask state, environment rejection, operator rejection, and unavailable reason as different values.
- [ ] Test PPO probabilities sum within tolerance only over valid actions when exposed; test entropy/value nullable reasons.
- [ ] Test bandit scores retain raw scale and are never normalized or renamed probability/confidence.
- [ ] Test reward exact-sum integrity against the authoritative 12 components and preserve current/cumulative separation.
- [ ] Emit stable timeline event IDs for congestion, SLA risk, failure, FRR, recommendation, action, recovery, and stabilization without inventing absent events.
- [ ] Implement counterfactual evaluation only on clones, verify the running fingerprint is unchanged, and label the result `simulated_estimate`.
- [ ] Return a typed unavailable response where cloning or required telemetry is unsupported.
- [ ] Run `py -m pytest -q tests/test_product_serializers.py tests/test_decision_contract.py tests/test_counterfactual.py tests/test_evidence_api.py`.
- [ ] Commit: `feat(product-api): expose versioned decisions timelines and previews`.

### Task 6: Build the shared Dispatch Atlas shell and topology

**Files:**
- Create: `frontend/css/tokens.css`
- Create: `frontend/css/shell.css`
- Create: `frontend/css/topology-atlas.css`
- Create: `frontend/css/responsive.css`
- Create: `frontend/js/product/shell.js`
- Create: `frontend/js/product/context-rail.js`
- Create: `frontend/js/product/provenance.js`
- Create: `frontend/js/product/topology-atlas.js`
- Create: `frontend/js/product/topology-list.js`
- Create: `frontend/js/product/help.js`
- Create: `tests/frontend/shell.test.mjs`
- Create: `tests/frontend/topology-atlas.test.mjs`
- Modify: `frontend/index.html`

**Interfaces:**
- Exactly three primary mode controls with roving focus and route synchronization.
- Persistent context rail and provenance treatment.
- Stable SVG topology with synchronized accessible list alternative.
- `MomentRef` focus for router, link, demand, path, incident, decision, action, and reward event.

- [ ] Write failing DOM tests for landmarks, three modes, provenance text, keyboard traversal, focus restoration, and non-color status labels.
- [ ] Add the DESIGN.md tokens literally; add no unapproved gradient, glow, glass, or external font.
- [ ] Render the 18-router curated layout with city/role leading and ID secondary.
- [ ] Implement deliberate link bends, hit targets, directional utilization, status patterns/icons, selection, and readable label avoidance.
- [ ] Keep node positions stable during data changes and resize.
- [ ] Synchronize SVG selection with the accessible topology list in both directions.
- [ ] Implement Explain this moment and direct navigation without modal traps.
- [ ] Run frontend tests plus `py -m pytest -q tests/test_presentation.py tests/test_study_ui.py` to prove legacy surfaces still work.
- [ ] Commit: `feat(ui): establish Dispatch Atlas shell and topology`.

### Task 7: Migrate Presentation Mode and Guided Story

**Files:**
- Create: `frontend/css/presentation-mode.css`
- Create: `frontend/js/product/modes/presentation.js`
- Create: `frontend/js/product/guided-story.js`
- Create: `frontend/js/product/presenter-cockpit.js`
- Create: `frontend/js/product/comparison-lane.js`
- Create: `tests/frontend/presentation-mode.test.mjs`
- Create: `tests/frontend/guided-story.test.mjs`
- Modify: `frontend/present.html`
- Modify: `docs/PRESENTATION_MODE.md`

**Interfaces:**
- Audience view, hideable presenter cockpit, fullscreen, play/pause, next/back, progress, bookmarks, Q&A jumps, Decision Lens, recommendation/outcome card, and comparison lane.

- [ ] Write failing tests for the full keyboard map and verify typing contexts do not trigger global shortcuts.
- [ ] Render scenario, phase, state/time, policy, current action, interval/cumulative reward, network condition, incident, change, recommendation, observed outcome, comparison, and study conclusion.
- [ ] Make the topology the dominant 16:9 region; keep controls and numerals spatially stable.
- [ ] Implement Decision Lens as non-mutating preview with old, proposed, and comparator paths labelled independently.
- [ ] Implement the exact Guided Story beats from the specification using `demo_evening`; do not conflate the L20 Kayseri-Samsun event with the L11 Ankara-Kayseri Q&A example.
- [ ] Show `Masked Bandit suggests [actual action]` only when an actual bandit decision payload exists; otherwise use the selected real policy name or unavailable state.
- [ ] Keep expected and observed telemetry in separate rows and mark clone results `SIMULATED ESTIMATE`.
- [ ] End with the complete governed conclusion, including negative and mixed findings.
- [ ] Verify same-scenario/seed/start comparison fingerprints before rendering the comparison lane.
- [ ] Run the focused frontend suite and presentation API tests.
- [ ] Commit: `feat(presentation): migrate Guided Story into topology-first mode`.

### Task 8: Migrate Network Information Mode

**Files:**
- Create: `frontend/css/network-mode.css`
- Create: `frontend/js/product/modes/network.js`
- Create: `frontend/js/product/network-filters.js`
- Create: `frontend/js/product/object-inspector.js`
- Create: `frontend/js/product/demand-risk-table.js`
- Create: `frontend/js/product/incident-timeline.js`
- Create: `tests/frontend/network-mode.test.mjs`
- Modify: `frontend/index.html`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Traffic class, congestion, SLA risk, failure/degraded/recovery filters.
- Router/link/demand/path inspection; current/prior deltas; primary/alternate routes; bottlenecks; demand/SLA table; FRR/restoration timeline.

- [ ] Write failing tests for bidirectional topology/table focus and filter intersections.
- [ ] Render capacity, utilization direction, status, modeled delay/loss, demand traffic, SLA risk, bottleneck, path chain, and availability labels from typed fields only.
- [ ] Separate current affected counts from cumulative intervals.
- [ ] Separate FRR/protection events from TE reroutes and label modeled behavior versus real-operator considerations.
- [ ] Render reroute, reversal, flap, churn, dwell, moved bandwidth, recovery, and stabilization only where the source exposes them.
- [ ] Add previous/current/delta presentation guarded by matching generation and adjacent sequence.
- [ ] Verify filters never hide the active focused object without announcing and moving focus predictably.
- [ ] Run focused frontend and serializer tests.
- [ ] Commit: `feat(network-ui): add MPLS operations workspace`.

### Task 9: Build the RL Decision Observatory

**Files:**
- Create: `frontend/css/rl-mode.css`
- Create: `frontend/js/product/modes/rl.js`
- Create: `frontend/js/product/observation-inspector.js`
- Create: `frontend/js/product/action-space.js`
- Create: `frontend/js/product/policy-outputs.js`
- Create: `frontend/js/product/reward-waterfall.js`
- Create: `frontend/js/product/model-provenance.js`
- Create: `tests/frontend/rl-mode.test.mjs`
- Modify: `frontend/study.html`

**Interfaces:**
- Pipeline: observation -> mask -> policy outputs -> selected action -> safety -> transition -> reward components -> next observation.
- Searchable/grouped observations, complete action space, learner-specific diagnostics, reward waterfall, integrity, and model provenance.

- [ ] Write failing tests for the exact 604-feature V2 grouping, search, prior/current/delta, normalization labels, and stable semantic descriptions.
- [ ] Name ranked changes `Changed features`; include explicit non-causal help text.
- [ ] Render all 69 actions with no-op separated from 17 demand groups of four candidate paths.
- [ ] Show valid, invalid, chosen, runner-up, no-op, and unavailable states plus rejection reasons.
- [ ] Render PPO top probabilities, entropy, and value only when exposed; otherwise show the backend reason.
- [ ] Render bandit scores/immediate reward estimates without percentages, probability language, or confidence language.
- [ ] Render all 12 reward components in authoritative order, exact-sum indicator, current reward, and cumulative reward.
- [ ] Render action and no-op distributions with the correct grain labels.
- [ ] Render checkpoint hash, root, training stage, evaluation stage, seed-ledger provenance, integrity, and safety without treating final evidence as live.
- [ ] Run focused frontend tests and decision/schema API tests.
- [ ] Commit: `feat(rl-ui): add truthful decision observatory`.

### Task 10: Integrate governed study and aggregate recorded replay

**Files:**
- Create: `frontend/js/product/governed-study.js`
- Create: `frontend/js/product/recorded-trace.js`
- Create: `tests/frontend/governed-study.test.mjs`
- Create: `tests/frontend/recorded-trace.test.mjs`
- Modify: `frontend/js/product/adapters/recorded-v2.js`
- Modify: `frontend/js/product/adapters/evidence-v2.js`
- Modify: `frontend/study.html`
- Modify: `docs/V2_EVIDENCE_AUDIT.md` only for UI access instructions, not findings

- [ ] Write fixtures from existing evidence API responses; do not copy or rewrite result artifacts.
- [ ] Assert final evidence always displays FINAL EVIDENCE, frozen identity, and non-live language.
- [ ] Assert development evidence cannot appear in a final-holdout comparison.
- [ ] Render recorded replay as aggregate time-series/table data with RECORDED status and an explicit `No per-link utilization was recorded` state.
- [ ] Do not mount, animate, or color a live topology from recorded aggregate rows.
- [ ] Preserve all final numbers, scenario/root results, PPO-only deceptive-local-optimum win, safety finding, moved-bandwidth trade-off, and planning limitation.
- [ ] Render episode-mean and step-pooled no-op values in separate components with denominators.
- [ ] Run evidence, replay, claims, and focused frontend tests.
- [ ] Commit: `feat(evidence-ui): integrate frozen study and aggregate replay`.

### Task 11: Responsive, accessibility, motion, and performance hardening

**Files:**
- Create: `tests/accessibility/test_product_accessibility.py`
- Create: `tests/visual/test_product_viewports.py`
- Create: `tests/performance/test_frontend_budget.py`
- Modify: `frontend/css/responsive.css`
- Modify: all new mode/component CSS only where a failing check requires it
- Modify: `docs/RELEASE_CHECKLIST.md` with additive product checks only

- [ ] Capture deterministic screenshots at 1920x1080, 1440x900, 1280x800, 768x1024, and 390x844 for every primary mode and key provenance state.
- [ ] Test no horizontal page overflow, no clipped focus, readable node plates, stable KPI widths, and usable 200% browser zoom.
- [ ] Run automated WCAG checks and a manual keyboard script covering shell, topology/list, dialogs/drawers, timeline, tables, Guided Story, fullscreen, and Q&A jumps.
- [ ] Verify WCAG 2.1 AA text/UI contrast, visible focus, semantic headings/landmarks, live-region restraint, and non-color status encoding.
- [ ] Verify `prefers-reduced-motion` removes path drawing, number interpolation, and panel travel while retaining state changes.
- [ ] Measure controls at 120-180 ms, panels at 200-320 ms, topology events at 400-800 ms, and confirm no ambient loop.
- [ ] Profile a dense topology step; eliminate forced layout loops and confirm the target device sustains smooth event animation near 60 fps.
- [ ] Confirm all scripts/styles/assets are vendored and the app works with network access disabled after server start.
- [ ] Run full frontend, accessibility, visual, performance, and existing presentation/study tests.
- [ ] Commit: `test(ui): harden responsive accessible motion and performance`.

### Task 12: Cut over routes, remove duplication safely, and release

**Files:**
- Modify: `server/main.py`
- Modify: `frontend/index.html`
- Modify: `frontend/present.html`
- Modify: `frontend/study.html`
- Modify: `frontend/js/app.js`
- Modify: `frontend/js/present.js`
- Modify: `frontend/js/study.js`
- Modify: `frontend/css/app.css`
- Modify: `frontend/css/present.css`
- Modify: `frontend/css/study.css`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/API.md`
- Modify: `docs/PRESENTATION_MODE.md`
- Modify: `docs/RELEASE_CHECKLIST.md`
- Modify: route and UI tests as required

- [ ] Change `/present` to select Presentation, `/` and `/advanced` to select Network Information, and `/study` to select RL Information at the governed-study section while preserving query parameters and deep links.
- [ ] Retain lightweight compatibility documents or redirects so bookmarked URLs and browser back/forward behavior remain correct.
- [ ] Remove legacy DOM/CSS/JS only after each route's parity test passes; do not combine cleanup with scientific or backend behavior changes.
- [ ] Run a repository search for `AI Advisor`, fourth-mode navigation, false probability/confidence language, and recorded-live ambiguity; replace only product-copy occurrences validated by tests.
- [ ] Run `py -m pytest -q` and require at least the starting 524 tests plus every new test to pass.
- [ ] Run frontend unit, integration, accessibility, visual, and performance suites.
- [ ] Run `git diff --check`.
- [ ] Confirm `git diff -- results/environment_v2_validation/manifest.json` is empty and its pre-implementation hash matches exactly.
- [ ] Confirm no learner, reward, scenario, training, experiment, frozen report, result, checkpoint, or other worktree file changed.
- [ ] Exercise all four compatibility URLs, all three modes, Guided Story, source switching, a live V1 session, configured V2 live demo or truthful unavailable state, paired comparison, recorded aggregate replay, development evidence, and final evidence.
- [ ] Update documentation with screenshots and current commands, then run all checks again.
- [ ] Request code review using `superpowers:requesting-code-review`; address only evidence-backed findings.
- [ ] Use `superpowers:verification-before-completion` immediately before the final claim.
- [ ] Commit: `feat(product): ship unified Dispatch Atlas experience`.
- [ ] Push the implementation branch only after the final verification record is captured.

## Prompt 2 completion record

The implementing agent must append a dated verification record to the pull request or task response containing:

- branch and final commit;
- exact commands and pass counts;
- screenshot/viewports inspected;
- supported and unavailable policy/environment pairs;
- protected manifest hash before and after;
- changed-file allowlist review;
- confirmation that results, learners, training, reward/scenario semantics, and other worktrees were untouched;
- any accepted risk, with owner and follow-up issue.

Do not mark Prompt 2 complete if any required source state can be mistaken for another, if a recorded trace renders invented link telemetry, if a policy output is mislabeled, if route compatibility is broken, or if the protected-file check is not clean.
