# Release notes

Branch `final`, built on the Part 1 tip `5ed6d16`
(`feat/live-v2-foundation-presentation-mode`).

---

## What Part 2 added

### 1 · The paired decision comparison

The comparison lane used to render a four-column summary table. It now renders
the full side-by-side comparison: a verdict with a signed gap, per-lane decision
cards with the validator's rejection reason, an interval metric table with a
leader per row, cumulative movement counters kept in three attributions, and the
first interval at which the two lanes decided differently.

The refusal path was strengthened at the same time. When the pairing proof
breaks, the payload now carries **no** verdict, metric row or gap — not a
disabled one, not a caveated one: the keys are absent, and the components that
render them are never called.

New: `mplssim/product/comparison.py`. `serialize.comparison_state` delegates to
it, so every existing caller is unchanged.

### 2 · The results surface

Three record classes — the live run, runs kept by Reset run, and the closed V2
study — in three sections that share no table and no aggregate. The study
section holds a pointer and a reason, never a transcribed figure: the frozen
record has exactly one renderer, so it cannot drift into a second copy.

New: `mplssim/product/results.py`, `frontend/js/product/results.js`,
`GET /api/product/results`.

### 3 · Retained runs, with a decided lifetime

Reset run archives to the session. Full reset hands the session's archive **and
the run that was on screen** to a process-level store, so returning to the
configuration form does not discard what you just watched. A restart drops
everything, deliberately — persisting a demonstration number is the first step
towards someone citing it. `GET /api/simulation/retained-runs` no longer 404s
without a session, since a full reset is exactly when you want to read it.

### 4 · The advisor fast-forward asymmetry, closed

Part 1 disclosed it in a response body. Part 2 makes it a decision the operator
makes and can count: `run-until` under advisor execution requires
`delegate: true`, the UI asks before sending it, and the stretch enters the
approval ledger as **one** `delegated_batch` — reported separately from
proposals, rendered as its own `delegation` timeline event, and counted in the
control panel.

Full reasoning: `docs/ADR-003-results-retention-and-delegated-fast-forward.md`.

### 5 · `/api/export/save-run` fixed under V2

It was broken. It called V1's `summarize_records`, which reads `jain_fairness`,
`p95_delay_ms`, `priority_sla_success`, `carried_mbps`, `reroutes`, `flaps`,
`frr_events` and `engine.path_change_count` — none of which the frozen V2
interval record has. It raised.

V2 now has its own summarizer over its own columns. Padding the V2 record with
zeros would have produced a row that looks like a V1 result and is not one, so
instead a V2 summary declares what it cannot measure. Every saved row carries
`environment_version`, `record_class` and `is_evidence: false`.

Also verified under V2 and pinned by tests: `/api/lsps`, `/api/links`,
`/api/metrics/history`, `/api/export/results`.

### 6 · Guided Story beats 3–11

Part 1 exercised beats 1–2 interactively. Two real defects surfaced when the
rest were walked:

- **Beat 8 did nothing.** It declared `advance: { kind: "approve" }` and the
  runner handled only `step`, `propose` and `runUntil`. Pressing Next advanced
  the copy — *"The move ran"* — without applying anything.
- **Beats 4 and 5 declared `select` and it was never applied**, so the copy
  named an object the topology did not highlight.

Both are fixed, and all eleven beats were then walked live end to end under V2
advisor pacing.

### 7 · Deferred details from the Part 1 handoff

- The RL rail said *"a live decision from the V1 runner"* under a V2 session. It
  now reads the environment and checkpoint from the actual context.
- Three mojibake `Â·` separators in `governed-study.js` are fixed.
- The moment rail drops from eight cells to four in audience view, with larger
  values, for a projector.
- Leaving audience view could drop keyboard focus onto `<body>` where the header
  toggle is hidden; focus now falls back to a landmark that always exists.

---

## One regression, found and fixed during the browser pass

Adding a second record shape to `advisor_history` broke
`/api/simulation/moment`: `timeline._advisor_events` assumed every record was a
proposal and raised `KeyError: 't_min'` on the delegated batch, taking down the
whole atomic moment read. Found by walking Guided Story in a real browser, not
by the test suite.

Fixed by giving the delegated batch its own event kind and putting `step`/`t_min`
on both record shapes. Pinned by
`test_a_delegated_batch_is_its_own_timeline_event_not_a_recommendation`.

---

## Verification

| Check | Result |
|---|---|
| Full suite (`py -m pytest -q`) | **811 passed, 0 failed** (Part 1: 758; baseline: 654) |
| New Part 2 suites | 46 tests across comparison, results and V2 endpoints |
| JavaScript parse (`node --check`, all modules) | clean |
| Python parse (`ast.parse` over `mplssim/`, `server/`, `tests/`) | clean |
| V1 byte-identity against the audited base | passes unchanged |
| `git status --porcelain` under `results/`, `runs/`, `models/` | no change |
| Guided Story beats 1–11, live, V2 advisor pacing | all eleven, no errors |
| Paired V2 comparison, live | verdict, metrics, movement, divergence all correct |
| Reset run → full reset → retained runs, live | archive survives full reset |
| Delegated fast-forward, live | 45 delegated intervals vs 3 individual approvals, reported separately |

### The governance allowlist

`tests/test_v1_v2_compatibility.py` carries an explicit allowlist that every
authorized stage extends in a reviewable commit. Part 2's four new source files,
three new test files, two new docs and this release directory were added with
the reason. **No assertion was relaxed**, and the substantive guard —
`test_models_results_figures_and_v1_configs_are_byte_identical_to_the_base` —
passes unchanged. Part 2 modified only files the list already authorized.

---

## Scientific confirmation

No training, tuning, evaluation, checkpoint selection or reselection, holdout
environment access, holdout-informed decision, evidence mutation or
scientific-semantics change was performed.

Environment semantics, observations, actions, masks, rewards, candidate paths,
simulator topology, traffic, failures, algorithms, hyperparameters, governed
checkpoints, frozen seeds and internal IDs are unchanged. No telemetry,
probability, explanation, counterfactual, reward or replay link utilization is
fabricated. The frozen study truth is rendered only from governed evidence, and
`mplssim/product/results.py` contains no transcribed study figure by
construction — a test asserts it.

---

## Known limitations carried forward

1. PPO entropy and value estimates are still not exposed by the live runner;
   both report their reason rather than showing a number.
2. Recorded replay has no per-link utilization; the reference topology is shown
   with the reason.
3. V2 has no manual traffic multiplier or burst injector. Both endpoints return
   `409` with the reason rather than emulating one.
4. Retained runs do not survive a server restart. This is a decision, not a gap
   — see ADR-003 §2.
5. The V1 `provenance-word` element id is replaced on first render
   (pre-existing).
6. `mplssim/product/checkpoints_v2.py` is the one product module permitted to
   import the learner classes, for inference only. A test asserts that no
   product module can train or save.
