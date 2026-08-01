# Part 2 handoff — comparison, results, and the release

Completes every task in `OPUS5_PART1_HANDOFF.md` § 9.

## 1. Branch, commit, worktree, status

| Field | Value |
|---|---|
| Branch | `final` |
| Base commit | `5ed6d16` (`feat/live-v2-foundation-presentation-mode`, Part 1 tip) |
| Worktree | `.claude/worktrees/final` |
| Tree status after commit | clean |

No existing worktree was reused, deleted or repurposed. `.worktrees/seed42`,
`.worktrees/continuity_v2`, `.worktrees/final_holdout_v2`,
`.worktrees/post_study_productization`, `.worktrees/three_mode_ui` and
`.claude/worktrees/rl-mpls-ui-part1-c51fff` are untouched.

## 2. Protected manifest hash

`results/environment_v2_validation/manifest.json` was never read for
modification, staged, reverted or normalized.

| Worktree | SHA-256 before | SHA-256 after |
|---|---|---|
| `final` | `5227C91A1990C52101A510BC7115C049E138D5B34C2CDD649C66C8AA726A979D` | identical |

Prohibited-artifact audit: `git status --porcelain -uall` shows **no** change
under `results/`, `runs/` or `models/`, and no `.zip` or `.pt` file.

## 3. Part 1 § 9 tasks, one by one

| # | Task | Outcome |
|---|---|---|
| 1 | Comparison presentation | **Done.** `mplssim/product/comparison.py` + rewritten `comparison-lane.js`: verdict, lane cards, metric table, movement counters, first divergence. Refusal path hardened — a broken proof emits no verdict, metric or gap at all. |
| 2 | Retained runs | **Done.** Consumed by the results surface. Lifetime decided: session → process on full reset → dropped on restart (ADR-003 § 2). |
| 3 | Cross-mode results system | **Done.** `mplssim/product/results.py`, `GET /api/product/results`, `results.js`. Three record classes, three sections, no aggregate anywhere. |
| 4 | Final evidence integration | **Done.** The study section carries a pointer and a reason, never a transcribed figure. A test asserts no frozen number appears in `results.py`. Three mojibake separators in `governed-study.js` fixed. |
| 5 | Advisor fast-forward asymmetry | **Decided and closed.** Explicit delegation required; recorded as one `delegated_batch` in the ledger, its own timeline event kind, counted in the panel (ADR-003 § 3). |
| 6 | `/api/export/save-run` and `/api/lsps` under V2 | **Verified — save-run was broken and is fixed.** V2 now has its own summarizer. `/api/lsps`, `/api/links`, `/api/metrics/history` and `/api/export/results` verified and pinned. |
| 7 | Guided Story beats 3–11 under V2 advisor pacing | **Done, and two real defects fixed.** Beat 8's `approve` advance was unhandled; beats 4–5's `select` was never applied. All eleven beats then walked live end to end. |
| 8 | `final/` release assembly | **Done.** `final/README.md`, `RUNNING_IT_AGAIN.md`, `OPERATING_THE_UI.md`, `RELEASE_NOTES.md`. |
| 9 | Deferred detail | **Addressed.** RL rail no longer hardcodes V1; moment rail drops to four cells in audience view; audience-exit focus no longer falls to `<body>`. The `provenance-word` id replacement is unchanged and still disclosed. |

## 4. Files changed

### New modules

| File | Purpose |
|---|---|
| `mplssim/product/comparison.py` | The paired decision comparison, and its refusals |
| `mplssim/product/results.py` | Three record classes, the process store, no cross-class aggregate |
| `mplssim/product/run_summary.py` | Per-environment episode summaries for save-run |
| `frontend/js/product/results.js` | The three-section results surface |
| `tests/test_part2_comparison.py` | 15 tests: shape, refusals, verdict, movement, divergence, surface |
| `tests/test_part2_results.py` | 17 tests: class separation, retention lifetime, no-persistence |
| `tests/test_part2_v2_endpoints.py` | 15 tests: save-run under V2, lsps, delegation, the timeline regression |
| `docs/ADR-003-results-retention-and-delegated-fast-forward.md` | The three decisions and their reasoning |
| `docs/RESULTS_AND_COMPARISON.md` | Reference for both surfaces |
| `final/{README,RUNNING_IT_AGAIN,OPERATING_THE_UI,RELEASE_NOTES}.md` | The release |

### Modified

| File | Change |
|---|---|
| `server/session.py` | `run_until(delegate=...)`, `DELEGATION_REQUIRED`, the `delegated_batch` ledger record, `kind` on proposals, `advisor_status` splits proposals from batches |
| `server/main.py` | `RunUntilRequest.delegate`; `stop` hands over to the process store; `retained-runs` reads both stores and no longer 404s; `save-run` is per-environment and 409s before the first interval |
| `server/product_api.py` | `GET /api/product/results` |
| `mplssim/product/serialize.py` | `comparison_state` delegates to `comparison.py` |
| `mplssim/product/timeline.py` | `delegation` event kind; a delegated batch is never a recommendation |
| `frontend/js/product/comparison-lane.js` | Full rewrite |
| `frontend/js/product/main.js` | `fastForward` with the delegation confirm, results loading, beat `approve` + `select`, `BEATS.length`, delegated batches excluded from the card |
| `frontend/js/product/control-panel.js` | Section 5 · Results; delegation disclosure in section 3; evidence heading numbered |
| `frontend/js/product/modes/presentation.js` | Results panel, delegation notice, projector rail density |
| `frontend/js/product/modes/rl.js` | Rail reads the live environment instead of hardcoding V1 |
| `frontend/js/product/shell.js` | `restoreFocusAfterAudience`, results handlers |
| `frontend/js/product/store.js` | `data.results`, `delegation`, `savedRun` |
| `frontend/js/product/adapters/live-v1.js` | `results`, `saveRun`, `runUntil(…, delegate)` |
| `frontend/js/product/governed-study.js` | Three mojibake separators |
| `frontend/css/presentation-mode.css` | Comparison, results and projector-density rules |
| `docs/{API,ARCHITECTURE,PRODUCT_UI}.md`, `README.md` | Documented above |
| `tests/test_presentation_controls.py`, `tests/test_v1_v2_compatibility.py` | Story-pacing contract; allowlist extended with reasons |

### API surface added or changed

```
GET  /api/product/results            three record classes, no session required
GET  /api/simulation/retained-runs   session + process store, no session required
GET  /api/simulation/comparison      now carries the full `detail` block
POST /api/simulation/run-until       gains `delegate`; advisor refuses without it (409)
POST /api/export/save-run            per-environment summaries; 409 before step 1
POST /api/simulation/stop            hands the archive to the process store
```

## 5. Test results

| Run | Command | Result |
|---|---|---|
| Full suite | `py -m pytest -q` | **813 passed, 0 failed** in 109.6 s |
| Part 1 baseline | — | 758 — exceeded by 55 |
| Original baseline | — | 654 |
| New: comparison | `py -m pytest tests/test_part2_comparison.py -q` | 15 passed |
| New: results | `py -m pytest tests/test_part2_results.py -q` | 17 passed |
| New: V2 endpoints | `py -m pytest tests/test_part2_v2_endpoints.py -q` | 15 passed |
| V1/V2 compatibility | `py -m pytest tests/test_v1_v2_compatibility.py -q` | passes; byte-identity guard unchanged |
| JavaScript parse | `node --check` over all 44 modules | all pass |
| Python parse | `ast.parse` over `mplssim/`, `server/`, `tests/` | all pass |

The governance allowlist was extended, with reasons, for Part 2's four new
source files, three test files, two docs and the release directory. **No
assertion was relaxed.** Part 2 modified only files the list already authorized,
so `ALLOWED_MODIFIED_FILES` needed no new entry.

## 6. Browser QA

Server: `python -m uvicorn server.main:app --port 8222`, route `/present`.

Exercised live, end to end:

- **Paired V2 comparison** — started `masked_bandit` + `greedy` on
  `demo_evening`/42 from the panel, stepped four intervals. Verdict read
  *"greedy leads this run by 0.5793 points of cumulative operational return"*;
  divergence read *"first decided differently at step 4: A moved a demand, B
  made no TE change"*; metric rows and the three-attribution movement table
  correct.
- **Single-controller session** — renders *"This session runs one controller.
  There is nothing to compare against"*, with no verdict and no metric table.
- **Retention** — Reset run → *"1 kept in this session"*; Full reset → *"0 kept
  in this session · 1 kept from earlier sessions"*, archive intact.
- **Guided Story, all eleven beats**, V2 advisor pacing, no errors. Beat 8 now
  applies the action (*"The move ran … the interval scored 0.792"*); beat 9 hit
  the real Kayseri–Samsun failure; beat 11 opened the governed conclusion
  rendering 18.221 / 9.036 / −2.327 / 1.107 from the frozen artifacts.
- **Delegation** — 45 delegated intervals against 3 individual approvals,
  reported separately by `/api/advisor/status`, shown as three `delegation`
  timeline events, and disclosed in the panel.
- **Audience view** — exit control visible, `Escape` leaves it, rail drops to
  four cells at projector density.

Browser console: no errors. Server log: no application errors after the
regression below was fixed.

### The one regression, found here and fixed

Adding a second record shape to `advisor_history` broke
`/api/simulation/moment`: `timeline._advisor_events` assumed every record was a
proposal and raised `KeyError: 't_min'` on a delegated batch, taking the whole
atomic moment read down with it. The test suite did not catch it; walking the
story in a real browser did. Fixed by giving the batch its own event kind and
putting `step`/`t_min` on both shapes, and pinned by
`test_a_delegated_batch_is_its_own_timeline_event_not_a_recommendation`.

## 7. Scientific confirmation

No training, tuning, evaluation, checkpoint selection or reselection, holdout
environment access, holdout-informed decision, evidence mutation or
scientific-semantics change was performed. Environment semantics, observations,
actions, masks, rewards, candidate paths, simulator topology, traffic, failures,
algorithms, hyperparameters, governed checkpoints, frozen seeds and internal IDs
are unchanged. No telemetry, probability, explanation, counterfactual, reward or
replay link utilization is fabricated. The frozen study truth is rendered only
from governed evidence, and `mplssim/product/results.py` deliberately does not
load it — a test asserts no frozen figure appears there.

## 8. Remaining limitations

Carried forward from Part 1, all still disclosed in the product:

1. PPO entropy and value estimates are not exposed by the live runner.
2. Recorded replay has no per-link utilization.
3. V2 has no manual traffic multiplier or burst injector; both endpoints 409.
4. Retained runs do not survive a restart — a decision, not a gap (ADR-003 § 2).
5. The V1 `provenance-word` id is replaced on first render (pre-existing).

## 9. Commands

```bash
python -m uvicorn server.main:app --port 8000
```

```bash
py -m pytest -q
```

```bash
py -m pytest tests/test_part2_comparison.py tests/test_part2_results.py tests/test_part2_v2_endpoints.py -q
```

Operator documentation is in `final/`. Start with `final/README.md`.
