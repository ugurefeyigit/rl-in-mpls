# ADR-003 — Results retention, class separation, and the delegated fast-forward

**Status:** accepted
**Supersedes:** nothing. Extends ADR-002 (unified three-mode product shell).
**Context:** Part 2 of the product build. The Part 1 handoff left three open
decisions; this record closes all three.

---

## 1. Three record classes, never merged

### Decision

The product recognises exactly three classes of number and reports them in
three separate sections that share no table, no chart and no aggregate:

| Class | What it is | May support a conclusion? |
|---|---|---|
| `live_demonstration` | the run on screen: one scenario, one seed, one pass, still moving | no |
| `retained_demonstration` | a run archived by *Reset run* inside this server process | no |
| `governed_evidence` | the closed V2 study: five holdout seeds per scenario, three training roots, evaluated once after the study closed to selection | **yes** |

`mplssim/product/results.py` emits all three. There is no function in that
module that takes two classes and returns one row, and `tests/test_part2_results.py`
asserts that no such function appears.

### Why

A live demonstration and a holdout result answer different questions. A
demonstration says *this is how the controller behaves*; a holdout result says
*this is what we established*. Averaging them, ranking them together, or even
placing them in one table produces a number that inherits the weaker one's
epistemic status while wearing the stronger one's appearance. That is the single
most damaging thing this surface could do, so the architecture makes it
impossible rather than discouraged.

### The study's numbers are not loaded into the results module

`results.py` emits a **pointer** to the governed routes (`/api/v2/*`) and the
reason the class stays separate. It does not read the frozen artifacts and
contains no transcribed figure. A literal copy of `18.221` in a second renderer
is a second version of the study's record, free to drift from the artifacts
silently. The frozen record is rendered in exactly one place —
`frontend/js/product/governed-study.js`, from the evidence API — and the results
surface links to it.

---

## 2. Retention lifetime

### Decision

| Action | What happens to the run |
|---|---|
| **Reset run** | archived on the session; readable at `/api/simulation/retained-runs` |
| **Full reset** | the session's archive *and the run that was on screen* are handed to a process-level store, so returning to the configuration form does not discard what the operator just watched |
| **Server restart** | everything is dropped |

Implemented as a module-level list in `results.py`. Nothing writes it to disk;
`tests/test_part2_results.py` asserts that the module contains no `open(`,
`write_text`, `to_csv` or `json.dump`.

### Why a restart drops it

Persisting a demonstration number to disk is the first step towards someone
citing it. These runs are unaudited, single-seed, single-pass, and were not
selected for anything. Surviving a restart would give them a durability that
their evidential status does not justify, and a stale `retained-runs` file found
six months later would be indistinguishable from a result.

An operator who *does* want a run kept has an explicit path:
`POST /api/export/save-run`, which writes a row labelled `record_class:
live_demonstration`, `is_evidence: false`. Keeping is then a decision someone
made, not a side effect of not restarting the server.

---

## 3. The advisor fast-forward asymmetry

### The problem Part 1 disclosed

Advisor execution holds every proposed action for an operator decision — except
`run-until`, which applies the controller's own actions for a stretch of
intervals in one gesture. Part 1 returned `approval_bypassed: true` with a note
and said so in the story copy, but the note lived in a response body the
operator never sees, and the approval history recorded the delegated intervals
as nothing at all.

### Options considered

1. **Refuse `run-until` under advisor execution.** Honest, and unusable: Guided
   Story needs to reach a scheduled failure, and stepping there one approval at
   a time is thirty clicks.
2. **Batch the stretch into a single approval.** Attractive but false — the
   operator would be approving actions the controller has not chosen yet, so the
   "approval" would be of an unknown thing.
3. **Keep it, with the Part 1 disclosure.** Rejected: an asymmetry that only
   appears in an API response is not disclosed to the person it affects.
4. **Require explicit delegation, and record it.** Chosen.

### Decision

`run_until` under advisor execution requires `delegate=true`. Without it the
session raises, and the API returns **409** with
`SimSession.DELEGATION_REQUIRED`, which names the asymmetry in full.

With it:

- the stretch runs;
- **one** `delegated_batch` record enters `advisor_history`, carrying
  `from_step`, `to_step`, `steps`, the condition and a note stating that no
  individual action was approved;
- the record is reported separately from proposals — `advisor_status()` returns
  `proposals`, `delegated_batches` and `delegated_intervals`;
- the timeline renders it as its own `delegation` event kind, never as a
  `recommendation`;
- the control panel shows a running count: *"45 interval(s) in this run were
  delegated, not approved individually."*

The UI asks once, at the point of the gesture, with the consequence spelled out.
Guided Story's own fast-forwards pass `delegate=true` and its beat copy already
states that those intervals were not approved one by one.

### Why this is the honest answer

The asymmetry is real and cannot be designed away — a fast-forward is
delegation. What was wrong was that the delegation was implicit. Now the
operator makes it, sees it, and can count it against the actions they approved
individually. The approval ledger stops overstating what the operator decided.

---

## 4. Consequences

- `POST /api/simulation/run-until` is a **breaking change for advisor sessions**
  only: automatic execution is unaffected, and advisor clients must add
  `delegate: true`. The refusal names the field.
- `advisor_history` now holds two record shapes. Every consumer must branch on
  `kind`. The `t_min`/`step` fields are present on both so a record can be
  placed on the timeline without knowing which it is. Missing this is what took
  `/api/simulation/moment` down during Part 2's browser pass; the regression is
  pinned by `test_a_delegated_batch_is_its_own_timeline_event_not_a_recommendation`.
- `POST /api/export/save-run` returns different fields for V1 and V2 rows. See
  `docs/RESULTS_AND_COMPARISON.md` § *Saving a run*.
- Nothing here trains, tunes, evaluates, selects a checkpoint, reads a holdout
  seed or writes under `results/`, `runs/` or `models/`.
