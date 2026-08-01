# Results and comparison

Two surfaces are documented here: the **paired comparison lane**, which shows two
controllers on one synchronized run, and the **results surface**, which shows
three classes of record without ever combining them.

Both were built in Part 2. The decisions behind them are recorded in
[ADR-003](ADR-003-results-retention-and-delegated-fast-forward.md).

---

## The paired comparison

### When it renders

`mplssim/product/pairing.py` answers *may we compare*. Only if it says yes does
`mplssim/product/comparison.py` produce anything comparative.

| Session | What the lane shows |
|---|---|
| one controller | "This session runs one controller. There is nothing to compare against." No verdict, no metric table. This is **not** a failed comparison. |
| two controllers, different environment versions | refused before any engine is read: the same action number addresses a different candidate path in V1 and V2 |
| two controllers, exogenous inputs disagree | refused, with the mismatched fields named |
| two controllers, proof holds | the full comparison below |

A refusal is the payload. When the proof is broken the response carries **no**
`verdict`, `metric_rows` or `divergence` key at all, and the lane components
that render them are not called. A reader must not be able to lift a number out
of a broken comparison.

### What it shows when it renders

- **Verdict** — cumulative operational return for each lane, the signed gap, and
  which lane leads. `is_evidence` is always `false`: one paired live run at one
  seed demonstrates behaviour, it does not establish a result.
- **Decision cards** — what each lane did this interval, its interval and
  cumulative return, and the validator's rejection reason when the environment
  refused the move.
- **Metric table** — the latest completed interval, lane by lane, with the gap
  and which lane is ahead. A metric with no better direction (`mean_util`) names
  no leader.
- **Movement table** — cumulative movement counters, kept apart.
- **Divergence** — the first interval in which the two lanes made different
  movement decisions, read from their own recorded histories.

### Three rules the comparison keeps

**No percentage on a signed score.** Operational return is signed; a ratio of
two signed numbers is meaningless. The gap is a subtraction, reported in score
units. There is no division of one return by another anywhere in
`comparison.py`, and a test asserts it.

**Controller, protection and recovery stay apart.** V2 keeps
`accepted_te_changes`, `frr_changes` and `recovery_restorations` in three
counters. They are reported as three counters. Summing them would credit the
policy with what the engine's fast reroute did.

**Endpoints are named only when the payload has them.** V1's decoded action
carries `src`/`dst`; V2's does not. The V2 lane says `D5 to candidate 1` rather
than inventing a city pair.

### Telling the lanes apart

Lane A carries the token **A** and a solid left border; lane B carries **B** and
a dashed one. Colour is a third signal, never the only one.

---

## The results surface

### Three classes

| Class | Grain | Evidence? |
|---|---|---|
| **Live run** | one scenario, one seed, one pass, still running | no |
| **Earlier runs kept in this session** | one scenario, one seed, one pass each, finished | no |
| **Closed V2 study** | five holdout seeds per scenario, three training roots, evaluated once | **yes** |

They appear in three sections with three headings and three tables. There is no
combined table, no leaderboard and no cross-class aggregate — not hidden, not
disabled: absent.

### Where the study's numbers come from

Not from this surface. The study section carries a pointer to `/api/v2/*` and
the reason the class stays separate. The frozen record is rendered in exactly
one place, `frontend/js/product/governed-study.js`, from the artifacts
themselves. A transcribed figure in a second renderer would be a second copy
free to drift.

### Retention

| Action | Effect |
|---|---|
| Reset run | the replaced run is archived on the session |
| Full reset | the session's archive **and the run on screen** move to the process store |
| Restart | everything is dropped |

Nothing is written to disk. To keep a run deliberately, use
`POST /api/export/save-run`.

---

## Saving a run

`POST /api/export/save-run` summarizes each lane **for its own environment**.

V1 and V2 record different interval columns. `summarize_records` in
`mplssim/experiments/runner.py` is a V1 summarizer: it reads `jain_fairness`,
`p95_delay_ms`, `priority_sla_success`, `carried_mbps`, `reroutes`, `flaps`,
`frr_events` and `engine.path_change_count`, and the frozen V2 record has none
of them. Under V2 it raised — this was the unverified item Part 1 flagged.

V2 now has `mplssim/product/run_summary.py`, over V2's own columns. The
alternative — padding the V2 record with zeros — would have produced a row that
looks like a V1 result and is not one. Instead a V2 summary declares what it
cannot measure:

```json
{
  "environment_version": "v2",
  "accepted_te_changes_total": 3,
  "frr_changes_total": 2,
  "not_measured": ["jain_fairness", "p95_delay_ms",
                   "priority_sla_success", "path_changes_per_demand"],
  "not_measured_reason": "The frozen V2 interval record does not carry these V1 quantities. They are reported as absent rather than padded with zeros.",
  "record_class": "live_demonstration",
  "is_evidence": false
}
```

A V1 row and a V2 row therefore have different fields, and each carries
`environment_version` so nothing downstream can average them by accident.

---

## API summary

| Route | Returns |
|---|---|
| `GET /api/simulation/comparison` | synchronization proof plus the full `detail` block (or the refusal) |
| `GET /api/simulation/moment` | the same `comparison` block inside the atomic moment |
| `GET /api/product/results` | the three record classes, in three sections |
| `GET /api/simulation/retained-runs` | session store plus process store; answers with no active session |
| `POST /api/export/save-run` | per-environment episode summaries |
| `POST /api/simulation/run-until` | gains `delegate`; advisor execution refuses without it (409) |

Verified under V2 in Part 2: `/api/lsps`, `/api/links`, `/api/metrics/history`,
`/api/export/results`, `/api/export/save-run`.
