# Three-mode product UI

## Audiences and routes

| Primary mode | Audience | Compatibility routes |
|---|---|---|
| Presentation | Presenter and network-aware audience | `/present` |
| Network Information | MPLS engineer and operator | `/`, `/advanced` |
| RL Information | ML engineer, reviewer, and study auditor | `/study` |

The routes all serve `frontend/app.html`. Client routing selects a mode, source,
RL subview, workflow, and optional selection. Guided Story belongs to
Presentation. Governed Study and Model Provenance are secondary RL Information
views, not primary modes.

## Record types

| Stamp | Plain wording in the setup path | Executes | Link telemetry | Intended use |
|---|---|---:|---:|---|
| LIVE | Live simulation | yes | yes | Running V2 (or explicitly V1) simulation and synchronized comparison |
| RECORDED | Recorded episode playback | no | no | Immutable V2 interval aggregates |
| DEVELOPMENT | Pilot and continuity results (before the holdout) | no | no | Selection-stage evidence, explicitly not holdout |
| FINAL EVIDENCE | Final study result (frozen, read-only) | no | no | Frozen one-shot holdout conclusion |

The bare stamp words are for the provenance ledger, where a projector-legible
token matters. Everywhere a first-time user makes a choice, the plain wording is
used instead, and the three evidence records live in a **Study evidence and
results** region of the control panel — never beside the scenario or model
pickers. None of them can be run, compared live or chosen as a model.

## The control panel

Presentation Mode has one persistent left column. Everything that configures or
drives a run is there, top to bottom, in the order a newcomer needs it:

1. Environment — **V2 by default**, the governed study environment.
2. Scenario — a real repository scenario, by its display name.
3. Seed — validated; frozen holdout seeds are refused with the reason.
4. Execution — automatic, or manual/advisor approval.
5. Controller A — a learner or a baseline that genuinely runs in that
   environment. An unavailable one is disabled with its verification reason.
6. Optional comparison and Controller B.
7. Checkpoint root (V2) — 42 by default; see the neutral rule below.
8. Speed, then Start run.
9. Run it: play/pause, step, skip to next event, stop, **reset run**,
   **full reset**.
10. Approve or reject, only in advisor execution.
11. Guided Story entry with manual and automatic pacing.
12. Results — refresh, save this run, and how many earlier runs are kept.
13. Study evidence and results — the frozen records, in their own region.

Nothing that starts or steers a run lives in the header, a bottom bar or a
drawer. The bottom presenter cockpit is gone.

**Reset run** recreates the same environment, scenario, seed and controllers at
step zero and retains the run it replaced. **Full reset** stops the runners,
hands the session's archive *and the run that was on screen* to the process
store, clears active and transient UI state, closes Guided Story and the advisor
workflow, and returns to the initial configuration. Neither mutates a model, a
checkpoint or any evidence artifact. A server restart drops every retained run;
see [ADR-003](ADR-003-results-retention-and-delegated-fast-forward.md).

**Skip to next event** is the one control that does not respect per-action
approval: it applies the controller's own actions for a stretch of intervals in
one gesture. Under advisor execution it asks the operator to delegate that
stretch before running it, and the panel then shows a running count of intervals
delegated rather than approved. The server refuses an undelegated fast-forward
outright.

## The results surface

Presentation Mode carries one **Results** panel with three sections that share
no table and no aggregate: the live run, earlier runs kept in this session, and
the closed V2 study. The study section holds a pointer and a reason, never a
transcribed figure — the frozen record has exactly one renderer, under RL
Information → Governed Study, reading its own artifacts. The comparison lane
sits directly above it and shows a verdict only while the pairing proof holds;
both surfaces are documented in
[RESULTS_AND_COMPARISON.md](RESULTS_AND_COMPARISON.md).

**Audience view** hides the working chrome. Its exit control is deliberately
rendered outside that chrome, is pinned visible at every viewport including
fullscreen, and `Escape` always leaves audience view before it leaves
fullscreen. Focus returns to the toggle that opened it. Nothing reloads.

## Automatic execution versus advisor approval

In automatic execution the policy acts. The card beneath the map explains a
**completed decision**; there is no proposal, no approval affordance and no
fabricated preview of something that already ran. In advisor execution the
proposed action is held — `Step` produces a proposal rather than advancing the
clock — and only Approve or Reject moves the run on. Automatic Guided Story
playback stops at every recommendation and waits.

A fast-forward (`Skip to next event`, and the Guided Story beats that use one)
applies the controller's own actions for that stretch without individual
approval, and both the API response and the story copy say so.

Changing source clears incompatible live state and invalidates outstanding
source requests. Late live responses cannot repopulate recorded or evidence
views. Development and final evidence never render together.

## Network Information

Network Information combines the fixed topology with router/link/demand/path
inspection, traffic-class and condition filters, a risk-ordered demand table,
current-versus-prior values, alternate paths, and a restoration/change timeline.
Table and topology selection share one object selection. FRR is labeled as the
engine’s built-in protection and is not counted as a policy decision.

The simulator does not model packet headers, label-stack operations,
RSVP-TE/IGP convergence, control-plane timing, or a real operator network.
Those fields are absent or explicitly unavailable.

## RL Information

Decision Observatory renders the eight-stage pipeline:

`observation → action mask → policy output → selected action → safety validation → environment transition → reward → next observation`

It includes searchable observation groups, descriptive changed-feature ranking,
all 69 actions (`no-op + 17 × 4`), validator reasons, chosen action, runner-up,
no-op comparison, the complete authoritative reward component set, exact-sum
status, model/checkpoint provenance, and a clone-only Decision Lens.

MaskablePPO values are called action probabilities only when exposed. Masked
bandit values remain action scores or immediate-reward estimates. Changed
observations are not causal feature importance. Episode-mean no-op frequency
and step-pooled no-op share retain separate names and denominators.

## Topology and layout

The UI intentionally reuses the stable pre-redesign engineering schematic from
`configs/topology.yaml`, scaled only in the display response. It does not claim
geographic placement. City and role are primary; `PE1`, `P2`, and other internal
IDs remain secondary. One display-only waypoint preserves the legacy curved
`L18` clearance. No simulator topology, ID, link, path, or routing semantic is
changed.

Capacity uses discrete line weight. Utilization uses discrete states plus
printed values and pressure ticks. Failure, recovery, selected routes, current
paths, and alternates have non-color encodings. Zoom, fit, reset, mouse, touch,
keyboard, and the equivalent list view share the same selection.

## Data absence

An absent value is never replaced with a plausible estimate. The UI states why
recorded link telemetry, live V2, PPO entropy/value, a counterfactual, or a replay
episode is unavailable. Evidence load failures surface as named outages rather
than zero-filled charts.
