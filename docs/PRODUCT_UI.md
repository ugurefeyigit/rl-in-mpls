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

| Stamp | Executes | Link telemetry | Intended use |
|---|---:|---:|---|
| LIVE | yes | yes | Running V1 simulation and synchronized comparison |
| RECORDED | no | no | Immutable V2 interval aggregates |
| DEVELOPMENT | no | no | Selection-stage evidence, explicitly not holdout |
| FINAL EVIDENCE | no | no | Frozen one-shot holdout conclusion |

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
