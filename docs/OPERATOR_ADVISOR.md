# Operator Advisor

The advisor turns the trained policy from an autopilot into a **recommender
that waits for a human**. It is the mode used in the live demonstration,
because "the model chose to move this flow, and here is what it expects to
happen" is a far more defensible claim than "the model runs the network".

---

## The flow

```
                    POST /api/advisor/propose
                              │
        policy prediction ────┤  (masked, deterministic)
        safety check ─────────┤  engine.validate_action, no mutation
        lookahead ────────────┘  two cloned engines, one interval each
                              │
                     session → PAUSED
                     pending_proposal set
                     WS broadcast {type:"advisor"}
                              │
              ┌───────────────┴───────────────┐
    POST /api/advisor/approve        POST /api/advisor/reject
    applies the exact action          applies a no-op
              └───────────────┬───────────────┘
                    both advance ONE interval
                    both append to advisor_history
                    WS broadcast {type:"tick"}
```

## Endpoints

| Endpoint | Behaviour |
|---|---|
| `POST /api/advisor/propose` | Generates a recommendation, pauses the session, sets `awaiting_decision`. **Idempotent** — calling it again returns the same pending proposal. 409 if the scenario has finished. |
| `POST /api/advisor/approve` | Applies exactly the proposed action, steps one interval, records the outcome. 409 if nothing is pending. |
| `POST /api/advisor/reject` | Applies a no-op, steps one interval, records the outcome. 409 if nothing is pending. |
| `GET /api/advisor/status` | `{pending, history[-20:], enabled}` — used to restore the card after a page reload. |

While a proposal is pending, `POST /api/simulation/resume` and
`/api/simulation/step` return **409**: the run cannot move until the operator
decides. Both UIs disable those buttons rather than letting the click fail.

## Proposals never mutate the engine

This is the property the whole mode rests on.

- The **safety check** calls `engine.validate_action(...)`, which is a pure
  predicate. It does not apply anything.
- The **lookahead** (`AlgoRunner.evaluate_action_vs_noop`) runs on
  `engine.clone()` — a deep copy — **twice**: once stepping an interval with
  the action applied, once stepping the same interval with no action. The live
  engine is never touched, so a proposal that is never approved leaves no
  trace in the simulation.
- Nothing is written to `metrics_history`, `action_log` or the reward totals
  until approve or reject is called.

`tests/test_state_machine.py` asserts that a proposal followed by no decision
leaves the step counter and metrics history unchanged.

## Predicted vs actual

The recommendation card shows two numbers for the busiest link, and they mean
different things:

- **Projected** — `lookahead.action.max_util`: the busiest link after replaying
  *this* interval on a copy, with the action applied.
- **Measured** — `actual.max_util`, recorded after approve/reject actually
  stepped the live engine.

They are usually close but rarely identical, because the live interval also
carries the traffic change that occurred during it, plus the paired
comparison controller's independent behaviour on its own engine. The card says
this in plain language; do not present the projection as a guarantee.

`lookahead.delta_max_util` is `action − noop` on the *same* cloned interval, so
it isolates the action's effect from the traffic change. That is the honest
number to quote for "what did the move buy us".

## What the history records

Each entry in `advisor_history` is the full proposal plus:

| Field | Meaning |
|---|---|
| `approved` | whether the operator accepted it |
| `applied_action` | the action index actually applied (`0` on reject) |
| `actual` | measured `max_util`, `mean_delay_ms`, `loss_ratio`, `sla_violations`, `delivered_ratio` |
| `reward` | the reward the interval earned |
| `operator_response_s` | wall-clock seconds between proposing and deciding |

`operator_response_s` is a demonstration statistic, not a performance claim —
it measures how long the presenter took to press a button.

## Using it from the UIs

**Presentation Mode** (`/present`) — the guided story calls `propose` at two
scripted moments; Approve/Reject are large buttons and the `A` / `R` keys.

**Engineering console** (`/`) — tick *Operator Advisor* before starting a
session, then use **Recommend / Approve / Reject** in the control rail. The
Scoreboard tab shows the pending proposal with the raw lookahead figures and
the technical action encoding (`D3 p0→p1`, action index, safety reason).

## Limitations

- The advisor recommends **one action per interval**, the same budget the
  policy had during training.
- `safety_ok:false` proposals are still offered. Approving one applies the
  policy's chosen action, and the engine's own constraint checker rejects the
  move — which is a useful thing to demonstrate deliberately.
- The lookahead is one interval deep. It says nothing about whether the move
  is still right ten minutes later.
- Action probabilities are read from the policy distribution when Torch makes
  them available; when it does not, the field is absent rather than estimated.
