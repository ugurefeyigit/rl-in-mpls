# Presentation Mode

A second frontend at **`/present`**, built for a live audience that knows
nothing about reinforcement learning or MPLS. It is not a reskin of the
engineering console — it answers different questions, in plain language, with
city names instead of router IDs.

The engineering console stays at **`/`** (or `/advanced`); the two link to each
other from the header.

---

## What it shows

| Region | Content |
|---|---|
| Header | Scenario display name, network clock (HH:MM), story-phase chip, session state, connection dot, fullscreen, print, link to the engineering view |
| Incident spine | One block per completed five-minute interval, coloured by that interval's **peak** link utilization; notches mark scheduled events; the white marker is "now" |
| Map | Large Cytoscape topology, city labels only, utilization colours, failed links dashed red with a ✕, the advisor's proposed route highlighted in violet |
| KPI cards | Overall network score · Busiest link right now · Services with SLA problems · Traffic delivered · Route changes |
| Recommendation | Appears only while a proposal is pending: plain-language headline, old vs proposed route, why, expected effect from the real lookahead, safety-check result, Approve / Reject |
| Comparison | AI Advisor vs the comparison controller, with one honest sentence of interpretation |
| Story timeline | Threshold crossings, failures and recoveries, proposals and operator decisions, in city names |
| Published results | Per-scenario winner table read from `results/eval_stats.csv` |

### Two different "busiest link" numbers

Both are correct and both are labelled:

- **KPI card and story text** use the *instantaneous* busiest link from the
  snapshot — the same value the map is coloured by. This keeps the headline
  number, the map and the narration from ever disagreeing on screen.
- **The incident spine and the engineering console** use
  `metrics.max_util`, the *peak within* each five-minute interval. It is
  higher, because it catches sub-interval spikes.

---

## Launching

```bash
python scripts/demo.py
```

Starts the backend with `ALLOW_TRAINING=false`, creates the fixed demo session
(`demo_evening`, RL vs greedy, seed 42, advisor mode, paused at t=0) and opens
`/present`.

Other forms:

```bash
python scripts/demo.py --advanced
```

```bash
python scripts/demo.py --allow-training
```

```bash
python -m uvicorn server.main:app --port 8000
```

---

## Controls

The presenter bar sits under the map.

| Control | What it does |
|---|---|
| Start Guided 5-Minute Story | Runs the scripted sequence — see [Guided story](#guided-story) |
| Pause / Resume | Free-runs the clock at the session speed, or stops it |
| Next event | Fast-forwards to the next scripted event **and through it** (see note below) |
| Approve / Reject | Resolves a pending recommendation |
| Fail backbone link / Recover link | Manual intervention on the selected link |
| Reset story | Rebuilds the same run from t=0 with the identical configuration |
| Compare against | Greedy (default) or fixed routing; changing it restarts the run |
| Scaled national view (10×) | Display-only scale — see [Scaling disclosure](#scaling-disclosure) |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Pause / resume — or dismiss the story card and continue |
| `→` | Next event — or dismiss the story card and continue |
| `A` | Approve the pending recommendation |
| `R` | Reject the pending recommendation |
| `F` | Fail the selected link |
| `Esc` | Close the story card, or leave fullscreen |

Shortcuts are ignored while a form control has focus.

### Why "Next event" takes one extra interval

The engine applies scripted link events over the half-open window
`[interval_start, interval_end)`. `POST /api/simulation/run-until` with
`condition:"next_event"` stops as soon as the clock *reaches* the event time —
at which point the event has not been applied yet. Presentation Mode therefore
runs one further interval so the audience lands on the far side of the event,
with the link actually down. This is a frontend decision; the API contract is
unchanged.

---

## Scaling disclosure

The **Scaled national view (10×)** toggle multiplies **only** displayed traffic
volumes and link capacities by ten. It mirrors `mplssim/display.py :: scale_mbps`
and the JS constant `DISPLAY_SCALE` in `frontend/js/fmt.js`; the two are pinned
together by `tests/test_presentation.py::test_js_display_scale_matches_python`.

Because loads and capacities scale identically, **utilization, delay, loss, SLA
counts, actions and rewards are unchanged**. While the toggle is on, a banner
stating exactly that stays visible, and the sentence is repeated in the printed
summary.

The topology itself is fictional. The footer of both UIs carries:

> Fictional scaled national backbone for demonstration — not a real operator topology.

---

## Guided story

See [PRESENTATION_SCRIPT.md](PRESENTATION_SCRIPT.md) for what to say. The
sequence is:

1. Intro card
2. Fast-forward until the busiest link crosses 85 %
3. Ask the advisor → **wait for Approve or Reject**
4. Predicted vs measured card
5. Fast-forward through the live-event surge (t=180)
6. Fast-forward through the backbone failure (t=195) → fast-reroute explanation
7. Ask the advisor again → **wait for Approve or Reject**
8. Fast-forward through the repair (t=240)
9. Final comparison card

Every step reads real backend state. Narration describes what the run actually
produced — for example, on seed 42 the network is already near capacity at
17:00, and the card says so rather than claiming a build-up the audience did
not see.

**Reloading the page mid-story drops the script** (the session and all data
survive; only the step pointer is lost). Continue with *Next event* and the
Approve/Reject buttons, or press *Start Guided 5-Minute Story* to restart.

---

## Failure handling

- **Connection lost** — the dot next to the clock turns red and the page
  reconnects with backoff. A reconnect replays the current snapshot; a
  recommendation that is still pending stays on screen.
- **Session error** — a full-screen overlay states that the run has halted and
  shows the backend message, the scenario and the interval. Nothing on screen
  is stale.
- **A rejected action** — every intervention reports back from the server's
  `changed` flag, so "already failed" never masquerades as a fresh failure.

## Print / Save as PDF

The header button renders a summary card from live state: final scores, peak
utilization, SLA totals, delivered ratio, route changes and flaps for both
controllers, plus the full story timeline and the disclaimer. Use the browser's
"Save as PDF" destination.


---

## The third surface: `/study`

Presentation Mode and the engineering console both drive a **live** simulation
session. The study surface does not: it is a read-only record of the **closed**
V2 study, served from the committed evidence files.

| Surface | Drives a session | Data source |
|---|---|---|
| `/` and `/advanced` | yes | live `SimSession` |
| `/present` | yes | live `SimSession` |
| `/study` | **no** | `results/v2_*` via `/api/v2/*` |

Use it when the question is *what did the study find and can I trust it*, rather
than *what does the controller do right now*.

### What it shows

1. **Verdict** - the frozen conclusions, including both halves of the planning
   statement, which always appear together.
2. **Final holdout** - the five-method aggregate and the per-root learner
   comparison. Learner rows are the mean of three training-root means; baselines
   have no training root and ran once.
3. **Scenarios** - the seven-scenario comparison. The one scenario PPO wins
   points its bar the other way, so the negative result is the most visible thing
   on the chart rather than a footnote.
4. **Operations and churn** - delivery, SLA, utilization, congestion, delay,
   loss, reroutes, reversals, flaps, moved bandwidth, dwell, TE changes, FRR,
   disconnections, restorations, the 12-component reward breakdown with its
   exact-sum residual, and the action distribution.
5. **Development** - a visually distinct region with a permanent
   *development / continuity - not holdout evidence* ribbon, carrying the
   learning curves and checkpoint selection. Nothing here may be averaged with
   the holdout, and the two never arrive from the same API route.
6. **Provenance** - safety and integrity counters, the six checkpoints with their
   payload and sidecar hashes and source bindings, runtime and device, artifact
   locations, and the invalidated / superseded / failed / repaired run
   disclosures behind progressive disclosure.
7. **Recorded replay** - preserved episodes played back from the traces the
   one-shot evaluation wrote.

### Two figures that look like one

Where two published statistics share a name, both are shown with their grain
named. **No-op share** has a pooled-step form (87.09% bandit) and an
episode-mean form (82.10%); the final-holdout report quotes the pooled one.
**Wall time** has a whole-runner form (152.093 s, including baselines and setup)
and a six-checkpoint-evaluations form (115.213 s). Neither pair is
interchangeable. See [V2_EVIDENCE_AUDIT.md](V2_EVIDENCE_AUDIT.md).

### Replay is recorded, never live

Every replay payload is marked `kind: "recorded_replay"` and `live: false`, and
the page refuses to render anything that is not. Replay never runs a controller
or evaluates a checkpoint.

The step traces are large and live outside Git. Set `V2_FULL_ARTIFACTS` to the
directory named in `results/v2_final_holdout/manifest.json` under
`full_artifact_path`. Without it the catalogue still lists all 315 episodes and
explains how to configure the path.

The traces carry aggregate utilization rather than per-link utilization, so
replay shows the real per-interval operational record - reward, actions, busiest
link, delay, loss, SLA - and deliberately does not animate a topology it has no
data for.

### If nothing loads

A missing or inconsistent artifact is reported as an outage, not as zeros: the
section says what failed and why, and the page shows its empty state. That is
the intended behaviour - the surface will not render an approximate number.
