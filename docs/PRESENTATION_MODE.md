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
