# UI acceptance tests — manual checklist

What the automated suite cannot assert: that the pages *look* right and behave
correctly under a presenter's hands. Run this before presenting.

```bash
python scripts/demo.py --no-browser
```

Then open `http://127.0.0.1:8000/present` and `http://127.0.0.1:8000/advanced`.
Keep the browser console open for the whole pass — **zero errors** is a
criterion, not a nicety.

Automated coverage that backs these up lives in `tests/test_presentation.py`
(page smoke, element IDs, no external assets, module graph, display-scale
constant, websocket reconnect, no-training-on-launch, benchmark honesty).

---

## A · Advanced console (`/`)

| # | Criterion | How to verify |
|---|---|---|
| A1 | City labels on the map | Nodes read *İstanbul*, *Ankara*, … — no `PE1` visible |
| A2 | Internal ID reachable | Hover a node → tooltip shows `İstanbul (PE1, PE_IN)` |
| A3 | Link dropdown | Reads `Ankara–Kayseri link (P2–P5, L11) — 2.0 Gbps` |
| A4 | Demand dropdown | Reads `İstanbul → Erzurum video traffic (D2)` |
| A5 | Scenario dropdown | Shows display names; description appears underneath |
| A6 | Demands table routes | Route column is a city chain, ID in the dimmed line below the name |
| A7 | Decision tape | Plain-language sentence with city names; `D2 p0→p3` only in the dimmed technical column |
| A8 | Number formats | Rates `740 Mbps` / `1.2 Gbps`; utilization `128%`; loss `0.82%`; delay `21.4 ms`; clock `19:10` |
| A9 | Disclaimer | Visible at the right end of the legend strip |
| A10 | Presentation Mode link | Header button opens `/present` |

### State machine and buttons

| # | Criterion | How to verify |
|---|---|---|
| A11 | State chip | Reflects `idle` / `running` / `paused` / `completed` / `error` with distinct colours |
| A12 | Pause disabled unless running | Before start: greyed. After Resume: enabled |
| A13 | Step disabled while running | Press Resume → Step greys out; Pause → Step enabled |
| A14 | Resume disabled when completed | Run a short scenario to the end → Resume greys out |
| A15 | Approve/Reject disabled with no proposal | Greyed until *Recommend* is pressed |
| A16 | Recommend disabled while pending | After *Recommend*, it greys out until you decide |

### Interventions report what actually happened

| # | Criterion | How to verify |
|---|---|---|
| A17 | Failure confirms with real counts | Fail `L20` → toast reads `… failed — N fast-reroute move(s)` |
| A18 | Repeat failure says "already" | Fail `L20` again → toast reads `… was already failed — no change` and is styled as an error |
| A19 | Recovery confirms | Recover → toast names the link and remaining failed count |
| A20 | Errors are visible | Press Step while running → red toast with the 409 detail |
| A21 | Reset confirms preserved config | Reset → toast lists scenario, controllers, seed, model, safety filter, speed |

### Panels

| # | Criterion | How to verify |
|---|---|---|
| A22 | Scoreboard is live | Scoreboard tab updates each interval: total reward, mean/interval, busiest link, peak, SLA now, demand-interval SLA total, delivered, route changes, flaps |
| A23 | Delta is absolute points | Compare mode → header reads `… ahead by N reward points`, **never** a percentage |
| A24 | Delta colouring | Green ahead / red behind / neutral grey when \|Δ\| < 5 |
| A25 | Churn warning | Run RL on `full_day` past ~12 intervals → amber badge about high route churn appears |
| A26 | Benchmark table | Benchmark tab shows per-algorithm 5-seed table + winner + the one-seed caveat |
| A27 | Benchmark for demo scenario | With `demo_evening` selected, the panel says it was not part of the published evaluation and shows the cross-scenario table |
| A28 | Events tab | Populates from `/api/events` and refreshes while the tab is open |
| A29 | SLA terminology | Table headers and tooltips say "demand-interval SLA violations" |
| A30 | Training gate | Training tab: pressing *Start training job* opens a confirmation dialog first |
| A31 | Training disabled banner | Launched via `scripts/demo.py`: banner reads "Training is disabled during presentation mode." and the button is greyed |

---

## B · Presentation Mode (`/present`)

| # | Criterion | How to verify |
|---|---|---|
| B1 | City labels only | No router ID anywhere on the default view |
| B2 | Legible at distance | Stand back 3 m: clock, phase chip and KPI values all readable |
| B3 | Phase chip tracks the story | `Normal` → `Traffic rising` → `Congestion detected` → `Recommendation ready` → `Change applied` → `Incident` → `Recovery` → `Complete` |
| B4 | Failed link rendering | After the failure: dashed red with a ✕ on the map |
| B5 | Legend | Five utilization colours + failed + recommended route |
| B6 | Disclaimer | Right end of the legend strip |
| B7 | Incident spine | One block per completed interval, coloured; three event notches; white "now" marker advances |

### KPI cards

| # | Criterion | How to verify |
|---|---|---|
| B8 | Five cards | Overall network score, Busiest link right now, Services with SLA problems, Traffic delivered, Route changes |
| B9 | Busiest link is named | Reads e.g. `98%` with `Adana–Malatya — close to capacity` and `Carrying 490 Mbps of 500 Mbps` |
| B10 | No fake zeros | Before the first interval completes, score and delivered show `—`, not `0` |
| B11 | Card matches the map | The busiest-link percentage matches the colour of the reddest link |
| B12 | Glossary tooltips | Hover "What is this?" → definition from `/api/display.glossary` |

### Recommendation card

| # | Criterion | How to verify |
|---|---|---|
| B13 | Appears only when pending | Hidden otherwise |
| B14 | Plain-language headline | e.g. *Move İzmir → Erzurum bulk data traffic away from the congested İzmir–Ankara corridor* |
| B15 | Old → new route | Both shown as city chains; old struck through, new in violet |
| B16 | "Why" names a link on **this** route | Cross-check the named corridor against the *Now goes via* chain — it must appear there |
| B17 | Expected effect is real | Shows `117.3% → 103.6%` and states it came from replaying an interval on a copy |
| B18 | Safety result shown | Green tick or red cross with the engine's reason |
| B19 | Proposed route on the map | The violet highlight follows the proposed chain |
| B20 | Technical footer | Proposal number, interval, action index, `D3 p0→p1`, policy confidence |
| B21 | Buttons are big | Approve / Reject usable from a lectern |

### Comparison and benchmark

| # | Criterion | How to verify |
|---|---|---|
| B22 | Both controllers | Score, busiest now, peak, SLA problems, delivered, route changes |
| B23 | Honest verdict sentence | When greedy leads, it says so explicitly and calls it a genuine result |
| B24 | Never a percentage delta | Verdict quotes absolute reward points |
| B25 | Comparator switch | Changing it warns that the run restarts, and restarts it |
| B26 | Benchmark panel | Per-scenario winner table with the "wins N of M, not all" note and the churn caveat |
| B27 | Benchmark matches the file | Spot-check `full_day` RL **153.8** / greedy **149.9** against `results/eval_stats.csv` |

### Controls

| # | Criterion | How to verify |
|---|---|---|
| B28 | `Space` | Pauses / resumes; dismisses an open story card |
| B29 | `→` | Advances to the next event, through it |
| B30 | `A` / `R` | Approve / reject when a recommendation is pending |
| B31 | `F` | Fails the selected link |
| B32 | `Esc` | Closes the story card, then leaves fullscreen |
| B33 | Shortcuts ignored in fields | Type in the comparator select — no shortcut fires |
| B34 | Fullscreen | ⛶ toggles; `Esc` exits |

### Scaling, resilience, print

| # | Criterion | How to verify |
|---|---|---|
| B35 | Scale toggle scales rates only | `490 Mbps of 500 Mbps` → `4.9 Gbps of 5.0 Gbps` |
| B36 | Utilization invariant | The percentage does **not** change when the toggle flips |
| B37 | Banner stays visible | Amber banner with the full scaling sentence remains while the toggle is on |
| B38 | Reconnect indicator | Stop the server → dot turns red; restart → reconnects |
| B39 | Reconnect keeps a pending proposal | With a recommendation pending, reload the page — the card is still there |
| B40 | Error overlay | Force an error → full-screen overlay with the backend message |
| B41 | Print summary | *Print / Save as PDF* renders scores, totals, timeline and the disclaimer |

### Guided story

| # | Criterion | How to verify |
|---|---|---|
| B42 | Nine steps run through | Intro → congestion → propose → decision → live event → failure → propose → repair → summary |
| B43 | Blocks on a pending decision | Press `→` while a recommendation is pending → toast asks you to approve or reject |
| B44 | Failure has actually applied | At the failure card, the map shows a dashed link and the timeline logs it with a fast-reroute count |
| B45 | FRR is credited correctly | The card states fast reroute is a protection mechanism, **not** the AI |
| B46 | Narration matches reality | The congestion card describes what the run produced, not a scripted claim |
| B47 | Summary uses real numbers | Final table matches the comparison panel |
| B48 | Reset story | Rebuilds t=0 with the same configuration and clears the timeline |

---

## C · Cross-cutting

| # | Criterion | How to verify |
|---|---|---|
| C1 | Zero console errors | Both pages, whole pass |
| C2 | No external requests | Network tab: everything from `/static/` or `/api/` |
| C3 | No training executed | `/api/training/progress` reports `active:false`; no new directory under `models/` |
| C4 | Published results untouched | `git status` shows nothing modified under `results/` or `models/` |
| C5 | Keyboard focus visible | Tab through both pages — focus ring on every control |
| C6 | Reduced motion respected | Enable OS "reduce motion" → no entry animations |
