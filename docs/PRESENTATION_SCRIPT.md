# Presentation script — 20–25 minutes

Scenario **Guided Operator Demonstration** (`demo_evening`), seed **42**,
**RL vs greedy**, advisor mode on.

Before you walk in:

```bash
python scripts/demo.py
```

Then press **F11**, or the ⛶ button. The training endpoint is already disabled.

Numbers in *italics* below are read off the screen during the talk — never
quoted from memory. Numbers in **bold** come from `results/eval_stats.csv` and
are fixed.

---

## 0 · Before you start (30 s)

Say the disclaimer out loud once, then leave it in the footer:

> "This is a fictional national backbone built for this project. It is not a
> real operator's topology, and the city names are labels I attached to make
> it readable."

---

## 1 · What the audience is looking at (2 min)

**Do:** nothing. Leave the map up.

**Say:** eighteen cities, thirty-two links. Colours are how full each link is —
green is quiet, red is at capacity, pink is over capacity and dropping traffic.
Seventeen traffic flows: voice calls, video, enterprise VPNs, consumer
internet. Every five minutes of simulated time, a controller may move one flow
onto a different route.

**Say:** two controllers run side by side on identical copies of the same
network, seeing byte-identical traffic. One is a trained AI advisor. The other
is the kind of rule-based controller networks use today.

Point at the KPI cards. "Overall network score" is a combined benchmark score —
say explicitly: *it is not money and not an industry KPI.*

---

## 2 · Start the story (1 min)

**Click:** `Start Guided 5-Minute Story` → **Start the evening**.

**Say:** it is 17:00. Evening traffic is about to build.

---

## 3 · The network is already tight (2 min)

**Click:** `Continue` (or `→`).

The card reads *"Already close to the limit"* — on seed 42 one corridor is at
roughly *98 %* within the first interval.

**Say:** this is the point. Even before anything goes wrong, a corridor is
nearly full. Traffic engineering is not about emergencies; it is about the
ordinary evening.

**Say:** past 100 % the link is not "a bit slow" — packets are dropped.

---

## 4 · The first recommendation (4 min) — the centrepiece

**Click:** `Ask the AI Advisor`.

Walk the card top to bottom. It is all live data:

- **Headline** — which flow, and away from which corridor.
- **Now goes via / Would go via** — the two routes as city chains.
- **Why** — the tightest link on *this flow's own route*, with its measured
  utilization, and the tightest point on the proposed route.
- **Expected effect** — e.g. *117.3 % → 103.6 %*.

**Say — this is the sentence to get right:**

> "That prediction was produced by copying the network, replaying the next five
> minutes twice — once with the change, once without — and comparing. The live
> network was not touched. If I reject this, nothing at all has happened to it."

**Say:** the safety check is a separate, conventional constraint checker. The
model proposes; the rules still have a veto.

**Then pause and ask the room to decide.** Press `A` to approve.

The next card shows **projected vs measured**. Say why they differ: the
measured interval also contains the traffic change itself. Point at
`delta_max_util` in the engineering view if an engineer asks for the isolated
effect.

---

## 5 · The live event (2 min)

**Click:** `Continue`.

**Say:** a major live event just started; traffic to the east has doubled.
Nothing has failed — this is pure demand. Watch the map turn orange.

---

## 6 · The backbone failure (3 min)

**Click:** `Continue`.

The card reads *"A backbone link has failed"*, and the timeline logs, for
example, *"Kayseri–Samsun link failed. The network immediately moved 5 affected
traffic flows onto backup paths (fast reroute)."*

**Say, clearly, because this is where people over-credit the AI:**

> "That instant recovery is **not** the AI. That is fast reroute — a standard
> protection mechanism. It restores connectivity in milliseconds, and it does
> not care whether the backup path is already busy. The AI's job starts
> *after* that: deciding what to do about the mess fast reroute just made."

---

## 7 · The second recommendation (2 min)

**Click:** `Ask the AI Advisor again`.

Consider **rejecting** this one (`R`). It demonstrates that the operator is in
control and that rejecting is a first-class outcome with a recorded result.

---

## 8 · The repair and the score (2 min)

**Click:** `Continue` → the final card.

Read the table out loud, including the parts that do not flatter the model.
On seed 42 with a reject at step 7, greedy typically finishes ahead.

**Say:**

> "On this single run the traditional controller won. I am showing you that
> because a single run with one seed is not evidence — and because it is true."

---

## 9 · The honest evaluation (3–4 min)

Point at **Published results across every scenario** (5 seeds each, from
`results/eval_stats.csv`).

**Where RL wins:**

| Scenario | RL | Best other | |
|---|---|---|---|
| Normal National Traffic Day | **153.8** | **149.9** (greedy) | RL |
| Hidden Shared Bottleneck | **72.9** | **70.0** (greedy) | RL |

**Say:** these are the two scenarios that reward *looking ahead*. On the hidden
bottleneck, the greedy controller takes the locally attractive path and walks
into a shared choke point; the trained policy avoids it.

**Where RL loses — say this without softening it:**

| Scenario | RL | Greedy | |
|---|---|---|---|
| Major Live Event Traffic Surge | **−94.6** | **−79.2** | greedy |
| Ankara–Kayseri Backbone Failure | **−76.9** | **−46.1** | greedy |

**Say:**

> "Greedy wins the reactive incidents, and it is not close. When the right move
> is obvious and immediate, a rule that reacts to current congestion beats a
> policy that learned an average."

**The churn weakness — do not skip this:**

> "There is a second cost. On a normal day the AI advisor makes **288** route
> changes to greedy's **70**, with **264** of them flapping — moving traffic
> back to a path it just left. Every one of those is a real change to a real
> network. An operations team would not accept that, and the reward function
> does not charge enough for it. That is a limitation of my reward design, not
> a detail."

---

## 10 · Closing (2 min)

**Say:**

- The question was whether RL can beat conventional traffic engineering when
  demand changes over time. The answer is **partly**: on sustained,
  look-ahead-rewarding conditions, yes; on sudden incidents, no.
- It is a flow-level simulation. No packets, no TCP, no RSVP-TE signalling,
  instant reroutes, one topology.
- The usable outcome is not "replace your controller". It is the advisor
  pattern: a policy that proposes, a constraint checker that vetoes, and a
  human who approves — with a measurable prediction attached to every
  suggestion.

Offer the engineering view (`/advanced`) for questions: full tables, reward
components, action probabilities, counterfactuals, event log, CSV export.

---

## Fallback plan

| If | Do |
|---|---|
| The story desyncs | `Reset story`, then drive manually with `→`, `A`, `R` |
| A link is already failed | The toast says so explicitly; use `Recover link` |
| The page reloads | The session survives; the script pointer does not — continue with `Next event` and Approve/Reject |
| The backend errors | A full-screen overlay appears with the message; `Reset story` rebuilds the identical run |
| Someone asks for raw numbers | `/advanced` → Metrics / Links / Events tabs, or the CSV export button |

## Questions you should expect

**"Is that reward score meaningful?"** — It is a benchmark score combining
delivery, congestion, service quality and stability, defined in
`configs/reward.yaml`. Not money, not an SLA. The per-metric rows underneath it
are the ones with physical meaning.

**"Would this work on my network?"** — Unknown. The policy was trained on this
topology only; nothing here demonstrates topology generalization.

**"What is a demand-interval SLA violation?"** — One count per traffic demand
per five-minute interval that missed its latency or loss target. A single
demand suffering all evening scores many.

**"Why not just use greedy?"** — On several of these scenarios, you should.
That is the finding.
