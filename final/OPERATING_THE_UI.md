# Operating the interface

Every control, what it does, and what it refuses to do.

If you have five minutes, read **§1 The shortest path** and **§3 The control
panel**. Everything else is reference.

---

## 1. The shortest path

1. Open <http://127.0.0.1:8000/present>.
2. In the left panel, press **Start run**.
3. Press **Step once** a few times, or **Resume** to let it run.
4. Read the card under the map: that is the decision the controller just made.

The defaults are already the interesting ones: the governed V2 environment, the
`demo_evening` scenario, seed 42, and the masked contextual bandit on training
root 42.

To see two controllers side by side: tick **Compare two controllers on the same
run** *before* pressing Start run, then step. The comparison lane fills in under
the map.

---

## 2. The three modes

One application, three depths over the **same moment**. Switching mode never
restarts the run, never changes the source and never loses your selection.

| Mode | Shortcut | What it is for |
|---|---|---|
| **Presentation** | `Alt+1` | driving and narrating a run: the map, the decision, the comparison, the results |
| **Network Information** | `Alt+2` | the network as an engineer reads it: links, demands, paths, SLA risk, filters |
| **RL Information** | `Alt+3` | the policy's decision: observation, action mask, policy output, reward terms — and the governed study record |

The mode buttons are in the header. The control panel is present in Presentation
mode.

---

## 3. The control panel

The single left column. Everything that configures or drives a run is here, in
the order a first-time user needs it. Nothing that starts a run lives anywhere
else.

### 1 · Set up the run

These are locked once a run is loaded — start a new run to change them.

| Control | What it does |
|---|---|
| **Environment** | `V2` (default) is the governed study environment: 604-value observation, 69 actions, 12 reward components. `V1` is the earlier environment and must be asked for explicitly. |
| **Scenario** | Which evening the simulation plays. `demo_evening` includes a scheduled Kayseri–Samsun link failure. |
| **Seed** | Any whole number. The same seed reproduces the same traffic and the same events exactly. Frozen holdout seeds 1001–1005 are refused with the reason. |
| **Execution** | **Automatic** — the controller acts on its own, and each completed decision is explained afterwards. **Manual · advisor approval** — the run pauses before each proposed action; nothing is applied until you approve it. |
| **Controller A** | The controller that drives the run. A controller with no verified checkpoint is shown *disabled with the verification reason*, never hidden and never quietly swapped. |
| **Compare two controllers** | Tick to add **Controller B**. Both lanes get the same scenario, seed, starting state, traffic, failures and interventions. If that cannot be proved, no comparison is shown. |
| **Checkpoint root** (V2 only) | Which of the study's three continuity training roots the checkpoint comes from. 42 is the default — the study's primary scientific root, chosen by fixed identity and never by performance. |
| **Speed** | Presentation pacing, not real time. `1x` is one 5-simulated-minute control interval every 2 seconds. |
| **Start run** | Builds the session and loads it paused at step zero. If it is disabled, the reason is printed directly beneath it. |

### 2 · Run it

| Button | What it does |
|---|---|
| **Start run / Pause / Resume** | The primary transport. Its label always says what pressing it will do. |
| **Step once** | Advance exactly one control interval. In advisor execution this produces a *proposal* instead and holds it. |
| **Skip to next event** | Fast-forward to the next scheduled scenario event. **Under advisor execution this asks you to delegate the stretch first** — see §5. |
| **Stop** | Pause the ticking loop. The run stays loaded. |
| **Reset run** | Same environment, scenario, seed, controllers and root, back at step zero. The run it replaces is **kept** and appears under Results. |
| **Full reset** | Stops everything, clears the session, closes Guided Story and audience view, and returns to the configuration form. Runs kept so far survive this. |

Neither reset mutates a model, a checkpoint or any study artifact.

The status line underneath reads, for example:
`Running · step 12 · 1 earlier run(s) kept`.

### 3 · Approve or reject *(advisor execution only)*

| Button | What it does |
|---|---|
| **Approve** | Applies the held action and advances one interval. |
| **Reject · no TE change** | Advances one interval applying nothing (action 0). |

Both are disabled unless an action is actually being held. While one is held the
panel says *"The proposed action is held. Nothing has been applied yet."*

In automatic execution this section instead explains that there is nothing to
approve — the policy already acted, and the card under the map explains what it
did.

### 4 · Guided Story

An eleven-beat walk through one real `demo_evening` session, in the governed V2
environment with advisor approval. **It advances the actual engine; it does not
script the network.** A beat whose event has not happened says so rather than
narrating it into existence.

| Button | What it does |
|---|---|
| **Start / End Guided Story** | Starts a fresh story session, or leaves the story (the run stays). |
| **Previous** | Reviews an earlier beat. It does **not** rewind the network, and the copy tells you where the network actually is. |
| **Next** | Advances to the next beat, performing whatever that beat requires — a step, a proposal, an approval, or a fast-forward. |
| **Play automatically** | Paces the beats for you. It **stops at every recommendation** and waits for Approve or Reject. It never answers for you. |
| **Restart** | Rebuilds the story session from beat one. |

The eleven beats: establish the network → read the initial evening → traffic
rises → the bottleneck becomes visible → SLA risk appears → the policy
recommends → inspect or approve → observe the transition → demand surge and
failure → compare decisions → repair and the governed conclusion.

Beat 11 opens the frozen study conclusion.

### 5 · Results

| Button | What it does |
|---|---|
| **Refresh results** | Re-reads the results surface. It updates on its own during a run; this is for when nothing is running. |
| **Save this run** | Writes a summary row per lane to `results/runs.db`. This is the only way a run survives a server restart. |

The status line says how many runs are kept in this session and how many
survived an earlier full reset.

### 6 · Study evidence and results

The three frozen, read-only records of the closed study. **These are not
simulation settings.** None of them can be run, compared live or chosen as a
model — the region exists precisely so they can never sit beside the scenario or
model pickers.

| Button | Opens |
|---|---|
| The record buttons | the selection-stage record, the final holdout record, or a recorded replay |
| **What the study concluded** | the frozen conclusion, over whatever is on screen |
| **Q&A jumps** | shortcuts to the surface that answers a common question |

---

## 4. Reading the main surfaces

### The moment rail (top of Presentation mode)

Eight fixed cells: phase, time, incident, busiest link, action, interval reward,
cumulative reward, SLA risks now. **The geometry never moves** — only the values
change — so a presenter can point at a cell and it stays put.

In audience view the rail drops to the four cells a room can read from the back.

Underneath, one sentence describes what changed since the previous step.

### The topology

A **fixed engineering schematic**, not geography, and a fictional scaled
network, not a real operator's. Line weight is capacity; line treatment is
pressure; a failed link is drawn as failed. The map summarizes an undirected
link with its busier direction and says so in inspection.

| Control | What it does |
|---|---|
| **Zoom in / Zoom out / Fit / Reset view** | camera only; the geometry itself never moves |
| **List view** | the same topology as an accessible list, for screen readers or when a projector loses fine lines |

Click a city, link or demand to select it. Arrow keys move between cities.

### The recommendation card (under the map)

In **automatic** execution: an explanation of a decision that already happened.
There is no approval affordance and no fabricated preview of something that
already ran.

In **advisor** execution: the proposal being held, exactly what it would move,
and an expected outcome computed on a *copy* of the current state — which is an
estimate, labelled as one, and shown beside the observed outcome once the
interval runs. Where they differ, the difference stays visible.

### The comparison lane

Only appears with two controllers, and only shows a comparison while the backend
can **prove** both lanes are running the same experiment.

- **Verdict** — cumulative return for each lane and the signed gap. It is stated
  in score units, never as a percentage: operational return is a signed score,
  and a ratio of signed numbers is meaningless.
- **Lane cards** — what each controller did this interval, and the validator's
  reason if the environment refused the move. Lane A carries an **A** token and a
  solid border; lane B a **B** token and a dashed one, so they are still
  distinguishable in greyscale or on a bad projector.
- **Metric table** — the latest interval side by side, with the gap and which
  lane is ahead. A measure with no better direction names no leader.
- **Movement table** — controller TE changes, FRR protection moves and
  post-recovery restorations, in three separate counters. They are never summed:
  protection is not a policy decision.
- **First divergence** — the earliest interval in which the two lanes chose
  differently. Everything before it is identical by construction.

**If the proof breaks**, the lane shows the reason and the fields that disagree
and *no comparative number at all*. That is deliberate: a wrong comparison is
worse than no comparison.

### The results panel

Three sections that never share a table:

1. **Live run** — what is on screen. One seed, one pass, unaudited.
2. **Earlier runs kept in this session** — archived by Reset run.
3. **Closed V2 study** — a pointer to the frozen record, and the reason it stays
   separate. Its numbers are rendered in exactly one place, under RL Information
   → Governed Study, from the artifacts themselves.

The first two are demonstrations and are labelled as such. Only the third
supports a conclusion.

---

## 5. Delegated fast-forward — the one asymmetry

Advisor execution holds every proposed action for your decision, with one
exception: a **fast-forward** applies the controller's own actions for a stretch
of intervals in a single gesture. Nothing in that stretch is individually
approved.

Rather than doing that silently, the application:

1. **asks you first** — Skip to next event shows a confirmation naming the
   consequence, and the server refuses an undelegated fast-forward outright;
2. **records it as one delegated batch**, not as a run of approvals;
3. **shows it on the timeline** as its own event: *"7 interval(s) delegated"*;
4. **counts it in the panel**: *"45 interval(s) in this run were delegated, not
   approved individually."*

Guided Story's own skips are delegated stretches, and the beat copy says so.

---

## 6. Audience view and fullscreen

| Control | What it does |
|---|---|
| **Audience view** (header) | Hides the working chrome: the control panel, the ledger, the tool row. The map and the moment rail get the room. |
| **Exit audience view** | Always visible while audience view is on. It is deliberately rendered *outside* the chrome that audience view hides, so the only way out can never be hidden by the same rule. |
| **Fullscreen** (header) | Browser fullscreen. |
| **Escape** | Closes a drawer first, then leaves audience view, then leaves fullscreen — in that order. Audience view is always escapable, including while fullscreen. Nothing reloads. |

---

## 7. Keyboard

| Key | Action |
|---|---|
| `Alt+1` / `Alt+2` / `Alt+3` | Presentation / Network / RL mode |
| `Space` | play / pause |
| `→` | step once (or next beat, during Guided Story) |
| `←` | previous beat, during Guided Story |
| `G` | toggle Guided Story (in Presentation mode) |
| `E` | explain this moment |
| `?` | keyboard reference |
| `[` / `]` | jump to previous / next timeline event |
| `/` | focus the search box in the current mode |
| `Escape` | close drawer → leave audience view → leave fullscreen |

---

## 8. What the interface will not do

Worth knowing, because each is a deliberate refusal rather than a missing
feature:

- **It will not show a comparison it cannot prove.** Two lanes in different
  environment versions are refused before either engine is read.
- **It will not put a percentage on a signed return.**
- **It will not average a demonstration with the study's evidence.**
- **It will not fabricate a V2 traffic override.** V1 has a manual traffic
  multiplier and a burst injector; the frozen V2 engine does not, and those
  endpoints return an error naming the reason instead of emulating one.
- **It will not substitute V1 for an unavailable V2 checkpoint.** It reports the
  exact verification step that failed.
- **It will not show zero where the engine has no value.** An absent value is
  absent, with the reason.
- **It will not pad a V2 result into the shape of a V1 result.** A saved V2 run
  lists what it could not measure.
- **It will not claim a bandit score is a probability.** The label comes from
  the controller's declared output semantics, never from the shape of the
  numbers.
