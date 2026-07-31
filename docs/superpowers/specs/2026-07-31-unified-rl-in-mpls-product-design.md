# Unified RL-in-MPLS Product Design Specification

**Date:** 2026-07-31

**Status:** Approved design package for Prompt 2

**Scope:** Product, UX, visual, data, architecture, migration, and acceptance specification only. No production UI is implemented by this document.

## 1. Outcome

RL-in-MPLS becomes one coherent application with exactly three primary modes:

1. **Presentation** — the whole project at speaking depth.
2. **Network Information** — the MPLS operations workspace.
3. **RL Information** — the inference and governed-evidence observatory.

Guided Story is a polished workflow inside Presentation. It is not a fourth mode.

All three modes share one shell, the same selected network moment, the same topology geometry, and the same provenance grammar. They differ in explanation depth and inspector density. A mode change never implies that the underlying data source changed.

## 2. Hard boundaries

The redesign does not change:

- any V1 or V2 scientific definition;
- reward, observation, action, mask, scenario, seed, baseline, metric, horizon, transition, or evaluation semantics;
- a learner, checkpoint, compact table, manifest, raw trace, replay buffer, or training artifact;
- internal router, link, demand, scenario, policy, checkpoint, or action identifiers;
- the distinction between live V1 behavior and closed V2 evidence;
- the fail-closed read-only behavior of `mplssim/evidence/` and `/api/v2/*`.

The protected file `results/environment_v2_validation/manifest.json` is outside the implementation scope.

## 3. Product truth model

### 3.1 Provenance is a type, not a badge

Every view has exactly one `SourceKind`:

```text
live_session
recorded_replay
development_evidence
final_holdout_evidence
```

| SourceKind | Persistent label | May execute a policy? | May show topology telemetry? | May show scientific conclusions? |
|---|---|---:|---:|---:|
| `live_session` | `LIVE` | Yes, only implemented/configured policies | Yes, from the running engine | Only in a separate governed-evidence panel |
| `recorded_replay` | `RECORDED` | No | Only fields actually recorded; V2 traces have no per-link utilization | Yes, as linked context, never as the replay result |
| `development_evidence` | `DEVELOPMENT` | No | No fabricated episode topology | Development findings and selection only |
| `final_holdout_evidence` | `FINAL EVIDENCE` | No | No live or fabricated topology | Yes, the closed one-shot result |

Source changes require an explicit control or direct link. Mode changes do not change source.

### 3.2 Required context tuple

The shell shows the applicable parts of this tuple at all times:

```text
source kind · environment version · scenario · seed · policy/checkpoint
comparator · session/evidence state · network/evidence time · selected object
```

Unavailable tuple members are omitted only when they do not exist for that source. They are never replaced by a plausible default.

### 3.3 Vocabulary contract

| Concept | Required wording | Forbidden wording |
|---|---|---|
| A model proposes an action | policy recommendation; “Masked Bandit suggests…” | AI advisor; the model thinks/wants/knows |
| PPO distribution | action probability | confidence, unless a separately defined calibrated confidence exists |
| Bandit output | action score; immediate-reward estimate | probability; confidence |
| Clone-based lookahead | simulated one-interval estimate | prediction without qualification; what will happen |
| Feature change list | changed-feature ranking | causal importance; reason the model acted |
| Recorded trace | recorded replay | live; rerun; evaluation in progress |
| V2 result | final-holdout evidence | live comparison |
| SLA counter | demand-interval SLA violations | number of customers/services affected, unless showing current demand count |

### 3.4 No-op grains

The product always names both forms in full:

- `Step-pooled no-op share`: action 0 count divided by all recorded steps.
- `Episode-level mean no-op frequency`: the mean of each episode's own no-op fraction.

No component may display an unlabeled “no-op rate” when both grains are possible.

## 4. Information architecture

### 4.1 Unified shell

```text
Application shell
├─ Primary mode control
│  ├─ Presentation
│  ├─ Network Information
│  └─ RL Information
├─ Persistent context ledger
│  ├─ provenance state
│  ├─ environment/scenario/seed
│  ├─ policy/checkpoint/comparator
│  ├─ time/step/session state
│  └─ selected object
├─ Shared topology/moment stage
├─ Explain this moment
├─ Direct-object navigation
├─ Source switcher
│  ├─ Live session
│  ├─ Recorded replay
│  ├─ Development evidence
│  └─ Final evidence
└─ Mode surface
   ├─ Presentation story and audience controls
   ├─ Network operations inspectors
   └─ RL pipeline and evidence inspectors
```

The source switcher does not resemble the primary mode control. Modes answer “how deeply do I want to understand this?” Sources answer “what kind of record am I looking at?”

### 4.2 Routes and backward compatibility

| URL | Initial application context | Backward-compatible behavior |
|---|---|---|
| `/` | Network Information, current or idle live session | Preserves the engineering-console destination |
| `/advanced` | Network Information, current or idle live session | Stable alias; same shell and deep context as `/` |
| `/present` | Presentation, current or idle live session | Preserves Presentation destination |
| `/present?workflow=guided-story` | Presentation, Guided Story setup | Direct story launch |
| `/study` | RL Information, Governed Study, Final Evidence | Preserves sealed-study destination without creating a fourth mode |
| `/?mode=rl&source=recorded&policy_id=…&scenario=…&seed=…&step=…` | RL Information, recorded episode deep link | Reconstructs exact recorded context when available |
| `/?mode=network&object=link:L11&event=…` | Network Information, selected link/event | Opens the object and timeline incident |

The server should render the shell directly for each route. Do not rely on client-side redirect flashes.

### 4.3 Persistent navigation

- `Alt+1`: Presentation.
- `Alt+2`: Network Information.
- `Alt+3`: RL Information.
- `Space`: play/pause when focus is not in a form control.
- `ArrowRight`: next step or next story beat.
- `ArrowLeft`: prior recorded step or prior story beat; live sessions never rewind the engine.
- `G`: open Guided Story from Presentation.
- `/`: focus object/feature search in Network or RL Information.
- `E`: Explain this moment.
- `[` / `]`: prior/next incident bookmark.
- `Esc`: close topmost drawer/dialog, then leave fullscreen.

Keyboard shortcuts are discoverable in help and disabled while typing. Every shortcut has a visible control equivalent.

### 4.4 Explain this moment

The mechanism opens a depth switch without changing the moment:

| Depth | Answers |
|---|---|
| Presentation | What happened? Why does it matter to this story? What changed? |
| Network | Which demand, route, link, failure, SLA, and restoration facts support that statement? |
| RL | Which observation features, mask states, policy outputs, action, safety result, transition, and reward components exist? |

Explanations are deterministic templates over exposed facts. They never generate hidden rationale. Each sentence links to the source object or pipeline stage that supports it.

### 4.5 Direct navigation

Every incident, recommendation, action, reward event, link, demand, and path has a stable in-session `MomentRef`:

```text
{sourceKind, sessionOrEvidenceId, step, eventId, objectType, objectId}
```

Selecting a reference:

- focuses the topology object if real link/path telemetry exists;
- opens the relevant Network or RL inspector;
- moves the timeline cursor to the event;
- retains the reference through mode changes;
- shows “Topology detail unavailable for this recorded grain” instead of inventing a route when it does not exist.

## 5. Presentation mode

### 5.1 Job

Help a presenter explain the complete project to a network-aware audience while the topology remains readable from the room. The audience should always know the scenario, source, current event, selected policy, action, outcome, and study conclusion.

### 5.2 Content hierarchy

1. Persistent provenance and current context.
2. Large topology stage.
3. Current phase, incident, and change since prior step.
4. Recommendation/decision card directly beneath the topology when applicable.
5. Stable primary figures: current interval reward, cumulative reward, current action, network condition.
6. Synchronized comparison lane.
7. Story progress, bookmarks, and Q&A jumps.
8. Governed conclusion drawer.

The current five equal KPI cards are replaced by one prioritized moment rail. The topology gets at least 65% of the presentation content area at 1920×1080.

### 5.3 Top-of-stage moment rail

The rail contains:

- phase: normal / rising / pressure / failure / recovery / stabilized;
- network time and step;
- active incident or `No active incident`;
- action: actual action label or `No TE change`;
- current interval reward and cumulative reward, both explicitly labeled;
- one sentence describing what changed since the prior completed step.

Values stay in fixed-width cells. The rail never grows vertically during a run.

### 5.4 Recommendation card

The card appears only for a real policy output:

```text
Masked Bandit suggests moving İstanbul → Erzurum bulk traffic

Demand       D13 · bulk · [actual offered Mbps]
Old path     [actual current path]
Proposed     [actual candidate path]
Grounding    [measured path pressure/mask facts only]
Before       [actual pre-action telemetry]
Expected     [clone-based simulated estimate, if available]
Score        [bandit immediate-reward estimate] OR Probability [PPO]
Safety       Valid / Invalid · [actual validator reason]
Observed     [post-action telemetry after execution, initially pending]
Reward       [actual interval reward, initially pending]
```

Rules:

- The policy name is part of the headline.
- Bandit score and PPO probability use different labels and visual encodings.
- “Grounding” is a telemetry summary, not internal reasoning.
- Expected telemetry is visually and verbally marked `SIMULATED ESTIMATE`.
- The Decision Lens can preview this card without execution.
- Approve is present only in configured operator-approval workflows. Ordinary live comparison auto-applies each controller's real action.
- Invalid actions cannot be approved into a false “applied” state. The resulting validator outcome is shown.

### 5.5 Audience view and presenter cockpit

Audience view hides the cockpit, object inspector, source switcher, and nonessential help. It retains provenance, scenario, clock, story progress, topology, moment rail, recommendation, comparison lane, and disclosure.

Presenter cockpit is a compact bottom drawer containing:

- play/pause/step;
- next/back story beat;
- speed;
- incident bookmarks;
- policy/comparator selection;
- fullscreen and audience-view toggle;
- Guided Story controls;
- Q&A jumps;
- optional operator approve/reject.

The cockpit never overlays nodes or the recommendation card.

### 5.6 Comparison lane

The lane shows the primary policy and one implemented comparator under a synchronized context:

- same environment version;
- same scenario;
- same non-holdout demonstration seed;
- one cloned starting engine state;
- identical scripted/manual interventions;
- equal step horizon;
- synchronization fingerprint `matched` before each paired step.

Each lane shows actual action, reward, cumulative reward, busiest-link state, demand-interval SLA count, moved bandwidth, and route changes when available. The lane never calls a final-holdout aggregate a live comparator.

If synchronization cannot be proven, comparison controls are disabled and the product states which proof is missing.

### 5.7 Governed conclusion

The conclusion opens as `FINAL EVIDENCE`, even during a `LIVE` session. It contains:

- Masked bandit `18.221`;
- MaskablePPO `9.036`;
- Greedy `-2.327`;
- bandit won all three roots and six of seven scenarios;
- PPO led only `deceptive_local_optimum` by `1.107`;
- safety remained intact;
- bandit moved more bandwidth than PPO but drastically less than greedy;
- the evidence did not positively establish a need for temporal planning in this formulation;
- this does not establish that planning is generally irrelevant to MPLS or traffic engineering;
- direct link to RL Information → Governed Study.

Its border, provenance ledger, and wording make it impossible to mistake for the live run above it.

### 5.8 Q&A jumps

| Jump | Destination |
|---|---|
| What is MPLS? | Presentation explanation over the selected demand/LSP; Network deep link available |
| Why this action? | Recommendation grounding, then RL pipeline if requested |
| Is it safe? | Mask/validator result and protected-class rule |
| Did temporal planning help? | Final Evidence conclusion with both required halves |
| How was this result validated? | Final Evidence integrity, roots, seeds, one-shot workflow, hashes |

## 6. Network Information mode

### 6.1 Job

Provide a serious MPLS-TE operations workspace where topology, tables, and incident time are bidirectionally linked. It is not a generic infrastructure dashboard and does not invent unmodeled protocols.

### 6.2 Content hierarchy

1. Topology with stable Turkey layout.
2. Current/previous delta rail and incident state.
3. Selected router/link/demand/path inspector.
4. Demand and SLA-risk table linked to topology.
5. Event timeline.
6. Filters and overlays.
7. Modeled-vs-real-operator disclosure.

### 6.3 Filters and overlays

- Traffic class: voice, video, enterprise VPN, consumer internet, bulk data, critical services.
- Condition: congested, SLA risk, failed, degraded, recovering.
- Object: routers, links, demands, active LSPs, candidate paths.
- Path: primary/current, alternate candidates, selected comparator path.
- Change: changed since previous step, rerouted, reversed, flapped, restored.

Filters hide labels and overlays, not raw state. A visible `Clear filters` control and count state prevent an apparently empty network.

### 6.4 Router inspection

Shows city, role, internal ID, neighbors, number of traversing LSPs, current affected demands, and incident membership. It does not invent CPU, memory, label-table, BGP, RSVP, or interface counters.

### 6.5 Link inspection

Shows internal link ID, endpoint cities/IDs, direction, capacity, load, utilization, available capacity, propagation delay, modeled queue delay, modeled loss, admin weight, status, congestion, LSP count, current/prior delta, and selected demand contribution when derivable.

For an undirected physical link, both directions are visible separately. The map may summarize with the worse direction but the inspector names that rule.

### 6.6 Demand and path inspection

Shows demand ID, endpoints, class, priority, protection, base/offered/carried traffic, current candidate index and full route, alternate routes, current delay/loss/SLA state, bottleneck, path changes, last reroute, cooldown or TE dwell, previous TE path, reversal/flap status, and moved bandwidth when exposed.

V1 fields are labeled V1 (`cooldown`). V2 fields are labeled V2 (`hard TE dwell`, previous TE path, projected gross bottleneck). The inspector does not merge them.

### 6.7 Bottleneck and SLA-risk table

Default order:

1. disconnected protected demands;
2. disconnected unprotected demands;
3. SLA-violating demands by priority-weighted severity;
4. demands whose current path crosses a congested link;
5. remaining demands by offered traffic.

Selecting a row focuses the current path and its tightest modeled hop. Selecting a link filters/highlights the rows that cross it. The table has a screen-reader caption and preserves internal IDs.

### 6.8 Incident timeline

Event types:

```text
congestion threshold · SLA risk · scripted/manual failure · FRR change
restoration · policy recommendation · accepted/rejected TE action
reversal · flap · repair · stabilization
```

Each event contains time, environment, source, object IDs, before/after facts, policy actor when applicable, and a direct link. FRR is explicitly labeled as built-in protection, not a learner action.

### 6.9 Modeled versus real-operator considerations

A restrained disclosure drawer states:

- modeled: flow-level demand, candidate LSP choice, analytic delay/loss, scripted/manual link state, FRR-style repair, dwell/cooldown, TE actions;
- not modeled: packets, TCP behavior, RSVP-TE/IGP convergence, label signaling, exact GIS, real operator topology, production control-plane timing.

No unimplemented MPLS feature appears as an operational control.

## 7. RL Information mode

### 7.1 Job

Make the actual inference and evidence chain interrogable without turning 604 values or 69 actions into an unreadable dump.

### 7.2 Primary pipeline

```text
Observation
  → Action mask
  → Policy output
  → Selected action
  → Safety validation
  → Environment transition
  → 12 reward components
  → Next observation
```

The current stage is highlighted. Every stage links to its data source and exposes an unavailable state.

### 7.3 Context subviews

RL Information has three secondary views, not primary modes:

- `Decision Observatory`: live inference or recorded step.
- `Governed Study`: final and development evidence with strict separation.
- `Model Provenance`: checkpoint, hyperparameters, roots, seeds, hashes, and diagnostics.

### 7.4 Observation inspector

The V2 default observation is grouped from `configs/experiments/rl_observation_v2.yaml`:

- directed-link input utilization;
- directed-link up state;
- demand offered/base;
- priority;
- protected state;
- measured delay/SLA;
- measured loss/SLA;
- current path age;
- TE dwell remaining;
- disconnected state;
- current path one-hot;
- previous TE path one-hot;
- candidate live;
- candidate propagation delay;
- candidate projected gross bottleneck.

Each row shows:

```text
semantic label · object label/ID · raw current · normalized current
prior normalized · delta · transform · reference boundary · source offset
```

Search covers feature names, cities, internal IDs, class, candidate index, and offset. Filters include changed, threshold-crossing, selected demand/path, link features, demand health, path history, and candidate feasibility.

Changed-feature ranking sorts by absolute normalized delta between prior and current observations. It is explicitly labeled `descriptive change, not causal importance`.

For live V1, the inspector uses the 586-value V1 schema and labels it V1. It never presents V1 values as the V2 604-value schema.

Recorded V2 replay does not currently contain observations. It shows `Observation unavailable in recorded trace` and names the fields that are available.

### 7.5 Action space and mask

The complete action grid contains:

- action `0`: no TE change;
- actions `1–68`: `17 demands × 4 candidate paths` using `1 + 4*d + p`.

Rows show action ID, demand/city/class, candidate index/path, selected/current/no-op state, mask validity, validator reason, policy output, and relationship to the selected action.

States are exhaustive:

- chosen and valid;
- valid runner-up;
- valid no-op;
- invalid with a real reason;
- unavailable because the source does not carry mask detail.

Mask reasons are produced by the authoritative validator. The UI does not infer a reason from a boolean mask.

### 7.6 Policy outputs

#### MaskablePPO

When exposed:

- top masked action probabilities;
- selected action probability;
- runner-up and no-op probability;
- entropy of the masked distribution;
- value estimate;
- algorithm and hyperparameters.

Probability bars sum over the valid masked distribution. Invalid actions show no probability bar. If entropy or value is not exposed, the row says so.

#### Masked contextual bandit

When exposed:

- score/immediate-reward estimate for each valid action;
- selected score, runner-up score, and no-op score;
- score margin;
- deterministic/epsilon-greedy inference mode;
- network and training hyperparameters.

Scores may be negative and do not sum to one. No percent symbol or probability language is used.

### 7.7 Selected action and safety

Shows action ID, decoded demand/path, source policy, whether it was selected or operator-overridden, authoritative mask value, validator result/reason, protected-class rule where relevant, accepted/rejected transition record, reversal, volume share, and edge divergence.

An operator rejection is distinct from an environment rejection.

### 7.8 Reward waterfall

The V2 waterfall contains all 12 components in defined order:

```text
delivery
protected_disconnect
unprotected_disconnect
sla_severity
max_util
overload
potential
move_fixed
move_volume
move_divergence
reversal
invalid
```

It shows signed values, exact defined-order sum, scalar interval reward, residual, and an `EXACT SUM` integrity state. Current and cumulative reward are separate. The potential term exposes `phi_current` and `phi_next` only when present.

For V1 live sessions, the component set is labeled V1 and is not padded to look like the V2 12-component reward.

### 7.9 Counterfactual preview

Decision Lens may request a counterfactual only for a live session whose engine supports exact cloning.

The response compares no-op and the selected valid action on deep clones of the same current state, without changing the real session. It is labeled:

> Simulated one-interval estimate from cloned state. It is not an observed outcome and not final evidence.

After execution, the observed outcome appears beside the estimate. The UI does not hide differences.

Recorded, development, and final-evidence sources show `Counterfactual unavailable for this source` unless a counterfactual was itself recorded.

### 7.10 Model and evidence provenance

Shows:

- environment/observation/action/reward/transition/config/seed-protocol versions;
- algorithm and hyperparameter summary;
- checkpoint transition;
- training root;
- development selection rule and seeds;
- evaluation stage;
- payload and sidecar SHA-256;
- training and evaluation source SHAs;
- episode seed ledger reference;
- deterministic inference and device where recorded;
- integrity counters and safety state.

### 7.11 Governed Study presentation

Final and development evidence use separate regions and adapter types. They cannot share a chart series or aggregate function.

The final result leads with the negative planning finding, the bandit/PPO/greedy returns, the 3/3 roots, the 6/7 scenarios, PPO's one scenario, safety, and movement trade-off. Development curves live behind a persistent `DEVELOPMENT — NOT HOLDOUT` heading. Recorded replay lives under `RECORDED` and never shows link-level topology telemetry.

## 8. Guided Story

### 8.1 Setup

Default source is `LIVE`, environment `V2` when a configured governed checkpoint is available for demonstration, scenario `demo_evening`, non-holdout demonstration seed `42`, and comparator `greedy`. The policy picker lists only configured real policies and shows checkpoint root/transition.

If a V2 learner checkpoint is unavailable, the setup clearly offers the currently installed V1 MaskablePPO demo. It does not silently substitute it. The story copy and observation/reward links change to V1.

Using a frozen checkpoint in a live demonstration does not create or modify final evidence. Holdout seeds `1001–1005` are rejected for live demonstration sessions.

### 8.2 Story beats and interaction sequence

| Beat | Stage action | Audience message | Truth source |
|---:|---|---|---|
| 1. Establish the network | Show the 18-city fictional topology, six classes, selected policy/comparator, `LIVE` ledger | “This is one simulated backbone, not a real operator network.” | topology/display/scenario APIs |
| 2. Read the initial evening | Step once and describe the actual state; do not promise a quiet start | “At 17:00 the network is [actual condition].” | live snapshot |
| 3. Traffic rises | Advance until the actual pressure threshold or next relevant event | Identify the real busiest corridor and time | cloned synchronized live engines |
| 4. Bottleneck becomes visible | Select the actual bottleneck and affected demands | Explain capacity, utilization, direction, and candidate paths | link/demand telemetry |
| 5. SLA risk appears | Focus the highest-priority actual at-risk demand, or state none yet | Separate current affected-demand count from cumulative demand-interval violations | demand telemetry/history |
| 6. Policy recommends | Open Decision Lens; do not execute | “`[Actual policy]` suggests `[actual action]`.” | actual inference/mask/validator |
| 7. Inspect or approve | Presenter opens Presentation/Network/RL explanation depth, then approves or continues auto-apply | Show old → simulated estimate; keep expected distinct from observed | clone counterfactual |
| 8. Observe the transition | Advance one interval and settle new route | Show observed telemetry and actual reward beside the estimate | live post-action state |
| 9. Demand surge and failure | Advance through the actual 20:00 flash crowd and 20:15 `L20` Kayseri–Samsun failure | Explain FRR as built-in protection, then restoration pressure | `demo_evening` events and action log |
| 10. Compare decisions | Show paired policy actions and outcomes under matched fingerprints | State who leads this run and the movement/churn cost; no percentage delta on signed returns | synchronized comparison lane |
| 11. Repair and conclusion | Advance through the 21:00 repair, show stabilization, then open `FINAL EVIDENCE` | Give the closed result and both temporal-planning conclusion halves | live session, then `/api/v2/study` and final holdout |

Story progress is visible but quiet: beat number, short label, and incident bookmarks on the time-distance band. Back moves through explanatory beats but never rewinds a live engine. If the presenter goes back past a live event, the UI says `Reviewing beat 6 · network remains at 20:20`.

### 8.3 Incident bookmarks

- pressure threshold crossed;
- first SLA risk;
- policy recommendation;
- accepted/rejected action;
- 20:00 flash crowd;
- 20:15 Kayseri–Samsun (`L20`, `P5–P8`) failure;
- FRR completion;
- second recommendation;
- 21:00 repair;
- stabilization;
- governed conclusion.

The separate `link_failure` scenario exposes the Ankara–Kayseri (`L11`, `P2–P5`) failure as a Q&A jump and Network Information scenario, not as a false event in `demo_evening`.

## 9. Topology visual and interaction specification

### 9.1 Curated Turkey layout

Display coordinates are presentation metadata in `mplssim/display.py` or a new display-only registry. Scientific `configs/topology.yaml` positions remain unchanged.

Normalized coordinates:

| Internal ID | City | Role label | x | y |
|---|---|---|---:|---:|
| PE1 | İstanbul | LER · ingress | 8 | 30 |
| PE2 | İzmir | LER · ingress | 7 | 64 |
| PE3 | Bursa | LER · ingress | 16 | 45 |
| PE4 | Antalya | LER · ingress | 28 | 87 |
| P1 | Eskişehir | LSR | 28 | 48 |
| P2 | Ankara | LSR | 41 | 40 |
| P3 | Konya | LSR | 41 | 70 |
| P4 | Bolu | LSR | 31 | 31 |
| P5 | Kayseri | LSR | 54 | 53 |
| P6 | Adana | LSR | 55 | 82 |
| P7 | Gaziantep | LSR | 67 | 79 |
| P8 | Samsun | LSR | 57 | 21 |
| A1 | Sivas | LSR · aggregation | 65 | 48 |
| A2 | Malatya | LSR · aggregation | 71 | 62 |
| PE5 | Trabzon | LER · egress | 71 | 17 |
| PE6 | Erzurum | LER · egress | 83 | 35 |
| PE7 | Diyarbakır | LER · egress | 80 | 70 |
| PE8 | Van | LER · egress | 94 | 53 |

The map carries a small `Curated geographic layout · not exact GIS` note.

### 9.2 Crossing management

- Use display-only bend points per physical link.
- Bundle shared west-to-central and central-to-east trunks without hiding link identity.
- Keep `L11` Ankara–Kayseri and `L20` Kayseri–Samsun visually separable.
- Reserve a clear central corridor for selected route overlays.
- Labels use collision-aware fixed offsets defined in the display registry, not force-directed movement.
- A selected link may temporarily lift above a crossing; nonselected links remain stable.

### 9.3 Interaction

- Click/tap selects one object.
- `Enter` opens the object's inspector.
- Arrow keys move to the nearest geographic node; `L` enters adjacent links; `D` lists demands traversing the selection.
- Hover is supplementary only.
- Zoom/pan never moves node relationships and provides `Reset view`.
- Selecting a demand highlights current path; `Show alternates` adds candidate routes with numbered route keys.
- Decision Lens uses old solid-muted, proposed selection-orange, and comparator parallel rail.
- Execution transitions once from old to observed route; no path glow remains afterward.

### 9.4 Accessible alternative

The topology has a synchronized list/tree:

```text
Routers
  Ankara · LSR · P2 · 3 neighbors · 4 active LSPs
Links from Ankara
  Ankara → Kayseri · L11 · up · 62% · 2.0 Gbps
Demands through selected object
  D13 · İstanbul → Van · bulk · SLA OK
```

List selection and map selection are one state. Screen readers are not required to traverse canvas internals.

### 9.5 Recorded evidence rule

For recorded V2 traces, the stage may show a static reference topology only when labeled `REFERENCE TOPOLOGY · NO RECORDED LINK TELEMETRY`. It never colors, animates, or marks individual links from aggregate `max_util`.

## 10. Data-source matrix

`Existing` means the value is truthfully available now. `Proposed` means Prompt 2 must add the named non-scientific read/serialization capability.

| Visible value or control | Source now | Status / proposed source | Rules |
|---|---|---|---|
| Mode | client route/store | Existing | Exactly three values |
| Provenance state | `/api/v2/*` stage/kind/live; live session type | Proposed `GET /api/product/capabilities` plus adapter type | Never inferred from URL styling alone |
| Environment version | V2 evidence identity; live runner class | Proposed in session status/payload | Must distinguish V1 586 from V2 604 |
| Scenario/name/description/events | `/api/scenarios`, `/api/display` | Existing | Display name plus key in technical detail |
| Seed | session status; V2 provenance/replay | Existing | Holdout seeds blocked from live demo |
| Policy/comparator | session status; V2 policy/checkpoint records | Existing but inconsistent IDs; catalog proposed | Only implemented/configured methods |
| Checkpoint root/transition/hash | `/api/v2/final-holdout/provenance` | Existing for evidence; proposed live binding | Full hash in detail, short hash in ledger |
| Session state/time/step | status and WebSocket payload | Existing | Idle/running/paused/completed/error |
| Topology routers/links | `/api/topology` | Existing | Internal graph unchanged |
| City/role/internal ID | `/api/display`, topology | Existing; display layout proposed | City and role lead, ID secondary |
| Curated coordinates/bend points/label offsets | none | Proposed display-only registry under `/api/display` | Never edit topology config for aesthetics |
| Link capacity/direction/weight/propagation | topology and live snapshot | Existing | Capacity per direction |
| Link load/utilization/up/congestion/available/LSPs | live snapshot | Existing V1; proposed V2 snapshot serializer | Worst direction summary must be named |
| Link queue delay/loss | live snapshot | Existing modeled V1; proposed V2 snapshot serializer | Label as modeled |
| Prior link values/delta | client retained prior snapshot | Existing for uninterrupted UI; proposed payload sequence/generation guard | Never compare across reset/generation |
| Demand base/offered/carried traffic | traffic classes/live snapshot | Existing V1; proposed V2 serializer | Name which traffic measure is shown |
| Demand class/priority/protection/SLA thresholds | traffic classes/live snapshot | Existing | No invented customer count |
| Current/candidate paths | live demand snapshot | Existing V1; proposed V2 serializer | Full router chain and candidate index |
| Demand delay/loss/SLA/disconnected/bottleneck | live demand snapshot | Existing V1; proposed V2 serializer | Current affected demands ≠ cumulative intervals |
| Reroute/path-change/cooldown | live snapshot/history | Existing V1 | Version label |
| V2 path age/dwell/previous path/projected gross | V2 engine | Proposed V2 serializer/decision payload | Not backfilled into V1 |
| FRR/restoration/reversal/flap/moved bandwidth | V2 interval/info and evidence tables; partial V1 history | Proposed typed live event timeline | Separate built-in FRR from TE actions |
| Current action/decoded action | live decision; recorded trace action | Existing | Recorded trace may lack path decoding |
| Action mask boolean/count | live V1 decision count; V2 trace count; V2 env matrix | Proposed full versioned decision payload | Source-specific availability |
| Per-action rejection reason | validator | Proposed `GET /api/session/decision` mask reason list | UI never reverse-engineers reasons |
| PPO selected/top probabilities | V1 live `server/session.py` | Existing V1; proposed V2 decision payload | Valid masked distribution only |
| PPO entropy/value | policy object | Proposed V2/V1 diagnostic fields | Show unavailable when extraction fails |
| Bandit action scores | bandit network | Proposed V2 decision payload | Immediate-reward estimates, not probabilities |
| Bandit epsilon/inference mode | checkpoint config/runtime | Proposed decision/model provenance | Deterministic live demo names epsilon unused |
| Observation 586/604 | env `_obs()` | Proposed versioned decision payload plus schema | Do not store in final evidence API |
| Observation raw semantics/normalization/offsets | env state and YAML schema | Proposed schema API and serializer | Changed ranking is descriptive |
| Safety/mask acceptance reason | validators/decoded action | Existing partial; proposed complete payload | Operator vs environment rejection distinct |
| Clone-based expected telemetry | V1 clone lookahead; V2 `clone()` exists | Existing advisor subset; proposed generic counterfactual endpoint | Label simulated estimate |
| Observed post-action telemetry | live next snapshot/history | Existing | Never populate before step completes |
| Interval/cumulative reward | live decision/history | Existing | Separate fields |
| V2 12 reward components/order/sum | V2 info/evidence reward API | Existing evidence; proposed live V2 payload | Exact-sum indicator |
| V1 reward components | live decision | Existing | Keep V1 labels, no fake V2 padding |
| Action/no-op distributions | `/api/v2/final-holdout/actions` | Existing final evidence | Both no-op grains named |
| Final result and conclusions | `/api/v2/study`, `/api/v2/final-holdout*` | Existing | Both planning statements travel together |
| Development curves/selections | `/api/v2/development/*` | Existing | Development region only |
| Integrity/safety/provenance | `/api/v2/final-holdout/integrity|provenance` | Existing | Fail closed |
| Recorded episode index/steps | `/api/v2/replay/*` | Existing | Refuse non-recorded payload |
| Recorded per-link topology telemetry | absent from trace | Intentionally unavailable | Static reference topology only |
| Incident/recommendation/action timeline | `/api/events`, live history, advisor history | Partial; proposed typed session timeline | Stable event IDs and object refs |
| Explain this moment | client fact templates | Proposed client/domain presenter over typed payloads | No generative hidden rationale |
| Sync fingerprint | engine state/traffic RNG and config | Proposed paired-session metadata | Comparison disabled on mismatch |

## 11. Backend/API gap analysis

### P0: Required for truthful core experience

1. **Capability and policy catalog** — `GET /api/product/capabilities` returns supported sources, environment versions, policies, baselines, checkpoint bindings, output semantics (`probabilities`, `scores`, or neither), clone support, and availability reasons.
2. **Versioned session start** — extend the request with `environment_version`, stable policy IDs, checkpoint IDs, and `comparison_mode`. Preserve current fields for compatibility.
3. **Live V2 demonstration runner** — a read-only-inference runner using `MplsTeEnvV2` and configured frozen checkpoints. It must reject holdout seeds, write no governed artifact, expose no training/evaluation path, and label output `LIVE DEMONSTRATION`.
4. **Exact paired start** — construct one starting engine, clone it for each runner, apply identical interventions, and expose a per-step synchronization fingerprint. Independent same-seed constructors are no longer the sole proof.
5. **V2 snapshot serializer** — JSON-safe router/link/demand/path/metrics state from `SimulationEngineV2`, without changing its transitions.
6. **Decision observatory payload** — `GET /api/simulation/decision` returns observation/schema version, prior observation, action mask and validator reasons, policy-output type and values, selected action, safety result, transition fields, reward components, exact sum, next observation, and provenance. Unsupported fields are nullable with a reason.
7. **Display-only map metadata** — add curated city coordinates, link bend points, and label offsets to `/api/display` or a new GET-only display route.
8. **Typed session timeline** — `GET /api/simulation/timeline` returns stable event IDs and before/after facts for congestion, SLA risk, failure, FRR, recommendation, action, recovery, and stabilization.

### P1: Required for full Decision Lens and robust context

9. **Generic counterfactual preview** — `POST /api/simulation/counterfactual` accepts current session generation, step, and a valid action; evaluates action and no-op on clones; returns `kind: simulated_estimate`, source fingerprint, metrics, and reward estimate. It never advances the session.
10. **Schema endpoint** — `GET /api/rl/schema?environment=v1|v2` returns observation groups, offsets, transforms, action formula, reward names/order, and semantic definitions from the real Python/YAML sources.
11. **Object snapshot endpoint** — a focused GET route for router/link/demand/path detail reduces repeated client derivation and names field availability by environment.
12. **Session generation/sequence** — every live payload carries session ID, generation, monotonically increasing sequence, and step so prior/current deltas cannot cross reset or reconnect boundaries.

### P2: Useful after core acceptance

13. **Stable deep-link resolver** — resolve a `MomentRef` to the nearest available live/recorded/evidence context.
14. **Server-generated printable session summary** — optional deterministic HTML/JSON built from stored live history, not required for initial migration.

### Explicitly not required

- No new training, evaluation, tuning, checkpoint selection, or evidence-writing endpoint.
- No generative explanation service.
- No per-link data synthesis for V2 replay.
- No change to `/api/v2/*` arithmetic or stage types.

## 12. Component inventory

### Shell and context

- `AppShell`
- `PrimaryModeControl`
- `ContextLedger`
- `ProvenanceStamp`
- `SourceSwitcher`
- `ConnectionState`
- `GlobalCommandHelp`
- `ExplainMomentPanel`
- `DirectLinkControl`

### Shared topology and time

- `AtlasStage`
- `TopologyCanvas`
- `TopologyObjectList`
- `NodePlate`
- `LinkRail`
- `PathOverlay`
- `DecisionLensOverlay`
- `IncidentTimeband`
- `IncidentBookmark`
- `MomentCursor`
- `TopologyLegend`

### Presentation

- `MomentRail`
- `RecommendationCard`
- `ComparisonLane`
- `StoryProgress`
- `PresenterCockpit`
- `AudienceViewToggle`
- `QuestionJumpMenu`
- `GovernedConclusionDrawer`

### Network Information

- `NetworkFilterBar`
- `ObjectInspector`
- `RouterInspector`
- `LinkInspector`
- `DemandInspector`
- `PathInspector`
- `DemandRiskTable`
- `TelemetryDeltaRow`
- `RestorationSequence`
- `ModeledRealityDisclosure`

### RL Information

- `InferencePipeline`
- `ObservationInspector`
- `FeatureRow`
- `ChangedFeatureRanking`
- `ActionGrid`
- `MaskReason`
- `PpoProbabilityPlot`
- `BanditScorePlot`
- `SelectedActionPanel`
- `SafetyValidationPanel`
- `RewardWaterfall`
- `ExactSumIndicator`
- `CounterfactualPanel`
- `ModelProvenancePanel`
- `EvidenceStageRegion`
- `RecordedReplayPlayer`
- `UnavailableEvidenceState`

### Feedback and accessibility

- `StatusMessage`
- `InlineError`
- `EvidenceOutage`
- `LoadingSkeleton` with static geometry
- `Drawer`
- `Dialog`
- `SkipLinks`
- `LiveRegion`

## 13. Application state model

```text
AppState
├─ route
│  ├─ mode: presentation | network | rl
│  ├─ workflow: none | guided_story
│  └─ rlView: decision | study | provenance
├─ source
│  ├─ kind: live_session | recorded_replay | development_evidence | final_holdout_evidence
│  ├─ availability
│  └─ provenance payload
├─ context
│  ├─ environmentVersion
│  ├─ scenario
│  ├─ seed
│  ├─ policy
│  ├─ checkpoint
│  ├─ comparator
│  ├─ sessionId / evidenceId
│  ├─ generation
│  ├─ step / recordedStep
│  └─ time
├─ selection
│  ├─ objectType
│  ├─ objectId
│  ├─ eventId
│  └─ actionId
├─ playback
│  ├─ state: idle | running | paused | completed | error
│  ├─ speed
│  └─ synchronization
├─ story
│  ├─ active
│  ├─ beat
│  ├─ reviewBeat
│  ├─ bookmarks
│  └─ awaitingDecision
├─ filters
│  ├─ network
│  └─ rl
└─ ui
   ├─ audienceView
   ├─ fullscreen
   ├─ openDrawer
   ├─ explainDepth
   └─ reducedMotion
```

Invariants:

- `source.kind` is always present after boot.
- `mode` changes never mutate `source` or live engine state.
- `generation` changes clear prior-delta, pending recommendation, and MomentRef state from the old generation.
- `recorded_replay` implies `live === false` and disables execution controls.
- `development_evidence` and `final_holdout_evidence` cannot populate the same `EvidenceStageRegion` instance.
- `awaitingDecision` disables play, step, scenario change, and source change until approve/reject/cancel resolves it.

## 14. Precise layouts

### 14.1 Presentation — 1920×1080

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ RL-in-MPLS  [Presentation] [Network Information] [RL Information]   ?   ⛶ │ 64
├───────────────┬──────────────────────────────────────────────────────────────┤
│ LIVE ▶        │ V2 · demo_evening · seed 42 · Bandit r42/250k vs Greedy   │ 44
├───────────────┴──────────────────────────────────────────────────────────────┤
│ 20:15  FAILURE      L20 Kayseri–Samsun       action  D13 → p2              │ 64
│ interval reward +1.34        cumulative +18.7        2 SLA risks           │
├───────────────────────────────────────────────────────────────┬──────────────┤
│                                                               │ WHAT CHANGED │
│                                                               │ link failed  │
│                 CURATED TURKEY TOPOLOGY                       │ 4 FRR moves  │
│                 (about 68% of viewport)                       │ route pressure│
│                                                               ├──────────────┤
│          stable nodes · route rails · incident focus          │ COMPARISON   │
│                                                               │ Bandit / G   │
│                                                               │ actual actions│
├───────────────────────────────────────────────────────────────┴──────────────┤
│ Masked Bandit suggests [actual action]                                      │
│ old route ─────────  proposed route ━━━━━  SIMULATED ESTIMATE  safety ✓     │ 148
├──────────────────────────────────────────────────────────────────────────────┤
│ 17:00 ───── pressure ── recommendation ── surge ── FAILURE ● ─ repair ──── │ 40
├──────────────────────────────────────────────────────────────────────────────┤
│ ◀ Beat 9/11 ▶   Play/Pause   Approve   Incident bookmarks   Q&A   Audience │ 56
└──────────────────────────────────────────────────────────────────────────────┘
```

Audience view removes the final cockpit row and expands the topology/recommendation area.

### 14.2 Network Information — 1440×900

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ RL-in-MPLS  [Presentation] [Network Information] [RL Information]          │
├──────────────────────────────────────────────────────────────────────────────┤
│ LIVE · V1 · link_failure · seed 42 · PPO vs Greedy · 18:10 · paused         │
├──────────────────────────────────────────────────────────────────────────────┤
│ [class: all] [SLA risk] [failed/degraded/recovering] [paths]  Search /      │
├───────────────────────────────────────────────────┬──────────────────────────┤
│                                                   │ LINK                    │
│          CURATED TURKEY TOPOLOGY                  │ Ankara → Kayseri        │
│                                                   │ L11 · P2→P5             │
│  selected L11 + affected demand/path overlays     │ 2.0 Gbps · 93% · +8 pp │
│                                                   │ delay/loss/LSPs/status  │
├───────────────────────────────────────────────────┼──────────────────────────┤
│ Demand & SLA risk table                           │ current / previous delta │
│ city pair · class · offered · path · SLA · dwell  │ primary / alternates    │
├───────────────────────────────────────────────────┴──────────────────────────┤
│ 17:00 congestion ─ 18:00 FAILURE ─ FRR ─ recommendation ─ recovery ────── │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 14.3 RL Information — 1440×900

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ RL-in-MPLS  [Presentation] [Network Information] [RL Information]          │
├──────────────────────────────────────────────────────────────────────────────┤
│ LIVE · V2 · demo_evening · seed 42 · Bandit r42/250k · step 23 · paused    │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Decision Observatory] [Governed Study] [Model Provenance]                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ OBSERVATION → MASK → SCORES → ACTION → SAFETY → TRANSITION → REWARD → NEXT │
├──────────────────────────────────────┬───────────────────────────────────────┤
│ Observation 604                      │ Action space 69                       │
│ Search /  changed  selected demand   │ no-op / D1..D17 × p0..p3             │
│ group · feature · raw · norm · Δ     │ valid · reason · score               │
│ changed-feature ranking              │ chosen / runner-up / no-op           │
├──────────────────────────────────────┼───────────────────────────────────────┤
│ Selected action & safety             │ Reward waterfall                     │
│ actual decoding · mask · validator   │ 12 ordered components                │
│ old → simulated → observed           │ sum = reward · residual · EXACT      │
├──────────────────────────────────────┴───────────────────────────────────────┤
│ checkpoint hash · root · transition · schema versions · integrity           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 14.4 Tablet — 768 px

```text
┌──────────────────────────────┐
│ RL-in-MPLS   mode menu   ⋯  │
│ LIVE · V2 · 20:15 · paused │
├──────────────────────────────┤
│ moment rail                  │
├──────────────────────────────┤
│ topology                     │
│ fixed 16:9-ish stage         │
├──────────────────────────────┤
│ recommendation / selection   │
├──────────────────────────────┤
│ timeline                     │
├──────────────────────────────┤
│ [Open inspector] [Explain]   │
└──────────────────────────────┘
```

Network and RL inspectors open as bottom sheets at 768 px. The selected-object summary remains inline.

### 14.5 Narrow — 390 px

```text
┌──────────────────────┐
│ RL-in-MPLS       ☰  │
│ FINAL EVIDENCE      │
├──────────────────────┤
│ Mode: RL Information│
├──────────────────────┤
│ current moment      │
├──────────────────────┤
│ topology viewport OR│
│ accessible object list│
├──────────────────────┤
│ focused object/card │
├──────────────────────┤
│ primary content     │
│ one column          │
└──────────────────────┘
```

Primary mode controls become a labeled menu but remain the first navigation group. Tables scroll inside their region; the page itself never scrolls horizontally.

## 15. Motion specification

| Event | Motion | Duration | Reduced motion |
|---|---|---:|---|
| Hover/focus/control | color/keyline state | 120–160 ms | immediate |
| Mode change | topology stays; inspectors cross-fade/slide 8 px | 220–280 ms | immediate, focus moves to mode heading |
| Drawer/dialog | translate from owned edge | 240–320 ms | immediate |
| Link pressure change | interpolate line color/pressure ticks once | 400–600 ms | discrete swap |
| Failure | line breaks once; × appears | 500–700 ms | discrete broken line |
| Recovery | segmented line resolves to solid | 500–800 ms | discrete state plus text |
| Decision Lens | proposed path draws once, old path remains | 450–650 ms | both paths appear with labels |
| Executed reroute | old path fades to ghost; observed path settles | 500–800 ms | before/after labels update |
| Recommendation | enters after causal event | 220–280 ms | immediate |
| Timeline advance | cursor moves to next exact interval | 200–320 ms | discrete |
| Telemetry update | numeric cross-fade in fixed box | 120–180 ms | immediate |

Performance target is 60 fps on the supported demo machine. No animation owns the meaning; text and state are complete before motion starts.

## 16. Responsive behavior

| Width | Presentation | Network Information | RL Information |
|---:|---|---|---|
| 1920×1080 | Full stage, comparison rail, hideable cockpit | Full stage + inspector + table | Two-column observatory + pipeline |
| 1440×900 | Stage + narrower moment rail | Stage 60–65%, right inspector | Two-column, provenance footer |
| 1280 | Comparison collapses below stage | Inspector becomes overlay drawer | One main column + secondary drawer |
| 768 | Cockpit bottom sheet; recommendation inline | Topology + risk summary; sheets | Pipeline scrolls horizontally inside its own region; inspectors stack |
| 390 | One-column story/moment/topology | Topology or accessible list toggle | One stage at a time with persistent pipeline step selector |

At every width:

- no page-level horizontal overflow;
- topology has a meaningful minimum height;
- city labels remain readable or move to the accessible list, never microscopic;
- context abbreviates labels but retains full provenance word and source details on activation;
- controls never overlay topology nodes;
- browser zoom and text resize preserve control access.

## 17. Accessibility requirements

### Semantics and keyboard

- Landmarks: header, primary navigation, context status, main, complementary inspector, footer.
- Primary mode control uses tabs only if the panels share one document context and proper roving focus; otherwise use navigation links with `aria-current=page`.
- Canvas topology has a synchronized semantic tree/list.
- Every drawer traps focus while open and returns focus on close.
- Error and provenance state changes use appropriate live regions; telemetry ticks do not spam announcements.
- Incident timeline events are buttons/links with time in their accessible name.

### Non-color encoding

- Provenance: word + icon + pattern.
- Failure: broken line + × + label.
- Congestion/SLA risk: pressure ticks/triangle + numeric value.
- Recovery: segmented-to-solid convention + recovery label.
- Policy series: letter token, label, line style, and color.
- Selection: outer keyline plus pointer notch.

### Contrast and graphics

- Text contrast ≥ 4.5:1, large text ≥ 3:1.
- Meaningful link/node boundaries and focus indicators ≥ 3:1 against adjacent colors.
- Selected, focused, and failed states must remain distinguishable under common color-vision deficiencies and grayscale.

### Motion and cognition

- Respect `prefers-reduced-motion` globally.
- No timeout for reading a recommendation or evidence conclusion.
- Presenter-controlled automatic progress pauses at decisions and modals.
- Plain-language and technical labels are adjacent, not hidden in hover-only tooltips.

## 18. Failure, empty, and unavailable states

| State | Required behavior |
|---|---|
| No live session | Explain what a session is; offer only configured scenarios/policies; no zero telemetry |
| WebSocket lost | Freeze displayed time, mark connection lost, retain last values as `last received`, reconnect with backoff |
| Session error | Halt playback; name error, scenario, step, and recovery action; do not display data as current |
| Evidence 503 | Name evidence error; show no approximate values |
| Replay traces unconfigured | Catalogue 315 episodes; explain `V2_FULL_ARTIFACTS`; disable load |
| Recorded field absent | State unavailable field and trace grain |
| Checkpoint unavailable | Show policy in evidence context but disable live selection with reason |
| Sync mismatch | Pause paired comparison; show mismatch fields; no comparative verdict |
| No valid reroute | Show no-op as legal and explain mask state; do not call it indecision |
| Counterfactual unavailable | State source/engine limitation; no forecast placeholder |
| Filter removes all objects | Show active filters and `Clear filters` |

## 19. Migration strategy

### Phase A — foundations without route replacement

- Add shared tokens, context types, adapters, capability catalog, display layout metadata, and route-independent tests.
- Serve the new shell at an internal migration route such as `/app`.
- Leave `/`, `/advanced`, `/present`, and `/study` unchanged.

### Phase B — Presentation and Guided Story

- Migrate current live behavior, keyboard controls, comparison, cloned lookahead, story truth, fullscreen, print, and failure handling.
- Add V2 live demonstration only behind capability detection.
- Route `/present` to the shell after presentation parity and 16:9 acceptance.

### Phase C — Network Information

- Migrate topology, interventions, LSP/link tables, current/prior deltas, filters, object inspection, and timeline.
- Route `/` and `/advanced` to Network Information after API and accessibility parity.

### Phase D — RL Information and evidence

- Build the Decision Observatory and schema views.
- Port the sealed V2 study without changing evidence arithmetic or stage separation.
- Port recorded replay with aggregate-only rules.
- Route `/study` to RL Information → Governed Study.

### Phase E — retirement

- Run route contract, full suite, accessibility, visual regression, offline, and evidence integrity gates.
- Remove legacy frontend modules only when no route or test imports them.
- Preserve Git history; do not delete evidence or experiment artifacts.

## 20. Test and acceptance plan

### 20.1 Unit tests

- SourceKind exhaustive handling and provenance wording.
- Route-to-mode/source mapping.
- mode switch preserves context and selection.
- generation change clears stale deltas and pending decisions.
- PPO probability and bandit score formatters cannot cross-label.
- two no-op grains require full labels.
- observation schema groups cover exactly 586 or 604 offsets with no overlap/gap.
- 69-action decoding covers no-op plus `17×4` exactly.
- reward waterfall uses all 12 V2 components in defined order and exact sum.
- recorded adapter rejects `live !== false` or wrong `kind`.
- final and development adapters cannot combine series.
- topology display coordinates do not modify internal IDs or scientific topology.
- Explanation templates cite only present facts and emit unavailable states.

### 20.2 API and contract tests

- Capability catalog matches actual configured runners/checkpoints.
- Live V2 session rejects holdout seeds and never writes under governed paths.
- Paired runners begin from an identical cloned fingerprint and remain traffic-synchronized.
- Manual/scripted interventions apply to all paired engines.
- Decision payload observation/mask/action/reward shapes match environment versions.
- Per-action mask reason agrees with authoritative validator.
- PPO probabilities sum to one across valid actions within tolerance.
- Bandit scores are returned as unnormalized scores with no probability field.
- Counterfactual clones leave real session state, RNG, history, and step unchanged.
- V2 snapshot serializer agrees with engine arrays and metrics.
- Timeline event IDs and sequence survive reconnect; reset changes generation.
- `/api/v2/*` remains GET-only and fail closed.

### 20.3 Component and interaction tests

- Primary modes are unmistakable and exactly three.
- Guided Story is reachable only within Presentation.
- Audience view hides cockpit but retains provenance and story state.
- Decision Lens never executes an action.
- Approve/reject state machine disables conflicting controls.
- Link/demand/table selection is bidirectional.
- Screen-reader topology list and canvas share selection.
- Deep links open the correct mode/source/object/event.
- Recorded replay exposes no execution control or colored link replay.
- Unavailable states are reachable and actionable.

### 20.4 Visual regression viewports

- 1920×1080 Presentation, cockpit shown and hidden.
- 1440×900 all three modes.
- 1280×800 all three modes with drawer behavior.
- 768×1024 all three modes.
- 390×844 all three modes.
- 200% zoom at 1280 px.
- Reduced motion, Windows high contrast, grayscale, and deuteranopia simulation.

No screenshot may show horizontal page overflow, clipped city plates, unstable KPI widths, control overlap, illegible topology labels, or evidence state encoded only by color.

### 20.5 Accessibility testing

- Automated WCAG scan for every route/state.
- Full keyboard walkthrough of mode switch, topology list, story, drawers, Decision Lens, filters, action grid, replay, and study.
- Screen-reader smoke tests for provenance, topology alternative, recommendation, reward sum, evidence outage, and recorded state.
- Focus order and restoration after dialogs/drawers.
- Contrast measurement for all semantic combinations.

### 20.6 Performance and offline

- Initial shell loads with network disabled after local server start; no external request.
- Cytoscape redraw and inspector updates stay under the 16.7 ms frame budget for ordinary ticks on the demo machine.
- No continual animation or background chart work while paused.
- Resize and orientation changes do not create overflow.
- Memory remains bounded through a full 288-step episode and replay scrubbing.

### 20.7 Scientific regression gates

Run the existing evidence, presentation/API, freeze/pin, compatibility, and full test suites. Verify before and after:

- no changes under `results/`, `runs/`, `models/`, scientific `configs/`, learner/training modules, or frozen manifests;
- both temporal-planning conclusion halves remain together;
- PPO's `deceptive_local_optimum` win remains visible;
- both no-op grains and both wall-time grains remain separate;
- replay refuses non-recorded payloads;
- protected manifest content and hash are unchanged from preflight.

### 20.8 Acceptance coverage targets

| Area | Target |
|---|---|
| New pure state/adapters/formatters | ≥ 95% branch coverage |
| New backend serializers and decision contracts | ≥ 90% branch coverage |
| Critical provenance, evidence, action, mask, and counterfactual paths | 100% named behavior coverage |
| UI route/mode/source combinations | Every supported combination has at least one interaction test |
| Viewports/accessibility states | Every listed viewport and state has a recorded acceptance result |

## 21. Prompt 2 phased implementation

The detailed executable plan is in `docs/superpowers/plans/2026-07-31-unified-rl-in-mpls-ui.md`.

1. Guardrails and contracts.
2. Capability catalog, display metadata, and versioned schemas.
3. Typed context store and source adapters.
4. Exact paired session and V2 live demonstration runner.
5. Versioned snapshot, decision, timeline, and counterfactual APIs.
6. Shared Dispatch Atlas shell and topology.
7. Presentation and Guided Story migration.
8. Network Information migration.
9. RL Decision Observatory.
10. Governed Study and recorded replay migration.
11. Responsive, accessibility, performance, and offline hardening.
12. Route cutover, legacy retirement, full verification, and release documentation.

Every phase ends in a working, reviewable slice and preserves old routes until its parity gate passes.

## 22. Features intentionally excluded

- A fourth primary mode for Guided Story, Study, Settings, Training, or Replay.
- Training controls in the redesigned product shell.
- V2 training, tuning, resume, evaluation, reselection, sweep, retry, or new holdout access.
- New policies, baselines, reward terms, observation features, actions, scenarios, protocol behaviors, or scientific metrics.
- A real-operator or exact-GIS topology claim.
- Packet, TCP, RSVP-TE, IGP convergence, label-stack, or production control-plane simulation.
- Generative/LLM explanations, anthropomorphic policy copy, causal feature attribution, or model “confidence” not exposed by the learner.
- Fake link-level recorded replay, invented paths, synthetic telemetry, or missing-value interpolation.
- More than two live synchronized policies at once.
- Automatic live rewind or branching session history.
- A new frontend framework, build pipeline, CDN, external font, WebGL engine, or design dependency during Prompt 2.
- Decorative glass, glow, parallax, typewriter effects, ambient animation, gamification, or marketing landing-page sections.
- Mobile feature parity for dense multi-panel inspection; 390 px preserves core truth and access through progressive disclosure.
- Editing frozen evidence, manifests, checkpoints, experiment worktrees, or scientific configuration.

## 23. Self-review

- No unresolved placeholder, fabricated value, invented policy, or deferred product decision remains.
- The three primary modes and Guided Story relationship are explicit.
- Live, recorded, development, and final evidence are distinct types throughout.
- V1 live and V2 evidence/live-demonstration capabilities are never silently merged.
- Recorded V2 topology limits, no-op grains, reward-component order, action count, observation size, final findings, and negative result are preserved.
- Every requested visible value maps to an existing or proposed source.
- The implementation plan can be written without deciding product or visual behavior anew.
