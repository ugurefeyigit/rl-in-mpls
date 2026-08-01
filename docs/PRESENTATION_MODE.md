# Presentation Mode

Presentation is the audience-facing depth of the unified application at
`/present`. It shares the same session, source, selection, topology, number
formatting, and provenance rules as Network Information and RL Information.
Guided Story is a workflow inside this mode; it is never a fourth primary mode.

## Launch

```bash
python scripts/demo.py
```

The launcher disables training, creates the reproducible demonstration, and
opens `/present`. A manual server also works:

```bash
python -m uvicorn server.main:app --port 8000
```

## Start here

Everything you need is in the persistent left control panel: environment
(**V2** by default), scenario, seed, execution style, controller, optional
comparison, checkpoint root, speed, then **Start run**. Below it are the
transport controls, reset run and full reset, the approve/reject pair when you
chose advisor execution, Guided Story, and the Study evidence region. Nothing
that starts or steers a run lives anywhere else.

A newcomer should be able to pick a scenario and a controller and press Start
run without reading this file.

The persistent source stamp is part of the presentation. `LIVE`, `RECORDED`,
`DEVELOPMENT`, and `FINAL EVIDENCE` are different record types, not cosmetic
badges. Only LIVE can execute a policy or color links from utilization. In the
setup path those records are named in plain language and grouped under **Study
evidence and results**, because a finished study record is not a simulation
setting.

Audience view has a pinned exit control at every viewport, and `Escape` always
leaves it — including from fullscreen, without reloading.

## Composition

The topology is the dominant region. Above it, the current-moment rail shows
phase, clock, incident, action, interval and cumulative reward, busiest link,
and current SLA risk. The right rail explains what changed, the network
condition, and the relationship to the governed study. The incident band,
recommendation card, comparison lane, and presenter cockpit use the same live
snapshot and provenance.

Audience view removes the cockpit, deep analysis, source controls, and context
rail while retaining the topology, current facts, source stamp, and disclaimer.
Fullscreen uses the browser Fullscreen API. Neither option changes the session.

The fixed topology is the proven pre-redesign left-to-right engineering
schematic, optimized for path readability rather than geography. Node plates
lead with city and MPLS role, retain the internal ID below, never move during a
session, and never overlap. The footer reads: fixed engineering schematic, not
geographic; fictional scaled network, not a real operator.

## Presenter controls

| Control | Behavior |
|---|---|
| Play / Pause | Resumes or pauses the real live session |
| Step | Advances exactly one five-minute control interval while paused |
| Next event | Advances through the next real scenario event |
| Speed | Selects `1x`, `5x`, or `20x` live pacing |
| Start / End Guided Story | Starts a fresh `demo_evening`, seed-42, PPO-versus-greedy advisor session or leaves the workflow |
| Back / Next | Reviews a prior beat without rewinding the engine, or advances the real story action |
| Auto story | Optional 6.5-second presenter pacing; can be paused at any beat and never removes manual controls |
| Preview recommendation | Requests the real advisor proposal without mutating the engine |
| Approve / Reject | Applies or rejects only a pending proposal |
| Incident bookmarks | Moves among recorded live events |
| Q&A jumps | Opens the relevant product depth or the governed conclusion |

## Guided Story

The eleven beats are:

1. Establish the fictional network.
2. Read and advance the initial evening interval.
3. Advance until congestion.
4. Inspect the bottleneck.
5. Inspect current SLA risk.
6. Request a policy recommendation.
7. Inspect the observation, mask, safety, route, and estimate.
8. Approve when a recommendation is pending and show the observed outcome.
9. Advance through the real surge/failure sequence and explain built-in FRR.
10. Compare synchronized live lanes.
11. Advance toward repair and open the honest governed-study conclusion.

Starting the workflow always creates its own real `demo_evening` seed-42
session. It does not reuse an unrelated or already-progressed run. Going back
reviews copy only; it does not rewind the engine. Optional automatic pacing
invokes the same Next behavior as the visible control.

The recommendation card says “MaskablePPO suggests …” and includes only values
returned by the engine. Counterfactual telemetry is labeled a simulated
estimate and is computed on a clone with a session fingerprint. If no truthful
estimate exists, the UI says `Outcome estimate unavailable`. Observed telemetry
and actual reward appear only after execution.

## Comparison and governed evidence

The live comparison lane is shown only when the backend proves both runners
share scenario, seed, starting state, and exogenous inputs. A failed proof
disables the verdict and prints the reason. Signed simulation returns are never
converted to percentages.

The final holdout is a read-only record, never an interactive live comparator.
Development and final evidence render in mutually exclusive regions selected by
the source control. Both halves of the planning conclusion stay together.

## Keyboard

| Key | Action |
|---|---|
| `Alt+1`, `Alt+2`, `Alt+3` | Presentation, Network Information, RL Information |
| `Space` | Play or pause |
| `→` | Next step, or next Guided Story beat |
| `←` | Review the prior Guided Story beat |
| `G` | Start or end Guided Story |
| `E` | Explain this moment |
| `[` / `]` | Previous / next incident bookmark |
| `?` | Keyboard help |
| `Esc` | Close the current drawer, then leave fullscreen |

Shortcuts are ignored while typing in a form control. Every shortcut has a
visible control equivalent. See [ACCESSIBILITY.md](ACCESSIBILITY.md) for the
full keyboard and reduced-motion contract.

## Honest unavailable states

- This installation has V1 live checkpoints only; a V2 live demonstration is
  shown as unavailable rather than silently substituting V1.
- Recorded V2 traces contain interval aggregates but no per-link utilization,
  so replay shows a reference topology and never invents link animation.
- PPO entropy and value are unavailable because the live runner does not expose
  them.
- Recorded episodes require `V2_FULL_ARTIFACTS`; without it the catalogue still
  lists all 315 episodes and explains the configuration requirement.
