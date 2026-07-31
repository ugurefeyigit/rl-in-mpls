# Technical defense — the governed V2 study

Written to be defended out loud. Each section is a question a reviewer is likely
to ask, and the answer the evidence actually supports.

**Author:** Uğur Efe Yiğit. The simulation, the V2 environment, the governance
tooling, both learners' training and evaluation harnesses, the analysis and this
write-up are the author's own work. Third-party libraries are listed in
`THIRD_PARTY_NOTICES.md`.

---

## 1. What question was being asked

> Does a controller that reasons about the *future* of a traffic-engineering
> problem beat one that only reacts to the *present* state?

The V1 work had shown a MaskablePPO agent could beat static and heuristic routing
on some scenarios. That does not answer the question above, because PPO's
advantage could come from temporal credit assignment **or** simply from being a
strong learned state-to-action map.

V2 was designed to separate those. It puts MaskablePPO — which optimises a
discounted return over a horizon — against a **masked contextual bandit**, which
is explicitly myopic: it maximises the immediate reward of the current interval
and has no notion of the future at all. Both see the same observation, share the
same action space, and are constrained by the same authoritative action mask.

If planning matters in this formulation, PPO should win. It did not.

## 2. The system being controlled

An 18-router MPLS backbone simulated at flow level: 4 ingress PEs, 4 egress PEs,
8 P cores, 2 aggregation nodes; 32 undirected links (64 directed) at 100–2000
Mbps; 17 demands across 6 traffic classes with diurnal profiles, seeded AR(1)
noise, bursts, flash crowds and scripted failures.

The topology carries deliberate stress: a hidden shared bottleneck (P5→P8), a
longer-but-better detour region, a redundancy ring, and a 2 Gbps backbone link
(P2–P5) whose failure forces mass rerouting.

**Traffic engineering, concretely.** Each demand is carried by an LSP pinned to
one of *k* precomputed candidate paths. The controller acts once per 5-minute
control interval and may move one demand to a different candidate path. A flow
solver then assigns load; utilization drives analytic delay and loss; SLA
compliance is counted per demand per interval. FRR protection, dwell/hold-down,
and a protected-bandwidth safety check all sit between the controller's intent
and the network's state.

**What the abstraction does not model:** packets, TCP dynamics, RSVP-TE or IGP
convergence, label signalling, or congestion backoff on offered traffic. Reroutes
take effect at the next interval. These are documented in `README.md` and
`docs/REPORT.md` and they bound every claim below.

## 3. The V2 environment and its governance

`MplsTeEnvV2`: 604-dimensional observation, 69 discrete actions (one no-op plus
68 demand/path moves), 12 named reward components whose sum is the operational
return on every step.

The environment definition was **frozen and signed off** at
`dca533b5c6fa9953307d01470c23cac512eb2961` before any V2 learner was trained. A
freeze/pin test suite fails if a scientific definition — reward, observation,
action, topology, scenario, seed, mask, horizon or metric semantics — changes
underneath a trained checkpoint.

Governance ran on three rules:

1. **Preregistration.** Training roots (42, 314159, 271828), the seven scenarios,
   the development seeds (101–105) and the holdout seeds (1001–1005) were fixed
   in advance, along with the checkpoint-selection rule.
2. **Separation of selection from testing.** Every checkpoint decision was made
   on development seeds. The holdout seeds were never constructed, evaluated,
   inspected or debugged with until the single authorized final evaluation.
3. **Fail closed.** Tooling refuses to run rather than silently widen scope. That
   property caused a real, disclosed failure — see §7.

## 4. The two learners

| | MaskablePPO | Masked contextual bandit |
| --- | --- | --- |
| Objective | discounted return over the episode | immediate reward of this interval |
| Temporal credit | yes (GAE, value function) | none by construction |
| Action masking | authoritative mask, same source | authoritative mask, same source |
| Observation | identical 604-d vector | identical 604-d vector |
| Budget | 400,000 transitions per root | 400,000 transitions per root |
| Checkpoints | every 50,000 transitions | every 50,000 transitions |

The bandit is not a weakened PPO. It is the *control condition* for the planning
hypothesis: the one thing it lacks is the ability to trade present reward for
future reward.

Both trained under identical vectorisation (16 environments), identical device
(CUDA, RTX 4070 Laptop), identical seeding discipline, and produced byte-identical
episode-seed ledgers within each root — which is how we know they faced the same
traffic.

## 5. Roots, selection, continuity, holdout

Four stages, in order, with a one-way door at the end:

**Stage 1 — seed-42 pilot** (`ca64b62…`). One training root. Established that the
bandit learns a useful policy (peaking at 22.65) while PPO's checkpoint curve is
non-monotonic (`-8.51, -2.87, 10.33, -16.79, 13.45, 2.86, -8.75, -17.80`). One
root proves nothing about generalization, and the pilot said so.

**Stage 2 — three-root continuity** (`6a8a406…`). Roots 314159 and 271828 added.
Each of the 48 checkpoints was evaluated on development seeds 101–105 across the
seven scenarios. The **preregistered rule — highest valid mean return on the
development seeds — selected six checkpoints**: root 42 → 250k/250k, root 314159
→ PPO 350k / bandit 300k, root 271828 → PPO 150k / bandit 400k. Bandit won all
three roots (25.741 vs 16.699 aggregate).

**Stage 3 — authorization repair** (`f7ed0f4…`). See §7.

**Stage 4 — final holdout** (`f7ed0f4…`). The six fixed checkpoints plus three
baselines, evaluated **once**, on seeds 1001–1005 only, across the seven
scenarios. 35 episodes each, 315 total. No selection input existed in the
workflow: no checkpoint, seed or scenario could be chosen at that point.

> The single most important property of this design: **no holdout number was
> available to anyone at the moment any checkpoint was selected.** Selection was
> complete and committed before the holdout ran.

## 6. What the holdout found

| Method | Return | Delivered | SLA intervals | Reroutes/h | Moved Mbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| masked_bandit | **18.221** | 0.9492 | 174.94 | 2.148 | 1,602.86 |
| maskable_ppo | 9.036 | 0.9459 | 208.17 | 2.148 | 1,291.00 |
| greedy | -2.327 | 0.9444 | 200.34 | 4.913 | 9,963.93 |
| cspf | -28.339 | 0.9347 | 249.57 | 0.636 | 683.18 |
| static | -101.851 | 0.8998 | 353.43 | 0.366 | 205.87 |

Learner rows are the mean of three training-root means. Baselines have no
training root and ran once.

- Bandit beat PPO on **3 of 3** training roots (margins 9.560, 12.250, 5.746).
- Bandit beat PPO in **6 of 7** scenarios. PPO led `deceptive_local_optimum` by
  **1.107**. The largest bandit edge was **20.183** in `link_failure`.
- **Safety held.** Zero invalid actions, mask disagreements, reward mismatches,
  non-finite values, solver failures, protected safety failures. All 315 episodes
  truncated normally. Protected and unprotected disconnection accounting was
  identical across every method, as were FRR disconnections and restorations.
- **Churn is acceptable but not free.** Both learners reroute at 2.148/hour. The
  bandit reverses less (1.58 vs 2.00) and flaps less (0.093 vs 0.118) — but
  **moves more bandwidth** (1,603 vs 1,291 Mbps/episode). That is a real cost and
  it is reported. Greedy moves 9,964 Mbps with 12.89 reversals.
- Every step passed the exact 12-component reward-sum check; largest aggregation
  residual 1.7053e-13.

### The conclusion, stated precisely

> **The frozen evidence does not positively support a need for temporal planning
> in this formulation.** The explicitly myopic learner remained stronger, on the
> untouched holdout, across roots and across nearly all scenarios.

> **This is not evidence that planning is generally irrelevant to MPLS or traffic
> engineering.** It is a result about *these* learners, *this* observation and
> reward design, *this* topology and *these* scenarios.

Both halves travel together everywhere they appear — in `/api/v2/study`, on the
`/study` surface, and in every report. A test enforces it.

**Why the negative result is the more interesting one.** A confirmed "planning
helps" would have been the expected outcome and easy to over-read. Instead the
control condition won, which points at the observation design: if the 604-d
observation already encodes enough of the network's state, a strong myopic map
can be near-optimal for a problem whose actions are largely reversible at the
next interval. That is a hypothesis this study does **not** test — it is V3 work.

## 7. The invalidated run, disclosed

Two things went wrong. Both are preserved and neither contaminated a reported
result.

**The SB3 seed-propagation bug (invalidated).** The first seed-42 PPO run was
thrown away. Its episode-seed ledger exposed roots 42–57 rather than 42:
Stable-Baselines3 forwarded `model_seed + worker_rank` to a V2 environment that
already derived child seeds from `root + worker_rank`, counting rank twice. The
experiment wrapper now preserves the governed root, a focused regression
reproduces the SB3 behaviour, and training fails closed if any recorded root
differs. The failed PPO run and the comparison that included it are preserved
under `runs/v2/` and were not used. The seed-42 bandit run was scientifically
valid but was repeated unchanged so both learners bind to one source SHA.

**The authorization gate (repaired before the holdout).** The original gate
rejected holdout seeds unconditionally *and* required the evaluation checkout SHA
to equal each checkpoint's training SHA. Because the six checkpoints were bound to
**two** approved sources, that made a single safe evaluation impossible. Commit
`f7ed0f4` added an evaluation-only workflow: an immutable six-checkpoint registry;
exact payload, sidecar, root, algorithm, transition and source binding;
descendant-only cross-source loading; an allowlist limited to evaluation,
governance, tests and compact results with scientific-definition changes
rejected; an explicit complete-holdout seed mode; **no checkpoint, seed or
scenario selection inputs**; fail-closed new output directories; and no retry
path. It was committed and pushed *before* the evaluation ran.

This is the honest shape of it: **the gate was changed before the holdout.** What
makes that defensible rather than a loophole is that the change removed selection
capability rather than adding it, was made without any holdout number in hand,
and was verified first — 81 focused learning/compatibility tests, 14 freeze/pin
tests, and the 440-test full suite, all passing, before access.

Three run statuses are kept distinct and never merged:
**invalidated** (scientifically void), **superseded** (valid, replaced for
source-identity reasons), **repaired** (a tooling defect fixed before the gate).

## 8. Limitations

- **Three training roots.** Direction was stable (3/3) but magnitude was not
  (5.746 to 12.250).
- **Five holdout seeds, seven scenarios, one topology.** Nothing here shows
  topology generalization.
- **Scenario heterogeneity dominates variance.** Episode SD is ~155 for both
  learners while the root-mean SD is ~2–3. Aggregate dispersion is mostly "which
  scenario", not "which learner".
- **Baselines ran once** — they have no training root, so their rows carry no
  root spread.
- **One reward design.** The conclusion is about the objective as specified;
  a different weighting could change which learner wins.
- **Flow-level abstraction** with an instant control plane (§2).
- **The result cannot establish that memory or planning would never help** under a
  different learner, observation design, action space or task.

## 9. Questions to expect

**"Did you tune anything on the holdout?"** No. `manifest.json` asserts
`training_performed`, `tuning_performed`, `checkpoint_selection_performed`,
`checkpoint_sweep_performed` and `holdout_used_for_debugging` are all false, and
the evidence loader refuses to serve the data if any of them is not false. The
holdout workflow accepts no selection input.

**"Why is the bandit's win not just PPO being under-trained?"** PPO's instability
is documented, not hidden — its curve is non-monotonic on every root and its
selected checkpoints are non-monotonic across roots (250k, 350k, 150k). The
honest reading is that PPO was harder to train stably in this formulation under
an equal budget. That is a finding about the difficulty of the temporal problem,
not a claim that no PPO configuration could do better. A better-tuned PPO is
explicitly V3 work and would need its own preregistration.

**"Isn't 3 roots too few?"** Yes for a magnitude claim; the report says so. It is
enough for a direction claim only in combination with the 6/7 scenario split and
the untouched holdout.

**"Why is greedy negative?"** The reward charges for utilization, SLA severity and
movement. Greedy achieves comparable delivery but pays heavily for churn — 12.89
reversals and 9,964 Mbps moved per episode. The learners' advantage is
substantially "same outcome, far less thrash".

**"What would change your mind?"** A preregistered V3 in which a recurrent or
planning learner, given an observation that genuinely hides future-relevant state,
beats the myopic control condition on untouched seeds.

## 10. Where the evidence lives

| What | Where |
| --- | --- |
| Final holdout, compact | `results/v2_final_holdout/` |
| Continuity (development) | `results/v2_three_root_continuity/` |
| Seed-42 pilot (development) | `results/v2_seed42/` |
| Independent reconciliation | [V2_EVIDENCE_AUDIT.md](V2_EVIDENCE_AUDIT.md) |
| Closeout narrative | `NEXT_STAGE_HANDOFF.md` |
| Browsable record | `/study` (see [PRESENTATION_MODE.md](PRESENTATION_MODE.md)) |
| Unapproved future work | [V3_RESEARCH_BACKLOG.md](V3_RESEARCH_BACKLOG.md) |

Full step traces and every checkpoint stay outside Git, in the preserved
experiment worktrees named in each manifest.
