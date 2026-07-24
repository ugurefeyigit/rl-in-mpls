# Technical report — RL-based traffic engineering on a simulated MPLS backbone

*Generated with the repository at the commit that ships the pretrained
`ppo_te` model. All numbers reproduce with the listed commands and seeds.*

## 1. Problem definition

Given an MPLS backbone with time-varying traffic demands, choose LSP
placements per demand at discrete control intervals to minimize congestion,
delay, loss and SLA violations while limiting control churn. We compare a
learned controller (MaskablePPO) against conventional policies on paired
scenarios and report where each wins, honestly.

## 2. MPLS abstraction

- Each demand is a forwarding-equivalence class (FEC) bound to exactly one
  explicit LSP (ordered router sequence, loop-free by construction).
- k = 4 candidate paths per demand, Yen's algorithm over administrative
  weights, hop cap of ⌈2.5 × min-hops⌉ + 1.
- Ingress/egress LERs are PE routers; transit LSRs are P/AGG routers.
- Link failure triggers local repair (move to best surviving candidate —
  a stand-in for pre-signalled backup LSPs / FRR); controllers reoptimize
  afterwards at their control cadence.
- Not modeled: label stacks, PHP, RSVP-TE/LDP signaling, IGP convergence
  timers, microloops. Reroutes take effect at the next control interval.

## 3. Network model

18 routers (4+4 PEs, 8 P, 2 AGG), 32 undirected links = 64 directed links,
100–2000 Mbps. Flow-level state per directed link: offered load (sum of
demand volumes whose active LSP traverses it), utilization ρ, queueing delay
`min(1.5·ρ′/(1−ρ′), 60) ms` with ρ′ = min(ρ, 0.98), loss 0 below 90%
utilization, quadratic soft loss to 2% at 100%, `1 − 0.98/ρ` above (excess
dropped). End-to-end demand delay = Σ (propagation + 0.2 ms processing +
queueing); demand loss = 1 − Π(1 − loss_link). Upstream drops are not
cascaded downstream (documented conservative simplification).

## 4. Traffic model

17 demands across 6 classes (voice, video, VPN, best-effort, bulk,
critical) with class SLAs (60–400 ms latency, 0.2–5% loss). Volume =
base × diurnal profile(hour) × scenario multiplier × event factor × AR(1)
noise (φ = 0.9, per-class burstiness), all driven by `numpy.default_rng(seed)`.
Traffic is exogenous — independent of routing — so identical
(scenario, seed) pairs give paired comparisons across algorithms.

Scenarios: full_day, morning_ramp, evening_peak, night_consolidation,
flash_crowd, link_failure (L11 backbone), demand_forecast_error,
deceptive_local_optimum (shared hidden bottleneck P5→P8), demo_evening,
random_day (randomized training scenario), ood_double_failure (two
overlapping failures never seen in training), overload_stress (1.6× global).

## 5. RL formulation

- **Observation** (586-dim float32, all in [0, 1]): per directed link
  utilization/2, queue delay/60 ms, loss, up-flag, EWMA utilization; per
  demand volume, priority, SLA thresholds, protected flag, current-path
  one-hot, candidate bottleneck utilizations, cooldown, disconnected flag;
  globals: sin/cos hour, max/mean/std utilization, mean delay, loss, SLA
  fraction, delivered ratio, recent reroutes, episode progress. No future
  information.
- **Actions**: Discrete(69) = no-op + (17 demands × 4 candidate paths), with
  action masking for failed paths, cooldowns (3 intervals), same-path moves
  and bandwidth-infeasible moves of protected classes.
- **Reward** (configs/reward.yaml): + delivered ratio, + priority-weighted
  SLA success, − max-util excess above 60%, − utilization spread, − delay,
  − loss, − SLA violations, − overload, − reroute (0.08), − flap (0.25),
  − invalid action, − disconnections. Components normalized to ~[0,1] and
  logged individually (TensorBoard + UI).
- **Episode**: one scenario; 24 h = 288 steps of 5 simulated minutes
  (5 × 1-minute micro-ticks each). Termination = total disconnection
  (catastrophic); truncation = scenario end.
- **Algorithm**: MaskablePPO (sb3-contrib 2.x), MlpPolicy 256×256, lr 3e-4,
  γ = 0.995, GAE λ = 0.95, clip 0.2, ent 0.01, 8 vec envs, 400k steps,
  seed 42. Training scenario: random_day (randomized bursts/failures per
  episode seed). Training cost: ≈ 30–40 min on a laptop CPU.

## 6. Baselines

| Name | Behavior | Information used |
|---|---|---|
| static | pin every demand to its lowest-admin-weight candidate; return to it after failures | topology only |
| greedy | if any link > 85%: move the largest demand crossing the hottest link to the candidate with the lowest bottleneck utilization (margin 5%, cooldown 3) | current telemetry |
| cspf | every 6 intervals, re-place all demands in priority order onto the cheapest candidate whose bottleneck reservation stays under 90% capacity; 8% hysteresis; ≤3 moves | current telemetry + reservations |
| random | uniformly random valid action (sanity floor) | mask only |

All baselines pass through the same engine validity rules and the same
reward accounting as the RL agent. None sees future traffic.

## 7. Evaluation methodology

`scripts/evaluate.py`: 7 scenarios × 5 seeds (101–105) × 5 algorithms,
paired by (scenario, seed). Reported: mean ± std and 95% t-CI per cell, plus
paired RL-minus-baseline deltas. The demo (seed 42) is separate and
illustrative only. Decision latency is measured per call.

## 8. Results

*(numbers inserted from results/eval_stats.csv — see repository for the
machine-readable files; figures in results/figures/)*

Mean over 5 paired seeds (101–105); full tables with std/95% CI in
`results/eval_stats.csv`, per-episode rows in `results/eval_summary.csv`.
Reward is the shared multi-objective score (§5) accounted identically for
all controllers.

**Episode reward (higher is better)**

| Scenario | static | random | greedy | cspf | **rl** |
|---|---:|---:|---:|---:|---:|
| full_day | −425.6 | −162.9 | 149.9 | 124.7 | **153.8** |
| deceptive_local_optimum | −26.3 | 10.4 | 70.0 | 39.9 | **72.9** |
| evening_peak | −240.0 | −204.3 | **−26.2** | −78.9 | −34.5 |
| flash_crowd | −199.8 | −175.2 | **−79.2** | −86.1 | −94.6 |
| link_failure | −164.4 | −140.3 | **−46.1** | −55.8 | −76.9 |
| ood_double_failure | −84.7 | −82.8 | **−20.6** | −57.4 | −38.1 |
| overload_stress | −210.2 | −204.7 | **−146.4** | −170.4 | −170.3 |

**Mean max-link-utilization / total SLA violations**

| Scenario | static | greedy | cspf | **rl** |
|---|---:|---:|---:|---:|
| full_day | 1.33 / 1345 | 0.87 / 318 | 0.91 / 361 | **0.80 / 201** |
| deceptive_local_optimum | 1.05 / 180 | 0.78 / 20 | 0.88 / 63 | **0.68 / 4** |
| evening_peak | 1.64 / 618 | 1.06 / 186 | 1.15 / 294 | **1.01** / 195 |
| flash_crowd | 1.84 / 504 | 1.34 / **232** | 1.33 / 262 | **1.28** / 264 |
| link_failure | 1.55 / 441 | **1.13 / 173** | 1.14 / 200 | 1.16 / 231 |
| ood_double_failure | 1.06 / 271 | 0.83 / **138** | 0.96 / 199 | **0.83** / 140 |
| overload_stress | 2.40 / 524 | **1.54 / 381** | 1.86 / 426 | 1.58 / 450 |

**Paired RL-minus-baseline deltas pooled over all 35 paired episodes
(mean ± 95% CI):**

| Comparison | Δ reward | Δ mean max-util | Δ SLA violations |
|---|---:|---:|---:|
| RL − static | **+166.2 ± 61.9** | **−0.50 ± 0.07** | **−342.5 ± 119.9** |
| RL − greedy | −12.7 ± 6.7 | −0.03 ± 0.02 | +5.3 ± 23.0 (n.s.) |
| RL − cspf | **+13.8 ± 8.7** | **−0.13 ± 0.04** | **−45.9 ± 24.3** |
| RL − random | **+110.3 ± 37.5** | **−0.37 ± 0.06** | **−242.1 ± 78.1** |

**Reading (the honest version):**

- RL decisively beats static routing and the CSPF reoptimizer, and clearly
  beats random — the policy genuinely learned (training eval went −173 →
  +161 over 400k steps; `results/figures/training_curve.png`).
- RL **wins where anticipation and global coordination pay**: the full
  diurnal cycle (max-util 0.80 vs 0.87, SLA violations 201 vs 318 against
  greedy) and the deceptive-local-optimum scenario built around a hidden
  shared bottleneck (SLA violations 4 vs 20, max-util 0.68 vs 0.78).
- RL **loses to the greedy heuristic on purely reactive scenarios** (flash
  crowd, single link failure, uniform overload): when the correct response
  is simply "move the biggest flow off the hottest link *now*", a
  telemetry-driven heuristic is sufficient and slightly better. Pooled over
  everything, greedy edges out RL on reward (−12.7 ± 6.7).
- **Failure recovery**: mean steps to clear SLA violations after the L11
  failure — greedy 10.0, cspf 12.2, **rl 26.8**, static 48. RL restores
  service far faster than static but noticeably slower than the reactive
  heuristic on this in-distribution failure; interestingly, on the
  out-of-distribution double failure RL recovers in 1.0 step (greedy 0.4).
- **Control churn is RL's visible weakness**: on full_day it rerouted at
  essentially every interval (288 moves, 264 flagged as flaps, vs greedy's
  70). The 0.08 reroute penalty was too cheap relative to the utilization
  gains. This is exactly the kind of behavior a network operator would
  veto, and reproducing it honestly is more useful than hiding it.
- **Decision latency**: RL inference ≈ 0.85 ms/decision (CPU), greedy
  ≈ 0.13 ms, cspf ≈ 0.10 ms — all trivially real-time at a 5-minute control
  interval.

## 9. Failure cases (deliberate)

Shown because the brief demands them, and because they are true:

1. **Flash crowd (partially out-of-distribution).** Training saw flash
   crowds in only ~30% of episodes. At evaluation the RL policy reacts but
   trails greedy by ~15 reward points (−94.6 vs −79.2): it hesitates to make
   the large, obvious move that greedy executes immediately.
2. **Single backbone failure.** RL needs ~27 control intervals to fully
   clear SLA violations vs 10 for greedy. The policy converges on a decent
   allocation but takes several intervals of exploration-shaped small moves
   while cooldowns bind.
3. **Route churn / flapping on calm days.** 288 reroutes per full_day
   episode, 264 counted as flaps. Utilization metrics look great, but this
   trades operational stability for marginal reward — a mis-tuned reward,
   not a law of nature. Raising the reroute penalty is the obvious fix and
   is left visible as future tuning (Experiment 6 in the brief).
4. **Uniform overload.** With 1.6× demand everywhere there is nothing
   clever to learn — every path is wet. Greedy's simple triage matches or
   beats RL (−146 vs −170); RL burns reroutes (48/48 intervals) for no gain.
5. **Safety filter finding.** With action masking active at inference, the
   safety filter never rejected a single action across all evaluation
   episodes (identical results with `--no-safety`; see
   `results/nosafety_*.csv`). It is defense-in-depth for unmasked or stale
   policies, not an active constraint on a masked one — a useful
   architectural observation for deployment design.

## 10. Reward ablations

Two variants were retrained with individual reward terms zeroed
(`scripts/train.py --zero-weight …`, 80k steps each — shorter than the main
400k run, so compare direction, not magnitude; scored below under the FULL
standard reward, 3 seeds, results/abl_*_stats.csv):

| Policy (training reward) | full_day reward | full_day SLA viol. | evening_peak reward | mean max-util (full_day) |
|---|---:|---:|---:|---:|
| main `ppo_te` (400k, full reward) | **153.8** | **201** | −34.5 | **0.80** |
| `ablate_stability` (no reroute/flap penalties, 80k) | 117.3 | 219 | **−31.9** | 0.85 |
| `ablate_congestion` (no loss/overload/max-util penalties, 80k) | −18.4 | 607 | −145.0 | 0.94 |

- **Removing the congestion terms is catastrophic**: SLA violations triple
  on full_day and the evening peak collapses (−145 vs −34.5) — the policy
  happily parks demands on hot links because only delivered-ratio and SLA
  terms push back, too weakly. Reward design demonstrably matters.
- **Removing the stability terms** degrades full_day moderately (117 vs 154
  under the standard reward, which charges it for the flaps it never
  learned to avoid) and matches on evening_peak. Notably the *main* policy
  already reroutes nearly every interval — evidence that at 0.08 the
  reroute penalty is barely load-bearing and would need to be several times
  larger to purchase operational stability (ties into failure case #3).

## 11. Statistical caveats

Five seeds bound experiment runtime on a laptop; CIs are wide accordingly.
Paired scenario design removes traffic variance between algorithms but not
across seeds. No hypothesis tests are claimed beyond the reported CIs;
distributional assumptions (t-CI) are an approximation at n = 5.

## 12. Limitations

1. Flow-level abstraction — no packet dynamics, no TCP feedback, no jitter.
2. Analytic delay/loss curves (documented above); real queues differ.
3. Instantaneous, loss-free path switchover at interval boundaries.
4. Offered traffic independent of congestion (no elastic backoff).
5. Single topology; no claim of topology generalization.
6. Telemetry is perfect and instantaneous (the design allows adding noise
   and delay; not evaluated here).
7. Reward weights hand-tuned once; sensitivity only partially explored
   (§10).
8. The optimizer-style upper bound (LP min-max-utilization) is not
   implemented; CSPF is the strongest classical baseline used.

## 13. Real-world deployment sketch (what this is NOT yet)

Telemetry (SNMP/gNMI streaming, sampled NetFlow → traffic-matrix estimation)
→ feature pipeline → policy inference in **shadow mode** (actions logged,
never applied) → advisory mode (operator approves each action; the safety
filter's constraint checks — capacity, protected classes, cooldowns — stay
mandatory) → limited closed loop via PCEP/SR-Policy or RSVP-TE with
automatic rollback on SLA regression. Digital-twin evaluation, gradual
rollout percentages, and change-freeze windows are prerequisites. The
simulation-to-reality gap (traffic realism, convergence dynamics, partial
telemetry) is the dominant risk; nothing in this repository closes it.

## 14. Operator modes implemented vs conceptual

- **Safe automatic** (implemented): safety filter validates every RL action;
  UI shows proposed vs accepted with reasons.
- **Experimental** (implemented): safety filter off.
- **Advisory / shadow** (conceptual here): the decision panel already shows
  the full recommendation payload an operator would approve; wiring an
  approve/reject gate is future work.

## Acceptance checklist

The 28 criteria from the project brief, verified:

1. ✅ Runs without editing source (`scripts/demo.py`, configs in YAML)
2. ✅ Topology visible (Cytoscape view)
3. ✅ Traffic changes over time (diurnal + noise + events; test `test_traffic_differs_across_seeds_and_time`)
4. ✅ Link utilization responds to traffic (test `test_flow_conservation_on_links`)
5. ✅ Routing choices affect metrics (test `test_actions_change_outcomes_vs_noop`)
6. ✅ Baselines run (static/greedy/cspf/random, evaluated)
7. ✅ Genuine RL agent trains (MaskablePPO, curve −173→+161)
8. ✅ Model save/load (checkpoints, best_model, `/api/agent/load` via model_tag on session start)
9. ✅ Actions from live observations (`AlgoRunner._step_rl` — predict on current obs)
10. ✅ Not timestamp-predetermined (policy input = telemetry; traffic exogenous but decisions computed)
11. ✅ Frontend shows selected actions (decision tape + agent panel)
12. ✅ Old and new paths shown (decision panel, LSP highlighting)
13. ✅ Reward components visible (agent panel bars, TensorBoard)
14. ✅ RL vs baseline on identical scenarios (compare mode, paired eval)
15. ✅ Link failures injectable (UI + API + scripted)
16. ✅ Traffic bursts injectable (UI + API + scripted)
17. ✅ Results exportable (CSV/JSON endpoints + results/)
18. ✅ Multi-seed evaluation (5 seeds, CIs)
19. ✅ Setup instructions work (README quick start; pip freeze in requirements.txt)
20. ✅ README states limitations honestly
21. ✅ No fake charts/static metrics (all UI data from engine snapshots over WS)
22. ✅ Locally demonstrable (Windows laptop, CPU only)
23. ✅ Critical logic tested (32 pytest cases)
24. ✅ Fixed-seed demonstration (`demo.py`, seed 42, disclosed)
25. ✅ Meaningful learned policy shown (deceptive_local_optimum: SLA 4 vs 20)
26. ✅ Limitation/failure case shown (five, §9)
27. ✅ Fuzzie.API decision justified (ADR-001)
28. ✅ Internally coherent (single Python service, one config source of truth)

## 15. Future work

LP/ILP min-max-utilization bound; GNN policy for topology transfer;
continuous split ratios (SAC) for ECMP-style TE; telemetry noise/delay
robustness; multi-agent per-domain control; imitation warm-start from CSPF;
risk-sensitive rewards for tail SLAs.
