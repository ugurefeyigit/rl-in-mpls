# Hostile reviews

Three adversarial reviews, written to find weaknesses before a real reviewer
does. **Round 1** reviews the inherited closed study plus the state of this
study at 01:00 UTC on 2026-10-07 (reproduction finished, E1 partial, E2–E6
running). Each point is answered with what was done about it. **Round 2**
(below) re-reviews the finished paper.

---

## Round 1

### A. Networking (TE/MPLS) reviewer

| # | Criticism | Severity | Response / action |
|---|---|---|---|
| N1 | "One TE change per 5 minutes, network-wide" is not how MPLS-TE controllers work; PCE/SDN controllers re-optimize many LSPs at once. The single-move rule *creates* the sequential structure you then study. | major | Accepted as the central scoping caveat. The paper frames the problem as *incremental* reconfiguration (make-before-break churn limits, change windows) and states that the conclusions are about this formulation. The MILP-track baseline shows what one-shot optimization achieves when it is forced through the same single-move constraint. |
| N2 | Why learn at all? A min-max-utilization LP/MILP per interval is the standard TE answer and was missing. | major | **Fixed.** Added MILP-track (HiGHS). Result: statistically indistinguishable from the bandit, better than PPO. This is now a headline result, not a footnote. |
| N3 | The traffic model is synthetic (hand-written diurnal curves, AR(1)), traffic is inelastic, and the "deceptive local optimum" scenario is named by intent. | major | Accepted (`docs/LIMITATIONS.md` 1–2, 6). The deceptive scenario's PPO advantage did not reproduce; the paper no longer treats that scenario as special without measured evidence. |
| N4 | Candidate paths are precomputed (k = 4) and never change; real CSPF computes paths online against reservations. | moderate | Accepted, stated. Path-count sensitivity not run (environment frozen; changing k changes the action space). |
| N5 | Zero actuation latency: the controller's change carries traffic in the same interval. | major for the question asked | Turned into an experiment (E4, delayed activation). |
| N6 | The reward is a weighted sum of 12 terms chosen by the authors; operators care about SLA violations, utilization headroom and churn separately. | moderate | Operational metrics (delivered ratio, SLA-violating demand-intervals, max utilization, reroutes/h, moved bandwidth) are reported for every policy next to the return. |
| N7 | 18 routers / 17 demands is small; scalability is not addressed. | moderate | Accepted; no scaling claim is made. Inference latency is reported. |

### B. RL reviewer

| # | Criticism | Severity | Response / action |
|---|---|---|---|
| R1 | Three training roots cannot support "bandit beats PPO". | major | Primary comparison extended to 5 roots; root-level intervals and sign counts reported; the reproduction itself demonstrates how wide PPO's root distribution is. |
| R2 | PPO was run with one untuned configuration inherited from a different (V1) environment. | major | E3: 8-configuration one-factor sensitivity on validation seeds, best configuration retrained on fresh roots (E3b). Bandit deliberately left untuned. |
| R3 | Horizon and algorithm are confounded: PPO differs from the bandit in γ *and* in being an on-policy policy-gradient method. | major | E2: PPO with γ ∈ {0, 0.9} and a masked Q-learner with γ ∈ {0.5, 0.9, 0.99} that is otherwise identical to the bandit (γ = 0 is bit-identical). |
| R4 | The bandit is trained on a shaped reward; potential-based shaping is policy-invariant only for γ = 0.995, so the "myopic" learner gets one step of lookahead on Φ for free. | moderate | E6: bandit trained with the shaping coefficient set to 0, evaluated on the primary reward. |
| R5 | Checkpoint selection and reporting used the same continuity seeds in the closed study. | moderate | New protocol: validation seeds for selection, disjoint test seeds for reporting; final-checkpoint results reported too. |
| R6 | "Zero invalid actions" is meaningless if the wrapper aborts on any invalid action. | moderate | Correct. Independent mask audit (1M+ state-action checks; independent projection recomputation; PPO probability mass on illegal actions). |
| R7 | Is the problem even sequential? If not, the comparison is uninformative. | major | Sequentiality audit: clairvoyant and frozen-exogenous rollouts; oracle ladder (H = 1, 3, 6) as policies; model-fidelity analysis of what each learner's actions are worth over 1 and 6 intervals. |
| R8 | Training/evaluation distribution shift (train on `random_day`, test on scripted scenarios). | minor | Stated; it applies identically to both learners. |
| R9 | Budget: 400k transitions may favour the sample-efficient method. | moderate | E5: 1.2M-transition runs; learning curves reported. |

### C. Paper reviewer (venue-agnostic ML/networking)

**Summary.** A simulation study comparing PPO with a masked contextual bandit
for incremental MPLS-TE, with a reproduction of a prior internal study.

**Strengths.** Unusually careful engineering (frozen definitions, exact
reproducibility of the simulator, paired evaluation); honest reporting of a
non-reproduction.

**Weaknesses.** (W1) Novelty: Teal and DOTE already argue that TE is
essentially one-shot and that RL is overkill; "a bandit beats PPO" is
expected. (W2) A single small topology. (W3) No theory. (W4) Risk that the
result is about PPO's optimization rather than about TE.

**Questions.** Does any γ > 0 learner beat the bandit? Is there any regime in
which sequential RL wins? How far are the learners from a strong
optimization baseline?

**Score rationale (round 1).** Weak reject as an ML paper unless the
horizon/algorithm attribution and a regime where sequential RL is necessary
are shown; borderline accept for a networking workshop as a careful negative
result.

**Actions.** W1 → the paper positions itself relative to Teal/DOTE (the
persistent-state, incremental, costed setting those papers do not cover) and
the contribution is the *measurement* and *attribution*, not "simple wins".
W4 → E2/E3. "Regime where sequential RL is necessary" → E4. W2/W3 → stated as
limitations; no theory is claimed.

---

## Round 2

Written at about 04:50 UTC on 2026-10-07 by an independent reviewing agent. It was
given the compiled paper, code, tables and docs, and told to verify every
quantitative claim against the stored results and not to modify files. Some
experiments were still running (E3, E5, E6, E7, PPO γ = 0.9, Q γ = 0.99).
Each criticism was checked against the repository before acting; the
"verified" column records that check.

### A. Networking reviewer

| # | Criticism | Sev. | Verified? | Response / action |
|---|---|---|---|---|
| N1 | MILP-track is an untuned heuristic: `eps`, `min_gain` never tuned; it ignores move cost and moves the largest-volume (most expensive) demand; it has +6.3 more utility than the bandit and wins `ood_double_failure` on 5/5 roots, so "matches" hides a trade-off. | major | yes | **Experiment:** validation-seed grid over min_gain × move selection (largest/smallest volume), selected config tested (log #32, #43, #44). **Outcome: the reviewer was right** — tuned MILP-track (30.9) beats the bandit on 5/5 roots; abstract, introduction, RQ1, discussion and conclusion now say so. **Text:** "no detectable difference in return", utility/cost trade-off stated, per-scenario bandit − MILP table added (App.), reroute-ratio sentence now says MILP's objective has no churn term. |
| N2 | Test regime is chronic overload (mean max-util > 1 for every controller); TE is loss triage, so the sequential value of headroom planning is absent. | major | yes (1.00–1.05 for MILP/bandit/PPO, 1.38 no-op) | New *Load regime* limitation with generated numbers; discussion conditions the Teal/DOTE remark on this. An operational-load scenario would need a new environment version (roadmap). |
| N3 | "Anticipation the controller cannot observe" is a design choice (no clock). | major | yes | Reworded to "not provided by our observation"; E7 (clock) is the test, reported in RQ3. |
| N4 | "Sequential RL is necessary" under delay overclaims: MILP-track beats PPO there; a lag-aware myopic target would recover; the delay lengthens the lock and cancelled requests keep their charge. | major | yes | RQ5 renamed "When does a myopic learner fail?"; claim narrowed to "a decision-time-reward learner fails"; side effects disclosed. **Experiment:** Q-learner γ = 0.5 (≈ 2-interval horizon) under delay (log #33). Title changed (`paper/TITLES.md`). |
| N5 | The observation contains the projected bottleneck of every candidate move — a hand-built one-step model that favours the bandit. | moderate | yes (features 536–604) | Disclosed as a limitation; not ablated (frozen environment). |
| N6 | Inference-latency argument is irrelevant at 300 s; gaps should be translated to operational units. | minor | yes | Latency now reported as non-discriminating; operational metrics (delivered %, SLA, reroutes/h) are in Table 2 and RQ1 text. |

### B. RL reviewer

| # | Criticism | Sev. | Verified? | Response / action |
|---|---|---|---|---|
| R1 | With 2–3 roots the two-stage bootstrap ≈ range of root means; calling it a 95 % CI and using it in the abstract is misleading. | critical | yes (e.g. delay roots +21.5/+41.6 → [20.6, 42.6]) | Statistics paragraph rewritten; abstract and RQ1 lead with the root t-interval; ablation tables (delay) report per-root values; sign counts emphasised. |
| R2 | "Floating-point differences alone" is false: SB3 samples PPO actions from the device generator; versions differ. | major | yes (manifests: torch 2.11.0+cu128, numpy 2.3.0, py 3.13.4) | Corrected in paper, intro, reproduction report, overview; CPU-root spread (SD) added as the like-for-like evidence. |
| R3 | "Horizon, not algorithm" is confounded (unnormalized ~200-step returns for PPO; (1−γ) scaling for Q). | major | partly | Abstract hedged ("points to"); pending PPO γ = 0.9, Q γ = 0.99 and E3 `rewnorm`/`gae08` decide it; RQ2/RQ3 will report whichever way they fall. |
| R4 | Q-learner is under-resourced (≈ 6.2k updates, ≈ 25 target syncs, truncation as terminal). | major | yes | Stated as a limitation; Q results described as budget-bound. |
| R5 | No-op continuation cannot see dwell or coupling effects; frozen Δ_H ≈ H·g − c, so 88 % agreement is close to guaranteed; claims that §6.4 "measures" the discarded terms are false. | major | yes (claims in §3 and §8) | Both claims rewritten; RQ4 now states exactly what the frozen rollout measures (whether cost amortization changes the ranking) and that later-move value is bounded only by the closed-loop ladder. Reactive continuation not run (cost). |
| R6 | Seqdiag (33 % agreement) and oracle ladder (+1.4 to +2.9) are in tension: open-loop non-myopic value is largely not realisable by a re-planning controller. | major | yes | RQ4 restructured around the closed-loop result; open-loop numbers presented as an overstatement of realisable value; abstract uses the ladder. |
| R7 | Delay: bandit failure is tautological; PPO "unaffected" hides ±10 root shifts and PPO r42 still flaps; Q (1 root) CI misleading. | major | yes | Per-root table; text says the bandit fails by construction and that PPO r42 still flaps; Q results per root. |
| R8 | Shaping argument (per-episode ≈ 0) says nothing about per-step argmax effects. | moderate | yes | Argument to rest on E6 (no-shaping bandit) when it lands. |
| R9 | GAE mechanism asserted, not measured; one PPO root has 0 reversals but low utility; costs explain 15.3 of 24.9. | moderate | yes | Mechanism now "we argue"; text states that 2 roots fail through utility; decomposition split generated (62 % costs). Critic explained variance reported where logged. |
| R10 | 3 of 5 primary roots are reproduction runs whose sign was known; frozen diagnostic added after an early look; H5(a) falsified and not reported. | moderate | yes | Disclosed; hypothesis-outcome table added (`docs/HYPOTHESES.md`, paper appendix), including H5(a) not supported. |
| R11 | Per-scenario CIs, win rate and d_z treat root × episode pairs as iid. | moderate | yes | Per-scenario intervals are now two-stage (roots, then episodes) with root sign counts; d_z labelled as pooled. |
| R12 | Only scripted OOD scenarios tested; train/test shift may differ between learners. | moderate | yes | Limitation reworded; `random_day` test not run (compute). |

### C. Paper reviewer

| # | Criticism | Sev. | Response / action |
|---|---|---|---|
| P1 | Abstract/conclusion overclaim ("the cause is the horizon"; "regime where sequential RL is necessary"). | major | Both rewritten (see R3, N4); title changed. |
| P2 | "Matches"/"statistically indistinguishable" without an equivalence margin. | moderate | Replaced by "no detectable difference at five roots" with t-interval; per-scenario differences shown. |
| P3 | Fig. 4 caption state count is wrong for H < 24. | minor | Caption gives the range of states per H. |
| P4 | Stale PDF (App. B grid; missing RQ3). | minor | Recompiled; RQ3 section added. |
| P5 | Table 4/5 caption inaccuracies. | minor | Captions corrected (roots per row; interval type). |
| P6 | Loose wording ("most informative", "plans over 17 hours", "stronger and cheaper"). | minor | Reworded. |

**Verified by the reviewer (no action needed):** main gap, intervals, root
counts, win rate, d_z; per-root returns; bandit − MILP and PPO − MILP;
reproduction numbers; horizon sweep; delay table values; seqdiag H = 24
values; oracle ladder; model-fidelity means; decomposition; gradient-step
counts.

---

## Round 3

Written at about 09:15 UTC by an independent reviewing agent on the paper
revised after round 2, with the E3/E3b, E6, E7, Q γ = 0.99, PPO γ = 0.9 and
tuned-MILP results in. Responses checked against the repository.

### Unresolved round-2 items

| # | Criticism | Sev. | Response / action |
|---|---|---|---|
| R8 | §5 still argued that shaping (≤ 0.02 per episode) cannot explain the bandit's advantage, though E6 shows removing it moves the bandit by ≈ 7. | major | Sentence replaced: the evaluated contribution is tiny, but that does not bound its effect on learning; points to E6. |
| R6 | "Bounds the value of lookahead" is wrong: Oracle-H uses a no-change continuation, so its gains are what one planner achieves, not an upper bound; Oracle-6 < Oracle-3 suggests continuation bias. | major | Abstract, RQ4, discussion, conclusion and `SEQUENTIALITY_AUDIT.md` reworded; H12 (greedy continuation) is the check. |
| R3 | `rewnorm`/`gae08` "probe" the mechanism but are never interpreted. | moderate | RQ3 now interprets them, with the one-root caveat. |
| R1 | Reproduction gap still labelled "95 % CI" with 3 roots. | moderate | Per-root values and the (uninformative) t-interval reported instead. |
| P2 | "PPO with γ = 0 matches the bandit" is equivalence wording. | minor | "Not detectably different at three roots" (abstract, conclusion, README, overview). |
| R5 | Intro/conclusion overstate what the frozen rollout shows. | minor | Qualified to "value visible to a no-change continuation". |

### Networking, RL and paper reviewers

| # | Criticism | Sev. | Response / action |
|---|---|---|---|
| N1 | Tuned-MILP grid edge not disclosed. | moderate | Stated in RQ1 (and log #43); the lead is likely conservative. |
| N2 | Tuned MILP missing from the main table. | major | Row added to Table 2 and README results block. |
| N3 | "Without such a model" ignores the bandit's one-step projection features. | moderate | Reworded. |
| N4 | Oracle-1's lead mixes an exact reward model with clairvoyant traffic. | moderate | Stated as not separated; claim softened. A persistence oracle was not run. |
| N5 | "Environment frozen" is not a reason for not ablating features (variants were run). | minor | Reason given as compute. |
| N6 | E7's negative result not used in the discussion. | moderate | Integrated. |
| R-a | "Tuned MILP beats every learner" is false for the shaping-free bandit on one root (31.2 vs 30.9); tuning was asymmetric. | major | Verified. Restated as "both learners of the main comparison"; per-root shaping-free bandit − tuned MILP reported; asymmetry stated. |
| R-b | Shaping confounds the horizon sweep (policy-invariant only for γ = 0.995; ≈ 7-point effect at γ = 0). | major | **Experiment E6b** (Q γ = 0.9, PPO γ = 0, PPO γ = 0.995 without shaping; H13, registered before running). RQ2 will state the confound and the E6b outcome. |
| R-c | No within-root noise floor for the 2–8-point effects of E6/E7/Q γ = 0.5. | moderate | **Experiment E8** (bandit, root 42, two learner seeds on identical traffic; H14). |
| R-d | "Monotonically" holds for means only. | minor | "Mean return falls … on two of three roots individually". |
| R-e | EV argument used non-oscillating roots. | minor | Now uses the PPO runs whose final checkpoint oscillates (generated). |
| R-f | Pre-write both readings of pending outcomes. | — | H11–H14 state support and falsification conditions before running. |
| P-a | Stale text (E3 "finished", "two further roots", "H9–H11"). | minor | Fixed. |
| P-b | H8 abridged too strongly. | moderate | "Closer to the myopic optimum than PPO; neither is close". |
| P-c | "Frozen" has three meanings. | minor | L = 0 environment renamed "base environment". |
| P-d | Abstract too long; §5 title. | minor | Abstract cut to ≈ 240 words; §5 retitled "Why PPO Falls Behind". |
| P-e | Interval labels. | minor | Final-checkpoint gap labelled as two-stage bootstrap; ladder n stated. |
| P-f | Strongest practical finding buried in intro bullet 2. | minor | Bullet reordered: tuned MILP first. |

**Verdict recorded by the reviewer:** weak accept for a networking venue as a
careful negative and measurement study; borderline for an ML venue until the
lookahead claims are framed as properties of a no-change-continuation planner
(done) and the shaping confound is resolved (E6b running).
