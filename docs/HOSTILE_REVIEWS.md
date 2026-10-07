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

*(written after the final results; see bottom of this file once complete)*
