# Hypotheses and predictions (recorded before the new learner results)

Committed before any E0–E5 learner evaluation existed (see git history of this
file). Known at the time of writing: the closed study's reported numbers
(bandit 18.2 vs PPO 9.0 holdout return, 3/3 roots), the exact reproduction of
the baselines, and an early, partial sequentiality diagnostic on `full_day`
(myopic action = 24-step-best action in 35 % of states under clairvoyant
no-op continuation).

| ID | Hypothesis | Experiment | Prediction that would **support** it | Outcome that would **falsify** it |
|---|---|---|---|---|
| H0 | The historical bandit > PPO result reproduces on CPU. | E0 (historical protocol) | bandit − PPO > 0 on most roots; gap of similar order (≈ 9) | gap ≤ 0 on ≥ 2 of 3 roots |
| H1 | The gap survives an independent protocol and more roots. | E0+E1, validation→test | two-stage CI of bandit − PPO excludes 0 | CI includes 0 or sign flips |
| H2 | **Horizon is not what PPO's extra machinery buys.** Shortening PPO's horizon does not hurt it. | E2: PPO γ ∈ {0, 0.9, 0.995} | PPO(γ=0) ≥ PPO(γ=0.995) − small | PPO(γ=0) clearly worse than PPO(γ=0.995) |
| H3 | **Bootstrapping future value does not help a value learner at this budget.** | E2: Q γ ∈ {0, 0.5, 0.9, 0.99} | return non-increasing in γ, or flat | a γ > 0 Q-learner clearly beats the bandit on ≥ 2/3 roots |
| H4 | PPO's deficit is not a tuning artefact. | E3 | no configuration's validation return exceeds the bandit's on the same root | some configuration reaches or exceeds the bandit on validation *and* test |
| H5 | The decision problem has measurable non-myopic structure, but most of it comes from (a) one-off move costs vs persistent gains and (b) clairvoyant knowledge of future traffic. | seqdiag live vs frozen; oracle ladder H=1/3/6 | frozen agreement > live agreement; oracle-H gains over oracle-1 small relative to learner–oracle-1 gaps | oracle-6 ≫ oracle-1 and frozen ≈ live |
| H6 | When effects are delayed (L = 1), the myopic learner fails and sequential learners do better. | E4 | bandit(L=1) ≪ bandit(L=0) and < PPO(L=1), Q-γ0.9(L=1) | bandit(L=1) ≈ PPO(L=1), or both collapse equally |
| H7 | PPO does not catch up with 3× budget. | E5 | PPO at 1.2M still below bandit at 400k | PPO at 1.2M ≥ bandit |
| H8 | The bandit wins because it is a good myopic optimizer, not because PPO finds non-myopic value. | model fidelity | bandit top-1 agreement with exact myopic optimum and 1-step regret better than PPO's | PPO's actions have worse 1-step reward but better H-step value |

Interpretation rules fixed in advance:

* A difference is called "robust across roots" only if its two-stage bootstrap
  interval excludes 0 **and** it has the same sign on every root.
* With 3 roots, any claim is stated as "on all three roots", not as a
  population statement.
* If H6 holds but H2/H3 also hold, the conclusion is conditional: sequential
  RL is unnecessary *in the frozen environment because its effects are
  immediate*, not unnecessary for TE in general.

## Post-hoc additions (added after some results were known; treat as exploratory)

| ID | Added | Motivation | Hypothesis | Experiment | Falsified if |
|---|---|---|---|---|---|
| H9 | 2026-10-07 01:30 UTC, after the frozen-exogenous diagnostic on `full_day` | With traffic held fixed, the myopic action is the 24-step best in 91 % of states; with true future traffic only 36 %. The non-myopic value is anticipation of exogenous change, which the default observation (no clock) can barely predict. | Adding time of day helps a γ = 0.9 Q-learner more than it helps the bandit. | E7 (obs-v2.0-time-606; bandit and Q γ=0.9, roots 42, 314159) | the bandit gains as much or more from time features, or neither gains |
| H10 | 2026-10-07 01:15 UTC, after the return decomposition | PPO's deficit is dominated by reversal costs from flapping at hold-down expiry. | Removing PPO's horizon (γ = 0) removes most of the flapping. | E2 `ppog0` | PPO γ=0 flaps as much as PPO γ=0.995 |
| H11 | 2026-10-07 04:44 UTC, after E4 (delay reverses the ranking) and before running the diagnostic on the delayed variant | The sequentiality diagnostic detects the regime where the myopic learner fails: with L = 1 the agreement between the myopic-best and the 24-step-best action falls far below its L = 0 value, *also with traffic frozen* (L = 0 frozen: 88 %), because the immediate reward of a move contains only its cost. | Frozen H24 agreement at L = 1 is much lower than at L = 0 (no numeric threshold fixed; reported with per-scenario values). | seqdiag on `{"variant": "delayed", "delay_steps": 1}`, greedy reference, seeds 4001–4003, clairvoyant and frozen | frozen agreement at L = 1 stays near the L = 0 value: then the diagnostic does not predict when sequential RL is needed |
| H12 | 2026-10-07 05:10 UTC, after round-2 review R5/R6, before running | With a *reactive* continuation (greedy controller after the first move), the first-interval best move agrees with the 24-interval best move more often than with the no-change continuation, under the true future; i.e. much of the open-loop non-myopic value is obtainable by reacting later. | live H24 agreement with greedy continuation clearly above the no-change value (33 %) | seqdiag, `--continuation greedy`, greedy reference, seeds 4001–4003, clairvoyant and frozen | live agreement with greedy continuation ≤ the no-change value: the first move matters for long-run value even when the controller reacts afterwards |
| H13 | 2026-10-07 09:17 UTC, after round-3 review R-b, before running E6b | The horizon effect is not an artifact of shaping: without shaping, Q γ = 0.9 is below the shaping-free bandit and PPO γ = 0.995 is far below it, while PPO γ = 0 is close to it. | ordering bandit-noshaping ≈ PPO γ0-noshaping > Q γ0.9-noshaping ≫ PPO γ0.995-noshaping on both roots | E6b (roots 42, 314159) vs E6 | Q γ0.9 or PPO γ0.995 without shaping reaches the shaping-free bandit on both roots |
| H14 | 2026-10-07 09:17 UTC, after round-3 review R-c, before running E8 | Within-root learner noise of the bandit is small relative to the E6/E7 effects (≈ 2–8 points). | the two reseeded root-42 bandits are within a few points of 18.3 | E8 | spread across reseeds comparable to the E6 effect (≥ 7 points): then E6/E7 differences are not distinguishable from learner noise |

## Outcomes

The authoritative, generated outcome table is in the paper
(`paper/appendix/hypotheses.tex`; statuses of pending experiments come from
`paper/generated/status_*.tex`, written by `scripts/study/analyze.py`).
Outcomes are reported for every hypothesis, including those that failed:

* **H0**: direction reproduced (3/3 roots), magnitude not (30.1 vs 9.2).
* **H1**: supported (24.9; t-interval over 5 roots [3.4, 46.4]; 5/5 roots).
* **H2**: supported (PPO γ = 0: 23.6; γ = 0.995: −6.4).
* **H3**: supported (Q γ = 0.5 ≈ bandit, 1/3 roots above; γ = 0.9 −4.8, 0/3; γ = 0.99 −20.9, 0/3).
* **H4**: falsified on the selection root (tuned PPO 20.7 vs bandit 18.3 on root 42 test), supported on fresh roots (E3b: tuned PPO below the bandit on 4/4, mean −28.6).
* **H9**: not supported (clock adds +1.9 to the bandit, +0.4 to Q γ = 0.9; both within learner noise).
* **H14**: falsified — three learner seeds on identical root-42 traffic give the bandit 18.3 / 17.8 / 28.2.
* **H5**: prediction met, but component (a), one-off cost vs persistent gain,
  is **not** supported: with traffic frozen, cost amortization changes the
  best move in only 12 % of states. The non-myopic value is (b),
  anticipation of exogenous change.
* **H6**: the bandit fails under delay (22.0 → −30.3); PPO and Q γ = 0.9 are
  above it on 2/2 roots, but neither reaches MILP-track (11.3).
* **H8**: supported (bandit top-1 55 %, PPO 7 %; PPO's actions worth less than
  no-op over 6 intervals on 3/3 roots).
* **H10**: supported (reversals 52.8 → 1.0).
* **H13**: supported — without shaping, PPO γ = 0 stays close to the shaping-free bandit (−1.3 / −7.2), Q γ = 0.9 is below it (−13.6 / −15.4) and PPO γ = 0.995 far below (−46.1 / −16.3).
* **H7**: supported on root 42 (PPO at 1.2M 10.7 < bandit at 400k 18.3; still improving).
* **H11**: supported — under delay the frozen diagnostic's myopic choice captures 0 % of the 24-interval gain (97 % without delay).
* **H12**: see the generated table (pending at the time of writing).
