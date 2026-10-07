# Research overview

*Two-page summary. Numbers come from `results/tables/` (generated); the paper
is `paper/main.pdf`.*

## Question

In an incremental MPLS traffic-engineering problem with persistent routing
state — one LSP move per 5-minute interval, move costs, a 3-interval hold-down,
protected-class capacity rules, all enforced by an action mask — does a
sequential RL agent (MaskablePPO, γ = 0.995) outperform a learner that only
predicts each action's immediate reward (a masked neural contextual bandit,
γ = 0)? Why, and under which conditions would the answer change?

## The decision problem (`docs/MATHEMATICAL_FORMULATION.md`)

State = routing state (current/previous path, dwell, path age) × exogenous
process (time, traffic noise, failures). Actions = no-op or move demand d to
candidate p (69, masked). Traffic is exogenous; a move carries traffic in the
interval in which it is made. Formally an MDP: routes persist, moves lock a
demand, reversals are penalized, and the protected-class mask couples demands.
The agent's observation omits the clock and future events (POMDP w.r.t. the
exogenous process).

## Findings

| # | Finding | Evidence |
|---|---|---|
| F1 | The closed predecessor study's bandit result reproduces; its PPO result does not. Training traffic was byte-identical; floating-point differences alone moved PPO's per-root holdout return by up to 41 points (bandit ≤ 4). The "PPO wins the deceptive scenario" claim does not reproduce. | `docs/REPRODUCTION_REPORT.md` |
| F2 | The bandit beats PPO on all 5 training roots (paired +24.9, 95 % CI [12.2, 39.1]) and is statistically indistinguishable from a non-learning per-interval min-max-utilization MILP controller. | `results/tables/main_*.csv` |
| F3 | Most of PPO's deficit is reconfiguration cost (15.3 of the 24.9-point gap; 9.6 is lower network utility): on 3 of 5 roots most of its moves are reversals at hold-down expiry, and its utility varies widely across roots (7.8–32.2 vs 20.9–31.6 for the bandit). Over 6 intervals PPO's chosen actions are worth less than doing nothing on 3/3 analysed roots. | `decomposition_by_root.csv`, `model_fidelity.csv`, case study |
| F4 | It is the horizon, not the algorithm: PPO with γ = 0 matches the bandit (−0.6 [−5.2, 2.7]) and stops oscillating; a bootstrapped Q-learner (γ = 0.9) is worse than the bandit (−4.8 [−7.0, −2.1]). | `horizon_sweep.csv` |
| F5 | Per decision the problem is not myopic under the true future (myopic = 24-step best in 33 % of states), but with traffic held fixed it is (88 %). Almost all non-myopic value is anticipation of exogenous change. Clairvoyant lookahead beyond one interval adds +1.4 to +2.9 return, versus ≈ 22 for knowing the next interval exactly. | `docs/SEQUENTIALITY_AUDIT.md`, `oracle_ladder.csv` |
| F6 | Delaying a move's effect by one interval reverses the ranking: PPO − bandit = +31.5 [20.6, 42.6]; the bandit collapses (22.0 → −30.3), PPO is unaffected. | `delay_results.csv` |
| F7 | The action mask is consistent and safe: 0 violations of 14 checks over 1,068,120 state–action pairs. | `docs/MASK_VALIDATION.md` |

Pending at the time of writing (see the paper for final status): PPO tuning
(E3), PPO γ = 0.9, Q γ = 0.99, time-of-day observation (E7), shaping-free
bandit (E6), 1.2M-transition budget (E5).

## What is (and is not) contributed

* **Engineering:** a reproducible study layer on a frozen, governed simulator
  (registry, bit-exact trainer equivalence, paired evaluation, exact
  baseline reproduction, mask audit).
* **Experimental:** a controlled attribution of the bandit–PPO gap to the
  discount horizon, and a regime (delayed effects) where the ranking reverses.
* **Methodological:** a cheap simulator-based sequentiality diagnostic
  (clairvoyant vs frozen-exogenous rollouts) that predicts whether sequential
  RL has anything to exploit, applicable before training anything.
* **Not contributed:** a new algorithm, a theorem, topology generalization,
  or evidence about real networks.

## Interpretation in one paragraph

When an action's effect is felt in the interval in which it is taken and the
exogenous future is not predictable from the observation, the immediate
reward already ranks actions almost as well as a long-horizon criterion could
— so a myopic learner (or a per-interval optimizer) is the right tool, and a
long horizon only adds noise that hides small, certain costs. Sequential RL
becomes necessary when effects are delayed relative to decisions. Having
persistent state is not, by itself, a reason to use it.
