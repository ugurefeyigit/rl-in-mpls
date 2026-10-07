# Research overview

*Two-page summary. Numbers come from `results/tables/` (generated); the paper
is `paper/main.pdf`.*

## Question

*Title: "Persistent State Is Not Enough: A Controlled Study of Planning
Horizons in Incremental MPLS Traffic Engineering" (`paper/TITLES.md`).*

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
| F1 | The closed predecessor study's bandit result reproduces; its PPO result does not. Training traffic was byte-identical; a change of device and library versions (which alters PPO's action-sampling stream and arithmetic) moved PPO's per-root holdout return by up to 41 points (bandit ≤ 4). The "PPO wins the deceptive scenario" claim does not reproduce. | `docs/REPRODUCTION_REPORT.md` |
| F2 | The bandit beats PPO on all 5 training roots (paired +24.9; t-interval over roots [3.4, 46.4]). Its return is not detectably different from a non-learning per-interval min-max-utilization MILP controller (+1.2, t-interval [−4.0, 6.3]; not an equivalence claim). MILP-track has higher network utility and higher move costs, and it wins the double-failure scenario. | `results/tables/main_*.csv` |
| F3 | Most of PPO's deficit is reconfiguration cost (15.4 of the 24.9-point gap; 9.5 is lower network utility): on 3 of 5 roots most of its moves are reversals at hold-down expiry, and its utility varies widely across roots (7.8–32.2 vs 20.9–31.6 for the bandit). Over 6 intervals PPO's chosen actions are worth less than doing nothing on 3/3 analysed roots. | `decomposition_by_root.csv`, `model_fidelity.csv`, case study |
| F4 | Varying the discount within each family points to the horizon, not the algorithm: PPO with γ = 0 matches the bandit (−0.6; 2/3 roots above) and stops oscillating, and a bootstrapped Q-learner (γ = 0.9) is worse than the bandit (−4.8; 0/3 roots above). A PPO-specific interaction with long horizons is not yet excluded (E3 reward normalization / GAE λ pending). | `horizon_sweep.csv` |
| F5 | Closed loop: a clairvoyant controller that re-plans every interval gains only +2.9 (H = 3) and +1.4 (H = 6) over H = 1, while knowing the next interval exactly is worth ≈ 22 over the bandit. Open loop, a move's 24-step value under a no-change continuation ranks moves like the first interval does in 33 % of states under the true future, but in 88 % with traffic held fixed: the open-loop non-myopic value is anticipation of exogenous change, which a re-planning controller can largely obtain by reacting. | `docs/SEQUENTIALITY_AUDIT.md`, `oracle_ladder.csv` |
| F6 | Delaying a move's effect by one interval makes the bandit fail by construction (22.0 → −30.3); PPO is ahead of it on 2/2 roots (mean +31.5), but MILP-track (11.3) stays ahead of every learner. The experiment shows when a decision-time-reward learner fails, not that long-horizon RL is needed. | `delay_results.csv` |
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
reward already ranks actions almost as well as a longer criterion would, and
a controller that re-plans every interval loses little by not looking ahead.
A myopic learner, or a per-interval optimizer, is then the right tool, and a
long horizon mainly adds estimation noise. Learners of the decision-time
reward fail when effects are delayed relative to decisions; whether a short
or a long horizon is then needed is open. Having persistent state is not, by
itself, a reason to use a sequential learner. All of this holds in a heavily
loaded, flow-level, single-topology simulator (`docs/LIMITATIONS.md`).
