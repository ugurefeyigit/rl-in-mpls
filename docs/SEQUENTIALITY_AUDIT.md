# Sequentiality audit

*How much does the current action affect future decision quality, and is
that effect predictable from what the controller observes?*

Numbers in this document are generated (`results/tables/seqdiag_summary.csv`,
`experiments/raw/seqdiag_greedy*/`); the paper uses the same tables through
`paper/generated/numbers.tex`.

## 1. Structural facts (from the code; `docs/MATHEMATICAL_FORMULATION.md`)

| Question | Answer in Environment V2 |
|---|---|
| Does `a_t` change `s_{t+1}`? | Yes, through the routing state only: current path, dwell, previous-TE path, path age. Traffic and failures are exogenous. |
| Do routing decisions persist? | Yes, until the controller or FRR changes them. |
| Do they alter future congestion? | Yes: a moved demand loads its new path in every later interval. |
| Are demands processed independently? | No: they share links, and the protected-class mask makes one demand's placement change which moves are legal for others. |
| Is there delayed reward? | No: a move carries traffic in the interval in which it is made, so its first-interval effect is in `r_t`. (Delayed effects are introduced only by the E4 variant.) |
| Path dependence? | Through the 3-interval hold-down and the reversal cost (6-interval window). |
| Resets? | Episodes are one scenario (48–288 intervals); nothing resets within an episode. |
| Is γ = 0.995 meaningful? | Its horizon (≈ 200 intervals ≈ 17 h) exceeds six of the seven evaluation episodes (48–84 intervals). |

So the problem is formally an MDP, and every one of the bandit's omissions
(persistence, dwell opportunity cost, reversal cost, coupled legality) is real.
The question is quantitative.

## 2. Method: action-value rollouts with a fixed continuation

On states visited by a reference controller (greedy; diagnostic seeds
4001–4003; every 4th decision; 495 states), for every legal action `a` we roll
the cloned simulator forward for up to 24 intervals: first `a`, then no-op.
Prefix sums give `G_H(s, a)` for H ∈ {1, 2, 3, 6, 12, 24} from a single rollout;
`Δ_H(s, a) = G_H(s, a) − G_H(s, no-op)`. `Δ_1` is exactly what the bandit
estimates. Per state we record whether the myopic choice `argmax Δ_1` equals
the H-step choice `argmax Δ_H` (agreement), the H-step value it forgoes
(regret), the share of the best H-step gain it captures, and whether the
H-step best action has negative immediate advantage ("sacrifice").

This is the effective-horizon idea of Laidlaw et al. (2023) with a no-op base
policy: if the first-step choice is already right, little lookahead is needed.

Two variants:

* **clairvoyant (live):** the clone carries the true future traffic, bursts and
  failures of the episode;
* **frozen exogenous:** offered traffic and link state are held at their
  current values during the rollout. This isolates the value of a move that is
  predictable from the current observation (persistence vs one-off cost) from
  the value that comes from anticipating exogenous change.

## 3. Results

<!-- SEQ:BEGIN -->
| Rollout | H | States | Agreement | Gain captured | Best move is a sacrifice | Best is no-op |
|---|---:|---:|---:|---:|---:|---:|
| clairvoyant | 1 | 495 | 100 % | 100 % | 0 % | 51 % |
| clairvoyant | 2 | 495 | 81 % | 89 % | 13 % | 42 % |
| clairvoyant | 3 | 495 | 71 % | 78 % | 21 % | 37 % |
| clairvoyant | 6 | 474 | 54 % | 64 % | 35 % | 27 % |
| clairvoyant | 12 | 453 | 43 % | 59 % | 46 % | 19 % |
| clairvoyant | 24 | 390 | 33 % | 52 % | 59 % | 13 % |
| frozen exogenous | 1 | 495 | 100 % | 100 % | 0 % | 59 % |
| frozen exogenous | 2 | 495 | 94 % | 99 % | 5 % | 54 % |
| frozen exogenous | 3 | 495 | 93 % | 98 % | 6 % | 52 % |
| frozen exogenous | 6 | 474 | 89 % | 97 % | 8 % | 50 % |
| frozen exogenous | 12 | 453 | 88 % | 97 % | 9 % | 50 % |
| frozen exogenous | 24 | 390 | 88 % | 97 % | 8 % | 53 % |

Per scenario at H = 24 (agreement / gain captured):

| Scenario | Clairvoyant | Frozen |
|---|---:|---:|
| full_day | 36 % / 49 % | 91 % / 96 % |
| evening_peak | 29 % / 42 % | 85 % / 97 % |
| flash_crowd | 23 % / 65 % | 83 % / 99 % |
| link_failure | 30 % / 57 % | 93 % / 99 % |
| deceptive_local_optimum | 40 % / 52 % | 87 % / 92 % |
| ood_double_failure | 23 % / 36 % | 93 % / 97 % |
| overload_stress | 24 % / 67 % | 67 % / 96 % |
<!-- SEQ:END -->

## 4. Interpretation

1. **Per decision, the problem is not myopic under the true future.** With
   clairvoyant rollouts the myopic choice agrees with the 24-interval best
   choice in only about a third of states and the long-horizon best move
   usually has a *negative* immediate advantage.
2. **Almost all of that non-myopic value is anticipation of exogenous
   change.** Holding traffic and link state fixed, the myopic choice is the
   24-interval best in the large majority of states and captures nearly all of
   the available gain; sacrifice moves become rare. Moves whose one-off cost
   exceeds one interval's gain but which pay off by persistence alone exist but
   are uncommon, because move costs are small relative to congestion
   penalties.
3. **The observation can barely support that anticipation.** It contains the
   current traffic but not the clock, the AR(1) state or scheduled events. A
   learner can exploit the non-myopic value only to the extent it can predict
   traffic from current volumes. E7 (time of day added) tests this directly.
4. Hence the bandit's advantage is consistent with the structure: the part of
   the problem that is predictable from the observation is close to myopic,
   and the part that is not myopic is mostly not predictable. This is a
   statement about this environment, its observation and its traffic model,
   not about TE in general.

## 5. Caveats

* The continuation is no-op. A better continuation can only increase the
  value of any first action; the diagnostic measures structure, not an
  optimal policy.
* States come from one reference controller; other controllers visit other
  states.
* Three diagnostic seeds; per-scenario estimates rest on 45–216 states.
