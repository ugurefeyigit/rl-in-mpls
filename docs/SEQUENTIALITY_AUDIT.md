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

*Revised after the round-2 hostile review (R5, R6). The first version read
the clairvoyant open-loop numbers as "non-myopic value that a better learner
could exploit"; the closed-loop oracle ladder shows that most of it cannot be
realised by a controller that re-plans every interval.*

1. **Closed loop, a simple clairvoyant planner gains little from lookahead.**
   Oracle-H re-plans every interval with the true future, scoring each move
   with a no-change continuation. Oracle-3 and Oracle-6 gain only +2.9 and
   +1.4 return over Oracle-1 (5 test seeds), while Oracle-1 is ≈ 22 above the
   bandit (`results/tables/oracle_ladder.csv`). Oracle-1's lead combines an
   exact reward model with clairvoyant next-interval traffic (not separated).
   These gains are a property of this planner, not an upper bound on the
   value of lookahead (round-3 review R6); Oracle-6 < Oracle-3 is consistent
   with continuation bias.
2. **Open loop, per decision, the problem looks non-myopic under the true
   future.** With clairvoyant no-change rollouts the myopic choice agrees with
   the 24-interval best choice in about a third of states, and the
   long-horizon best move often has a negative immediate advantage. These
   open-loop values overstate realisable value: a move that pays off only
   after traffic changes can also be made when the change arrives.
3. **That open-loop non-myopic value is anticipation of exogenous change.**
   With traffic and link state held fixed, the myopic choice is the
   24-interval best in the large majority of states. With constant exogenous
   inputs and no further moves, a move's H-interval value is close to H times
   its per-interval gain minus its one-off cost, so this number measures one
   thing: how often amortizing a move's cost over a longer stay changes which
   move is best. It rarely does. It does **not** measure dwell opportunity
   cost or protected-class coupling, because the continuation never moves
   again. Those are bounded only by the closed-loop ladder, and checked by
   a reactive (greedy) continuation (H12).
4. **The observation does not provide what anticipation needs.** It contains
   the current traffic but not the clock, the AR(1) state or scheduled events.
   That is a design choice of the environment, not a property of TE. E7 (time
   of day added) tests the clock.
5. This is a statement about this environment, its observation, its traffic
   model and its heavily loaded regime, not about TE in general.

## 5. Caveats

* The default continuation is no-op. It cannot see value that arises from
  later moves; a greedy continuation is run as a check (H12). The diagnostic
  measures structure, not an optimal policy.
* States come from one reference controller; other controllers visit other
  states.
* Three diagnostic seeds; per-scenario estimates rest on 45–216 states.
