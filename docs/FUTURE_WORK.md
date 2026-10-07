# Future-work roadmap

This roadmap follows from the evidence in this study. Every item names the
finding that motivates it, the result that would change a conclusion of the
paper, and a rough cost on the 4-core CPU machine used here (one 400k-transition
run ≈ 0.5–1 h). **Nothing below has been run** unless the status column says so.
Each item needs its own registry entry and preregistered hypothesis, under the
seed discipline of `mplssim/study/protocol.py`: new validation and test seeds;
the closed study's holdout 1001–1005 is not reused for selection.

The older `docs/V3_RESEARCH_BACKLOG.md` was written before this study. The
table at the end maps its items to what this study has since settled.

## A. Where would sequential RL be needed? (highest priority)

| # | Item | Motivating finding | Result that would change a conclusion | Cost |
|---|---|---|---|---|
| A1 | **Delay sweep** L ∈ {0, 1, 2, 3} for bandit, Q-learner (γ ∈ {0.5, 0.9}) and PPO (default and tuned), ≥ 3 roots | E4: at L = 1 the ranking reverses (bandit collapses, PPO unaffected); only L ∈ {0, 1} measured | the crossover is a single point; a sweep gives the *shape* (does Q γ = 0.5 suffice at L = 1? does PPO degrade at L = 3?) | ≈ 30 runs |
| A2 | **Partial delay**: a fraction α of the moved volume takes effect immediately, the rest at t + L (make-before-break with gradual shift) | E4 is the extreme case α = 0 | if the bandit stays competitive for α ≥ 0.5, then "delay reverses the ranking" needs restating as a threshold, not a switch | ≈ 15 runs |
| A3 | **Sequentiality diagnostic on the delayed variant** (clairvoyant and frozen) | the diagnostic predicted myopic adequacy at L = 0; a predictive tool should also predict the failure at L = 1 | if frozen agreement at L = 1 stays high, then the diagnostic does not predict when sequential RL is needed, and the methodological claim fails | ≈ 4 h CPU, no training |
| A4 | **Forecast-augmented observation**: add a noisy k-step forecast of demand volumes (noise σ swept) | seqdiag: non-myopic value is anticipation of exogenous change that the observation cannot see; E7 (clock in the observation) tests the weakest form of this | if PPO or Q (γ > 0) beats the bandit once forecasts are informative, then "persistent state is not enough, predictability is needed" is directly supported | ≈ 20 runs |

## B. Strengthening the existing comparison

| # | Item | Motivating finding | What it would settle | Cost |
|---|---|---|---|---|
| B1 | PPO tuning on more than one root (E3 selected on root 42 only) | per-root shifts of up to 41 points from numerical noise alone (reproduction) | whether any PPO configuration with γ = 0.995 closes the gap reliably | ≈ 24 runs for 3 roots × 8 configs |
| B2 | Bandit tuning with the same budget as PPO tuning | the bandit was deliberately left untuned | the size of the gap under symmetric tuning (the ranking cannot reverse unless tuning *hurts* the bandit) | ≈ 8 runs |
| B3 | Recurrent PPO (sb3-contrib RecurrentPPO, masked) | V3-2; the observation omits the clock, so history could carry phase information | whether memory substitutes for a clock (compare against E7) | ≈ 6 runs |
| B4 | Non-clairvoyant MPC: Oracle-H with a forecast in place of the true future | Oracle-H uses the true future, so it bounds rather than competes | how much of Oracle-1's lead over the bandit (≈ 22) a realistic forecaster recovers; this separates "lookahead" from "knowing the future" | ≈ 1 day CPU, no training |
| B5 | Multi-move MILP (re-optimize all demands each interval, churn-penalized) | MILP-track is limited to one move per interval like the learners | the cost of the single-move restriction itself (networking reviewer N1) | ≈ 2 h CPU |

## C. External validity

| # | Item | Notes |
|---|---|---|
| C1 | Second and third topology (e.g. Abilene, GÉANT from TopologyZoo or SNDlib) with their published traffic matrices | requires a new frozen environment version; the action space changes size, so the learners need retraining. The question is whether the frozen-exogenous agreement stays high. |
| C2 | Real traffic matrices (SNDlib, Abilene TM) in place of the synthetic diurnal process | removes the hand-written diurnal model (limitation 1) |
| C3 | Elastic traffic (TCP-like rate response to loss) | creates genuine endogenous dynamics: a move would change *future demand*, which is exactly the coupling this formulation lacks |
| C4 | Packet-level validation of selected policies (ns-3 or similar) | checks that flow-level fixed-point conclusions survive queueing dynamics |
| C5 | Larger action spaces (k > 4 candidate paths; multiple moves per interval) | tests whether masking and the bandit's per-action head scale |

## D. Methodology

| # | Item |
|---|---|
| D1 | Package the sequentiality diagnostic (clairvoyant vs frozen-exogenous rollout agreement) as a stand-alone tool for any Gymnasium environment that supports state snapshot/restore, and run it on published RL-for-networking environments. |
| D2 | Report a theory-backed horizon quantity (Laidlaw et al.'s effective horizon) alongside the empirical agreement rates. |

## Mapping of the earlier V3 backlog

| V3 item | Status after this study |
|---|---|
| V3-1 observation leaks the future | partly answered: the observation has *no* clock, and seqdiag shows the non-myopic value is unobservable anticipation. E7 (adding time of day) tests the reverse direction; A4 extends it. |
| V3-2 recurrent policy | still open (B3) |
| V3-3 explicit planner | clairvoyant Oracle-H run as an upper reference; a non-clairvoyant planner is still open (B4) |
| V3-4 A2C and other on-policy methods | lower priority: E2 shows that PPO with γ = 0 matches the bandit, so the algorithm family is not the explanation |
| V3-5 reward-design sensitivity | partly: E6 removes shaping; reward-weight sensitivity still open |
| V3-6 topology generalization | still open (C1) |
| V3-7 more training roots | done for the main comparison (5 roots per learner); ablations use 2–3 |
