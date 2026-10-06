# Limitations

Ordered roughly by how much each could change the conclusions.

## Validity of the simulator (networking realism)

1. **Flow-level abstraction.** No packets, queues or TCP. Delay and loss are
   analytic functions of link utilization (M/M/1-shaped delay capped at 60 ms,
   a quadratic soft-loss onset at 90 % and proportional drop above 100 %).
   Learned routing policies trained in fluid models are known not to transfer
   to packet-level dynamics (Boltres et al., 2024). All conclusions are about
   this model.
2. **Exogenous, inelastic traffic.** Offered load never reacts to congestion
   (no TCP back-off, no admission control). This is what makes paired
   comparisons exact, but it removes one natural source of temporal coupling:
   in a real network, a routing decision that causes loss changes future
   offered load.
3. **Instantaneous control plane.** A TE change carries traffic in the same
   5-minute interval in which it is decided; RSVP-TE signalling,
   make-before-break, IGP convergence and controller-to-device latency are not
   modelled. This is the most consequential assumption for the
   sequentiality question; the delayed-activation variant (`E4`) is our
   controlled probe of it, not a realistic control-plane model.
4. **Fixed candidate paths.** Four candidates per demand, computed once by
   Yen's algorithm with role/hop/delay filters. No online CSPF path
   computation, no bandwidth reservation state, no splitting of a demand over
   several LSPs (unlike MATE/TeXCP or ECMP-style split ratios). Path count and
   ordering are known to affect heuristic baselines (Doherty et al., 2025); we
   did not vary them.
5. **One TE change per interval.** Network-wide rate limit chosen by the
   environment design, not derived from operational data. Many controllers
   can re-optimize many LSPs at once; the single-move rule is what makes the
   problem incremental.
6. **Small, hand-designed topology and traffic.** 18 routers, 17 demands, six
   classes, hand-written diurnal profiles and scripted scenarios
   ("deceptive local optimum", "flash crowd", ...). No operator traffic
   matrices or real failure traces. Scenario names describe the designer's
   intent; whether a scenario actually has the named property must be (and
   for sequentiality, is) measured.
7. **Perfect telemetry.** Utilization and per-demand health are exact and
   current. No measurement noise or delay.
8. **FRR is free and instantaneous.** Failures cause no transient loss; the
   cheapest live candidate is always used.

## Validity of the learning comparison (RL methodology)

9. **Five training roots** for the primary comparison and three for most
   ablations. Root-level confidence intervals are wide. Episode-level
   intervals are narrow but reflect evaluation uncertainty for a *fixed*
   trained policy, not training variability; we report both.
10. **Budget.** 400k transitions (≈1,390 training days) per run; one run per
    learner at 1.2M. Conclusions are "at this budget". Lower discount factors
    are known to help with limited data (Jiang et al., 2015; Amit et al., 2020),
    so a myopic win does not prove that the problem lacks sequential structure.
11. **Tuning asymmetry, deliberately against the bandit.** PPO received a
    bounded one-factor-at-a-time sensitivity study (8 configurations on one
    root plus γ variations); the bandit was used with its preregistered
    configuration only. The PPO search is not exhaustive (no joint search, no
    network-architecture search beyond width, no observation normalization,
    no learning-rate schedules).
12. **Function approximation.** Both learners are MLPs on a flat 604-feature
    vector; neither exploits graph structure. A GNN policy might change the
    picture, especially for generalization.
13. **Training/evaluation shift.** Learners train only on `random_day`; all
    evaluation scenarios are scripted and partly out of distribution
    (`overload_stress` at 1.6× load, `ood_double_failure`). This tests
    robustness, not in-distribution optimality.
14. **Reward specification.** All conclusions are about the V2 operational
    reward (a hand-weighted sum of 12 components). Operational metrics are
    reported alongside, but a different operator objective could reorder
    methods. The potential-based shaping term is policy-invariant only for
    γ = 0.995; for the γ = 0 bandit it is a small (≈0.01 per episode) reward
    modification.
15. **Clairvoyant oracles are not optimal policies.** H-step oracles see the
    true future traffic but only consider "one move, then no change". They
    bound what H-step-greedy reasoning can gain; they do not bound the optimal
    policy. The MILP baseline optimizes gross maximum utilization only (no
    loss, delay or protected-class terms) and is tracked one move at a time.
16. **The sequentiality diagnostic uses a fixed reference trajectory**
    (greedy controller) and no-op continuations on three diagnostic seeds; it
    characterizes the states that controller visits.

## Reproducibility

17. The closed V2 study's checkpoints and per-episode holdout data were not
    archived in the repository. Its learner results are reproduced
    statistically on different hardware (CPU instead of CUDA), not exactly; its
    baselines reproduce exactly.
18. Five normative design documents referenced by the V2 code were never
    committed; the formulation was reconstructed from the code.

## Scope of claims

19. No claim of topology generalization (one topology).
20. No claim of deployability: nothing here was tested against real routers,
    controllers or traffic.
21. No algorithmic novelty is claimed: the bandit, the Q-learner and PPO are
    standard methods; the contribution is the controlled comparison and the
    measurement of temporal coupling.
