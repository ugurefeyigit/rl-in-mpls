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
    (greedy controller) on three diagnostic seeds, so it characterizes the
    states that controller visits. With the default no-change continuation it
    cannot see value from later moves (dwell opportunity cost, protected-class
    coupling); a reactive (greedy) continuation is run as a check (H12), and
    the closed-loop oracle ladder bounds that value under perfect foresight.
    Open-loop values overstate what a re-planning controller can realise.

17a. **Heavily loaded regime** (round-2 review N2). In the test scenarios the
    mean per-interval maximum link utilization is about 1.0 even for the best
    controllers (MILP-track 1.00, bandit 1.03, no TE 1.38), and about 4.5 % of
    offered traffic is lost. TE here mostly decides where traffic is lost.
    Headroom planning in a provisioned backbone is a plausible source of
    sequential value that is not tested.
17b. **The observation contains a one-step model** (round-2 review N5): for
    every candidate move, the projected gross bottleneck utilization it would
    create (features 536–604). This makes the immediate consequence of a move
    nearly observable and favours a learner of the immediate reward. Not
    ablated (frozen environment).
17c. **Q-learner budget** (round-2 review R4): one gradient step per 64
    transitions (≈ 6,200 in total), target update every 250 steps, time-limit
    ends treated as terminal. γ > 0 results are statements about this budget.
17d. **Delayed-activation side effects**: a pending request blocks further
    moves of its demand (effective lock L + 3 intervals), and a request that is
    illegal at activation is cancelled but keeps its charge.
17e. **Small-root ablations**: with 2–3 roots the two-stage bootstrap stays
    close to the range of root means; those comparisons are reported per root
    with sign counts.

## Reproducibility

17. The closed V2 study's checkpoints and per-episode holdout data were not
    archived in the repository. Its learner results are reproduced
    statistically on a different machine and software stack (CPU instead of
    CUDA; different torch/numpy/Python versions; PPO's action-sampling stream
    therefore differs), not exactly; its baselines reproduce exactly.
18. Five normative design documents referenced by the V2 code were never
    committed; the formulation was reconstructed from the code.

## Scope of claims

19. No claim of topology generalization (one topology).
20. No claim of deployability: nothing here was tested against real routers,
    controllers or traffic.
21. No algorithmic novelty is claimed: the bandit, the Q-learner and PPO are
    standard methods; the contribution is the controlled comparison and the
    measurement of temporal coupling.
