# Title candidates

The title must state the finding without overclaiming. It was revised after
the round-2 hostile review (`docs/HOSTILE_REVIEWS.md`, P1/N4): the earlier
title promised to say *when sequential RL is needed*, but the delay
experiment shows only that a learner of the decision-time reward fails, and
the optimization baseline stays ahead of every learner there.

| # | Candidate | Assessment |
|---|---|---|
| 1 | Persistent State Is Not Enough: When Incremental MPLS Traffic Engineering Needs Sequential Reinforcement Learning | *Previous title.* Overclaims: the study does not establish that sequential RL is needed in any regime. |
| 2 | Persistent State Is Not Enough: A Controlled Study of Planning Horizons in Incremental MPLS Traffic Engineering | **Chosen.** States the main negative finding and what was varied (the horizon) without implying a positive result for RL. |
| 3 | Does Incremental MPLS Traffic Engineering Need Sequential Reinforcement Learning? | Accurate, but a question title hides the answer. |
| 4 | A Masked Contextual Bandit Beats PPO for Incremental MPLS Traffic Engineering | Headline result only; invites a "PPO was badly tuned" reading and omits the mechanism and the measurement. |
| 5 | Measuring the Value of Lookahead in Incremental MPLS Traffic Engineering | Describes the method contribution well but undersells the learner comparison. |
