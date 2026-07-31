"""Frozen identity of the closed governed V2 study.

These are assertions about evidence that already exists and can never change.
Nothing here is configuration. If a value disagrees with an artifact, the artifact
is not the study this repository closed, and the loader fails closed.

Sources of truth for each constant:
  results/v2_final_holdout/manifest.json          seeds, scenarios, episode counts
  results/v2_final_holdout/checkpoint_provenance.csv   source SHAs, roots
  results/v2_three_root_continuity/REPORT.md      environment, pin SHA
  NEXT_STAGE_HANDOFF.md                           closeout identity
"""

from __future__ import annotations

# --- source identities -------------------------------------------------------
#: Evaluation-only tooling repair; the one-shot holdout ran from here.
EVALUATION_SOURCE_SHA = "f7ed0f407c50c5472ecff89f977bc656439a8c49"
#: Seed-42 scientific training source.
SEED42_SOURCE_SHA = "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
#: Roots 314159 and 271828 training/evaluation source.
CONTINUATION_SOURCE_SHA = "6a8a4068b98bf9a71dead6e547595b4bbd755689"
#: Signed-off V2 environment pin.
SIGNED_OFF_ENV_SHA = "dca533b5c6fa9953307d01470c23cac512eb2961"
#: Approved ancestor of the governed work.
APPROVED_ANCESTOR_SHA = "859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac"
#: The commit that closed the study.
CLOSEOUT_SHA = "d7d2b3f8623ec26ef802dcc07b768978a81c2e19"

TRAINING_SOURCE_SHAS: tuple[str, ...] = (SEED42_SOURCE_SHA, CONTINUATION_SOURCE_SHA)

# --- experimental design -----------------------------------------------------
TRAINING_ROOTS: tuple[int, ...] = (42, 314159, 271828)
#: The untouched final-holdout seeds. Used exactly once, after the study closed to
#: selection.
HOLDOUT_SEEDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)
#: Development/continuity seeds. Checkpoint selection used these and only these.
CONTINUITY_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)

SCENARIOS: tuple[str, ...] = (
    "full_day",
    "evening_peak",
    "flash_crowd",
    "link_failure",
    "deceptive_local_optimum",
    "ood_double_failure",
    "overload_stress",
)

LEARNER_ALGORITHMS: tuple[str, ...] = ("maskable_ppo", "masked_bandit")
BASELINE_ALGORITHMS: tuple[str, ...] = ("static", "greedy", "cspf")
ALL_ALGORITHMS: tuple[str, ...] = LEARNER_ALGORITHMS + BASELINE_ALGORITHMS

#: Six learner checkpoints plus three baselines.
POLICY_COUNT = 9
#: Seven scenarios by five holdout seeds.
EPISODES_PER_POLICY = 35
TOTAL_HOLDOUT_EPISODES = POLICY_COUNT * EPISODES_PER_POLICY  # 315

#: Fixed horizon per scenario, in 5-minute control intervals. These are frozen
#: scenario properties, not a single global episode length: one seed covers
#: 288+84+60+60+60+60+48 = 660 steps, so each policy recorded 3,300.
SCENARIO_STEPS: dict[str, int] = {
    "full_day": 288,
    "evening_peak": 84,
    "flash_crowd": 60,
    "link_failure": 60,
    "deceptive_local_optimum": 60,
    "ood_double_failure": 60,
    "overload_stress": 48,
}
STEPS_PER_SEED = sum(SCENARIO_STEPS.values())          # 660
STEPS_PER_POLICY = STEPS_PER_SEED * len(HOLDOUT_SEEDS)  # 3300

# --- environment -------------------------------------------------------------
ENVIRONMENT = "MplsTeEnvV2"
OBSERVATION_DIM = 604
ACTION_COUNT = 69
NOOP_ACTION = 0

#: The exact 12 component names the environment emits. Their sum equals the
#: operational return on every step of every episode.
REWARD_COMPONENTS: tuple[str, ...] = (
    "delivery",
    "protected_disconnect",
    "unprotected_disconnect",
    "sla_severity",
    "max_util",
    "overload",
    "potential",
    "move_fixed",
    "move_volume",
    "move_divergence",
    "reversal",
    "invalid",
)

# --- stage labels ------------------------------------------------------------
#: Evidence stages must never be blurred together in any presentation.
STAGE_FINAL_HOLDOUT = "final_holdout"
STAGE_DEVELOPMENT = "development"
