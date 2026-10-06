"""Seed sets and scenario list of the post-V2 study (single source of truth).

Seed namespaces are disjoint from each other and from training episode seeds
(``root + rank + 1024*episode`` with roots 42/314159/271828/... never lands in
these small ranges for the episodes used here; see ``tests/test_study.py``).

* ``HISTORICAL_CONTINUITY`` / ``HISTORICAL_HOLDOUT`` -- the closed V2 study's
  selection and final-holdout seeds. Used only to reproduce historical numbers.
* ``VALIDATION`` -- checkpoint selection and hyper-parameter selection for
  every new experiment. Never used for a reported test number.
* ``TEST`` -- reported results of new experiments. Never used for any choice.
* ``DIAGNOSTIC`` -- structural (clairvoyant) diagnostics that select nothing.
"""

from __future__ import annotations

EVAL_SCENARIOS: tuple[str, ...] = (
    "full_day", "evening_peak", "flash_crowd", "link_failure",
    "deceptive_local_optimum", "ood_double_failure", "overload_stress",
)

TRAINING_SCENARIO = "random_day"

HISTORICAL_CONTINUITY: tuple[int, ...] = (101, 102, 103, 104, 105)
HISTORICAL_HOLDOUT: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)
VALIDATION: tuple[int, ...] = (2001, 2002, 2003, 2004, 2005)
TEST: tuple[int, ...] = tuple(range(3001, 3021))
DIAGNOSTIC: tuple[int, ...] = (4001, 4002, 4003)

#: Training roots. The first three are the closed study's preregistered roots.
TRAINING_ROOTS: tuple[int, ...] = (42, 314159, 271828, 161803, 141421)

SEED_SETS = {
    "historical_continuity": HISTORICAL_CONTINUITY,
    "historical_holdout": HISTORICAL_HOLDOUT,
    "validation": VALIDATION,
    "test": TEST,
    "diagnostic": DIAGNOSTIC,
}
