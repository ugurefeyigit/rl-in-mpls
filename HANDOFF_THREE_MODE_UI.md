# Handoff — three-mode UI follow-up

## Current state

- Design authority: `feat/post-study-productization` at
  `3bb791e9a2ec6d7e1a402ecaa70a61f37d11f52a`, based on `c49da5f`.
- Implementation branch: `feat/three-mode-ui-implementation`.
- Worktree: `.worktrees/three_mode_ui`.
- The application is functional at `/`, `/advanced`, `/present`, and `/study`.
  All routes use one shell with exactly Presentation, Network Information, and
  RL Information as primary modes. Guided Story is a Presentation workflow.
- Router positions use a stable, non-geographic engineering schematic derived
  from the earlier readable layout. The simulator topology and router IDs are
  unchanged.

## Verified behavior

- Live, recorded, development-evidence, and final-evidence states stay distinct.
- Guided Story establishes its own live `demo_evening`, seed-42 paired session
  and advances through real congestion, recommendation, failure, and recovery.
- RL Information exposes the real decision pipeline, all 69 actions, masks, and
  12 reward components. Bandit scores are never labelled as probabilities.
- Recorded replay does not invent per-link utilization.
- The page has no document-level horizontal overflow at 1920, 1440, 1280, 768,
  or 390 px; the narrow topology owns its intentional internal pan.
- Keyboard focus, accessible topology list, reduced motion, and responsive
  landmarks are implemented.
- The Impeccable detector was run once. Its only relevant advisory was the
  drafting grid on the topology stage, retained as the design-specified
  measurement surface. The deliberate correction pass is complete.

## Follow-up scope

Keep follow-up narrow: browser/device spot checks, copy polish, and review of
small responsive details. Do not redesign the application or reopen product
definition. Start with `docs/PRODUCT_UI.md`, `docs/ACCESSIBILITY.md`, and the
approved specification under `docs/superpowers/specs/`.

Before changing anything, verify the branch is clean and run `py -m pytest -q`.
Do not modify or stage `results/environment_v2_validation/manifest.json`; its
expected SHA-256 in this worktree is
`5227C91A1990C52101A510BC7115C049E138D5B34C2CDD649C66C8AA726A979D`.

## Scientific boundaries

No training, tuning, evaluation, checkpoint selection, holdout execution, or
scientific-semantics change is authorized. Final evidence remains read-only.
The known honest limitations are documented in `docs/RELEASE_CHECKLIST.md`:
V2 live demonstration requires configured V2 checkpoints, recorded topology
animation is unavailable without per-link traces, and PPO entropy/value are not
exposed by the current live runner.
