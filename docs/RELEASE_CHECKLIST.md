# Release checklist — post-study productization

The scientific work is finished. This checklist covers releasing the *product*
around it without disturbing the record.

## Before any change

- [ ] `git status` shows only the intended working set. The one expected
      pre-existing change is `results/environment_v2_validation/manifest.json`,
      which is **preserved unstaged and never committed**.
- [ ] In the implementation worktree, that file's SHA-256 is
      `5227c91a1990c52101a510bc7115c049e138d5b34c2cdd649c66c8aa726a979d`.
      A mismatch stops the release.

      ```bash
      sha256sum results/environment_v2_validation/manifest.json
      ```

- [ ] Work happens in a dedicated worktree. `feat/rl-environment-v2` and the
      three experiment worktrees (`seed42`, `continuity_v2`, `final_holdout_v2`)
      stay untouched.

## Scientific boundary — non-negotiable

The V2 study is closed. A release must never:

- [ ] train, tune, or resume a learner;
- [ ] load a checkpoint for evaluation;
- [ ] reselect a checkpoint or run a sweep;
- [ ] re-run, retry, expand or reopen the final holdout;
- [ ] construct new holdout evidence;
- [ ] change reward, observation, action, topology, scenario, seed, metric, mask,
      horizon, baseline or evaluation semantics;
- [ ] modify a frozen definition or its hash;
- [ ] rewrite a manifest, compact table, checkpoint, sidecar or raw artifact;
- [ ] use a holdout result to motivate V2 tuning;
- [ ] add a fourth controller to the V2 comparison;
- [ ] pool episodes as if they were independent training roots;
- [ ] round a figure into a different conclusion or omit a reported result.

Verify mechanically rather than by inspection:

```bash
git status --porcelain results/ runs/ models/ configs/
```

Must be empty apart from the one preserved manifest.

## Evidence integrity

- [ ] Every displayed figure reconciles with the frozen files.

      ```bash
      python -m pytest tests/test_evidence_loader.py tests/test_evidence_claims.py -q
      ```

- [ ] The two look-alike pairs are still shown separately with their grain
      stated: no-op share (pooled-step vs episode-mean) and wall time (whole
      runner vs six checkpoint evaluations). See
      [V2_EVIDENCE_AUDIT.md](V2_EVIDENCE_AUDIT.md).
- [ ] Development and final-holdout evidence appear in separate regions and are
      never averaged.
- [ ] LIVE, RECORDED, DEVELOPMENT, and FINAL EVIDENCE remain distinct after
      repeated source switching; no stale live timeline survives a switch.
- [ ] Both halves of the planning statement appear together.

## Tests

- [ ] Evidence layer and surface:

      ```bash
      python -m pytest tests/test_evidence_loader.py tests/test_evidence_claims.py tests/test_evidence_replay.py tests/test_evidence_api.py tests/test_study_ui.py -q
      ```

- [ ] Existing presentation and API contracts:

      ```bash
      python -m pytest tests/test_presentation.py tests/test_api_e2e.py -q
      ```

- [ ] Unified shell, product API, and accessibility contracts:

      ```bash
      python -m pytest tests/test_product_contracts.py tests/test_product_api.py tests/test_product_ui.py tests/test_product_accessibility.py -q
      ```

- [ ] Freeze / pin gates and V1↔V2 compatibility:

      ```bash
      python -m pytest tests/ -q -k "freeze or pin or frozen"
      python -m pytest tests/test_v1_v2_compatibility.py -q
      ```

- [ ] Full suite:

      ```bash
      python -m pytest tests/ -q
      ```

Never run a command that regenerates a scientific artifact or writes into a
governed output directory. If a test writes a cache, keep it outside `results/`
and `runs/` and do not commit it.

## Product

- [ ] `/`, `/advanced`, `/present`, and `/study` all serve the unified shell.
- [ ] Exactly three primary modes are visible; Guided Story is nested inside
      Presentation.
- [ ] No page pulls an external asset — the demo machine may be offline.
- [ ] `/api/v2/*` is GET-only and exposes no train, tune, evaluate, select, sweep
      or rerun route.
- [ ] A broken artifact surfaces as HTTP 503 with a named error, never as zeros.
- [ ] Replay refuses any payload not marked `recorded_replay`.
- [ ] With `V2_FULL_ARTIFACTS` unset, the replay catalogue still lists all 315
      episodes and explains how to configure the path.

## Visual and accessibility

- [ ] Fresh loads checked at 1920×1080, 1440×900, 1280 px, 768 px, and 390 px.
- [ ] Zero horizontal page overflow at every width; wide tables scroll inside
      their own container.
- [ ] Charts never paint wider than the box CSS gave them.
- [ ] WCAG 2.1 AA contrast on all body, table and label text.
- [ ] No colour-only encoding — every series also carries a letter token and a
      printed number.
- [ ] Visible keyboard focus; skip link; landmarks.
- [ ] `prefers-reduced-motion` honoured.
- [ ] Loading, empty, failure and replay-unavailable states all reachable.
- [ ] Fixed router plates do not overlap, leave the atlas, or move during a
      session; the schematic/non-geographic disclaimer remains visible.

## Commit and push

- [ ] Full diff reviewed, including untracked files.
- [ ] Nothing large or prohibited staged:

      ```bash
      git diff --cached --stat
      git diff --cached --name-only | grep -Ei '\.(zip|pt|pth|gz|npz|pkl|db)$'
      ```

      The second command must print nothing.

- [ ] Protected manifest hash re-verified.
- [ ] Push **only** the dedicated post-study branch. Never push
      `feat/rl-environment-v2`.
- [ ] Protected manifest hash verified once more after the push.

## Known limitations of this release

- **No custom display typeface.** The frontend is build-free with no CDN and no
  npm, and committing a font binary is out of scope for a post-study release, so
  the page uses the system UI sans and spends its typographic identity on the
  tabular monospace numeric readout instead.
- **Replay is a timeline, not a topology animation.** The recorded step traces
  carry aggregate utilization (`max_util`, `mean_util`, `util_std`,
  `congested_links`) but not per-link utilization, so replay shows the real
  per-interval operational record. Colouring individual links would require
  inventing data and is deliberately not done.
- **V2 live policy demonstration is unavailable in this checkout.** Only V1
  live checkpoints are configured; the capability catalogue prints the reason.
- **PPO entropy/value are unavailable.** The live runner does not expose them,
  so the decision surface prints an unavailable reason rather than deriving
  substitutes.
- The scientific limitations of the study itself are unchanged; see
  [TECHNICAL_DEFENSE.md](TECHNICAL_DEFENSE.md) §8.
