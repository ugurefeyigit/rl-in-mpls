# Handoff for Codex

Everything you need to pick this up cold. Read § 1–3 before touching anything.

---

## 1. Where you are

| Field | Value |
|---|---|
| Repository | `C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls` |
| Working directory | `.claude/worktrees/final` — a git worktree, a complete self-contained copy |
| Branch | `final` (pushed to `origin/final`) |
| Head commit | `15cca12` — *feat(ui): paired comparison, results surface, and the release* |
| Tag | `v2.0.0` (pushed) |
| Parent commit | `5ed6d16` — Part 1 tip, branch `feat/live-v2-foundation-presentation-mode` |
| Tree status | clean apart from the CSS edit in § 6 |

```bash
cd .claude/worktrees/final && git status
```

Other worktrees under `.worktrees/` (`seed42`, `continuity_v2`,
`final_holdout_v2`, `post_study_productization`, `three_mode_ui`) hold frozen
training artifacts. **Do not touch them.** `.claude/worktrees/rl-mpls-ui-part1-c51fff`
is Part 1's worktree and is also untouched.

---

## 2. Non-negotiable constraints

These are enforced by tests, not by convention. Breaking one fails the suite.

1. **No training, tuning, evaluation, checkpoint selection or reselection, and
   no holdout access.** Holdout seeds 1001–1005 are refused for live sessions.
2. **Never write under `results/`, `runs/` or `models/`.** No `.zip`, no `.pt`.
3. **V1 stays byte-identical to the audited base.** Guarded by
   `test_models_results_figures_and_v1_configs_are_byte_identical_to_the_base`
   in `tests/test_v1_v2_compatibility.py`.
4. **`results/environment_v2_validation/manifest.json` is protected.** SHA-256
   must stay `5227C91A1990C52101A510BC7115C049E138D5B34C2CDD649C66C8AA726A979D`.
5. **`mplssim/sim/engine_v2.py` and `mplssim/rl/env_v2.py` are frozen
   scientific definitions.** Wrap them (`mplssim/product/live_v2.py` shows how);
   do not edit them.
6. **New or modified files must be added to the allowlist** in
   `tests/test_v1_v2_compatibility.py` (`ALLOWED_NEW_FILES` /
   `ALLOWED_MODIFIED_FILES`) **with a written reason**, in the same commit. This
   is deliberate friction — it makes every widening reviewable.

### Product-truth rules the code keeps

- A value the engine does not have is **absent with a reason**, never zero.
- A bandit score is never called a probability; the label comes from the
  controller's *declared* `output_semantics`, never from the shape of the number.
- A signed operational return never gets a percentage difference.
- Controller TE changes, FRR protection moves and post-recovery restorations are
  three counters, never summed.
- A live demonstration and the frozen holdout record never share a table.
- The frozen study's numbers are rendered from artifacts in exactly **one**
  place (`frontend/js/product/governed-study.js`). Never transcribe a figure.

---

## 3. Commands

```bash
python -m uvicorn server.main:app --port 8000
```
Then <http://127.0.0.1:8000/present>.

```bash
py -m pytest -q
```
Expected: **813 passed, 0 failed**. Takes ~110 s. Baselines: Part 1 was 758, the
original was 654 — never go below.

```bash
py -m pytest tests/test_part2_comparison.py tests/test_part2_results.py tests/test_part2_v2_endpoints.py -q
```

```bash
py -m pytest tests/test_v1_v2_compatibility.py -q
```

Optional: set `V2_LIVE_CHECKPOINTS` to a directory containing `seed42` and
`continuity_v2` if the training worktrees are not under `.worktrees/`. Without
it the two V2 learners show as unavailable **with the verification reason** and
the baselines still run. Nothing is ever silently substituted.

---

## 4. Architecture, in the order data flows

```
mplssim/sim/engine_v2.py        frozen simulator      ← never edit
mplssim/rl/env_v2.py            frozen environment    ← never edit
  ↓
mplssim/product/live_v2.py      EngineV2View: read-only product-shaped view
server/session.py               AlgoRunnerV2, SimSession, the state machine
  ↓
mplssim/product/serialize.py    typed snapshot
mplssim/product/decision.py     observation → mask → output → action → reward
mplssim/product/pairing.py      MAY we compare (fingerprint proof)
mplssim/product/comparison.py   WHAT the comparison shows (only if pairing said yes)
mplssim/product/results.py      three record classes, no cross-class aggregate
mplssim/product/run_summary.py  per-environment episode summaries for save-run
mplssim/product/timeline.py     typed event timeline
  ↓
server/product_api.py           /api/product/*, /api/simulation/*
server/evidence_api.py          /api/v2/* — GET-only frozen evidence
  ↓
frontend/js/product/            26+ ES modules, no build step, no CDN
```

**Key invariant:** only `frontend/js/product/adapters/*.js` may call `fetch`.
A test asserts no other module does.

**Atomicity:** `/api/simulation/moment` reads snapshot + decision + timeline +
comparison + advisor under the session lock and reconciles their provenance. The
frontend consumes that composite, not the individual routes. A test pins the
literal `const moment = await liveApi.moment()`.

---

## 5. What Part 2 changed (this commit)

Full detail in `OPUS5_PART2_HANDOFF.md`. Summary:

| Area | Change |
|---|---|
| Comparison | `mplssim/product/comparison.py` + rewritten `comparison-lane.js`. Broken proof ⇒ **no** verdict/metric/gap keys at all. |
| Results | `mplssim/product/results.py`, `GET /api/product/results`, `results.js`. Three record classes, three sections. |
| Retention | Reset run → session; full reset → process store; restart → dropped. Never written to disk. |
| Advisor fast-forward | `run-until` requires `delegate=true` under advisor execution; recorded as one `delegated_batch`. |
| save-run under V2 | Was broken (called V1's summarizer). V2 got its own; declares `not_measured` rather than zero-padding. |
| Guided Story | Beat 8's `approve` advance and beats 4–5's `select` were unhandled. Fixed; all 11 beats walked live. |

### Two traps that will bite you

1. **`session.advisor_history` holds two record shapes.** Branch on
   `record["kind"]`: `"proposal"` or `"delegated_batch"`. Both carry
   `step`/`t_min` so they can be placed on a timeline, but a batch has no
   `decoded`, no `action` and no `lookahead`. Assuming one shape is what broke
   `/api/simulation/moment` during Part 2's browser pass. Pinned by
   `test_a_delegated_batch_is_its_own_timeline_event_not_a_recommendation`.

2. **V1 and V2 decoded actions have different fields.** V1's carries `src`/`dst`;
   V2's does not. Never render a city pair from a V2 decoded action — see
   `_action_summary` in `comparison.py` for the guard.

---

## 6. The one uncommitted change

`frontend/css/presentation-mode.css` has two edits made after the commit, in
response to the Impeccable design hook:

- `.results__rule` — dropped a `border-left: 3px solid var(--state-pressure)`
  accent stripe; it now sets itself apart with `--surface-sunken` alone.
- `.panel--notice` — the same stripe became `border-top: 2px solid`, matching
  the `.cmp__verdict` idiom. It is a disclosure, not an alarm.

**Two other hook findings were classified as false positives and left alone:**
`.cmp__lane` (L58–60) and `.cmp__grid tbody tr[data-leader]` (L72–73). Those
left borders are not decoration — they are one of three redundant channels that
distinguish lane A from lane B (letter token, border *style* solid-vs-dashed,
colour), so the pairing survives greyscale and a colour-blind reader.
`test_lanes_are_distinguishable_without_colour` asserts the dashed style;
removing them would fail it and remove a real accessibility affordance.

Verified after the edit: `py -m pytest tests/test_part2_comparison.py
tests/test_presentation_controls.py tests/test_product_accessibility.py -q` →
92 passed. **The full suite has not been re-run since this edit** — do that
first, then commit it.

```bash
cd .claude/worktrees/final && py -m pytest -q && git add -A && git commit
```

---

## 7. Open items

Nothing from the Part 1 handoff is outstanding — all nine § 9 tasks are done.
What remains:

1. **Commit the CSS edit above** after a full-suite run.
2. **Create the GitHub Release object.** The tag `v2.0.0` is pushed but there is
   no Release. `gh` is not installed on this machine. Either install it, or use
   <https://github.com/ugurefeyigit/rl-in-mpls/releases/new?tag=v2.0.0> and paste
   `final/RELEASE_NOTES.md` as the body.
3. **Decide whether `final` merges to `main`.** It currently sits alongside it;
   there is no PR.

### Known limitations, all disclosed in the product

- PPO entropy and value estimates are not exposed by the live runner.
- Recorded replay has no per-link utilization.
- V2 has no manual traffic multiplier or burst injector; both endpoints 409.
- Retained runs do not survive a restart — a decision (ADR-003 § 2), not a gap.
- The V1 `provenance-word` element id is replaced on first render (pre-existing).
- `mplssim/product/checkpoints_v2.py` is the one product module permitted to
  import learner classes, for inference only. A test asserts no product module
  can train or save.

---

## 8. Documentation map

| Read | For |
|---|---|
| `final/README.md` | the release entry point |
| `final/RUNNING_IT_AGAIN.md` | exact commands, exact directories, troubleshooting |
| `final/OPERATING_THE_UI.md` | every control button-by-button, keyboard map |
| `final/RELEASE_NOTES.md` | what this release contains and refuses to do |
| `OPUS5_PART2_HANDOFF.md` | this stage's full record |
| `OPUS5_PART1_HANDOFF.md` | the previous stage, including checkpoint hashes |
| `docs/ADR-003-...md` | why demonstrations and evidence never merge; retention; delegation |
| `docs/RESULTS_AND_COMPARISON.md` | reference for both new surfaces |
| `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/PRODUCT_UI.md` | the system |
| `docs/TECHNICAL_DEFENSE.md` | methodology, roots, selection, limitations |

---

## 9. How to verify you have not broken anything

```bash
py -m pytest -q                                    # 813 passed
py -m pytest tests/test_v1_v2_compatibility.py -q  # V1 byte-identity
git status --porcelain -uall | grep -E "results/|runs/|models/|\.zip|\.pt"   # must be empty
```

```bash
py -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('results/environment_v2_validation/manifest.json').read_bytes()).hexdigest().upper())"
```
Must print `5227C91A1990C52101A510BC7115C049E138D5B34C2CDD649C66C8AA726A979D`.

For UI work, run the server and walk Guided Story end to end. Part 2 found two
real defects and one regression that way that the test suite did not catch.
