# Performance and Evaluation Audit

Runtime profiling, behaviour-preserving optimization, and an independent
integrity review of the offline evaluation path.

Branch: `optimize/runtime-evaluation`
Baseline commit: `652558e` (profiler added, engine untouched)
Scope: simulation engine, traffic model, RL environment observation/mask
construction, episode runner, evaluation script. No model was retrained, no
reward weight, configuration file or published result was modified.

---

## 1. Method

### 1.1 Environment

| | |
|---|---|
| Python | 3.13.4 |
| Platform | Windows-11-10.0.26200-SP0 (AMD64) |
| CPU | Intel(R) Core(TM) i7-14700HX |
| NumPy / pandas / SciPy | 2.3.0 / 2.3.0 / 1.18.0 |
| stable-baselines3 | 2.9.0 |
| Model | `models/ppo_te/best_model.zip` (loaded, never retrained) |
| Seed | 101 everywhere |
| Profiling scenario | `evening_peak` (420 min = 84 control intervals) |

### 1.2 Harness

`scripts/profile_runtime.py`. For each benchmark:

- **Warm-up**: 3 untimed iterations, after a global warm-up that primes the
  YAML config caches, the candidate-path cache and the import graph. These are
  process-lifetime caches; leaving them cold would measure them once and
  nothing thereafter.
- **Setup outside the timed region**: every repetition rebuilds its fixture
  (`setup()`) before the clock starts, so the measurement isolates the body.
- **Repetitions**: 11 per benchmark (`--quick` halves this). Operations faster
  than the clock resolution are repeated `inner` times inside one timed region
  and divided out.
- **Reported**: median, p95 and min. Median and p95 are the headline numbers;
  min is recorded because on a loaded desktop the median still absorbs
  background interference while the minimum is the most reproducible estimate
  of the code's own cost.

### 1.3 Why the numbers below are A/B, not before/after-in-time

A first pass measured "baseline" and "after" in separate sessions and produced
obvious artefacts — `env.reset` appeared 91% *slower* when a tight direct loop
showed it 23% *faster*. Machine state (thermal, background load) dominated.

All headline figures are therefore from **interleaved A/B rounds**: a git
worktree pinned at the baseline commit `652558e` and the optimized tree are
profiled alternately, back to back, with the *same* profiler script copied into
both. Rows where the profiler remained noise-dominated are re-measured with a
tight direct loop and reported as such.

Reproduce:

```bash
git worktree add /tmp/base_ab 652558e
cp scripts/profile_runtime.py /tmp/base_ab/scripts/
for r in 1 2 3 4 5; do
  (cd /tmp/base_ab && python scripts/profile_runtime.py --label base_r$r --no-model)
  python scripts/profile_runtime.py --label head_r$r --no-model
done
# RL rows (slower, 2 rounds is enough for episode-length benchmarks):
for r in 1 2; do
  (cd /tmp/base_ab && python scripts/profile_runtime.py --label rlbase_r$r --only rl. eval. episode.)
  python scripts/profile_runtime.py --label rlhead_r$r --only rl. eval. episode.
done
```

The profiler guards benchmarks for methods that do not exist on older revisions
(`fast_clone`, `_lsp_counts`), so the same script runs unmodified against the
baseline commit.

Raw timings land in `results/runtime_audit_*.json`, which is git-ignored
(`results/*.json`); the tables below are the record.

---

## 2. Baseline

Median across 5 interleaved baseline rounds at commit `652558e` (RL rows from
the 2 model-enabled rounds).

| Component | Median | Throughput |
|---|---|---|
| Engine control interval (`step_interval`) | 1.392 ms | 718 intervals/s |
| Micro-tick (interval / 5) | 278.5 us | 3 591 micro-ticks/s |
| `_compute_tick` | 329.9 us | 3 031 ticks/s |
| Action-mask generation (68 actions) | 134.2 us | 7 450 masks/s |
| Observation vector (586 floats) | 215.6 us | 4 638 obs/s |
| API snapshot | 3.359 ms | 298 snapshots/s |
| Snapshot + `json.dumps` | 3.771 ms | 265 payloads/s |
| Engine clone (`copy.deepcopy`) | 1.363 ms | 734 clones/s |
| Clone + one interval (counterfactual) | 3.090 ms | 324/s |
| `path_available` x68 | 60.0 us | — |
| `validate_action` x68 | 119.6 us | — |
| `candidate_info` (all 17 demands) | 721.9 us | — |
| `candidate_info` (one demand) | 43.9 us | — |
| Engine construction | 407.1 us | — |
| RL env `reset` | 611.8 us | — |
| RL env `step` | 2.216 ms | 451 steps/s |
| RL `predict` only | 1.306 ms | 766 predictions/s |
| RL predict + env step | 6.938 ms | 144 decisions/s |
| Full episode, `full_day` (engine only) | 569.2 ms | 1.8 episodes/s |
| Full episode, `demo_evening` (engine only) | 123.3 ms | 8.1 episodes/s |
| Evaluation, 1 scenario x 4 baselines x 1 seed | 616.8 ms | 1.6 sweeps/s |
| Evaluation, 1 scenario x 5 algorithms x 1 seed | 1.241 s | 0.8 sweeps/s |

Peak traced allocation per call (baseline): interval 10.4 KiB, snapshot
82.6 KiB, deepcopy clone 192.2 KiB, action mask 1.9 KiB, whole `demo_evening`
episode 91.8 KiB.

### 2.1 Where the time actually went

`cProfile`, baseline, `evening_peak` after 20 intervals:

| Hot path | Dominant cost | Share |
|---|---|---|
| `snapshot` | `n_lsps` generator: `i in path` membership scan for every (link, demand) pair | **49%** |
| `snapshot` | `candidate_info` (4 NumPy reductions per candidate, 272 calls) | 38% |
| `_obs` | `path_bottleneck_util` called 68 times | **54%** |
| `action_masks` | `path_available`'s per-hop generator | 46% |
| `action_masks` | projected-load array copies for protected demands | 25% |
| `_compute_tick` | per-demand delay/loss loop | **49%** |
| `_compute_tick` | `np.percentile` for p95 delay | 16% |

---

## 3. Results

### 3.1 Headline comparison

Interleaved A/B, 5 rounds of the final code (2 rounds for the RL rows).
Positive = faster. `d%med` compares medians; `d%min` compares the minimum
sample, which is less contaminated by background load.

| Component | Base | After | d%med | d%min |
|---|---|---|---|---|
| **API snapshot** | 3.359 ms | **319.7 us** | **+90.5%** | +88.1% |
| Snapshot + `json.dumps` | 3.771 ms | 577.2 us | +84.7% | +80.9% |
| **Observation vector** | 215.6 us | **37.9 us** | **+82.4%** | +78.9% |
| `candidate_info` (one demand) | 43.9 us | 12.7 us | +71.2% | +69.7% |
| `candidate_info` (all demands) | 721.9 us | 215.1 us | +70.2% | +69.0% |
| **Action-mask generation** | 134.2 us | **55.6 us** | **+58.5%** | +58.9% |
| RL env `step` | 2.216 ms | 1.055 ms | +52.4% | +45.0% |
| Episode `full_day` (engine only) | 569.2 ms | 287.8 ms | +49.4% | +46.7% |
| Episode `demo_evening` (engine only) | 123.3 ms | 65.1 ms | +47.2% | +45.2% |
| Episode `demo_evening` (static controller) | 160.4 ms | 84.7 ms | +47.2% | +41.9% |
| Episode `full_day` (static controller) | 553.3 ms | 303.4 ms | +45.2% | +45.0% |
| RL predict + env step | 6.938 ms | 3.895 ms | +43.9% | +31.9% |
| `_compute_tick` | 329.9 us | 190.3 us | +42.3% | +42.9% |
| RL episode `demo_evening` | 473.9 ms | 281.6 ms | +40.6% | +41.4% |
| **Evaluation, 1 scenario x 5 algos x 1 seed** | 1.241 s | **751.4 ms** | **+39.5%** | +39.5% |
| **Engine control interval** | 1.392 ms | **896.2 us** | **+35.6%** | +43.0% |
| Evaluation, 1 scenario x 4 baselines x 1 seed | 616.8 ms | 439.1 ms | +28.8% | +41.7% |
| Clone + one interval (counterfactual) | 3.090 ms | 2.288 ms | +26.0% | +19.5% |
| RL env `reset` | 611.8 us | 506.5 us | +17.2% | +20.8% |
| `path_available` x68 | 60.0 us | 54.6 us | +9.1% | +12.3% |
| `validate_action` x68 | 119.6 us | 109.5 us | +8.4% | +5.4% |
| Engine clone (`copy.deepcopy`) | 1.363 ms | 1.338 ms | +1.8% | +0.3% |
| Engine construction | 407.1 us | (noise) | -31.8% | +1.1% |

`clone()` is unchanged by design — the deepcopy path was left exactly as it was
so existing callers keep their semantics; the 1.8% is noise. Engine
construction is the one row the profiler could not measure stably (median and
min disagree in sign); a tight direct loop puts it at 394.7 us -> 396.1 us,
i.e. unchanged — the extra precomputation costs about 1 us.

New capability (no baseline equivalent):

| Component | After | vs `clone()` |
|---|---|---|
| `fast_clone()` | **36.5 us** | **37x faster than `copy.deepcopy` (1.338 ms)** |
| `fast_clone()` + one interval | 1.033 ms | 2.2x faster than clone + interval |
| `_lsp_counts()` | 14.0 us | replaces ~1.6 ms of membership scanning |

### 3.2 On measurement stability

Short benchmarks on a loaded laptop are noisy, and an early two-round pass
produced figures that later rounds contradicted (`step_interval` read +11%
then, +35.6% over five rounds; `env.reset` once read 91% *slower* where a tight
direct loop showed it 21% *faster*). Everything above therefore comes from five
interleaved rounds, with the min column as a cross-check, and the two rows that
still disagree between median and min are called out explicitly rather than
having the flattering number quoted.

The whole-episode and evaluation figures are the most trustworthy: they run for
hundreds of milliseconds, so scheduler noise is a small fraction, and they agree
closely between median and min (+45% to +49%).

### 3.3 Acceptance targets

| Target | Result | Met |
|---|---|---|
| >= 20% faster raw engine stepping | control interval **+35.6%** (min +43.0%); `_compute_tick` +42.3%; engine-only episodes +47.2% / +49.4% | yes |
| >= 30% faster snapshot generation | **+90.5%** | yes |
| >= 25% faster action-mask generation | **+58.5%** | yes |
| >= 20% faster one-seed evaluation | **+39.5%** (5 algorithms incl. RL); +28.8% for the 4 baselines alone | yes |
| Materially faster lightweight clone | `fast_clone` **36.5 us vs 1.338 ms** = 37x | yes |

### 3.4 Memory

Peak traced allocation per call, min across rounds:

| Operation | Base | After | Change |
|---|---|---|---|
| `fast_clone` | n/a | **27.9 KiB** | **86% less than deepcopy** |
| Engine interval | 10.4 KiB | 10.2 KiB | -2.5% |
| Whole `demo_evening` episode | 91.8 KiB | 91.0 KiB | -0.9% |
| Snapshot | 82.6 KiB | 88.0 KiB | +6.5% |
| Deepcopy clone | 192.2 KiB | 206.1 KiB | +7.3% |
| Action mask | 1.9 KiB | 3.9 KiB | +104% |
| RL env step | 10.7 KiB | 12.7 KiB | +18.1% |

**Allocation went up in four places, and that is a real trade.** The vectorized
candidate matrices and mask arrays are transient allocations bought in exchange
for large time savings; in absolute terms the increases are 2 KiB (mask),
2 KiB (env step), 5 KiB (snapshot) and 14 KiB (deepcopy clone, which now
copies a few more precomputed arrays). Steady-state memory per episode is
unchanged. The one place where allocation mattered — cloning for
counterfactuals — improved by 86%.

---

## 4. Optimizations retained

Each one preserves floating-point operand order, so results are bit-identical
rather than merely close. See section 6 for the evidence.

1. **Per-directed-link operational-state array** (`_dlink_up`).
   `link_up` stays the authoritative, API-visible dict; the array mirrors it and
   is rebuilt wholesale by `set_link_state` (link changes are rare, and a full
   rebuild cannot drift out of sync with a partial patch). The array exists for
   whole-matrix reductions, not for scalar lookups — see section 5.

2. **Padded candidate/link incidence matrices.** `_cand_pad` (n_demands, k,
   max_hops) pads short paths by *repeating the path's own first link*. Because
   `max`, `min` and `all` are idempotent, a duplicated entry cannot change the
   result — which is what makes the vectorized reductions bit-identical to the
   per-path loops. Non-existent candidates are masked by `_cand_exists`.
   `candidate_matrices()` computes availability, bottleneck utilization,
   projected bottleneck and available bandwidth for all 68 candidates in four
   whole-array reductions instead of 272 tiny ones.

3. **Second padding scheme for non-idempotent reductions** (`_cand_pad_sum`).
   Sum and product cannot be padded by repetition, so these rows point at a
   sentinel slot holding the identity element (0.0 for the delay sum, 1.0 for
   the delivered-fraction product). Used by the vectorized per-demand
   delay/loss/SLA computation.

4. **LSP counting by accumulation** (`_lsp_counts`). One pass over demands
   incrementing a per-link counter, replacing an `i in path` membership scan for
   every (link, demand) pair — the single largest cost in the baseline snapshot.

5. **Cached immutable snapshot data.** Topology-derived constants (per-link ids,
   endpoints, capacity, weight, propagation delay; router payloads) are built
   once per engine. **Key order in the payload is preserved exactly** as the
   frontend has always seen it, and the router dicts are copied on the way out
   so a consumer cannot mutate the cache.

6. **Projected-load base computed once per demand**, not once per candidate.
   A link shared by the old and new path goes through the same `(load - vol) +
   vol` sequence either way, so the values are identical; the array copies drop
   from 68 to 17 per snapshot. Two separate scratch buffers (`_proj_buf`,
   `_sweep_buf`) so a sweep holding a base cannot have it clobbered by an
   unrelated single-shot projection call.

7. **Precomputed per-demand constants**: class priorities, SLA thresholds,
   candidate counts, base rates — previously rebuilt on every micro-tick.

8. **Vectorized per-demand delay/loss/SLA** via sentinel-padded gather, with a
   **structural fallback**. Padding a sum with zeros is only bit-exact while
   NumPy sums the row sequentially; above 8 elements it switches to pairwise
   blocking, which regroups the real terms. `_vectorize_demand_metrics` is
   therefore `max_hops < 8` (currently 6), and the scalar loop is kept as a live
   fallback. Both branches are asserted to agree bit-for-bit, and the fallback
   is separately asserted to reproduce the golden episode summaries — so it
   cannot rot into untested dead code.

9. **`fast_clone()`.** Shares immutable topology/configuration; copies every
   piece of mutable state including the traffic RNG and AR(1) noise. `clone()`
   is untouched and still does a full `copy.deepcopy`, so existing callers
   (`server/session.py` uses it in three places) keep exactly the semantics they
   were written against. Documented difference: `fast_clone` copies
   `action_log` and `metrics_history` as *lists*, sharing the append-only record
   objects; appending to one engine never affects the other, and nothing mutates
   a recorded entry in place. A `_SHAREABLE_ATTRS` declaration plus an
   exhaustive test guard the boundary — the test walks the real attribute dict,
   so the next mutable field added to the engine fails loudly rather than being
   silently shared with a counterfactual.

10. **Traffic model precomputation.** AR(1) sigmas, base rates and per-event
    demand masks are computed once; the distinct diurnal profiles are
    interpolated once per curve per tick instead of once per demand (17 demands
    share a handful of curves). The volume product keeps its exact operand
    order: `((((base * profile) * demand_multiplier) * event) * noise)`.

11. **Vectorized observation and action mask.** The mask applies candidate
    existence, path availability, same-path and cooldown as array operations;
    only the protected-class bandwidth check still runs per candidate, and only
    for candidates that survived the cheap checks. Verified equal to a per-action
    `validate_action` sweep, including in the disconnected regime.

12. **Per-metric evaluation buffering fixes** — see section 7.

---

## 5. Optimizations attempted and rejected

Reported because "we tried X and it did not help" is as much a result as a
speedup.

### 5.1 Rejected: `np.round` in place of Python `round` in `snapshot`

`snapshot` calls Python `round` about 590 times per payload (~0.15 ms).
Replacing them with vectorized `np.round` looked attractive.

**Rejected — they are not the same function.** They agreed on all 290 live
telemetry values at every precision tested, which is exactly what makes this
dangerous: the divergence only appears at representable half-way points.

```
round(2.675, 2)    -> 2.67      # Python: 2.675 is really 2.67499...
np.round(2.675, 2) -> 2.68      # NumPy rounds the binary value differently
```

The snapshot payload is an API contract. Swapping a rounding rule to save
0.15 ms is an unforced correctness risk, and the golden snapshot test would
have to be re-baselined to accept it.

### 5.2 Rejected: NumPy indexing for scalar `path_available`

Initially `path_available` used the new boolean array
(`self._dlink_up[links].all()`). Measured over 68 candidates:

| Implementation | Time |
|---|---|
| dict lookups per hop (original) | 43.3 us |
| boolean-array indexing | 56.9 us |
| **all 68 at once as a matrix** | **1.9 us** |

**Reverted to dict lookups.** A path is only 4–6 hops, so two ufunc calls cost
more than the lookups they replace; the array only pays off when every
candidate is reduced together. This is why the codebase now deliberately uses
*both* representations: dicts for scalar queries, the array for matrix
reductions. They are asserted to agree across a whole episode. The final
`path_available` is 9.1% faster than baseline; the intermediate array-based
version was ~4% *slower*, and shipping it would have been a regression hidden
inside an otherwise-positive changeset.

### 5.3 Revised: `candidate_info` rebuilding all matrices for one demand

The first version had `candidate_info(d)` call `candidate_matrices()` and take
one row. Snapshot (which needs all rows) got much faster, but a single-demand
query got **14% slower** (41.9 us -> 47.8 us), and 17 separate calls were 21%
slower than baseline.

Fixed by adding `candidate_row(d)`, which reduces over that demand's rows only.
Result: 43.9 us -> 12.7 us (**+71.2%**), and the two are asserted to agree
element-for-element. Worth recording as a near-miss: the aggregate benchmark
looked excellent while a real API path had regressed.

### 5.4 Rejected: unguarded padded summation

The vectorized delay sum is only bit-exact while NumPy stays on its sequential
summation path. Shipping it without a guard would have made bit-exactness an
accident of a NumPy implementation detail (the pairwise-blocking threshold) that
a dependency upgrade or a longer candidate path could silently break. Adopted
only with the `_MAX_SEQUENTIAL_SUM` guard and a live, tested fallback (retained
item 8).

### 5.5 Not attempted: reducing work

The brief rules out optimizing by cutting micro-ticks, scenarios, demands,
paths, metrics or precision, and none of that was done. `np.percentile` for the
p95 delay remains ~16% of a micro-tick and was left alone: every alternative
changes the statistic.

### 5.6 Not pursued: duplicate engine construction in `MplsTeEnv`

`MplsTeEnv.__init__` builds an engine that `reset()` immediately discards, so
every episode constructs two. Measured cost is ~400 us against a ~280 ms RL
episode (0.14%). Removing it would require making the constructor's derived
attributes lazy, for no measurable benefit. Documented, not changed.

---

## 6. Numerical-equivalence evidence

`tests/test_runtime_equivalence.py` — 50 tests, all passing.

### 6.1 Design

These are **characterization tests**: every golden literal was captured from the
engine at commit `4b8de03`, *before* any optimization, and embedded in the test
file. Regenerating a golden to make a failure disappear would defeat the file's
purpose, and the header says so.

Three layers:

1. **Golden trace** — a 40-step scripted episode over `link_failure` (L11 drops
   at step 12, is restored at step 36). The action script deliberately exercises
   every constraint: 24 accepted reroutes, 16 rejections, 27 reroutes,
   **3 flaps**, **3 FRR repairs**, 253 SLA violations, 99 congested link-steps.
   Pinned exactly: link loads, utilization, queueing delay, loss, per-demand
   delay/loss, SLA state, offered traffic volumes, EWMA trend, current
   placement, all 20 aggregated interval metrics for all 40 steps, all 12 reward
   components for all 40 steps, the 69-bit action mask, the 586-float
   observation (sampled values, sum, min, max **and a SHA-256 of the raw
   bytes**), the complete snapshot payload (every per-link, per-demand and
   per-candidate field), and whole-episode summaries for all four baseline
   controllers on two scenarios.
2. **Reference cross-checks** — the engine is compared against deliberately
   naive re-implementations (scalar link-load accumulation, per-hop availability
   from the `link_up` dict, per-action `validate_action` mask sweep, whole-array
   projected loads, the analytic delay/loss formulas), sampled at eight points
   across an episode. This catches semantic changes in states no golden covers.
3. **Isolation** — engines, clones and shared caches must not leak mutable state
   or perturb each other's RNG.

Tolerance policy: **strict equality** (`assert_exact`), not `allclose`. The
optimizations were designed to preserve operand order, so exact equality is
achievable and is the strongest available evidence. No assertion was loosened.

### 6.2 The tests were mutation-tested

A characterization test that passes proves nothing until it is shown to fail on
a real change. Eight deliberate semantic mutations were injected and reverted:

| Mutation | Caught by |
|---|---|
| `PROC_DELAY_MS` 0.2 -> 0.2000001 | 5 tests |
| Reversed link-load accumulation order | 7 tests |
| Dropped the protected-class bandwidth check | 2 tests |
| `n_lsps` ignores the disconnected filter | 1 test |
| Cooldown off-by-one (`<` -> `<=`) | 9 tests |
| Disconnected demand keeps offering load | 1 test |
| EWMA coefficient 0.8/0.2 -> 0.81/0.19 | 2 tests |
| Flap window 6 -> 8 | 4 tests |

**The first run found two genuine blind spots**, which is the point of doing it:

- *`n_lsps` ignoring the disconnected filter was caught by nothing.* No demand
  ever disconnects in the golden trace, so the branch was untested. Added
  `_disconnect_engine()` (fails every link on demand 0's candidates, producing
  8 disconnected demands) and tests for load exclusion, LSP counting,
  delay/loss/SLA handling, snapshot flags, recovery, and mask equivalence in
  that regime.
- *Dropping the protected-class bandwidth check was caught only by the episode
  summary* — the probe-step mask happened not to change. Added
  `GOLDEN_VALIDATE_REASONS`: the *reason* for all 68 actions at all 40 steps,
  encoded one character per action. This pins each constraint separately
  (same-path, failed link, cooldown, protected bandwidth), so one constraint
  cannot hide behind another still rejecting the action. A companion test
  asserts the trace still exercises all five outcomes.

After closing both gaps, all eight mutations are caught.

### 6.3 Isolation guarantees, specifically tested

- Two engines stepped interleaved match engines run in isolation.
- Four parallel engines (different scenarios and seeds) do not contaminate one
  another across 15 mutable state arrays.
- No mutable array, dict or list is shared between two engines.
- Any ndarray shared between engines must be flagged read-only (the test
  discovers shared arrays by object identity rather than trusting a list).
- Cloning does not advance or alter the original's RNG, and driving the clone
  hard afterwards still does not.
- Writing into any of the clone's 15 mutable arrays leaves the original
  untouched; clone histories, path histories and burst lists are independent.
- `fast_clone` shares nothing mutable that is not explicitly declared in
  `_SHAREABLE_ATTRS`, and everything declared shareable really is shared.
- `fast_clone` and `clone` produce identical snapshots and identical interval
  metrics when driven with the same actions for six intervals.
- Offered traffic is bit-identical between an engine that never reroutes and one
  that reroutes on every interval, including the AR(1) noise state.
- Scripted failures land at the same simulated instant regardless of controller
  behaviour.

---

## 7. Evaluation integrity audit

`scripts/audit_evaluation.py` — 18 executable checks; **15 pass, 3 findings,
0 failures**. Every claim below is re-verifiable by running that script.

### 7.1 Verified sound

| Property | Evidence |
|---|---|
| Identical traffic per (scenario, seed) across algorithms | 4 algorithms, 60 steps, `offered_mbps` exactly equal |
| Traffic RNG unaffected by controller behaviour | AR(1) noise, generator state and volumes identical after 40 intervals with vs without reroutes |
| All runners advance the same intervals and simulated time | steps=60, final `t_min`=300.0 for every algorithm |
| `all_disconnected` never fires in the evaluated scenarios | peak simultaneous disconnections: `ood_double_failure` 4/17, all others 0/17 |
| FRR distinguished from controller actions | `action_log` sources `{rl: 60, frr: 3}`; accepted FRR count equals summed `frr_events` |
| Flap attribution excludes FRR | baselines slice `action_log` from the pre-decision mark; RL reads the flag off its own action |
| `recovery_steps` computed consistently | matches an independent recomputation for all four algorithms |
| Paired deltas align on (scenario, seed) | both sides indexed by the same intersection object; no duplicate keys |
| `dropped_gbit_total` units | `sum((offered - carried) Mbps * 300 s) / 1000` = 4195.459 Gbit, confirmed |
| Outputs truncate, never append | every writer uses truncating `to_csv` / `write_text` |
| `summarize_records` is pure | does not mutate the frame it is given |
| Benchmark sources are regenerable | command recorded in section 7.4 |

### 7.2 Findings fixed (in owned files, additively)

All four fixes are **additive** — no existing column, metric or value changes
meaning, so historical benchmark interpretation is preserved.

1. **`n_seeds` could overstate the sample behind a statistic.**
   `scripts/evaluate.py` set `n_seeds = len(g)` once per (scenario, algorithm)
   group, then computed each metric over `g[metric].dropna()`. `recovery_steps`
   is `None` for any scenario without a failure, so its mean/std/CI could rest on
   fewer samples than `n_seeds` advertised, with nothing recording the real
   count. Additionally `ci95()` returns 0.0 for n < 2, which renders as a
   zero-width interval rather than "undefined".
   **Fix**: a `<metric>_n` column now records the true per-metric sample size,
   which also distinguishes a `0.0` CI from a genuinely tight one. Verified: in
   a 2-seed x 2-scenario sample, `recovery_steps_n` is NaN for `evening_peak`
   (no failure) while `n_seeds` is 2.

2. **Duplicate seeds would silently corrupt paired deltas.** A repeated
   `--seeds` value creates duplicate (scenario, seed) index keys, making the
   paired `.loc` lookups expand instead of aligning one-to-one.
   **Fix**: `evaluate.py` now rejects duplicate seeds with an explicit error.

3. **Hardcoded control interval in `summarize_records`.** `dropped_gbit_total`
   used a literal `5 * 60` seconds, not derived from `EngineConfig`; a changed
   `control_interval_min` would have silently mis-scaled a headline metric.
   **Fix**: the interval is now read from the trace itself (`t_min` after the
   first interval *is* one control interval). The value is unchanged at the
   current 5 minutes — the golden episode summaries assert this.

4. **Final message named the wrong files.** `evaluate.py` printed
   `eval_summary.csv` regardless of `--prefix`.
   **Fix**: it names the files actually written, and additionally warns when
   older `<prefix>_steps_*.csv` files from a previous, wider run are still
   present — those are *not* overwritten by a narrower re-run and would still be
   picked up by `make_figures.py`. (35 such files currently exist in
   `results/`.) Use `--prefix` to isolate a run.

### 7.3 Findings documented, deliberately not fixed

These would change evaluation semantics or the meaning of a published metric.
Fixing them is a call for the owner of the evaluation methodology, not a
performance change to smuggle in here.

1. **RL and baseline runners use different termination conditions.**
   `_run_rl` stops on `terminated or truncated`, where `terminated` is
   `engine.all_disconnected`; `_run_baseline` stops only on `eng.done`
   (scenario duration reached). If every demand were ever simultaneously
   disconnected, the RL episode would be summarized over fewer intervals than
   the baselines it is paired against, and per-step means would not be
   comparable.
   *Exposure*: measured — it never fires. Peak simultaneous disconnection is
   4/17 (`ood_double_failure`); every other evaluated scenario reaches 0.
   *Mitigation applied*: the RL summary now carries `terminated_early`, so any
   future occurrence is visible instead of silently shortening an episode. The
   loop itself is unchanged, because `terminated` is part of the Gymnasium
   contract.
   *Recommended*: either run RL to the scenario duration like the baselines, or
   drop `terminated_early` episodes from paired comparisons.

2. **Decision timing measures different things for RL and baselines.**
   `_run_baseline` times `ctl.decide(eng)`, which *includes* each heuristic's own
   feasibility scanning. `_run_rl` computes `env.action_masks()` **outside** the
   timed region and times only `model.predict()` — so the safety filter's mask
   generation, a real part of an RL decision, is excluded while the baselines'
   equivalent work is included.
   *Magnitude*: mask generation is ~56 us/step after optimization (~134 us
   before), against an RL `predict` of ~0.8 ms.
   *Mitigation applied*: `mean_mask_time_ms` is now reported separately.
   `mean_decision_time_ms` keeps its original meaning so historical numbers stay
   comparable; add the two for a like-for-like comparison.

3. **`recovery_steps` conflates three different situations.**
   `summarize_records` returns `None` when a scenario has no failure, but a
   *censored* value (`n - f0`, meaning "never recovered") when SLA violations
   never return to zero. Downstream, a censored value is indistinguishable from
   a genuine fast recovery, and it is a lower bound that depends on episode
   length.
   *Recommended*: report recovery as a (recovered: bool, steps) pair, or emit a
   separate `recovery_censored` flag, so means are not taken over a mixture.

4. **Baselines are never charged the invalid-action penalty.** `_run_baseline`
   calls `compute_reward(..., invalid=False)` unconditionally, while `_run_rl`
   passes the real flag.
   *Exposure*: measured — zero baseline actions were rejected in
   `link_failure`/seed 101 across all four baselines, so no reward has been
   affected. It remains a latent asymmetry if a baseline ever proposes an
   invalid move (`static` and `cspf` do not pre-check the protected-class
   bandwidth constraint, so this is reachable in principle).
   *Not fixed*: passing the real flag would change baseline reward sums and
   therefore the interpretation of published results.

### 7.4 Regenerating the benchmark sources

`server/main.py` reads `results/eval_stats.csv`. All three benchmark files are
present and regenerable:

```bash
python scripts/evaluate.py --model ppo_te \
  --scenarios full_day evening_peak flash_crowd link_failure \
    deceptive_local_optimum ood_double_failure overload_stress \
  --seeds 101 102 103 104 105 \
  --algorithms static greedy cspf random rl
```

No published result file was regenerated or overwritten during this work.
Verification runs used isolated prefixes (`--prefix runtime_audit_smoke`) and
were deleted afterwards; `scripts/audit_evaluation.py --write-outputs` writes
only to `results/runtime_audit_eval/`.

---

## 8. Known consequence: a speed-sensitive server test

`tests/test_state_machine.py::test_manual_step_requires_pause_and_advances_exactly_one`
became timing-sensitive. Measured behaviour:

- **In isolation: fails deterministically** (8/8 runs, and 3/3 when running
  `tests/test_state_machine.py` alone).
- **In the full suite: intermittent** — 128 passed, 128 passed, 1 failed across
  three consecutive runs. Concurrent CPU load slows the session loop back below
  the threshold, so the result depends on machine load.
- On the baseline commit it passed 5/5 in isolation.

This is a real consequence of the speedup, not a pre-existing flake, and it is
**not fixable from this branch** — the test and `server/` belong to other
agents.

Cause, measured directly: the test starts an `evening_peak` session (84
intervals) at `speed="fast"` (no pacing delay), sleeps 0.2 s, and assumes the
episode is still running. Time for that session to run to completion:

| | Time to complete |
|---|---|
| Baseline | **0.462 s** (> 0.2 s, so still running — test passes) |
| Optimized | **0.178 s** (< 0.2 s, so already `COMPLETED`) |

The first assertion then passes for the wrong reason — the 409 comes from
`step_manual`'s *"scenario finished"* branch rather than *"pause before
stepping manually"* — and the subsequent step returns 409 instead of 200.

Simulation semantics are unchanged; only wall-clock speed moved. Suggested
fixes for the owner: start with `autostart=False` and assert on reported state
instead of sleeping; or use `speed="1x"` for this test; or assert the 409
*reason* so the two cases cannot be confused.

---

## 9. Compatibility

Unchanged and verified by the test suite:

- **Pretrained model** — `models/ppo_te` loads and runs; no retraining.
- **Observation ordering and normalizations** — SHA-256 of the raw observation
  bytes is pinned. Note the vector is laid out *feature-major*
  (`dm` is `(features, demands)` and is raveled row-wise); the vectorized
  rewrite preserves this exactly.
- **Action numbering** — `1 + d * k + p`, mask asserted bit-identical.
- **Reward semantics** — all 12 components pinned for all 40 golden steps; no
  weight touched.
- **Seeded traffic traces** — offered volumes bit-identical.
- **Scenario timing** — failure/recovery land on the same steps.
- **API snapshot shape** — key sets asserted for the payload, links, demands,
  candidates and routers; key *order* also preserved.
- **Number of candidate paths** — unchanged (k = 4).
- **Action masking semantics** — asserted equal to a per-action
  `validate_action` sweep, including in the disconnected regime.
- **Existing tests** — all 78 pre-existing tests still pass; total suite is 128
  (78 existing + 50 new). See section 8 for the one test that became
  speed-sensitive.

Files changed on this branch:

| File | Change |
|---|---|
| `mplssim/sim/engine.py` | optimizations, `fast_clone`, incidence matrices |
| `mplssim/traffic/model.py` | precomputation, vectorized volumes |
| `mplssim/rl/env.py` | vectorized observation and action mask |
| `mplssim/experiments/runner.py` | derived interval length, `mean_mask_time_ms`, `terminated_early` |
| `scripts/evaluate.py` | per-metric `_n`, duplicate-seed guard, accurate output message |
| `scripts/profile_runtime.py` | new — profiling harness |
| `scripts/audit_evaluation.py` | new — evaluation integrity checks |
| `tests/test_runtime_equivalence.py` | new — 50 equivalence and isolation tests |
| `docs/PERFORMANCE_AND_EVALUATION_AUDIT.md` | new — this document |

Not touched: `mplssim/baselines/**`, `server/**`, `frontend/**`, `configs/**`,
`models/**`, `results/eval_*`, training scripts, CI files, existing
presentation documents, reward weights.
