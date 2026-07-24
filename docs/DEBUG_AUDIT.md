# Debug audit — presentation hardening

The V1 application was treated as an external submission and audited against
the full checklist in the hardening brief. Each finding lists the observable
symptom, the **root cause**, and the fix. Verified by the test files
`tests/test_state_machine.py` and `tests/test_correctness_fixes.py`
(all listed tests pass).

## A. Session lifecycle bugs (fixed in `server/session.py`)

### A1. Resume could create two concurrent simulation loops
**Symptom:** rapid pause→resume produced double-speed ticking; step counter
jumped by 2 per interval.
**Root cause:** V1 `pause()` only set `running = False`; the loop task kept
executing its current iteration/sleep. `start()` then unconditionally created
a *new* task while the old one could observe `running == True` again and
continue — two loops stepping one engine.
**Fix:** explicit state machine (`idle/running/paused/completed/error`) with a
single `asyncio.Lock` around every mutation; `resume()` only spawns a task if
none is alive (a paused-but-alive task simply continues); repeated
pause/resume are idempotent no-ops.
**Test:** `test_repeated_pause_resume_idempotent_single_loop`.

### A2. Step counter could advance after pause responded
**Symptom:** UI showed +1–2 steps after clicking Pause.
**Root cause:** no synchronization between the HTTP handler and the tick loop;
pause returned before the in-flight interval finished, and a second interval
could start because the loop checked the flag only at iteration top.
**Fix:** the loop holds the session lock for the whole interval computation;
`pause()` acquires the same lock, so at most one in-flight interval completes
during the transition and none can start after.
**Test:** `test_pause_freezes_step_counter`.

### A3. Reset silently discarded model tag and safety-filter setting
**Symptom:** a session started with `safety_filter=false` or a custom model
tag reverted to defaults after Reset (silent fallback to the default model).
**Root cause:** V1 rebuilt the session from only
`(scenario, algorithms, seed, speed)` — the other fields of the start request
were dropped.
**Fix:** immutable `SessionConfig` dataclass stores the complete
configuration; reset rebuilds runners from that exact config.
**Test:** `test_reset_preserves_full_configuration`.

### A4. Old session could keep ticking (and broadcasting) after reset/start
**Symptom:** occasional "ghost tick" with stale time right after reset; charts
got one point from the previous run.
**Root cause:** V1 reset constructed a new `SimSession` but never cancelled or
awaited the old loop task; its final broadcast raced the swap.
**Fix:** reset bumps a generation counter under the lock, cancels and awaits
the old task; loops re-check their generation before broadcasting, so a stale
payload can never be delivered.
**Tests:** `test_reset_while_running_stops_session`,
`test_failure_works_while_running_and_after_reset`.

### A5. Completed sessions could be resumed
**Root cause:** no terminal state existed; `start()` merely checked `done` at
spawn time.
**Fix:** `completed` is a real state; resume/step in it return HTTP 409 with
"reset before resuming".
**Test:** `test_completed_session_requires_reset`.

### A6. Exceptions inside the loop died silently
**Root cause:** V1's loop had no exception handler; a raise left `running =
True` forever with no task — UI showed "running" with frozen values.
**Fix:** `error` trap state with the exception recorded in `status.error`,
logged via the structured event log, broadcast to clients.

## B. Intervention bugs

### B1. Manual failure injection was invisible until the next tick
**Symptom:** Fail Link appeared to "not work", especially while paused
(nothing changed on screen at all until Step/Resume).
**Root cause:** V1 intervention endpoints mutated the engine but never
broadcast; the UI only repainted on tick payloads.
**Fix:** every intervention broadcasts an immediate out-of-band
`type: "intervention"` snapshot (no clock advance).
**Tests:** `test_failure_broadcasts_immediately_even_while_paused`,
`test_failure_affects_both_directions_and_no_clock_advance` (also proves both
directed edges go down — the engine keyed failures by physical link ID in V1
already, which was correct).

### B2. Repeated intervention clicks double-applied semantics unclear
**Root cause:** V1 `set_link_state` was already idempotent internally but the
API gave no signal, so the UI couldn't tell "applied" from "already failed",
and double-clicks looked like silent failures.
**Fix:** endpoints return `changed: true/false`; UI shows explicit
confirmation; failing an already-failed link is a reported no-op.
**Test:** `test_failure_idempotent_and_recovery`.

### B3. Interventions raced the tick loop
**Root cause:** V1 applied interventions from the HTTP handler thread-context
while `step_once` ran in a worker thread — engine arrays could be mutated
mid-interval (e.g. FRR while loads were being summed).
**Fix:** interventions acquire the session lock, serializing them with ticks.
**Test:** `test_failure_works_while_running_and_after_reset` (running case).

## C. Data-integrity bugs

### C1. Saved live runs recorded reward = 0 for every step
**Symptom:** `/api/export/save-run` stored `reward_sum = 0`; exported CSVs
disagreed with the on-screen cumulative reward.
**Root cause:** V1's session tracked only a cumulative float; the persistence
path filled per-step reward with a literal `0.0` placeholder.
**Fix:** `AlgoRunner.history` is the single authoritative per-step record
(reward at full precision + components + interval metrics); exports, saved
summaries, the metrics API and the scoreboard all derive from it.
**Test:** `test_exported_rewards_match_cumulative` (sum of exported per-step
rewards == displayed cumulative reward, and ≠ 0).

### C2. Rounded rewards drifted from the cumulative tracker
**Root cause:** decision payloads rounded to 4 decimals and V1 stored those
rounded values; summing them diverged from the full-precision accumulator.
**Fix:** history stores full precision; rounding happens only at display.

### C3. `dropped_gbit_total` was 1000× too small
**Root cause:** Mbps × seconds gives megabits; the code divided by 1000
twice (once inline, once after the sum).
**Fix:** single conversion; unit test with a hand-computable case
(100 Mbps dropped × 2 × 300 s = 60 Gbit). Published V1 CSVs are preserved
unchanged (see CURRENT_SYSTEM_BASELINE.md); the column is correct for all
new runs.
**Test:** `test_dropped_gbit_conversion`.

## D. Correctness bugs in the RL plumbing (model-compatible fixes)

### D1. Observation-dimension formula only worked for k_paths = 4
**Root cause:** `demand_features = 10 + k + 1` happens to equal the true
layout size `7 + 2k` only at k = 4.
**Fix:** corrected formula; parametrized tests for k ∈ {2, 3, 4, 5} incl. the
Gymnasium checker; `ppo_te` shapes (586/69) verified unchanged.
**Tests:** `test_observation_dim_formula_for_any_k_paths`,
`test_pretrained_model_still_compatible`.

### D2. Protected-class bandwidth check double-counted the moving demand
**Root cause:** headroom was computed from raw current loads; when the old
and new paths share links, the demand's own traffic on the shared link was
counted against itself.
**Fix:** `projected_link_loads_after_move()` — remove own traffic from the
current path, add to the candidate, then check headroom; used by validation,
masking, candidate info and explanations.
**Tests:** `test_projected_loads_conserve_volume_and_avoid_double_count`,
`test_protected_check_uses_projected_load` (synthetic double-count case:
raw check rejects, projected check correctly accepts).

### D3. Hardcoded topology facts (17 demands, D1–D17, PE5–PE8, 28/56 comment)
**Root cause:** literals baked in during initial development.
**Fix:** reward normalization uses `interval["n_demands"]`; randomized
scenarios draw bursts from configured demand IDs and flash-crowd targets from
configured egress routers; topology header comment corrected to 32/64.
**Test:** `test_random_day_uses_configured_demands_and_egress`.

### D4. Random baseline didn't match its documentation
**Root cause:** V1 sampled a (demand, path) uniformly over ALL pairs and
acted only if valid — the effective distribution over valid actions was
non-uniform and the effective no-op rate exceeded the documented 50%.
**Fix:** documented rule implemented exactly: 50% no-op, otherwise uniform
over the same validity mask the RL policy sees; chi-square uniformity test.
Published V1 "random" rows are preserved as-is; the corrected controller
applies to new runs.
**Test:** `test_random_baseline_uniform_over_valid_actions`.

### D5. No configuration/model validation
**Root cause:** none existed; a mismatched checkpoint produced an opaque
SB3 shape error.
**Fix:** `mplssim/validation.py` validates all YAML at startup (duplicate
IDs, unknown endpoints/links/demands, malformed SLAs, event times outside
scenarios, reward params) and `check_model_compatibility()` raises a clear
message naming both the expected and configured shapes;
`models/ppo_te/metadata.json` records the training-time shapes and config
hashes.
**Tests:** `test_config_validation_passes_on_shipped_configs`,
`test_model_mismatch_message_is_clear`.

## E. Items audited and found NOT broken

- **Both directions on physical failure** — V1 engine already keyed failures
  by undirected link ID; kept, now covered by a test.
- **Counterfactual isolation** — clones were true deep copies in V1; the new
  advisor lookahead keeps that invariant (`test_advisor_propose_does_not_mutate_engine`).
- **Paired interventions in compare mode** — V1 applied to all runners;
  now lock-guarded and tested (`test_failure_paired_in_compare_mode`).
- **Invalid actions presented as successful** — V1 already returned
  accepted/reason and the UI showed REJECTED badges; unchanged.
- **Silent model fallback** — V1 fell back best→final model file *within the
  requested tag* only (never to another tag or heuristic); retained, but
  model loads are now logged with the file name, and shape-checked.

## F. Structured logging

All lifecycle transitions, interventions, advisor decisions, WS connects,
model loads and loop exceptions are logged through `server/events.py`
(`log_event`) with scenario/algorithm/seed/step/simulated-time context,
mirrored into a 500-entry ring buffer at `GET /api/events`, and shown in the
Advanced UI's Events panel.

## G. Terminology corrections (documentation)

- "SLA violations" is now labeled **demand-interval SLA violations** (a count
  of violating demands summed over control intervals), in the UI and report.
- "Recovery time" is documented as **post-FRR traffic-engineering recovery**
  (first interval with zero SLA violations after a failure): every controller
  receives the same FRR-style emergency repair, so the metric measures how
  fast the controller cleans up afterwards — not raw restoration.
