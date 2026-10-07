# Action-mask validation

The closed study reported zero invalid actions, mask disagreements and safety
failures. Those counters are fail-fast (`AuditedV2Env` aborts the run on the
first violation), so "zero" only certifies that runs completed. This audit
checks the mask independently, on states that random legal play generates
(no-op with probability 0.5, otherwise a uniform legal move), including
randomized training days with link failures and disconnections.

Script: `scripts/study/run_mask_audit.py` (module `mplssim/study/maskaudit.py`).
Raw output: `experiments/raw/mask_audit/summary.json` (pass 2; pass 1 with the
engine-side checks only is kept as `summary_pass1_engine_checks.json`).
Episodes: the 7 evaluation scenarios × seeds 5000–5005, and `random_day` ×
seeds 6000–6039. Runtime: 15 min (pass 2, CPU).

## Sample

| Quantity | Count |
|---|---:|
| States audited (= transitions) | 15,480 |
| State–action legality checks | 1,068,120 |
| Legal TE moves applied to a cloned engine | 711,848 |
| …of which moves of protected (voice/critical) demands | 155,408 |
| Accepted moves along the trajectories | 7,832 |
| FRR reroutes along the trajectories | 146 |
| States with at least one failed link | 804 |
| States with at least one disconnected demand | 144 |
| States at which MaskablePPO's masked distribution was checked | 15,480 |

## Checks (all must be zero)

| Check | What it verifies | Violations |
|---|---|---:|
| no-op illegal | action 0 is always legal, so no state has an empty legal set | 0 |
| decode mismatch | action `1 + 4d + p` addresses an existing (demand, candidate) | 0 |
| mask ≠ validator | vectorized mask equals the scalar validator for all 69 actions | 0 |
| masked action accepted | every masked action is rejected when applied to a clone | 0 |
| masked action side effect | a rejected action changes no route, dwell, path age, previous-TE slot or counter | 0 |
| legal action rejected | every legal action is accepted when applied | 0 |
| legal action wrong path | after acceptance the demand is on the requested candidate | 0 |
| legal action without dwell | acceptance starts the 3-interval hold-down | 0 |
| legal over failed link | no legal move routes over a failed link | 0 |
| protected projection (engine) | protected moves have projected gross bottleneck ≤ 100 % | 0 |
| protected projection (independent) | same, recomputed from scratch by summing every connected demand's full offered rate over its post-move path | 0 |
| projection mismatch | independent and engine projections agree to 1e-9 | 0 |
| mask not idempotent | computing the mask twice gives the same result (no hidden state) | 0 |
| PPO illegal probability mass | an (untrained) MaskablePPO policy's masked distribution assigns exactly 0 probability to illegal actions | 0 |

**Result: 0 violations of 14 checks over 1,068,120 state–action pairs.**

## What this does and does not establish

* It establishes that the mask is internally consistent with the engine's
  validator, that both agree with an independent recomputation of the only
  non-trivial legality rule (protected-class projected utilization), that
  illegal actions are side-effect free, and that the policy-gradient learner
  cannot sample an illegal action.
* It does not establish that the *rules themselves* are the right operational
  constraints (e.g. whether 100 % projected gross utilization is the right
  admission threshold for protected traffic); those are design choices of the
  frozen environment.
* On average 46.0 TE moves are legal per state (711,848 / 15,480), i.e. 47 of
  69 actions including no-op; about a third of the action space is masked.
