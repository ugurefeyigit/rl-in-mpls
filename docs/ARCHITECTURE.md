# Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ frontend/ (static ES modules, no build step)                           │
│   app.html     shared shell · fixed SVG topology · four modes          │
│   js/product/ typed source adapters · store guards · mode renderers    │
│         ▲ REST (fetch)                  ▲ WebSocket /ws/telemetry      │
├─────────┴───────────────────────────────┴──────────────────────────────┤
│ server/ (FastAPI, one process)                                         │
│   main.py    REST + WS endpoints, training-job subprocess control      │
│   evidence_api.py  GET-only /api/v2/* over the frozen V2 evidence      │
│   product_api.py   additive typed product snapshot/decision endpoints  │
│   session.py SimSession → AlgoRunner(s): paced stepping, decisions,    │
│              explanations, counterfactuals, paired compare mode        │
│   db.py      SQLite run summaries                                      │
├────────────────────────────────────────────────────────────────────────┤
│ mplssim/ (importable library — server-independent)                     │
│   core/      Router, LinkDef, DirectedLink, TrafficClass, Demand,      │
│              Topology loader (configs/topology.yaml)                   │
│   paths/     Yen k-shortest candidate paths with hop cap               │
│   traffic/   diurnal profiles, AR(1) noise, scenario events, seeding   │
│   sim/       SimulationEngine (flow-level state machine),              │
│              delay/loss models (documented approximations)             │
│   baselines/ static SP · greedy utilization · CSPF reopt · random      │
│   rl/        MplsTeEnv (Gymnasium, masked Discrete actions), reward    │
│   experiments/ paired episode runner + summary metrics                 │
│   evidence/  read-only access to the CLOSED V2 study                   │
│   product/   display metadata, source contracts, schemas, serializers  │
│              identity.py frozen study constants (no I/O)               │
│              loader.py   fail-closed schema/identity/integrity checks  │
│              claims.py   every scientific calculation, one place       │
│              replay.py   recorded step traces, never an evaluation     │
├────────────────────────────────────────────────────────────────────────┤
│ scripts/   train.py · evaluate.py · make_figures.py · demo.py          │
│ configs/   topology · traffic_classes · scenarios · reward ·           │
│            training · baselines   (all YAML, human-editable)           │
│ tests/     unit + gym-checker + API e2e (pytest)                       │
└────────────────────────────────────────────────────────────────────────┘
```

## Key design decisions

### Flow-level simulation (not packet-level)
Offered demand volumes are placed onto explicit LSP paths; per-directed-link
utilization drives analytic delay/queue/loss approximations
(`mplssim/sim/models.py`). This is the standard abstraction in TE research;
packet-level fidelity is explicitly NOT claimed (see REPORT.md, Limitations).

### Exogenous traffic ⇒ paired comparisons
Offered traffic depends only on `(scenario, seed, time)`, never on routing.
Two algorithms run on the same `(scenario, seed)` face identical inputs, so
metric differences are caused by routing decisions alone. Comparison mode in
the UI and `scripts/evaluate.py` both rely on this.

### MPLS abstraction
Each demand is a FEC bound to exactly one active LSP, chosen from k=4
precomputed loop-free candidate paths (Yen's algorithm over admin weights,
hop-capped). Ingress = PE_IN (LER), transit = P/AGG (LSR), egress = PE_OUT.
Failures trigger local repair (FRR-style: best surviving candidate);
controllers then reoptimize at control intervals. Label operations below the
path level (label stacks, PHP) are not modeled.

### Control loop
One control interval = 5 simulated minutes = 5 × 1-minute micro-ticks.
The controller (RL policy or baseline) may move at most one demand per
interval (CSPF: batched every 6 intervals). A per-demand cooldown (3
intervals) and flap detection discourage oscillation.

### RL formulation
- Observation: 586-dim normalized vector (5×64 link features + 15×17 demand
  features + 11 globals) — exact layout documented in `mplssim/rl/env.py`.
- Actions: `Discrete(69)` = no-op + (demand × candidate path), with
  invalid-action masking (failed links, cooldowns, same-path,
  bandwidth-infeasible moves for protected classes).
- Reward: weighted sum of normalized components (configs/reward.yaml), every
  component logged separately (TensorBoard + UI).
- Algorithm: MaskablePPO (sb3-contrib). Masking is required because the valid
  action set changes with failures/cooldowns; PPO is the stable default for
  this scale. DQN would need action-space surgery to respect masks natively.

### Safety filter
`SimulationEngine.validate_action` doubles as the safe-RL constraint checker:
the UI shows proposed vs accepted actions and rejection reasons. Evaluation
can disable it (`--no-safety`) for the experimental-mode comparison.

### Shared shell and typed sources
`frontend/app.html` serves `/`, `/advanced`, `/present`, `/study`, and `/compare`.
Client routing selects one of four modes without creating separate products. The
store guards live snapshots by session generation and monotonic sequence, and
guards asynchronous reads by source revision so a late LIVE response cannot
populate a RECORDED or evidence surface.

Exp 2.1 extends the existing process-memory results boundary with exactly two
completed-run slots. A/B records are normalized copies of recorded interval
history, never model objects and never evidence. Full Reset clears both slots;
server restart drops them by construction. The `/compare` renderer uses authored
SVG plus table twins and deep-links only the aggregate interval fields that were
actually retained.

Four adapters preserve the data boundary: live V1 may execute and render link
telemetry; recorded V2 is immutable and has no per-link utilization;
development evidence is selection-stage only; final evidence is the frozen
one-shot holdout. Source changes clear incompatible snapshots, timelines,
decisions, comparisons, recommendations, selections, and story state.

The SVG atlas reads a display-only map built from the existing schematic
coordinates. It never writes to `configs/topology.yaml`, moves a node during a
session, or feeds display coordinates back to the simulator.

### Why the frontend has no build step
The demo must work offline without npm or a CDN. The application is plain local
HTML, CSS, and ES modules served by FastAPI; legacy vendor assets remain for
compatibility surfaces but the unified topology has no runtime framework.

The unified product reads a live interval through
`GET /api/simulation/moment`. The endpoint holds the session lock while it
serializes the snapshot, decision, timeline, comparison, and recommendation, so
the interface cannot combine fields from different simulation intervals.


## The read-only evidence path

The governed V2 study is closed. Its artifacts under `results/v2_*` are immutable
inputs, and one component reads them:

```
results/v2_final_holdout/*.csv,*.json ─┐
results/v2_three_root_continuity/*     ├─► mplssim/evidence/loader.py
results/v2_seed42/*                    ┘        │  validates schema, study
                                                │  identity and integrity;
                                                │  raises rather than degrades
                                                ▼
                                        mplssim/evidence/claims.py
                                                │  root-aware aggregation,
                                                │  scenario comparison,
                                                │  reward reconciliation
                                                ▼
                                        server/evidence_api.py  (GET only)
                                                ▼
                                        frontend/study.html
```

Three properties hold by construction and are covered by tests:

1. **Read-only.** Nothing on this path opens a file under `results/` or `runs/`
   for writing, imports a learner, constructs an environment, or loads a
   checkpoint. `tests/test_evidence_loader.py` and `tests/test_evidence_api.py`
   assert the absence of writes to governed paths.
2. **Fail closed.** A missing file, a missing column, a wrong episode count, a
   foreign source SHA, an unexpected seed set or a failed integrity flag raises.
   The API turns that into a 503 with a named cause rather than serving zeros.
3. **One place for arithmetic.** The API and the frontend do no scientific
   calculation. Aggregation is root-aware — the aggregate is the unweighted mean
   of the three training-root means, never a pool of the 105 episodes — and
   development and final-holdout evidence are separate types with no operation
   that combines them.

Recorded replay reads the preserved step traces named in the holdout manifest.
Their location is configurable via `V2_FULL_ARTIFACTS`; when unset, the catalogue
still lists every episode and reports it unavailable.


## Live V2 (Part 1)

`MplsTeEnvV2` is the live default. One session runs one environment version;
mixing versions in a paired comparison is refused, because the same action
number addresses a different candidate path in each.

- `mplssim/product/checkpoints_v2.py` — the immutable six-checkpoint registry
  (pre-holdout continuity selection), its SHA-256 payload and sidecar hashes,
  and the fail-closed loader. Verification is artifact presence → payload and
  sidecar hashes → sidecar declaring V2, the expected root and transition →
  the stored environment identity validating against the live environment. Any
  failure raises `CheckpointUnavailable`; V1 is never substituted.
- `mplssim/product/live_v2.py` — `EngineV2View`, a read-only product-shaped view
  over the frozen `SimulationEngineV2`. It translates only the names the product
  layer reads under a V1 spelling (offered traffic, TE dwell, the route-change
  log) and refuses to emulate what V2 does not have, such as a manual traffic
  multiplier. `mplssim/sim/engine_v2.py` is not edited.
- `server/session.py` — `AlgoRunnerV2`. Every V2 controller, learner or baseline,
  drives a real `MplsTeEnvV2`; a baseline only supplies the action integer, so
  the mask, the validator reason and the twelve reward components come from the
  governed environment in every lane.

The default training root is 42: the study's primary seed-42 scientific root,
first in the registered root order. It is chosen by fixed identity, never from
final-holdout performance.

## Results, comparison and record classes

Three product modules turn recorded state into surfaces, and each is a *read*
over state that already exists.

- `mplssim/product/pairing.py` answers **may we compare** — it fingerprints the
  exogenous inputs of both lanes and either proves they share one experiment or
  names the fields that broke the proof.
- `mplssim/product/comparison.py` answers **what the comparison shows**, and
  only when pairing said yes. On a broken proof it emits no verdict, no metric
  row and no gap; the refusal *is* the payload. It never divides one signed
  operational return by another, and it keeps controller TE changes, FRR
  protection moves and post-recovery restorations in three separate counters.
- `mplssim/product/results.py` reports three record classes —
  `live_demonstration`, `retained_demonstration` and `governed_evidence` — in
  three sections that share no table and no aggregate. It deliberately does
  **not** load the study's numbers: it emits a pointer to `/api/v2/*`, so the
  frozen record has exactly one renderer and cannot drift into a second copy.
- `mplssim/product/run_summary.py` summarizes a saved run for its own
  environment. V1 and V2 record different interval columns, so a V2 row declares
  what it cannot measure rather than padding V1's columns with zeros.

Retention is process-scoped and never written to disk: reset run archives to the
session, full reset hands the archive to the process, a restart drops
everything. The reasoning is in
[ADR-003](ADR-003-results-retention-and-delegated-fast-forward.md), along with
the decision that a fast-forward under advisor execution must be explicitly
delegated and recorded as one batch in the approval ledger.
