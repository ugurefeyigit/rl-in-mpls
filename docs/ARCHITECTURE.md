# Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ frontend/ (static, no build step)                                      │
│   index.html   Cytoscape topology · ECharts · decision tape (live)     │
│   present.html Presentation Mode (live)                                │
│   study.html   V2 sealed evidence record (read-only, no session)       │
│         ▲ REST (fetch)                  ▲ WebSocket /ws/telemetry      │
├─────────┴───────────────────────────────┴──────────────────────────────┤
│ server/ (FastAPI, one process)                                         │
│   main.py    REST + WS endpoints, training-job subprocess control      │
│   evidence_api.py  GET-only /api/v2/* over the frozen V2 evidence      │
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
│   evidence/  read-only access to the CLOSED V2 study:                  │
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

### Why the frontend has no build step
The demo machine has no Node toolchain, and a presentation must not depend on
npm or a CDN. Cytoscape.js + ECharts are vendored under `frontend/vendor/`;
the app is plain ES modules served by FastAPI. Cytoscape.js was chosen for
first-class network-graph styling (per-edge data-driven color/width, overlay
classes for LSP highlighting) without a framework dependency.


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
