# Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ frontend/ (static, no build step)                                      │
│   Cytoscape.js topology · ECharts metrics · decision tape · tables     │
│         ▲ REST (fetch)                  ▲ WebSocket /ws/telemetry      │
├─────────┴───────────────────────────────┴──────────────────────────────┤
│ server/ (FastAPI, one process)                                         │
│   main.py    REST + WS endpoints, training-job subprocess control      │
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
