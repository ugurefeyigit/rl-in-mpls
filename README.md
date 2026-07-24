# RL-in-MPLS — Reinforcement-Learning Traffic Engineering on a Simulated MPLS Backbone

An interactive engineering experiment answering one question honestly:

> **Can an RL controller make better MPLS traffic-engineering decisions than
> conventional static or heuristic routing when traffic demand changes over time?**

A flow-level simulation of an 18-router MPLS backbone is driven by either a
trained MaskablePPO agent or conventional controllers (static shortest path,
utilization-aware greedy, CSPF-style periodic reoptimization). All
controllers face byte-identical seeded traffic, so every comparison is
paired. A live NOC-style dashboard shows the topology, LSPs, decisions,
reward breakdowns and metrics in real time.

**This is a simulation study, not a production controller.** Limitations are
listed below and in [docs/REPORT.md](docs/REPORT.md).

---

## Quick start (Windows / macOS / Linux, Python 3.11+)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### One-command demo

```bash
python scripts/demo.py
```

Starts the backend, loads the pretrained agent (`models/ppo_te/`), creates the
fixed demo scenario (evening peak → flash crowd → backbone failure →
recovery; RL vs static side-by-side; seed 42, paused at t=0) and opens the
dashboard. Press **Resume** to run, or **Step** to advance one 5-minute
control interval at a time. The demo seed is fixed for reproducibility — the
multi-seed evidence lives in `results/` (see Evaluation).

### Manual start

```bash
python -m uvicorn server.main:app --port 8000
# dashboard:  http://127.0.0.1:8000
# OpenAPI:    http://127.0.0.1:8000/docs
```

### Docker

```bash
docker compose up --build    # http://127.0.0.1:8000
```

## Training

```bash
python scripts/train.py                    # full run (configs/training.yaml, ~40 min CPU)
python scripts/train.py --timesteps 30000  # quick sanity run
tensorboard --logdir runs                  # curves incl. per-component rewards
```

Checkpoints, best model and eval traces land in `models/<tag>/`.

## Evaluation

```bash
python scripts/evaluate.py                 # 7 scenarios × 5 seeds × 5 algorithms
python scripts/make_figures.py             # PNGs into results/figures/
```

Produces `results/eval_summary.csv` (per episode), `eval_stats.csv`
(mean/std/95% CI), `eval_summary.json` (incl. paired RL-minus-baseline
deltas) and per-step traces for the figure scripts.

## Tests

```bash
python -m pytest tests/ -q      # 32 tests: engine invariants, gym checker, API e2e
```

---

## What you see in the dashboard

| View | Content |
|---|---|
| **Topology** | Cytoscape network; link color = utilization, width = capacity, dashed red = failed; click an LSP row to highlight its path; hover for per-direction load/delay/loss/LSP counts |
| **Decision tape** | Every controller action as a syslog-style stream with reward |
| **Agent** | Selected action, action probabilities, safety-filter verdict with rejection reason, reward component breakdown, post-hoc no-op counterfactual, and an *engineering interpretation* generated from measured telemetry (labeled as such — it is not the policy's internal reasoning) |
| **Metrics** | Time series per algorithm: max/mean utilization, delay, p95, loss, SLA violations, reroutes, fairness; cumulative reward |
| **Matrix** | Live ingress→egress traffic heatmap |
| **LSPs / Links** | Full tables incl. candidate paths, SLA state, per-algorithm load deltas |
| **Training / Runs** | Launch & monitor training jobs; stored run summaries |

Controls: scenario, single-vs-compare mode, algorithm A/B, model tag, seed,
safety filter on/off, speed (1×/5×/20×/fast), pause/step/reset, link
failure/recovery injection, demand bursts, global demand multiplier, CSV/JSON
export.

## Repository layout

```
configs/      topology.yaml · traffic_classes.yaml · scenarios.yaml ·
              reward.yaml · training.yaml · baselines.yaml
mplssim/      simulation core, RL env, baselines, experiment runner
server/       FastAPI app, live session manager, SQLite persistence
frontend/     build-free dashboard (Cytoscape.js + ECharts, vendored)
scripts/      train.py · evaluate.py · make_figures.py · demo.py
tests/        pytest suite (unit + gymnasium checker + API e2e)
docs/         ARCHITECTURE.md · API.md · REPORT.md · DEMO_SCRIPT.md · ADR-001
models/       trained checkpoints (ppo_te = pretrained demo agent)
results/      evaluation CSV/JSON + figures/
```

## The network

18 routers (4 ingress PEs, 4 egress PEs, 8 P cores, 2 aggregation), 32
undirected links (64 directed) with 100–2000 Mbps capacities, 17 demands in 6
traffic classes (voice, video, VPN, best-effort, bulk, critical) following
diurnal profiles with seeded AR(1) noise, bursts, flash crowds and scripted
link failures. Engineered stress points: a hidden shared bottleneck (P5→P8),
a longer-but-better detour region, a redundancy ring, and a 2 Gbps backbone
link (P2–P5) whose failure forces mass rerouting. Details:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Honest limitations (read before presenting)

- **Flow-level abstraction** — no packets, no TCP dynamics; delay/loss are
  documented analytic functions of utilization (`mplssim/sim/models.py`).
- **Instant control plane** — reroutes take effect at the next interval; no
  RSVP-TE/IGP convergence, no label signaling.
- Offered traffic is **routing-independent** (no congestion backoff), which
  favors clean comparisons over TCP realism.
- The RL agent is trained on *this* topology; nothing here demonstrates
  topology generalization.
- One control action per 5-minute interval; sub-interval dynamics are
  averaged.
- The demo scenario/seed was **chosen to be illustrative**; aggregate claims
  rely on the multi-seed evaluation only.
- See docs/REPORT.md for failure cases, including where RL loses.

## Acceptance criteria

The 28-point checklist from the project brief is tracked in
[docs/REPORT.md](docs/REPORT.md#acceptance-checklist).
