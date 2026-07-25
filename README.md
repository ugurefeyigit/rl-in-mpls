# RL-in-MPLS — Reinforcement-Learning Traffic Engineering on a Simulated MPLS Backbone

> **Author:** Uğur Efe Yiğit · **License:** Proprietary — all rights reserved.
> This repository is publicly readable but **not** open source. No permission is
> granted to use, copy, modify or redistribute it without written permission.
> See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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

Starts the backend **with the training endpoint disabled**, loads the
pretrained agent (`models/ppo_te/`), creates the fixed demo session (evening
peak → flash crowd → backbone failure → recovery; RL vs greedy side-by-side;
seed 42, advisor mode, paused at t=0) and opens **Presentation Mode**. The demo
seed is fixed for reproducibility — the multi-seed evidence lives in `results/`
(see Evaluation).

Open the engineering console instead:

```bash
python scripts/demo.py --advanced
```

Re-enable training (never do this during a presentation):

```bash
python scripts/demo.py --allow-training
```

### Manual start

```bash
python -m uvicorn server.main:app --port 8000
```

| URL | What it is |
|---|---|
| `http://127.0.0.1:8000/present` | Presentation Mode — storytelling UI for a live audience |
| `http://127.0.0.1:8000/` or `/advanced` | Engineering console — full telemetry |
| `http://127.0.0.1:8000/docs` | OpenAPI, always current |

To serve manually with training disabled, as the demo launcher does:

```bash
ALLOW_TRAINING=false python -m uvicorn server.main:app --port 8000
```

On PowerShell:

```bash
$env:ALLOW_TRAINING="false"; python -m uvicorn server.main:app --port 8000
```

### Docker

```bash
docker compose up --build    # http://127.0.0.1:8000
```

The compose service sets `ALLOW_TRAINING=false`; override it in
`docker-compose.yml` if you intend to train inside the container.

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
python -m pytest tests/ -q
```

78 tests: engine invariants, gymnasium checker, API end-to-end, session state
machine, correctness fixes, and the presentation contract (page smoke, element
IDs, no external assets, display-scale agreement between Python and JS,
websocket reconnect, no-training-on-launch, benchmark honesty).

Manual UI checks that the suite cannot cover are in
[docs/UI_ACCEPTANCE_TESTS.md](docs/UI_ACCEPTANCE_TESTS.md).

---

## Presentation Mode

A second frontend at **`/present`**, for showing this to people who do not know
what MPLS or RL is. It is a separate UI, not a reskin: city names instead of
router IDs, five large KPI cards, a story timeline, an operator recommendation
card with Approve/Reject, and a guided five-minute story driven entirely by
real backend state.

| Doc | Contents |
|---|---|
| [docs/PRESENTATION_MODE.md](docs/PRESENTATION_MODE.md) | What it shows, how to launch, keyboard shortcuts, scaling disclosure |
| [docs/PRESENTATION_SCRIPT.md](docs/PRESENTATION_SCRIPT.md) | The 20–25 minute script: what to say, what to click, where RL loses |
| [docs/OPERATOR_ADVISOR.md](docs/OPERATOR_ADVISOR.md) | propose / approve / reject, predicted vs actual, why proposals never mutate the engine |
| [docs/CITY_DISPLAY_MAPPING.md](docs/CITY_DISPLAY_MAPPING.md) | The router → city table and the rule that internal IDs never change |

**Display names are a presentation layer only.** `PE1`, `L11`, `D2` and the
scenario keys are the contract shared by the pretrained model, the configs, the
tests and the committed results; they are never renamed. The mapping lives in
one place, `mplssim/display.py`, and reaches both frontends via
`GET /api/display`.

**Scaled national view.** Presentation Mode has an optional 10× display scale.
It multiplies traffic volumes and link capacities only — utilization, delay,
loss, SLA counts, actions and rewards are unchanged — and it keeps a banner on
screen saying exactly that while it is on.

**Terminology: "demand-interval SLA violations".** The SLA counters are not a
count of unhappy services. They count *one per traffic demand per five-minute
interval* that missed its latency or loss target, so one demand suffering for a
whole evening contributes many. Both UIs label the metric this way.

---

## What you see in the engineering console

| View | Content |
|---|---|
| **Topology** | Cytoscape network with city labels; link color = utilization, width = capacity, dashed red = failed; click a demand row to highlight its route; hover for per-direction load/delay/loss/LSP counts and the internal IDs |
| **Decision tape** | Every controller action as a plain-language stream with reward; internal encoding (`D2 p0→p3`) in a dimmed technical column |
| **Scoreboard** | Live per-controller totals — reward, mean/interval, busiest link, peak, SLA problems now, demand-interval SLA total, delivered ratio, route changes, flaps — plus the absolute reward-point delta and a high-churn warning |
| **Decision** | Selected action, action probabilities, safety-filter verdict with rejection reason, reward component breakdown, post-hoc no-op counterfactual, and an *engineering interpretation* generated from measured telemetry (labeled as such — it is not the policy's internal reasoning) |
| **Metrics** | Time series per controller: max/mean utilization, delay, p95, loss, demand-interval SLA violations, route changes, fairness; cumulative reward |
| **Benchmark** | Published 5-seed results for the selected scenario, read live from `results/eval_stats.csv`, plus the cross-scenario winner table |
| **Matrix** | Live source→destination traffic heatmap |
| **Demands / Links** | Full tables incl. candidate paths, SLA state, per-controller load deltas |
| **Events** | Structured backend event log from `GET /api/events` |
| **Training / Runs** | Launch & monitor training jobs (confirmation required, disabled in demo mode); stored run summaries |

Controls: scenario, single-vs-compare mode, controller A/B, model tag, seed,
safety filter, operator advisor, speed (1×/5×/20×/fast), pause/step/reset,
Recommend/Approve/Reject, link failure/recovery injection, demand bursts,
global demand multiplier, CSV/JSON export.

A state chip in the header tracks `idle / running / paused / completed / error`
and every control that is invalid in the current state is disabled rather than
allowed to fail. Interventions report back from the server's `changed` flag, so
"already failed" is never presented as a fresh failure.

## Repository layout

```
configs/      topology.yaml · traffic_classes.yaml · scenarios.yaml ·
              reward.yaml · training.yaml · baselines.yaml
mplssim/      simulation core, RL env, baselines, experiment runner,
              display.py (the single city/scenario/label registry)
server/       FastAPI app, live session manager, event log, SQLite persistence
frontend/     build-free UIs (Cytoscape.js + ECharts, vendored — no CDN, no npm)
              index.html + js/app.js        engineering console
              present.html + js/present.js  Presentation Mode
              js/display.js · js/fmt.js     shared labels and number formatting
scripts/      train.py · evaluate.py · make_figures.py · demo.py
tests/        pytest suite (unit + gymnasium checker + API e2e + presentation)
docs/         ARCHITECTURE.md · API.md · REPORT.md · DEMO_SCRIPT.md · ADR-001
              PRESENTATION_MODE.md · PRESENTATION_SCRIPT.md ·
              OPERATOR_ADVISOR.md · CITY_DISPLAY_MAPPING.md ·
              UI_ACCEPTANCE_TESTS.md · DEBUG_AUDIT.md
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

## Configuration guide

Everything tunable lives in `configs/` as commented YAML; no source edits needed:

| File | What you change there |
|---|---|
| `topology.yaml` | Routers (id/role/position), links (capacity, delay, admin weight). Adding/removing links automatically reshapes candidate paths, observations and the UI |
| `traffic_classes.yaml` | Class SLAs/priorities, diurnal profiles (hour, multiplier control points), the demand matrix (src/dst/class/base Mbps) |
| `scenarios.yaml` | Scenario windows, demand multipliers, scripted events (`link_down`, `link_up`, `burst`, `flash_crowd`, `multiplier`), randomization ranges for training |
| `reward.yaml` | All reward weights and normalization params — the formula is documented in the file header |
| `training.yaml` | PPO hyperparameters, timesteps, seeds, control-interval/k-paths/cooldown (`env:` block feeds both training and the live server) |
| `baselines.yaml` | Greedy trigger/margin/cooldown, CSPF period/headroom/hysteresis |

Changing `env:` values (e.g. `k_paths`) changes the observation/action shapes —
retrain before loading a model trained under different shapes.

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
- **RL does not win everywhere, and the losses are not small.** On the
  published 5-seed evaluation it wins the Normal National Traffic Day
  (153.8 vs greedy 149.9) and the Hidden Shared Bottleneck (72.9 vs 70.0), and
  loses the reactive incidents clearly — Major Live Event Traffic Surge
  (−94.6 vs greedy −79.2) and Ankara–Kayseri Backbone Failure (−76.9 vs −46.1).
- **Route churn is the agent's real weakness.** On a normal day it makes 288
  route changes to greedy's 70, of which 264 are flaps — traffic moved back to
  a path it just left. The reward function does not charge enough for this, and
  no operations team would accept it. Both UIs surface a churn warning rather
  than hiding it.
- See docs/REPORT.md for failure cases, including where RL loses.

## Screenshots

None are committed — the UIs are live views and a stale PNG would misrepresent
them. To capture the current state, run `python scripts/demo.py`, play through
the guided story, and use **Print / Save as PDF** in the Presentation Mode
header: it renders a summary card with the run's real scores, totals, story
timeline and the fictional-topology disclaimer.

## Acceptance criteria

The 28-point checklist from the project brief is tracked in
[docs/REPORT.md](docs/REPORT.md#acceptance-checklist).
