# API reference

Interactive OpenAPI docs are always available at `http://127.0.0.1:8000/docs`.
All bodies are JSON. Errors return `{"detail": "..."}` with 4xx status.

## V2 study evidence (read-only)

The governed V2 study is closed. These routes read the committed compact
artifacts under `results/v2_*` and the preserved step traces. Every one is a
**GET**; there is deliberately no route that can train, tune, evaluate a
checkpoint, reselect one, sweep, or reopen the final holdout.

Every payload carries `stage` — `final_holdout` or `development` — plus the
source SHA and the artifact path it was read from. Final-holdout and development
evidence never arrive from the same route, so no client can average them.

| Method | Path | Description |
|---|---|---|
| GET | `/api/v2/study` | Status, identity, seeds, scenarios, and the frozen conclusions |
| GET | `/api/v2/final-holdout` | One-shot result: episode accounting, learner comparison, five-method aggregate, safety, churn, runtime, both no-op grains |
| GET | `/api/v2/final-holdout/scenarios` | Seven-scenario comparison, root-averaged, with baselines |
| GET | `/api/v2/final-holdout/reward-components` | 12-component breakdown and the exact-sum residuals |
| GET | `/api/v2/final-holdout/actions` | All 621 action rows plus no-op share at both grains |
| GET | `/api/v2/final-holdout/integrity` | Safety counters and integrity status |
| GET | `/api/v2/final-holdout/provenance` | Six checkpoints, hashes, source bindings, runtime |
| GET | `/api/v2/development/continuity` | Three-root continuity results and learning curves |
| GET | `/api/v2/development/seed42` | Seed-42 pilot results and checkpoint curve |
| GET | `/api/v2/disclosures` | Invalidated, superseded, failed and repaired runs |
| GET | `/api/v2/replay/index` | Catalogue of all 315 recorded episodes, with per-episode availability |
| GET | `/api/v2/replay/episode` | `?policy_id=&scenario=&seed=` — one recorded episode's provenance and step sequence |

**Errors.** Unlike the live-session routes, these return a structured detail:
`{"detail": {"error": "IntegrityError", "message": "..."}}`. An unreadable or
inconsistent artifact is a **503**, never a page of zeros. An out-of-study
request — a non-holdout seed, an unknown scenario or policy — is a **400**.

**Replay availability.** The recorded step traces are large and live outside Git.
Set `V2_FULL_ARTIFACTS` to the directory named in
`results/v2_final_holdout/manifest.json` under `full_artifact_path`. Without it,
`/api/v2/replay/index` still lists all 315 episodes with `available: false` and
`/api/v2/replay/episode` returns 503 with the configuration hint.

Replay is a tape player. Every payload is marked `kind: "recorded_replay"` and
`live: false`, and the `/study` page refuses to render anything that is not.

## Unified product API

These additive endpoints feed the three-mode shell. They preserve all existing
API paths. Every live payload declares provenance; routes fail closed when no
session exists. The only POST below is a clone-only estimate that refuses stale
generation/step fingerprints and never advances the running engine.

| Method | Path | Description |
|---|---|---|
| GET | `/api/product/capabilities` | Installed policies, sources, checkpoints, and truthful unavailable reasons |
| GET | `/api/product/contracts` | Three modes, nested workflows, routes, source permissions, output semantics, no-op grains |
| GET | `/api/product/display-map` | Fixed display-only engineering schematic, city/role labels, capacity and utilization classes |
| GET | `/api/rl/schema?environment=v1\|v2` | Observation groups, all 69 actions, and authoritative reward-component schema |
| GET | `/api/simulation/snapshot` | Typed live topology, demand, metric, incident, session, and provenance snapshot |
| GET | `/api/simulation/decision` | Observation → mask → output → action → safety → transition → reward pipeline |
| GET | `/api/simulation/timeline` | Typed live incident, action, FRR, recovery, and stabilization events |
| GET | `/api/simulation/comparison` | Paired-lane state plus synchronization proof; no verdict when proof fails |
| GET | `/api/simulation/object/{kind}/{id}` | Focused router, link, demand, or path details |
| POST | `/api/simulation/counterfactual` | One-interval simulated estimate on a clone; stale requests fail closed |

PPO outputs are named action probabilities only when exposed. Masked-bandit
outputs retain their actual action-score or immediate-reward-estimate names.
Changed observations are descriptive deltas, never causal importance.

## Static information

| Method | Path | Description |
|---|---|---|
| GET | `/api/topology` | Routers (id, role, x, y) and undirected links (capacity, delay, weight) |
| GET | `/api/scenarios` | Scenario names, descriptions, durations, scripted events |
| GET | `/api/traffic-classes` | Class SLA definitions and the demand matrix |
| GET | `/api/checkpoints` | Trained model files under `models/<tag>/` |

## Session control

| Method | Path | Body | Description |
|---|---|---|---|
| POST | `/api/simulation/start` | `{scenario, environment, algorithms, seed, training_root, model_tag, safety_filter, speed, autostart, execution}` | Create (and optionally start) a session; 2 algorithms = paired compare mode |
| POST | `/api/simulation/pause` | – | Pause the ticking loop |
| POST | `/api/simulation/resume` | – | Resume |
| POST | `/api/simulation/step` | – | Advance exactly one control interval (must be paused) |
| POST | `/api/simulation/run-until` | `{condition, max_steps, util_threshold}` | Paused fast-forward to `next_event`, `congestion`, real `failure`, real `recovery`, or `end`; unknown conditions fail closed |
| POST | `/api/simulation/reset` | – | **Reset run**: rebuild the same experiment at t=0 and retain the run it replaced |
| POST | `/api/simulation/stop` | – | **Full reset**: stop the runners and clear the active session |
| GET | `/api/simulation/retained-runs` | – | Runs archived by reset run, summarized |
| POST | `/api/simulation/speed` | `{speed: "1x"\|"5x"\|"20x"\|"fast"}` | Presentation pacing (1x = one 5-min interval / 2 s) |
| GET | `/api/simulation/status` | – | Clock, step, running/done flags |
| GET | `/api/simulation/moment?algorithm=rl` | – | Atomic product read of snapshot, decision, timeline, comparison, and advisor state under the session lock |

`environment` defaults to **`v2`**, the governed study environment. Valid
`algorithms` depend on it:

| `environment` | Valid `algorithms` | Checkpoint selector |
|---|---|---|
| `v2` (default) | `masked_bandit`, `maskable_ppo`, `greedy`, `cspf`, `static` | `training_root` ∈ {42, 314159, 271828}, default 42 |
| `v1` | `rl`, `static`, `greedy`, `cspf`, `random` | `model_tag`, default `ppo_te` |

`rl` is V1's generic controller slot and `ppo_te` is the V1 checkpoint tag it
loads; they are not two different controllers. V2's learners are named for what
they are. A V1 controller requested in V2 (or the reverse) is a `400`; V2 is
never silently substituted for V1 or the reverse.

A V2 learner whose frozen checkpoint cannot be verified returns `409` with the
verification reason — missing artifact root, payload or sidecar SHA-256
mismatch, a sidecar that does not declare V2 and the expected training root, or
a stored environment identity that does not validate against the live
environment. No fallback is ever attempted.

`execution` is `automatic` (the policy acts; each completed decision is
explained) or `advisor` (each proposed action is held until `approve` or
`reject`). In advisor execution `POST /api/simulation/step` produces a
*proposal* rather than advancing the clock, and `/api/simulation/run-until`
returns `approval_bypassed: true` because a fast-forward is one delegated
gesture rather than many individual approvals.

`/api/traffic/burst` and `/api/traffic/multiplier` return `409` in V2: the
frozen V2 engine has no manual traffic override and none is fabricated.

Frozen final-holdout seeds `1001–1005` are rejected for live sessions before
the current session is changed. They remain available only through the
read-only governed evidence APIs.

## Interventions (applied to every engine in the session — comparisons stay paired)

| Method | Path | Body |
|---|---|---|
| POST | `/api/failure/inject` | `{link: "L20"}` |
| POST | `/api/failure/recover` | `{link: "L20"}` |
| POST | `/api/traffic/burst` | `{demand: "D5", factor: 2.0, duration_min: 60}` |
| POST | `/api/traffic/multiplier` | `{factor: 1.25}` (global demand scale) |

## Telemetry & analysis

| Method | Path | Description |
|---|---|---|
| GET | `/api/telemetry/current` | Full snapshot payload (same shape as a WS tick) |
| GET | `/api/metrics/history` | Per-interval metric history per algorithm |
| GET | `/api/lsps` | LSP table incl. candidate paths, SLA state, path changes |
| GET | `/api/links` | Directed-link table (both algorithms in compare mode) |
| GET | `/api/agent/status` | Last decision + cumulative reward per algorithm |

## Export & persistence

| Method | Path | Description |
|---|---|---|
| GET | `/api/export/results?fmt=csv\|json` | Download the session's step metrics |
| POST | `/api/export/save-run` | Persist run summaries to SQLite (`results/runs.db`) |
| GET | `/api/runs` | Stored run summaries |

## Training

| Method | Path | Body / Description |
|---|---|---|
| POST | `/api/agent/train` | `{timesteps, tag, seed}` — launches `scripts/train.py` as a subprocess |
| GET | `/api/training/progress` | `{active, log: [...], exit_code}` |

## WebSocket `ws://…/ws/telemetry`

Server → client messages:

```jsonc
{
  "type": "tick",
  "status": {"scenario": "...", "step": 12, "hour": 18.0, "running": true, ...},
  "runs": [
    {
      "algorithm": "rl",
      "snapshot": {            // full engine state
        "routers": [...], "links": [...], "demands": [...],
        "failed_links": [...], "metrics": {...}, "recent_actions": [...]
      },
      "decision": {            // controller decision for this interval
        "action": 23, "decoded": {...}, "reward": -0.41,
        "components": {"delivered": 1.0, "loss": -0.2, ...},
        "action_probability": 0.61, "top_actions": [...],
        "counterfactual": {...}, "explanation": "Rerouted bulk demand D5 ..."
      }
    }
  ]
}
```

`{"type": "status"}` is sent when a scenario finishes. The client may send any
text as a keepalive; it is ignored.
