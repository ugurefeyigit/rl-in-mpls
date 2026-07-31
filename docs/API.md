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
| POST | `/api/simulation/start` | `{scenario, algorithms: ["rl","static"], seed, model_tag, safety_filter, speed, autostart}` | Create (and optionally start) a session; 2 algorithms = paired compare mode |
| POST | `/api/simulation/pause` | – | Pause the ticking loop |
| POST | `/api/simulation/resume` | – | Resume |
| POST | `/api/simulation/step` | – | Advance exactly one control interval (must be paused) |
| POST | `/api/simulation/reset` | – | Rebuild the same session at t=0 |
| POST | `/api/simulation/speed` | `{speed: "1x"\|"5x"\|"20x"\|"fast"}` | Presentation pacing (1x = one 5-min interval / 2 s) |
| GET | `/api/simulation/status` | – | Clock, step, running/done flags |

Valid `algorithms`: `rl`, `static`, `greedy`, `cspf`, `random`.

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
