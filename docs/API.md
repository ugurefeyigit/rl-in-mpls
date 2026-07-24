# API reference

Interactive OpenAPI docs are always available at `http://127.0.0.1:8000/docs`.
All bodies are JSON. Errors return `{"detail": "..."}` with 4xx status.

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
