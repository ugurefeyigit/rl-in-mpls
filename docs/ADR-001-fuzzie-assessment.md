# ADR-001: Fuzzie.API assessment and backend decision

**Status:** accepted · **Date:** 2026-07-24

## Context

A team member's repository, [Fuzzie.API](https://github.com/ahmetburakkir/Fuzzie.API),
was proposed as a possible starting point for the MPLS-TE RL simulation.

### What Fuzzie.API is

A Python 3.12 / FastAPI **network monitoring & telemetry** backend built with
Clean Architecture / DDD:

- **Domain:** `Device`, `Port`, `Alarm`, `Event`, and an L2/L3 `Packet` model
  (ARP/DHCP/DNS/ICMP/TCP; MAC learning, flooding, NAT actions).
- **Infrastructure:** SQLAlchemy 2.x + Alembic (PostgreSQL documented, SQLite
  committed), plus real-device collectors (SNMP, ICMP ping, SSH, NetFlow).
- **Simulation:** a tick scheduler that mutates device health metrics and
  generates alarms for scenarios such as "Normal Network" and "DDoS Attack".
- **Presentation:** REST routers + a WebSocket live-metrics stream; 65 unit
  tests pass; `requirements.txt` is empty (deps only in the README).

## Decision

**Build a purpose-built project; adopt Fuzzie's proven patterns, not its code.**
The whole system is one Python service: simulation core + Gymnasium env +
Stable-Baselines3 + FastAPI + static frontend.

## Rationale

1. **Domain mismatch.** Fuzzie models *device fault monitoring* (per-device
   CPU/memory/alarms, L2 packet walkthroughs). MPLS-TE needs *flow-level
   traffic engineering*: link capacities, demands, explicit LSPs, bandwidth
   reservation, path computation. No entity, service, or table carries over.
2. **Control inversion.** Fuzzie's background scheduler owns time; an RL
   environment must own time through `env.step()`. Retrofitting Gym semantics
   onto its threading scheduler would produce exactly the fragile architecture
   the project brief warns against.
3. **No two-service split needed.** The "existing backend + Python RL sidecar"
   pattern makes sense when the backend is another language. Fuzzie is already
   Python/FastAPI, so a separate RL microservice would add serialization and
   deployment cost with zero benefit.
4. **Name/scope conflict.** The repo is a monitoring dashboard ("Fuzzie") with
   DDoS/alarm features; presenting MPLS-TE work inside it would confuse the
   audience and the git history.

## What was reused (as patterns)

- FastAPI + WebSocket connection-manager publish pattern (`server/main.py::WsHub`).
- Scenario-manager concept (`configs/scenarios.yaml` + `ScenarioSpec`).
- Tick-based engine lifecycle: start/pause/step/reset (`server/session.py`).
- Layered separation: domain (`mplssim/core`) / application (`mplssim/sim`,
  `mplssim/rl`, `mplssim/baselines`) / presentation (`server/`).

## What was consciously dropped

- SQLAlchemy/Alembic/PostgreSQL → stdlib SQLite (`server/db.py`). A laptop demo
  must not require a database server; run summaries are small JSON blobs.
- The DDD entity ceremony → frozen dataclasses + numpy state arrays, because
  the inner loop must run at hundreds of steps per second for RL training.
- Real-device collectors (SNMP/SSH) — out of scope for a simulation, kept in
  mind for the deployment discussion in docs/REPORT.md.

## Consequences

- One `pip install`, one process, one command to demo.
- Fuzzie.API remains untouched and usable by its author; nothing here forks it.
- If the team later wants Fuzzie as an umbrella dashboard, this service's REST
  + WS API is small and stable enough to be proxied by it.
