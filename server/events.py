"""Structured backend event logging with an in-memory ring buffer.

Every operationally relevant action (session lifecycle, interventions,
advisor decisions, websocket connects, model loads, exceptions) is logged
through `log_event`, which both writes a structured line to the standard
logger and appends a JSON-friendly record to a ring buffer served at
GET /api/events and shown in the Advanced UI's event panel.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger("mplssim.server")

_RING: deque[dict[str, Any]] = deque(maxlen=500)


def log_event(event: str, *, level: int = logging.INFO,
              scenario: str | None = None, algorithm: str | None = None,
              seed: int | None = None, step: int | None = None,
              t_min: float | None = None, **extra: Any) -> dict[str, Any]:
    record = {
        "ts": time.strftime("%H:%M:%S"),
        "event": event,
        "scenario": scenario,
        "algorithm": algorithm,
        "seed": seed,
        "step": step,
        "t_min": t_min,
        **{k: v for k, v in extra.items() if v is not None},
    }
    _RING.append(record)
    logger.log(level, "%s | scenario=%s algo=%s seed=%s step=%s t=%s %s",
               event, scenario, algorithm, seed, step, t_min,
               " ".join(f"{k}={v}" for k, v in extra.items() if v is not None))
    return record


def recent_events(limit: int = 100) -> list[dict[str, Any]]:
    return list(_RING)[-limit:]


def clear_events() -> None:
    _RING.clear()
