"""SQLite persistence for run summaries and exports (stdlib sqlite3, no ORM)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "results" / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,           -- 'live' | 'evaluation'
    scenario TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    seed INTEGER NOT NULL,
    summary_json TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def save_run(kind: str, scenario: str, algorithm: str, seed: int,
             summary: dict[str, Any]) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (created_at, kind, scenario, algorithm, seed, summary_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), kind, scenario, algorithm, seed,
             json.dumps(summary)),
        )
        return int(cur.lastrowid)


def list_runs(limit: int = 100) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, created_at, kind, scenario, algorithm, seed, summary_json"
            " FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [
        {"id": r[0], "created_at": r[1], "kind": r[2], "scenario": r[3],
         "algorithm": r[4], "seed": r[5], "summary": json.loads(r[6])}
        for r in rows
    ]
