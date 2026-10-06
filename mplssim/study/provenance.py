"""Provenance records attached to every study artifact."""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "torch", "gymnasium",
                "stable_baselines3", "sb3_contrib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - optional
            out[mod] = "unavailable"
    return out


def hardware() -> dict[str, Any]:
    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "gpu": "none (CPU-only run)"}


def run_record(**fields: Any) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "software": software_versions(),
        "hardware": hardware(),
        **fields,
    }
