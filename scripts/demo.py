"""One-command demonstration mode.

Starts the backend, waits until it is healthy, creates the fixed demo session
(demo_evening scenario, RL vs static comparison, seed 42, paused at t=0) and
opens the dashboard in the default browser.

Usage:
    python scripts/demo.py [--port 8000] [--algorithms rl static] [--seed 42]
                           [--speed 1x] [--autostart]

The demo seed is fixed so the presentation is reproducible; multi-seed
honesty lives in scripts/evaluate.py (see docs/REPORT.md).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--algorithms", nargs="*", default=["rl", "static"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="demo_evening")
    ap.add_argument("--speed", default="1x")
    ap.add_argument("--model", default="ppo_te")
    ap.add_argument("--autostart", action="store_true",
                    help="start ticking immediately instead of paused")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app",
         "--port", str(args.port), "--log-level", "warning"],
        cwd=ROOT,
    )
    print(f"backend starting on {base} (pid {proc.pid}) …")
    try:
        for _ in range(60):
            try:
                httpx.get(base + "/api/scenarios", timeout=1.0)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("backend did not become healthy in 30 s")

        r = httpx.post(base + "/api/simulation/start", json={
            "scenario": args.scenario,
            "algorithms": args.algorithms,
            "seed": args.seed,
            "model_tag": args.model,
            "safety_filter": True,
            "speed": args.speed,
            "autostart": args.autostart,
        }, timeout=60.0)
        r.raise_for_status()
        print("demo session ready:", r.json())
        print("controls: Resume to run, Step for one 5-min interval at a time.")
        webbrowser.open(base)
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
