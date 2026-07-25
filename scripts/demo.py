"""One-command demonstration mode.

Starts the backend with training disabled, waits until it is healthy, creates
the fixed demo session (demo_evening, RL vs greedy, seed 42, advisor mode,
paused at t=0) and opens Presentation Mode in the default browser.

Usage:
    python scripts/demo.py                       # Presentation Mode, no training
    python scripts/demo.py --advanced            # open the engineering console
    python scripts/demo.py --allow-training      # re-enable the training endpoint
    python scripts/demo.py --port 8000 --seed 42 --algorithms rl greedy

Training is disabled by default (ALLOW_TRAINING=false in the child process):
a stray click on the training tab during a live presentation would spawn a
long-running job on the presenting machine. Pass --allow-training if you
actually intend to train.

The demo seed is fixed so the presentation is reproducible; multi-seed honesty
lives in scripts/evaluate.py (see docs/REPORT.md).
"""

from __future__ import annotations

import argparse
import os
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
    ap.add_argument("--algorithms", nargs="*", default=["rl", "greedy"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="demo_evening")
    ap.add_argument("--speed", default="1x")
    ap.add_argument("--model", default="ppo_te")
    ap.add_argument("--autostart", action="store_true",
                    help="start ticking immediately instead of paused")
    ap.add_argument("--advanced", action="store_true",
                    help="open the engineering console instead of Presentation Mode")
    ap.add_argument("--allow-training", action="store_true",
                    help="re-enable POST /api/agent/train (disabled by default)")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ)
    env["ALLOW_TRAINING"] = "true" if args.allow_training else "false"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app",
         "--port", str(args.port), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    mode = "engineering console" if args.advanced else "Presentation Mode"
    print(f"backend starting on {base} (pid {proc.pid}) …")
    print(f"training endpoint: {'ENABLED' if args.allow_training else 'disabled'}")
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
            "advisor": True,
            "interface_mode": "advanced" if args.advanced else "present",
        }, timeout=120.0)
        r.raise_for_status()
        status = r.json()
        print(f"demo session ready: {status['scenario']} · seed {status['seed']} · "
              f"{' vs '.join(status['algorithms'])} · state {status['state']}")
        print(f"opening {mode} at {base}{'/advanced' if args.advanced else '/present'}")
        if args.advanced:
            print("controls: Start session, then Resume / Step, or Recommend for advisor mode.")
        else:
            print("controls: 'Start Guided 5-Minute Story', then Space / → / A / R / F.")
        if not args.no_browser:
            webbrowser.open(base + ("/advanced" if args.advanced else "/present"))
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
