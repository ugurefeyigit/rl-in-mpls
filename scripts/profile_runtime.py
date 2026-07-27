"""Runtime profiler for the MPLS-TE simulation, RL environment and evaluation.

Measures the components that dominate interactive (server) and offline
(evaluation) latency, under fixed seeds, with warm-up and repetitions. Each
benchmark reports median and p95 so that a single slow sample (GC pause, OS
scheduling) cannot be mistaken for a regression.

Usage:
    python scripts/profile_runtime.py                     # full suite
    python scripts/profile_runtime.py --quick             # fewer repetitions
    python scripts/profile_runtime.py --only step mask    # substring filter
    python scripts/profile_runtime.py --label after       # tag the output file

Output:
    results/runtime_audit_<label>.json   machine-readable timings + environment
    stdout                               markdown table

The script never trains, never writes into `results/eval_*`, and never mutates
configuration. It is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mplssim.baselines import make_baseline
from mplssim.experiments.runner import run_episode
from mplssim.factory import engine_config_from_training, make_engine
from mplssim.rl.env import MplsTeEnv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED = 101
PROFILE_SCENARIO = "evening_peak"   # 420 min => 84 control intervals


# ----------------------------------------------------------------- harness
class Bench:
    """Collects timing samples for named benchmarks."""

    def __init__(self, quick: bool = False, only: list[str] | None = None) -> None:
        self.quick = quick
        self.only = only
        self.results: dict[str, dict[str, Any]] = {}

    def selected(self, name: str) -> bool:
        return not self.only or any(o in name for o in self.only)

    def run(
        self,
        name: str,
        setup: Callable[[], Any],
        body: Callable[[Any], Any],
        *,
        unit_per_call: float = 1.0,
        unit_name: str = "calls",
        reps: int = 7,
        warmup: int = 2,
        inner: int = 1,
    ) -> None:
        """Time ``body(state)`` where ``state`` comes from a fresh ``setup()``.

        ``setup`` runs outside the timed region for every repetition, so the
        measurement isolates ``body``. ``inner`` repeats ``body`` inside one
        timed region for operations too fast to time individually; the reported
        per-call time divides it out. ``unit_per_call`` converts one ``body``
        call into domain units (e.g. micro-ticks) for the throughput column.
        """
        if not self.selected(name):
            return
        if self.quick:
            reps = max(3, reps // 2)
            warmup = 1
        for _ in range(warmup):
            body(setup())
        samples: list[float] = []
        for _ in range(reps):
            state = setup()
            t0 = time.perf_counter()
            for _ in range(inner):
                body(state)
            samples.append((time.perf_counter() - t0) / inner)
        med = statistics.median(samples)
        p95 = float(np.percentile(samples, 95))
        self.results[name] = {
            "median_s": med,
            "p95_s": p95,
            "min_s": min(samples),
            "reps": reps,
            "inner": inner,
            "samples_s": samples,
            "unit_name": unit_name,
            "unit_per_call": unit_per_call,
            "throughput_per_s": (unit_per_call / med) if med > 0 else float("inf"),
        }
        print(f"  {name:38s} median={_fmt(med)}  p95={_fmt(p95)}  "
              f"{self.results[name]['throughput_per_s']:12,.1f} {unit_name}/s")

    def derive(self, name: str, source: str, unit_per_call: float, unit_name: str) -> None:
        """Add a row that re-expresses ``source``'s samples in another unit.

        Used where one measurement answers two questions (a control interval is
        exactly ``micro_ticks_per_interval`` micro-ticks), so both rows stay
        numerically consistent instead of drifting apart as separate runs.
        """
        src = self.results.get(source)
        if src is None:
            return
        med = src["median_s"] / unit_per_call
        self.results[name] = {
            **{k: src[k] / unit_per_call for k in ("median_s", "p95_s", "min_s")},
            "reps": src["reps"], "inner": src["inner"],
            "samples_s": [s / unit_per_call for s in src["samples_s"]],
            "unit_name": unit_name, "unit_per_call": 1.0,
            "throughput_per_s": (1.0 / med) if med > 0 else float("inf"),
            "derived_from": source,
        }
        print(f"  {name:38s} median={_fmt(med)}  p95={_fmt(self.results[name]['p95_s'])}  "
              f"{self.results[name]['throughput_per_s']:12,.1f} {unit_name}/s")

    def memory(self, name: str, setup: Callable[[], Any], body: Callable[[Any], Any]) -> None:
        """Record peak traced allocation for one ``body`` call."""
        if not self.selected(name):
            return
        state = setup()
        body(state)                      # warm caches so we measure steady state
        state = setup()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        body(state)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.results[name] = {
            "peak_alloc_bytes": int(peak - base),
            "retained_bytes": int(cur - base),
        }
        print(f"  {name:38s} peak_alloc={(peak - base)/1024:10.1f} KiB  "
              f"retained={(cur - base)/1024:10.1f} KiB")


def _fmt(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:8.3f} s "
    if seconds >= 1e-3:
        return f"{seconds*1e3:8.3f} ms"
    return f"{seconds*1e6:8.3f} us"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def cpu_name() -> str:
    name = platform.processor() or ""
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                name = out.splitlines()[0].strip()
        except Exception:
            pass
    return name or "unknown"


def environment() -> dict[str, Any]:
    import pandas
    import scipy
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "git_commit": git_commit(),
        "seed": SEED,
        "profile_scenario": PROFILE_SCENARIO,
    }
    try:
        import stable_baselines3
        env["stable_baselines3"] = stable_baselines3.__version__
    except Exception:
        env["stable_baselines3"] = None
    return env


# ---------------------------------------------------------------- fixtures
def fresh_engine(scenario: str = PROFILE_SCENARIO, seed: int = SEED):
    return make_engine(scenario, seed=seed, cfg=engine_config_from_training())


def advanced_engine(steps: int = 20):
    """An engine warmed to a mid-episode state (non-trivial loads and history)."""
    eng = fresh_engine()
    for _ in range(steps):
        eng.step_interval()
    return eng


def fresh_env(scenario: str = PROFILE_SCENARIO, seed: int = SEED) -> MplsTeEnv:
    env = MplsTeEnv(scenario=scenario, base_seed=seed)
    env.reset(options={"episode_seed": seed})
    return env


def advanced_env(steps: int = 20) -> MplsTeEnv:
    env = fresh_env()
    for _ in range(steps):
        env.step(0)
    return env


def load_rl_model():
    """Load the pretrained MaskablePPO checkpoint, or None if unavailable.

    Loads the archive directly rather than through ``server.session`` so that
    profiling produces no event-log side effects.
    """
    try:
        from sb3_contrib import MaskablePPO
    except Exception:
        return None
    base = ROOT / "models" / "ppo_te"
    path = base / "best_model.zip"
    if not path.exists():
        path = base / "final_model.zip"
    if not path.exists():
        return None
    try:
        return MaskablePPO.load(path, device="cpu")
    except Exception as exc:                                   # pragma: no cover
        print(f"  (model load failed: {exc})")
        return None


# -------------------------------------------------------------- benchmarks
def bench_engine(b: Bench) -> None:
    print("\n[engine]")
    cfg = engine_config_from_training()
    micro = cfg.micro_ticks_per_interval

    b.run("engine.step_interval", advanced_engine,
          lambda e: e.step_interval(),
          unit_name="intervals", inner=20)
    b.derive("engine.micro_tick", "engine.step_interval",
             unit_per_call=micro, unit_name="micro-ticks")
    b.run("engine._compute_tick", advanced_engine,
          lambda e: e._compute_tick(),
          unit_name="ticks", inner=200)
    b.run("engine.path_available", advanced_engine,
          lambda e: [e.path_available(d, p) for d in range(e.n_demands) for p in range(4)],
          unit_per_call=68.0, unit_name="checks", inner=50)
    b.run("engine.validate_action_sweep", advanced_engine,
          lambda e: [e.validate_action(d, p, "rl")
                     for d in range(e.n_demands) for p in range(4)],
          unit_per_call=68.0, unit_name="validations", inner=20)
    b.run("engine.candidate_info_all", advanced_engine,
          lambda e: [e.candidate_info(d) for d in range(e.n_demands)],
          unit_name="sweeps", inner=20)
    b.run("engine.snapshot", advanced_engine,
          lambda e: e.snapshot(),
          unit_name="snapshots", inner=20)
    b.run("engine.snapshot_json", advanced_engine,
          lambda e: json.dumps(e.snapshot(), default=str),
          unit_name="payloads", inner=10)
    b.run("engine.clone", advanced_engine,
          lambda e: e.clone(),
          unit_name="clones", inner=5)
    b.run("engine.fast_clone", advanced_engine,
          lambda e: e.fast_clone(),
          unit_name="clones", inner=50)
    b.run("engine.clone_and_step", advanced_engine,
          lambda e: e.clone().step_interval(),
          unit_name="counterfactuals", inner=5)
    b.run("engine.fast_clone_and_step", advanced_engine,
          lambda e: e.fast_clone().step_interval(),
          unit_name="counterfactuals", inner=20)
    b.run("engine._lsp_counts", advanced_engine,
          lambda e: e._lsp_counts(),
          unit_name="counts", inner=200)
    b.run("engine.construct", lambda: None,
          lambda _: fresh_engine(),
          unit_name="engines", inner=5)


def bench_env(b: Bench) -> None:
    print("\n[rl env]")
    b.run("env.step(noop)", advanced_env,
          lambda e: e.step(0),
          unit_name="steps", inner=20)
    b.run("env.action_masks", advanced_env,
          lambda e: e.action_masks(),
          unit_name="masks", inner=100)
    b.run("env._obs", advanced_env,
          lambda e: e._obs(),
          unit_name="observations", inner=200)
    b.run("env.reset", lambda: MplsTeEnv(scenario=PROFILE_SCENARIO, base_seed=SEED),
          lambda e: e.reset(options={"episode_seed": SEED}),
          unit_name="resets", inner=3)


def bench_episodes(b: Bench) -> None:
    print("\n[episodes]")
    for scenario in ("full_day", "demo_evening"):
        b.run(f"episode.engine_only[{scenario}]",
              lambda s=scenario: fresh_engine(s),
              _drain_engine, unit_name="episodes", reps=5)
        b.run(f"episode.static[{scenario}]", lambda: None,
              lambda _, s=scenario: run_episode("static", s, SEED),
              unit_name="episodes", reps=5)


def _drain_engine(eng) -> None:
    while not eng.done:
        eng.step_interval()


def bench_rl(b: Bench, model: Any) -> None:
    print("\n[rl inference]")
    if model is None:
        print("  (no model available - skipped)")
        return

    def predict_and_step(env: MplsTeEnv) -> None:
        obs = env._obs()
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        env.step(int(action))

    def predict_only(env: MplsTeEnv) -> None:
        obs = env._obs()
        mask = env.action_masks()
        model.predict(obs, deterministic=True, action_masks=mask)

    b.run("rl.predict_only", advanced_env, predict_only,
          unit_name="predictions", inner=20)
    b.run("rl.predict_plus_env_step", advanced_env, predict_and_step,
          unit_name="decisions", inner=20)
    b.run("episode.rl[demo_evening]", lambda: None,
          lambda _: run_episode("rl", "demo_evening", SEED, model=model),
          unit_name="episodes", reps=3)


def bench_evaluation(b: Bench, model: Any) -> None:
    print("\n[offline evaluation]")
    algos = ["static", "greedy", "cspf", "random"] + (["rl"] if model else [])

    def one_seed_all_algos(_: Any) -> None:
        for algo in algos:
            run_episode(algo, "evening_peak", SEED, model=model)

    name = f"eval.evening_peak_x{len(algos)}algos_x1seed"
    b.run(name, lambda: None, one_seed_all_algos, unit_name="sweeps", reps=3)

    for algo in algos:
        b.run(f"eval.single[{algo}]", lambda: None,
              lambda _, a=algo: run_episode(a, "evening_peak", SEED, model=model),
              unit_name="episodes", reps=3)


def bench_memory(b: Bench) -> None:
    print("\n[memory]")
    b.memory("mem.engine.step_interval", advanced_engine, lambda e: e.step_interval())
    b.memory("mem.engine.snapshot", advanced_engine, lambda e: e.snapshot())
    b.memory("mem.engine.clone", advanced_engine, lambda e: e.clone())
    b.memory("mem.engine.fast_clone", advanced_engine, lambda e: e.fast_clone())
    b.memory("mem.env.action_masks", advanced_env, lambda e: e.action_masks())
    b.memory("mem.env.step", advanced_env, lambda e: e.step(0))
    b.memory("mem.episode.demo_evening", lambda: fresh_engine("demo_evening"), _drain_engine)


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer repetitions")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter over benchmark names")
    ap.add_argument("--label", default="baseline",
                    help="output tag: results/runtime_audit_<label>.json")
    ap.add_argument("--no-model", action="store_true", help="skip RL model benchmarks")
    args = ap.parse_args()

    env_info = environment()
    print("=" * 78)
    for k, v in env_info.items():
        print(f"  {k:22s} {v}")
    print("=" * 78)

    # Warm-up: import graphs, YAML configs and the candidate-path cache are all
    # process-lifetime caches; touching them here keeps them out of the samples.
    print("\n[warm-up]")
    warm = fresh_engine()
    warm.step_interval()
    warm.snapshot()
    MplsTeEnv(scenario=PROFILE_SCENARIO, base_seed=SEED).action_masks()
    print("  caches primed")

    model = None if args.no_model else load_rl_model()
    if model is not None:
        print(f"  model loaded: models/ppo_te")

    b = Bench(quick=args.quick, only=args.only)
    bench_engine(b)
    bench_env(b)
    bench_episodes(b)
    bench_rl(b, model)
    bench_evaluation(b, model)
    bench_memory(b)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"runtime_audit_{args.label}.json"
    out.write_text(json.dumps(
        {"environment": env_info, "results": b.results}, indent=1), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"| {'benchmark':38s} | {'median':>12s} | {'p95':>12s} | {'throughput':>16s} |")
    print(f"|{'-'*40}|{'-'*14}|{'-'*14}|{'-'*18}|")
    for name, r in b.results.items():
        if "median_s" not in r:
            continue
        print(f"| {name:38s} | {_fmt(r['median_s']):>12s} | {_fmt(r['p95_s']):>12s} | "
              f"{r['throughput_per_s']:10,.1f} {r['unit_name'][:5]:5s} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
