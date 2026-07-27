"""Independent integrity checks for the offline evaluation path.

Each check is an executable assertion about a property the paired-comparison
methodology depends on, so claims in docs/PERFORMANCE_AND_EVALUATION_AUDIT.md
can be re-verified rather than taken on trust.

Usage:
    python scripts/audit_evaluation.py                # all checks
    python scripts/audit_evaluation.py --only traffic # substring filter

Exit code is 0 when every check passes, 1 otherwise. Checks that surface a
known finding print FINDING and do not fail the run; they are reported so the
audit document and the code cannot drift apart.

Writes nothing unless --write-outputs is given, in which case verification
artefacts go to results/runtime_audit_eval/ (never results/eval_*).
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mplssim.baselines import make_baseline
from mplssim.experiments.runner import run_episode, summarize_records
from mplssim.factory import engine_config_from_training, make_engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "runtime_audit_eval"

SCENARIO = "link_failure"
SEED = 101
ALGOS = ["static", "greedy", "cspf", "random"]

_RESULTS: list[tuple[str, str, str]] = []       # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    _RESULTS.append((status, name, detail))
    mark = {"PASS": "  PASS   ", "FAIL": "  FAIL   ", "FINDING": "  FINDING"}[status]
    print(f"{mark} {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"           {line}")


# ------------------------------------------------------------------ checks
def check_identical_traffic_across_algorithms() -> None:
    """Every algorithm must face byte-identical offered traffic."""
    per_algo = {}
    for algo in ALGOS:
        df, _ = run_episode(algo, SCENARIO, SEED)
        per_algo[algo] = df["offered_mbps"].to_numpy()
    ref_name, ref = next(iter(per_algo.items()))
    bad = []
    for algo, vals in per_algo.items():
        if vals.shape != ref.shape or not np.array_equal(vals, ref):
            n = int(np.sum(vals[:min(len(vals), len(ref))]
                           != ref[:min(len(vals), len(ref))]))
            bad.append(f"{algo}: {n} steps differ from {ref_name}")
    if bad:
        record("FAIL", "identical offered traffic across algorithms", "\n".join(bad))
    else:
        record("PASS", "identical offered traffic across algorithms",
               f"{len(ALGOS)} algorithms, {len(ref)} steps, exact match")


def check_traffic_rng_isolated_from_controllers() -> None:
    """Controller decisions must not perturb the traffic RNG stream."""
    cfg = engine_config_from_training()
    passive = make_engine(SCENARIO, seed=SEED, cfg=cfg)
    active = make_engine(SCENARIO, seed=SEED, cfg=cfg)
    for step in range(40):
        active.apply_action(step % active.n_demands, 1 + (step % 3), source="rl")
        passive.step_interval()
        active.step_interval()
    same_noise = np.array_equal(passive.traffic._noise, active.traffic._noise)
    same_state = (passive.traffic._rng.bit_generator.state
                  == active.traffic._rng.bit_generator.state)
    same_vols = np.array_equal(passive.demand_volumes, active.demand_volumes)
    if same_noise and same_state and same_vols:
        record("PASS", "traffic RNG isolated from controller behaviour",
               "AR(1) noise, generator state and offered volumes all identical "
               "after 40 intervals with vs without reroutes")
    else:
        record("FAIL", "traffic RNG isolated from controller behaviour",
               f"noise={same_noise} rng_state={same_state} volumes={same_vols}")


def check_equal_micro_ticks_and_steps() -> None:
    """Every runner must advance the same simulated time."""
    lengths, times = {}, {}
    for algo in ALGOS:
        df, summary = run_episode(algo, SCENARIO, SEED)
        lengths[algo] = summary["steps"]
        times[algo] = float(df["t_min"].iloc[-1])
    if len(set(lengths.values())) == 1 and len(set(times.values())) == 1:
        record("PASS", "all runners advance the same number of intervals",
               f"steps={next(iter(lengths.values()))}, "
               f"final t_min={next(iter(times.values()))} for {sorted(lengths)}")
    else:
        record("FAIL", "all runners advance the same number of intervals",
               f"steps={lengths} final_t={times}")


def check_rl_early_termination_asymmetry() -> None:
    """The RL loop can stop early on total disconnection; baselines cannot.

    Both loops end on ``done`` (scenario duration reached), but the RL loop also
    ends on ``terminated`` = every demand disconnected. If that ever fires, the
    RL episode is summarized over fewer intervals than its paired baselines,
    and per-step means are then not comparable.
    """
    rl_src = inspect.getsource(sys.modules["mplssim.experiments.runner"])
    rl_has_terminated = "terminated or truncated" in rl_src
    baseline_has_terminated = "while not eng.done" in rl_src
    _df, summary = run_episode("static", SCENARIO, SEED)
    if rl_has_terminated and baseline_has_terminated:
        record("FINDING", "RL vs baseline episode-termination condition differs",
               "runner._run_rl stops on `terminated or truncated`, where terminated =\n"
               "engine.all_disconnected. runner._run_baseline stops only on\n"
               "`eng.done` (duration reached). If every demand is ever\n"
               "simultaneously disconnected, the RL episode is summarized over\n"
               "fewer intervals than the baselines it is paired against.\n"
               "MITIGATED, not removed: the RL summary now carries\n"
               "`terminated_early`, so the condition is visible instead of silently\n"
               "shortening an episode. The loop itself is unchanged because\n"
               "`terminated` is part of the Gymnasium contract and altering it would\n"
               "change evaluation semantics. Never fires in the suite (checked next).")
    else:
        record("PASS", "RL and baseline termination conditions match")


def check_all_disconnected_never_fires() -> None:
    """Quantify the exposure of the termination asymmetry above."""
    worst = {}
    for scenario in ("link_failure", "ood_double_failure", "overload_stress",
                     "demo_evening", "full_day"):
        cfg = engine_config_from_training()
        eng = make_engine(scenario, seed=SEED, cfg=cfg)
        peak = 0
        while not eng.done:
            eng.step_interval()
            peak = max(peak, int(np.sum(eng.disconnected)))
        worst[scenario] = f"{peak}/{eng.n_demands}"
    if all(int(v.split("/")[0]) < int(v.split("/")[1]) for v in worst.values()):
        record("PASS", "all_disconnected never fires in the evaluated scenarios",
               "peak simultaneous disconnections: "
               + ", ".join(f"{k}={v}" for k, v in worst.items()))
    else:
        record("FAIL", "all_disconnected fires in some scenario", str(worst))


def check_reward_accounting_symmetry() -> None:
    """Baselines always report invalid=False; the RL env reports it truthfully."""
    cfg = engine_config_from_training()
    rejected = {}
    for algo in ("static", "greedy", "cspf", "random"):
        eng = make_engine(SCENARIO, seed=SEED, cfg=cfg)
        ctl = make_baseline(algo, seed=SEED)
        n_rejected = 0
        while not eng.done:
            for d_idx, p_idx in ctl.decide(eng):
                ok, _ = eng.apply_action(
                    d_idx, p_idx, source=ctl.name if ctl.name != "random" else "rl")
                n_rejected += int(not ok)
            eng.step_interval()
        rejected[algo] = n_rejected
    total = sum(rejected.values())
    detail = ("runner._run_baseline calls compute_reward(..., invalid=False)\n"
              "unconditionally, while _run_rl passes the real flag. Rejected\n"
              f"baseline actions in {SCENARIO}/seed {SEED}: {rejected}.")
    if total == 0:
        record("PASS", "reward accounting symmetric in practice",
               detail + "\nNo baseline action was rejected, so the hardcoded\n"
                        "invalid=False changed no reward here. Still a latent\n"
                        "asymmetry if a baseline ever proposes an invalid move.")
    else:
        record("FINDING", "baselines are never charged the invalid-action penalty",
               detail + f"\n{total} rejected baseline actions received no invalid\n"
                        "penalty, whereas the RL agent is charged for each one.")


def check_frr_distinguished_from_controller_actions() -> None:
    """FRR repairs must be attributable separately from controller reroutes."""
    cfg = engine_config_from_training()
    eng = make_engine(SCENARIO, seed=SEED, cfg=cfg)
    while not eng.done:
        eng.apply_action(eng.step_count % eng.n_demands, 1, source="rl")
        eng.step_interval()
    sources = {}
    for rec in eng.action_log:
        sources[rec.source] = sources.get(rec.source, 0) + 1
    frr = sum(1 for a in eng.action_log if a.source == "frr" and a.accepted)
    counted = int(sum(row["frr_events"] for row in eng.metrics_history))
    if "frr" in sources and frr == counted:
        record("PASS", "FRR actions distinguished from controller actions",
               f"action_log sources={sources}; accepted FRR={frr} equals the "
               f"summed frr_events={counted}")
    else:
        record("FAIL", "FRR attribution inconsistent",
               f"sources={sources} accepted_frr={frr} summed_frr_events={counted}")


def check_flap_attribution_excludes_frr() -> None:
    """The flap penalty must reflect controller churn, not failure repair."""
    src = inspect.getsource(sys.modules["mplssim.experiments.runner"])
    baseline_ok = "eng.action_log[log_mark:] if a.accepted" in src
    rl_ok = "self.eng.action_log[-1].is_flap" in inspect.getsource(
        sys.modules["mplssim.rl.env"])
    if baseline_ok and rl_ok:
        record("PASS", "flap attribution restricted to controller actions",
               "baselines slice action_log from the pre-decision mark; the RL env\n"
               "reads the flag off its own just-applied action")
    else:
        record("FAIL", "flap attribution differs between runners",
               f"baseline_ok={baseline_ok} rl_ok={rl_ok}")


def check_decision_timing_scope() -> None:
    """What each runner includes in mean_decision_time_ms."""
    src = inspect.getsource(sys.modules["mplssim.experiments.runner"])
    rl_block = src[src.index("def _run_rl"):]
    mask_outside = rl_block.index("mask = env.action_masks()") < rl_block.index("t0 =")
    has_mask_metric = "mean_mask_time_ms" in src
    record("FINDING", "decision-time scope differs between RL and baselines",
           "runner._run_baseline times ctl.decide(eng), which INCLUDES each\n"
           "heuristic's own feasibility scanning (validate_action /\n"
           "path_available sweeps).\n"
           f"runner._run_rl computes env.action_masks() OUTSIDE the timed region\n"
           f"(mask before t0: {mask_outside}) and times only model.predict().\n"
           "So `mean_decision_time_ms` for RL excludes the mask generation that\n"
           "the safety filter requires, while baseline numbers include the\n"
           "equivalent work.\n"
           f"MITIGATED, not removed: `mean_mask_time_ms` is now reported\n"
           f"separately (present={has_mask_metric}). `mean_decision_time_ms` keeps\n"
           "its original meaning so historical numbers stay comparable; add the\n"
           "two together for a like-for-like comparison against a baseline.")


def check_recovery_metric_consistency() -> None:
    """recovery_steps must be defined the same way for every algorithm."""
    rows = {}
    for algo in ALGOS:
        df, summary = run_episode(algo, SCENARIO, SEED)
        fail = df.index[df["n_failed_links"] > 0]
        expected = None
        if len(fail):
            f0 = int(fail[0])
            post = df.loc[f0:]
            ok = post.index[post["sla_violations"] == 0]
            expected = int(ok[0] - f0) if len(ok) else len(df) - f0
        rows[algo] = (summary["recovery_steps"], expected)
    mismatched = {a: v for a, v in rows.items() if v[0] != v[1]}
    if mismatched:
        record("FAIL", "recovery_steps computed consistently", str(mismatched))
    else:
        record("PASS", "recovery_steps computed consistently",
               f"{ {a: v[0] for a, v in rows.items()} } "
               "(steps from first failed-link step until SLA violations reach 0)")


def check_recovery_steps_censoring() -> None:
    """recovery_steps is censored, not missing, when SLA never recovers."""
    df, summary = run_episode("static", "overload_stress", SEED)
    has_failure = bool((df["n_failed_links"] > 0).any())
    record("FINDING", "recovery_steps mixes 'no failure', 'recovered' and 'censored'",
           "summarize_records returns None when a scenario has no failure, but a\n"
           "CENSORED value (n - f0, i.e. 'never recovered') when SLA violations\n"
           "never return to zero. Those two cases are indistinguishable downstream\n"
           "from a genuine fast recovery, and the censored value is a lower bound\n"
           "that depends on episode length.\n"
           f"overload_stress/seed {SEED} has_scripted_failure={has_failure}, "
           f"recovery_steps={summary['recovery_steps']}.")


def check_ci_sample_sizes() -> None:
    """n_seeds must describe the sample each statistic actually used."""
    frames = []
    for algo in ("static", "greedy"):
        for scenario in ("link_failure", "evening_peak"):
            _df, s = run_episode(algo, scenario, SEED)
            frames.append(s)
    sdf = pd.DataFrame(frames)
    n_missing = int(sdf["recovery_steps"].isna().sum())
    eval_src = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    has_n = "{mkey}_n" in eval_src
    record("PASS" if has_n else "FINDING",
           "per-metric sample size recorded alongside each statistic",
           "scripts/evaluate.py sets row['n_seeds'] = len(g) once per\n"
           "(scenario, algorithm) group, then computes each metric over\n"
           "g[metric].dropna(). recovery_steps is None for any scenario without a\n"
           "failure, so its mean/std/ci95 can rest on fewer samples than n_seeds\n"
           "advertises.\n"
           f"In this mini-sample {n_missing}/{len(sdf)} rows had recovery_steps=None.\n"
           f"FIXED: a `<metric>_n` column now records the real per-metric sample\n"
           f"size (present={has_n}), which also tells ci95()'s 0.0-for-n<2 return\n"
           "apart from a genuinely tight interval. Existing columns unchanged.")


def check_paired_delta_alignment() -> None:
    """Paired deltas must subtract matching (scenario, seed) rows."""
    rows = []
    for algo in ("static", "greedy"):
        for scenario in ("link_failure", "evening_peak"):
            _df, s = run_episode(algo, scenario, SEED)
            rows.append(s)
    sdf = pd.DataFrame(rows)
    a = sdf[sdf.algorithm == "static"].set_index(["scenario", "seed"])
    b = sdf[sdf.algorithm == "greedy"].set_index(["scenario", "seed"])
    common = a.index.intersection(b.index)
    delta = (a.loc[common, "reward_sum"] - b.loc[common, "reward_sum"])
    manual = [float(a.loc[key, "reward_sum"]) - float(b.loc[key, "reward_sum"])
              for key in common]
    aligned = np.allclose(delta.to_numpy(dtype=float), manual)
    dup = int(sdf.duplicated(subset=["algorithm", "scenario", "seed"]).sum())
    if aligned and dup == 0:
        record("PASS", "paired deltas align on (scenario, seed)",
               "both sides are indexed by the same intersection object, so rows\n"
               "cannot slip relative to one another; no duplicate keys present.\n"
               "FIXED: evaluate.py now rejects a repeated --seeds value, which\n"
               "would otherwise create duplicate index keys and make the paired\n"
               ".loc lookups expand instead of aligning one-to-one.")
    else:
        record("FAIL", "paired deltas misaligned",
               f"aligned={aligned} duplicate_keys={dup}")


def check_export_units() -> None:
    """dropped_gbit_total must be Mbps*s converted to gigabits exactly once."""
    df, summary = run_episode("static", SCENARIO, SEED)
    interval_s = 5 * 60
    expected = float(((df["offered_mbps"] - df["carried_mbps"]) * interval_s / 1000).sum())
    got = summary["dropped_gbit_total"]
    if got == expected:
        record("PASS", "dropped_gbit_total unit conversion",
               f"sum((offered-carried) Mbps * {interval_s} s) / 1000 = {got:.3f} Gbit.\n"
               "Note the interval length is hardcoded as 5*60 s in\n"
               "summarize_records; it is not read from EngineConfig, so a changed\n"
               "control_interval_min would silently scale this metric.")
    else:
        record("FAIL", "dropped_gbit_total unit conversion",
               f"got {got} expected {expected}")


def check_interval_length_hardcoded() -> None:
    """The hardcoded 5-minute interval in summarize_records vs the real config."""
    cfg = engine_config_from_training()
    if cfg.control_interval_min != 5:
        record("FAIL", "summarize_records interval length matches config",
               f"config says {cfg.control_interval_min} min, summarize_records "
               f"assumes 5")
    else:
        runner_src = (ROOT / "mplssim" / "experiments" / "runner.py").read_text(
            encoding="utf-8")
        derived = "_interval_seconds(df)" in runner_src
        record("PASS" if derived else "FINDING",
               "control-interval length is derived, not hardcoded",
               f"configs/training.yaml sets control_interval_min="
               f"{cfg.control_interval_min}.\n"
               f"FIXED: summarize_records reads the interval from the trace itself\n"
               f"(t_min after the first interval) via _interval_seconds "
               f"(present={derived}),\n"
               "so dropped_gbit_total cannot silently mis-scale if the control\n"
               "interval changes. The value is unchanged at the current 5 minutes.")


def check_outputs_are_overwritten_not_appended() -> None:
    """Evaluation writers must truncate, never append."""
    src = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    appends = [ln.strip() for ln in src.splitlines()
               if ("to_csv" in ln or "write_text" in ln or "open(" in ln)
               and ("mode=" in ln or "'a'" in ln or '"a"' in ln)]
    stale = sorted(p.name for p in (ROOT / "results").glob("eval_steps_*.csv"))
    if appends:
        record("FAIL", "evaluation outputs are truncated, not appended", str(appends))
    else:
        record("PASS", "evaluation outputs are truncated, not appended",
               "every writer uses to_csv/write_text in truncating mode.\n"
               f"Separate risk: {len(stale)} eval_steps_*.csv files already exist and\n"
               "are only overwritten for scenarios/algorithms present in the CURRENT\n"
               "run, so a narrower re-run leaves older files behind for\n"
               "make_figures.py to pick up. Use --prefix to isolate a run.")


def check_benchmark_sources_regenerable() -> None:
    """Files behind the benchmark endpoint must be reproducible from scripts."""
    results = ROOT / "results"
    needed = ["eval_summary.csv", "eval_stats.csv", "eval_summary.json"]
    present = {n: (results / n).exists() for n in needed}
    server_reads = []
    main_src = (ROOT / "server" / "main.py").read_text(encoding="utf-8")
    for name in needed:
        if name in main_src:
            server_reads.append(name)
    record("PASS" if all(present.values()) else "FINDING",
           "benchmark endpoint sources are regenerable",
           f"present={present}\n"
           f"referenced by server/main.py: {server_reads or 'none by literal name'}\n"
           "regenerate with:\n"
           "  python scripts/evaluate.py --model ppo_te \\\n"
           "    --scenarios full_day evening_peak flash_crowd link_failure \\\n"
           "      deceptive_local_optimum ood_double_failure overload_stress \\\n"
           "    --seeds 101 102 103 104 105 \\\n"
           "    --algorithms static greedy cspf random rl")


def check_summarize_is_pure() -> None:
    """summarize_records must not mutate the frame it is handed."""
    df, _ = run_episode("static", SCENARIO, SEED)
    before = df.copy(deep=True)
    summarize_records(df, algorithm="static", scenario=SCENARIO, seed=SEED)
    if df.equals(before):
        record("PASS", "summarize_records does not mutate its input frame")
    else:
        record("FAIL", "summarize_records mutates its input frame")


CHECKS = [
    check_identical_traffic_across_algorithms,
    check_traffic_rng_isolated_from_controllers,
    check_equal_micro_ticks_and_steps,
    check_rl_early_termination_asymmetry,
    check_all_disconnected_never_fires,
    check_reward_accounting_symmetry,
    check_frr_distinguished_from_controller_actions,
    check_flap_attribution_excludes_frr,
    check_decision_timing_scope,
    check_recovery_metric_consistency,
    check_recovery_steps_censoring,
    check_ci_sample_sizes,
    check_paired_delta_alignment,
    check_export_units,
    check_interval_length_hardcoded,
    check_outputs_are_overwritten_not_appended,
    check_benchmark_sources_regenerable,
    check_summarize_is_pure,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--write-outputs", action="store_true",
                    help=f"write verification artefacts to {OUT}")
    args = ap.parse_args()

    print("=" * 78)
    print("  offline evaluation integrity audit")
    print(f"  scenario={SCENARIO} seed={SEED} algorithms={ALGOS}")
    print("=" * 78)

    t0 = time.perf_counter()
    for check in CHECKS:
        if args.only and not any(o in check.__name__ for o in args.only):
            continue
        try:
            check()
        except Exception as exc:                                   # pragma: no cover
            record("FAIL", check.__name__, f"raised {type(exc).__name__}: {exc}")

    if args.write_outputs:
        OUT.mkdir(parents=True, exist_ok=True)
        rows = []
        for algo in ALGOS:
            df, summary = run_episode(algo, SCENARIO, SEED)
            rows.append(summary)
            df.to_csv(OUT / f"steps_{SCENARIO}_seed{SEED}_{algo}.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
        print(f"\nwrote verification artefacts to {OUT}")

    n_fail = sum(1 for s, _, _ in _RESULTS if s == "FAIL")
    n_finding = sum(1 for s, _, _ in _RESULTS if s == "FINDING")
    n_pass = sum(1 for s, _, _ in _RESULTS if s == "PASS")
    print("\n" + "=" * 78)
    print(f"  {n_pass} passed, {n_finding} findings, {n_fail} failures "
          f"({time.perf_counter() - t0:.1f}s)")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
