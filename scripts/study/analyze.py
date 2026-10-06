"""Build every processed table and every number used in the paper.

    python scripts/study/analyze.py

Reads raw evaluation outputs (via mplssim.study.collect), writes
experiments/processed/*.csv, results/tables/*.csv, paper/tables/*.tex and
paper/generated/numbers.tex (LaTeX macros; the manuscript contains no
hand-typed result numbers). Sections whose inputs do not exist yet are
skipped with a message.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mplssim.study.collect import PROCESSED, RAW, collect  # noqa: E402
from mplssim.study.protocol import EVAL_SCENARIOS  # noqa: E402
from mplssim.study.stats import (compare_fixed, compare_learners, paired_frame,  # noqa: E402
                                 per_scenario, root_t_interval, stratified_bootstrap_mean,
                                 two_stage_bootstrap)

TABLES = ROOT / "results" / "tables"
PTABLES = ROOT / "paper" / "tables"
NUMBERS: dict[str, str] = {}
SCEN_LABEL = {"full_day": "Full day", "evening_peak": "Evening peak",
              "flash_crowd": "Flash crowd", "link_failure": "Link failure",
              "deceptive_local_optimum": "Deceptive", "ood_double_failure": "Double failure",
              "overload_stress": "Overload"}
POLICY_LABEL = {"bandit": "Masked bandit", "ppo": "MaskablePPO", "static": "Static SP",
                "greedy": "Greedy", "cspf": "CSPF", "noop": "No-op",
                "random_valid": "Random valid", "milp_track": "MILP-track",
                "oracle_h1": "Oracle-1$^\\dagger$", "oracle_h3": "Oracle-3$^\\dagger$",
                "oracle_h6": "Oracle-6$^\\dagger$"}


def num(name: str, value: float | int | str, fmt: str = "{:.1f}") -> None:
    """Register a LaTeX macro \\<name> holding a formatted result."""
    if isinstance(value, str):
        NUMBERS[name] = value
    elif isinstance(value, (int, np.integer)) and fmt == "{:.1f}":
        NUMBERS[name] = f"{int(value):,}".replace(",", "{,}")
    else:
        NUMBERS[name] = fmt.format(value)


def ci_str(est: float, lo: float, hi: float, d: int = 1) -> str:
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write_table(df: pd.DataFrame, name: str, latex: str | None = None) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    if latex is not None:
        PTABLES.mkdir(parents=True, exist_ok=True)
        (PTABLES / f"{name}.tex").write_text(latex)
    print("table", name, len(df), "rows")


def latex_rows(rows: list[list[str]]) -> str:
    return "\n".join(" & ".join(r) + r" \\" for r in rows)


# ------------------------------------------------------------------ helpers
def learner_test(ep: pd.DataFrame, policy: str, role: str = "selected") -> pd.DataFrame:
    return ep[(ep.policy == policy) & (ep.role == role)]


def ref_test(ep: pd.DataFrame, policy: str) -> pd.DataFrame:
    return ep[(ep.policy == policy) & (ep.kind != "learner")]


def policy_summary(df: pd.DataFrame, learner: bool) -> dict[str, float]:
    """Mean return with the appropriate interval plus operational metrics."""
    out = {}
    if learner:
        g = df.groupby("root")["operational_return"].mean()
        pairs = df.assign(diff=df.operational_return)
        tb = two_stage_bootstrap(pairs) if len(g) > 1 else None
        out.update(mean=g.mean(), lo=tb.low if tb else np.nan, hi=tb.high if tb else np.nan,
                   roots=len(g), root_sd=g.std(ddof=1) if len(g) > 1 else np.nan)
    else:
        ci = stratified_bootstrap_mean(df.operational_return.to_numpy(), df.scenario.to_numpy())
        out.update(mean=ci.estimate, lo=ci.low, hi=ci.high, roots=0, root_sd=np.nan)
    steps = df.episode_length.to_numpy()
    out.update(
        per_interval=float((df.operational_return / df.episode_length).mean()),
        delivered=float(df.delivered_ratio_mean.mean()),
        sla=float(df.sla_violations_demand_intervals.mean()),
        maxutil=float(df.max_utilization_mean.mean()),
        loss=float(df.loss_ratio_mean.mean()),
        delay=float(df.delay_ms_mean.mean()),
        reroutes=float(df.reroutes_per_hour.mean()),
        reversals=float(df.te_reversals.mean()),
        moved=float(df.moved_mbps_total.mean()),
        noop=float(df.noop_frequency.mean()),
        decision_ms=float(df.mean_decision_time_ms.mean()) if "mean_decision_time_ms" in df else np.nan,
        episodes=len(df), steps=int(steps.sum()))
    return out


def fixed_pairs(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Pair a (learner, with roots) with b (reference or learner) on scenario/seed[/root]."""
    on = ["scenario", "seed"] + (["root"] if (a.root >= 0).all() and (b.root >= 0).all() else [])
    j = a.merge(b, on=on, suffixes=("_a", "_b"))
    j["diff"] = j.operational_return_a - j.operational_return_b
    if "root" not in j:
        j["root"] = j["root_a"]
    return j


# ---------------------------------------------------------------- sections
def section_reproduction(data: dict) -> None:
    ep = data.get("episodes_historical_holdout")
    if ep is None or not (ep.kind == "learner").any():
        print("skip reproduction: no reproduced learner evaluations")
        return
    hist_root = pd.read_csv(ROOT / "results/v2_final_holdout/per_root_metrics.csv")
    hist_scen = pd.read_csv(ROOT / "results/v2_final_holdout/scenario_metrics.csv")
    # baseline rows carry training_root == "baseline"; keep learner rows with integer roots
    hist_root = hist_root[hist_root.algorithm.isin(["masked_bandit", "maskable_ppo"])].copy()
    hist_root["training_root"] = hist_root.training_root.astype(int)
    hist_scen = hist_scen[hist_scen.algorithm.isin(["masked_bandit", "maskable_ppo"])].copy()
    hist_scen["training_root"] = hist_scen.training_root.astype(int)
    sel_hist = pd.read_csv(ROOT / "results/v2_three_root_continuity/checkpoint_selection.csv")
    rows = []
    sel = ep[(ep.kind == "learner") & (ep.role == "selected") & (ep.family == "E0_repro")]
    for (pol, root), g in sel.groupby(["policy", "root"]):
        alg = "masked_bandit" if pol == "bandit" else "maskable_ppo"
        h = hist_root[(hist_root.algorithm == alg) & (hist_root.training_root == root)]
        hs = sel_hist[(sel_hist.algorithm == alg) & (sel_hist.training_root == root)]
        sj = json.loads((RAW / "learner_eval" / f"E0_repro__{pol}__r{root}" /
                         "selection_historical.json").read_text())
        rows.append({"policy": pol, "root": root,
                     "selected_ckpt_repro": sj["selected_checkpoint"],
                     "selected_ckpt_hist": int(hs.checkpoint_transition.iloc[0]),
                     "holdout_return_repro": g.operational_return.mean(),
                     "holdout_return_hist": float(h.operational_return_mean.iloc[0]),
                     "continuity_return_repro": sj["validation_curve"][str(sj["selected_checkpoint"])]
                     if str(sj["selected_checkpoint"]) in sj["validation_curve"]
                     else sj["validation_curve"].get(sj["selected_checkpoint"]),
                     "continuity_return_hist": float(hs.mean_operational_return.iloc[0])})
    rep = pd.DataFrame(rows)
    rep["holdout_diff"] = rep.holdout_return_repro - rep.holdout_return_hist
    write_table(rep, "reproduction_per_root")
    # learner gap under the historical protocol
    pairs = paired_frame(sel.assign(policy=sel.policy), "bandit", "ppo")
    if pairs.root.nunique() >= 2:
        cl = compare_learners(pairs)
        num("ReproGap", cl["boot_est"])
        num("ReproGapLo", cl["boot_lo"])
        num("ReproGapHi", cl["boot_hi"])
        num("ReproRootsPositive", int(cl["roots_positive"]), "{}")
        num("ReproRoots", int(cl["roots"]), "{}")
        hb = hist_root[hist_root.algorithm == "masked_bandit"].operational_return_mean.mean()
        hp = hist_root[hist_root.algorithm == "maskable_ppo"].operational_return_mean.mean()
        num("HistGap", hb - hp)
        num("HistBandit", hb)
        num("HistPPO", hp)
        num("ReproBandit", sel[sel.policy == "bandit"].groupby("root").operational_return.mean().mean())
        num("ReproPPO", sel[sel.policy == "ppo"].groupby("root").operational_return.mean().mean())
        # historical gap uncertainty from per-(root,scenario) means: t over roots
        hr = hist_root.pivot_table(index="training_root", columns="algorithm",
                                   values="operational_return_mean")
        tt = root_t_interval(hr.masked_bandit - hr.maskable_ppo)
        num("HistGapTLo", tt.low)
        num("HistGapTHi", tt.high)
        write_table(pd.DataFrame([cl]), "reproduction_gap")
    # per scenario: reproduced vs historical, root-averaged bandit - ppo
    hs = hist_scen[hist_scen.algorithm.isin(["masked_bandit", "maskable_ppo"])]
    hsg = hs.pivot_table(index=["scenario", "training_root"], columns="algorithm",
                         values="operational_return_mean").reset_index()
    hsg["diff"] = hsg.masked_bandit - hsg.maskable_ppo
    hist_diff = hsg.groupby("scenario")["diff"].mean()
    ps = per_scenario(pairs)
    ps["hist_diff"] = ps.scenario.map(hist_diff)
    write_table(ps, "reproduction_per_scenario")


def section_main(data: dict) -> None:
    ep = data.get("episodes_test")
    if ep is None:
        print("skip main: no test episodes")
        return
    learners = [p for p in ("bandit", "ppo") if (ep.policy == p).any()]
    refs = [p for p in ("noop", "static", "random_valid", "cspf", "greedy", "milp_track",
                        "oracle_h1") if ((ep.policy == p) & (ep.kind != "learner")).any()]
    rows, latex = [], []
    for p in learners + refs:
        df = learner_test(ep, p) if p in learners else ref_test(ep, p)
        s = policy_summary(df, p in learners)
        s["policy"] = p
        rows.append(s)
        latex.append([POLICY_LABEL.get(p, p), ci_str(s["mean"], s["lo"], s["hi"]),
                      f"{s['per_interval']:.3f}", f"{s['delivered']:.4f}", f"{s['sla']:.0f}",
                      f"{s['maxutil']:.3f}", f"{s['reroutes']:.2f}", f"{s['moved']:.0f}"])
        key = {"bandit": "Bandit", "ppo": "PPO", "greedy": "Greedy", "cspf": "Cspf",
               "static": "Static", "noop": "Noop", "milp_track": "Milp",
               "oracle_h1": "OracleOne", "random_valid": "Random"}[p]
        num(f"Test{key}", s["mean"])
        num(f"Test{key}Lo", s["lo"])
        num(f"Test{key}Hi", s["hi"])
        num(f"Test{key}Roots", int(s["roots"]), "{}")
        num(f"Test{key}Reroutes", s["reroutes"], "{:.2f}")
        num(f"Test{key}Sla", s["sla"], "{:.0f}")
        num(f"Test{key}Delivered", 100 * s["delivered"], "{:.2f}")
        num(f"Test{key}DecisionMs", s["decision_ms"], "{:.2f}")
    main = pd.DataFrame(rows)
    write_table(main, "main_results", latex_rows(latex))
    if {"bandit", "ppo"} <= set(learners):
        sel = ep[(ep.kind == "learner") & (ep.role == "selected")]
        pairs = paired_frame(sel, "bandit", "ppo")
        cl = compare_learners(pairs)
        write_table(pd.DataFrame([cl]), "main_gap")
        for k, v in (("Gap", cl["boot_est"]), ("GapLo", cl["boot_lo"]), ("GapHi", cl["boot_hi"]),
                     ("GapTLo", cl["t_lo"]), ("GapTHi", cl["t_hi"])):
            num(k, v)
        num("GapRootsPositive", int(cl["roots_positive"]), "{}")
        num("GapRoots", int(cl["roots"]), "{}")
        num("GapWinRate", 100 * cl["episode_win_rate"], "{:.0f}")
        num("GapDz", cl["d_z_pooled"], "{:.2f}")
        ps = per_scenario(pairs)
        write_table(ps, "main_per_scenario", latex_rows([[
            SCEN_LABEL[r.scenario], ci_str(r.diff_est, r.diff_lo, r.diff_hi),
            f"{100 * r.win_rate:.0f}\\,\\%"] for r in ps.itertuples()]))
        for r in ps.itertuples():
            num("Gap" + SCEN_LABEL[r.scenario].replace(" ", ""), r.diff_est)
        # final (selection-free) checkpoints
        fin = ep[(ep.kind == "learner") & (ep.role == "final")]
        pf = paired_frame(fin, "bandit", "ppo")
        cf = compare_learners(pf)
        num("GapFinal", cf["boot_est"])
        num("GapFinalLo", cf["boot_lo"])
        num("GapFinalHi", cf["boot_hi"])
        num("GapFinalRootsPositive", int(cf["roots_positive"]), "{}")
        # per-root table
        pr = sel.groupby(["policy", "root"]).operational_return.mean().unstack(0)
        write_table(pr.reset_index(), "main_per_root")
    # learners vs references (episode-paired, root means)
    comps = []
    for p in learners:
        a = learner_test(ep, p)
        for r in refs:
            b = ref_test(ep, r)
            j = fixed_pairs(a, b)
            cl = compare_learners(j[["root", "scenario", "seed", "diff"]])
            comps.append({"learner": p, "reference": r, **cl})
            key = {"bandit": "Bandit", "ppo": "PPO"}[p] + "Minus" + {
                "greedy": "Greedy", "milp_track": "Milp", "oracle_h1": "OracleOne", "cspf": "Cspf",
                "static": "Static", "noop": "Noop", "random_valid": "Random"}[r]
            num(key, cl["boot_est"])
            num(key + "Lo", cl["boot_lo"])
            num(key + "Hi", cl["boot_hi"])
    if comps:
        write_table(pd.DataFrame(comps), "learners_vs_references")
    # references among themselves (fixed policies, episode-paired)
    rr = []
    for a_, b_ in (("milp_track", "greedy"), ("oracle_h1", "greedy"), ("oracle_h1", "milp_track"),
                   ("greedy", "cspf")):
        if a_ in refs and b_ in refs:
            j = ref_test(ep, a_).merge(ref_test(ep, b_), on=["scenario", "seed"],
                                       suffixes=("_a", "_b"))
            j["diff"] = j.operational_return_a - j.operational_return_b
            rr.append({"a": a_, "b": b_, **compare_fixed(j)})
    if rr:
        write_table(pd.DataFrame(rr), "references_pairwise")


def section_horizon(data: dict) -> None:
    ep = data.get("episodes_test")
    if ep is None:
        return
    sel = ep[(ep.kind == "learner") & (ep.role == "selected")]
    roots = (42, 314159, 271828)
    rows = []
    fams = [("Q", 0.0, "bandit"), ("Q", 0.5, "E2:qg05"), ("Q", 0.9, "E2:qg09"),
            ("Q", 0.99, "E2:qg099"), ("PPO", 0.0, "E2:ppog0"), ("PPO", 0.9, "E2:ppog09"),
            ("PPO", 0.995, "ppo")]
    base = sel[(sel.policy == "bandit") & sel.root.isin(roots)]
    for fam, gamma, pol in fams:
        df = sel[(sel.policy == pol) & sel.root.isin(roots)]
        if df.root.nunique() == 0:
            continue
        g = df.groupby("root").operational_return.mean()
        j = df.merge(base, on=["root", "scenario", "seed"], suffixes=("_a", "_b"))
        j["diff"] = j.operational_return_a - j.operational_return_b
        cl = compare_learners(j[["root", "scenario", "seed", "diff"]]) if len(j) else {}
        rows.append({"family": fam, "gamma": gamma, "policy": pol, "roots": len(g),
                     "mean": g.mean(), "root_sd": g.std(ddof=1) if len(g) > 1 else np.nan,
                     "vs_bandit": cl.get("boot_est", np.nan), "vs_bandit_lo": cl.get("boot_lo"),
                     "vs_bandit_hi": cl.get("boot_hi"),
                     "reroutes": df.reroutes_per_hour.mean(), "noop": df.noop_frequency.mean()})
    if len(rows) > 2:
        h = pd.DataFrame(rows)
        write_table(h, "horizon_sweep", latex_rows([[
            r.family, f"{r.gamma:g}", str(r.roots), f"{r.mean:.1f}",
            "--" if np.isnan(r.vs_bandit) or (r.family == "Q" and r.gamma == 0)
            else ci_str(r.vs_bandit, r.vs_bandit_lo, r.vs_bandit_hi),
            f"{r.reroutes:.2f}"] for r in h.itertuples()]))
        for r in h.itertuples():
            tag = f"{r.family}{str(r.gamma).replace('.', 'p')}"
            num(f"H{tag}", r.mean)
            if not np.isnan(r.vs_bandit):
                num(f"H{tag}Vs", r.vs_bandit)
                num(f"H{tag}VsLo", r.vs_bandit_lo)
                num(f"H{tag}VsHi", r.vs_bandit_hi)


def section_tuning(data: dict) -> None:
    cur = data.get("validation_curves")
    ep = data.get("episodes_test")
    if cur is None or cur.empty:
        return
    t = cur[(cur.seedset == "validation") & ((cur.family == "E3_ppo_tuning") |
                                             ((cur.policy == "ppo") & (cur.root == 42)))]
    if t.family.eq("E3_ppo_tuning").sum() == 0:
        print("skip tuning: no E3 runs")
        return
    best = t.groupby(["policy", "run_id"]).mean_return.max().reset_index()
    best = best.sort_values("mean_return", ascending=False)
    if ep is not None:
        sel = ep[(ep.kind == "learner") & (ep.role == "selected") & (ep.root == 42)]
        best["test_mean"] = best.policy.map(sel.groupby("policy").operational_return.mean())
    b42 = cur[(cur.seedset == "validation") & (cur.policy == "bandit") & (cur.root == 42)]
    num("TuneBanditValBest", b42.mean_return.max() if len(b42) else float("nan"))
    write_table(best, "ppo_tuning", latex_rows([[
        r.policy.replace("E3:", "").replace("_", "\\_"), f"{r.mean_return:.1f}",
        f"{r.test_mean:.1f}" if "test_mean" in best and not pd.isna(r.test_mean) else "--"]
        for r in best.itertuples()]))
    num("TuneBest", best.policy.iloc[0].replace("E3:", ""))
    num("TuneBestVal", best.mean_return.iloc[0])
    num("TuneConfigs", int(best.policy.nunique()), "{}")


def section_delay(data: dict) -> None:
    ep = data.get("episodes_test")
    if ep is None:
        return
    sel = ep[(ep.kind == "learner") & (ep.role == "selected") & (ep.family == "E4_delay")]
    if sel.empty:
        print("skip delay: no E4 runs")
        return
    rows = []
    for pol, g in sel.groupby("policy"):
        r = g.groupby("root").operational_return.mean()
        rows.append({"policy": pol, "roots": len(r), "mean": r.mean(),
                     "reroutes": g.reroutes_per_hour.mean(), "noop": g.noop_frequency.mean()})
    refdir = RAW / "references_delay1"
    for f in sorted(refdir.glob("*__seed*.json")) if refdir.exists() else []:
        pass
    if refdir.exists():
        sys.path.insert(0, str(ROOT / "scripts/study"))
        from run_oracles import load_dir
        d = load_dir(refdir)
        for pol, g in d.groupby("algorithm"):
            rows.append({"policy": f"ref:{pol}", "roots": 0, "mean": g.operational_return.mean(),
                         "reroutes": g.reroutes_per_hour.mean(), "noop": g.noop_frequency.mean()})
    out = pd.DataFrame(rows)
    write_table(out, "delay_results")
    for r in out.itertuples():
        key = r.policy.replace("E4:L1_", "").replace("ref:", "Ref").replace("_", "")
        num(f"Delay{key}", r.mean)


def section_seqdiag() -> None:
    rows = []
    for name, frozen in (("seqdiag_greedy", False), ("seqdiag_greedy_frozen", True)):
        files = glob.glob(str(RAW / name / "*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files])
        for scope, g in [("all", df)] + list(df.groupby("scenario")):
            for h in (1, 2, 3, 6, 12, 24):
                if f"agree_h{h}" not in g:
                    continue
                gg = g.dropna(subset=[f"best_delta_h{h}"])
                gain = gg[f"best_delta_h{h}"].mean()
                rows.append({"frozen": frozen, "scope": scope, "H": h, "states": len(gg),
                             "agree": gg[f"agree_h{h}"].mean(),
                             "regret": gg[f"myopic_regret_h{h}"].mean(),
                             "best_gain": gain,
                             "captured": 1 - gg[f"myopic_regret_h{h}"].mean() / gain if gain > 0 else 1.0,
                             "spearman": gg[f"spearman_h1_h{h}"].mean() if h > 1 else 1.0,
                             "best_is_sacrifice": gg[f"best_is_sacrifice_h{h}"].mean(),
                             "best_is_noop": (gg[f"best_a_h{h}"] == 0).mean()})
    if not rows:
        print("skip seqdiag")
        return
    s = pd.DataFrame(rows)
    write_table(s, "seqdiag_summary")
    for frozen, tag in ((False, "Live"), (True, "Frozen")):
        a = s[(s.frozen == frozen) & (s.scope == "all")]
        for h in (1, 3, 6, 12, 24):
            r = a[a.H == h]
            if len(r):
                num(f"Seq{tag}AgreeH{h}", 100 * r.agree.iloc[0], "{:.0f}")
                num(f"Seq{tag}CapturedH{h}", 100 * r.captured.iloc[0], "{:.0f}")
                num(f"Seq{tag}SacrificeH{h}", 100 * r.best_is_sacrifice.iloc[0], "{:.0f}")
                num(f"Seq{tag}States", int(r.states.iloc[0]), "{}")


def section_compute(data: dict) -> None:
    runs = data.get("runs")
    if runs is None or runs.empty:
        return
    main = runs[runs.family.isin(["E0_repro", "E1_main"])]
    rows = []
    for pol, g in main.groupby("policy"):
        rows.append({"policy": pol, "runs": len(g), "wall_min_mean": g.wall_time_seconds.mean() / 60,
                     "tps_mean": g.transitions_per_second.mean(),
                     "parameters": g.parameters.dropna().mean() if g.parameters.notna().any() else np.nan})
    c = pd.DataFrame(rows)
    write_table(c, "compute")
    for r in c.itertuples():
        k = {"bandit": "Bandit", "ppo": "PPO"}.get(r.policy, r.policy)
        num(f"Wall{k}", r.wall_min_mean, "{:.0f}")
        num(f"Tps{k}", r.tps_mean, "{:.0f}")


def section_environment() -> None:
    """Environment specification tables generated from the YAML configs."""
    import yaml
    cfg = ROOT / "configs"
    topo = yaml.safe_load((cfg / "topology.yaml").read_text())
    tc = yaml.safe_load((cfg / "traffic_classes.yaml").read_text())
    sc = yaml.safe_load((cfg / "scenarios.yaml").read_text())["scenarios"]
    caps = [l["capacity_mbps"] for l in topo["links"]]
    roles = pd.Series([r["role"] for r in topo["routers"]]).value_counts()
    num("NRouters", len(topo["routers"]), "{}")
    num("NLinks", len(topo["links"]), "{}")
    num("NDirLinks", 2 * len(topo["links"]), "{}")
    num("NDemands", len(tc["demands"]), "{}")
    num("TotalCapacityGbps", 2 * sum(caps) / 1000, "{:.1f}")
    num("PeakOfferedGbps", sum(d["base_mbps"] for d in tc["demands"]) / 1000, "{:.2f}")
    counts = pd.Series([d["class"] for d in tc["demands"]]).value_counts()
    rows = []
    for name, c in tc["classes"].items():
        rows.append([name, str(c["priority"]), str(c["max_latency_ms"]), f"{c['max_loss_pct']:g}",
                     "yes" if c["protected"] else "no", c["profile"].replace("_", "\\_"),
                     str(int(counts.get(name, 0)))])
    write_table(pd.DataFrame(rows, columns=["class", "priority", "delay_ms", "loss_pct", "protected",
                                            "profile", "demands"]), "env_classes", latex_rows(rows))
    rows = []
    for name in EVAL_SCENARIOS + ("random_day",):
        s_ = sc[name]
        ev = s_.get("events", [])
        kinds = ", ".join(sorted({e["type"].replace("_", " ") for e in ev})) or (
            "randomized" if s_.get("randomize") else "none")
        rows.append([name.replace("_", "\\_"), f"{s_['start_hour']:g}",
                     str(s_["duration_min"] // 5), f"{s_['demand_multiplier']:g}",
                     f"{s_['noise_sigma']:g}", kinds])
    write_table(pd.DataFrame(rows, columns=["scenario", "start_h", "decisions", "multiplier",
                                            "noise", "events"]), "env_scenarios", latex_rows(rows))
    write_table(pd.DataFrame([{"role": k, "routers": v} for k, v in roles.items()]), "env_roles")


def _tex_escape(text: str) -> str:
    import re
    text = re.sub(r"`([^`]*)`", r"\\texttt{\1}", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\\textbf{\1}", text)
    for a, b in (("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("_", "\\_"), ("≤", "$\\le$"),
                 ("≈", "$\\approx$"), ("→", "$\\to$"), ("×", "$\\times$"), ("–", "--"),
                 ("—", "---"), ("·", "$\\cdot$"), ("γ", "$\\gamma$")):
        text = text.replace(a, b)
    return text.replace("\\texttt{", "\\texttt{").replace("\\_}", "_}")


def section_doc_tables() -> None:
    gen = ROOT / "docs" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (ROOT / "docs" / "EXPERIMENT_LOG.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].isdigit():
            rows.append(cells)
    body = "\n".join(f"{c[0]} & {_tex_escape(c[2])} & {_tex_escape(c[3])} \\\\" for c in rows)
    (gen / "experiment_log_table.tex").write_text(
        "{\\small\\begin{longtable}{@{}rp{0.5\\textwidth}p{0.38\\textwidth}@{}}\\toprule\n"
        "\\# & Event & Outcome \\\\\\midrule\n" + body + "\n\\bottomrule\\end{longtable}}\n")
    f = RAW / "mask_audit" / "summary.json"
    if f.exists():
        d = json.loads(f.read_text())
        c, v = d["counts"], d["violations"]
        lines = [f"{k.replace('_', ' ')} & {int(val):,} \\\\" for k, val in c.items()]
        lines += ["\\midrule"] + [f"violation: {k.replace('_', ' ')} & {int(val)} \\\\"
                                     for k, val in v.items()]
        (gen / "mask_audit_table.tex").write_text(
            "\\begin{center}\\small\\begin{tabular}{@{}lr@{}}\\toprule\n" + "\n".join(lines)
            + "\n\\bottomrule\\end{tabular}\\end{center}\n")
        num("MaskStates", int(c["states"]))
        num("MaskChecks", int(c["actions_checked"]))
        num("MaskLegalApplied", int(c["legal_te_actions"]))
        num("MaskProtectedMoves", int(c["protected_legal_moves"]))
        num("MaskViolations", int(sum(v.values())), "{}")
        num("MaskFailedLinkStates", int(c.get("states_with_failed_link", 0)))
    else:
        (gen / "mask_audit_table.tex").write_text("(mask audit pending)\n")


def write_numbers() -> None:
    out = ROOT / "paper" / "generated" / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated by scripts/study/analyze.py -- do not edit."]
    for k, v in sorted(NUMBERS.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out.write_text("\n".join(lines) + "\n")
    (PROCESSED / "numbers.json").write_text(json.dumps(NUMBERS, indent=1, sort_keys=True))
    print(f"wrote {len(NUMBERS)} macros to {out}")


def main() -> None:
    data = collect(copy=True)
    for f in (section_reproduction, section_main, section_horizon, section_tuning,
              section_delay, section_compute):
        try:
            f(data)
        except Exception as exc:  # keep going; report clearly
            print(f"!! {f.__name__} failed: {type(exc).__name__}: {exc}")
            raise
    section_seqdiag()
    section_environment()
    section_doc_tables()
    write_numbers()


if __name__ == "__main__":
    main()
