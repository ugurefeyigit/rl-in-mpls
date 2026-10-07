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


_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
          "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", "12": "Twelve", "24": "TwentyFour",
          "05": "PointFive", "09": "PointNine", "099": "PointNineNine", "0995": "Default"}


def macro_name(name: str) -> str:
    """LaTeX control words may contain letters only: spell out digit runs."""
    import re
    name = name.replace("0p995", "Default").replace("0p99", "PointNineNine") \
               .replace("0p9", "PointNine").replace("0p5", "PointFive").replace("0p0", "Zero")
    out = re.sub(r"\d+", lambda m: _WORDS.get(m.group(0), "N" + "".join(
        _WORDS[c] for c in m.group(0))), name)
    if not re.fullmatch(r"[A-Za-z]+", out):
        raise ValueError(f"cannot make a LaTeX macro name from {name!r}")
    return out


def num(name: str, value: float | int | str, fmt: str = "{:.1f}") -> None:
    """Register a LaTeX macro \\<name> holding a formatted result."""
    name = macro_name(name)
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
    for pol, k in (("bandit", "Bandit"), ("ppo", "PPO")):
        num(f"ReproMaxShift{k}", rep[rep.policy == pol].holdout_diff.abs().max(), "{:.0f}")
    br = PROCESSED / "baseline_reproduction.csv"
    if br.exists():
        b = pd.read_csv(br)
        num("BaselineReproCells", len(b), "{}")
        worst = b[[c for c in b.columns if c.startswith("abs_diff")]].to_numpy().max()
        mant, exp = f"{worst:.0e}".split("e")
        num("BaselineReproMaxDiff", f"{mant}\\cdot10^{{{int(exp)}}}")
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
    write_table(ps, "reproduction_per_scenario", latex_rows([[
        SCEN_LABEL[r.scenario], f"{r.hist_diff:+.1f}", ci_str(r.diff_est, r.diff_lo, r.diff_hi)]
        for r in ps.itertuples()]))
    d = ps.set_index("scenario").loc["deceptive_local_optimum"]
    num("ReproDeceptive", d.diff_est)
    num("ReproDeceptiveLo", d.diff_lo)
    num("ReproDeceptiveHi", d.diff_hi)
    num("HistDeceptive", d.hist_diff, "{:+.1f}")
    cur = data["validation_curves"]
    cc = cur[(cur.family == "E0_repro") & (cur.seedset == "historical_continuity")]
    best = cc.groupby(["policy", "root"]).mean_return.max()
    num("ReproPPONeverPositive", int((best.loc["ppo"] < 0).sum()), "{}")
    for r in rep.itertuples():
        tag = {42: "A", 314159: "B", 271828: "C"}[int(r.root)]
        k = "Bandit" if r.policy == "bandit" else "PPO"
        num(f"Repro{k}Root{tag}", r.holdout_return_repro)
        num(f"Hist{k}Root{tag}", r.holdout_return_hist)
    write_table(rep, "reproduction_per_root", latex_rows([[
        ("Bandit" if r.policy == "bandit" else "PPO"), str(int(r.root)),
        f"{r.selected_ckpt_hist // 1000}k / {r.selected_ckpt_repro // 1000}k",
        f"{r.holdout_return_hist:.1f}", f"{r.holdout_return_repro:.1f}"]
        for r in rep.sort_values(["policy", "root"]).itertuples()]))


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
        fr = fin.groupby(["policy", "root"]).operational_return.mean().unstack(0)
        sc = sel.groupby(["policy", "root"]).checkpoint.first().unstack(0)
        write_table(pr.reset_index(), "main_per_root", latex_rows([[
            str(int(r)), f"{int(sc.loc[r, 'bandit']) // 1000}k", f"{pr.loc[r, 'bandit']:.1f}",
            f"{fr.loc[r, 'bandit']:.1f}", f"{int(sc.loc[r, 'ppo']) // 1000}k",
            f"{pr.loc[r, 'ppo']:.1f}", f"{fr.loc[r, 'ppo']:.1f}"]
            for r in pr.index if r in sc.index and not pd.isna(pr.loc[r, "ppo"])]))
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
        lv = pd.DataFrame(comps)
        write_table(lv, "learners_vs_references", latex_rows([[
            POLICY_LABEL[r.learner], POLICY_LABEL.get(r.reference, r.reference),
            ci_str(r.boot_est, r.boot_lo, r.boot_hi), f"{int(r.roots_positive)}/{int(r.roots)}"]
            for r in lv.itertuples()]))
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


UTIL_COMPONENTS = ["rc_delivery", "rc_protected_disconnect", "rc_unprotected_disconnect",
                   "rc_sla_severity", "rc_max_util", "rc_overload"]
COST_COMPONENTS = ["rc_move_fixed", "rc_move_volume", "rc_move_divergence", "rc_reversal",
                   "rc_invalid"]


def section_decomposition(data: dict) -> None:
    """Return = network utility + shaping + move costs, per policy instance."""
    ep = data.get("episodes_test")
    if ep is None:
        return
    s = ep[(ep.role == "selected") | (ep.kind != "learner")].copy()
    s["utility"] = s[UTIL_COMPONENTS].sum(axis=1)
    s["costs"] = s[COST_COMPONENTS].sum(axis=1)
    s["shaping"] = s["rc_potential"]
    g = s.groupby(["policy", "root"]).agg(
        ret=("operational_return", "mean"), utility=("utility", "mean"), shaping=("shaping", "mean"),
        costs=("costs", "mean"), reversal_cost=("rc_reversal", "mean"),
        moves=("accepted_te_changes", "mean"), reversals=("te_reversals", "mean"),
        noop=("noop_frequency", "mean")).reset_index()
    g["reversal_share"] = g.reversals / g.moves.where(g.moves > 0)
    write_table(g, "decomposition_by_root")
    pol = g.groupby("policy")[["ret", "utility", "shaping", "costs", "moves", "reversals"]].mean()
    order = [p for p in ("bandit", "ppo", "milp_track", "greedy", "cspf", "static", "noop",
                         "oracle_h1") if p in pol.index]
    write_table(pol.loc[order].reset_index(), "decomposition", latex_rows([[
        POLICY_LABEL.get(p, p), f"{pol.loc[p, 'ret']:.1f}", f"{pol.loc[p, 'utility']:.1f}",
        f"{pol.loc[p, 'costs']:.1f}", f"{pol.loc[p, 'shaping']:.2f}", f"{pol.loc[p, 'moves']:.1f}",
        f"{pol.loc[p, 'reversals']:.1f}"] for p in order]))
    for p, k in (("bandit", "Bandit"), ("ppo", "PPO"), ("milp_track", "Milp")):
        if p in pol.index:
            num(f"Util{k}", pol.loc[p, "utility"])
            num(f"Cost{k}", pol.loc[p, "costs"])
            num(f"Rev{k}", pol.loc[p, "reversals"])
            num(f"Moves{k}", pol.loc[p, "moves"])
    b = g[g.policy == "bandit"]
    num("ShapingMaxAbs", s.shaping.abs().groupby(s.policy).mean().max(), "{:.2f}")
    pp = g[g.policy == "ppo"]
    if len(pp):
        num("PPORevShareMax", 100 * pp.reversal_share.max(), "{:.0f}")
        num("PPOCostMin", pp.costs.min())
        num("PPOCostMax", pp.costs.max())
        num("PPOUtilMax", pp.utility.max())
        num("BanditUtilMax", b.utility.max())
        num("BanditCostMin", b.costs.min())
        num("BanditCostMax", b.costs.max())
        num("PPOFlappingRoots", int((pp.reversal_share > 0.5).sum()), "{}")


def section_fidelity() -> None:
    files = sorted((RAW / "model_fidelity").glob("*__r*.csv"))
    files = [f for f in files if not f.name.endswith(".h1only.csv")]
    if not files:
        return
    rows = []
    for f in files:
        d = pd.read_csv(f)
        if "g6_policy" not in d:
            continue
        rid = f.stem
        rows.append({"run_id": rid, "policy": "bandit" if "__bandit__" in rid else "ppo",
                     "root": int(rid.rsplit("__r", 1)[1]), "states": len(d),
                     "top1": d.agree_top1.mean(), "regret1": (d.r_best - d.r_policy).mean(),
                     "noop_policy": d.policy_is_noop.mean(), "noop_optimal": d.best_is_noop.mean(),
                     "g6_policy": (d.g6_policy - d.g6_noop).mean(),
                     "g6_myopic": (d.g6_myopic_best - d.g6_noop).mean(),
                     "spearman": d.spearman.mean() if "spearman" in d else np.nan})
    if not rows:
        return
    t = pd.DataFrame(rows).sort_values(["policy", "root"])
    write_table(t, "model_fidelity", latex_rows([[
        ("Bandit" if r.policy == "bandit" else "PPO"), str(r.root), f"{100 * r.top1:.0f}\\,\\%",
        f"{r.regret1:.2f}", f"{100 * r.noop_policy:.0f} / {100 * r.noop_optimal:.0f}\\,\\%",
        f"{r.g6_policy:+.2f}", f"{r.g6_myopic:+.2f}"] for r in t.itertuples()]))
    for pol, k in (("bandit", "Bandit"), ("ppo", "PPO")):
        g = t[t.policy == pol]
        if len(g):
            num(f"Fid{k}TopOne", 100 * g.top1.mean(), "{:.0f}")
            num(f"Fid{k}Regret", g.regret1.mean(), "{:.2f}")
            num(f"Fid{k}NoopPolicy", 100 * g.noop_policy.mean(), "{:.0f}")
            num(f"Fid{k}NoopOptimal", 100 * g.noop_optimal.mean(), "{:.0f}")
            num(f"Fid{k}GSix", g.g6_policy.mean(), "{:+.2f}")
            num(f"Fid{k}GSixMyopic", g.g6_myopic.mean(), "{:+.2f}")
            num(f"Fid{k}Roots", len(g), "{}")
            num(f"Fid{k}NegRoots", int((g.g6_policy < 0).sum()), "{}")


def section_final_flapping(data: dict) -> None:
    ep = data.get("episodes_test")
    if ep is None:
        return
    f = ep[(ep.policy == "ppo") & (ep.role == "final")]
    if f.empty:
        return
    g = f.groupby("root").agg(moves=("accepted_te_changes", "mean"), rev=("te_reversals", "mean"))
    share = g.rev / g.moves.where(g.moves > 0)
    num("PPOFinalFlappingRoots", int((share > 0.5).sum()), "{}")
    num("PPOFinalRoots", len(g), "{}")


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
                     "reroutes": df.reroutes_per_hour.mean(), "noop": df.noop_frequency.mean(),
                     "reversals": df.te_reversals.mean(),
                     "roots_better": int((g > base[base.root.isin(g.index)]
                                          .groupby("root").operational_return.mean()
                                          .reindex(g.index)).sum())})
    if len(rows) > 2:
        h = pd.DataFrame(rows)
        write_table(h, "horizon_sweep", latex_rows([[
            ("Q-learner" if r.family == "Q" else "PPO") + (" (bandit)" if r.family == "Q" and r.gamma == 0 else ""),
            f"{r.gamma:g}", str(r.roots), f"{r.mean:.1f}",
            "--" if np.isnan(r.vs_bandit) or (r.family == "Q" and r.gamma == 0)
            else ci_str(r.vs_bandit, r.vs_bandit_lo, r.vs_bandit_hi),
            f"{r.reroutes:.2f}", f"{r.reversals:.1f}"] for r in h.itertuples()]))
        for r in h.itertuples():
            tag = f"{r.family}{str(r.gamma).replace('.', 'p')}"
            num(f"H{tag}", r.mean)
            num(f"H{tag}Rev", r.reversals)
            num(f"H{tag}Roots", int(r.roots), "{}")
            num(f"H{tag}RootsBetter", int(r.roots_better), "{}")
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
    sel = ep[(ep.kind == "learner") & (ep.role == "selected")]
    d1 = sel[sel.family == "E4_delay"]
    if d1.empty:
        print("skip delay: no E4 runs")
        return
    roots = sorted(d1.root.unique())
    l0 = {"bandit": sel[sel.policy == "bandit"], "ppo": sel[sel.policy == "ppo"],
          "qg09": sel[sel.policy == "E2:qg09"]}
    rows = []
    for tag, pol in (("bandit", "E4:L1_bandit"), ("ppo", "E4:L1_ppo"), ("qg09", "E4:L1_qg09")):
        g = d1[d1.policy == pol]
        if g.empty:
            continue
        rr = g.groupby("root").operational_return.mean()
        base = l0[tag][l0[tag].root.isin(rr.index)].groupby("root").operational_return.mean()
        row = {"policy": tag, "roots": len(rr), "mean_L1": rr.mean(),
               "mean_L0_same_roots": base.mean() if len(base) else np.nan,
               "reroutes": g.reroutes_per_hour.mean(), "reversals": g.te_reversals.mean()}
        if tag != "bandit":
            b = d1[d1.policy == "E4:L1_bandit"]
            j2 = g.merge(b, on=["root", "scenario", "seed"], suffixes=("_a", "_b"))
            j2["diff"] = j2.operational_return_a - j2.operational_return_b
            if len(j2):
                cl = compare_learners(j2[["root", "scenario", "seed", "diff"]])
                row.update(vs_bandit=cl["boot_est"], vs_bandit_lo=cl["boot_lo"],
                           vs_bandit_hi=cl["boot_hi"], roots_better=cl["roots_positive"])
        rows.append(row)
    refdir = RAW / "references_delay1"
    if refdir.exists():
        sys.path.insert(0, str(ROOT / "scripts/study"))
        from run_oracles import load_dir
        dd = load_dir(refdir)
        base0 = ep[(ep.kind != "learner")]
        for pol, g in dd.groupby("algorithm"):
            rows.append({"policy": f"ref:{pol}", "roots": 0, "mean_L1": g.operational_return.mean(),
                         "mean_L0_same_roots": base0[base0.policy == pol].operational_return.mean(),
                         "reroutes": g.reroutes_per_hour.mean(), "reversals": g.te_reversals.mean()})
    out = pd.DataFrame(rows)
    write_table(out, "delay_results")
    lab = {"bandit": "Masked bandit", "ppo": "MaskablePPO ($\\gamma=0.995$)",
           "qg09": "Q-learner ($\\gamma=0.9$)", "ref:milp_track": "MILP-track",
           "ref:greedy": "Greedy", "ref:cspf": "CSPF", "ref:static": "Static SP", "ref:noop": "No-op"}
    lat = []
    for r in out.itertuples():
        if r.policy not in lab:
            continue
        vs = (ci_str(r.vs_bandit, r.vs_bandit_lo, r.vs_bandit_hi)
              if "vs_bandit" in out and not pd.isna(getattr(r, "vs_bandit", np.nan)) else "--")
        lat.append([lab[r.policy], str(int(r.roots)) if r.roots else "--", f"{r.mean_L0_same_roots:.1f}",
                    f"{r.mean_L1:.1f}", vs, f"{r.reroutes:.2f}"])
    (PTABLES / "delay_results.tex").write_text(latex_rows(lat))
    for r in out.itertuples():
        key = {"bandit": "Bandit", "ppo": "PPO", "qg09": "QNine"}.get(
            r.policy, r.policy.replace("ref:", "Ref").replace("_", "").capitalize())
        num(f"Delay{key}", r.mean_L1)
        num(f"Delay{key}LZero", r.mean_L0_same_roots)
        if "vs_bandit" in out and not pd.isna(getattr(r, "vs_bandit", np.nan)):
            num(f"Delay{key}Vs", r.vs_bandit)
            num(f"Delay{key}VsLo", r.vs_bandit_lo)
            num(f"Delay{key}VsHi", r.vs_bandit_hi)
            num(f"Delay{key}RootsBetter", int(r.roots_better), "{}")
            num(f"Delay{key}Roots", int(r.roots), "{}")

def section_oracle_ladder(data: dict) -> None:
    ep = data.get("episodes_test")
    if ep is None:
        return
    seeds = list(range(3001, 3006))
    refs = ep[(ep.kind != "learner") & ep.seed.isin(seeds)]
    have = [p for p in ("oracle_h1", "oracle_h3", "oracle_h6") if (refs.policy == p).sum() == 35]
    if len(have) < 2:
        print("skip oracle ladder: oracle H>1 incomplete")
        return
    rows = []
    sel = ep[(ep.kind == "learner") & (ep.role == "selected") & ep.seed.isin(seeds)]
    for p in ("noop", "greedy", "milp_track", *have):
        d = refs[refs.policy == p]
        rows.append({"policy": p, "mean": d.operational_return.mean(), "n": len(d)})
    for p in ("bandit", "ppo"):
        d = sel[sel.policy == p]
        rows.append({"policy": p, "mean": d.groupby("root").operational_return.mean().mean(),
                     "n": len(d)})
    lad = pd.DataFrame(rows)
    base = refs[refs.policy == "oracle_h1"].set_index(["scenario", "seed"]).operational_return
    for p in have[1:]:
        d = refs[refs.policy == p].set_index(["scenario", "seed"]).operational_return
        diff = (d - base).dropna().reset_index(name="diff")
        ci = stratified_bootstrap_mean(diff["diff"].to_numpy(), diff.scenario.to_numpy())
        h = p[-1]
        num(f"Oracle{h}MinusOne", ci.estimate)
        num(f"Oracle{h}MinusOneLo", ci.low)
        num(f"Oracle{h}MinusOneHi", ci.high)
        lad.loc[lad.policy == p, "minus_h1"] = ci.estimate
    for r in lad.itertuples():
        num("Ladder" + r.policy.replace("_", "").replace("oracleh", "OracleH").capitalize(), r.mean)
    write_table(lad, "oracle_ladder")


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
    # Markdown block for docs/SEQUENTIALITY_AUDIT.md
    md = ["| Rollout | H | States | Agreement | Gain captured | Best move is a sacrifice | Best is no-op |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in s[s.scope == "all"].sort_values(["frozen", "H"]).itertuples():
        md.append(f"| {'frozen exogenous' if r.frozen else 'clairvoyant'} | {r.H} | {r.states} | "
                  f"{100 * r.agree:.0f} % | {100 * r.captured:.0f} % | {100 * r.best_is_sacrifice:.0f} % | "
                  f"{100 * r.best_is_noop:.0f} % |")
    md += ["", "Per scenario at H = 24 (agreement / gain captured):", "",
           "| Scenario | Clairvoyant | Frozen |", "|---|---:|---:|"]
    h24 = s[(s.H == 24) & (s.scope != "all")]
    for sc in EVAL_SCENARIOS:
        cells = []
        for fr in (False, True):
            r = h24[(h24.scope == sc) & (h24.frozen == fr)]
            cells.append(f"{100 * r.agree.iloc[0]:.0f} % / {100 * r.captured.iloc[0]:.0f} %"
                         if len(r) else "pending")
        md.append(f"| {sc} | {cells[0]} | {cells[1]} |")
    lat = []
    for sc in EVAL_SCENARIOS:
        cells = [SCEN_LABEL[sc]]
        for fr in (False, True):
            r = h24[(h24.scope == sc) & (h24.frozen == fr)]
            cells += ([f"{100 * r.agree.iloc[0]:.0f}", f"{100 * r.captured.iloc[0]:.0f}",
                       f"{100 * r.best_is_sacrifice.iloc[0]:.0f}"] if len(r) else ["--"] * 3)
        cells.insert(1, str(int(h24[(h24.scope == sc) & (~h24.frozen)].states.iloc[0]))
                     if len(h24[(h24.scope == sc) & (~h24.frozen)]) else "--")
        lat.append(cells)
    (PTABLES / "seqdiag_per_scenario.tex").write_text(latex_rows(lat))
    doc = ROOT / "docs" / "SEQUENTIALITY_AUDIT.md"
    text = doc.read_text()
    a, b = "<!-- SEQ:BEGIN -->", "<!-- SEQ:END -->"
    if a in text:
        pre, rest = text.split(a, 1)
        doc.write_text(pre + a + "\n" + "\n".join(md) + "\n" + b + rest.split(b, 1)[1])
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
    bf = RAW / "benchmark" / "inference.json"
    if bf.exists():
        b = json.loads(bf.read_text())["results"]
        lab = {"masked_bandit": "Masked bandit", "maskable_ppo": "MaskablePPO",
               "milp_track": "MILP-track", "greedy": "Greedy", "cspf": "CSPF",
               "static": "Static SP", "oracle_h1": "Oracle-1$^\\dagger$"}
        wall = {"masked_bandit": c.set_index("policy").wall_min_mean.get("bandit"),
                "maskable_ppo": c.set_index("policy").wall_min_mean.get("ppo")}
        tps = {"masked_bandit": c.set_index("policy").tps_mean.get("bandit"),
               "maskable_ppo": c.set_index("policy").tps_mean.get("ppo")}
        rows_l = []
        for k in ("masked_bandit", "maskable_ppo", "milp_track", "greedy", "cspf", "static", "oracle_h1"):
            v = b[k]
            rows_l.append([lab[k], f"{v['parameters']:,}".replace(",", "{,}") if v["parameters"] else "--",
                           f"{wall[k]:.0f}" if k in wall and wall[k] == wall[k] else "--",
                           f"{tps[k]:.0f}" if k in tps and tps[k] == tps[k] else "--",
                           f"{v['mean_ms']:.3g}", f"{v['p95_ms']:.3g}"])
            key = {"masked_bandit": "Bandit", "maskable_ppo": "PPO", "milp_track": "Milp",
                   "greedy": "Greedy", "cspf": "Cspf", "static": "Static", "oracle_h1": "OracleOne"}[k]
            num(f"Inf{key}Ms", v["mean_ms"], "{:.3g}")
            if v["parameters"]:
                num(f"Params{key}", int(v["parameters"]))
        (PTABLES / "compute.tex").write_text(latex_rows(rows_l))
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


ARTIFACTS = [
    # (paper id, output, script, inputs, experiments, seeds)
    ("Fig. topology", "results/figures/paper/fig_topology.pdf", "scripts/make_paper_figures.py topology",
     "configs/topology.yaml", "--", "--"),
    ("Tab. classes / scenarios", "paper/tables/env_classes.tex, env_scenarios.tex",
     "scripts/study/analyze.py (section_environment)", "configs/traffic_classes.yaml, configs/scenarios.yaml",
     "--", "--"),
    ("Tab. reproduction", "paper/tables/reproduction_per_root.tex, results/tables/reproduction_*.csv",
     "scripts/study/analyze.py (section_reproduction)",
     "experiments/raw/learner_eval/E0_repro__*/historical_*.csv, results/v2_final_holdout/*.csv",
     "E0_repro (6 runs)", "select 101-105, test 1001-1005"),
    ("Baseline exact reproduction", "experiments/processed/baseline_reproduction.csv",
     "scripts/study/check_baseline_reproduction.py", "experiments/raw/references_historical_holdout/",
     "--", "1001-1005"),
    ("Tab. main results, Fig. main", "paper/tables/main_results.tex, main_per_scenario.tex, fig_main.pdf",
     "scripts/study/analyze.py (section_main); scripts/make_paper_figures.py main",
     "experiments/processed/episodes_test.csv", "E0_repro, E1_main, references", "select 2001-2005, test 3001-3020"),
    ("Fig. learning curves", "results/figures/paper/fig_learning_curves.pdf", "scripts/make_paper_figures.py curves",
     "experiments/processed/validation_curves.csv", "E0_repro, E1_main", "2001-2005"),
    ("Tab./Fig. horizon sweep", "paper/tables/horizon_sweep.tex, fig_horizon.pdf",
     "scripts/study/analyze.py (section_horizon); make_paper_figures.py horizon",
     "experiments/processed/episodes_test.csv", "E0_repro, E2_horizon", "3001-3020"),
    ("Tab. PPO tuning", "paper/tables/ppo_tuning.tex", "scripts/study/analyze.py (section_tuning)",
     "experiments/processed/validation_curves.csv, episodes_test.csv", "E3_ppo_tuning, E3b_ppo_best",
     "select 2001-2005, test 3001-3020"),
    ("Tab./Fig. delay", "results/tables/delay_results.csv, fig_delay.pdf", "scripts/study/analyze.py (section_delay)",
     "experiments/processed/episodes_test.csv, experiments/raw/references_delay1/", "E4_delay", "3001-3020"),
    ("Fig. sequentiality", "results/figures/paper/fig_seqdiag.pdf, results/tables/seqdiag_summary.csv",
     "scripts/study/run_seqdiag.py; analyze.py (section_seqdiag); make_paper_figures.py seqdiag",
     "experiments/raw/seqdiag_greedy*/", "--", "4001-4003"),
    ("Mask audit", "experiments/raw/mask_audit/summary.json, docs/generated/mask_audit_table.tex",
     "scripts/study/run_mask_audit.py", "--", "--", "5000-5005, 6000-6039"),
    ("Tab. compute", "results/tables/compute.csv", "scripts/study/analyze.py (section_compute)",
     "experiments/processed/runs.csv", "E0_repro, E1_main", "--"),
    ("All in-text numbers", "paper/generated/numbers.tex (+ experiments/processed/numbers.json)",
     "scripts/study/analyze.py", "all of the above", "all", "all"),
]


def section_manifest() -> None:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    lines = ["# Results manifest", "",
             "*Generated by `scripts/study/analyze.py`; do not edit by hand.* Every number, table "
             "and figure in `paper/` and `report/` is produced by the script listed here from the "
             "listed inputs. Inputs under `experiments/raw/` are versioned; checkpoints are not "
             "(see `docs/REPRODUCIBILITY.md`).", "",
             f"Analysis run at commit `{commit[:12]}` (the commit *before* the regenerated outputs "
             "were committed).", "",
             "| Paper artifact | Output | Produced by | Inputs | Experiments | Seeds |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {a} | `{b}` | `{c}` | `{d}` | {e} | {f} |" for a, b, c, d, e, f in ARTIFACTS]
    runs = PROCESSED / "runs.csv"
    if runs.exists():
        r = pd.read_csv(runs)
        lines += ["", "## Training runs included", "",
                  "| Run id | Algorithm | Root | Transitions | Wall (min) | Commit |", "|---|---|---|---|---|---|"]
        for x in r.sort_values("run_id").itertuples():
            wall = f"{x.wall_time_seconds / 60:.0f}" if pd.notna(x.wall_time_seconds) else "--"
            com = str(x.git_commit)[:10] if pd.notna(x.git_commit) else "1457e9b (worktree)"
            lines.append(f"| `{x.run_id}` | {x.algorithm} | {x.root} | {x.transitions} | {wall} | `{com}` |")
    (ROOT / "docs" / "RESULTS_MANIFEST.md").write_text("\n".join(lines) + "\n")


def section_readme() -> None:
    f = TABLES / "main_results.csv"
    if not f.exists():
        return
    m = pd.read_csv(f)
    lab = {"bandit": "Masked contextual bandit (γ=0)", "ppo": "MaskablePPO (γ=0.995)",
           "milp_track": "MILP-track (per-interval min-max-util, no learning)", "greedy": "Greedy",
           "cspf": "CSPF", "static": "Static shortest path", "noop": "No-op",
           "random_valid": "Random valid", "oracle_h1": "Oracle-1 † (exact next-interval reward)"}
    rows = ["| Policy | Test return [95 % CI] | Roots | Delivered | SLA viol. | Reroutes/h |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in m.sort_values("mean", ascending=False).itertuples():
        partial = "" if r.episodes % 140 == 0 else f" (partial: {r.episodes} episodes)"
        rows.append(f"| {lab.get(r.policy, r.policy)}{partial} | {r.mean:.1f} [{r.lo:.1f}, {r.hi:.1f}] | "
                    f"{int(r.roots) if r.roots else '–'} | {100 * r.delivered:.2f} % | {r.sla:.0f} | "
                    f"{r.reroutes:.2f} |")
    g = TABLES / "main_gap.csv"
    if g.exists():
        x = pd.read_csv(g).iloc[0]
        rows += ["", f"Bandit − PPO, paired: **{x.boot_est:.1f}** [{x.boot_lo:.1f}, {x.boot_hi:.1f}], "
                     f"positive on {int(x.roots_positive)}/{int(x.roots)} roots."]
    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    if a in text and b in text:
        pre, rest = text.split(a, 1)
        _, post = rest.split(b, 1)
        readme.write_text(pre + a + "\n" + "\n".join(rows) + "\n" + b + post)


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
    section_fidelity()
    for f in (section_reproduction, section_main, section_decomposition, section_final_flapping,
              section_horizon, section_tuning,
              section_delay, section_compute):
        try:
            f(data)
        except Exception as exc:  # keep going; report clearly
            print(f"!! {f.__name__} failed: {type(exc).__name__}: {exc}")
            raise
    section_oracle_ladder(data)
    section_seqdiag()
    section_environment()
    section_doc_tables()
    section_manifest()
    section_readme()
    write_numbers()


if __name__ == "__main__":
    main()
