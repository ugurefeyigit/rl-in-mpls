"""Generate every paper figure from processed results (no numbers are typed in).

    python scripts/make_paper_figures.py            # all figures whose inputs exist
    python scripts/make_paper_figures.py topology seqdiag

Inputs: experiments/processed/*.csv (built by scripts/study/analyze.py) and
experiments/raw/seqdiag_*/. Outputs: results/paper_figures/*.pdf (+ .png
previews). Colors: the validated reference categorical palette, fixed per
method (color follows the method in every figure), with marker shapes as
secondary encoding.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "paper_figures"
PROC = ROOT / "experiments" / "processed"
RAW = ROOT / "experiments" / "raw"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COLOR = {"bandit": "#2a78d6", "ppo": "#eb6834", "greedy": "#1baf7a", "cspf": "#eda100",
         "static": "#4a3aa7", "noop": "#8a8984", "random_valid": "#b5b4ae",
         "oracle": "#0b0b0b", "q": "#8c564b", "milp_track": "#e87ba4", "ppog0": "#eb6834",
         "milp_tuned": "#b8326f"}
MARK = {"bandit": "o", "ppo": "s", "greedy": "^", "cspf": "D", "static": "v",
        "noop": "x", "random_valid": "+", "oracle": "*", "q": "P", "milp_track": "h", "ppog0": "s",
        "milp_tuned": "H"}
LABEL = {"bandit": "Masked bandit", "ppo": "MaskablePPO", "greedy": "Greedy", "cspf": "CSPF",
         "static": "Static SP", "noop": "No-op", "random_valid": "Random valid",
         "milp_track": "MILP-track", "milp_tuned": "MILP-track (tuned)", "ppog0": "MaskablePPO, γ = 0",
         "oracle_h1": "Oracle H=1", "oracle_h3": "Oracle H=3", "oracle_h6": "Oracle H=6"}
SCEN_SHORT = {"full_day": "full day", "evening_peak": "evening peak", "flash_crowd": "flash crowd",
              "link_failure": "link failure", "deceptive_local_optimum": "deceptive",
              "ood_double_failure": "double failure", "overload_stress": "overload"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# --------------------------------------------------------------- topology
def fig_topology() -> None:
    topo = yaml.safe_load((ROOT / "configs" / "topology.yaml").read_text())
    pos = {r["id"]: (r["x"], -r["y"]) for r in topo["routers"]}
    role = {r["id"]: r["role"] for r in topo["routers"]}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.grid(False)
    ax.set_axis_off()
    widths = {250: 0.6, 500: 1.1, 1000: 1.8, 2000: 3.0}
    for link in topo["links"]:
        (x1, y1), (x2, y2) = pos[link["a"]], pos[link["z"]]
        hot = link["id"] in ("L11", "L20")
        xs, ys = [x1, x2], [y1, y2]
        if x1 == x2 and abs(y1 - y2) > 200:  # long vertical link would hide the ring
            t = np.linspace(0, 1, 40)
            xs = list(x1 + 70 * np.sin(np.pi * t))
            ys = list(y1 + (y2 - y1) * t)
        ax.plot(xs, ys, color=COLOR["ppo"] if hot else "#9b9a94",
                lw=widths[link["capacity_mbps"]], zorder=1, solid_capstyle="round")
        if hot:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 10, link["id"], fontsize=6,
                    color=INK, ha="center", va="bottom", zorder=4)
    shape = {"PE_IN": "s", "PE_OUT": "s", "P": "o", "AGG": "D"}
    fill = {"PE_IN": COLOR["bandit"], "PE_OUT": COLOR["bandit"], "P": "#ffffff", "AGG": "#ffffff"}
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=95, marker=shape[role[n]], facecolor=fill[role[n]],
                   edgecolor=INK, linewidth=0.8, zorder=3)
        ax.text(x, y, n, fontsize=4.6, ha="center", va="center", zorder=5,
                color="#ffffff" if role[n].startswith("PE") else INK)
    for cap, w in widths.items():
        ax.plot([], [], color="#9b9a94", lw=w, label=f"{cap} Mb/s")
    ax.plot([], [], color=COLOR["ppo"], lw=1.8, label="L11 / L20 (stress links)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=5, fontsize=5.5,
              handlelength=1.4, columnspacing=0.8)
    save(fig, "fig_topology")


# ------------------------------------------------------------ seqdiag
def load_seqdiag(name: str) -> pd.DataFrame:
    files = glob.glob(str(RAW / name / "*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def fig_seqdiag() -> None:
    live = load_seqdiag("seqdiag_greedy")
    frozen = load_seqdiag("seqdiag_greedy_frozen")
    if live.empty:
        print("skip seqdiag: no data")
        return
    H = [1, 2, 3, 6, 12, 24]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    series = [(live, "clairvoyant (true future traffic)", "-", "o", COLOR["bandit"]),
              (frozen, "frozen exogenous (current traffic held)", "--", "s", COLOR["bandit"]),
              (load_seqdiag("seqdiag_greedy_delay1"), "delay L = 1, clairvoyant", "-", "o",
               COLOR["ppo"]),
              (load_seqdiag("seqdiag_greedy_delay1_frozen"), "delay L = 1, frozen", "--", "s",
               COLOR["ppo"]),
              (load_seqdiag("seqdiag_greedy_contgreedy"), "greedy continuation, clairvoyant", "-",
               "^", COLOR["q"]),
              (load_seqdiag("seqdiag_greedy_contgreedy_frozen"), "greedy continuation, frozen", "--",
               "^", COLOR["q"])]
    for df, name, ls, mk, col in series:
        if df.empty:
            continue
        agree = [df[f"agree_h{h}"].mean() for h in H]
        regret = [df[f"myopic_regret_h{h}"].mean() / max(df[f"best_delta_h{h}"].mean(), 1e-12)
                  for h in H]
        axes[0].plot(H, agree, ls=ls, marker=mk, color=col, label=name)
        axes[1].plot(H, [1 - r for r in regret], ls=ls, marker=mk, color=col, label=name)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(H, [str(h) for h in H])
        ax.set_xlabel("lookahead horizon H (control intervals of 5 min)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("P(myopic action = H-step best)")
    axes[1].set_ylabel("share of H-step gain captured\nby the myopic action")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.33)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_seqdiag")



# ------------------------------------------------------------ results
def _read(name: str) -> pd.DataFrame:
    f = PROC / f"{name}.csv"
    try:
        return pd.read_csv(f)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def fig_learning_curves() -> None:
    cur = _read("validation_curves")
    if cur.empty:
        print("skip curves")
        return
    cur = cur[(cur.seedset == "validation") & cur.policy.isin(["bandit", "ppo", "E2:ppog0"])]
    cur = cur.assign(policy=cur.policy.replace({"E2:ppog0": "ppog0"}))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for pol, ls in (("bandit", "-"), ("ppo", "-"), ("ppog0", "--")):
        g = cur[cur.policy == pol]
        if g.empty:
            continue
        if pol != "ppog0":
            for _, r in g.groupby("root"):
                ax.plot(r.checkpoint / 1e3, r.mean_return, color=COLOR[pol], lw=0.6, alpha=0.3)
        m = g.groupby("checkpoint").mean_return.mean()
        lab = LABEL[pol] + (", γ = 0.995" if pol == "ppo" else "")
        ax.plot(m.index / 1e3, m.values, color=COLOR[pol], marker=MARK[pol], ls=ls,
                mfc="white" if pol == "ppog0" else COLOR[pol],
                label=f"{lab} ({g.root.nunique()} roots)")
    ax.set_xlabel("training transitions (thousands)")
    ax.set_ylabel("mean validation return")
    ax.legend(loc="lower right")
    save(fig, "fig_learning_curves")


def fig_main() -> None:
    f = ROOT / "results" / "tables" / "main_per_scenario.csv"
    m = ROOT / "results" / "tables" / "main_results.csv"
    if not f.exists() or not m.exists():
        print("skip main")
        return
    ps, main = pd.read_csv(f), pd.read_csv(m)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    order = [p for p in ("noop", "static", "random_valid", "cspf", "greedy", "milp_track",
                         "milp_tuned", "ppo", "bandit", "oracle_h1") if p in set(main.policy)]
    main = main.set_index("policy").loc[order].reset_index().sort_values("mean").reset_index(drop=True)
    y = np.arange(len(main))
    for i, r in main.iterrows():
        key = "oracle" if r.policy.startswith("oracle") else r.policy
        c = COLOR.get(key, INK2)
        ax.text(r["hi"] + 3, i, f"{r['mean']:.1f}", va="center", fontsize=6.5, color=INK2)
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                    fmt=MARK.get(key, "o"), color=c, ms=5, capsize=2, lw=1.2)
    ax.set_yticks(y, [LABEL.get(p, p) + (" †" if p.startswith("oracle") else "") for p in main.policy])
    ax.set_xlim(right=main["hi"].max() + 22)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("mean test return (95% CI)")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ps["label"] = ps.scenario.map(SCEN_SHORT)
    ps = ps.iloc[::-1].reset_index(drop=True)
    for i, r in ps.iterrows():
        ax.errorbar(r.diff_est, i, xerr=[[r.diff_est - r.diff_lo], [r.diff_hi - r.diff_est]],
                    fmt="o", color=COLOR["bandit"], ms=4.5, capsize=2, lw=1.2)
    ax.set_yticks(range(len(ps)), ps.label)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("bandit − PPO return per episode (95% CI)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_main")


def fig_horizon() -> None:
    f = ROOT / "results" / "tables" / "horizon_sweep.csv"
    if not f.exists():
        print("skip horizon")
        return
    h = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for fam, key, lab in (("Q", "bandit", "masked value learner (γ=0: bandit)"),
                          ("PPO", "ppo", "MaskablePPO")):
        g = h[h.family == fam].sort_values("gamma")
        x = g.gamma.map(lambda v: {0: 0, 0.5: 1, 0.9: 2, 0.99: 3, 0.995: 3.3}[v])
        ax.plot(x, g["mean"], marker=MARK[key], color=COLOR[key], label=lab)
        if "root_sd" in g:
            ax.errorbar(x, g["mean"], yerr=g["root_sd"], fmt="none", color=COLOR[key], capsize=2, lw=0.8)
    ax.set_xticks([0, 1, 2, 3, 3.3], ["0", "0.5", "0.9", "0.99", "   .995"])
    ax.set_xlabel("discount γ")
    ax.set_ylabel("mean test return (±1 SD over roots)")
    ax.legend(loc="lower left")
    save(fig, "fig_horizon")


def fig_delay() -> None:
    """Dumbbell: test return without delay vs with L = 1, same roots/episodes."""
    f = ROOT / "results" / "tables" / "delay_results.csv"
    if not f.exists():
        print("skip delay")
        return
    d = pd.read_csv(f).set_index("policy")
    rows = [("bandit", "bandit", "Masked bandit"), ("qg05", "q", "Q-learner γ = 0.5"),
            ("qg09", "q", "Q-learner γ = 0.9"),
            ("ppo", "ppo", "MaskablePPO γ = 0.995"), ("ref:milp_track", "milp_track", "MILP-track"),
            ("ref:greedy", "greedy", "Greedy"), ("ref:cspf", "cspf", "CSPF")]
    rows = [r for r in rows if r[0] in d.index]
    fig, ax = plt.subplots(figsize=(3.4, 0.35 * len(rows) + 0.5))
    for i, (key, ck, lab) in enumerate(rows):
        a, b = d.loc[key, "mean_L0_same_roots"], d.loc[key, "mean_L1"]
        c = COLOR.get(ck, INK2)
        ax.annotate("", xy=(b, i), xytext=(a, i),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, shrinkA=3, shrinkB=3))
        ax.plot([a], [i], marker=MARK.get(ck, "o"), mfc=SURF, mec=c, ls="none")
        ax.plot([b], [i], marker=MARK.get(ck, "o"), color=c, ls="none")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("mean test return: no delay (hollow) → L = 1 (filled)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_delay")


def fig_case() -> None:
    f = RAW / "case_studies" / "deceptive_local_optimum_seed3001.csv"
    if not f.exists():
        print("skip case")
        return
    d = pd.read_csv(f)
    names = {"E0_repro__bandit__r314159": ("bandit", "Masked bandit"),
             "E0_repro__ppo__r314159": ("ppo", "MaskablePPO"),
             "milp_track": ("cspf", "MILP-track"), "oracle_h1": ("oracle", "Oracle-1 †")}
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.3), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    for pol, (key, lab) in names.items():
        g = d[d.policy == pol]
        tot = g.reward.sum()
        ax.plot(g.t_min / 60, g.max_util, color=COLOR[key], lw=1.3,
                label=f"{lab} (return {tot:.0f})")
    ax.axhline(1.0, color=INK2, lw=0.6, ls="--")
    ax.axvspan(45 / 60, 225 / 60, color=GRID, alpha=0.6, lw=0)
    ax.text(48 / 60, 0.42, "burst on D2, D4, D5 (×1.8)", fontsize=6.5, color=INK2, va="bottom")
    ax.set_ylabel("max link utilization")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.5)
    ax = axes[1]
    ax.grid(axis="y", visible=False)
    for i, (pol, (key, lab)) in enumerate(names.items()):
        g = d[(d.policy == pol) & (d.action > 0) & d.action_accepted]
        rev = g.te_reversals > 0
        ax.scatter(g.t_min[~rev] / 60, np.full((~rev).sum(), i), marker="|", s=60,
                   color=COLOR[key], lw=1.4)
        ax.scatter(g.t_min[rev] / 60, np.full(rev.sum(), i), marker="x", s=22,
                   color=COLOR[key], lw=1.0)
    ax.set_yticks(range(len(names)), [v[1] for v in names.values()])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("hours since 09:00 (deceptive-local-optimum scenario, test seed 3001)")
    ax.scatter([], [], marker="|", color=INK2, label="accepted move")
    ax.scatter([], [], marker="x", color=INK2, label="reversal (back to previous path)")
    ax.legend(loc="lower right", ncol=2, fontsize=6.5, bbox_to_anchor=(1.0, 0.98))
    fig.tight_layout()
    save(fig, "fig_case")


FIGURES = {"case": fig_case, "topology": fig_topology, "seqdiag": fig_seqdiag, "curves": fig_learning_curves,
           "main": fig_main, "horizon": fig_horizon, "delay": fig_delay}


def main() -> None:
    style()
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n]()


if __name__ == "__main__":
    main()
