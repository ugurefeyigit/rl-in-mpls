"""Generate every paper figure from processed results (no numbers are typed in).

    python scripts/make_paper_figures.py            # all figures whose inputs exist
    python scripts/make_paper_figures.py topology seqdiag

Inputs: experiments/processed/*.csv (built by scripts/study/analyze.py) and
experiments/raw/seqdiag_*/. Outputs: results/figures/paper/*.pdf (+ .png
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
OUT = ROOT / "results" / "figures" / "paper"
PROC = ROOT / "experiments" / "processed"
RAW = ROOT / "experiments" / "raw"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COLOR = {"bandit": "#2a78d6", "ppo": "#eb6834", "greedy": "#1baf7a", "cspf": "#eda100",
         "static": "#4a3aa7", "noop": "#8a8984", "random_valid": "#b5b4ae",
         "oracle": "#0b0b0b", "q": "#1baf7a"}
MARK = {"bandit": "o", "ppo": "s", "greedy": "^", "cspf": "D", "static": "v",
        "noop": "x", "random_valid": "+", "oracle": "*", "q": "P"}
LABEL = {"bandit": "Masked bandit", "ppo": "MaskablePPO", "greedy": "Greedy", "cspf": "CSPF",
         "static": "Static SP", "noop": "No-op", "random_valid": "Random valid",
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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for df, name, ls, mk in ((live, "clairvoyant (true future traffic)", "-", "o"),
                             (frozen, "frozen exogenous (current traffic held)", "--", "s")):
        if df.empty:
            continue
        agree = [df[f"agree_h{h}"].mean() for h in H]
        regret = [df[f"myopic_regret_h{h}"].mean() / max(df[f"best_delta_h{h}"].mean(), 1e-12)
                  for h in H]
        axes[0].plot(H, agree, ls=ls, marker=mk, color=COLOR["bandit"], label=name)
        axes[1].plot(H, [1 - r for r in regret], ls=ls, marker=mk, color=COLOR["bandit"],
                     label=name)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(H, [str(h) for h in H])
        ax.set_xlabel("lookahead horizon H (control intervals of 5 min)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("P(myopic action = H-step best)")
    axes[1].set_ylabel("share of H-step gain captured\nby the myopic action")
    axes[0].legend(loc="lower left")
    save(fig, "fig_seqdiag")


FIGURES = {"topology": fig_topology, "seqdiag": fig_seqdiag}


def main() -> None:
    style()
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n]()


if __name__ == "__main__":
    main()
