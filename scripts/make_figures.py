"""Generate presentation-ready figures from evaluation outputs.

Reads results/eval_summary.csv, results/eval_steps_*.csv and the training
evaluations file models/<tag>/evaluations.npz; writes PNGs to results/figures.

Usage: python scripts/make_figures.py [--model ppo_te]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

ALGO_COLORS = {"rl": "#1f77b4", "static": "#7f7f7f", "greedy": "#2ca02c",
               "cspf": "#e0a020", "random": "#d62728"}
ALGO_ORDER = ["static", "random", "greedy", "cspf", "rl"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _algos_in(df: pd.DataFrame) -> list[str]:
    return [a for a in ALGO_ORDER if a in set(df["algorithm"])]


def fig_summary_bars(sdf: pd.DataFrame) -> None:
    metrics = [("max_util_mean", "Mean of max link utilization"),
               ("sla_violations_total", "Total SLA violations"),
               ("mean_delay_ms", "Mean end-to-end delay (ms)"),
               ("reroutes_total", "Total reroutes")]
    scenarios = sorted(sdf["scenario"].unique())
    for mkey, title in metrics:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        algos = _algos_in(sdf)
        width = 0.8 / len(algos)
        x = np.arange(len(scenarios))
        for i, algo in enumerate(algos):
            means, errs = [], []
            for scen in scenarios:
                g = sdf[(sdf.scenario == scen) & (sdf.algorithm == algo)][mkey].dropna()
                means.append(g.mean() if len(g) else np.nan)
                errs.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0)
            ax.bar(x + i * width, means, width, yerr=errs, capsize=2,
                   label=algo, color=ALGO_COLORS[algo])
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{title} — mean ± 95% CI across seeds")
        ax.legend(ncol=len(algos), fontsize=8)
        if mkey == "max_util_mean":
            ax.axhline(1.0, color="red", ls=":", lw=1)
        fig.tight_layout()
        fig.savefig(FIGS / f"summary_{mkey}.png", dpi=150)
        plt.close(fig)


def fig_timeseries(scenario: str, metric: str, ylabel: str, fname: str,
                   hline: float | None = None) -> None:
    files = sorted(RESULTS.glob(f"eval_steps_{scenario}_seed*_*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for f in files:
        df = pd.read_csv(f)
        algo = df["algorithm"].iloc[0]
        if algo not in ALGO_COLORS:
            continue
        ax.plot(df["t_min"] / 60, df[metric], label=algo,
                color=ALGO_COLORS[algo], lw=1.4)
    if hline is not None:
        ax.axhline(hline, color="red", ls=":", lw=1)
    # mark failure windows
    df0 = pd.read_csv(files[0])
    if "n_failed_links" in df0 and df0["n_failed_links"].max() > 0:
        fail = df0["n_failed_links"] > 0
        t = df0["t_min"] / 60
        ax.fill_between(t, 0, 1, where=fail, transform=ax.get_xaxis_transform(),
                        alpha=0.12, color="red", label="link failed")
    ax.set_xlabel("simulated hours into scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{scenario}: {ylabel} (paired traffic, same seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_training_curve(tag: str) -> None:
    npz = ROOT / "models" / tag / "evaluations.npz"
    if not npz.exists():
        return
    data = np.load(npz)
    steps, results = data["timesteps"], data["results"]  # (n_evals, n_episodes)
    mean, std = results.mean(axis=1), results.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean, color="#1f77b4", lw=1.6, label="eval episode return")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="#1f77b4")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("evaluation episode return")
    ax.set_title(f"Training curve ({tag}) — eval on full_day, 3 episodes/point")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curve.png", dpi=150)
    plt.close(fig)


def fig_reward_distribution(sdf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    scen = "full_day" if "full_day" in set(sdf.scenario) else sdf.scenario.iloc[0]
    data, labels, colors = [], [], []
    for algo in _algos_in(sdf):
        vals = sdf[(sdf.scenario == scen) & (sdf.algorithm == algo)]["reward_sum"].dropna()
        if len(vals):
            data.append(vals.to_numpy())
            labels.append(f"{algo}\n(n={len(vals)})")
            colors.append(ALGO_COLORS[algo])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("episode return")
    ax.set_title(f"Episode-return distribution across seeds — {scen}")
    fig.tight_layout()
    fig.savefig(FIGS / "reward_distribution.png", dpi=150)
    plt.close(fig)


def fig_link_util_heatmap(scenario: str = "evening_peak") -> None:
    """Per-link mean utilization: RL vs static (needs a live rerun for
    per-link data, cheap at ~0.5 s per episode)."""
    from mplssim.experiments.runner import run_episode  # noqa: F401  (import check)
    from mplssim.factory import make_engine, engine_config_from_training
    from mplssim.baselines import make_baseline
    from server.session import load_model
    from mplssim.rl.env import MplsTeEnv

    seed = 101
    utils: dict[str, np.ndarray] = {}
    # static
    eng = make_engine(scenario, seed=seed, cfg=engine_config_from_training())
    ctl = make_baseline("static")
    acc = []
    while not eng.done:
        for d, p in ctl.decide(eng):
            eng.apply_action(d, p, source="static")
        eng.step_interval()
        acc.append(eng.link_util.copy())
    utils["static"] = np.mean(acc, axis=0)
    link_ids = [dl.id for dl in eng.topo.dlinks]
    # rl
    try:
        model = load_model("ppo_te")
    except FileNotFoundError:
        return
    env = MplsTeEnv(scenario=scenario, base_seed=seed)
    obs, _ = env.reset(options={"episode_seed": seed})
    acc, done = [], False
    while not done:
        a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, _, te, tr, _ = env.step(int(a))
        acc.append(env.eng.link_util.copy())
        done = te or tr
    utils["rl"] = np.mean(acc, axis=0)

    order = np.argsort(-utils["static"])[:28]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mat = np.vstack([utils["static"][order], utils["rl"][order]])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1.2)
    ax.set_yticks([0, 1], ["static", "rl"])
    ax.set_xticks(range(len(order)),
                  [link_ids[i] for i in order], rotation=90, fontsize=6)
    ax.set_title(f"Mean directed-link utilization — {scenario}, seed {seed} "
                 f"(28 hottest links under static)")
    fig.colorbar(im, label="utilization")
    fig.tight_layout()
    fig.savefig(FIGS / "link_util_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_te")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    sdf = pd.read_csv(RESULTS / "eval_summary.csv")
    fig_summary_bars(sdf)
    fig_reward_distribution(sdf)
    fig_training_curve(args.model)
    for scen in sorted(sdf["scenario"].unique()):
        fig_timeseries(scen, "max_util", "max link utilization",
                       f"ts_maxutil_{scen}.png", hline=1.0)
        fig_timeseries(scen, "sla_violations", "SLA violations",
                       f"ts_sla_{scen}.png")
        fig_timeseries(scen, "mean_delay_ms", "mean delay (ms)",
                       f"ts_delay_{scen}.png")
    fig_link_util_heatmap()
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
