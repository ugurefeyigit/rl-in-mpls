"""Statistics for paired policy comparisons.

Experimental structure. Every policy is evaluated on the same
(scenario, seed) episodes, whose exogenous traffic is identical across
policies, so episode-level comparisons are *paired*. Learners additionally
have a training root; the two learners of one root share the training seed
ledger byte-for-byte, so roots are paired as well.

Two sources of uncertainty are reported separately, never mixed silently:

* **Evaluation uncertainty** for a *fixed* trained policy: scenario-stratified
  paired bootstrap over test episodes.
* **Training uncertainty** across independent training roots: a two-stage
  bootstrap (resample roots, then episodes within scenario strata), and a
  t-interval over root-level mean differences. With 3-5 roots these intervals
  are wide; that is reported, not hidden.

No hypothesis test is used as a decision rule. Point estimates are always
accompanied by an interval; effect sizes are paired (d_z) and in reward units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as sps

KEY = ["scenario", "seed"]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n: int
    method: str

    def fmt(self, digits: int = 2) -> str:
        return f"{self.estimate:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}est": self.estimate, f"{prefix}lo": self.low,
                f"{prefix}hi": self.high, f"{prefix}n": self.n}


def _strata_indices(strata: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(strata == s) for s in np.unique(strata)]


def stratified_bootstrap_mean(values: np.ndarray, strata: np.ndarray, n_boot: int = 10_000,
                              alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile CI of the mean, resampling within strata (equal stratum sizes kept)."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    groups = _strata_indices(np.asarray(strata))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        boots[b] = values[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(float(values.mean()), float(lo), float(hi), len(values),
                    "scenario-stratified paired bootstrap")


def paired_frame(df: pd.DataFrame, a: str, b: str, metric: str = "operational_return",
                 policy_col: str = "policy") -> pd.DataFrame:
    """Rows (scenario, seed[, root]) with columns a, b and diff = a - b."""
    idx = KEY + (["root"] if "root" in df.columns else [])
    wide = df[df[policy_col].isin([a, b])].pivot_table(index=idx, columns=policy_col,
                                                       values=metric).dropna()
    wide["diff"] = wide[a] - wide[b]
    return wide.reset_index()


def paired_effect(diff: np.ndarray) -> float:
    """Paired standardized mean difference d_z = mean / sd."""
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd > 0 else float("inf") * np.sign(np.mean(diff))


def compare_fixed(pairs: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """Evaluation-uncertainty summary for two fixed policies."""
    d = pairs["diff"].to_numpy()
    ci = stratified_bootstrap_mean(d, pairs["scenario"].to_numpy(), seed=seed)
    return {**ci.as_dict("diff_"), "d_z": paired_effect(d),
            "win_rate": float(np.mean(d > 0)), "tie_rate": float(np.mean(d == 0))}


def two_stage_bootstrap(pairs: pd.DataFrame, n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> Interval:
    """Resample training roots, then episodes within scenario strata (per root)."""
    rng = np.random.default_rng(seed)
    roots = sorted(pairs["root"].unique())
    per_root = {r: (g["diff"].to_numpy(), _strata_indices(g["scenario"].to_numpy()))
                for r, g in pairs.groupby("root")}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        means = []
        for r in rng.choice(roots, size=len(roots), replace=True):
            vals, groups = per_root[r]
            idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
            means.append(vals[idx].mean())
        boots[b] = np.mean(means)
    root_means = pairs.groupby("root")["diff"].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(float(root_means.mean()), float(lo), float(hi), len(roots),
                    "two-stage bootstrap (roots, then scenario-stratified episodes)")


def root_t_interval(root_means: Iterable[float], alpha: float = 0.05) -> Interval:
    x = np.asarray(list(root_means), float)
    n = len(x)
    if n < 2:
        return Interval(float(x.mean()), float("nan"), float("nan"), n, "t over roots")
    half = sps.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return Interval(float(x.mean()), float(x.mean() - half), float(x.mean() + half), n,
                    "t-interval over root means")


def compare_learners(pairs: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """Training-uncertainty summary; ``pairs`` must contain a ``root`` column."""
    root_means = pairs.groupby("root")["diff"].mean()
    tb = two_stage_bootstrap(pairs, seed=seed)
    tt = root_t_interval(root_means)
    return {**tb.as_dict("boot_"), **{k: v for k, v in tt.as_dict("t_").items() if k != "t_est"},
            "roots": int(len(root_means)), "roots_positive": int((root_means > 0).sum()),
            "root_means": ";".join(f"{r}:{m:.3f}" for r, m in root_means.items()),
            "episode_win_rate": float(np.mean(pairs["diff"] > 0)),
            "d_z_pooled": paired_effect(pairs["diff"].to_numpy())}


def per_scenario(pairs: pd.DataFrame, seed: int = 0, n_boot: int = 10_000) -> pd.DataFrame:
    """Per-scenario paired mean difference with a two-stage bootstrap CI.

    Within a scenario, roots are resampled, then episodes within each resampled
    root (so between-root variability is not treated as episode noise). With a
    single root this reduces to an episode bootstrap. ``win_rate`` is the share
    of paired episodes with a positive difference (all roots pooled).
    """
    rows = []
    for s, g in pairs.groupby("scenario"):
        rng = np.random.default_rng(seed)
        groups = [grp["diff"].to_numpy() for _, grp in g.groupby("root")] if "root" in g \
            else [g["diff"].to_numpy()]
        root_means = np.array([x.mean() for x in groups])
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(len(groups), size=len(groups))
            boots[b] = np.mean([groups[i][rng.integers(len(groups[i]), size=len(groups[i]))].mean()
                                for i in idx])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        d = g["diff"].to_numpy()
        rows.append({"scenario": s, "diff_est": root_means.mean(), "diff_lo": lo, "diff_hi": hi,
                     "n": len(d), "roots": len(groups),
                     "roots_positive": int((root_means > 0).sum()),
                     "win_rate": float(np.mean(d > 0))})
    return pd.DataFrame(rows)
