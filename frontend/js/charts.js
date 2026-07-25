// ECharts: metric time series (per algorithm), cumulative reward,
// reward-component breakdown, and the traffic matrix heatmap.
// Series are named with the audience-facing controller label; colours stay
// keyed by the internal algorithm id.

import { algoLabel, city } from "./display.js";
import { rate } from "./fmt.js";

const ALGO_COLORS = { rl: "#58a6ff", static: "#8494a7", greedy: "#3fb950",
                      cspf: "#e3b341", random: "#db61a2" };
const DARK = {
  textStyle: { color: "#8494a7", fontSize: 10 },
  axisLine: { lineStyle: { color: "#2c3644" } },
  splitLine: { lineStyle: { color: "#1b232e" } },
};

function baseChart(el) {
  const c = echarts.init(el, null, { renderer: "canvas" });
  window.addEventListener("resize", () => c.resize());
  return c;
}

export class Charts {
  constructor() {
    this.metric = baseChart(document.getElementById("chart-metric"));
    this.reward = baseChart(document.getElementById("chart-reward"));
    this.components = baseChart(document.getElementById("chart-components"));
    this.matrix = baseChart(document.getElementById("chart-matrix"));
    this.metricKey = "max_util";
    this.history = {};        // algo -> [interval metrics]
    this.rewardSeries = {};   // algo -> [[t, cumreward]]
    this.compSeries = {};     // algo -> last components
  }

  reset() {
    this.history = {}; this.rewardSeries = {}; this.compSeries = {};
    this.render();
  }

  push(algo, metrics, decision) {
    (this.history[algo] = this.history[algo] || []).push(metrics);
    if (decision) {
      (this.rewardSeries[algo] = this.rewardSeries[algo] || []).push(
        [metrics.t_min, decision.cumulative_reward]);
      this.compSeries[algo] = decision.components;
    }
  }

  /**
   * Replace the accumulated series from GET /api/metrics/history — the
   * authoritative per-interval record (AlgoRunner.history). Used on page
   * reload and after a fast-forward, where intermediate intervals are stepped
   * on the server but only the final payload is broadcast.
   */
  setHistories(runs) {
    this.history = {};
    this.rewardSeries = {};
    for (const r of runs) {
      this.history[r.algorithm] = r.history;
      let cum = 0;
      this.rewardSeries[r.algorithm] = r.history.map((h) => {
        cum += h.reward || 0;
        return [h.t_min, cum];
      });
    }
  }

  /** Highest interval index seen per algorithm — used to de-duplicate ticks. */
  lastStep(algo) {
    const h = this.history[algo];
    return h && h.length ? (h[h.length - 1].step ?? h.length) : 0;
  }

  render() {
    const key = this.metricKey;
    const series = Object.entries(this.history).map(([algo, hist]) => ({
      name: algoLabel(algo), type: "line", showSymbol: false, smooth: 0.15,
      lineStyle: { width: 1.6, color: ALGO_COLORS[algo] },
      itemStyle: { color: ALGO_COLORS[algo] },
      data: hist.map((h) => [h.t_min / 60, h[key]]),
    }));
    const markLine = (key === "max_util")
      ? { silent: true, symbol: "none",
          lineStyle: { color: "#f85149", type: "dotted" },
          data: [{ yAxis: 1.0 }, { yAxis: 0.9 }] }
      : undefined;
    if (series.length && markLine) series[0].markLine = markLine;
    this.metric.setOption({
      ...DARK, animation: false,
      grid: { left: 42, right: 12, top: 26, bottom: 22 },
      legend: { top: 0, textStyle: { color: "#8494a7", fontSize: 10 } },
      tooltip: { trigger: "axis", backgroundColor: "#0b0f14", borderColor: "#2c3644",
                 textStyle: { color: "#dbe4ee", fontSize: 11 } },
      xAxis: { type: "value", name: "h", ...DARK, axisLabel: { color: "#8494a7" } },
      yAxis: { type: "value", ...DARK, axisLabel: { color: "#8494a7" },
               scale: key === "jain_fairness" || key === "delivered_ratio" },
      series,
    }, { notMerge: true });

    this.reward.setOption({
      ...DARK, animation: false,
      title: { text: "Cumulative reward", left: 0, top: 0,
               textStyle: { color: "#8494a7", fontSize: 11 } },
      grid: { left: 46, right: 12, top: 26, bottom: 20 },
      tooltip: { trigger: "axis", backgroundColor: "#0b0f14", borderColor: "#2c3644",
                 textStyle: { color: "#dbe4ee", fontSize: 11 } },
      xAxis: { type: "value", ...DARK, axisLabel: { color: "#8494a7" } },
      yAxis: { type: "value", ...DARK, axisLabel: { color: "#8494a7" }, scale: true },
      series: Object.entries(this.rewardSeries).map(([algo, data]) => ({
        name: algoLabel(algo), type: "line", showSymbol: false,
        lineStyle: { width: 1.6, color: ALGO_COLORS[algo] },
        itemStyle: { color: ALGO_COLORS[algo] },
        data: data.map(([t, v]) => [t / 60, v]),
      })),
    }, { notMerge: true });

    const algos = Object.keys(this.compSeries);
    const compKeys = algos.length ? Object.keys(this.compSeries[algos[0]]) : [];
    this.components.setOption({
      ...DARK, animation: false,
      title: { text: "Reward components (last interval)", left: 0, top: 0,
               textStyle: { color: "#8494a7", fontSize: 11 } },
      grid: { left: 80, right: 12, top: 24, bottom: 20 },
      tooltip: { backgroundColor: "#0b0f14", borderColor: "#2c3644",
                 textStyle: { color: "#dbe4ee", fontSize: 11 } },
      xAxis: { type: "value", ...DARK, axisLabel: { color: "#8494a7" } },
      yAxis: { type: "category", data: compKeys,
               axisLabel: { color: "#8494a7", fontSize: 9 } },
      series: algos.map((algo) => ({
        name: algoLabel(algo), type: "bar", barGap: "10%",
        itemStyle: { color: ALGO_COLORS[algo] },
        data: compKeys.map((k) => this.compSeries[algo][k]),
      })),
    }, { notMerge: true });
  }

  renderMatrix(snapshot, onCell) {
    if (!snapshot) return;
    const srcs = [...new Set(snapshot.demands.map((d) => d.src))].sort();
    const dsts = [...new Set(snapshot.demands.map((d) => d.dst))].sort();
    const cell = {};
    for (const d of snapshot.demands) {
      const k = d.src + "|" + d.dst;
      cell[k] = (cell[k] || 0) + d.volume_mbps;
    }
    const data = [];
    let vmax = 1;
    srcs.forEach((s, i) => dsts.forEach((t, j) => {
      const v = Math.round(cell[s + "|" + t] || 0);
      vmax = Math.max(vmax, v);
      data.push([j, i, v]);
    }));
    this.matrix.setOption({
      animation: false,
      tooltip: { formatter: (p) =>
                   `${city(srcs[p.value[1]])} → ${city(dsts[p.value[0]])}: ${rate(p.value[2])}`,
                 backgroundColor: "#0b0f14", borderColor: "#2c3644",
                 textStyle: { color: "#dbe4ee", fontSize: 11 } },
      grid: { left: 76, right: 60, top: 24, bottom: 46 },
      xAxis: { type: "category", data: dsts.map(city),
               axisLabel: { color: "#8494a7", rotate: 40 },
               name: "destination city", nameGap: 32, nameTextStyle: { color: "#8494a7" } },
      yAxis: { type: "category", data: srcs.map(city), axisLabel: { color: "#8494a7" },
               name: "source city", nameTextStyle: { color: "#8494a7" } },
      visualMap: {
        min: 0, max: vmax, calculable: false, orient: "vertical", right: 0, top: "center",
        inRange: { color: ["#10161f", "#1f4e8c", "#d29922", "#f85149"] },
        textStyle: { color: "#8494a7", fontSize: 9 },
      },
      series: [{
        type: "heatmap", data,
        label: { show: true, color: "#dbe4ee", fontSize: 10, fontFamily: "Consolas" },
      }],
    }, { notMerge: true });
    this.matrix.off("click");
    this.matrix.on("click", (p) => onCell(srcs[p.value[1]], dsts[p.value[0]]));
  }
}
