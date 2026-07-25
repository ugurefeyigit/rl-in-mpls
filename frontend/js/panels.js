// Right-rail panels: agent decision card, scoreboard, benchmark, events,
// LSP/link tables, decision tape, training monitor, saved runs.
//
// Every user-facing label goes through display.js (city names); internal IDs
// (D2, p3, L11) appear only in the dimmed "technical" line.

import {
  city, pathLabel, pathVia, linkLabel, linkTechnical, demandLabel,
  algoLabel, algoTech,
} from "./display.js";
import {
  esc, rate, util, delay, lossPct, loss, reward as fmtReward, signed, ratio,
  simTime,
} from "./fmt.js";

const CLASS_COLORS = { voice: "#e15759", video: "#4e79a7", vpn: "#59a14f",
                       besteffort: "#9c755f", bulk: "#f28e2b", critical: "#b07aa1" };

// A reroute burst this large in a 12-interval window is worth calling out:
// the policy is trading route stability for utilization (see docs/REPORT.md).
export const CHURN_WINDOW = 12;
export const CHURN_THRESHOLD = 8;

// ------------------------------------------------------- decision description
/**
 * Turn a controller decision into plain language using city names.
 * Returns {text, technical, cls} — `technical` is the engineer-only detail.
 */
export function describeDecision(decision, snapshot) {
  const byId = {};
  if (snapshot) for (const d of snapshot.demands) byId[d.id] = d;

  const routersFor = (demandId, pathIdx) => {
    const d = byId[demandId];
    if (!d || !d.candidates) return null;
    const c = d.candidates.find((x) => x.path_idx === pathIdx);
    return c ? c.routers : null;
  };
  const nameFor = (demandId) => {
    const d = byId[demandId];
    return d ? demandLabel(d.src, d.dst, d.class) : demandId;
  };

  if (decision.decoded) {
    const dec = decision.decoded;
    if (dec.type !== "reroute") {
      return { text: "No change — current routing is holding.",
               technical: "action 0 (no-op)", cls: "ok" };
    }
    const from = routersFor(dec.demand, dec.from_path);
    const to = routersFor(dec.demand, dec.path_idx);
    const technical =
      `${dec.demand} p${dec.from_path}→p${dec.path_idx}`;
    if (dec.accepted === false) {
      return {
        text: `Rejected: move ${nameFor(dec.demand)} via ` +
              `${to ? pathVia(to) : `path ${dec.path_idx}`} — ${dec.reason}.`,
        technical, cls: "rej",
      };
    }
    return {
      text: `Move ${nameFor(dec.demand)} via ` +
            `${to ? pathVia(to) : `path ${dec.path_idx}`}` +
            `${from ? ` (was via ${pathVia(from)})` : ""}.`,
      technical, cls: "warn",
    };
  }

  const moves = decision.moves || [];
  if (!moves.length) {
    return { text: "No change — current routing is holding.",
             technical: "no move", cls: "ok" };
  }
  return {
    text: moves.map((m) => {
      const to = routersFor(m.demand, m.path_idx);
      const verb = m.accepted === false ? "Rejected move of" : "Move";
      return `${verb} ${nameFor(m.demand)} via ` +
             `${to ? pathVia(to) : `path ${m.path_idx}`}`;
    }).join("; ") + ".",
    technical: moves.map((m) => `${m.demand}→p${m.path_idx}`).join(", "),
    cls: moves.some((m) => m.accepted === false) ? "rej" : "warn",
  };
}

// ------------------------------------------------------------ decision tape
export function tapeAppend(algo, decision, hour, snapshot) {
  const box = document.getElementById("tape-lines");
  if (!box) return;
  const { text, technical, cls } = describeDecision(decision, snapshot);
  const line = document.createElement("div");
  line.className = `tape-line ${cls}`;
  const who = decision.algorithm === "rl" ? "who-rl" : "who-b";
  line.innerHTML =
    `<span class="t">${simTime(hour)}</span>` +
    `<span class="${who}">${esc(algoLabel(decision.algorithm))}</span>` +
    `<span class="msg">${esc(text)}</span>` +
    `<span class="tech">${esc(technical)}</span>` +
    `<span class="t">r=${signed(decision.reward, 2)}</span>`;
  box.prepend(line);
  while (box.children.length > 120) box.removeChild(box.lastChild);
}

// ------------------------------------------------------ agent decision card
export function renderDecision(container, runs) {
  container.innerHTML = "";
  for (const run of runs) {
    const dec = run.decision;
    if (!dec) continue;
    const snap = run.snapshot;
    const { text, technical } = describeDecision(dec, snap);
    const card = document.createElement("div");
    card.className = "dec-card";

    let badge = `<span class="badge noop">NO CHANGE</span>`;
    if (dec.decoded && dec.decoded.type === "reroute") {
      badge = dec.decoded.accepted
        ? `<span class="badge ok">APPLIED</span>`
        : `<span class="badge rej">BLOCKED BY SAFETY CHECK</span>`;
    } else if (dec.moves && dec.moves.length) {
      badge = `<span class="badge ok">APPLIED</span>`;
    }

    // full route chains, only for an accepted single-demand reroute
    let routeHtml = "";
    if (dec.decoded && dec.decoded.type === "reroute" && snap) {
      const d = snap.demands.find((x) => x.id === dec.decoded.demand);
      if (d && d.candidates) {
        const from = d.candidates.find((c) => c.path_idx === dec.decoded.from_path);
        const to = d.candidates.find((c) => c.path_idx === dec.decoded.path_idx);
        if (from && to) {
          routeHtml = `<dl class="kv route-kv">
            <dt>previous route</dt><dd>${esc(pathLabel(from.routers))}</dd>
            <dt>new route</dt><dd>${esc(pathLabel(to.routers))}</dd>
            <dt>traffic volume</dt><dd>${esc(rate(d.volume_mbps))}</dd>
          </dl>`;
        }
      }
    }

    let probRows = "";
    if (dec.top_actions) {
      probRows = `<h4>Policy action probabilities</h4>` + dec.top_actions.map((a) => `
        <div class="bar-row">
          <span class="lbl">${esc(a.desc.length > 26 ? a.desc.slice(0, 26) + "…" : a.desc)}</span>
          <span class="bar-track"><span class="bar-fill prob" style="width:${(a.prob * 100).toFixed(1)}%"></span></span>
          <span class="val">${(a.prob * 100).toFixed(1)}%</span>
        </div>`).join("");
    }
    const comps = Object.entries(dec.components || {});
    const maxAbs = Math.max(0.01, ...comps.map(([, v]) => Math.abs(v)));
    const compRows = comps.map(([k, v]) => `
      <div class="bar-row">
        <span class="lbl">${esc(k)}</span>
        <span class="bar-track"><span class="bar-fill ${v >= 0 ? "pos" : "neg"}"
          style="width:${(Math.abs(v) / maxAbs * 50).toFixed(1)}%"></span></span>
        <span class="val">${signed(v, 3)}</span>
      </div>`).join("");

    let cfHtml = "";
    if (dec.counterfactual) {
      const cf = dec.counterfactual;
      cfHtml = `<h4>Counterfactual (post-hoc)</h4>
        <dl class="kv">
          <dt>busiest-link utilization if no change</dt><dd>${util(cf.noop.max_util, 1)}</dd>
          <dt>busiest-link utilization actual</dt><dd>${util(cf.actual.max_util, 1)}</dd>
          <dt>Δ busiest-link utilization</dt><dd>${signed(cf.delta_max_util * 100, 2)} pp</dd>
          <dt>SLA-violating demand-intervals, no change / actual</dt>
          <dd>${cf.noop.sla_violations} / ${cf.actual.sla_violations}</dd>
        </dl>`;
    }

    card.innerHTML = `
      <h4><span>${esc(algoLabel(run.algorithm))} — interval ${dec.step}</span>${badge}</h4>
      <div class="dec-action">${esc(text)}</div>
      <div class="tech-line mono">${esc(algoTech(run.algorithm))} · ${esc(technical)}</div>
      ${routeHtml}
      ${dec.action_probability !== undefined
        ? `<dl class="kv"><dt>action probability</dt><dd>${(dec.action_probability * 100).toFixed(1)}%</dd>
           <dt>valid actions in mask</dt><dd>${dec.mask_valid_actions}</dd></dl>` : ""}
      <div class="dec-explan"><span class="tag">engineering interpretation — computed from telemetry, not the network's internal reasoning</span>
        ${esc(dec.explanation || "")}</div>
      <h4>Reward ${signed(dec.reward, 3)}
        <span class="mono">Σ ${fmtReward(dec.cumulative_reward)}</span></h4>
      ${compRows}
      ${probRows}
      ${cfHtml}`;
    container.appendChild(card);
  }
}

// -------------------------------------------------------------- scoreboard
/** Totals derived from the per-interval history Charts already accumulates. */
export function totalsFor(hist) {
  const t = { intervals: hist.length, sla: 0, reroutes: 0, flaps: 0,
              delivered: 0, maxUtil: 0, churn: 0 };
  hist.forEach((h, i) => {
    t.sla += h.sla_violations || 0;
    t.reroutes += h.reroutes || 0;
    t.flaps += h.flaps || 0;
    t.delivered += h.delivered_ratio || 0;
    t.maxUtil = Math.max(t.maxUtil, h.max_util || 0);
    if (i >= hist.length - CHURN_WINDOW) t.churn += h.reroutes || 0;
  });
  t.deliveredMean = t.intervals ? t.delivered / t.intervals : 0;
  return t;
}

/**
 * Live scoreboard. `runs` comes from the WS payload, `histories` is
 * charts.history (algo -> [interval metrics]). Deltas are ABSOLUTE reward
 * points — never a percentage, because baseline totals are often negative and
 * a ratio would be meaningless.
 */
export function renderScoreboard(container, runs, histories, status) {
  if (!runs || !runs.length) {
    container.innerHTML =
      `<div class="placeholder">Start a session to see the live scoreboard.</div>`;
    return;
  }
  const cols = runs.map((run) => {
    const hist = histories[run.algorithm] || [];
    const t = totalsFor(hist);
    const m = (run.snapshot && run.snapshot.metrics) || {};
    const cum = run.decision ? run.decision.cumulative_reward : 0;
    const slaNow = run.snapshot
      ? run.snapshot.demands.filter((d) => !d.sla_ok || d.disconnected).length : 0;
    return { run, t, m, cum, slaNow, mean: t.intervals ? cum / t.intervals : 0 };
  });

  let deltaHtml = "";
  if (cols.length === 2) {
    const [a, b] = cols;
    const d = a.cum - b.cum;
    const lead = d >= 0 ? a : b;
    const trail = d >= 0 ? b : a;
    const cls = Math.abs(d) < 5 ? "neutral" : (d >= 0 ? "good" : "bad");
    // "ahead by N reward points" — absolute, direction stated explicitly.
    deltaHtml = `<div class="score-delta ${cls}">
      ${Math.abs(d) < 5
        ? `${esc(algoLabel(a.run.algorithm))} and ${esc(algoLabel(b.run.algorithm))} are level
           (within ${fmtReward(Math.abs(d))} reward points)`
        : `${esc(algoLabel(lead.run.algorithm))} ahead by
           ${fmtReward(Math.abs(d))} reward points vs ${esc(algoLabel(trail.run.algorithm))}`}
    </div>`;
  }

  const churn = cols.filter((c) => c.t.churn >= CHURN_THRESHOLD);
  const churnHtml = churn.length ? `<div class="warn-badge" role="status">
      ⚠ High route churn — ${churn.map((c) =>
        `${esc(algoLabel(c.run.algorithm))} made ${c.t.churn} route changes in the
         last ${Math.min(CHURN_WINDOW, c.t.intervals)} intervals`).join("; ")}.
      This policy trades stability for utilization.</div>` : "";

  const row = (label, get, title = "") => `<tr title="${esc(title)}">
      <th scope="row">${esc(label)}</th>${cols.map((c) =>
        `<td class="num">${get(c)}</td>`).join("")}</tr>`;

  container.innerHTML = `
    ${deltaHtml}
    ${churnHtml}
    <table class="score-tbl"><thead><tr><th></th>${cols.map((c) =>
      `<th class="algo-${esc(c.run.algorithm)}">${esc(algoLabel(c.run.algorithm))}</th>`).join("")}
    </tr></thead><tbody>
      ${row("Total reward", (c) => fmtReward(c.cum),
            "Cumulative simulation score — a benchmark score, not money or an industry KPI.")}
      ${row("Mean reward / interval", (c) => fmtReward(c.mean))}
      ${row("Busiest link now", (c) => util(c.m.max_util || 0))}
      ${row("Peak busiest link", (c) => util(c.t.maxUtil))}
      ${row("Services with SLA problems now", (c) => c.slaNow,
            "Demands currently over their latency or loss target, or disconnected.")}
      ${row("Demand-interval SLA violations (total)", (c) => c.t.sla,
            "One count per demand per 5-minute interval that missed its SLA target.")}
      ${row("Traffic delivered (mean)", (c) => ratio(c.t.deliveredMean))}
      ${row("Route changes (total)", (c) => c.t.reroutes)}
      ${row("Route flaps (total)", (c) => c.t.flaps,
            "A reroute back to a path the demand recently left.")}
    </tbody></table>
    <div class="hint">Simulated time ${esc(simTime(status ? status.hour : 0))} ·
      interval ${status ? status.step : 0} of
      ${status ? Math.floor(status.duration_min / 5) : 0} ·
      one seed. Multi-seed results are in the Benchmark tab.</div>`;
}

// --------------------------------------------------------------- benchmark
export function renderBenchmark(container, bench, scenarioKey) {
  if (!bench) {
    container.innerHTML = `<div class="placeholder">Published results unavailable.</div>`;
    return;
  }
  const entry = bench.scenarios[scenarioKey];
  if (!entry) {
    // Some scenarios (the guided demo) are presentation-only and were never
    // part of the 5-seed evaluation. Say so, and show the overview instead of
    // an empty panel.
    container.innerHTML = `
      <div class="hint honest">This scenario is a guided demonstration and was not
        part of the published 5-seed evaluation. The table below is the published
        result for every scenario that was.</div>
      ${benchmarkOverviewHtml(bench)}
      <div class="hint mono">source: ${esc(bench.source)}</div>`;
    return;
  }
  const algos = Object.entries(entry.algorithms)
    .sort(([, a], [, b]) => b.reward_mean - a.reward_mean);
  container.innerHTML = `
    <h3>${esc(entry.display_name)}</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>controller</th><th>reward (mean ± CI95)</th>
        <th>busiest link</th><th>demand-interval SLA violations</th>
        <th>route changes</th></tr></thead>
      <tbody>${algos.map(([a, s]) => `
        <tr class="${a === entry.winner ? "winner" : ""}">
          <td>${esc(algoLabel(a))}${a === entry.winner ? ' <span class="badge ok">best</span>' : ""}
            <span class="tech-line mono">${esc(a)}</span></td>
          <td class="num">${s.reward_mean.toFixed(1)} ± ${s.reward_ci95.toFixed(1)}</td>
          <td class="num">${util(s.max_util_mean, 1)}</td>
          <td class="num">${s.sla_violations_mean.toFixed(1)}</td>
          <td class="num">${s.reroutes_mean.toFixed(1)}</td>
        </tr>`).join("")}
      </tbody></table></div>
    <div class="hint honest">Live run = one seed; benchmark = 5-seed mean ± CI.
      RL does not beat greedy everywhere — strongest on Normal Day and Hidden
      Bottleneck; greedy wins several reactive incidents.</div>
    ${benchmarkOverviewHtml(bench)}
    <div class="hint mono">source: ${esc(bench.source)}</div>`;
}

/**
 * Compact per-scenario winner table across the whole published evaluation.
 * Shows RL and the best non-RL controller side by side so the panel can never
 * be read as "RL wins everywhere" — because it does not.
 */
export function benchmarkOverviewHtml(bench) {
  const rows = Object.entries(bench.scenarios).map(([key, s]) => {
    const rl = s.algorithms.rl;
    const rivals = Object.entries(s.algorithms).filter(([a]) => a !== "rl");
    const best = rivals.sort(([, x], [, y]) => y.reward_mean - x.reward_mean)[0];
    const rlWins = s.winner === "rl";
    return `<tr>
      <td>${esc(s.display_name)}<span class="tech-line mono">${esc(key)}</span></td>
      <td class="num ${rlWins ? "sla-ok" : ""}">${rl ? rl.reward_mean.toFixed(1) : "—"}</td>
      <td class="num ${rlWins ? "" : "sla-ok"}">${best ? best[1].reward_mean.toFixed(1) : "—"}
        <span class="tech-line mono">${best ? esc(algoLabel(best[0])) : ""}</span></td>
      <td>${esc(algoLabel(s.winner))}</td>
    </tr>`;
  }).join("");
  return `<h3>Published results, every scenario</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>scenario</th><th>AI Advisor reward</th>
        <th>best other controller</th><th>winner</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

// ------------------------------------------------------------------ events
export function renderEvents(container, events) {
  if (!events || !events.length) {
    container.innerHTML = `<div class="placeholder">No backend events yet.</div>`;
    return;
  }
  container.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>time</th><th>event</th><th>detail</th></tr></thead>
    <tbody>${[...events].reverse().map((e) => {
      const detail = Object.entries(e)
        .filter(([k, v]) => !["ts", "event", "scenario", "algorithm", "seed"].includes(k)
                            && v !== null && v !== undefined)
        .map(([k, v]) => `${k}=${v}`).join(" ");
      return `<tr><td>${esc(e.ts)}</td><td>${esc(e.event)}</td>
        <td class="mono ev-detail">${esc(detail)}</td></tr>`;
    }).join("")}</tbody></table></div>`;
}

// ------------------------------------------------------------------ tables
export function renderLspTable(tbl, runs, onSelect, selectedId) {
  const run = runs[0];
  if (!run) return;
  const rows = run.snapshot.demands.map((d) => {
    const cc = CLASS_COLORS[d.class] || "#888";
    const sla = d.disconnected
      ? `<span class="sla-bad">DISCONNECTED</span>`
      : d.sla_ok ? `<span class="sla-ok">OK</span>` : `<span class="sla-bad">VIOLATED</span>`;
    return `<tr data-id="${esc(d.id)}" class="${d.id === selectedId ? "selected" : ""}">
      <td>${esc(demandLabel(d.src, d.dst, d.class))}<span class="tech-line mono">${esc(d.id)}</span></td>
      <td><span class="chip-class" style="background:${cc}">${esc(d.class)}</span></td>
      <td class="num">${esc(rate(d.volume_mbps))}</td>
      <td class="route-cell">${esc(pathLabel(d.current_path))}</td>
      <td class="num">${esc(delay(d.delay_ms))}</td>
      <td class="num">${esc(lossPct(d.loss_pct))}</td>
      <td class="num">${util(d.bottleneck_util)}</td>
      <td>${sla}</td>
      <td class="num">${d.path_changes}</td>
    </tr>`;
  }).join("");
  tbl.innerHTML = `<thead><tr>
    <th>traffic demand</th><th>class</th><th>volume</th><th>route</th>
    <th>delay</th><th>loss</th><th>busiest hop</th>
    <th title="Current SLA state of this demand">SLA</th><th>route changes</th>
  </tr></thead><tbody>${rows}</tbody>`;
  tbl.querySelectorAll("tbody tr").forEach((tr) =>
    tr.addEventListener("click", () => onSelect(tr.dataset.id)));
}

export function renderLinkTableHtml(tbl, runs) {
  const algoNames = runs.map((r) => algoLabel(r.algorithm));
  const rows = [];
  const first = runs[0].snapshot.links;
  for (let i = 0; i < first.length; i++) {
    const l = first[i];
    const other = runs[1] ? runs[1].snapshot.links[i] : null;
    const cls = !l.up ? "sla-bad" : (l.congested ? "sla-bad" : "");
    rows.push(`<tr>
      <td>${esc(linkLabel(l.link))} <span class="dir">${esc(city(l.src))}→${esc(city(l.dst))}</span>
        <span class="tech-line mono">${esc(linkTechnical(l.link))}</span></td>
      <td class="num">${esc(rate(l.capacity_mbps))}</td>
      <td class="num">${esc(rate(l.load_mbps))}</td>
      <td class="num ${cls}">${util(l.utilization, 1)}</td>
      ${other ? `<td class="num">${esc(rate(other.load_mbps))}</td>
                 <td class="num ${other.congested ? "sla-bad" : ""}">${util(other.utilization, 1)}</td>
                 <td class="num">${signed((l.utilization - other.utilization) * 100, 1)}</td>` : ""}
      <td class="num">${esc(delay(l.queue_delay_ms))}</td>
      <td class="num">${esc(loss(l.loss_fraction))}</td>
      <td class="num">${l.n_lsps}</td>
      <td>${l.up ? (l.congested ? "⚠ congested" : "up") : "✖ failed"}</td>
    </tr>`);
  }
  tbl.innerHTML = `<thead><tr>
    <th>link (one direction)</th><th>capacity</th>
    <th>${esc(algoNames[0])} load</th><th>${esc(algoNames[0])} util</th>
    ${runs[1] ? `<th>${esc(algoNames[1])} load</th><th>${esc(algoNames[1])} util</th><th>Δutil pp</th>` : ""}
    <th>queue delay</th><th>loss</th><th>LSPs</th><th>state</th>
  </tr></thead><tbody>${rows.join("")}</tbody>`;
}

// ---------------------------------------------------------------- training
export function renderCheckpoints(tbl, cps) {
  tbl.innerHTML = `<thead><tr><th>tag</th><th>file</th><th>MB</th><th>modified</th></tr></thead>
    <tbody>${cps.map((c) => `<tr><td>${esc(c.tag)}</td><td>${esc(c.file)}</td>
      <td class="num">${c.size_mb}</td><td class="num">${esc(c.modified)}</td></tr>`).join("")}</tbody>`;
}

export function renderRuns(tbl, runs) {
  tbl.innerHTML = `<thead><tr><th>id</th><th>time</th><th>scenario</th><th>controller</th>
    <th>seed</th><th>busiest link</th><th>delay</th>
    <th title="One count per demand per 5-minute interval">demand-interval SLA violations</th>
    <th>route changes</th></tr></thead>
    <tbody>${runs.map((r) => `<tr><td class="num">${r.id}</td>
      <td class="num">${esc(r.created_at)}</td><td>${esc(r.scenario)}</td>
      <td>${esc(algoLabel(r.algorithm))}</td><td class="num">${r.seed}</td>
      <td class="num">${util(r.summary.max_util_mean ?? 0, 1)}</td>
      <td class="num">${esc(delay(r.summary.mean_delay_ms ?? 0))}</td>
      <td class="num">${r.summary.sla_violations_total ?? "—"}</td>
      <td class="num">${r.summary.reroutes_total ?? "—"}</td></tr>`).join("")}</tbody>`;
}
