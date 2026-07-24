// Right-rail panels: agent decision card, LSP/link tables, decision tape,
// training monitor, saved runs.

const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const CLASS_COLORS = { voice: "#e15759", video: "#4e79a7", vpn: "#59a14f",
                       besteffort: "#9c755f", bulk: "#f28e2b", critical: "#b07aa1" };

// ------------------------------------------------------------ decision tape
export function tapeAppend(algo, decision, hour) {
  const box = document.getElementById("tape-lines");
  const hh = String(Math.floor(hour)).padStart(2, "0");
  const mm = String(Math.round((hour % 1) * 60)).padStart(2, "0");
  let cls = "ok", msg;
  if (decision.decoded) {
    const d = decision.decoded;
    if (d.type === "noop") { msg = "no-op"; cls = "ok"; }
    else if (d.accepted === false) { msg = `REJECTED ${d.demand}→p${d.path_idx} (${d.reason})`; cls = "rej"; }
    else { msg = `reroute ${d.demand}: p${d.from_path}→p${d.path_idx}`; cls = "warn"; }
  } else {
    const moves = decision.moves || [];
    msg = moves.length
      ? "reroute " + moves.map((m) => `${m.demand}→p${m.path_idx}`).join(", ")
      : "no-op";
    cls = moves.length ? "warn" : "ok";
  }
  const line = document.createElement("div");
  line.className = `tape-line ${cls}`;
  const who = decision.algorithm === "rl" ? "who-rl" : "who-b";
  line.innerHTML =
    `<span class="t">${hh}:${mm}</span>` +
    `<span class="${who}">${esc(decision.algorithm).padEnd(6)}</span>` +
    `<span class="msg">${esc(msg)}</span>` +
    `<span class="t">r=${decision.reward >= 0 ? "+" : ""}${decision.reward.toFixed(2)}</span>`;
  box.prepend(line);
  while (box.children.length > 80) box.removeChild(box.lastChild);
}

// ------------------------------------------------------ agent decision card
export function renderDecision(container, runs) {
  container.innerHTML = "";
  for (const run of runs) {
    const dec = run.decision;
    if (!dec) continue;
    const card = document.createElement("div");
    card.className = "dec-card";
    let badge = `<span class="badge noop">NO-OP</span>`;
    let actionTxt = "hold current LSP placement";
    if (dec.decoded && dec.decoded.type === "reroute") {
      badge = dec.decoded.accepted
        ? `<span class="badge ok">ACCEPTED</span>`
        : `<span class="badge rej">REJECTED — ${esc(dec.decoded.reason)}</span>`;
      actionTxt = `${dec.decoded.demand}: path ${dec.decoded.from_path} → ${dec.decoded.path_idx}`;
    } else if (dec.moves) {
      actionTxt = dec.moves.length
        ? dec.moves.map((m) => `${m.demand}→p${m.path_idx}`).join(", ")
        : "no reroute";
      badge = dec.moves.length ? `<span class="badge ok">APPLIED</span>` : `<span class="badge noop">NO-OP</span>`;
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
        <span class="val">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span>
      </div>`).join("");
    let cfHtml = "";
    if (dec.counterfactual) {
      const cf = dec.counterfactual;
      cfHtml = `<h4>Counterfactual (post-hoc)</h4>
        <dl class="kv">
          <dt>max util if no-op</dt><dd>${(cf.noop.max_util * 100).toFixed(1)}%</dd>
          <dt>max util actual</dt><dd>${(cf.actual.max_util * 100).toFixed(1)}%</dd>
          <dt>Δ max util</dt><dd>${cf.delta_max_util >= 0 ? "+" : ""}${(cf.delta_max_util * 100).toFixed(2)} pp</dd>
          <dt>SLA viol. no-op / actual</dt><dd>${cf.noop.sla_violations} / ${cf.actual.sla_violations}</dd>
        </dl>`;
    }
    card.innerHTML = `
      <h4><span>${esc(run.algorithm)} — step ${dec.step}</span>${badge}</h4>
      <div class="dec-action">${esc(actionTxt)}</div>
      ${dec.action_probability !== undefined
        ? `<dl class="kv"><dt>action probability</dt><dd>${(dec.action_probability * 100).toFixed(1)}%</dd>
           <dt>valid actions in mask</dt><dd>${dec.mask_valid_actions}</dd></dl>` : ""}
      <div class="dec-explan"><span class="tag">engineering interpretation — computed from telemetry, not the network's internal reasoning</span>
        ${esc(dec.explanation || "")}</div>
      <h4>Reward ${dec.reward >= 0 ? "+" : ""}${dec.reward.toFixed(3)}
        <span class="mono">Σ ${dec.cumulative_reward.toFixed(1)}</span></h4>
      ${compRows}
      ${probRows}
      ${cfHtml}`;
    container.appendChild(card);
  }
}

// ------------------------------------------------------------------ tables
export function renderLspTable(tbl, runs, onSelect, selectedId) {
  const run = runs[0];
  if (!run) return;
  const rows = run.snapshot.demands.map((d) => {
    const cc = CLASS_COLORS[d.class] || "#888";
    const sla = d.disconnected
      ? `<span class="sla-bad">DISC</span>`
      : d.sla_ok ? `<span class="sla-ok">OK</span>` : `<span class="sla-bad">VIOL</span>`;
    return `<tr data-id="${d.id}" class="${d.id === selectedId ? "selected" : ""}">
      <td>${d.id}</td>
      <td><span class="chip-class" style="background:${cc}">${d.class}</span></td>
      <td>${d.src}→${d.dst}</td>
      <td class="num">${d.volume_mbps.toFixed(0)}</td>
      <td class="num">${esc(d.current_path.join("·"))}</td>
      <td class="num">${d.delay_ms.toFixed(1)}</td>
      <td class="num">${d.loss_pct.toFixed(2)}</td>
      <td class="num">${(d.bottleneck_util * 100).toFixed(0)}%</td>
      <td>${sla}</td>
      <td class="num">${d.path_changes}</td>
    </tr>`;
  }).join("");
  tbl.innerHTML = `<thead><tr>
    <th>LSP</th><th>class</th><th>route</th><th>Mbps</th><th>path</th>
    <th>delay ms</th><th>loss %</th><th>bneck</th><th>SLA</th><th>moves</th>
  </tr></thead><tbody>${rows}</tbody>`;
  tbl.querySelectorAll("tbody tr").forEach((tr) =>
    tr.addEventListener("click", () => onSelect(tr.dataset.id)));
}

export function renderLinkTableHtml(tbl, runs) {
  const algoNames = runs.map((r) => r.algorithm);
  const rows = [];
  const first = runs[0].snapshot.links;
  for (let i = 0; i < first.length; i++) {
    const l = first[i];
    const other = runs[1] ? runs[1].snapshot.links[i] : null;
    const cls = !l.up ? "sla-bad" : (l.congested ? "sla-bad" : "");
    rows.push(`<tr>
      <td>${l.id}</td>
      <td class="num">${l.capacity_mbps}</td>
      <td class="num">${l.load_mbps.toFixed(0)}</td>
      <td class="num ${cls}">${(l.utilization * 100).toFixed(1)}%</td>
      ${other ? `<td class="num">${other.load_mbps.toFixed(0)}</td>
                 <td class="num ${other.congested ? "sla-bad" : ""}">${(other.utilization * 100).toFixed(1)}%</td>
                 <td class="num">${((l.utilization - other.utilization) * 100).toFixed(1)}</td>` : ""}
      <td class="num">${l.queue_delay_ms.toFixed(2)}</td>
      <td class="num">${(l.loss_fraction * 100).toFixed(2)}</td>
      <td class="num">${l.n_lsps}</td>
      <td>${l.up ? (l.congested ? "⚠ cong" : "up") : "✖ down"}</td>
    </tr>`);
  }
  tbl.innerHTML = `<thead><tr>
    <th>dlink</th><th>cap</th><th>${algoNames[0]} load</th><th>${algoNames[0]} util</th>
    ${runs[1] ? `<th>${algoNames[1]} load</th><th>${algoNames[1]} util</th><th>Δutil pp</th>` : ""}
    <th>qdel</th><th>loss%</th><th>lsps</th><th>state</th>
  </tr></thead><tbody>${rows.join("")}</tbody>`;
}

// ---------------------------------------------------------------- training
export function renderCheckpoints(tbl, cps) {
  tbl.innerHTML = `<thead><tr><th>tag</th><th>file</th><th>MB</th><th>modified</th></tr></thead>
    <tbody>${cps.map((c) => `<tr><td>${esc(c.tag)}</td><td>${esc(c.file)}</td>
      <td class="num">${c.size_mb}</td><td class="num">${esc(c.modified)}</td></tr>`).join("")}</tbody>`;
}

export function renderRuns(tbl, runs) {
  tbl.innerHTML = `<thead><tr><th>id</th><th>time</th><th>scenario</th><th>algo</th>
    <th>seed</th><th>maxU</th><th>delay</th><th>SLA viol</th><th>reroutes</th></tr></thead>
    <tbody>${runs.map((r) => `<tr><td class="num">${r.id}</td>
      <td class="num">${esc(r.created_at)}</td><td>${esc(r.scenario)}</td>
      <td>${esc(r.algorithm)}</td><td class="num">${r.seed}</td>
      <td class="num">${(r.summary.max_util_mean ?? 0).toFixed(3)}</td>
      <td class="num">${(r.summary.mean_delay_ms ?? 0).toFixed(1)}</td>
      <td class="num">${r.summary.sla_violations_total ?? "—"}</td>
      <td class="num">${r.summary.reroutes_total ?? "—"}</td></tr>`).join("")}</tbody>`;
}
