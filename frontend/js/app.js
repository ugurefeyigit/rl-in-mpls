// Advanced console bootstrap: state, WebSocket, control wiring, render loop.
//
// Every user-facing string routes through display.js (city names) and fmt.js
// (number formatting). Internal IDs stay visible only in the dimmed technical
// lines and hover cards, because tests, configs and the pretrained model
// depend on them.

import { api, toast } from "./api.js";
import { TopoView } from "./topo.js";
import { Charts } from "./charts.js";
import {
  loadDisplay, scenarioLabel, algoLabel, city, linkFull, demandFull,
  disclaimer, pathLabel, demandLabel,
} from "./display.js";
import { simTime, rate, util, esc } from "./fmt.js";
import {
  renderBenchmark, renderCheckpoints, renderDecision, renderEvents,
  renderLspTable, renderLinkTableHtml, renderRuns, renderScoreboard,
  tapeAppend,
} from "./panels.js";

const $ = (id) => document.getElementById(id);

const state = {
  topology: null,
  scenarios: {},
  demands: [],
  benchmark: null,
  lastPayload: null,
  status: { state: "idle", running: false },
  advisor: null,          // pending proposal, or null
  selectedDemand: null,
  activeTab: "score",
  compare: false,
};

const hover = $("hovercard");
const topoA = new TopoView("cy-a", hover);
const topoB = new TopoView("cy-b", hover);
const charts = new Charts();

function phaseFor(hour) {
  if (hour < 6) return "night";
  if (hour < 12) return "morning";
  if (hour < 17) return "afternoon";
  if (hour < 22) return "evening";
  return "night";
}

// ------------------------------------------------------------ state machine
const STATE_TEXT = {
  idle: "idle", running: "running", paused: "paused",
  completed: "completed", error: "error",
};

function applyStatus(st) {
  state.status = st;
  const s = st.state || (st.running ? "running" : "idle");
  const chip = $("state-chip");
  chip.className = `state-chip state-${s}`;
  chip.textContent = STATE_TEXT[s] || s;
  const errBox = $("state-error");
  errBox.classList.toggle("hidden", !st.error);
  errBox.textContent = st.error || "";

  const hasSession = st.scenario !== undefined && st.scenario !== null;
  const pending = Boolean(st.awaiting_decision);
  const done = Boolean(st.done) || s === "completed";
  const err = s === "error";

  const dis = (id, off) => { $(id).disabled = off; };
  dis("btn-pause", !hasSession || s !== "running" || err);
  dis("btn-resume", !hasSession || err || done || pending || s === "running");
  dis("btn-step", !hasSession || err || done || pending || s === "running");
  dis("btn-reset", !hasSession);
  dis("btn-propose", !hasSession || err || done || pending
                     || !(st.algorithms || []).includes("rl"));
  dis("btn-approve", !pending);
  dis("btn-reject", !pending);
  for (const id of ["btn-fail", "btn-recover", "btn-burst", "btn-save-run"])
    dis(id, !hasSession || err);

  if (hasSession) {
    $("run-desc").textContent =
      `${scenarioLabel(st.scenario)} · seed ${st.seed} · ` +
      (st.algorithms || []).map(algoLabel).join(" vs ");
  }
  $("sim-clock").textContent = st.hour === undefined ? "--:--" : simTime(st.hour);
  $("sim-day-phase").textContent = st.hour === undefined
    ? "no session"
    : `${phaseFor(st.hour)} · interval ${st.step}/${Math.floor(st.duration_min / 5)}` +
      (done ? " · complete" : s === "running" ? "" : ` · ${s}`);
}

// ------------------------------------------------------------------ render
function onPayload(payload) {
  const isTick = payload.type === "tick";
  state.lastPayload = payload;
  applyStatus(payload.status);
  const st = payload.status;
  const runs = payload.runs || [];
  if (!runs.length) return;

  $("pane-title-a").innerHTML =
    `<span class="algo-a">${esc(algoLabel(runs[0].algorithm))}</span> — ` +
    `${esc(scenarioLabel(st.scenario))}`;
  topoA.update(runs[0].snapshot);
  if (runs[1]) {
    $("pane-title-b").innerHTML =
      `<span class="algo-b">${esc(algoLabel(runs[1].algorithm))}</span> — ` +
      `${esc(scenarioLabel(st.scenario))}`;
    topoB.update(runs[1].snapshot);
  }

  // Only genuine ticks advance the series; interventions and resets re-send
  // the last decision and would otherwise double-count it.
  if (isTick) {
    let gap = false;
    for (const run of runs) {
      const dec = run.decision;
      if (!dec || !run.snapshot.metrics) continue;
      if (dec.step !== run.snapshot.step) continue;
      if (dec.step <= charts.lastStep(run.algorithm)) continue;
      // A fast-forward (POST /api/simulation/run-until, or Presentation Mode
      // driving the same session) steps many intervals server-side but
      // broadcasts only the last payload. Refill from the authoritative record
      // rather than charting a series with holes in it.
      if (dec.step > charts.lastStep(run.algorithm) + 1) gap = true;
      else charts.push(run.algorithm, run.snapshot.metrics, dec);
      tapeAppend(run.algorithm, dec, st.hour, run.snapshot);
    }
    if (gap) { refreshHistories(); return; }
  }
  renderActiveTab();
}

function renderActiveTab() {
  const payload = state.lastPayload;
  if (state.activeTab === "benchmark") {
    renderBenchmark($("benchmark-body"), state.benchmark,
                    state.status.scenario || $("sel-scenario").value);
    return;
  }
  if (!payload) return;
  const runs = payload.runs || [];
  if (!runs.length) return;
  switch (state.activeTab) {
    case "score":
      renderScoreboard($("scoreboard"), runs, charts.history, state.status);
      renderAdvisorCard();
      break;
    case "agent":
      renderDecision($("agent-decision"), runs);
      break;
    case "metrics":
      charts.render();
      break;
    case "matrix":
      charts.renderMatrix(runs[0].snapshot, (src, dst) => {
        const list = runs[0].snapshot.demands
          .filter((d) => d.src === src && d.dst === dst)
          .map((d) => `${d.id} (${d.class}, ${rate(d.volume_mbps)}, ` +
                      `${d.sla_ok ? "SLA ok" : "SLA VIOLATED"})`);
        $("matrix-detail").textContent = list.length
          ? `${city(src)} → ${city(dst)}: ${list.join(" · ")}`
          : `${city(src)} → ${city(dst)}: no demands`;
      });
      break;
    case "lsps":
      renderLspTable($("tbl-lsps"), runs, (id) => {
        state.selectedDemand = id === state.selectedDemand ? null : id;
        topoA.selectDemand(state.selectedDemand);
        topoB.selectDemand(state.selectedDemand);
        renderActiveTab();
      }, state.selectedDemand);
      break;
    case "links":
      renderLinkTableHtml($("tbl-links"), runs);
      break;
  }
}

// ------------------------------------------------------------ advisor card
function renderAdvisorCard() {
  const box = $("advisor-card");
  const p = state.advisor;
  if (!p) { box.innerHTML = ""; return; }
  const la = p.lookahead || {};
  const d = p.decoded;
  let effect = "";
  if (la.noop && la.action) {
    effect = `<dl class="kv">
      <dt>busiest link if no change</dt><dd>${util(la.noop.max_util, 1)}</dd>
      <dt>busiest link if applied</dt><dd>${util(la.action.max_util, 1)}</dd>
      <dt>Δ busiest link</dt><dd>${(la.delta_max_util * 100).toFixed(1)} pp</dd>
    </dl>`;
  } else if (la.noop) {
    effect = `<dl class="kv"><dt>busiest link if no change</dt>
      <dd>${util(la.noop.max_util, 1)}</dd></dl>`;
  }
  box.innerHTML = `<div class="advisor-card">
    <h4>Recommendation pending <span class="badge ${p.safety_ok ? "ok" : "rej"}">
      ${p.safety_ok ? "safety check passed" : "safety check failed"}</span></h4>
    <div class="dec-action">${p.is_noop
      ? "Hold the current routing — no change recommended."
      : esc(`Move ${demandLabel(d.src, d.dst, d.class)} ` +
            `(${rate(d.volume_mbps)}) to ${pathLabel(d.to_routers)}`)}</div>
    ${p.is_noop ? "" : `<dl class="kv route-kv">
      <dt>current route</dt><dd>${esc(pathLabel(d.from_routers))}</dd>
      <dt>proposed route</dt><dd>${esc(pathLabel(d.to_routers))}</dd></dl>`}
    ${effect}
    <div class="tech-line mono">proposal #${p.id} · interval ${p.step} ·
      action ${p.action}${d ? ` · ${d.demand}→p${d.path_idx}` : ""} ·
      ${p.safety_reason}</div>
    <div class="hint">Use Approve / Reject in the control rail. Approving applies
      exactly this action; rejecting applies no change. Both advance one interval.</div>
  </div>`;
}

// --------------------------------------------------------------- websocket
let ws = null;
let wsRetry = 0;
function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/telemetry`);
  ws.onopen = () => { wsRetry = 0; $("conn-dot").classList.add("ok"); ws.send("hi"); };
  ws.onclose = () => {
    $("conn-dot").classList.remove("ok");
    wsRetry += 1;
    setTimeout(connectWs, Math.min(1500 * wsRetry, 10000));
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const payload = JSON.parse(ev.data);
    switch (payload.type) {
      case "advisor":
        state.advisor = payload.proposal;
        applyStatus(payload.status);
        highlightProposal(payload.proposal);
        renderActiveTab();
        toast(payload.proposal.is_noop
          ? "Recommendation: hold the current routing."
          : "Recommendation ready — approve or reject.");
        break;
      case "reset":
        state.advisor = null;
        clearProposalHighlight();
        onPayload(payload);
        break;
      case "status":
        applyStatus(payload.status);
        if (payload.status.state === "error")
          toast(`Session error: ${payload.status.error}`, true);
        else if (payload.status.done)
          toast(`Scenario complete: ${scenarioLabel(payload.status.scenario)}`);
        break;
      default:              // tick | intervention
        // The socket replays a "tick" on every (re)connect, so the message type
        // alone must not clear a still-pending recommendation.
        if (payload.type === "tick" && !payload.status.awaiting_decision) {
          state.advisor = null;
          clearProposalHighlight();
        }
        onPayload(payload);
    }
  };
}

function highlightProposal(p) {
  if (p && p.decoded) topoA.showProposedPath(p.decoded.to_routers);
}
function clearProposalHighlight() {
  topoA.showProposedPath(null);
}

// ---------------------------------------------------------------- controls
async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); }
}

/** Refresh the accumulated series from the authoritative server-side record. */
async function refreshHistories() {
  try {
    const hist = await api.metricsHistory();
    charts.setHistories(hist.runs);
    if (state.activeTab === "metrics") charts.render();
    renderActiveTab();
  } catch { /* no session yet */ }
}

function wireControls() {
  $("sel-mode").addEventListener("change", () => {
    state.compare = $("sel-mode").value === "compare";
    $("algo-b-wrap").classList.toggle("hidden", !state.compare);
    $("topo-pane-b").classList.toggle("hidden", !state.compare);
    setTimeout(() => { topoA.cy?.resize(); topoA.cy?.fit(undefined, 30);
                       topoB.cy?.resize(); topoB.cy?.fit(undefined, 30); }, 50);
  });
  $("sel-scenario").addEventListener("change", () => {
    const s = state.scenarios[$("sel-scenario").value];
    $("scenario-desc").textContent = s ? s.description : "";
    if (state.activeTab === "benchmark") renderActiveTab();
  });

  $("btn-start").addEventListener("click", () => guard(async () => {
    const algorithms = [$("sel-algo-a").value];
    if (state.compare) algorithms.push($("sel-algo-b").value);
    const advisor = $("chk-advisor").checked;
    charts.reset();
    $("tape-lines").innerHTML = "";
    state.selectedDemand = null;
    state.advisor = null;
    const st = await api.start({
      scenario: $("sel-scenario").value,
      algorithms,
      seed: parseInt($("inp-seed").value, 10) || 42,
      model_tag: $("sel-model").value || null,
      safety_filter: $("chk-safety").checked,
      speed: document.querySelector(".speed.active")?.dataset.speed || "1x",
      autostart: !advisor,
      advisor,
      interface_mode: "advanced",
    });
    applyStatus(st);
    toast(`Session started: ${scenarioLabel(st.scenario)} — ` +
          `${st.algorithms.map(algoLabel).join(" vs ")}, seed ${st.seed}`);
  }));

  $("btn-pause").addEventListener("click", () => guard(async () =>
    applyStatus(await api.pause())));
  $("btn-resume").addEventListener("click", () => guard(async () =>
    applyStatus(await api.resume())));
  $("btn-step").addEventListener("click", () => guard(api.step));
  $("btn-reset").addEventListener("click", () => guard(async () => {
    charts.reset(); $("tape-lines").innerHTML = ""; state.advisor = null;
    const st = await api.reset();
    applyStatus(st);
    // Reset preserves the FULL configuration — say so explicitly.
    toast(`Reset to interval 0. Configuration preserved: ` +
          `${scenarioLabel(st.scenario)}, ${st.algorithms.map(algoLabel).join(" vs ")}, ` +
          `seed ${st.seed}, model ${st.model_tag || "none"}, ` +
          `safety filter ${st.safety_filter ? "on" : "off"}, speed ${st.speed}.`);
  }));

  $("btn-propose").addEventListener("click", () => guard(async () => {
    const p = await api.advisorPropose();
    state.advisor = p;
    highlightProposal(p);
    renderActiveTab();
  }));
  $("btn-approve").addEventListener("click", () => guard(async () => {
    const rec = await api.advisorApprove();
    state.advisor = null; clearProposalHighlight();
    toast(`Approved. Busiest link after the change: ` +
          `${util(rec.actual.max_util, 1)} ` +
          `(predicted ${rec.lookahead.action ? util(rec.lookahead.action.max_util, 1) : "n/a"}).`);
  }));
  $("btn-reject").addEventListener("click", () => guard(async () => {
    const rec = await api.advisorReject();
    state.advisor = null; clearProposalHighlight();
    toast(`Rejected — no change applied. Busiest link: ${util(rec.actual.max_util, 1)}.`);
  }));

  $("speed-row").addEventListener("click", (ev) => {
    const b = ev.target.closest(".speed");
    if (!b) return;
    document.querySelectorAll(".speed").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    guard(() => api.speed(b.dataset.speed));
  });

  $("btn-fail").addEventListener("click", () => guard(async () => {
    const id = $("sel-fail-link").value;
    const r = await api.failLink(id);
    toast(r.changed
      ? `${linkFull(id)} failed — ${r.frr_reroutes} fast-reroute move(s).`
      : `${linkFull(id)} was already failed — no change.`, !r.changed);
  }));
  $("btn-recover").addEventListener("click", () => guard(async () => {
    const id = $("sel-fail-link").value;
    const r = await api.recoverLink(id);
    toast(r.changed
      ? `${linkFull(id)} recovered. Failed links remaining: ${r.failed_links.length}.`
      : `${linkFull(id)} was already up — no change.`, !r.changed);
  }));
  $("btn-burst").addEventListener("click", () => guard(async () => {
    const id = $("sel-burst-demand").value;
    const f = parseFloat($("inp-burst-factor").value) || 2.0;
    await api.burst(id, f, 60);
    const d = state.demands.find((x) => x.id === id);
    toast(`60-minute ×${f.toFixed(1)} burst injected on ` +
          `${d ? `${city(d.src)} → ${city(d.dst)} ${d.class} traffic` : id}.`);
  }));
  $("inp-mult").addEventListener("input", () => {
    $("mult-val").textContent = " " + parseFloat($("inp-mult").value).toFixed(2);
  });
  $("inp-mult").addEventListener("change", () => guard(async () => {
    const f = parseFloat($("inp-mult").value);
    await api.multiplier(f);
    toast(`Global demand multiplier set to ×${f.toFixed(2)}.`);
  }));

  $("btn-save-run").addEventListener("click", () => guard(async () => {
    const r = await api.saveRun();
    toast(`Saved run summaries: ${r.saved_run_ids.join(", ")}`);
  }));

  document.querySelectorAll("#tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#tabs button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      document.querySelectorAll(".tab-body").forEach((x) => x.classList.add("hidden"));
      $(`tab-${b.dataset.tab}`).classList.remove("hidden");
      state.activeTab = b.dataset.tab;
      if (state.activeTab === "training") refreshTraining();
      if (state.activeTab === "runs") refreshRuns();
      if (state.activeTab === "events") refreshEvents();
      renderActiveTab();
      setTimeout(() => charts.render(), 30);
    }));

  $("sel-metric").addEventListener("change", () => {
    charts.metricKey = $("sel-metric").value;
    charts.render();
  });

  $("btn-train").addEventListener("click", () => guard(async () => {
    const steps = parseInt($("inp-train-steps").value, 10) || 100000;
    const tag = $("inp-train-tag").value || "ppo_custom";
    const ok = window.confirm(
      `Start a NEW training job?\n\n` +
      `  timesteps: ${steps.toLocaleString()}\n  tag: ${tag}\n\n` +
      `This spawns a long-running background process on the server and writes ` +
      `to models/${tag}/. It does not touch the pretrained ppo_te model or any ` +
      `published results. Do not run this during a presentation.`);
    if (!ok) return;
    await api.trainStart({ timesteps: steps, tag,
                           seed: parseInt($("inp-seed").value, 10) || 42,
                           confirm: true });
    toast(`Training job started: ${tag}, ${steps.toLocaleString()} timesteps.`);
    refreshTraining();
  }));
  $("btn-refresh-runs").addEventListener("click", refreshRuns);
  $("btn-refresh-events").addEventListener("click", refreshEvents);
}

async function refreshTraining() {
  try {
    const p = await api.trainProgress();
    const blocked = p.allowed === false;
    $("train-disabled").classList.toggle("hidden", !blocked);
    $("btn-train").disabled = blocked;
    $("train-log").textContent = p.log.length
      ? p.log.join("\n")
      : (p.active ? "training starting…" : "no training job in this server session");
    renderCheckpoints($("tbl-checkpoints"), await api.checkpoints());
    if (p.active && state.activeTab === "training") setTimeout(refreshTraining, 3000);
  } catch { /* server restarting */ }
}

async function refreshRuns() {
  try { renderRuns($("tbl-runs"), await api.runs()); } catch { /* no db yet */ }
}

async function refreshEvents() {
  try { renderEvents($("events-body"), await api.events(80)); } catch { /* ignore */ }
}
let eventsTimer = null;

// --------------------------------------------------------------- bootstrap
async function boot() {
  await loadDisplay();
  state.topology = await api.topology();
  state.scenarios = await api.scenarios();
  const tc = await api.trafficClasses();
  state.demands = tc.demands;
  $("legend-disclaimer").textContent = disclaimer();

  const selScen = $("sel-scenario");
  for (const [name, s] of Object.entries(state.scenarios)) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = s.display_name || scenarioLabel(name);
    if (name === "demo_evening") o.selected = true;
    selScen.appendChild(o);
  }
  selScen.dispatchEvent(new Event("change"));

  const selLink = $("sel-fail-link");
  for (const l of state.topology.links) {
    const o = document.createElement("option");
    o.value = l.id;
    o.textContent = `${linkFull(l.id)} — ${rate(l.capacity_mbps)}`;
    if (l.id === "L20") o.selected = true;
    selLink.appendChild(o);
  }
  const selDemand = $("sel-burst-demand");
  for (const d of state.demands) {
    const o = document.createElement("option");
    o.value = d.id;
    o.textContent = demandFull(d);
    selDemand.appendChild(o);
  }
  const selModel = $("sel-model");
  const tags = [...new Set((await api.checkpoints()).map((c) => c.tag))];
  for (const t of tags.length ? tags : ["ppo_te"]) {
    const o = document.createElement("option");
    o.value = t; o.textContent = t;
    if (t === "ppo_te") o.selected = true;
    selModel.appendChild(o);
  }

  topoA.init(state.topology);
  topoB.init(state.topology);
  topoA.onDemandSelect = (id) => { state.selectedDemand = id; };

  wireControls();
  applyStatus(await api.status());
  connectWs();

  try { state.benchmark = await api.benchmark(); } catch { /* results absent */ }
  renderBenchmark($("benchmark-body"), state.benchmark,
                  state.status.scenario || $("sel-scenario").value);

  // if a session already exists (page reload), repopulate history + advisor
  await refreshHistories();
  try {
    const a = await api.advisorStatus();
    if (a.pending) { state.advisor = a.pending; highlightProposal(a.pending); }
  } catch { /* no session */ }
  renderActiveTab();

  eventsTimer = setInterval(() => {
    if (state.activeTab === "events") refreshEvents();
  }, 4000);
}

boot().catch((e) => toast(`Startup failed: ${e.message}`, true));
