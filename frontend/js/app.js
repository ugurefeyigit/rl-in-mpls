// App bootstrap: state, WebSocket, control wiring, render loop.

import { api, toast } from "./api.js";
import { TopoView } from "./topo.js";
import { Charts } from "./charts.js";
import {
  renderCheckpoints, renderDecision, renderLspTable, renderLinkTableHtml,
  renderRuns, tapeAppend,
} from "./panels.js";

const $ = (id) => document.getElementById(id);

const state = {
  topology: null,
  scenarios: {},
  demands: [],
  lastPayload: null,
  selectedDemand: null,
  activeTab: "agent",
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

// ------------------------------------------------------------------ render
function onTick(payload) {
  state.lastPayload = payload;
  const st = payload.status;
  const hour = st.hour;
  $("sim-clock").textContent =
    `${String(Math.floor(hour)).padStart(2, "0")}:${String(Math.round((hour % 1) * 60)).padStart(2, "0")}`;
  $("sim-day-phase").textContent =
    `${phaseFor(hour)} · step ${st.step}/${Math.floor(st.duration_min / 5)}` +
    (st.done ? " · END" : st.running ? "" : " · paused");
  $("run-desc").textContent =
    `${st.scenario} · seed ${st.seed} · ${st.algorithms.join(" vs ")}`;

  const runs = payload.runs;
  $("pane-title-a").innerHTML =
    `<span class="algo-a">${runs[0].algorithm}</span> — ${st.scenario}`;
  topoA.update(runs[0].snapshot);
  if (runs[1]) {
    $("pane-title-b").innerHTML =
      `<span class="algo-b">${runs[1].algorithm}</span> — ${st.scenario}`;
    topoB.update(runs[1].snapshot);
  }

  for (const run of runs) {
    if (run.snapshot.metrics && run.decision && run.decision.step === run.snapshot.step) {
      charts.push(run.algorithm, run.snapshot.metrics, run.decision);
      tapeAppend(run.algorithm, run.decision, hour);
    }
  }

  renderActiveTab();
}

function renderActiveTab() {
  const payload = state.lastPayload;
  if (!payload) return;
  const runs = payload.runs;
  switch (state.activeTab) {
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
          .map((d) => `${d.id} (${d.class}, ${d.volume_mbps.toFixed(0)} Mbps, ${d.sla_ok ? "SLA ok" : "SLA VIOLATED"})`);
        $("matrix-detail").textContent = list.length
          ? `${src} → ${dst}: ${list.join(" · ")}` : `${src} → ${dst}: no demands`;
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

// --------------------------------------------------------------- websocket
let ws = null;
function connectWs() {
  ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
  ws.onopen = () => { $("conn-dot").classList.add("ok"); ws.send("hi"); };
  ws.onclose = () => { $("conn-dot").classList.remove("ok"); setTimeout(connectWs, 1500); };
  ws.onmessage = (ev) => {
    const payload = JSON.parse(ev.data);
    if (payload.type === "tick") onTick(payload);
    else if (payload.type === "status" && payload.status.done)
      toast(`Scenario finished: ${payload.status.scenario}`);
  };
}

// ---------------------------------------------------------------- controls
async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); }
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
  });

  $("btn-start").addEventListener("click", () => guard(async () => {
    const algorithms = [$("sel-algo-a").value];
    if (state.compare) algorithms.push($("sel-algo-b").value);
    charts.reset();
    $("tape-lines").innerHTML = "";
    state.selectedDemand = null;
    await api.start({
      scenario: $("sel-scenario").value,
      algorithms,
      seed: parseInt($("inp-seed").value, 10) || 42,
      model_tag: $("sel-model").value || null,
      safety_filter: $("chk-safety").checked,
      speed: document.querySelector(".speed.active")?.dataset.speed || "1x",
      autostart: true,
    });
    toast(`Session started: ${$("sel-scenario").value} [${algorithms.join(" vs ")}]`);
  }));

  $("btn-pause").addEventListener("click", () => guard(api.pause));
  $("btn-resume").addEventListener("click", () => guard(api.resume));
  $("btn-reset").addEventListener("click", () => guard(async () => {
    charts.reset(); $("tape-lines").innerHTML = ""; await api.reset();
  }));
  $("btn-step").addEventListener("click", () => guard(api.step));

  $("speed-row").addEventListener("click", (ev) => {
    const b = ev.target.closest(".speed");
    if (!b) return;
    document.querySelectorAll(".speed").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    guard(() => api.speed(b.dataset.speed));
  });

  $("btn-fail").addEventListener("click", () =>
    guard(() => api.failLink($("sel-fail-link").value)));
  $("btn-recover").addEventListener("click", () =>
    guard(() => api.recoverLink($("sel-fail-link").value)));
  $("btn-burst").addEventListener("click", () =>
    guard(() => api.burst($("sel-burst-demand").value,
                          parseFloat($("inp-burst-factor").value) || 2.0, 60)));
  $("inp-mult").addEventListener("input", () => {
    $("mult-val").textContent = " " + parseFloat($("inp-mult").value).toFixed(2);
  });
  $("inp-mult").addEventListener("change", () =>
    guard(() => api.multiplier(parseFloat($("inp-mult").value))));

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
      renderActiveTab();
      setTimeout(() => charts.render(), 30);
    }));

  $("sel-metric").addEventListener("change", () => {
    charts.metricKey = $("sel-metric").value;
    charts.render();
  });

  $("btn-train").addEventListener("click", () => guard(async () => {
    await api.trainStart({
      timesteps: parseInt($("inp-train-steps").value, 10) || 100000,
      tag: $("inp-train-tag").value || "ppo_custom",
      seed: parseInt($("inp-seed").value, 10) || 42,
    });
    toast("Training job started");
    refreshTraining();
  }));
  $("btn-refresh-runs").addEventListener("click", refreshRuns);
}

async function refreshTraining() {
  try {
    const p = await api.trainProgress();
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

// --------------------------------------------------------------- bootstrap
async function boot() {
  state.topology = await api.topology();
  state.scenarios = await api.scenarios();
  const tc = await api.trafficClasses();
  state.demands = tc.demands;

  const selScen = $("sel-scenario");
  for (const name of Object.keys(state.scenarios)) {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    if (name === "demo_evening") o.selected = true;
    selScen.appendChild(o);
  }
  selScen.dispatchEvent(new Event("change"));

  const selLink = $("sel-fail-link");
  for (const l of state.topology.links) {
    const o = document.createElement("option");
    o.value = l.id; o.textContent = `${l.id} ${l.a}–${l.z} (${l.capacity_mbps} Mbps)`;
    if (l.id === "L20") o.selected = true;
    selLink.appendChild(o);
  }
  const selDemand = $("sel-burst-demand");
  for (const d of state.demands) {
    const o = document.createElement("option");
    o.value = d.id; o.textContent = `${d.id} ${d.src}→${d.dst} (${d.class})`;
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
  connectWs();

  // if a session already exists (page reload), repopulate history
  try {
    const hist = await api.metricsHistory();
    charts.setHistories(hist.runs);
  } catch { /* no session yet */ }
}

boot();
