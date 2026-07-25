// Presentation Mode — a storytelling wallboard for a live audience that knows
// nothing about RL or MPLS.
//
// Hard rule: every number on this page comes from the live simulation or from
// the committed benchmark file. Nothing here is hardcoded, estimated or
// invented. Where a value is unavailable the UI says so rather than guessing.

import { api, toast } from "./api.js";
import { TopoView } from "./topo.js";
import {
  loadDisplay, pathLabel, pathVia, linkLabel, linkFull,
  demandLabel, scenarioLabel, algoLabel, disclaimer, glossary,
} from "./display.js";
import {
  DISPLAY_SCALE, SCALE_NOTE, rate, util, ratio, delay, simTime, esc,
  reward as fmtReward,
} from "./fmt.js";

const $ = (id) => document.getElementById(id);

const DEMO = {
  scenario: "demo_evening",
  seed: 42,
  model_tag: "ppo_te",
  failLink: "L20",          // the scripted backbone failure in demo_evening
};

const state = {
  topology: null,
  benchmark: null,
  status: { state: "idle" },
  runs: [],
  history: {},              // algorithm -> [interval metrics]
  proposal: null,
  lastAdvisorRecord: null,
  comparator: "greedy",
  scale: 1,
  // story-derivation memory
  band: 0,
  failed: new Set(),
  storyItems: [],
  transientPhase: null,
  transientUntil: 0,
  scenarioEvents: [],
  durationMin: 360,
  lastRunSteps: 0,
  // guided story
  story: { active: false, step: -1, awaitingOperator: false },
};

let topo = null;

// ============================================================== boot
async function boot() {
  await loadDisplay();
  state.topology = await api.topology();
  $("map-disclaimer").textContent = disclaimer();

  topo = new TopoView("present-map", null,
                      { fontSize: 15, nodeScale: 1.3, failCross: true });
  topo.init(state.topology);

  const sel = $("sel-present-link");
  for (const l of state.topology.links) {
    const o = document.createElement("option");
    o.value = l.id;
    o.textContent = linkFull(l.id);
    if (l.id === DEMO.failLink) o.selected = true;
    sel.appendChild(o);
  }

  try {
    state.benchmark = await api.benchmark();
    renderBenchmarkStory();
  } catch {
    $("bm-body").innerHTML =
      `<p class="note">Published results file not available on this server.</p>`;
  }

  wireControls();
  wireKeyboard();
  wireGlossary();

  applyStatus(await api.status());
  await loadScenarioMeta();
  connectWs();

  // page reloaded mid-run: recover everything from the backend
  try {
    const payload = await api.telemetry();
    onPayload(payload, { silent: true });
    await refreshHistories();
  } catch { /* no session yet */ }
  try {
    const a = await api.advisorStatus();
    if (a.pending) setProposal(a.pending);
  } catch { /* no session */ }
  renderAll();
}

async function loadScenarioMeta() {
  try {
    const all = await api.scenarios();
    const s = all[state.status.scenario || DEMO.scenario];
    if (s) {
      state.scenarioEvents = s.events || [];
      state.durationMin = s.duration_min || 360;
    }
  } catch { /* keep defaults */ }
}

// ============================================================== websocket
let ws = null;
let wsRetry = 0;
function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/telemetry`);
  ws.onopen = () => {
    wsRetry = 0;
    $("conn-dot").classList.add("ok");
    $("conn-dot").title = "Live connection: connected";
    ws.send("hi");
  };
  ws.onclose = () => {
    $("conn-dot").classList.remove("ok");
    $("conn-dot").title = "Live connection: reconnecting…";
    wsRetry += 1;
    setTimeout(connectWs, Math.min(1500 * wsRetry, 10000));
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const p = JSON.parse(ev.data);
    if (p.type === "advisor") {
      applyStatus(p.status);
      setProposal(p.proposal);
      renderAll();
      return;
    }
    if (p.type === "status") {
      applyStatus(p.status);
      renderAll();
      return;
    }
    if (p.type === "reset") {
      resetDerivedState();
      onPayload(p);
      return;
    }
    // The socket replays a "tick" payload on every (re)connect, so the message
    // type alone must not clear a recommendation that is still pending —
    // status.awaiting_decision is the authority.
    if (p.type === "tick" && !p.status.awaiting_decision) setProposal(null);
    onPayload(p);
  };
}

function resetDerivedState() {
  state.history = {};
  state.band = 0;
  state.failed = new Set();
  state.storyItems = [];
  state.proposal = null;
  state.lastAdvisorRecord = null;
  state.transientPhase = null;
  $("story-timeline").innerHTML =
    `<p class="note">The story log fills in as the evening plays out.</p>`;
}

// ============================================================== state
function applyStatus(st) {
  if (!st) return;
  state.status = st;
  const s = st.state || "idle";
  $("state-dot").className = `state-dot ${s === "running" ? "running" : s === "paused" ? "paused" : ""}`;
  $("state-dot").textContent = `● ${s}`;
  $("present-clock").textContent = st.hour === undefined ? "--:--" : simTime(st.hour);

  if (st.scenario) {
    $("scenario-name").textContent = scenarioLabel(st.scenario);
    $("scenario-sub").textContent =
      `${(st.algorithms || []).map(algoLabel).join("  vs  ")} · seed ${st.seed}` +
      (st.duration_min ? ` · ${Math.round(st.duration_min / 60)}-hour evening` : "");
    if (st.duration_min) state.durationMin = st.duration_min;
  }

  const err = s === "error";
  $("error-overlay").classList.toggle("hidden", !err);
  if (err) $("error-detail").textContent =
    `${st.error}\n\nscenario ${st.scenario} · interval ${st.step} · ` +
    `network time ${simTime(st.hour)}`;

  $("btn-playpause").innerHTML =
    `${s === "running" ? "Pause" : "Resume"}<span class="kbd">Space</span>`;

  const pending = Boolean(st.awaiting_decision);
  const done = Boolean(st.done) || s === "completed";
  $("btn-playpause").disabled = err || (done && s !== "running") || pending;
  $("btn-next-event").disabled = err || done || pending || s === "running";
  $("btn-approve-bar").disabled = !pending;
  $("btn-reject-bar").disabled = !pending;
  $("btn-fail-bar").disabled = err || !st.scenario;
  $("btn-recover-bar").disabled = err || !st.scenario;
  $("btn-reset-story").disabled = !st.scenario;
}

function primaryRun() {
  return state.runs.find((r) => r.algorithm === "rl") || state.runs[0] || null;
}
function comparatorRun() {
  const p = primaryRun();
  return state.runs.find((r) => r !== p) || null;
}

function onPayload(payload, opts = {}) {
  applyStatus(payload.status);
  state.runs = payload.runs || [];
  if (!state.runs.length) return;

  const isTick = payload.type === "tick";
  for (const run of state.runs) {
    const m = run.snapshot.metrics;
    if (!m || !isTick) continue;
    const hist = state.history[run.algorithm] || (state.history[run.algorithm] = []);
    if (!hist.length || m.step > hist[hist.length - 1].step) hist.push(m);
  }

  const p = primaryRun();
  if (p) {
    topo.update(p.snapshot);
    if (!opts.silent) ingestStory(p, payload.status);
    else syncDerivedFromSnapshot(p);
  }
  renderAll();
}

/** After a reload we adopt the current world without narrating its history. */
function syncDerivedFromSnapshot(run) {
  state.failed = new Set(run.snapshot.failed_links || []);
  const u = (run.snapshot.metrics || {}).max_util || 0;
  state.band = u >= 1.0 ? 3 : u >= 0.9 ? 2 : u >= 0.75 ? 1 : 0;
}

/** Refresh accumulated series from the authoritative server-side record. */
async function refreshHistories() {
  try {
    const h = await api.metricsHistory();
    state.history = {};
    for (const r of h.runs) state.history[r.algorithm] = r.history;
  } catch { /* no session */ }
}

/**
 * Pull the authoritative current state after a fast-forward or an advisor
 * decision, rather than waiting for the websocket. A fast-forward steps many
 * intervals server-side but broadcasts only the last payload, so the series
 * must come from GET /api/metrics/history. Story ingestion is idempotent
 * (bands only ratchet up, failures are diffed against a set), so the duplicate
 * websocket delivery of the same payload is harmless.
 */
/**
 * Fast-forward until the next scripted event, then take the one further
 * interval in which it actually fires.
 *
 * The engine applies scripted link events over the half-open window
 * [interval_start, interval_end), and run-until stops as soon as the clock
 * *reaches* the event time — so at that moment the event has not been applied
 * yet. One more interval puts the audience on the far side of it.
 */
async function advanceThroughEvent() {
  await guard(() => api.runUntil("next_event", 120));
  if (!state.status.done) await guard(() => api.step());
  await syncNow();
}

async function syncNow() {
  await refreshHistories();
  try {
    const payload = await api.telemetry();
    onPayload(payload);
  } catch { renderAll(); }
}

// ============================================================== story log
const BANDS = [
  { at: 0.75, sev: "warn", word: "is filling up" },
  { at: 0.90, sev: "bad", word: "is close to capacity" },
  { at: 1.00, sev: "bad", word: "is over capacity — traffic is being dropped" },
];

function addStoryItem(hour, text, sev = "info") {
  const item = { hour, text, sev };
  state.storyItems.push(item);
  const box = $("story-timeline");
  if (box.querySelector(".note")) box.innerHTML = "";
  const el = document.createElement("div");
  el.className = `story-item sev-${sev}`;
  el.innerHTML = `<span class="st-time">${esc(simTime(hour))}</span>` +
                 `<span class="st-dot"></span><span>${text}</span>`;
  box.prepend(el);
  return item;
}

/**
 * The busiest link at this instant, from the snapshot.
 *
 * Presentation Mode reads utilization from the snapshot everywhere (KPI card,
 * phase chip, story thresholds) so the headline number, the map colour and the
 * narration can never disagree. The interval *peak* (metrics.max_util) is a
 * different, also-correct quantity; it drives the spine, which is labelled as
 * a per-interval peak, and it is what the engineering console reports.
 */
function busiestOf(snapshot) {
  let best = null;
  for (const dl of snapshot.links) {
    if (!dl.up) continue;
    if (!best || dl.utilization > best.utilization) best = dl;
  }
  return best;
}

/** "L11" -> "Ankara–Kayseri", for use inside a sentence. */
const corridor = (linkId) => linkLabel(linkId).replace(/ link$/, "");
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/**
 * The most loaded hop along a specific route, in the direction of travel.
 *
 * The recommendation card explains *why* a flow should move, so it must name a
 * link that flow actually crosses — not the globally busiest link, which may
 * be somewhere else entirely.
 */
function pathBottleneck(snapshot, routers) {
  if (!routers || routers.length < 2) return null;
  const byPair = {};
  for (const dl of snapshot.links) byPair[`${dl.src}|${dl.dst}`] = dl;
  let worst = null;
  for (let i = 0; i < routers.length - 1; i++) {
    const dl = byPair[`${routers[i]}|${routers[i + 1]}`];
    if (dl && (!worst || dl.utilization > worst.utilization)) worst = dl;
  }
  return worst;
}

function ingestStory(run, status) {
  const snap = run.snapshot;
  const m = snap.metrics;
  if (!m) return;
  const hour = status.hour;

  // --- link failures and recoveries -------------------------------------
  const failed = new Set(snap.failed_links || []);
  for (const l of failed) {
    if (!state.failed.has(l)) {
      const frr = m.frr_events || 0;
      addStoryItem(hour,
        `<strong>${esc(linkLabel(l))} failed.</strong> ` +
        (frr > 0
          ? `The network immediately moved ${plural(frr, "affected traffic flow",
             "affected traffic flows")} onto backup paths ` +
            `(<span class="gloss" data-gloss="Fast reroute">fast reroute</span>).`
          : `Traffic on that link is being re-routed onto backup paths.`),
        "bad");
      setTransient("Incident", 9000);
    }
  }
  for (const l of state.failed) {
    if (!failed.has(l)) {
      addStoryItem(hour, `<strong>${esc(linkLabel(l))} is back in service.</strong> ` +
                         `Traffic can return to its normal route.`, "good");
      setTransient("Recovery", 9000);
    }
  }
  state.failed = failed;

  // --- busiest-link threshold crossings ---------------------------------
  const busiest = busiestOf(snap);
  const u = busiest ? busiest.utilization : 0;
  const band = u >= 1.0 ? 3 : u >= 0.9 ? 2 : u >= 0.75 ? 1 : 0;
  if (band > state.band && busiest) {
    const b = BANDS[band - 1];
    addStoryItem(hour,
      `${esc(corridor(busiest.link))} reaches ${util(u)} of its capacity — it ${b.word}.`,
      b.sev);
  }
  state.band = band;

  if (status.done && !state.storyItems.some((i) => i.final)) {
    addStoryItem(hour, `<strong>The evening is complete.</strong>`, "info").final = true;
  }
}

function setTransient(phase, ms) {
  state.transientPhase = phase;
  state.transientUntil = Date.now() + ms;
}

// ============================================================== phase
function derivePhase() {
  const st = state.status;
  if (st.state === "error") return { text: "Halted", cls: "p-incident" };
  if (!st.scenario) return { text: "Not started", cls: "p-normal" };
  if (st.done || st.state === "completed") return { text: "Complete", cls: "p-complete" };
  if (st.awaiting_decision) return { text: "Recommendation ready", cls: "p-recommend" };
  if (state.transientPhase && Date.now() < state.transientUntil) {
    const t = state.transientPhase;
    return { text: t, cls: t === "Incident" ? "p-incident"
                       : t === "Recovery" ? "p-recovery" : "p-applied" };
  }
  if (state.failed.size) return { text: "Incident", cls: "p-incident" };
  const run = primaryRun();
  const b = run ? busiestOf(run.snapshot) : null;
  const u = b ? b.utilization : 0;
  if (u >= 0.9) return { text: "Congestion detected", cls: "p-congested" };
  if (u >= 0.75) return { text: "Traffic rising", cls: "p-rising" };
  return { text: "Normal", cls: "p-normal" };
}

// ============================================================== render
function renderAll() {
  const phase = derivePhase();
  $("phase-chip").textContent = phase.text;
  $("phase-chip").className = phase.cls;
  renderKpis();
  renderSpine();
  renderRecommendation();
  renderComparison();
}

function totals(algo) {
  const hist = state.history[algo] || [];
  const t = { n: hist.length, sla: 0, reroutes: 0, flaps: 0, delivered: 0, peak: 0 };
  for (const h of hist) {
    t.sla += h.sla_violations || 0;
    t.reroutes += h.reroutes || 0;
    t.flaps += h.flaps || 0;
    t.delivered += h.delivered_ratio || 0;
    t.peak = Math.max(t.peak, h.max_util || 0);
  }
  t.deliveredMean = t.n ? t.delivered / t.n : 0;
  return t;
}

function setKpi(id, value, note, tone) {
  const card = $(id);
  card.className = `kpi ${card.classList.contains("wide") ? "wide " : ""}${tone || ""}`;
  card.querySelector(".kpi-value").textContent = value;
  card.querySelector(".kpi-value").className = `kpi-value ${tone || ""}`;
  card.querySelector(".kpi-note").innerHTML = note;
}

function renderKpis() {
  const run = primaryRun();
  if (!run) return;
  const snap = run.snapshot;
  // Before the first interval completes the engine has primed link telemetry
  // but not yet produced interval metrics. Anything derived from metrics shows
  // an em dash rather than a zero that would read as a real measurement.
  const m = snap.metrics;
  const t = totals(run.algorithm);
  const cum = run.decision ? run.decision.cumulative_reward : null;

  // 1 — overall score (cumulative reward)
  setKpi("kpi-score", cum === null ? "—" : fmtReward(cum),
    cum === null
      ? `The run has not completed an interval yet.`
      : `Combined score for delivery, congestion, service quality and route ` +
        `stability over ${plural(t.n, "five-minute interval", "five-minute intervals")}. ` +
        `<span class="gloss" data-gloss="Total reward">What is this?</span>`,
    cum === null ? "" : cum >= 0 ? "info" : "warn");

  // 2 — busiest link
  const busiest = busiestOf(snap);
  if (busiest) {
    const u = busiest.utilization;
    const tone = u >= 0.9 ? "bad" : u >= 0.75 ? "warn" : "good";
    const word = u >= 1.0 ? "over capacity — traffic is being dropped"
               : u >= 0.9 ? "close to capacity"
               : u >= 0.75 ? "busy" : "comfortable";
    setKpi("kpi-busiest", util(u),
      `${esc(corridor(busiest.link))} — ${word}.<br>` +
      `Carrying ${esc(rate(busiest.load_mbps, state.scale))} of ` +
      `${esc(rate(busiest.capacity_mbps, state.scale))}. ` +
      `<span class="gloss" data-gloss="Link utilization">What is this?</span>`,
      tone);
  }

  // 3 — services with SLA problems
  const bad = snap.demands.filter((d) => !d.sla_ok || d.disconnected);
  setKpi("kpi-sla", String(bad.length),
    bad.length
      ? `Worst: ${esc(demandLabel(bad[0].src, bad[0].dst, bad[0].class))}. ` +
        `<span class="gloss" data-gloss="SLA problem">What is this?</span>`
      : `Every service is inside its delay and loss targets.`,
    bad.length === 0 ? "good" : bad.length <= 2 ? "warn" : "bad");

  // 4 — delivered
  const dr = m ? m.delivered_ratio : null;
  setKpi("kpi-delivered", dr === null || dr === undefined ? "—" : ratio(dr),
    dr === null || dr === undefined
      ? `Measured once the first five-minute interval completes.`
      : `of all offered traffic is reaching its destination right now.`,
    dr === null || dr === undefined ? "" : dr >= 0.999 ? "good" : dr >= 0.99 ? "warn" : "bad");

  // 5 — route changes
  setKpi("kpi-changes", String(t.reroutes),
    `${t.reroutes === 1 ? "route change" : "route changes"} so far` +
    (t.flaps ? `, of which ${t.flaps} moved traffic straight back` : "") +
    `. <span class="gloss" data-gloss="Route">What is a route?</span>`,
    "info");
}

/** The incident spine: one tick per interval, coloured by that interval's peak. */
function renderSpine() {
  const run = primaryRun();
  const track = $("spine-track");
  if (!run) { track.innerHTML = ""; $("spine-caption").textContent = ""; return; }
  const hist = state.history[run.algorithm] || [];
  const total = Math.max(1, Math.round(state.durationMin / 5));
  const w = 100 / total;
  const colour = (u) => u >= 1.0 ? "var(--u5)" : u >= 0.9 ? "var(--u4)"
                      : u >= 0.75 ? "var(--u3)" : u >= 0.5 ? "var(--u2)" : "var(--u1)";
  let html = hist.map((h, i) =>
    `<i style="left:${(i * w).toFixed(3)}%;width:${w.toFixed(3)}%;` +
    `background:${colour(h.max_util || 0)}"></i>`).join("");
  for (const ev of state.scenarioEvents) {
    const pos = (ev.t_min / state.durationMin) * 100;
    const cls = ev.type === "link_down" ? "ev-fail"
              : ev.type === "link_up" ? "ev-recover" : "ev-burst";
    html += `<span class="spine-mark ${cls}" style="left:${pos.toFixed(3)}%"></span>`;
  }
  const now = ((state.status.t_min || 0) / state.durationMin) * 100;
  html += `<span class="spine-mark now" style="left:${Math.min(now, 99.7).toFixed(3)}%"></span>`;
  track.innerHTML = html;
  $("spine-caption").textContent =
    `${simTime(state.status.hour || 0)} · interval ${state.status.step || 0} of ${total}`;
}

// ------------------------------------------------------------ recommendation
function setProposal(p) {
  const isNew = p && (!state.proposal || state.proposal.id !== p.id);
  state.proposal = p || null;
  topo.showProposedPath(p && p.decoded ? p.decoded.to_routers : null);
  if (isNew) {
    const d = p.decoded;
    addStoryItem(state.status.hour,
      p.is_noop
        ? `The AI Advisor recommends <strong>no change</strong> right now.`
        : `The AI Advisor recommends moving ` +
          `<strong>${esc(demandLabel(d.src, d.dst, d.class))}</strong> to a different route.`,
      "act");
  }
}

function renderRecommendation() {
  const box = $("recommendation");
  const p = state.proposal;
  if (!p) { box.classList.add("hidden"); box.classList.remove("active"); return; }
  box.classList.remove("hidden");
  box.classList.add("active");
  const d = p.decoded;
  const la = p.lookahead || {};

  // Why — the worst hop on THIS flow's own route, and the worst hop on the
  // proposed one. Both are measured; neither is inferred.
  const run = primaryRun();
  const snap = run ? run.snapshot : null;
  const hotNow = snap && d ? pathBottleneck(snap, d.from_routers) : null;
  const hotNew = snap && d ? pathBottleneck(snap, d.to_routers) : null;
  let why;
  if (!d) {
    why = `The policy sees no move worth making at this interval.`;
  } else if (hotNow) {
    why = `Its current route crosses ${esc(corridor(hotNow.link))}, which is at ` +
          `${util(hotNow.utilization)} of capacity — the tightest point on that route.` +
          (hotNew
            ? ` The proposed route's tightest point is ${esc(corridor(hotNew.link))} at ` +
              `${util(hotNew.utilization)} before the move.`
            : "");
  } else {
    why = `The policy selected this flow as the best one to move at this interval.`;
  }

  // Expected effect — REAL lookahead numbers, or an explicit "no prediction".
  let effect;
  if (la.noop && la.action) {
    const before = la.noop.max_util, after = la.action.max_util;
    const same = Math.abs(before - after) < 0.005;
    effect = `<div class="rec-effect">
      <div class="eyebrow">Expected effect on the busiest link</div>
      <div class="big">${util(before, 1)} → ${util(after, 1)}</div>
      <div>${same
        ? `No change to the busiest link this interval; the benefit shows up in
           delay (${delay(la.noop.mean_delay_ms)} → ${delay(la.action.mean_delay_ms)})
           and in where traffic sits afterwards.`
        : after < before
          ? `Relieves the busiest link by ${Math.abs((before - after) * 100).toFixed(1)}
             percentage points.`
          : `Adds ${Math.abs((after - before) * 100).toFixed(1)} percentage points to the
             busiest link — the policy is trading peak load for something else.`}</div>
      <div class="note">Measured by replaying one interval on a copy of the network,
        with and without the change. The live network is untouched.</div>
    </div>`;
  } else if (la.noop) {
    effect = `<div class="rec-effect">
      <div class="eyebrow">If nothing changes</div>
      <div class="big">${util(la.noop.max_util, 1)}</div>
      <div class="note">busiest link over the next five minutes</div></div>`;
  } else {
    effect = `<div class="rec-effect note">No forward projection available for this
      recommendation.</div>`;
  }

  // The headline only calls a corridor "congested" when the measured value on
  // this flow's own route says so.
  const headline = p.is_noop
    ? `Hold the current routing — no change recommended.`
    : hotNow && hotNow.utilization >= 0.75
      ? `Move ${esc(demandLabel(d.src, d.dst, d.class))} away from the congested ` +
        `${esc(corridor(hotNow.link))} corridor.`
      : `Move ${esc(demandLabel(d.src, d.dst, d.class))} onto a different route.`;

  $("rec-body").innerHTML = `
    <div class="rec-headline">${headline}</div>
    ${p.is_noop ? "" : `
      <dl class="rec-route">
        <dt>Traffic</dt><dd>${esc(rate(d.volume_mbps, state.scale))}</dd>
        <dt>Now goes via</dt><dd class="old">${esc(pathVia(d.from_routers))}</dd>
        <dt>Would go via</dt><dd class="new">${esc(pathVia(d.to_routers))}</dd>
      </dl>
      <div class="note">Full route: ${esc(pathLabel(d.to_routers))}</div>`}
    <div class="note" style="margin-top:8px"><strong>Why:</strong> ${why}</div>
    ${effect}
    <div class="rec-safety ${p.safety_ok ? "ok" : "bad"}">
      ${p.safety_ok
        ? `✔ Safety check passed — the move respects capacity, delay and stability rules.`
        : `✖ Safety check failed: ${esc(p.safety_reason)}. Approving would apply the
           policy's choice anyway; the rule engine would block the move.`}
    </div>
    <div class="rec-actions">
      <button class="btn-approve" id="btn-approve-card">Approve the change</button>
      <button class="btn-reject" id="btn-reject-card">Reject</button>
    </div>
    <div class="tech-detail">proposal #${p.id} · interval ${p.step} · action ${p.action}
      ${d ? `· ${esc(d.demand)} p${d.from_path}→p${d.path_idx}` : ""}
      ${p.action_probability !== null && p.action_probability !== undefined
        ? `· policy confidence ${(p.action_probability * 100).toFixed(1)}%` : ""}</div>`;

  $("btn-approve-card").addEventListener("click", approve);
  $("btn-reject-card").addEventListener("click", reject);
}

// --------------------------------------------------------------- comparison
function renderComparison() {
  const a = primaryRun();
  const b = comparatorRun();
  const body = $("cmp-body");
  if (!a) return;
  if (!b) {
    body.innerHTML = `<p class="note">This run has a single controller
      (${esc(algoLabel(a.algorithm))}). Start the guided story to compare two.</p>`;
    return;
  }
  const ta = totals(a.algorithm), tb = totals(b.algorithm);
  const ca = a.decision ? a.decision.cumulative_reward : 0;
  const cb = b.decision ? b.decision.cumulative_reward : 0;
  const d = ca - cb;
  const cls = Math.abs(d) < 5 ? "neutral" : d > 0 ? "good" : "bad";
  const slaA = a.snapshot.demands.filter((x) => !x.sla_ok || x.disconnected).length;
  const slaB = b.snapshot.demands.filter((x) => !x.sla_ok || x.disconnected).length;

  // One honest sentence. Never a percentage: baseline totals go negative, so a
  // ratio would be meaningless or misleading.
  let verdict;
  if (Math.abs(d) < 5) {
    verdict = `The two controllers are level so far — within
      ${fmtReward(Math.abs(d))} points of each other.`;
  } else if (d > 0) {
    verdict = `<strong>${esc(algoLabel(a.algorithm))} is ahead by
      ${fmtReward(Math.abs(d))} reward points.</strong> It is paying for that with
      ${ta.reroutes} route changes against ${tb.reroutes} — more churn for the
      network operations team.`;
  } else {
    verdict = `<strong>${esc(algoLabel(b.algorithm))} is ahead by
      ${fmtReward(Math.abs(d))} reward points.</strong> This is a real result, not a
      glitch: the traditional controller wins several reactive incidents.`;
  }

  body.innerHTML = `
    <table class="cmp-table">
      <thead><tr><th></th>
        <th class="a">${esc(algoLabel(a.algorithm))}</th>
        <th class="b">${esc(algoLabel(b.algorithm))}</th></tr></thead>
      <tbody>
        <tr><td>Overall score</td><td class="num">${fmtReward(ca)}</td>
            <td class="num">${fmtReward(cb)}</td></tr>
        <tr><td>Busiest link now</td>
            <td class="num">${util((busiestOf(a.snapshot) || {}).utilization || 0)}</td>
            <td class="num">${util((busiestOf(b.snapshot) || {}).utilization || 0)}</td></tr>
        <tr><td>Peak busiest link, whole run</td><td class="num">${util(ta.peak)}</td>
            <td class="num">${util(tb.peak)}</td></tr>
        <tr><td>Services with SLA problems</td><td class="num">${slaA}</td>
            <td class="num">${slaB}</td></tr>
        <tr><td>Traffic delivered</td><td class="num">${ratio(ta.deliveredMean)}</td>
            <td class="num">${ratio(tb.deliveredMean)}</td></tr>
        <tr><td>Route changes</td><td class="num">${ta.reroutes}</td>
            <td class="num">${tb.reroutes}</td></tr>
      </tbody></table>
    <div class="cmp-verdict ${cls}">${verdict}</div>`;
}

// ---------------------------------------------------------------- benchmark
function renderBenchmarkStory() {
  const bench = state.benchmark;
  if (!bench) return;
  const rows = Object.entries(bench.scenarios).map(([key, s]) => {
    const rl = s.algorithms.rl;
    const rivals = Object.entries(s.algorithms).filter(([x]) => x !== "rl")
      .sort(([, x], [, y]) => y.reward_mean - x.reward_mean);
    const best = rivals[0];
    const rlWins = s.winner === "rl";
    return `<tr>
      <td>${esc(s.display_name)}</td>
      <td class="num ${rlWins ? "win-rl" : ""}">${rl ? rl.reward_mean.toFixed(1) : "—"}</td>
      <td class="num ${rlWins ? "" : "win-other"}">${best ? best[1].reward_mean.toFixed(1) : "—"}</td>
      <td>${esc(algoLabel(s.winner))}</td></tr>`;
  }).join("");
  const wins = Object.values(bench.scenarios).filter((s) => s.winner === "rl").length;
  const n = Object.keys(bench.scenarios).length;
  $("bm-body").innerHTML = `
    <table class="bm-table">
      <thead><tr><th>Scenario</th><th>AI Advisor</th><th>Best other</th><th>Winner</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <div class="honest">The AI Advisor wins ${wins} of ${n} published scenarios,
      not all of them. It is strongest on a normal traffic day and on the hidden
      shared bottleneck. The traditional controller wins several reactive
      incidents, and the AI Advisor changes routes far more often — churn the
      operations team would feel.</div>
    <div class="note" style="margin-top:8px">5 seeds per scenario, mean total reward.
      Source: ${esc(bench.source)}</div>`;
}

// ============================================================== actions
async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); throw e; }
}
async function soft(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); return null; }
}

async function approve() {
  const rec = await soft(() => api.advisorApprove());
  if (!rec) return;
  state.lastAdvisorRecord = rec;
  setTransient("Change applied", 9000);
  await syncNow();
  const pred = rec.lookahead && rec.lookahead.action
    ? util(rec.lookahead.action.max_util, 1) : null;
  addStoryItem(state.status.hour,
    `<strong>The operator approved the change.</strong> Busiest link after it: ` +
    `${util(rec.actual.max_util, 1)}${pred ? ` (projected ${pred})` : ""}.`, "act");
  renderAll();
  if (state.story.active && state.story.awaitingOperator) {
    state.story.awaitingOperator = false;
    storyAdvance();
  }
}

async function reject() {
  const rec = await soft(() => api.advisorReject());
  if (!rec) return;
  state.lastAdvisorRecord = rec;
  setTransient("Change applied", 6000);
  await syncNow();
  addStoryItem(state.status.hour,
    `<strong>The operator rejected the change.</strong> Nothing was moved; ` +
    `busiest link ${util(rec.actual.max_util, 1)}.`, "act");
  renderAll();
  if (state.story.active && state.story.awaitingOperator) {
    state.story.awaitingOperator = false;
    storyAdvance();
  }
}

async function togglePlay() {
  const s = state.status.state;
  if (s === "running") await soft(() => api.pause());
  else await soft(() => api.resume());
}

async function nextEvent() {
  if (state.story.active) return storyAdvance();
  await soft(advanceThroughEvent);
}

async function failSelected() {
  const id = $("sel-present-link").value;
  const r = await soft(() => api.failLink(id));
  if (!r) return;
  toast(r.changed
    ? `${linkLabel(id)} failed — ${r.frr_reroutes} fast-reroute move(s).`
    : `${linkLabel(id)} was already failed — no change.`, !r.changed);
  renderAll();
}

async function recoverSelected() {
  const id = $("sel-present-link").value;
  const r = await soft(() => api.recoverLink(id));
  if (!r) return;
  toast(r.changed
    ? `${linkLabel(id)} is back in service.`
    : `${linkLabel(id)} was already up — no change.`, !r.changed);
  renderAll();
}

// ============================================================== guided story
// Every step reads real backend state. Narration describes what just happened;
// it never asserts a number the backend did not produce.
const STORY_STEPS = [
  {
    id: "intro",
    modal: () => ({
      title: "An evening on the national backbone",
      html: `<p>This is a simulated national network carrying voice, video,
        enterprise and consumer traffic between eighteen cities. Over the next few
        minutes the evening peak builds, a live event floods one region, and a
        backbone link fails — and you decide whether to take the AI Advisor's
        advice.</p>
        <p class="note">Two controllers run side by side on identical copies of the
        network: an <span class="gloss" data-gloss="AI Advisor">AI Advisor</span>
        trained with reinforcement learning, and a
        <span class="gloss" data-gloss="Traditional controller">traditional
        rule-based controller</span>.</p>`,
      cta: "Start the evening",
    }),
    run: async () => {
      const st = await guard(() => api.start({
        scenario: DEMO.scenario,
        algorithms: ["rl", state.comparator],
        seed: DEMO.seed,
        model_tag: DEMO.model_tag,
        safety_filter: true,
        speed: "1x",
        autostart: false,           // the story drives the clock, not a timer
        advisor: true,
        interface_mode: "present",
      }));
      resetDerivedState();
      applyStatus(st);
      await loadScenarioMeta();
      await syncNow();
      addStoryItem(state.status.hour ?? 17,
        `<strong>The evening begins.</strong> All links are comfortable.`, "info");
    },
  },
  {
    id: "to-congestion",
    label: "Run forward until the network gets busy",
    run: async () => {
      const r = await guard(() => api.runUntil("congestion", 60, 0.85));
      state.lastRunSteps = r.steps;
      await syncNow();
      const run = primaryRun();
      const b = run ? busiestOf(run.snapshot) : null;
      addStoryItem(state.status.hour,
        b ? `After ${plural(r.steps, "interval", "intervals")}, ` +
            `${esc(corridor(b.link))} is the busiest link at ${util(b.utilization)}.`
          : `Ran forward ${plural(r.steps, "interval", "intervals")}.`, "warn");
    },
    // Narration is written from what the run actually produced. On this seed
    // the network is already close to capacity at 17:00, so the card says that
    // rather than claiming a build-up the audience did not see.
    modal: () => {
      const run = primaryRun();
      const b = run ? busiestOf(run.snapshot) : null;
      const quick = (state.lastRunSteps || 0) <= 2;
      return {
        title: quick ? "Already close to the limit" : "The evening peak builds",
        html: `<p>${quick
          ? `The evening does not start quietly. Within
             ${plural(state.lastRunSteps || 1, "five-minute interval",
                      "five-minute intervals")} of 17:00 one corridor is already
             near its limit.`
          : `Video and consumer traffic climbed steadily for
             ${plural(state.lastRunSteps || 0, "interval", "intervals")}.`}</p>
          ${b ? `<p><strong>${esc(corridor(b.link))}</strong> is carrying
            ${esc(rate(b.load_mbps, state.scale))} of its
            ${esc(rate(b.capacity_mbps, state.scale))} capacity —
            <strong>${util(b.utilization)}</strong>. Every extra megabit on that
            corridor now turns into queueing delay and, past 100%, dropped
            packets.</p>` : ""}
          <p>This is the moment a traffic engineer has to decide whether to move
            something. Let's ask the AI Advisor.</p>`,
        cta: "Ask the AI Advisor",
      };
    },
  },
  {
    id: "propose-1",
    label: "Ask the AI Advisor for a recommendation",
    run: async () => { await guard(() => api.advisorPropose()); },
    awaitOperator: true,
  },
  {
    id: "after-decision",
    modal: () => {
      const rec = state.lastAdvisorRecord;
      if (!rec) return { title: "Decision recorded", html: `<p>Continuing.</p>`, cta: "Continue" };
      const pred = rec.lookahead && rec.lookahead.action ? rec.lookahead.action : null;
      return {
        title: rec.approved ? "Change applied" : "Recommendation rejected",
        html: `<p>${rec.approved
          ? `The route change was applied to the live network.`
          : `Nothing was moved — the network continued unchanged.`}</p>
          <table class="cmp-table"><tbody>
            <tr><td>Busiest link, projected before deciding</td>
              <td class="num">${pred ? util(pred.max_util, 1) : "—"}</td></tr>
            <tr><td>Busiest link, actually measured</td>
              <td class="num">${util(rec.actual.max_util, 1)}</td></tr>
            <tr><td>Services with SLA problems</td>
              <td class="num">${rec.actual.sla_violations}</td></tr>
            <tr><td>Operator decision time</td>
              <td class="num">${rec.operator_response_s.toFixed(1)} s</td></tr>
          </tbody></table>
          <p class="note">The projection came from replaying one interval on a copy of
          the network. The measured value also includes the traffic change over that
          interval, so the two are close but not identical.</p>`,
        cta: "Continue to the live event",
      };
    },
  },
  {
    id: "next-event-1",
    label: "Run forward to the live-event surge",
    run: async () => {
      await advanceThroughEvent();
      addStoryItem(state.status.hour,
        `<strong>A major live event starts.</strong> Traffic towards the east ` +
        `doubles over the next two hours.`, "warn");
    },
    modal: () => ({
      title: "A live event floods one region",
      html: `<p>Demand towards the eastern cities has just doubled. Watch the map:
        links that were comfortable a moment ago are turning orange and red.</p>
        <p class="note">Nothing has failed yet — this is pure demand.</p>`,
      cta: "Continue",
    }),
  },
  {
    id: "next-event-2",
    label: "Run forward to the backbone failure",
    run: async () => {
      await advanceThroughEvent();
    },
    modal: () => {
      const failed = [...state.failed];
      return {
        title: failed.length ? "A backbone link has failed" : "Incident",
        html: `<p>${failed.length
          ? `<strong>${esc(failed.map(linkLabel).join(", "))}</strong> just went down.`
          : `A scheduled incident has fired.`}</p>
          <p>The network did not wait for anyone: it immediately moved the affected
          traffic onto backup paths. That is
          <span class="gloss" data-gloss="Fast reroute">fast reroute</span> — a
          built-in protection mechanism, not the AI. It restores connectivity in
          milliseconds, but it does not care whether the backup path is already busy.</p>
          <p>That is exactly where a controller has to decide what to do next.</p>`,
        cta: "Ask the AI Advisor again",
      };
    },
  },
  {
    id: "propose-2",
    label: "Ask the AI Advisor after the failure",
    run: async () => { await guard(() => api.advisorPropose()); },
    awaitOperator: true,
  },
  {
    id: "next-event-3",
    label: "Run forward to the repair",
    run: async () => {
      await advanceThroughEvent();
    },
    modal: () => ({
      title: state.failed.size ? "Still running degraded" : "The link is repaired",
      html: `<p>${state.failed.size
        ? `${esc([...state.failed].map(corridor).join(", "))} is still down. The network
           is carrying the evening on backup paths.`
        : `The failed backbone link is back in service and traffic can return to its
           normal route.`}</p>
        <p class="note">Notice how many route changes each controller has made to get
        here — that number is in the comparison panel, and it matters to the people
        who run the network.</p>`,
      cta: "See the final result",
    }),
  },
  {
    id: "summary",
    modal: () => finalSummaryModal(),
  },
];

function finalSummaryModal() {
  const a = primaryRun(), b = comparatorRun();
  if (!a) return { title: "No run", html: `<p>Start the story first.</p>`, cta: "Close" };
  const ta = totals(a.algorithm);
  const ca = a.decision ? a.decision.cumulative_reward : 0;
  let rows = `<tr><th></th><th>${esc(algoLabel(a.algorithm))}</th>` +
             (b ? `<th>${esc(algoLabel(b.algorithm))}</th>` : "") + `</tr>`;
  const tb = b ? totals(b.algorithm) : null;
  const cb = b && b.decision ? b.decision.cumulative_reward : 0;
  const line = (label, x, y) =>
    `<tr><td>${label}</td><td class="num">${x}</td>` + (b ? `<td class="num">${y}</td>` : "") + `</tr>`;
  rows += line("Overall score", fmtReward(ca), fmtReward(cb));
  rows += line("Peak busiest link", util(ta.peak), tb ? util(tb.peak) : "");
  rows += line("Demand-interval SLA violations", ta.sla, tb ? tb.sla : "");
  rows += line("Traffic delivered", ratio(ta.deliveredMean), tb ? ratio(tb.deliveredMean) : "");
  rows += line("Route changes", ta.reroutes, tb ? tb.reroutes : "");

  const d = ca - cb;
  const honest = !b
    ? `<p class="note">Only one controller ran, so there is nothing to compare against.</p>`
    : Math.abs(d) < 5
      ? `<p>The two finished level — within ${fmtReward(Math.abs(d))} points.</p>`
      : d > 0
        ? `<p>The AI Advisor finished <strong>${fmtReward(Math.abs(d))} points ahead</strong>
           on this one run — but it also made ${ta.reroutes} route changes against
           ${tb.reroutes}. Stability has a cost that this score only partly captures.</p>`
        : `<p>The traditional controller finished
           <strong>${fmtReward(Math.abs(d))} points ahead</strong> on this run. That is a
           genuine result: the rule-based method reacts well to sudden incidents, and
           the published multi-seed evaluation shows it winning several of them.</p>`;

  return {
    title: "How the evening went",
    html: `<table class="cmp-table"><tbody>${rows}</tbody></table>
      ${honest}
      <p class="note">This is a single run with one random seed. The published
      evaluation in the panel on the right averages five seeds per scenario and is
      the number to quote.</p>`,
    cta: "Finish",
    final: true,
  };
}

// --- story driver -----------------------------------------------------------
async function storyStart() {
  state.story = { active: true, step: -1, awaitingOperator: false };
  $("btn-story-start").textContent = "Restart guided story";
  await storyAdvance();
}

async function storyAdvance() {
  const s = state.story;
  if (!s.active) return;
  if (s.awaitingOperator) {
    toast("Approve or reject the recommendation to continue.", true);
    return;
  }
  s.step += 1;
  if (s.step >= STORY_STEPS.length) {
    s.active = false;
    $("btn-story-start").textContent = "Start Guided 5-Minute Story";
    return;
  }
  const step = STORY_STEPS[s.step];

  if (step.run) {
    try { await step.run(); }
    catch { s.step -= 1; return; }   // stay put; the toast already explained why
  }
  await refreshHistories();
  renderAll();

  if (step.awaitOperator) {
    s.awaitingOperator = true;
    return;                          // the Approve/Reject handlers resume the story
  }
  if (step.modal) showStoryCard(step.modal());
  else await storyAdvance();
}

function showStoryCard(spec) {
  $("story-title").textContent = spec.title;
  $("story-text").innerHTML = spec.html;
  $("btn-story-continue").textContent = spec.cta || "Continue";
  $("btn-story-continue").classList.toggle("hidden", Boolean(spec.final));
  $("story-card").classList.remove("hidden");
  $("btn-story-continue").focus();
  wireGlossary();
}

function hideStoryCard() { $("story-card").classList.add("hidden"); }

async function storyReset() {
  state.story = { active: false, step: -1, awaitingOperator: false };
  $("btn-story-start").textContent = "Start Guided 5-Minute Story";
  hideStoryCard();
  const st = await soft(() => api.reset());
  if (st) {
    resetDerivedState();
    toast(`Reset to the start. Same run: ${scenarioLabel(st.scenario)}, ` +
          `${(st.algorithms || []).map(algoLabel).join(" vs ")}, seed ${st.seed}.`);
  }
}

// ============================================================== print
function buildPrintSummary() {
  const a = primaryRun(), b = comparatorRun();
  const st = state.status;
  if (!a) {
    $("print-summary").innerHTML = `<h1>National backbone demonstration</h1>
      <p>No run has been started yet.</p>`;
    return;
  }
  const ta = totals(a.algorithm);
  const tb = b ? totals(b.algorithm) : null;
  const ca = a.decision ? a.decision.cumulative_reward : 0;
  const cb = b && b.decision ? b.decision.cumulative_reward : 0;
  const row = (l, x, y) =>
    `<tr><td>${l}</td><td>${x}</td>${b ? `<td>${y}</td>` : ""}</tr>`;
  $("print-summary").innerHTML = `
    <h1>${esc(scenarioLabel(st.scenario))}</h1>
    <p>Simulated network time ${esc(simTime(st.hour || 0))} ·
       interval ${st.step} of ${Math.round(state.durationMin / 5)} ·
       seed ${st.seed} · model ${esc(st.model_tag || "none")}</p>
    <h2>Result</h2>
    <table><thead><tr><th></th><th>${esc(algoLabel(a.algorithm))}</th>
      ${b ? `<th>${esc(algoLabel(b.algorithm))}</th>` : ""}</tr></thead>
      <tbody>
        ${row("Overall score", fmtReward(ca), fmtReward(cb))}
        ${row("Peak busiest link", util(ta.peak), tb ? util(tb.peak) : "")}
        ${row("Demand-interval SLA violations", ta.sla, tb ? tb.sla : "")}
        ${row("Traffic delivered (mean)", ratio(ta.deliveredMean), tb ? ratio(tb.deliveredMean) : "")}
        ${row("Route changes", ta.reroutes, tb ? tb.reroutes : "")}
        ${row("Route flaps", ta.flaps, tb ? tb.flaps : "")}
      </tbody></table>
    <h2>What happened</h2>
    <ul>${[...state.storyItems].map((i) =>
      `<li>${esc(simTime(i.hour))} — ${i.text.replace(/<[^>]+>/g, "")}</li>`).join("")}</ul>
    <p class="foot">${esc(disclaimer())} Single run, one seed; the published
      multi-seed evaluation is in results/eval_stats.csv.
      ${state.scale > 1 ? esc(SCALE_NOTE) : ""}</p>`;
}

// ============================================================== controls
function wireControls() {
  $("btn-story-start").addEventListener("click", storyStart);
  $("btn-story-continue").addEventListener("click", () => {
    hideStoryCard();
    storyAdvance();
  });
  $("btn-story-close").addEventListener("click", hideStoryCard);
  $("btn-playpause").addEventListener("click", togglePlay);
  $("btn-next-event").addEventListener("click", nextEvent);
  $("btn-approve-bar").addEventListener("click", approve);
  $("btn-reject-bar").addEventListener("click", reject);
  $("btn-fail-bar").addEventListener("click", failSelected);
  $("btn-recover-bar").addEventListener("click", recoverSelected);
  $("btn-reset-story").addEventListener("click", storyReset);

  $("sel-comparator").addEventListener("change", async (ev) => {
    const next = ev.target.value;
    if (state.status.scenario) {
      const ok = window.confirm(
        `Compare the AI Advisor against "${algoLabel(next)}" instead?\n\n` +
        `This restarts the run from the beginning.`);
      if (!ok) { ev.target.value = state.comparator; return; }
      state.comparator = next;
      state.story = { active: false, step: -1, awaitingOperator: false };
      $("btn-story-start").textContent = "Start Guided 5-Minute Story";
      await storyStart();
    } else {
      state.comparator = next;
    }
  });

  $("chk-scale").addEventListener("change", (ev) => {
    state.scale = ev.target.checked ? DISPLAY_SCALE : 1;
    topo.setScale(state.scale);
    $("scale-banner").classList.toggle("hidden", state.scale === 1);
    $("scale-banner").textContent = SCALE_NOTE;
    renderAll();
  });

  $("btn-fullscreen").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  });

  $("btn-print").addEventListener("click", () => {
    buildPrintSummary();
    window.print();
  });
  window.addEventListener("beforeprint", buildPrintSummary);
  window.addEventListener("resize", () => topo.cy?.resize());
}

function wireKeyboard() {
  document.addEventListener("keydown", (ev) => {
    const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
    const cardOpen = !$("story-card").classList.contains("hidden");
    switch (ev.key) {
      case " ":
        ev.preventDefault();
        if (cardOpen) { hideStoryCard(); storyAdvance(); } else togglePlay();
        break;
      case "ArrowRight":
        ev.preventDefault();
        if (cardOpen) { hideStoryCard(); storyAdvance(); } else nextEvent();
        break;
      case "a": case "A":
        if (state.proposal) { ev.preventDefault(); approve(); }
        break;
      case "r": case "R":
        if (state.proposal) { ev.preventDefault(); reject(); }
        break;
      case "f": case "F":
        ev.preventDefault(); failSelected();
        break;
      case "Escape":
        if (cardOpen) hideStoryCard();
        else if (document.fullscreenElement) document.exitFullscreen();
        break;
      default: break;
    }
  });
}

// glossary tooltips, sourced from GET /api/display
function wireGlossary() {
  const tip = $("gloss-tip");
  for (const el of document.querySelectorAll(".gloss:not([data-wired])")) {
    el.dataset.wired = "1";
    el.tabIndex = 0;
    const show = () => {
      const term = el.dataset.gloss || el.textContent.trim();
      const text = glossary()[term];
      if (!text) return;
      tip.innerHTML = `<strong>${esc(term)}</strong><br>${esc(text)}`;
      tip.classList.remove("hidden");
      const r = el.getBoundingClientRect();
      tip.style.left = `${Math.min(r.left, window.innerWidth - 320)}px`;
      tip.style.top = `${Math.max(8, r.top - tip.offsetHeight - 10)}px`;
    };
    const hide = () => tip.classList.add("hidden");
    el.addEventListener("mouseenter", show);
    el.addEventListener("focus", show);
    el.addEventListener("mouseleave", hide);
    el.addEventListener("blur", hide);
  }
}

// re-wire glossary spans after every render (they are recreated by innerHTML)
const observer = new MutationObserver(() => wireGlossary());
observer.observe(document.body, { childList: true, subtree: true });

boot().catch((e) => toast(`Startup failed: ${e.message}`, true));
