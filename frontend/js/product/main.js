/* Product bootstrap and source orchestration. */

import { api as evidenceApi } from "./adapters/evidence-v2.js";
import { api as liveApi, connect, hasActiveSession } from "./adapters/live-v1.js";
import { api as replayApi } from "./adapters/recorded-v2.js";
import { beatAt, matchesStorySession, storySessionConfig } from "./guided-story.js";
import { renderConclusion } from "./governed-study.js";
import { questionDestination } from "./help.js";
import { proposalFromAdvisor } from "./recommendation-card.js";
import { onNavigate, readLocation, writeLocation } from "./router.js";
import { mountShell } from "./shell.js";
import { captureSource, createStore, isCurrentSource } from "./store.js";
import { TopologyAtlas } from "./topology-atlas.js";
import { coalesceLatest } from "./latest-refresh.js";

const store = createStore();
let shell = null;
let socket = null;
let storyTimer = null;

const refreshLive = coalesceLatest(
  refreshLiveOnce,
  () => store.state.source.kind === "live_session",
);

const atlas = new TopologyAtlas({
  onSelect: (type, id, detail) => {
    actions.selectObject(type, id);
    if (detail?.open) shell?.openDrawer("drawer-inspector");
  },
});

const actions = {
  setSource,
  selectObject: (objectType, objectId) => store.select(objectType, objectId),
  selectEvent: (event) => { store.selectEvent(event.id); if (event.object_type && event.object_id) store.select(event.object_type, event.object_id); },
  toggleClass: (value) => toggleFilter("classes", value),
  toggleCondition: (value) => toggleFilter("conditions", value),
  setNetworkSearch: (search) => store.patch({ filters: { search } }),
  clearFilters: () => store.patch({ filters: { classes: [], conditions: [], search: "" } }),
  setRlView: (rlView) => { store.patch({ rlView }); writeLocation(store.state); },
  setObservationSearch: (observationSearch) => store.patch({ ui: { observationSearch } }),
  toggleObservationChanged: () => store.patch({ ui: { observationChangedOnly: !store.state.ui.observationChangedOnly } }),
  toggleInvalidActions: () => store.patch({ ui: { showInvalidActions: !store.state.ui.showInvalidActions } }),
  selectAction: (row) => store.patch({ selection: { actionId: row.action } }),
  counterfactual: () => run(counterfactual),
  loadReplay: (policy, scenario, seed) => run(() => loadReplay(policy, scenario, seed)),
  scrubReplay: (currentStep) => store.patch({ data: { replay: { ...store.state.data.replay, currentStep } } }),
  loadEvidence: () => run(loadEvidence),
  toggleAudience: () => store.patch({ ui: { audienceView: !store.state.ui.audienceView } }),
  toggleFullscreen: () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(),
  playPause: () => run(playPause),
  step: () => run(step),
  nextEvent: () => run(() => ensureSession().then(() => liveApi.runUntil("next_event")).then(refreshLive)),
  setSpeed: (speed) => run(() => liveApi.speed(speed).then(refreshLive)),
  toggleStory: () => run(async () => { await toggleStory(); scheduleStoryAuto(); }),
  toggleStoryAuto,
  storyNext: () => run(async () => { await storyNext(); scheduleStoryAuto(); }),
  storyPrevious: () => { storyPrevious(); scheduleStoryAuto(); },
  propose: () => run(propose),
  approve: () => run(approve),
  reject: () => run(reject),
  jumpBookmark,
  questionJump,
};

async function boot() {
  shell = mountShell({ store, atlas, actions });
  const [capabilities, contracts, displayMap] = await Promise.all([
    liveApi.capabilities(), liveApi.contracts(), liveApi.displayMap(),
  ]);
  store.patch({ data: { capabilities, contracts, displayMap } });
  atlas.build(displayMap);
  applyRoute(readLocation());
  shell.render();
  socket = connect({
    onState: (connection) => store.patch({ connection }),
    onPayload: () => { if (store.state.source.kind === "live_session") run(refreshLive); },
  });
  await loadAppliedRoute();
}

function applyRoute(route) {
  store.setMode(route.mode);
  store.setSource(route.source);
  store.patch({
    workflow: route.workflow,
    rlView: route.rlView,
    selection: { ...route.selection, eventId: route.eventId, actionId: null },
  });
  store.patch({ story: {
    active: route.workflow === "guided-story",
    beat: route.workflow === "guided-story" ? 0 : store.state.story.beat,
    reviewBeat: null,
    auto: route.workflow === "guided-story" ? store.state.story.auto : false,
  } });
}

async function loadAppliedRoute() {
  if (store.state.story.active) await ensureStorySession();
  else await loadSource(store.state.source.kind);
}

async function setSource(kind) {
  store.setSource(kind);
  writeLocation(store.state);
  await loadSource(kind);
}

async function loadSource(kind) {
  const token = captureSource(store.state);
  if (token.kind !== kind) return;
  if (kind === "live_session") return refreshLive();
  if (kind === "recorded_replay") {
    const index = await replayApi.index();
    if (!isCurrentSource(store.state, token)) return;
    store.patch({ data: { replay: { index, episode: null, currentStep: 0 } }, connection: "open" });
    return;
  }
  await loadEvidence();
  store.patch({ connection: "open" });
}

async function loadEvidence() {
  const requests = {
    study: evidenceApi.study(),
    finalHoldout: evidenceApi.finalHoldout(),
    finalScenarios: evidenceApi.finalScenarios(),
    finalRewardComponents: evidenceApi.finalRewardComponents(),
    finalActions: evidenceApi.finalActions(),
    finalIntegrity: evidenceApi.finalIntegrity(),
    finalProvenance: evidenceApi.finalProvenance(),
    developmentContinuity: evidenceApi.developmentContinuity(),
    developmentSeed42: evidenceApi.developmentSeed42(),
    disclosures: evidenceApi.disclosures(),
  };
  const entries = await Promise.all(Object.entries(requests).map(async ([key, request]) => {
    try { return [key, await request]; }
    catch (error) { return [key, null, error]; }
  }));
  const evidence = {};
  const errors = [];
  for (const [key, value, error] of entries) {
    evidence[key] = value;
    if (error) errors.push(`${key}: ${error.message}`);
  }
  if (errors.length) evidence.error = errors.join(" · ");
  store.patch({ data: { evidence } });
  const conclusion = document.getElementById("conclusion-body");
  if (conclusion) conclusion.replaceChildren(...renderConclusion(store.state));
}

async function refreshLiveOnce() {
  if (store.state.source.kind !== "live_session") return;
  const sourceRequest = captureSource(store.state);
  const status = await liveApi.status();
  if (!isCurrentSource(store.state, sourceRequest)) return;
  if (!hasActiveSession(status)) {
    store.patch({ data: { snapshot: null, previousSnapshot: null, decision: null,
      timeline: null, comparison: null, recommendation: null }, connection: "open" });
    return;
  }
  const moment = await liveApi.moment();
  const { snapshot, decision, timeline, comparison, advisor } = moment;
  if (!isCurrentSource(store.state, sourceRequest)) return;
  if (!store.acceptSnapshot(snapshot)) return;
  const schema = store.state.data.schema?.environment_version === snapshot.provenance.environment_version
    ? store.state.data.schema : await liveApi.schema(snapshot.provenance.environment_version);
  const record = advisor.history?.length ? advisor.history[advisor.history.length - 1] : null;
  const recommendation = advisor.pending
    ? proposalFromAdvisor(advisor.pending, snapshot)
    : (record?.proposal ? proposalFromAdvisor(record.proposal, snapshot, { record }) : null);
  if (!isCurrentSource(store.state, sourceRequest)) return;
  store.patch({
    context: { comparator: snapshot.session.algorithms?.[1] || null },
    playback: { state: snapshot.session.state, speed: snapshot.session.speed,
      running: snapshot.session.running, awaitingDecision: snapshot.session.awaiting_decision },
    data: { decision, timeline, comparison, schema, recommendation },
    connection: "open", error: null,
    story: { bookmarks: timeline.events || [] },
  });
}

async function ensureSession() {
  const status = await liveApi.status();
  if (hasActiveSession(status)) return status;
  await liveApi.start({
    scenario: "demo_evening", algorithms: ["rl", "greedy"], seed: 42,
    model_tag: "ppo_te", safety_filter: true, speed: "1x", autostart: false,
    advisor: true, interface_mode: store.state.mode === "presentation" ? "present" : "advanced",
  });
  return refreshLive();
}

async function ensureStorySession() {
  if (store.state.source.kind !== "live_session") store.setSource("live_session");
  const status = await liveApi.status();
  if (!matchesStorySession(status)) await liveApi.start(storySessionConfig());
  await refreshLive();
}

async function playPause() {
  await ensureSession();
  if (store.state.data.snapshot?.session?.running) await liveApi.pause();
  else await liveApi.resume();
  await refreshLive();
}

async function step() {
  await ensureSession();
  if (store.state.data.snapshot?.session?.running) await liveApi.pause();
  await liveApi.step();
  await refreshLive();
}

async function propose() {
  await ensureSession();
  await liveApi.propose();
  await refreshLive();
}

async function approve() { await liveApi.approve(); await refreshLive(); }
async function reject() { await liveApi.reject(); await refreshLive(); }

async function counterfactual() {
  const decision = store.state.data.decision;
  const action = store.state.selection.actionId ?? decision?.selected_action?.action;
  if (action === null || action === undefined) throw new Error("Select an action before requesting an estimate.");
  const result = await liveApi.counterfactual({
    generation: store.state.context.generation,
    step: store.state.context.step,
    action,
  });
  store.patch({ data: { counterfactual: result } });
}

async function loadReplay(policy, scenario, seed) {
  const episode = await replayApi.episode(policy, scenario, seed);
  store.patch({ data: { replay: { ...store.state.data.replay, episode,
    policy_id: policy, scenario, seed, currentStep: 0 } } });
}

async function toggleStory() {
  if (store.state.story.active) {
    store.patch({ workflow: null, story: { active: false, auto: false, beat: 0, reviewBeat: null } });
    writeLocation(store.state);
    return;
  }
  store.setMode("presentation");
  if (store.state.source.kind !== "live_session") store.setSource("live_session");
  await liveApi.start(storySessionConfig());
  await refreshLive();
  store.patch({ workflow: "guided-story",
    story: { active: true, auto: false, beat: 0, reviewBeat: null } });
  writeLocation(store.state);
}

async function storyNext() {
  const current = store.state.story.reviewBeat ?? store.state.story.beat;
  const next = Math.min(10, current + 1);
  const beat = beatAt(next);
  if (beat.advance?.kind === "step") await step();
  else if (beat.advance?.kind === "propose") await propose();
  else if (beat.advance?.kind === "approve" && store.state.data.recommendation?.pending) await approve();
  else if (beat.advance?.kind === "runUntil") {
    await liveApi.runUntil(beat.advance.condition);
    await refreshLive();
  }
  store.patch({ story: { beat: Math.max(store.state.story.beat, next), reviewBeat: null } });
  if (beat.conclusion) { await loadEvidence(); shell.openDrawer("drawer-conclusion"); }
}

function storyPrevious() {
  const at = store.state.story.reviewBeat ?? store.state.story.beat;
  store.patch({ story: { reviewBeat: Math.max(0, at - 1) } });
}

function toggleStoryAuto() {
  if (!store.state.story.active) return;
  store.patch({ story: { auto: !store.state.story.auto } });
  scheduleStoryAuto();
}

function scheduleStoryAuto() {
  if (storyTimer) window.clearTimeout(storyTimer);
  storyTimer = null;
  if (!store.state.story.active || !store.state.story.auto || store.state.story.beat >= 10) return;
  storyTimer = window.setTimeout(() => run(async () => {
    await storyNext();
    scheduleStoryAuto();
  }), 6500);
}

function jumpBookmark(direction) {
  const events = store.state.data.timeline?.events || [];
  if (!events.length) return;
  const index = Math.max(-1, events.findIndex((event) => event.id === store.state.selection.eventId));
  const next = events[(index + direction + events.length) % events.length];
  actions.selectEvent(next);
}

function questionJump(question) {
  const destination = questionDestination(question);
  store.setMode(destination.mode);
  if (destination.rlView) store.patch({ rlView: destination.rlView });
  if (destination.source) {
    store.setSource(destination.source);
    run(() => loadSource(destination.source));
  }
  writeLocation(store.state);
  shell.closeDrawer();
  if (destination.conclusion) {
    run(loadEvidence);
    shell.openDrawer("drawer-conclusion");
  }
}

function toggleFilter(key, value) {
  const values = store.state.filters[key] || [];
  const next = values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
  store.patch({ filters: { [key]: next } });
}

async function run(operation) {
  try { await operation(); store.patch({ error: null }); }
  catch (error) { store.patch({ error: error?.message || String(error), connection: "error" }); }
}

onNavigate(async (route) => { applyRoute(route); await loadAppliedRoute(); });
window.addEventListener("beforeunload", () => {
  socket?.close();
  if (storyTimer) window.clearTimeout(storyTimer);
});
boot().catch((error) => store.patch({ error: error.message, connection: "error" }));
