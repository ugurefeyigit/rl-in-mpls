/* Product bootstrap and source orchestration. */

import { api as evidenceApi } from "./adapters/evidence-v2.js";
import { api as liveApi, connect, hasActiveSession } from "./adapters/live-v1.js";
import { api as replayApi } from "./adapters/recorded-v2.js";
import { BEATS, beatAt, matchesStorySession, storyContext, storySessionConfig } from "./guided-story.js";
import { renderConclusion } from "./governed-study.js";
import { questionDestination } from "./help.js";
import { explanationFromDecision, proposalFromAdvisor } from "./recommendation-card.js";
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
  setup: (partial) => { store.patch({ setup: partial }); reconcileSetup(); },
  startRun: () => run(startRun),
  resetRun: () => run(resetRun),
  fullReset: () => run(fullReset),
  pause: () => run(() => liveApi.pause().then(refreshLive)),
  openEvidence: (kind) => run(() => setSource(kind)),
  openConclusion: () => { run(loadEvidence); shell.openDrawer("drawer-conclusion"); },
  storyRestart: () => run(restartStory),
  exitAudience: () => store.patch({ ui: { audienceView: false } }),
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
  nextEvent: () => run(() => fastForward("next_event")),
  loadResults: () => run(loadResults),
  saveRun: () => run(saveRun),
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
  const [capabilities, contracts, displayMap, scenarios] = await Promise.all([
    liveApi.capabilities(), liveApi.contracts(), liveApi.displayMap(),
    liveApi.scenarios(),
  ]);
  store.patch({ data: { capabilities, contracts, displayMap, scenarios } });
  applyCapabilityDefaults(capabilities, scenarios);
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

/* ------------------------------------------------------------- fast-forward */

/**
 * Skip to a condition. Advisor execution refuses a fast-forward unless the
 * operator delegates the stretch, because it applies the controller's own
 * actions without individual approval. The confirmation is the delegation: it
 * is asked once, here, and recorded server-side as one delegated batch.
 */
async function fastForward(condition) {
  await ensureSession();
  const advisor = store.state.data.snapshot?.session?.execution === "advisor";
  if (advisor && !window.confirm(
    "This session runs with advisor approval.\n\n"
    + "Skipping ahead applies the controller's own actions for a stretch of "
    + "intervals in one gesture. Those intervals are NOT approved individually, "
    + "and the approval history will record this as one delegated batch.\n\n"
    + "Delegate this stretch?")) {
    return;
  }
  const outcome = await liveApi.runUntil(condition, 300, advisor);
  await refreshLive();
  if (outcome?.delegated) {
    store.patch({ delegation: { note: outcome.note, steps: outcome.steps } });
  }
}

/* ----------------------------------------------------------------- results */

async function loadResults() {
  const results = await liveApi.results();
  store.patch({ data: { results } });
}

async function saveRun() {
  const outcome = await liveApi.saveRun();
  store.patch({ savedRun: outcome });
  await loadResults();
}

async function refreshLiveOnce() {
  if (store.state.source.kind !== "live_session") return;
  const sourceRequest = captureSource(store.state);
  const status = await liveApi.status();
  if (!isCurrentSource(store.state, sourceRequest)) return;
  if (!hasActiveSession(status)) {
    store.patch({ data: { snapshot: null, previousSnapshot: null, decision: null,
      timeline: null, comparison: null, recommendation: null }, connection: "open" });
    await loadResults();
    return;
  }
  // The displayed moment is one atomic read and stays one atomic read. Results
  // are a separate concern — retained runs and a pointer to the study — so they
  // are read afterwards and never batched into the moment.
  const moment = await liveApi.moment();
  const results = await liveApi.results();
  const { snapshot, decision, timeline, comparison, advisor } = moment;
  if (!isCurrentSource(store.state, sourceRequest)) return;
  if (!store.acceptSnapshot(snapshot)) return;
  const schema = store.state.data.schema?.environment_version === snapshot.provenance.environment_version
    ? store.state.data.schema : await liveApi.schema(snapshot.provenance.environment_version);
  // Only a proposal record can become a recommendation card. A delegated batch
  // is an operator decision about a stretch of intervals, not an action to
  // explain, and rendering it as one would misreport what was approved.
  const proposals = advisor.proposals
    || (advisor.history || []).filter((row) => row.kind !== "delegated_batch");
  const record = proposals.length ? proposals[proposals.length - 1] : null;
  // Advisor execution has a live proposal to approve. Automatic execution has
  // no proposal at all: the card explains the decision the policy already made.
  const recommendation = advisor.pending
    ? proposalFromAdvisor(advisor.pending, snapshot, { execution: "advisor" })
    : (record
        ? proposalFromAdvisor(record, snapshot, { record, execution: "advisor" })
        : explanationFromDecision(decision, snapshot));
  if (!isCurrentSource(store.state, sourceRequest)) return;
  store.patch({
    context: { comparator: snapshot.session.algorithms?.[1] || null },
    playback: { state: snapshot.session.state, speed: snapshot.session.speed,
      running: snapshot.session.running, awaitingDecision: snapshot.session.awaiting_decision },
    data: { decision, timeline, comparison, schema, recommendation, advisor, results },
    connection: "open", error: null,
    story: { bookmarks: timeline.events || [] },
  });
}

/* --------------------------------------------------------------- run setup */

/** Seed the control panel from what this installation can actually run. */
function applyCapabilityDefaults(capabilities, scenarios) {
  const environment = capabilities?.default_environment || "v2";
  const policies = (capabilities?.live_policies || [])
    .filter((policy) => policy.environment_version === environment);
  const preferred = capabilities?.checkpoint_registry?.default_policy;
  const learner = policies.find((policy) => policy.id === preferred && policy.available)
    || policies.find((policy) => policy.family === "learner" && policy.available)
    || policies.find((policy) => policy.available);
  const comparator = policies.find(
    (policy) => policy.family === "baseline" && policy.id !== learner?.id);
  const scenario = Object.keys(scenarios || {}).includes(store.state.setup.scenario)
    ? store.state.setup.scenario : Object.keys(scenarios || {})[0];
  store.patch({ setup: {
    environment,
    scenario: scenario || store.state.setup.scenario,
    policyA: learner?.id || store.state.setup.policyA,
    policyB: comparator?.id || store.state.setup.policyB,
    trainingRoot: capabilities?.checkpoint_registry?.default_training_root
      ?? store.state.setup.trainingRoot,
  } });
}

/** Keep the pickers consistent when the environment changes under them. */
function reconcileSetup() {
  const setup = store.state.setup;
  const policies = (store.state.data.capabilities?.live_policies || [])
    .filter((policy) => policy.environment_version === setup.environment);
  if (!policies.length) return;
  const patch = {};
  if (!policies.some((policy) => policy.id === setup.policyA)) {
    patch.policyA = (policies.find((p) => p.family === "learner" && p.available)
      || policies[0]).id;
  }
  if (!policies.some((policy) => policy.id === setup.policyB)) {
    patch.policyB = (policies.find((p) => p.family === "baseline") || policies[0]).id;
  }
  if (Object.keys(patch).length) store.patch({ setup: patch });
}

function startConfig() {
  const setup = store.state.setup;
  const algorithms = setup.compare && setup.policyB && setup.policyB !== setup.policyA
    ? [setup.policyA, setup.policyB] : [setup.policyA];
  return {
    scenario: setup.scenario,
    environment: setup.environment,
    algorithms,
    seed: Number(setup.seed),
    training_root: Number(setup.trainingRoot),
    model_tag: setup.environment === "v1" ? "ppo_te" : null,
    safety_filter: true,
    speed: setup.speed,
    autostart: false,
    execution: setup.execution,
    advisor: setup.execution === "advisor",
    interface_mode: store.state.mode === "presentation" ? "present" : "advanced",
  };
}

async function startRun() {
  if (store.state.source.kind !== "live_session") store.setSource("live_session");
  store.patch({ story: { active: false, auto: false, beat: 0, reviewBeat: null },
                workflow: null });
  await liveApi.start(startConfig());
  await refreshLive();
}

async function resetRun() {
  await liveApi.reset();
  await refreshLive();
}

/** Full reset: stop the runners, drop transient UI state, keep the settings. */
async function fullReset() {
  if (storyTimer) window.clearTimeout(storyTimer);
  storyTimer = null;
  await liveApi.stop();
  shell?.closeDrawer();
  store.patch({
    workflow: null,
    story: { active: false, auto: false, beat: 0, reviewBeat: null, bookmarks: [] },
    selection: { objectType: null, objectId: null, eventId: null, actionId: null },
    ui: { audienceView: false, openDrawer: null, topologyList: false },
    data: { snapshot: null, previousSnapshot: null, decision: null, timeline: null,
            comparison: null, recommendation: null, counterfactual: null,
            advisor: null },
    playback: { state: "idle", speed: store.state.setup.speed, running: false,
                awaitingDecision: false },
    error: null,
  });
  await refreshLive();
}

async function ensureSession() {
  const status = await liveApi.status();
  if (hasActiveSession(status)) return status;
  await liveApi.start(startConfig());
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

async function approve() {
  await liveApi.approve();
  await refreshLive();
  scheduleStoryAuto();
}

async function reject() {
  await liveApi.reject();
  await refreshLive();
  scheduleStoryAuto();
}

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
  // A held recommendation is a hard stop for both manual and automatic pacing.
  // The beat does not advance until the operator approves or rejects it.
  if (store.state.data.recommendation?.pending) {
    throw new Error("This beat is waiting for you: approve or reject the "
      + "recommendation before continuing.");
  }
  const current = store.state.story.reviewBeat ?? store.state.story.beat;
  const next = Math.min(BEATS.length - 1, current + 1);
  const beat = beatAt(next);
  if (beat.advance?.kind === "step") await step();
  else if (beat.advance?.kind === "propose") await propose();
  else if (beat.advance?.kind === "approve") {
    // Beat 8 is "observe the transition", so it needs an applied action. If
    // nothing is held it proposes first rather than silently doing nothing.
    if (!store.state.data.snapshot?.session?.awaiting_decision) await propose();
    if (store.state.data.snapshot?.session?.awaiting_decision) await approve();
  } else if (beat.advance?.kind === "runUntil") {
    // The story runs with advisor approval, so its own fast-forwards are
    // delegated stretches. The beat copy says so; the server records it.
    await liveApi.runUntil(beat.advance.condition, 300, true);
    await refreshLive();
  }
  store.patch({ story: { beat: Math.max(store.state.story.beat, next), reviewBeat: null } });
  // A beat that names an object selects it, so the topology follows the copy.
  const selection = beat.select?.(storyContext(store.state));
  if (selection) store.select(selection.objectType, selection.objectId);
  if (beat.conclusion) { await loadEvidence(); shell.openDrawer("drawer-conclusion"); }
}

async function restartStory() {
  if (storyTimer) window.clearTimeout(storyTimer);
  storyTimer = null;
  await liveApi.start(storySessionConfig());
  await refreshLive();
  store.patch({ workflow: "guided-story",
    story: { active: true, auto: false, beat: 0, reviewBeat: null } });
  writeLocation(store.state);
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
  if (!store.state.story.active || !store.state.story.auto
      || store.state.story.beat >= BEATS.length - 1) return;
  // Automatic playback holds at a pending recommendation rather than answering
  // it. Approve or Reject resumes the schedule.
  if (store.state.data.recommendation?.pending) return;
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
