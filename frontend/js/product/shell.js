/* Shared application shell.
 *
 * This module owns navigation, drawers, keyboard behavior and composition. It
 * never fetches data and never executes an action; those capabilities stay in
 * the source adapter passed in by main.js.
 */

import { renderControlPanel } from "./control-panel.js";
import { $, fill, isTypingTarget, trapFocus } from "./dom.js";
import { renderExplain } from "./explain.js";
import { renderHelp, renderQuestions } from "./help.js";
import { renderNetwork, networkStageTitle } from "./modes/network.js";
import { renderPresentation } from "./modes/presentation.js";
import { renderRl } from "./modes/rl.js";
import { renderCompare } from "./modes/compare.js";
import { renderContextLedger, renderProvenance, renderSourceSwitch } from "./provenance.js";
import { renderRecommendation } from "./recommendation-card.js";
import { writeLocation } from "./router.js";
import { renderTimeband } from "./timeband.js";
import { renderTopologyList } from "./topology-list.js";

export function mountShell({ store, atlas, actions }) {
  const drawerState = { teardown: null, invoker: null };

  function render() {
    const state = store.state;
    document.body.dataset.mode = state.mode;
    document.body.dataset.audience = state.ui.audienceView ? "on" : "off";
    document.body.dataset.comparisonFocus = state.comparisonFocus.runId ? "on" : "off";

    for (const mode of ["presentation", "network", "rl", "compare"]) {
      const link = $(`mode-${mode}`);
      link.setAttribute("aria-current", state.mode === mode ? "page" : "false");
      $(`panel-${mode}`).hidden = state.mode !== mode;
    }
    $("mode-surface-title").textContent = ({
      presentation: "Presentation", network: "Network Information",
      rl: "RL Information", compare: "Comparative Run Results",
    })[state.mode];
    $("moment-rail").hidden = state.mode !== "presentation";
    renderControlPanel(state, controlHandlers());
    renderAudienceExit(state);

    renderProvenance(state);
    renderContextLedger(state);
    renderSourceSwitch(state, { onSelect: actions.setSource });
    renderConnection(state);
    renderStage(state);
    renderRecommendation(state);
    renderTimeband(state, { onSelectEvent: actions.selectEvent });

    if (state.mode === "presentation") renderPresentation(state, comparisonHandlers());
    else if (state.mode === "network") renderNetwork(state, networkHandlers());
    else if (state.mode === "rl") renderRl(state, rlHandlers());
    else renderCompare(state, comparisonHandlers());

    renderExplain(state);
    renderButtons(state);
  }

  function renderStage(state) {
    const storedFocus = Boolean(state.comparisonFocus.runId);
    const live = state.source.kind === "live_session" && !storedFocus;
    const snapshot = live ? state.data.snapshot : referenceSnapshot(state.data.displayMap);
    const note = $("atlas-unavailable");
    const showTelemetry = live && Boolean(state.data.snapshot);
    if (snapshot) {
      atlas.update(snapshot, {
        showTelemetry,
        selectedDemand: state.selection.objectType === "demand" ? state.selection.objectId : null,
      });
      atlas.select(state.selection.objectType, state.selection.objectId);
    }
    if (live) {
      note.hidden = true;
      note.textContent = "";
    } else {
      note.hidden = false;
      note.textContent = storedFocus
        ? "REFERENCE TOPOLOGY · STORED RUN HAS AGGREGATE INTERVAL DATA, NOT A PER-LINK SNAPSHOT"
        : state.source.kind === "recorded_replay"
        ? "REFERENCE TOPOLOGY · NO RECORDED LINK TELEMETRY"
        : "REFERENCE TOPOLOGY · THIS EVIDENCE CARRIES NO LINK TELEMETRY";
    }
    $("stage-title").textContent = storedFocus ? "Stored completed-run interval reference" : state.mode === "network"
      ? networkStageTitle(state.data.snapshot)
      : (state.mode === "presentation" ? "Current network moment" : "Network reference");
    $("atlas-disclaimer").textContent = `${state.data.displayMap?.layout_note || "Fixed engineering schematic · not geographic"} · Fictional scaled network, not a real operator.`;
    renderTopologyList(snapshot, {
      selection: state.selection,
      showTelemetry,
      onSelect: actions.selectObject,
    });
  }

  function renderConnection(state) {
    const host = $("conn-state");
    host.dataset.state = state.connection;
    $("conn-text").textContent = state.connection === "open" ? "Connected"
      : (state.connection === "lost" ? "Connection lost · values are last received"
        : state.connection === "error" ? "Unavailable" : "Connecting");
    $("error-banner").hidden = !state.error;
    $("error-banner").textContent = state.error || "";
  }

  function renderButtons(state) {
    const session = state.data.snapshot?.session;
    $("btn-session-primary").textContent = state.source.kind !== "live_session"
      ? "Switch to live session"
      : (session?.running ? "Pause live session" : (session ? "Resume live session" : "Start live session"));
    $("btn-audience").setAttribute("aria-pressed", state.ui.audienceView ? "true" : "false");
    $("btn-fullscreen").setAttribute("aria-pressed", document.fullscreenElement ? "true" : "false");
  }

  /* Audience view hides the working chrome, so its exit must live outside that
   * chrome. It renders whenever audience view is on, in every mode and in
   * fullscreen, and Escape does the same thing without reloading. */
  function renderAudienceExit(state) {
    $("btn-audience-exit").hidden = !state.ui.audienceView;
  }

  /* Leaving audience view must never drop focus onto <body>.
   *
   * The audience toggle lives in the header tool row, which the narrow-viewport
   * rules hide entirely — so on a phone the toggle is not focusable and the
   * obvious `focus()` silently did nothing. Fall back to the mode surface,
   * which is always present and always focusable. */
  function restoreFocusAfterAudience() {
    const toggle = $("btn-audience");
    if (toggle && toggle.offsetParent !== null) { toggle.focus(); return; }
    $("mode-surface").focus();
  }

  function controlHandlers() {
    return {
      onSetup: actions.setup,
      onStart: actions.startRun,
      onPlayPause: actions.playPause,
      onStep: actions.step,
      onNextEvent: actions.nextEvent,
      onPause: actions.pause,
      onResetRun: actions.resetRun,
      onFullReset: actions.fullReset,
      onSpeed: actions.setSpeed,
      onApprove: actions.approve,
      onReject: actions.reject,
      onToggleStory: actions.toggleStory,
      onToggleStoryAuto: actions.toggleStoryAuto,
      onStoryNext: actions.storyNext,
      onStoryPrevious: actions.storyPrevious,
      onStoryRestart: actions.storyRestart,
      onLoadResults: actions.loadResults,
      onSaveRun: actions.saveRun,
      onOpenEvidence: actions.openEvidence,
      onOpenConclusion: () => { actions.openConclusion(); },
      onOpenQuestions: () => openDrawer("drawer-questions"),
    };
  }

  function comparisonHandlers() {
    return {
      onAssign: actions.assignComparativeRun,
      onClear: actions.clearComparativeRun,
      onClearAll: actions.clearComparativeRuns,
      onSwap: actions.swapComparativeRuns,
      onSelectStep: actions.selectComparisonStep,
      onRewardView: actions.setComparisonRewardView,
      onResetView: actions.resetComparisonView,
    };
  }

  function networkHandlers() {
    return {
      onToggleClass: actions.toggleClass,
      onToggleCondition: actions.toggleCondition,
      onSearch: actions.setNetworkSearch,
      onClear: actions.clearFilters,
      onSelectDemand: (id) => actions.selectObject("demand", id),
    };
  }

  function rlHandlers() {
    return {
      onSetView: actions.setRlView,
      onObservationSearch: actions.setObservationSearch,
      onToggleObservationChanged: actions.toggleObservationChanged,
      onToggleInvalidActions: actions.toggleInvalidActions,
      onSelectAction: actions.selectAction,
      onCounterfactual: actions.counterfactual,
      onLoadReplay: actions.loadReplay,
      onScrubReplay: actions.scrubReplay,
    };
  }

  function openDrawer(id, invoker = document.activeElement) {
    closeDrawer(false);
    const drawer = $(id);
    if (!drawer) return;
    drawer.hidden = false;
    drawerState.invoker = invoker;
    drawerState.teardown = trapFocus(drawer, () => closeDrawer());
    store.patch({ ui: { openDrawer: id } });
    requestAnimationFrame(() => drawer.querySelector("button, [href], [tabindex]")?.focus());
  }

  function closeDrawer(restore = true) {
    const id = store.state.ui.openDrawer;
    if (id) $(id).hidden = true;
    drawerState.teardown?.();
    drawerState.teardown = null;
    store.patch({ ui: { openDrawer: null } });
    if (restore) drawerState.invoker?.focus?.();
  }

  function bind() {
    $("mode-nav").addEventListener("click", (event) => {
      const link = event.target.closest("[data-mode]");
      if (!link) return;
      event.preventDefault();
      store.setMode(link.dataset.mode);
      writeLocation(store.state);
      $("mode-surface").focus();
    });
    $("btn-topology-list").addEventListener("click", () => {
      const next = !store.state.ui.topologyList;
      store.patch({ ui: { topologyList: next } });
      $("topology-list").hidden = !next;
      $("atlas").hidden = next;
      $("btn-topology-list").setAttribute("aria-expanded", next ? "true" : "false");
    });
    $("btn-zoom-in").addEventListener("click", () => atlas.setZoom(atlas.zoom * 1.25));
    $("btn-zoom-out").addEventListener("click", () => atlas.setZoom(atlas.zoom / 1.25));
    $("btn-fit").addEventListener("click", () => atlas.fitTo(store.state.selection.objectId));
    $("btn-reset-view").addEventListener("click", () => atlas.resetView());
    $("btn-explain").addEventListener("click", (event) => openDrawer("drawer-explain", event.currentTarget));
    $("btn-help").addEventListener("click", (event) => openDrawer("drawer-help", event.currentTarget));
    document.querySelectorAll("[data-close]").forEach((button) =>
      button.addEventListener("click", () => closeDrawer()));
    document.querySelectorAll("[data-depth]").forEach((button) => button.addEventListener("click", () => {
      store.patch({ ui: { explainDepth: button.dataset.depth } });
    }));
    $("btn-audience").addEventListener("click", actions.toggleAudience);
    $("btn-audience-exit").addEventListener("click", () => {
      actions.exitAudience();
      restoreFocusAfterAudience();
    });
    $("btn-fullscreen").addEventListener("click", actions.toggleFullscreen);
    $("btn-session-primary").addEventListener("click", () => {
      if (store.state.source.kind !== "live_session") actions.setSource("live_session");
      else actions.playPause();
    });
    document.addEventListener("fullscreenchange", render);
    document.addEventListener("keydown", onKeyDown);
  }

  function onKeyDown(event) {
    if (isTypingTarget(event.target)) return;
    if (event.altKey && ["1", "2", "3", "4"].includes(event.key)) {
      event.preventDefault();
      store.setMode({ "1": "presentation", "2": "network", "3": "rl", "4": "compare" }[event.key]);
      writeLocation(store.state);
      return;
    }
    if (event.key === "Escape") {
      // Order matters: a drawer first, then audience view, then fullscreen.
      // Audience view must always be escapable, including while fullscreen.
      if (store.state.ui.openDrawer) { closeDrawer(); return; }
      if (store.state.ui.audienceView) {
        actions.exitAudience();
        restoreFocusAfterAudience();
        return;
      }
      if (document.fullscreenElement) document.exitFullscreen();
      return;
    }
    if (event.key === " ") { event.preventDefault(); actions.playPause(); }
    else if (event.key === "ArrowRight") {
      if (store.state.story.active) actions.storyNext(); else actions.step();
    } else if (event.key === "ArrowLeft" && store.state.story.active) actions.storyPrevious();
    else if (event.key.toLowerCase() === "g" && store.state.mode === "presentation") actions.toggleStory();
    else if (event.key.toLowerCase() === "e") openDrawer("drawer-explain");
    else if (event.key === "?") openDrawer("drawer-help");
    else if (event.key === "[") actions.jumpBookmark(-1);
    else if (event.key === "]") actions.jumpBookmark(1);
    else if (event.key === "/") { event.preventDefault(); document.querySelector(
      store.state.mode === "rl" ? "#obs-search" : "#network-search")?.focus(); }
  }

  renderHelp();
  renderQuestions({ onJump: actions.questionJump });
  bind();
  store.subscribe(render);
  render();
  return { render, openDrawer, closeDrawer };
}

function referenceSnapshot(map) {
  if (!map) return null;
  const counts = new Map(map.nodes.map((node) => [node.id, 0]));
  for (const link of map.links) {
    counts.set(link.a, (counts.get(link.a) || 0) + 1);
    counts.set(link.z, (counts.get(link.z) || 0) + 1);
  }
  return {
    nodes: map.nodes.map((node) => ({ ...node, n_links: counts.get(node.id), n_lsps: 0,
      worst_adjacent_utilization: null, has_failed_link: false })),
    links: map.links.map((link) => ({ ...link, up: true, state: "normal", band: "quiet",
      worst_direction: null, worst_utilization: null, pressure_ticks: 0 })),
    demands: [],
  };
}
