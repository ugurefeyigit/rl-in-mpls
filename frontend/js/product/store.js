/* The one application state.
 *
 * Three modes share this store, which is the whole point of the redesign: a
 * mode change is a depth change over the same moment, so it must not touch
 * `source`, the live engine, or the selected object.
 *
 * Two guards live here because getting them wrong produces a *plausible* lie
 * rather than a visible error:
 *
 * - `generation` changes on reset. Any prior snapshot, pending recommendation
 *   or moment reference from the old generation is dropped, because a delta
 *   computed across a reset is meaningless.
 * - `sequence` increases per emitted payload. A frame that arrives out of order
 *   is discarded instead of being rendered as the present.
 */

import { MODE_IDS, sourceProfile } from "./contracts.js";

const initial = {
  mode: "network",
  workflow: null,
  rlView: "decision",

  source: { kind: "live_session", availability: {}, provenance: null },

  context: {
    environmentVersion: "v1",
    scenario: null,
    scenarioLabel: null,
    seed: null,
    policyId: null,
    checkpointId: null,
    comparator: null,
    sessionId: null,
    generation: null,
    sequence: -1,
    step: null,
    hour: null,
  },

  selection: { objectType: null, objectId: null, eventId: null, actionId: null },
  playback: { state: "idle", speed: "1x", running: false, awaitingDecision: false },
  story: { active: false, beat: 0, reviewBeat: null, bookmarks: [] },
  filters: { classes: [], conditions: [], search: "" },
  ui: { audienceView: false, fullscreen: false, openDrawer: null, explainDepth: "presentation",
        topologyList: false, zoom: 1 },

  data: {
    capabilities: null, contracts: null, displayMap: null,
    snapshot: null, previousSnapshot: null, decision: null, timeline: null,
    comparison: null, recommendation: null, counterfactual: null,
    schema: null, evidence: {}, replay: null,
  },

  connection: "connecting",
  error: null,
};

export function createStore() {
  let state = structuredClone(initial);
  const listeners = new Set();

  function notify(changed) {
    for (const listener of listeners) listener(state, changed);
  }

  return {
    get state() { return state; },

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    /** Shallow-merge one or more top-level slices. */
    patch(partial) {
      const changed = new Set(Object.keys(partial));
      const next = { ...state };
      for (const [key, value] of Object.entries(partial)) {
        next[key] = (value && typeof value === "object" && !Array.isArray(value)
          && state[key] && typeof state[key] === "object" && !Array.isArray(state[key]))
          ? { ...state[key], ...value }
          : value;
      }
      state = next;
      notify(changed);
    },

    setMode(mode) {
      if (!MODE_IDS.includes(mode)) throw new Error(`unknown mode: ${mode}`);
      if (state.mode === mode) return;
      // A mode change never mutates source, context or selection.
      state = { ...state, mode, workflow: mode === "presentation" ? state.workflow : null };
      notify(new Set(["mode"]));
    },

    setSource(kind, availability = {}) {
      sourceProfile(kind);
      if (state.source.kind === kind) return;
      state = {
        ...state,
        source: { kind, availability, provenance: null },
        // Live-only artifacts cannot survive a move to a recorded or frozen record.
        data: { ...state.data, snapshot: null, previousSnapshot: null,
                decision: null, recommendation: null, counterfactual: null },
        selection: { objectType: null, objectId: null, eventId: null, actionId: null },
      };
      notify(new Set(["source", "data", "selection"]));
    },

    select(objectType, objectId) {
      state = { ...state, selection: { ...state.selection, objectType, objectId } };
      notify(new Set(["selection"]));
    },

    selectEvent(eventId) {
      state = { ...state, selection: { ...state.selection, eventId } };
      notify(new Set(["selection"]));
    },

    /**
     * Accept a live snapshot only if it belongs to the current generation and
     * is not older than what is already displayed.
     */
    acceptSnapshot(snapshot) {
      const provenance = snapshot?.provenance;
      if (!provenance) return false;
      const current = state.context;
      const sameRun = current.sessionId === provenance.session_id
        && current.generation === provenance.generation;

      if (sameRun && provenance.sequence < current.sequence) return false;

      const generationChanged = current.sessionId !== provenance.session_id
        || current.generation !== provenance.generation;

      state = {
        ...state,
        context: {
          ...current,
          environmentVersion: provenance.environment_version,
          scenario: provenance.scenario,
          scenarioLabel: provenance.scenario_label,
          seed: provenance.seed,
          policyId: provenance.policy_id,
          checkpointId: provenance.checkpoint_id,
          sessionId: provenance.session_id,
          generation: provenance.generation,
          sequence: provenance.sequence,
          step: provenance.step,
          hour: snapshot.time?.hour ?? null,
        },
        source: { ...state.source, provenance },
        data: {
          ...state.data,
          // A delta may only be drawn inside one generation.
          previousSnapshot: generationChanged ? null : state.data.snapshot,
          snapshot,
          ...(generationChanged
            ? { decision: null, recommendation: null, counterfactual: null }
            : {}),
        },
        ...(generationChanged
          ? { selection: { objectType: null, objectId: null, eventId: null, actionId: null },
              story: { ...state.story, reviewBeat: null } }
          : {}),
      };
      notify(new Set(["context", "data", "source"]));
      return true;
    },
  };
}
