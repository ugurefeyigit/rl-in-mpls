/* URL ↔ state.
 *
 * The four legacy URLs keep working and keep meaning what they meant:
 * `/present` is Presentation, `/` and `/advanced` are Network Information, and
 * `/study` is RL Information at the governed study. The server renders the shell
 * for each of them, so there is no redirect flash.
 *
 * Deep-link parameters reconstruct a moment; a parameter that cannot be honoured
 * is dropped rather than approximated.
 */

import { MODE_IDS, MODE_ROUTE, ROUTES, RL_VIEWS, SOURCE_KINDS } from "./contracts.js";

export function readLocation(loc = window.location) {
  const base = ROUTES[loc.pathname] || ROUTES["/"];
  const params = new URLSearchParams(loc.search);
  const state = {
    mode: base.mode,
    source: base.source,
    rlView: base.rlView || "decision",
    workflow: null,
    selection: { objectType: null, objectId: null },
    eventId: null,
    step: null,
  };

  const mode = params.get("mode");
  if (mode && MODE_IDS.includes(mode)) state.mode = mode;

  const source = params.get("source");
  if (source && source in SOURCE_KINDS) state.source = source;

  const view = params.get("view");
  if (view && RL_VIEWS.includes(view)) state.rlView = view;

  if (params.get("workflow") === "guided-story" && state.mode === "presentation") {
    state.workflow = "guided-story";
    state.source = "live_session";
  }

  const object = params.get("object");
  if (object && object.includes(":")) {
    const [objectType, objectId] = object.split(":", 2);
    if (["router", "link", "demand"].includes(objectType)) {
      state.selection = { objectType, objectId };
    }
  }

  const event = params.get("event");
  if (event) state.eventId = event;

  const step = params.get("step");
  if (step && /^\d+$/.test(step)) state.step = Number(step);

  return state;
}

/** Write the current mode and moment back to the address bar without a reload. */
export function writeLocation(state, { replace = false } = {}) {
  const url = locationForState(state);
  if (url === window.location.pathname + window.location.search) return;
  window.history[replace ? "replaceState" : "pushState"]({ url }, "", url);
}

export function locationForState(state) {
  const path = MODE_ROUTE[state.mode] || "/";
  const base = ROUTES[path] || ROUTES["/"];
  const params = new URLSearchParams();
  if (state.workflow) params.set("workflow", state.workflow);
  if (state.mode === "rl" && state.rlView
      && state.rlView !== (base.rlView || "decision")) {
    params.set("view", state.rlView);
  }
  if (state.source.kind !== base.source) params.set("source", state.source.kind);
  if (state.selection.objectType && state.selection.objectId) {
    params.set("object", `${state.selection.objectType}:${state.selection.objectId}`);
  }
  if (state.selection.eventId) params.set("event", state.selection.eventId);

  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

export function onNavigate(handler) {
  window.addEventListener("popstate", () => handler(readLocation()));
}
