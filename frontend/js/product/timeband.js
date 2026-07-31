/* The incident time band.
 *
 * A time-distance strip, not a chart. The cursor advances only when the
 * simulation or the recorded step advances; it never animates on its own.
 */

import { $, el, fill, icon } from "./dom.js";

const ICON_FOR_KIND = {
  congestion: "pressure", sla_risk: "pressure", failure: "failure",
  frr: "recovery", recovery: "recovery", stabilization: "recovery",
  recommendation: "recommendation", action: "route", reversal: "route",
  flap: "route",
};

/** Incident bookmarks a presenter jumps between. Ordinary TE actions are not. */
const BOOKMARK_KINDS = new Set([
  "congestion", "sla_risk", "failure", "frr", "recommendation",
  "recovery", "stabilization",
]);

export function renderTimeband(state, { onSelectEvent }) {
  const track = $("timeband-track");
  const timeline = state.data.timeline;

  if (!timeline || !timeline.events.length) {
    fill(track, el("li", {}, [
      el("p", { class: "tb-empty",
                text: state.source.kind === "live_session"
                  ? "No incident has occurred yet."
                  : "This record has no event timeline." }),
    ]));
    return [];
  }

  const currentEvent = state.selection.eventId;
  fill(track, timeline.events.map((event) => el("li", {}, [
    el("button", {
      type: "button",
      class: "tb-event",
      dataset: { kind: event.kind, eventId: event.id },
      "aria-current": event.id === currentEvent ? "step" : "false",
      onClick: () => onSelectEvent(event),
    }, [
      icon(ICON_FOR_KIND[event.kind] || "route"),
      el("span", { text: `${event.clock} ${event.title}` }),
    ]),
  ])));

  return timeline.events.filter((event) => BOOKMARK_KINDS.has(event.kind));
}

export function bookmarksFrom(timeline) {
  if (!timeline) return [];
  return timeline.events.filter((event) => BOOKMARK_KINDS.has(event.kind));
}
