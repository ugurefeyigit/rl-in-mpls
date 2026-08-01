/* Network Information: the MPLS traffic-engineering workspace.
 *
 * Topology, tables and incident time are one linked object. Selecting a link
 * filters the demands that cross it; selecting a demand focuses its route on the
 * map. There is one selection and one moment behind all three.
 */

import { $, el, fill, unavailable } from "../dom.js";
import { count, mbps, metricDelta, metricValue, percent } from "../format.js";
import { renderDemandRiskTable, riskSummary } from "../demand-risk-table.js";
import { matchesDemand, renderFilterBar } from "../network-filters.js";
import { crosses, renderInspector } from "../object-inspector.js";
import { renderComparisonContext } from "../comparison-context.js";

export function renderNetwork(state, handlers) {
  const snapshot = state.data.snapshot;
  const panel = $("panel-network");

  if (state.comparisonFocus.runId) {
    fill(panel, [renderComparisonContext(state, "network")]);
    fill($("rail"), []);
    return;
  }

  if (state.source.kind !== "live_session") {
    fill(panel, [unavailable("Network Information",
      "This mode reads a running session. The current record is " +
      `${state.source.kind.replace(/_/g, " ")}, which carries no live network state. ` +
      "Switch the record to LIVE to inspect the network.")]);
    fill($("rail"), []);
    return;
  }
  if (!snapshot) {
    fill(panel, [unavailable("Network Information",
      "No session is running. Start one to inspect the network.")]);
    return;
  }

  const previousDemands = state.data.previousSnapshot?.demands || [];
  const priorById = new Map(previousDemands.map((d) => [d.id, d]));
  const selectedLink = state.selection.objectType === "link"
    ? (snapshot.links || []).find((l) => l.id === state.selection.objectId)
    : null;

  let demands = (snapshot.demands || [])
    .filter((d) => matchesDemand(d, state.filters, priorById.get(d.id)));
  if (selectedLink) demands = demands.filter((d) => crosses(d, selectedLink));

  fill(panel, [
    renderFilterBar(state, {
      ...handlers,
      matched: demands.length,
      total: (snapshot.demands || []).length,
    }),
    selectedLink
      ? el("p", { class: "filters__state", role: "status",
          text: `Showing demands crossing ${selectedLink.a_city}–${selectedLink.z_city} ` +
                `(${selectedLink.id}).` })
      : null,
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Demand and SLA risk" }),
      renderDemandRiskTable(demands, {
        selection: state.selection,
        onSelect: handlers.onSelectDemand,
        previous: previousDemands,
      }),
    ]),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Restoration and change this run" }),
      restorationSequence(state),
    ]),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "What this simulator models" }),
      disclosure(),
    ]),
  ]);

  renderNetworkRail(state, snapshot);
}

function renderNetworkRail(state, snapshot) {
  fill($("rail"), [
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Current interval" }),
      metricsList(state, snapshot),
    ]),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Risk summary" }),
      riskSummary(snapshot.demands || []),
    ]),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Selected object" }),
      renderInspector(state),
    ]),
  ]);
}

function metricsList(state, snapshot) {
  const metrics = snapshot.metrics;
  if (!metrics.available) return unavailable("Interval metrics", metrics.reason);
  const rows = Object.entries(metrics.values);
  return el("table", { class: "grid" }, [
    el("caption", { text: metrics.has_previous
      ? "Current interval against the previous one."
      : "Current interval. No previous interval exists in this generation." }),
    el("thead", {}, [el("tr", {}, [
      el("th", { scope: "col", text: "Measure" }),
      el("th", { scope: "col", text: "Now" }),
      el("th", { scope: "col", text: "Δ" }),
    ])]),
    el("tbody", {}, rows.map(([key, row]) => el("tr", {}, [
      el("th", { scope: "row", text: row.label }),
      el("td", { text: metricValue(row.unit, row.value) }),
      el("td", { text: row.delta === null ? "—" : metricDelta(row.unit, row.delta) }),
    ]))),
  ]);
}

function restorationSequence(state) {
  const timeline = state.data.timeline;
  if (!timeline || !timeline.events.length) {
    return el("p", { class: "tb-empty", text: "No event has been recorded yet." });
  }
  const relevant = timeline.events.filter((e) =>
    ["failure", "frr", "recovery", "action", "reversal", "flap", "stabilization"]
      .includes(e.kind));
  if (!relevant.length) {
    return el("p", { class: "tb-empty",
      text: "No failure, protection move, reroute or restoration has happened yet." });
  }
  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid" }, [
      el("caption", { text: timeline.frr_note }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Time" }),
        el("th", { scope: "col", text: "Event" }),
        el("th", { scope: "col", text: "Actor" }),
        el("th", { scope: "col", text: "Object" }),
        el("th", { scope: "col", text: "Detail" }),
      ])]),
      el("tbody", {}, relevant.slice(-40).map((event) => el("tr", {}, [
        el("th", { scope: "row", text: event.clock }),
        el("td", { text: event.kind === "frr" ? "Protection (FRR)" : event.kind }),
        el("td", { text: event.actor_label || event.actor || "—" }),
        el("td", { text: event.object_id || "—" }),
        el("td", { text: event.detail }),
      ]))),
    ]),
  ]);
}

function disclosure() {
  return el("details", { class: "disclosure" }, [
    el("summary", { text: "Modeled behaviour versus a real operator network" }),
    el("div", { class: "prose" }, [
      el("p", { text: "Modeled: flow-level demand, candidate LSP choice, analytic " +
        "delay and loss, scripted and manual link state, FRR-style local repair, " +
        "reroute dwell and cooldown, and traffic-engineering actions." }),
      el("p", { text: "Not modeled: packets, TCP behaviour, RSVP-TE or IGP " +
        "convergence, label signaling, exact geography, real operator topology, " +
        "and production control-plane timing. No unimplemented MPLS feature is " +
        "offered here as a control." }),
    ]),
  ]);
}

/** Node telemetry summary used by the stage title in this mode. */
export function networkStageTitle(snapshot) {
  if (!snapshot) return "Network";
  const incident = snapshot.incident;
  if (incident.failed_links.length) {
    return `Network · ${count(incident.failed_links.length)} link(s) down`;
  }
  if (incident.congested_links.length) {
    return `Network · ${count(incident.congested_links.length)} congested`;
  }
  const max = snapshot.metrics.available
    ? snapshot.metrics.values.max_util?.value : null;
  return max === null || max === undefined
    ? "Network" : `Network · busiest ${percent(max, 0)}`;
}

export { mbps };
