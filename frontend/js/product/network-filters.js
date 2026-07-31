/* Network Information filters.
 *
 * Filters change what is emphasised, not what exists. They never hide the
 * currently focused object without saying so, and the bar always reports how
 * many objects match with a way back to everything — an apparently empty
 * network is the failure mode this guard exists for.
 */

import { el } from "./dom.js";
import { count } from "./format.js";

export const TRAFFIC_CLASSES = [
  ["voice", "Voice"], ["video", "Video"], ["vpn", "Enterprise VPN"],
  ["besteffort", "Consumer internet"], ["bulk", "Bulk data"],
  ["critical", "Critical services"],
];

export const CONDITIONS = [
  ["congested", "Congested"], ["sla_risk", "SLA risk"], ["failed", "Failed"],
  ["degraded", "Degraded"], ["recovering", "Recovering"], ["changed", "Changed since previous"],
];

export function matchesDemand(demand, filters, previousDemand) {
  if (filters.classes.length && !filters.classes.includes(demand.class)) return false;
  if (filters.search) {
    const needle = filters.search.toLowerCase();
    const haystack = `${demand.id} ${demand.src_city} ${demand.dst_city} ` +
                     `${demand.class_label} ${demand.current_path_label}`.toLowerCase();
    if (!haystack.includes(needle)) return false;
  }
  if (!filters.conditions.length) return true;
  return filters.conditions.some((condition) => {
    switch (condition) {
      case "congested": return demand.bottleneck_util >= 0.9;
      case "sla_risk": return !demand.sla_ok;
      case "failed": return demand.disconnected;
      case "degraded": return !demand.sla_ok || demand.disconnected;
      case "recovering": return demand.last_reroute_step >= 0 && demand.sla_ok;
      case "changed": return previousDemand
        && previousDemand.current_path_idx !== demand.current_path_idx;
      default: return true;
    }
  });
}

export function renderFilterBar(state, { onToggleClass, onToggleCondition, onSearch,
                                         onClear, matched, total }) {
  const filters = state.filters;
  const active = filters.classes.length + filters.conditions.length
    + (filters.search ? 1 : 0);

  return el("div", { class: "filters", role: "group", "aria-label": "Filters" }, [
    el("div", { class: "filters__group" }, [
      el("span", { class: "filters__legend", text: "Traffic class" }),
      ...TRAFFIC_CLASSES.map(([id, label]) => el("button", {
        type: "button", class: "chip",
        "aria-pressed": filters.classes.includes(id) ? "true" : "false",
        onClick: () => onToggleClass(id), text: label,
      })),
    ]),
    el("div", { class: "filters__group" }, [
      el("span", { class: "filters__legend", text: "Condition" }),
      ...CONDITIONS.map(([id, label]) => el("button", {
        type: "button", class: "chip",
        "aria-pressed": filters.conditions.includes(id) ? "true" : "false",
        onClick: () => onToggleCondition(id), text: label,
      })),
    ]),
    el("div", { class: "filters__group filters__group--search" }, [
      el("label", { class: "filters__legend", for: "network-search", text: "Search" }),
      el("input", {
        type: "search", id: "network-search", class: "field",
        placeholder: "City, demand, class or route",
        value: filters.search,
        onInput: (event) => onSearch(event.target.value),
      }),
    ]),
    el("p", { class: "filters__state", role: "status" }, [
      document.createTextNode(
        `${count(matched)} of ${count(total)} demands shown`),
      active
        ? el("button", { type: "button", class: "chip", onClick: onClear,
                         text: "Clear filters" })
        : null,
    ]),
  ]);
}
