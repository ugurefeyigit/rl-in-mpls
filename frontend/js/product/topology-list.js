/* The accessible twin of the topology.
 *
 * Map selection and list selection are one state, not two views that happen to
 * agree. A screen-reader user is never asked to traverse SVG internals, and the
 * list is a first-class way to work — not a fallback with less information.
 */

import { $, el, fill } from "./dom.js";
import { mbps, percent } from "./format.js";

export function renderTopologyList(snapshot, { selection, onSelect, showTelemetry }) {
  const body = $("topology-list-body");
  if (!snapshot) {
    fill(body, el("p", { class: "tb-empty", text: "No network state is loaded yet." }));
    return;
  }

  const selectedRouter = selection.objectType === "router" ? selection.objectId : null;
  const groups = [];

  groups.push(group("Cities", (snapshot.nodes || []).map((node) => item({
    selected: node.id === selectedRouter,
    name: `${node.city} · ${node.role_label}`,
    id: node.id,
    value: showTelemetry
      ? `${node.n_links} links · ${node.n_lsps} LSPs · busiest ${percent(node.worst_adjacent_utilization, 0)}`
      : `${node.n_links} links`,
    onSelect: () => onSelect("router", node.id),
  }))));

  const links = selectedRouter
    ? (snapshot.links || []).filter((l) => l.a === selectedRouter || l.z === selectedRouter)
    : (snapshot.links || []);
  groups.push(group(
    selectedRouter ? `Links at ${cityOf(snapshot, selectedRouter)}` : "Links",
    links.map((link) => item({
      selected: selection.objectType === "link" && link.id === selection.objectId,
      name: `${link.a_city} – ${link.z_city}`,
      id: link.id,
      value: showTelemetry
        ? `${link.up ? statusWord(link) : "failed"} · ${percent(link.worst_utilization, 0)} · ${mbps(link.capacity_mbps)}`
        : mbps(link.capacity_mbps),
      onSelect: () => onSelect("link", link.id),
    }))));

  const demands = selectedRouter
    ? (snapshot.demands || []).filter((d) => d.current_path.includes(selectedRouter))
    : (snapshot.demands || []);
  groups.push(group(
    selectedRouter ? `Demands through ${cityOf(snapshot, selectedRouter)}` : "Demands",
    demands.map((demand) => item({
      selected: selection.objectType === "demand" && demand.id === selection.objectId,
      name: `${demand.src_city} → ${demand.dst_city} ${demand.class_label}`,
      id: demand.id,
      value: `${demand.risk_label} · ${mbps(demand.offered_mbps)}`,
      onSelect: () => onSelect("demand", demand.id),
    }))));

  fill(body, groups);
}

function cityOf(snapshot, routerId) {
  return (snapshot.nodes || []).find((n) => n.id === routerId)?.city || routerId;
}

function statusWord(link) {
  if (link.state === "congested") return "congested";
  return "up";
}

function group(title, items) {
  return el("section", { class: "topo-group" }, [
    el("h3", { text: title }),
    el("ul", { class: "topo-list" },
      items.length ? items : [el("li", { class: "tb-empty", text: "Nothing matches." })]),
  ]);
}

function item({ selected, name, id, value, onSelect }) {
  return el("li", {}, [
    el("button", {
      type: "button",
      class: "topo-item",
      "aria-selected": selected ? "true" : "false",
      onClick: onSelect,
    }, [
      el("span", { class: "topo-item__name", text: name }),
      el("span", { class: "topo-item__id", text: id }),
      el("span", { class: "topo-item__value", text: value }),
    ]),
  ]);
}
