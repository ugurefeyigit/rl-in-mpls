/* Router, link, demand and path inspection.
 *
 * The inspector shows what the simulator models and names what it does not.
 * There is no CPU gauge, no label table and no BGP session count, because the
 * engine has none of those and inventing them would make the whole surface
 * untrustworthy.
 *
 * V1 and V2 fields carry their version label and are never merged.
 */

import { el, tag, unavailable } from "./dom.js";
import { count, mbps, metricDelta, ms, num, percent } from "./format.js";

export function renderInspector(state) {
  const { objectType, objectId } = state.selection;
  const snapshot = state.data.snapshot;

  if (!snapshot) {
    return unavailable("Inspector", "No network state is loaded.");
  }
  if (!objectType || !objectId) {
    return el("p", { class: "prose",
      text: "Select a city, a link or a demand — on the map, in the list or in the " +
            "table — to inspect it here." });
  }
  if (objectType === "link") return linkInspector(state, snapshot, objectId);
  if (objectType === "demand") return demandInspector(state, snapshot, objectId);
  return routerInspector(state, snapshot, objectId);
}

function head(title, subtitle, badges = []) {
  return el("header", { class: "insp__head" }, [
    el("h3", { class: "insp__title", text: title }),
    el("p", { class: "insp__sub", text: subtitle }),
    el("div", { class: "insp__badges" }, badges),
  ]);
}

function routerInspector(state, snapshot, routerId) {
  const node = (snapshot.nodes || []).find((n) => n.id === routerId);
  if (!node) return unavailable("Router", `No router ${routerId} in this snapshot.`);
  const links = (snapshot.links || []).filter((l) => l.a === routerId || l.z === routerId);
  const demands = (snapshot.demands || []).filter((d) => d.current_path.includes(routerId));
  const affected = demands.filter((d) => !d.sla_ok || d.disconnected);

  return el("div", { class: "insp" }, [
    head(node.city, `${node.role_label} · ${node.id}`,
      [node.has_failed_link ? tag("Adjacent failure", "failure") : tag("Up", "normal")]),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Neighbours" }),
      el("dd", { text: node.neighbors.map((n) => cityOf(snapshot, n)).join(", ") }),
      el("dt", { text: "Adjacent links" }),
      el("dd", { text: count(links.length) }),
      el("dt", { text: "Traversing LSPs" }),
      el("dd", { text: count(demands.length) }),
      el("dt", { text: "Affected demands now" }),
      el("dd", { text: count(affected.length) }),
      el("dt", { text: "Busiest adjacent link" }),
      el("dd", { text: percent(node.worst_adjacent_utilization, 1) }),
    ]),
    el("p", { class: "insp__note",
      text: "Not modeled: CPU, memory, label table, BGP, RSVP-TE and interface " +
            "counters. This simulator works at flow level." }),
  ]);
}

function linkInspector(state, snapshot, linkId) {
  const link = (snapshot.links || []).find((l) => l.id === linkId);
  if (!link) return unavailable("Link", `No link ${linkId} in this snapshot.`);
  const previous = state.data.previousSnapshot?.links?.find((l) => l.id === linkId);
  const crossing = (snapshot.demands || []).filter((d) => crosses(d, link));

  return el("div", { class: "insp" }, [
    head(`${link.a_city} – ${link.z_city}`, link.technical, [
      link.up ? tag(link.band_label, link.state === "congested" ? "pressure" : "normal")
              : tag("Failed", "failure"),
      tag(`${link.capacity_class}`, "comparison"),
    ]),

    el("table", { class: "grid" }, [
      el("caption", { text: "Both directions. The map summarizes with the busier one." }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Direction" }),
        el("th", { scope: "col", text: "Load" }),
        el("th", { scope: "col", text: "Utilization" }),
        el("th", { scope: "col", text: "Available" }),
        el("th", { scope: "col", text: "Queue delay" }),
        el("th", { scope: "col", text: "Loss" }),
        el("th", { scope: "col", text: "LSPs" }),
      ])]),
      el("tbody", {}, link.directions.map((d) => el("tr", {}, [
        el("th", { scope: "row", text: `${d.src_city} → ${d.dst_city}` }),
        el("td", { text: mbps(d.load_mbps) }),
        el("td", { text: percent(d.utilization, 1) }),
        el("td", { text: mbps(d.available_mbps) }),
        el("td", { text: ms(d.queue_delay_ms) }),
        el("td", { text: percent(d.loss_fraction, 3) }),
        el("td", { text: count(d.n_lsps) }),
      ]))),
    ]),

    el("dl", { class: "facts" }, [
      el("dt", { text: "Capacity per direction" }),
      el("dd", { text: mbps(link.capacity_mbps) }),
      el("dt", { text: "Propagation delay" }),
      el("dd", { text: ms(link.prop_delay_ms) }),
      el("dt", { text: "Admin weight" }),
      el("dd", { text: num(link.weight, 0) }),
      el("dt", { text: "Since previous step" }),
      el("dd", { text: previous
        ? `${metricDelta("share", link.worst_utilization - previous.worst_utilization)} on the busier direction`
        : "No comparable previous step in this generation." }),
      el("dt", { text: "Demands crossing" }),
      el("dd", { text: crossing.map((d) => d.id).join(", ") || "None" }),
    ]),

    el("p", { class: "insp__note",
      text: "Queue delay and loss are analytic functions of utilization, not packet " +
            "measurements. " + link.worst_direction_rule }),
  ]);
}

function demandInspector(state, snapshot, demandId) {
  const demand = (snapshot.demands || []).find((d) => d.id === demandId);
  if (!demand) return unavailable("Demand", `No demand ${demandId} in this snapshot.`);

  return el("div", { class: "insp" }, [
    head(`${demand.src_city} → ${demand.dst_city}`,
      `${demand.class_label} · ${demand.id} · priority ${demand.priority}` +
      `${demand.protected ? " · protected" : ""}`,
      [tag(demand.risk_label, riskState(demand))]),

    el("dl", { class: "facts" }, [
      el("dt", { text: "Base traffic" }),
      el("dd", { text: mbps(demand.base_mbps) }),
      el("dt", { text: "Offered now" }),
      el("dd", { text: mbps(demand.offered_mbps) }),
      el("dt", { text: "Carried now" }),
      el("dd", { text: mbps(demand.carried_mbps) }),
      el("dt", { text: "Current route" }),
      el("dd", { text: `p${demand.current_path_idx} · ${demand.current_path_label}` }),
      el("dt", { text: "Measured delay" }),
      el("dd", { text: `${ms(demand.delay_ms)} against a ${ms(demand.sla_max_latency_ms, 0)} limit` }),
      el("dt", { text: "Measured loss" }),
      el("dd", { text: `${num(demand.loss_pct, 3)}% against a ${num(demand.sla_max_loss_pct, 2)}% limit` }),
      el("dt", { text: "Tightest hop" }),
      el("dd", { text: percent(demand.bottleneck_util, 1) }),
      el("dt", { text: "Path changes" }),
      el("dd", { text: count(demand.path_changes) }),
      el("dt", { text: "Last reroute" }),
      el("dd", { text: demand.last_reroute_step >= 0
        ? `step ${count(demand.last_reroute_step)}` : "Never in this run" }),
      el("dt", { text: `${demand.cooldown_label} (V1)` }),
      el("dd", { text: demand.cooldown_until_step > (state.context.step ?? 0)
        ? `until step ${count(demand.cooldown_until_step)}`
        : "Not cooling down" }),
    ]),

    el("table", { class: "grid" }, [
      el("caption", { text: "Candidate routes" }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Route" }),
        el("th", { scope: "col", text: "Hops" }),
        el("th", { scope: "col", text: "Live" }),
        el("th", { scope: "col", text: "Tightest hop" }),
        el("th", { scope: "col", text: "If moved here" }),
        el("th", { scope: "col", text: "Headroom" }),
      ])]),
      el("tbody", {}, demand.candidates.map((c) => el("tr", {
        "aria-selected": c.is_current ? "true" : "false",
      }, [
        el("th", { scope: "row", text: `p${c.path_idx}${c.is_current ? " · current" : ""}` }),
        el("td", { text: count(c.hops) }),
        el("td", { text: c.available ? "yes" : "no" }),
        el("td", { text: percent(c.bottleneck_util, 1) }),
        el("td", { text: percent(c.projected_bottleneck_util, 1) }),
        el("td", { text: mbps(c.available_bandwidth_mbps) }),
      ]))),
    ]),

    el("p", { class: "insp__note",
      text: `Routes: ${demand.candidates.map((c) => `p${c.path_idx} ${c.path_label}`).join(" · ")}` }),
  ]);
}

function cityOf(snapshot, routerId) {
  return (snapshot.nodes || []).find((n) => n.id === routerId)?.city || routerId;
}

function riskState(demand) {
  if (demand.risk_rank <= 1) return "failure";
  if (demand.risk_rank === 2) return "pressure";
  if (demand.risk_rank === 3) return "pressure";
  return "normal";
}

function crosses(demand, link) {
  const path = demand.current_path;
  for (let i = 0; i < path.length - 1; i += 1) {
    if ((path[i] === link.a && path[i + 1] === link.z)
      || (path[i] === link.z && path[i + 1] === link.a)) return true;
  }
  return false;
}

export { crosses };
