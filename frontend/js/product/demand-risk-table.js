/* Demand and SLA-risk table.
 *
 * Ordered the way an operator would triage: disconnected protected demands
 * first, then disconnected unprotected, then SLA violations by priority, then
 * demands crossing a congested link, then everything else by offered traffic.
 *
 * Selecting a row selects the same demand the map does. There is one selection.
 */

import { el } from "./dom.js";
import { count, mbps, ms, percent } from "./format.js";

export function renderDemandRiskTable(demands, { selection, onSelect, previous }) {
  const rows = [...demands].sort((a, b) =>
    a.risk_rank - b.risk_rank
    || b.priority - a.priority
    || b.offered_mbps - a.offered_mbps);

  if (!rows.length) {
    return el("p", { class: "tb-empty",
      text: "No demand matches the active filters." });
  }

  const priorById = new Map((previous || []).map((d) => [d.id, d]));

  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid" }, [
      el("caption", {
        text: "Demands ordered by risk: disconnected protected classes first, then " +
              "disconnected, then SLA violations by priority, then demands crossing a " +
              "congested link.",
      }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Demand" }),
        el("th", { scope: "col", text: "ID" }),
        el("th", { scope: "col", text: "Class" }),
        el("th", { scope: "col", text: "Offered" }),
        el("th", { scope: "col", text: "Route" }),
        el("th", { scope: "col", text: "Tightest hop" }),
        el("th", { scope: "col", text: "Delay" }),
        el("th", { scope: "col", text: "Loss" }),
        el("th", { scope: "col", text: "State" }),
        el("th", { scope: "col", text: "Changed" }),
      ])]),
      el("tbody", {}, rows.map((demand) => {
        const prior = priorById.get(demand.id);
        const moved = prior && prior.current_path_idx !== demand.current_path_idx;
        return el("tr", {
          "aria-selected": selection.objectType === "demand"
            && selection.objectId === demand.id ? "true" : "false",
          tabindex: "0",
          onClick: () => onSelect(demand.id),
          onKeydown: (event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              onSelect(demand.id);
            }
          },
        }, [
          el("th", { scope: "row", text: `${demand.src_city} → ${demand.dst_city}` }),
          el("td", { text: demand.id }),
          el("td", { text: demand.class_label }),
          el("td", { text: mbps(demand.offered_mbps) }),
          el("td", { text: `p${demand.current_path_idx}` }),
          el("td", { text: percent(demand.bottleneck_util, 0) }),
          el("td", { text: ms(demand.delay_ms, 0) }),
          el("td", { text: `${demand.loss_pct.toFixed(2)}%` }),
          el("td", { text: demand.risk_label }),
          el("td", { text: moved ? `p${prior.current_path_idx} → p${demand.current_path_idx}` : "—" }),
        ]);
      })),
    ]),
  ]);
}

export function riskSummary(demands) {
  const disconnected = demands.filter((d) => d.disconnected);
  const violating = demands.filter((d) => !d.sla_ok && !d.disconnected);
  const pressured = demands.filter((d) => d.sla_ok && !d.disconnected
    && d.bottleneck_util >= 0.9);
  return el("dl", { class: "facts" }, [
    el("dt", { text: "Disconnected" }),
    el("dd", { text: count(disconnected.length) }),
    el("dt", { text: "Outside SLA" }),
    el("dd", { text: count(violating.length) }),
    el("dt", { text: "Crossing a congested link" }),
    el("dd", { text: count(pressured.length) }),
    el("dt", { text: "Grain" }),
    el("dd", { text: "Current affected demands. Cumulative demand-interval SLA " +
                     "violations are a separate metric." }),
  ]);
}
