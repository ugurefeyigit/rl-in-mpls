/* Exp 2.1 A/B controls and compact completed-run summary.
 *
 * A and B are identities, never winner/loser colors. Outcome words and
 * semantic classes come from backend metric direction, so higher delivery and
 * return are favorable while higher risk, utilization and churn are not.
 */

import { el, unavailable } from "./dom.js";
import { count, mbps, percent, points, shortHash, signed } from "./format.js";

const SUMMARY_IDS = new Set([
  "operational_return", "delivery", "sla_risk", "peak_utilization",
  "reroutes", "flaps", "moved_bandwidth",
]);

export function renderComparisonPicker(state, handlers, { compact = false } = {}) {
  const payload = state.data.comparativeRuns;
  if (!payload) {
    return unavailable("Completed-run comparison", "The temporary A/B store has not loaded.");
  }
  const { a, b } = payload.slots;
  return el("div", { class: `cr-picker${compact ? " cr-picker--compact" : ""}` }, [
    el("p", { class: "cr-picker__truth",
      text: "COMPLETED LIVE DEMONSTRATION · process memory only · not governed evidence" }),
    el("div", { class: "cr-picker__slots" }, [
      slotControl("a", "Run A", a, payload.candidates, handlers),
      slotControl("b", "Run B", b, payload.candidates, handlers),
    ]),
    el("div", { class: "cr-picker__actions", role: "group",
                "aria-label": "Completed-run comparison actions" }, [
      button("Swap A/B", handlers.onSwap, !a && !b),
      button("Clear A", () => handlers.onClear("a"), !a),
      button("Clear B", () => handlers.onClear("b"), !b),
      button("Clear All", handlers.onClearAll, !a && !b),
      ...(compact ? [el("a", { class: "ctl cr-picker__full", href: "/compare",
        text: "View Full Results" })] : []),
    ]),
    identityPair(a, b),
    synchronization(payload.pairing),
    summaryTable(payload.headline, a, b),
    el("p", { class: "cmp__proof", text: payload.lifetime }),
  ]);
}

function slotControl(slot, label, selected, candidates, handlers) {
  const options = [el("option", { value: "", text: "Choose a completed run" })];
  for (const candidate of candidates) {
    options.push(el("option", {
      value: candidate.run_id, selected: selected?.run_id === candidate.run_id,
      text: candidate.label,
    }));
  }
  return el("label", { class: `cr-slot cr-slot--${slot}` }, [
    el("span", { class: "cr-slot__token", text: slot.toUpperCase() }),
    el("span", { class: "cr-slot__label", text: label }),
    el("select", {
      class: "field", "aria-label": `${label} completed run`,
      value: selected?.run_id || "",
      onchange: (event) => event.target.value
        ? handlers.onAssign(slot, event.target.value) : handlers.onClear(slot),
    }, options),
  ]);
}

function button(text, onclick, disabled) {
  return el("button", { type: "button", class: "ctl", text, onclick, disabled });
}

function identityPair(a, b) {
  if (!a && !b) {
    return unavailable("No completed runs selected",
      "Finish runs, then choose them for A and B. Unfinished runs are never offered here.");
  }
  return el("div", { class: "cr-identities" }, [identity("a", a), identity("b", b)]);
}

function identity(slot, run) {
  if (!run) return el("section", { class: `cr-identity cr-identity--${slot}` }, [
    el("h3", { text: `${slot.toUpperCase()} · not selected` }),
  ]);
  const id = run.identity;
  return el("section", { class: `cr-identity cr-identity--${slot}` }, [
    el("h3", { text: `${slot.toUpperCase()} · ${id.controller}` }),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Scenario / seed" }),
      el("dd", { text: `${id.scenario} · ${id.seed}` }),
      el("dt", { text: "Environment / root" }),
      el("dd", { text: `${String(id.environment).toUpperCase()} · ${id.training_root ?? "not applicable"}` }),
      el("dt", { text: "Checkpoint" }),
      el("dd", { text: id.checkpoint_id || "No checkpoint · rule-based baseline" }),
      el("dt", { text: "Checkpoint hash" }),
      el("dd", { text: shortHash(id.checkpoint_sha256) }),
    ]),
  ]);
}

function synchronization(pairing) {
  const state = pairing.synchronized ? "normal" : (pairing.available ? "pressure" : "neutral");
  return el("p", { class: "cr-sync", dataset: { state },
                   role: "status", "aria-live": "polite",
                   text: `Synchronization · ${pairing.synchronized ? "synchronized" : "not synchronized"}. ${pairing.reason}` });
}

function summaryTable(rows, a, b) {
  const selected = rows.filter((row) => SUMMARY_IDS.has(row.id));
  if (!a || !b || !selected.length) {
    return unavailable("A/B summary", "Choose both completed runs to calculate absolute deltas.");
  }
  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid cr-summary" }, [
      el("caption", { text: "Completed-run summary. Deltas are A minus B in the printed unit." }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Measure" }),
        el("th", { scope: "col", text: "A" }),
        el("th", { scope: "col", text: "B" }),
        el("th", { scope: "col", text: "A − B" }),
        el("th", { scope: "col", text: "Outcome" }),
      ])]),
      el("tbody", {}, selected.map((row) => el("tr", {}, [
        el("th", { scope: "row", text: row.label }),
        metricCell(row, "a"), metricCell(row, "b"),
        el("td", { text: deltaText(row) }),
        el("td", { class: `outcome outcome--${row.a_outcome}`,
          text: outcomeText(row) }),
      ]))),
    ]),
  ]);
}

function metricCell(row, lane) {
  return el("td", { text: valueText(row.unit, row[lane]) });
}

function valueText(unit, value) {
  if (unit === "percent") return percent(value, 1);
  if (unit === "Mbps") return mbps(value);
  if (["changes", "flaps", "reversals", "violating demands"].includes(unit)) return count(value);
  return signed(value, 3);
}

function deltaText(row) {
  if (row.delta === null) return "Unavailable";
  if (row.unit === "percent") return points(row.delta, 1);
  if (row.unit === "Mbps") return `${signed(row.delta, 0)} Mbps`;
  if (["changes", "flaps", "reversals", "violating demands"].includes(row.unit)) {
    return `${signed(row.delta, 0)} ${row.unit}`;
  }
  return `${signed(row.delta, 3)} ${row.unit}`;
}

function outcomeText(row) {
  if (!row.leader) return "Unavailable";
  if (row.leader === "equal") return "Equal";
  return `${row.leader.toUpperCase()} better · ${row.direction} is favorable`;
}
