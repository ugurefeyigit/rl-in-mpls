/* Honest deep-link landing for a stored completed-run interval.
 * Stored histories contain aggregate interval metrics and actions, not a full
 * per-link snapshot or observation vector, so the destination says exactly
 * what can and cannot be reconstructed.
 */

import { el, unavailable } from "./dom.js";
import { count, mbps, percent, signed } from "./format.js";

export function renderComparisonContext(state, depth) {
  const focus = state.comparisonFocus;
  const payload = state.data.comparativeRuns;
  const runs = payload?.slots ? Object.values(payload.slots).filter(Boolean) : [];
  const run = runs.find((candidate) => candidate.run_id === focus.runId);
  if (!run) return unavailable("Stored run unavailable",
    "This A/B slot was cleared or the server restarted. Return to Comparative Run Results and choose it again.");
  const row = run.history.find((record) => record.step === focus.step);
  if (!row) return unavailable("Stored interval unavailable",
    `Step ${focus.step ?? "—"} is not present in this completed run.`);
  const metrics = row.metrics || {};
  const decision = row.decision || {};
  const decoded = decision.decoded || {};
  const selected = state.selection.objectType && state.selection.objectId
    ? `${state.selection.objectType} · ${state.selection.objectId}` : "No object selected";
  return el("section", { class: "panel comparison-context",
    "aria-labelledby": "comparison-context-title" }, [
    el("h2", { class: "panel__title", id: "comparison-context-title",
      text: `Stored ${depth === "network" ? "network" : "decision"} interval` }),
    el("p", { class: "cr-picker__truth",
      text: "COMPLETED LIVE DEMONSTRATION · aggregate stored interval · not governed evidence" }),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Run" }), el("dd", { text: run.label }),
      el("dt", { text: "Step / time" }), el("dd", { text: `${count(row.step)} · ${row.t_min} min` }),
      el("dt", { text: "Selected object" }), el("dd", { text: selected }),
      el("dt", { text: "Interval action" }), el("dd", { text: actionText(decision, decoded) }),
      el("dt", { text: "Interval reward" }), el("dd", { text: signed(row.reward, 4) }),
      el("dt", { text: "Delivery" }), el("dd", { text: percent(metrics.delivered_ratio, 1) }),
      el("dt", { text: "Peak utilization" }), el("dd", { text: percent(metrics.max_util, 1) }),
      el("dt", { text: "SLA risk" }), el("dd", { text: `${count(metrics.sla_violations)} violating demands` }),
      el("dt", { text: "Moved bandwidth" }), el("dd", { text: row.moved_mbps === null || row.moved_mbps === undefined
        ? "Unavailable · not recorded" : mbps(row.moved_mbps) }),
    ]),
    el("p", { class: "cmp__proof", text: depth === "network"
      ? "No per-link completed-run snapshot was stored, so the topology above is a fixed reference and carries no interval utilization coloring."
      : "The stored action and genuine reward components are available; observations, masks and policy outputs were not archived for this interval." }),
    el("a", { class: "ctl", href: `/compare?step=${row.step}`,
      text: "Back to Comparative Run Results" }),
  ]);
}

function actionText(decision, decoded) {
  if (decision.action === 0) return "No TE change";
  if (decision.action !== undefined) {
    return decoded.demand ? `Action ${decision.action} · ${decoded.demand} to path ${decoded.path_idx}`
      : `Action ${decision.action}`;
  }
  const moves = (decision.moves || []).filter((move) => move.accepted);
  return moves.length ? `${moves.length} accepted TE change(s)` : "No TE change";
}
