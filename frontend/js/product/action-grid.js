/* The complete 69-action space.
 *
 * No-op is separated from the 17 demand groups of four candidate paths, because
 * "hold position" is a different kind of decision from "move D13 to path 2".
 *
 * Every state is exhaustive and named: chosen and valid, valid runner-up, valid
 * no-op, invalid with the validator's own reason, or unavailable because the
 * source carries no mask detail. A boolean is never turned into a reason here.
 */

import { el, tag, unavailable } from "./dom.js";
import { count, policyValue } from "./format.js";

export function renderActionGrid(state, { onSelectAction, showInvalid, onToggleInvalid }) {
  const decision = state.data.decision;
  if (!decision) return unavailable("Action space", "No decision payload has been read.");
  const mask = decision.mask;
  if (!mask.available) return unavailable("Action space", mask.reason);

  const output = decision.policy_output;
  const semantics = output.semantics;
  const valueFor = new Map();
  if (output.available) {
    for (const row of output.top || []) valueFor.set(row.action, row.prob);
  }

  const runnerUp = output.available ? output.runner_up?.action : null;
  const noop = mask.actions[0];
  const groups = new Map();
  for (const row of mask.actions.slice(1)) {
    if (!groups.has(row.demand_id)) groups.set(row.demand_id, []);
    groups.get(row.demand_id).push(row);
  }

  const invalidCount = mask.actions.filter((a) => !a.valid).length;

  return el("div", { class: "acts" }, [
    el("div", { class: "acts__head" }, [
      el("p", { class: "filters__state",
        text: `${count(mask.valid_count)} of ${count(mask.count)} actions are legal ` +
              `this interval. Rejection reasons come from ${mask.reason_source}.` }),
      el("button", {
        type: "button", class: "chip",
        "aria-pressed": showInvalid ? "true" : "false",
        onClick: onToggleInvalid,
        text: `Show ${count(invalidCount)} masked actions`,
      }),
    ]),

    el("section", { class: "acts__noop" }, [
      el("h3", { class: "panel__title", text: "Action 0 · no TE change" }),
      el("div", { class: "acts__row" }, [
        actionButton(noop, {
          state, semantics, valueFor, runnerUp, onSelectAction, alwaysShow: true,
        }),
        el("p", { class: "acts__note",
          text: "No-op is always legal. Holding position is a decision, not indecision." }),
      ]),
    ]),

    el("div", { class: "acts__grid" }, [...groups.entries()].map(([demandId, rows]) => {
      const visible = showInvalid ? rows : rows.filter((r) => r.valid || r.selected);
      return el("section", { class: "acts__demand" }, [
        el("h3", { class: "acts__demand-title" }, [
          el("span", { text: rows[0].label }),
          el("span", { class: "acts__demand-id", text: demandId }),
        ]),
        visible.length
          ? el("div", { class: "acts__row" }, visible.map((row) => actionButton(row, {
              state, semantics, valueFor, runnerUp, onSelectAction,
            })))
          : el("p", { class: "acts__note", text: "Every candidate is masked this interval." }),
      ]);
    })),
  ]);
}

function actionButton(row, { semantics, valueFor, runnerUp, onSelectAction }) {
  const value = valueFor.get(row.action);
  const status = row.selected ? "chosen"
    : (row.action === runnerUp ? "runner-up"
      : (row.valid ? "valid" : "invalid"));

  return el("button", {
    type: "button",
    class: "act",
    dataset: { status },
    "aria-pressed": row.selected ? "true" : "false",
    title: row.valid ? (row.path_label || "No TE change") : row.reason,
    onClick: () => onSelectAction(row),
  }, [
    el("span", { class: "act__id", text: String(row.action) }),
    el("span", { class: "act__path",
      text: row.type === "noop" ? "no TE change" : `p${row.path_idx}` }),
    // Only a real distribution gets a numeric bar; an invalid action gets none.
    row.valid && value !== undefined
      ? el("span", { class: "act__value", text: policyValue(semantics, value) })
      : el("span", { class: "act__value act__value--none",
          text: row.valid ? "" : "masked" }),
    row.is_current_path && row.type !== "noop"
      ? el("span", { class: "act__flag", text: "current" }) : null,
  ]);
}

export function selectedActionPanel(state) {
  const decision = state.data.decision;
  const selected = decision?.selected_action;
  if (!selected?.available) {
    return unavailable("Selected action", selected?.reason
      || "No action has been taken in this generation.");
  }
  if (selected.kind === "baseline_moves") {
    return el("div", { class: "insp" }, [
      el("p", { class: "prose",
        text: `${selected.policy_id} proposes zero or more moves per interval rather ` +
              `than one action from the 69-action space.` }),
      el("dl", { class: "facts" }, [
        el("dt", { text: "Moves this interval" }),
        el("dd", { text: count(selected.n_moves) }),
        el("dt", { text: "Detail" }),
        el("dd", { text: selected.moves.map((m) =>
          `${m.demand} → p${m.path_idx} (${m.accepted ? "accepted" : m.reason})`)
          .join("; ") || "None" }),
      ]),
    ]);
  }

  const decoded = selected.decoded || {};
  const safety = decision.safety;
  return el("div", { class: "insp" }, [
    el("header", { class: "insp__head" }, [
      el("h3", { class: "insp__title",
        text: selected.is_noop ? "Action 0 · no TE change"
          : `Action ${selected.action} · ${decoded.demand} → path ${decoded.path_idx}` }),
      el("div", { class: "insp__badges" }, [
        selected.accepted === false ? tag("Rejected", "failure")
          : tag(selected.is_noop ? "No change" : "Applied", "normal"),
        safety.safety_filter ? tag("Safety filter on", "comparison") : null,
      ]),
    ]),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Source" }),
      el("dd", { text: selected.policy_id }),
      el("dt", { text: "From path" }),
      el("dd", { text: decoded.from_path === undefined ? "—" : `p${decoded.from_path}` }),
      el("dt", { text: "Validator" }),
      el("dd", { text: `${safety.validator} · ${selected.validator_reason || "ok"}` }),
      el("dt", { text: "Rejection" }),
      el("dd", { text: safety.environment_rejection ? "Environment rejection"
        : (safety.operator_rejection ? "Operator rejection" : "None") }),
      el("dt", { text: "Legal actions" }),
      el("dd", { text: count(selected.valid_action_count) }),
    ]),
    el("p", { class: "insp__note", text: selected.explanation || "" }),
    el("p", { class: "insp__note", text: selected.explanation_note || "" }),
  ]);
}
