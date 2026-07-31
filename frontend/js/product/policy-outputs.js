/* Per-action policy outputs.
 *
 * There is one renderer, and it is driven by the controller's *declared* output
 * semantics rather than by the shape of the numbers. A probabilities policy gets
 * percentages that sum over the valid masked distribution; a scores policy gets
 * signed unnormalized values, no percent sign, and the words "action score" or
 * "immediate-reward estimate".
 *
 * Entropy and value appear only when the backend exposes them. When it does not,
 * the row states the reason instead of showing a plausible zero.
 */

import { el, unavailable } from "./dom.js";
import { count, policyValue } from "./format.js";

export function renderPolicyOutputs(state) {
  const decision = state.data.decision;
  if (!decision) return unavailable("Policy output", "No decision payload has been read.");
  const output = decision.policy_output;

  if (!output.available) {
    return unavailable(output.label, output.reason);
  }

  const semantics = output.semantics;
  const max = Math.max(...output.top.map((r) => Math.abs(r.prob)), 1e-9);

  return el("div", { class: "pol" }, [
    el("p", { class: "pol__semantics", text: output.description }),

    el("ul", { class: "pol__bars", "aria-label": output.label },
      output.top.map((row) => el("li", { class: "pol__bar" }, [
        el("span", { class: "pol__action", text: `${row.action}` }),
        el("span", { class: "pol__desc", text: row.desc }),
        el("span", { class: "pol__track", "aria-hidden": "true" }, [
          el("span", {
            class: "pol__fill",
            style: `width:${Math.min(100, (Math.abs(row.prob) / max) * 100).toFixed(1)}%`,
          }),
        ]),
        el("span", { class: "pol__value", text: policyValue(semantics, row.prob) }),
      ]))),

    el("dl", { class: "facts" }, [
      el("dt", { text: `Selected · ${output.label.toLowerCase()}` }),
      el("dd", { text: policyValue(semantics, output.selected?.value) }),
      el("dt", { text: `Runner-up · ${output.label.toLowerCase()}` }),
      el("dd", { text: output.runner_up
        ? `action ${output.runner_up.action} · ${policyValue(semantics, output.runner_up.value)}`
        : "None reported" }),
      el("dt", { text: `No-op · ${output.label.toLowerCase()}` }),
      el("dd", { text: output.noop?.value === null || output.noop?.value === undefined
        ? (output.noop?.reason || "Not reported")
        : policyValue(semantics, output.noop.value) }),
      el("dt", { text: "Entropy" }),
      el("dd", { text: output.entropy === null
        ? output.entropy_reason : String(output.entropy) }),
      el("dt", { text: "Value estimate" }),
      el("dd", { text: output.value === null
        ? output.value_reason : String(output.value) }),
    ]),

    el("p", { class: "pol__note", text: output.distribution_note || "" }),
  ]);
}

export function counterfactualPanel(state, { onRequest }) {
  const result = state.data.counterfactual;
  const decision = state.data.decision;
  const supported = decision?.counterfactual?.available;

  if (!supported) {
    return unavailable("Counterfactual",
      decision?.counterfactual?.reason
      || "Counterfactual unavailable for this source.");
  }

  return el("div", { class: "cf" }, [
    el("p", { class: "prose",
      text: "A counterfactual runs the selected action and no-op on deep copies of " +
            "the current state. The live session is fingerprinted before and after " +
            "and must be unchanged, or no result is reported." }),
    el("button", { type: "button", class: "chip", onClick: onRequest,
                   text: "Estimate the selected action against no-op" }),
    result ? counterfactualResult(result) : null,
  ]);
}

function counterfactualResult(result) {
  if (result.kind !== "simulated_estimate") {
    return unavailable("Counterfactual", result.reason || "No estimate was produced.");
  }
  const keys = result.metrics_reported || [];
  return el("div", {}, [
    el("p", { class: "cf__label", text: result.label }),
    el("div", { class: "table-scroll" }, [
      el("table", { class: "grid" }, [
        el("caption", { text: `Cloned from step ${count(result.step)}. Session ` +
          `fingerprint ${result.session_unchanged ? "unchanged" : "CHANGED"}.` }),
        el("thead", {}, [el("tr", {}, [
          el("th", { scope: "col", text: "Measure" }),
          el("th", { scope: "col", text: "No-op" }),
          el("th", { scope: "col", text: `Action ${result.action}` }),
          el("th", { scope: "col", text: "Δ" }),
        ])]),
        el("tbody", {}, keys.map((key) => el("tr", {}, [
          el("th", { scope: "row", text: key }),
          el("td", { text: fmt(result.noop?.[key]) }),
          el("td", { text: fmt(result.action_metrics?.[key]) }),
          el("td", { text: fmt(result.delta?.[key]) }),
        ]))),
      ]),
    ]),
    result.action_metrics ? null
      : el("p", { class: "cf__label", text: result.action_reason }),
  ]);
}

function fmt(value) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(4);
}
