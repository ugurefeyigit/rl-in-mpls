/* The recommendation card.
 *
 * It renders only for a real policy output, directly beneath the topology, and
 * it keeps four things that are easy to blur strictly apart:
 *
 *   Before    what was measured before the move
 *   Expected  a clone-based estimate, marked SIMULATED ESTIMATE, or unavailable
 *   Observed  what was measured after the move actually ran
 *   Reward    the actual interval reward, pending until the step completes
 *
 * The headline names the policy. It never says "AI advisor", and it never
 * describes the model as thinking, wanting or knowing anything.
 */

import { $, el, fill, icon, tag, unavailable } from "./dom.js";
import { mbps, percent, points, policyValue, signed } from "./format.js";

export function renderRecommendation(state) {
  const host = $("recommendation");
  const proposal = state.data.recommendation;

  if (!proposal) { host.hidden = true; fill(host, []); return; }
  host.hidden = false;

  const policy = policyLabel(state);
  const decoded = proposal.decoded;
  const semantics = outputSemantics(state);

  const headline = decoded
    ? `${policy} suggests moving ${decoded.demandLabel}`
    : `${policy} suggests no TE change`;

  const rows = [];
  if (decoded) {
    rows.push(["Demand", `${decoded.demand} · ${decoded.classLabel} · ${mbps(decoded.volumeMbps)}`]);
    rows.push(["Old path", decoded.fromLabel]);
    rows.push(["Proposed", decoded.toLabel]);
  } else {
    rows.push(["Action", "0 · no TE change"]);
  }
  rows.push(["Grounding", proposal.grounding]);
  rows.push(["Before", proposal.before ? telemetryLine(proposal.before) : "—"]);

  const card = el("div", { class: "rec" }, [
    el("div", { class: "rec__head" }, [
      icon("recommendation"),
      el("h2", { class: "rec__headline", id: "rec-headline", text: headline }),
      proposal.safetyOk
        ? tag("Valid", "normal")
        : tag("Invalid", "failure"),
      proposal.pending ? tag("Preview only", "selection") : null,
    ]),

    el("dl", { class: "rec__facts" }, rows.flatMap(([term, value]) => [
      el("dt", { text: term }),
      el("dd", { text: String(value) }),
    ])),

    el("div", { class: "rec__outcomes" }, [
      outcomeBlock("Expected", proposal.expected, {
        estimate: true,
        unavailableReason: proposal.expectedReason,
      }),
      outcomeBlock("Observed", proposal.observed, {
        unavailableReason: proposal.observed ? null
          : "Pending. The observed outcome appears after the interval runs.",
      }),
    ]),

    el("dl", { class: "rec__facts" }, [
      el("dt", { text: semantics.label }),
      el("dd", { text: policyValue(semantics.id, proposal.outputValue) }),
      el("dt", { text: "Safety" }),
      el("dd", { text: `${proposal.safetyOk ? "Valid" : "Invalid"} · ${proposal.safetyReason}` }),
      el("dt", { text: "Reward" }),
      el("dd", { text: proposal.reward === null || proposal.reward === undefined
        ? "Pending" : signed(proposal.reward, 4) }),
    ]),

    el("p", { class: "rec__note", text: semantics.description }),
  ]);

  fill(host, card);
}

function outcomeBlock(title, telemetry, { estimate = false, unavailableReason } = {}) {
  if (!telemetry) {
    return el("div", { class: "rec__outcome" }, [
      el("h3", { class: "panel__title", text: title }),
      unavailable(title, unavailableReason || "Outcome estimate unavailable"),
    ]);
  }
  return el("div", { class: "rec__outcome", dataset: estimate ? { estimate: "true" } : {} }, [
    el("h3", { class: "panel__title" }, [
      document.createTextNode(title),
      estimate ? el("span", { class: "tag", dataset: { state: "selection" },
                              text: "Simulated estimate" }) : null,
    ]),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Busiest link" }),
      el("dd", { text: percent(telemetry.max_util, 1) }),
      el("dt", { text: "Mean delay" }),
      el("dd", { text: `${(telemetry.mean_delay_ms ?? 0).toFixed(1)} ms` }),
      el("dt", { text: "Loss" }),
      el("dd", { text: percent(telemetry.loss_ratio, 3) }),
      el("dt", { text: "SLA violations" }),
      el("dd", { text: String(telemetry.sla_violations ?? "—") }),
    ]),
    estimate && telemetry.delta_max_util !== undefined
      ? el("p", { class: "rec__delta",
                  text: `Busiest link versus no-op: ${points(telemetry.delta_max_util)}` })
      : null,
  ]);
}

function telemetryLine(t) {
  return `busiest ${percent(t.max_util, 1)} · delay ${(t.mean_delay_ms ?? 0).toFixed(1)} ms · ` +
         `loss ${percent(t.loss_ratio, 3)} · ${t.sla_violations ?? "—"} SLA`;
}

function policyLabel(state) {
  const id = state.context.policyId;
  const policy = state.data.capabilities?.live_policies?.find(
    (p) => p.id === id && p.environment_version === state.context.environmentVersion);
  return policy?.label || id || "The selected policy";
}

function outputSemantics(state) {
  const id = state.context.policyId;
  const policy = state.data.capabilities?.live_policies?.find(
    (p) => p.id === id && p.environment_version === state.context.environmentVersion);
  return {
    id: policy?.output_semantics || "none",
    label: policy?.output_label || "Per-action output",
    description: policy?.output_description
      || "This controller exposes no per-action numbers.",
  };
}

/**
 * Build the card model from a live advisor proposal. Every field is read from
 * the payload; nothing is derived by guessing.
 */
export function proposalFromAdvisor(proposal, snapshot, { record = null } = {}) {
  if (!proposal) return null;
  const decoded = proposal.decoded ? {
    demand: proposal.decoded.demand,
    demandLabel: demandLabel(proposal.decoded, snapshot),
    classLabel: proposal.decoded.class,
    volumeMbps: proposal.decoded.volume_mbps,
    fromLabel: routerLabel(proposal.decoded.from_routers, snapshot),
    toLabel: routerLabel(proposal.decoded.to_routers, snapshot),
    fromRouters: proposal.decoded.from_routers,
    toRouters: proposal.decoded.to_routers,
  } : null;

  const lookahead = proposal.lookahead || {};
  const expected = lookahead.action || (proposal.is_noop ? lookahead.noop : null);
  return {
    action: proposal.action,
    pending: !record,
    decoded,
    grounding: groundingText(proposal, snapshot),
    before: beforeTelemetry(snapshot),
    expected: expected ? { ...expected, delta_max_util: lookahead.delta_max_util } : null,
    expectedReason: expected ? null
      : "Outcome estimate unavailable for this action.",
    observed: record?.actual || null,
    reward: record?.reward ?? null,
    outputValue: proposal.action_probability,
    safetyOk: proposal.safety_ok,
    safetyReason: proposal.safety_reason,
  };
}

function beforeTelemetry(snapshot) {
  const metrics = snapshot?.metrics;
  if (!metrics?.available) return null;
  return {
    max_util: metrics.values.max_util?.value,
    mean_delay_ms: metrics.values.mean_delay_ms?.value,
    loss_ratio: metrics.values.loss_ratio?.value,
    sla_violations: metrics.values.sla_violations?.value,
  };
}

function groundingText(proposal, snapshot) {
  if (proposal.is_noop) {
    return "No candidate move was both legal and better than holding position.";
  }
  const demand = (snapshot?.demands || []).find((d) => d.id === proposal.decoded?.demand);
  if (!demand) return "Measured path pressure and mask state only.";
  const target = (demand.candidates || [])
    .find((c) => c.path_idx === proposal.decoded.path_idx);
  return `Current tightest hop ${percent(demand.bottleneck_util, 0)}; ` +
         `projected tightest hop on the proposed path ` +
         `${percent(target?.projected_bottleneck_util, 0)}.`;
}

function demandLabel(decoded, snapshot) {
  const demand = (snapshot?.demands || []).find((d) => d.id === decoded.demand);
  return demand ? `${demand.src_city} → ${demand.dst_city} ${demand.class_label}`
                : decoded.demand;
}

function routerLabel(routers, snapshot) {
  const cities = new Map((snapshot?.nodes || []).map((n) => [n.id, n.city]));
  return (routers || []).map((id) => cities.get(id) || id).join(" → ");
}
