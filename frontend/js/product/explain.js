/* Explain this moment.
 *
 * Deterministic sentences assembled from fields that are present in the current
 * payload. There is no generation step and no hidden rationale: if a fact is not
 * in the data, the explanation says the fact is unavailable rather than filling
 * the sentence in.
 *
 * The three depths answer three different questions about the *same* moment.
 * Changing depth never changes the moment.
 */

import { $, el, fill, unavailable } from "./dom.js";
import { count, mbps, ms, num, percent, policyValue, signed } from "./format.js";

export function explain(state, depth) {
  const snapshot = state.data.snapshot;
  if (state.source.kind !== "live_session") return explainRecord(state, depth);
  if (!snapshot) {
    return [unavailable("Nothing to explain yet",
      "No session snapshot has arrived. Start a session to get a moment to explain.")];
  }
  if (depth === "network") return explainNetwork(state, snapshot);
  if (depth === "rl") return explainRl(state, snapshot);
  return explainPresentation(state, snapshot);
}

function paragraph(text) { return el("p", { text }); }

function section(title, children) {
  return el("section", { class: "panel" }, [
    el("h3", { class: "panel__title", text: title }),
    el("div", { class: "prose" }, children),
  ]);
}

function explainPresentation(state, snapshot) {
  const incident = snapshot.incident;
  const metrics = snapshot.metrics;
  const time = snapshot.time;
  const out = [];

  out.push(section("What is happening", [
    paragraph(`At ${time.clock} the network is in ${incident.label.toLowerCase()}.` +
      (incident.active_incident ? ` The active incident is ${incident.active_incident}.`
                                : " No incident is active.")),
    metrics.available
      ? paragraph(
        `The busiest link is at ${percent(metrics.values.max_util?.value, 0)}, ` +
        `${count(metrics.values.sla_violations?.value)} demand-interval SLA ` +
        `violation(s) were recorded this interval, and ` +
        `${percent(metrics.values.delivered_ratio?.value, 1)} of offered traffic ` +
        `was delivered.`)
      : unavailable("Telemetry", metrics.reason),
  ]));

  const decision = state.data.decision;
  if (decision?.selected_action?.available) {
    out.push(section("What the controller did", [
      paragraph(describeAction(state, decision)),
      decision.reward?.available
        ? paragraph(`The interval scored ${signed(decision.reward.interval_reward)} ` +
                    `and the run total is ${signed(decision.reward.cumulative_reward)}. ` +
                    `This is a simulation score, not money and not an industry KPI.`)
        : null,
    ]));
  }

  out.push(section("Why it matters here", [
    paragraph(incident.demands_at_risk.length
      ? `${count(incident.demands_at_risk.length)} demand(s) are disconnected or ` +
        `outside their SLA right now: ${incident.demands_at_risk.join(", ")}.`
      : "No demand is currently disconnected or outside its SLA."),
    paragraph("The governed study's conclusion is a separate, frozen record. Open " +
              "the governed conclusion to read what it did and did not establish."),
  ]));
  return out;
}

function explainNetwork(state, snapshot) {
  const selection = state.selection;
  const out = [];
  const incident = snapshot.incident;

  out.push(section("Network facts behind that statement", [
    paragraph(incident.failed_links.length
      ? `Failed links: ${incident.failed_link_labels.join(", ")}.`
      : "Every link is operational."),
    paragraph(incident.congested_links.length
      ? `Links at or above the congestion threshold: ${incident.congested_links.join(", ")}.`
      : "No link is at the congestion threshold."),
  ]));

  if (selection.objectType === "link") {
    const link = (snapshot.links || []).find((l) => l.id === selection.objectId);
    if (link) {
      out.push(section(`Link ${link.id}`, [
        paragraph(`${link.a_city} – ${link.z_city}, ${mbps(link.capacity_mbps)} per ` +
          `direction. The map summarizes with the busier direction, ` +
          `${link.worst_direction} at ${percent(link.worst_utilization, 1)}.`),
        paragraph(`Modeled queue delay and loss on that direction: ` +
          `${ms(link.directions[0].queue_delay_ms)} and ` +
          `${percent(link.directions[0].loss_fraction, 3)}. Both are analytic ` +
          `approximations of utilization, not packet measurements.`),
      ]));
    }
  }
  if (selection.objectType === "demand") {
    const demand = (snapshot.demands || []).find((d) => d.id === selection.objectId);
    if (demand) {
      out.push(section(`Demand ${demand.id}`, [
        paragraph(`${demand.src_city} → ${demand.dst_city}, ${demand.class_label}, ` +
          `offering ${mbps(demand.offered_mbps)} and carrying ` +
          `${mbps(demand.carried_mbps)}.`),
        paragraph(`Current route: ${demand.current_path_label}. Tightest hop at ` +
          `${percent(demand.bottleneck_util, 1)}. ${demand.risk_label}.`),
      ]));
    }
  }
  out.push(section("What the simulator does not model", [
    paragraph("Packets, TCP behaviour, RSVP-TE or IGP convergence, label signaling, " +
      "exact geography and production control-plane timing are outside this model. " +
      "Delay and loss are analytic functions of utilization."),
  ]));
  return out;
}

function explainRl(state, snapshot) {
  const decision = state.data.decision;
  const out = [];
  if (!decision) {
    return [unavailable("Decision pipeline",
      "No decision payload has been read for this moment yet.")];
  }

  out.push(section("Observation", [
    decision.observation.available
      ? paragraph(`The policy saw a ${count(decision.observation.dim)}-value ` +
          `${state.context.environmentVersion.toUpperCase()} observation. ` +
          (decision.observation.changed_count === null
            ? "No prior observation exists in this generation, so no change list is shown."
            : `${count(decision.observation.changed_count)} value(s) changed since the ` +
              `prior observation.`))
      : unavailable("Observation", decision.observation.reason),
    paragraph(decision.observation.ranking_note || ""),
  ]));

  out.push(section("Mask and output", [
    decision.mask.available
      ? paragraph(`${count(decision.mask.valid_count)} of ` +
          `${count(decision.mask.count)} actions were legal. Each rejection reason ` +
          `comes from the engine's own validator.`)
      : unavailable("Action mask", decision.mask.reason),
    decision.policy_output.available
      ? paragraph(`${decision.policy_output.label}: the selected action scored ` +
          `${policyValue(decision.policy_output.semantics,
                         decision.policy_output.selected?.value)}. ` +
          decision.policy_output.description)
      : unavailable(decision.policy_output.label, decision.policy_output.reason),
  ]));

  out.push(section("Transition and reward", [
    paragraph(describeAction(state, decision)),
    decision.reward.available
      ? paragraph(`${count(decision.reward.component_count)} reward components sum to ` +
          `${num(decision.reward.component_sum, 4)} against a scalar interval reward of ` +
          `${num(decision.reward.interval_reward, 4)}; residual ` +
          `${num(decision.reward.residual, 6)}. ` +
          `${decision.reward.exact_sum ? "The sum is exact." : "The sum does not reconcile."}`)
      : unavailable("Reward", decision.reward.reason),
  ]));
  return out;
}

function describeAction(state, decision) {
  const selected = decision.selected_action;
  if (!selected?.available) return "No action has been taken in this generation yet.";
  if (selected.kind === "baseline_moves") {
    return selected.n_moves
      ? `${selected.policy_id} moved ${count(selected.n_moves)} demand(s) this interval.`
      : `${selected.policy_id} made no TE change this interval.`;
  }
  if (selected.is_noop) return "The controller made no TE change this interval.";
  const decoded = selected.decoded || {};
  const verdict = selected.accepted === false
    ? `The move was rejected by the safety validator: ${selected.validator_reason}.`
    : "The move was accepted.";
  return `Action ${selected.action} moved ${decoded.demand || "a demand"} from ` +
         `path ${decoded.from_path} to path ${decoded.path_idx}. ${verdict}`;
}

function explainRecord(state, depth) {
  const kind = state.source.kind;
  if (kind === "recorded_replay") {
    return [section("Recorded trace", [
      paragraph("This is playback of an immutable recorded episode from the one-shot " +
        "final holdout. No controller is running and no action is being taken now."),
      paragraph("The trace records interval aggregates. It contains no per-link " +
        "utilization, so no link-level topology state is shown for it."),
    ])];
  }
  const stage = kind === "final_holdout_evidence" ? "final holdout" : "development";
  return [section(`${stage} evidence`, [
    paragraph(kind === "final_holdout_evidence"
      ? "This is the untouched one-shot holdout result. It was evaluated once, after " +
        "the study closed to selection, and nothing was tuned on it."
      : "This is selection-stage evidence: the seed-42 pilot and the three-root " +
        "continuity runs, evaluated on seeds 101-105. Checkpoint selection happened here."),
    paragraph("Nothing on this record is live, and it is never used as a live comparator."),
  ])];
}

export function renderExplain(state) {
  fill($("explain-body"), explain(state, state.ui.explainDepth));
  for (const chip of document.querySelectorAll("#drawer-explain .depth-switch .chip")) {
    chip.setAttribute("aria-pressed",
      chip.dataset.depth === state.ui.explainDepth ? "true" : "false");
  }
}
