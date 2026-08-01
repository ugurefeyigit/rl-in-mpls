/* Presentation mode.
 *
 * The topology owns the viewport. Everything else is one prioritized moment
 * rail, the recommendation directly beneath the map, a quiet comparison lane,
 * and story copy — in that order, because that is the order a presenter speaks in.
 *
 * The five equal KPI cards the old wallboard used are gone: they gave a failed
 * link the same visual weight as the delivered ratio.
 */

import { $, el, fill, unavailable } from "../dom.js";
import { count, percent, signed } from "../format.js";
import { renderComparisonLane } from "../comparison-lane.js";
import { BEATS, beatAt, progressText, storyContext } from "../guided-story.js";

export function renderPresentation(state) {
  renderMomentRail(state);
  const snapshot = state.data.snapshot;
  fill($("panel-presentation"), [
    storySection(state),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Network condition" }),
      snapshot ? conditionList(snapshot)
               : unavailable("Condition", "No run has produced a snapshot yet."),
    ]),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Comparison lane" }),
      renderComparisonLane(state),
    ]),
  ]);
}

function renderMomentRail(state) {
  const rail = $("moment-rail");
  const snapshot = state.data.snapshot;
  rail.hidden = state.mode !== "presentation";
  if (state.mode !== "presentation") return;

  if (!snapshot) {
    fill($("moment-primary"), [cell("Session", "Not started", "phase")]);
    $("moment-change").textContent =
      "Choose a scenario and a controller on the left, then press Start run.";
    return;
  }

  const incident = snapshot.incident;
  const metrics = snapshot.metrics;
  const values = metrics.available ? metrics.values : {};
  const decision = state.data.decision;

  fill($("moment-primary"), [
    cell("Phase", incident.label, "phase", incident.phase),
    cell("Time", `${snapshot.time.clock} · step ${count(snapshot.time.step)}`, "time"),
    cell("Incident", incident.active_incident || "No active incident", "incident",
         incident.failed_links.length ? "failure"
           : (incident.congested_links.length ? "pressure" : "normal")),
    cell("Action", actionText(decision), "action"),
    cell("Interval reward", decision?.reward?.available
      ? signed(decision.reward.interval_reward, 3) : "—", "reward"),
    cell("Cumulative reward", decision?.reward?.available
      ? signed(decision.reward.cumulative_reward, 2) : "—", "reward"),
    cell("Busiest link", percent(values.max_util?.value, 0), "metric"),
    cell("SLA risks now", count(incident.demands_at_risk.length), "metric"),
  ]);

  $("moment-change").textContent = changeSentence(state);
}

function cell(label, value, kind, condition) {
  return el("div", {
    class: "moment-cell",
    dataset: { kind, condition: condition || "" },
  }, [
    el("span", { class: "moment-cell__label", text: label }),
    el("span", { class: "moment-cell__value", text: String(value) }),
  ]);
}

function actionText(decision) {
  const selected = decision?.selected_action;
  if (!selected?.available) return "No TE change";
  if (selected.kind === "baseline_moves") {
    return selected.n_moves ? `${count(selected.n_moves)} move(s)` : "No TE change";
  }
  if (selected.is_noop) return "No TE change";
  const decoded = selected.decoded || {};
  return `${decoded.demand} → path ${decoded.path_idx}`;
}

/** One sentence about what changed since the prior completed step. */
export function changeSentence(state) {
  const snapshot = state.data.snapshot;
  const previous = state.data.previousSnapshot;
  if (!snapshot?.metrics?.available) {
    return "Waiting for the first completed interval.";
  }
  if (!previous) {
    if ((snapshot.time?.step ?? 0) <= 1) {
      return "This is the first completed interval, so there is nothing to compare against yet.";
    }
    return "No prior browser snapshot is available for comparison with this progressed session.";
  }
  const parts = [];
  const nowFailed = new Set(snapshot.incident.failed_links);
  const wasFailed = new Set(previous.incident.failed_links);
  const newlyFailed = [...nowFailed].filter((id) => !wasFailed.has(id));
  const repaired = [...wasFailed].filter((id) => !nowFailed.has(id));
  if (newlyFailed.length) parts.push(`${newlyFailed.join(", ")} failed`);
  if (repaired.length) parts.push(`${repaired.join(", ")} came back up`);

  const maxNow = snapshot.metrics.values.max_util?.value;
  const maxBefore = previous.metrics.values?.max_util?.value;
  if (maxNow !== undefined && maxBefore !== undefined) {
    const delta = maxNow - maxBefore;
    if (Math.abs(delta) >= 0.01) {
      parts.push(`busiest link ${delta > 0 ? "rose" : "fell"} to ${percent(maxNow, 0)}`);
    }
  }
  const riskNow = snapshot.incident.demands_at_risk.length;
  const riskBefore = previous.incident.demands_at_risk.length;
  if (riskNow !== riskBefore) {
    parts.push(`${count(riskNow)} demand(s) now at risk, from ${count(riskBefore)}`);
  }
  return parts.length
    ? `Since the previous step: ${parts.join("; ")}.`
    : "Since the previous step: no material change.";
}

function storySection(state) {
  if (!state.story.active) {
    return el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Guided Story" }),
      el("p", { class: "prose",
        text: "Guided Story walks one real demo_evening session in eleven beats, " +
              "from normal operation through congestion, SLA risk, a policy " +
              "recommendation, a link failure and the governed conclusion. " +
              "It advances the actual engine; it does not script the network." }),
    ]);
  }

  const index = state.story.reviewBeat ?? state.story.beat;
  const beat = beatAt(index);
  const context = storyContext(state);

  return el("section", { class: "panel story" }, [
    el("h2", { class: "panel__title", text: `Guided Story · ${progressText(state)}` }),
    el("ol", { class: "story__beats", "aria-label": "Story beats" },
      BEATS.map((b, i) => el("li", {
        class: "story__beat",
        "aria-current": i === index ? "step" : "false",
        dataset: { done: i < state.story.beat ? "true" : "false" },
      }, [el("span", { text: b.label })]))),
    el("p", { class: "story__copy prose", text: beat.narrate(context) }),
    state.story.reviewBeat !== null && state.story.reviewBeat !== state.story.beat
      ? el("p", { class: "story__review",
          text: `You are reviewing an earlier beat. The live network has not been ` +
                `rewound — it remains at ${context.clock}.` })
      : null,
  ]);
}

function conditionList(snapshot) {
  const incident = snapshot.incident;
  return el("dl", { class: "facts" }, [
    el("dt", { text: "Failed links" }),
    el("dd", { text: incident.failed_link_labels.join(", ") || "None" }),
    el("dt", { text: "Congested links" }),
    el("dd", { text: incident.congested_links.join(", ") || "None" }),
    el("dt", { text: "Demands at risk" }),
    el("dd", { text: incident.demands_at_risk.join(", ") || "None" }),
  ]);
}
