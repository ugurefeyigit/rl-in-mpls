/* RL Information: the decision observatory and governed record.
 *
 * Decision, Study and Provenance are secondary views inside one primary mode.
 * They never appear in the primary navigation and never coerce one source kind
 * into another. Each renderer consumes only fields its typed adapter supplied.
 */

import { $, el, fill, unavailable } from "../dom.js";
import { renderActionGrid, selectedActionPanel } from "../action-grid.js";
import { renderGovernedStudy } from "../governed-study.js";
import { renderModelProvenance } from "../model-provenance.js";
import { renderObservationInspector } from "../observation-inspector.js";
import { counterfactualPanel, renderPolicyOutputs } from "../policy-outputs.js";
import { renderRecordedTrace } from "../recorded-trace.js";
import { renderRewardWaterfall } from "../reward-waterfall.js";

const PIPELINE_LABELS = {
  observation: "Observation",
  mask: "Action mask",
  policy_output: "Policy output",
  selected_action: "Selected action",
  safety: "Safety validation",
  transition: "Transition",
  reward: "Reward",
  next_observation: "Next observation",
};

export function renderRl(state, handlers) {
  const panel = $("panel-rl");
  fill(panel, [
    secondaryNavigation(state, handlers.onSetView),
    state.rlView === "study" ? studyView(state, handlers)
      : (state.rlView === "provenance" ? provenanceView(state)
        : decisionView(state, handlers)),
  ]);
  renderRlRail(state);
}

function secondaryNavigation(state, onSetView) {
  const views = [
    ["decision", "Decision Observatory"],
    ["study", "Governed Study"],
    ["provenance", "Model Provenance"],
  ];
  return el("nav", { class: "rl-subnav", "aria-label": "RL Information views" },
    views.map(([id, label]) => el("button", {
      type: "button", class: "chip",
      "aria-current": state.rlView === id ? "page" : "false",
      onClick: () => onSetView(id), text: label,
    })));
}

function decisionView(state, handlers) {
  if (state.source.kind !== "live_session") {
    return el("section", { class: "panel" }, [
      unavailable("Decision Observatory",
        state.source.kind === "recorded_replay"
          ? "This recorded trace contains aggregate interval rows, not observations, masks or policy outputs."
          : "A frozen evidence region is not a live policy decision. Switch to LIVE to inspect inference."),
    ]);
  }

  const decision = state.data.decision;
  return el("div", { class: "rl-observatory" }, [
    pipeline(decision),
    el("div", { class: "rl-grid" }, [
      region("Observation", renderObservationInspector(state, {
        search: state.ui.observationSearch || "",
        changedOnly: Boolean(state.ui.observationChangedOnly),
        onSearch: handlers.onObservationSearch,
        onToggleChanged: handlers.onToggleObservationChanged,
      })),
      region("Complete 69-action space", renderActionGrid(state, {
        showInvalid: Boolean(state.ui.showInvalidActions),
        onToggleInvalid: handlers.onToggleInvalidActions,
        onSelectAction: handlers.onSelectAction,
      })),
      region("Selected action and safety", selectedActionPanel(state)),
      region("Policy output", renderPolicyOutputs(state)),
      region("Decision Lens", counterfactualPanel(state, {
        onRequest: handlers.onCounterfactual,
      })),
      region("Reward components", renderRewardWaterfall(state)),
    ]),
  ]);
}
function pipeline(decision) {
  const stages = decision?.pipeline || Object.keys(PIPELINE_LABELS);
  const current = decision?.current_stage || "observation";
  return el("ol", { class: "pipeline", "aria-label": "Policy-decision pipeline" },
    stages.map((stage) => el("li", {
      class: "pipeline__stage",
      "aria-current": stage === current ? "step" : "false",
    }, [
      el("span", { class: "pipeline__name", text: PIPELINE_LABELS[stage] || stage }),
      el("span", { class: "pipeline__state", text: stageState(decision, stage) }),
    ])));
}

function stageState(decision, stage) {
  if (!decision) return "waiting";
  const value = decision[stage];
  if (value?.available === false) return "unavailable";
  if (stage === "safety") return decision.safety?.accepted === false ? "rejected" : "checked";
  if (stage === "transition") return decision.transition?.available ? "observed" : "pending";
  return value ? "available" : "pending";
}

function studyView(state, handlers) {
  return el("div", { class: "study-stack" }, [
    state.source.kind === "recorded_replay"
      ? region("Recorded replay", renderRecordedTrace(state, {
          onLoad: handlers.onLoadReplay, onScrub: handlers.onScrubReplay,
        }), "region region--recorded")
      : renderGovernedStudy(state),
  ]);
}

function provenanceView(state) {
  return el("div", { class: "prov" }, [renderModelProvenance(state)]);
}

function region(title, content, className = "region") {
  return el("section", { class: className }, [
    el("h2", { class: "region__title", text: title }),
    content,
  ]);
}

function renderRlRail(state) {
  fill($("rail"), [
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Evidence boundary" }),
      el("p", { class: "prose", text: state.source.kind === "live_session"
        ? "This is a live decision from the V1 runner. It is not final-holdout evidence."
        : `${state.source.kind.replaceAll("_", " ")} is read-only and cannot execute a policy.` }),
    ]),
    el("section", { class: "panel" }, [
      el("h2", { class: "panel__title", text: "Output language" }),
      el("p", { class: "prose", text:
        "MaskablePPO exposes action probabilities when available. A masked bandit exposes unnormalized action scores or immediate-reward estimates, never probabilities." }),
    ]),
  ]);
}
