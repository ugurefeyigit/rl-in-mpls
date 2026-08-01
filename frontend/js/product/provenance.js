/* Provenance rendering.
 *
 * The stamp carries a word, an icon and a pattern. Colour is the fourth signal,
 * never the first: a viewer must be able to tell LIVE from FINAL EVIDENCE in
 * grayscale, on a projector, from the back of a room.
 */

import { $, clear, el, fill, icon } from "./dom.js";
import { sourceProfile } from "./contracts.js";
import { clock, shortHash } from "./format.js";

export function renderProvenance(state) {
  const kind = state.source.kind;
  const profile = sourceProfile(kind);
  const stamp = $("provenance-stamp");
  stamp.dataset.kind = kind;
  document.body.dataset.source = kind;

  clear(stamp);
  stamp.appendChild(icon(profile.icon));
  stamp.appendChild(el("span", { class: "stamp__word", text: profile.label }));
  stamp.appendChild(el("span", { class: "stamp__detail", id: "provenance-detail",
                                 text: detailFor(state) }));
}

export function detailFor(state) {
  const kind = state.source.kind;
  const context = state.context;
  if (kind === "live_session") {
    const status = state.data.snapshot?.session;
    if (!status) return "No session running";
    return `${status.state} · step ${context.step ?? 0} · ${clock(context.hour)}`;
  }
  if (kind === "recorded_replay") {
    const trace = state.data.replay;
    return trace ? `${trace.policy_id} · ${trace.scenario} · seed ${trace.seed}`
                 : "No episode loaded";
  }
  if (kind === "final_holdout_evidence") {
    const provenance = state.data.evidence.finalProvenance;
    return provenance ? `one-shot · source ${shortHash(provenance.source_sha)}`
                      : "one-shot holdout";
  }
  return "selection stage · not holdout";
}

/** The persistent context tuple. Absent members are omitted, never defaulted. */
export function renderContextLedger(state) {
  const cells = [];
  const context = state.context;
  const kind = state.source.kind;

  if (kind === "live_session") {
    cells.push(["Environment", context.environmentVersion?.toUpperCase()]);
    cells.push(["Scenario", context.scenarioLabel || context.scenario]);
    cells.push(["Seed", context.seed]);
    cells.push(["Policy", policyLabel(state)]);
    if (context.comparator) cells.push(["Comparator", context.comparator]);
    cells.push(["Time", context.hour === null ? null : clock(context.hour)]);
    cells.push(["Step", context.step]);
  } else if (kind === "recorded_replay") {
    const trace = state.data.replay;
    cells.push(["Environment", "V2"]);
    cells.push(["Policy", trace?.policy_id]);
    cells.push(["Scenario", trace?.scenario]);
    cells.push(["Seed", trace?.seed]);
    cells.push(["Recorded step", trace?.currentStep]);
  } else {
    cells.push(["Environment", "V2"]);
    cells.push(["Stage", kind === "final_holdout_evidence" ? "final holdout" : "development"]);
    cells.push(["Grain", kind === "final_holdout_evidence"
      ? "root-averaged over 5 holdout seeds" : "selection stage"]);
  }

  const selection = state.selection;
  if (selection.objectType && selection.objectId) {
    cells.push(["Selected", `${selection.objectType} ${selection.objectId}`]);
  }

  fill($("context-ledger"), cells
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([term, value]) => el("div", {
      class: "context__cell",
      dataset: { term: term.toLowerCase().replaceAll(" ", "-") },
    }, [
      el("dt", { text: term }),
      el("dd", { text: String(value) }),
    ])));
}

function policyLabel(state) {
  const capabilities = state.data.capabilities;
  const id = state.context.policyId;
  if (!id) return null;
  const policy = capabilities?.live_policies?.find(
    (p) => p.id === id && p.environment_version === state.context.environmentVersion);
  const checkpoint = state.context.checkpointId;
  return checkpoint ? `${policy?.label || id} · ${checkpoint}` : (policy?.label || id);
}

/* The record switch says which record you are looking at. It uses the plain
 * wording, not the bare ledger stamp, and study evidence carries a "Study
 * result" prefix so it can never read as another thing you could run live. */
export function renderSourceSwitch(state, { onSelect }) {
  const container = document.querySelector("#source-switch .source-switch__options");
  const capabilities = state.data.capabilities;
  if (!capabilities) return;
  fill(container, capabilities.sources.map((source) => {
    const selected = source.kind === state.source.kind;
    const evidence = source.group === "study_evidence";
    return el("button", {
      type: "button",
      class: "chip",
      dataset: { group: source.group || "live" },
      "aria-pressed": selected ? "true" : "false",
      disabled: !source.available && !selected,
      title: source.available
        ? (source.plain_summary || source.description)
        : source.unavailable_reason,
      onClick: () => onSelect(source.kind),
      text: evidence
        ? `Study result · ${source.plain_label || source.label}`
        : (source.plain_label || source.label),
    });
  }));
}
