/*
THESIS: Completed-run comparison is one aligned operating ledger, not a dashboard card mosaic.
OWN-WORLD: Basalt atlas field, mineral rules, A violet solid circles, B orange dashed diamonds.
STORY: Verify identity and pairing, inspect reward and network condition, then trace decisions and terms.
FIRST VIEWPORT: A/B provenance and headline table lead; the full-width reward instrument follows immediately.
FORM: Dispatch Atlas analytical extension; shared interval cursor is the signature interaction.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
*/

import { $, el, fill, unavailable } from "../dom.js";
import { renderComparisonPicker } from "../comparison-picker.js";
import { renderChurnInstrument, renderComponentInstrument, renderDecisionInstrument,
         renderLineInstrument } from "../comparison-charts.js";

export function renderCompare(state, handlers) {
  const payload = state.data.comparativeRuns;
  const panel = $("panel-compare");
  const picker = renderComparisonPicker(state, handlers);
  if (!payload?.slots?.a || !payload?.slots?.b) {
    fill(panel, [el("div", { class: "compare-surface" }, [
      el("header", { class: "compare-intro" }, [
        el("h1", { text: "Comparative Run Results" }),
        el("p", { text: "Choose two completed live-demonstration runs. Nothing here is governed evidence." }),
      ]),
      picker,
      unavailable("Two completed runs required",
        "A and B remain empty until a run finishes. Unfinished runs are not eligible."),
    ])]);
    return;
  }

  const { a, b } = payload.slots;
  const paired = payload.pairing.synchronized;
  const selectedStep = paired ? state.comparisonView.selectedStep : null;
  const onSelectStep = paired ? handlers.onSelectStep : null;
  const cumulative = state.comparisonView.rewardView === "cumulative";
  fill(panel, [el("div", { class: "compare-surface", dataset: {
    synchronized: payload.pairing.synchronized ? "true" : "false",
  } }, [
    el("header", { class: "compare-intro" }, [
      el("h1", { text: "Comparative Run Results" }),
      el("p", { text: "Two completed live-demonstration records · process memory only · never merged with the closed study." }),
    ]),
    picker,
    viewControls(state, handlers),
    renderLineInstrument({
      id: "reward-chart", title: cumulative
        ? "How did cumulative reward evolve?"
        : "How did interval reward evolve?",
      note: cumulative
        ? "Cumulative reward is explicitly selected; this is not a second scale."
        : "Interval reward with an explicit zero line. Use the labelled control for cumulative reward.",
      unit: "signed operational return",
      series: pairedSeries(a, b, "reward", cumulative ? "cumulative" : "value"),
      references: [{ value: 0, label: "Zero reward" }], selectedStep,
      onSelect: onSelectStep,
    }),
    el("div", { class: "compare-grid compare-grid--network" }, [
      renderLineInstrument({
        id: "utilization-chart", title: "Where did link pressure differ?",
        note: "Maximum and mean aggregate utilization. Pressure and capacity references come from the printed percent scale.",
        unit: "percent", series: utilizationSeries(a, b),
        references: [{ value: 0.7, label: "70% pressure" },
                     { value: 1, label: "100% capacity" }],
        domain: [0, Math.max(1.05, utilizationPeak(a), utilizationPeak(b))],
        selectedStep, onSelect: onSelectStep,
      }),
      renderLineInstrument({
        id: "delivery-chart", title: "How much traffic was delivered?",
        note: "Delivered traffic as a ratio of offered traffic. Higher is favorable.",
        unit: "percent", series: pairedSeries(a, b, "delivery"), domain: [0, 1],
        selectedStep, onSelect: onSelectStep,
      }),
      renderLineInstrument({
        id: "sla-chart", title: "When did SLA risk appear?",
        note: "Violating demands per completed interval. This panel stays separate from delivery; there is no dual axis.",
        unit: "violating demands per interval", series: pairedSeries(a, b, "sla_risk"),
        domain: [0, Math.max(1, seriesPeak(a, "sla_risk"), seriesPeak(b, "sla_risk"))],
        selectedStep, onSelect: onSelectStep,
      }),
    ]),
    el("div", { class: "compare-grid compare-grid--decisions" }, [
      renderDecisionInstrument({ title: "When did controllers act and incidents occur?",
        a, b, selectedStep, onSelect: onSelectStep }),
      renderChurnInstrument({ title: "How much route churn did each run create?",
        headline: payload.headline }),
    ]),
    renderComponentInstrument({ title: "Which reward terms created the return?", a, b }),
    selectedInterval(payload, selectedStep),
  ])]);
}

function viewControls(state, handlers) {
  const cumulative = state.comparisonView.rewardView === "cumulative";
  return el("div", { class: "compare-controls", role: "group",
    "aria-label": "Comparison view controls" }, [
    el("button", { type: "button", class: "ctl", "aria-pressed": String(!cumulative),
      text: "Interval reward", onclick: () => handlers.onRewardView("interval") }),
    el("button", { type: "button", class: "ctl", "aria-pressed": String(cumulative),
      text: "Cumulative reward", onclick: () => handlers.onRewardView("cumulative") }),
    el("button", { type: "button", class: "ctl", text: "Reset view", onclick: () => {
      handlers.onResetView();
    } }),
  ]);
}

function pairedSeries(a, b, key, valueKey = "value") {
  return [["a", a], ["b", b]].map(([lane, run]) => {
    const source = run.series[key];
    return {
      lane, key: `${lane}-${key}`, label: `${lane.toUpperCase()} · ${run.identity.controller}`,
      direct: lane.toUpperCase(), reason: source.reason,
      values: (source.values || []).map((point) => ({ ...point, value: point[valueKey] })),
    };
  });
}

function utilizationSeries(a, b) {
  return [["a", a], ["b", b]].flatMap(([lane, run]) => [
    { lane, key: `${lane}-max-util`, label: `${lane.toUpperCase()} · maximum`,
      direct: `${lane.toUpperCase()} max`, values: run.series.utilization.max.values,
      reason: run.series.utilization.max.reason, className: "chart-line--maximum" },
    { lane, key: `${lane}-mean-util`, label: `${lane.toUpperCase()} · mean`,
      direct: `${lane.toUpperCase()} mean`, values: run.series.utilization.mean.values,
      reason: run.series.utilization.mean.reason, className: "chart-line--mean" },
  ]);
}

function utilizationPeak(run) {
  return Math.max(0, ...(run.series.utilization.max.values || []).map((point) => point.value));
}

function seriesPeak(run, key) {
  return Math.max(0, ...(run.series[key].values || []).map((point) => point.value));
}

function selectedInterval(payload, step) {
  if (!payload.pairing.synchronized) return el("section", { class: "selected-interval" }, [
    el("h2", { text: "Interval pairing unavailable" }),
    el("p", { text: "The completed runs remain separately inspectable, but shared interval selection and paired conclusions are disabled because their synchronization fields differ." }),
  ]);
  if (step === null) return el("section", { class: "selected-interval" }, [
    el("h2", { text: "Selected interval" }),
    el("p", { text: "Select or focus a point, then press Enter to link the same interval across every graph." }),
  ]);
  const { a, b } = payload.slots;
  const event = (a.timeline || []).find((row) => row.step === step);
  const object = event?.object_type && event?.object_id
    ? `&object=${encodeURIComponent(`${event.object_type}:${event.object_id}`)}` : "";
  const run = encodeURIComponent(a.run_id);
  return el("section", { class: "selected-interval", role: "status" }, [
    el("h2", { text: `Selected interval · step ${step}` }),
    el("p", { text: payload.pairing.synchronized
      ? "The shared cursor refers to the same scenario, seed and interval in A and B."
      : "The cursor shows the same numbered step in two completed runs. They are not synchronized, so no paired or causal conclusion is made." }),
    el("div", { class: "selected-interval__links" }, [
      el("a", { class: "ctl", href: `/advanced?comparison_run=${run}&step=${step}${object}`,
        text: "Open interval in Network Information" }),
      el("a", { class: "ctl", href: `/study?source=live_session&comparison_run=${run}&step=${step}${object}`,
        text: "Open decision in RL Information" }),
    ]),
  ]);
}
