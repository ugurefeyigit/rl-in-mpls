/* The cross-mode results surface.
 *
 * Three record classes, three sections, three headings, three separate tables.
 * There is deliberately no combined table and no "all runs" view in this file:
 * the one thing this surface must never let a reader do is average a live
 * demonstration with a holdout result.
 *
 * The frozen study numbers are NOT rendered here. This section carries a
 * pointer and a reason, and the governed record stays where it is rendered from
 * its own artifacts (`governed-study.js`). Copying a frozen number into a second
 * renderer is how two versions of a result start to disagree.
 */

import { el, unavailable } from "./dom.js";
import { count, metricValue, num, percent, signed } from "./format.js";

const RUN_COLUMNS = [
  ["operational_return", "Operational return", "return"],
  ["steps", "Intervals", "count"],
  ["max_util_peak", "Busiest link, peak", "share"],
  ["max_util_mean", "Busiest link, mean", "share"],
  ["delivered_ratio_mean", "Delivered traffic, mean", "share"],
  ["sla_violations_peak", "SLA violations, peak", "count"],
  ["mean_delay_ms", "Mean demand delay", "ms"],
];

export function renderResults(state) {
  const results = state.data.results;
  if (!results) {
    return el("p", { class: "tb-empty",
      text: "Results have not been read yet. Open this panel again after a run "
            + "has completed an interval." });
  }
  return el("div", { class: "results" }, [
    el("p", { class: "results__rule", text: results.separation_rule }),
    liveSection(results),
    retainedSection(results),
    studySection(results, state),
  ]);
}

/* ------------------------------------------------------------------- live */
function liveSection(results) {
  const live = results.live;
  const meta = results.record_classes.live_demonstration;
  return section("results-live", meta, [
    live.available
      ? el("p", { class: "results__sub",
          text: `${live.scenario} · seed ${live.seed} · ${live.environment.toUpperCase()}`
                + (live.training_root ? ` · root ${live.training_root}` : "")
                + ` · ${live.execution} execution · ${count(live.steps)} interval(s)` })
      : null,
    live.available ? runTable(live.runs)
      : unavailable("No live results yet", live.reason),
  ]);
}

/* --------------------------------------------------------------- retained */
function retainedSection(results) {
  const retained = results.retained;
  const meta = results.record_classes.retained_demonstration;
  return section("results-retained", meta, [
    el("p", { class: "results__sub", text: retained.lifetime }),
    retained.count
      ? el("div", {}, retained.runs.map((archive, index) => el("div", {
          class: "results__archive",
        }, [
          el("h4", { class: "results__archive-head",
            text: `Run ${index + 1} · ${archive.scenario} · seed ${archive.seed} · `
                  + `${String(archive.environment).toUpperCase()} · `
                  + `${count(archive.steps)} interval(s)` }),
          runTable(archive.runs),
        ])))
      : el("p", { class: "tb-empty",
          text: "No earlier run has been kept yet. Reset run archives the run it "
                + "replaces; Full reset keeps it for this server process." }),
  ]);
}

/* ------------------------------------------------------------------ study */
function studySection(results, state) {
  const study = results.study;
  const meta = results.record_classes.governed_evidence;
  const loaded = Boolean(state.data.evidence?.finalHoldout);
  return section("results-study", meta, [
    el("p", { class: "results__sub", text: study.grain }),
    el("p", { class: "results__pointer", text: study.reason }),
    el("p", { class: "cmp__proof",
      text: loaded
        ? "The frozen record is open in this session. It is rendered from its own "
          + "artifacts under RL Information → Governed Study, and no number from "
          + "it is copied into the sections above."
        : "Open it from Study evidence and results in the control panel. It is "
          + "never loaded into the sections above." }),
  ]);
}

/* ----------------------------------------------------------------- shared */
function section(id, meta, children) {
  return el("section", { class: "results__section", id,
                         "aria-labelledby": `${id}-title` }, [
    el("h3", { class: "results__title", id: `${id}-title`, text: meta.label }),
    el("p", { class: "results__grain", text: meta.grain }),
    el("p", { class: "results__class", text: meta.reason }),
    ...children,
  ]);
}

function runTable(runs) {
  if (!runs?.length) {
    return el("p", { class: "tb-empty", text: "This record holds no controller runs." });
  }
  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid" }, [
      el("caption", { text: "Every value is derived from the same per-interval "
        + "records the exports read, so a displayed number and an exported "
        + "number cannot disagree." }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Controller" }),
        ...RUN_COLUMNS.map(([, label]) => el("th", { scope: "col", text: label })),
      ])]),
      el("tbody", {}, runs.map((run) => el("tr", {}, [
        el("th", { scope: "row",
          text: run.checkpoint_id ? `${run.algorithm} · ${run.checkpoint_id}`
                                  : run.algorithm }),
        ...RUN_COLUMNS.map(([key, , unit]) => el("td", { text: cell(run[key], unit) })),
      ]))),
    ]),
    movementList(runs),
  ]);
}

function movementList(runs) {
  const keys = [...new Set(runs.flatMap((run) => Object.keys(run.movement || {})))];
  if (!keys.length) return null;
  return el("table", { class: "grid" }, [
    el("caption", { text: "Movement counters. Controller changes, protection "
      + "moves and restorations stay apart." }),
    el("thead", {}, [el("tr", {}, [
      el("th", { scope: "col", text: "Controller" }),
      ...keys.map((key) => el("th", { scope: "col", text: key.replace(/_/g, " ") })),
    ])]),
    el("tbody", {}, runs.map((run) => el("tr", {}, [
      el("th", { scope: "row", text: run.algorithm }),
      ...keys.map((key) => el("td", { text: count(run.movement?.[key] ?? null) })),
    ]))),
  ]);
}

function cell(value, unit) {
  if (value === null || value === undefined) return "—";
  // A signed operational return is never formatted as a percentage or a rate.
  if (unit === "return") return signed(value, 4);
  if (unit === "share") return percent(value, 1);
  if (unit === "count") return count(value);
  if (unit === "ms") return metricValue("ms", value);
  return num(value, 3);
}
