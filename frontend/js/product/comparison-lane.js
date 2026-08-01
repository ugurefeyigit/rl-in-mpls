/* The synchronized side-by-side decision comparison.
 *
 * It renders a verdict only while the backend can prove both runners share one
 * experiment. When the proof fails, the lane names the fields that broke it and
 * shows no comparative claim at all — not a greyed-out verdict, not a verdict
 * with a caveat. A wrong comparison is worse than no comparison.
 *
 * Three rules this file keeps, each mirroring one the backend keeps:
 *
 * - lanes are told apart by a letter token and a line style as well as colour;
 * - a signed operational return never grows a percentage difference;
 * - controller TE changes, FRR protection moves and post-recovery restorations
 *   are three columns, never one "reroutes" number.
 */

import { el, unavailable } from "./dom.js";
import { count, metricValue, signed } from "./format.js";

export function renderComparisonLane(state) {
  const comparison = state.data.comparison;

  if (!comparison) {
    return unavailable("Comparison", "No comparison state has been read yet.");
  }
  if (!comparison.comparison) {
    return unavailable("Comparison", comparison.reason);
  }
  if (!comparison.matched) {
    return el("div", { class: "cmp cmp--broken" }, [
      unavailable("Comparison disabled", comparison.reason),
      el("dl", { class: "facts" }, [
        el("dt", { text: "Proof required" }),
        el("dd", { text: comparison.proof || "—" }),
        el("dt", { text: "Fields that disagree" }),
        el("dd", { text: (comparison.mismatched_fields || []).join(", ") || "—" }),
      ]),
      el("p", { class: "cmp__proof",
        text: "No metric, gap or verdict is shown while the proof is broken." }),
    ]);
  }

  const detail = comparison.detail;
  if (!detail?.available) {
    return unavailable("Comparison", detail?.reason
      || "The comparison detail did not load.");
  }

  const [a, b] = detail.lanes;
  return el("div", { class: "cmp" }, [
    el("p", { class: "cmp__sync", id: "cmp-sync",
      text: `Synchronized on ${comparison.scenario}, seed ${comparison.seed}. `
            + `${detail.reason} Proof: ${detail.proof}.` }),
    verdictBlock(detail.verdict),
    decisionCards(a, b),
    metricTable(detail.metric_rows, a, b),
    movementTable(a, b, detail.movement_note),
    divergenceBlock(detail.divergence),
    el("p", { class: "cmp__proof", text: comparison.demonstration_note }),
  ]);
}

/* ------------------------------------------------------------------ verdict */
function verdictBlock(verdict) {
  if (!verdict) return null;
  return el("div", { class: "cmp__verdict", dataset: { leader: verdict.leader || "level" } }, [
    el("p", { class: "cmp__verdict-line", text: verdict.sentence }),
    el("dl", { class: "facts" }, [
      el("dt", { text: "A cumulative return" }),
      el("dd", { text: signed(verdict.a, 4) }),
      el("dt", { text: "B cumulative return" }),
      el("dd", { text: signed(verdict.b, 4) }),
      el("dt", { text: "Gap (A − B)" }),
      el("dd", { text: `${signed(verdict.gap, 4)} ${verdict.unit}` }),
    ]),
    el("p", { class: "cmp__proof", text: verdict.percentage_reason }),
  ]);
}

/* ------------------------------------------------------------ decision cards */
function decisionCards(a, b) {
  return el("div", { class: "cmp__lanes" }, [laneCard(a), laneCard(b)]);
}

function laneCard(lane) {
  const action = lane.action || {};
  return el("article", { class: "cmp__lane", dataset: { lane: lane.position } }, [
    el("h3", { class: "cmp__lane-head" }, [
      el("span", { class: "cmp__token", text: lane.token }),
      el("span", { class: "cmp__lane-name", text: lane.algorithm }),
    ]),
    el("p", { class: "cmp__lane-kind",
      text: `${lane.family === "learner" ? "Learned policy" : "Rule-based baseline"}`
            + (lane.checkpoint_id ? ` · ${lane.checkpoint_id}` : "") }),
    el("dl", { class: "facts" }, [
      el("dt", { text: "This interval" }),
      el("dd", { text: action.available ? action.text : (action.reason || "—") }),
      el("dt", { text: "Interval return" }),
      el("dd", { text: signed(lane.interval_reward, 3) }),
      el("dt", { text: "Cumulative return" }),
      el("dd", { text: signed(lane.cumulative_reward, 3) }),
      el("dt", { text: "Intervals completed" }),
      el("dd", { text: count(lane.steps_recorded) }),
    ]),
    action.validator_reason && action.accepted === false
      ? el("p", { class: "cmp__reject",
          text: `The environment refused this move: ${action.validator_reason}` })
      : null,
    action.explanation
      ? el("p", { class: "cmp__lane-note", text: action.explanation })
      : null,
  ]);
}

/* ------------------------------------------------------------- metric table */
function metricTable(rows, a, b) {
  if (!rows?.length) {
    return unavailable("Interval metrics",
      "Neither lane has completed an interval, so there is nothing to compare.");
  }
  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid cmp__grid" }, [
      el("caption", { text: "Latest completed interval, lane by lane. A blank "
        + "lead means neither direction is better by itself." }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Measure" }),
        el("th", { scope: "col", text: `A · ${a.algorithm}` }),
        el("th", { scope: "col", text: `B · ${b.algorithm}` }),
        el("th", { scope: "col", text: "A − B" }),
        el("th", { scope: "col", text: "Ahead" }),
      ])]),
      el("tbody", {}, rows.map((row) => el("tr", {
        dataset: { leader: row.leader || "" },
      }, [
        el("th", { scope: "row", text: row.label }),
        el("td", { text: metricValue(row.unit, row.a) }),
        el("td", { text: metricValue(row.unit, row.b) }),
        el("td", { text: gapText(row) }),
        el("td", { text: row.leader ? row.leader.toUpperCase() : "—" }),
      ]))),
    ]),
  ]);
}

function gapText(row) {
  if (row.gap === 0) return "level";
  if (row.unit === "count") return signed(row.gap, 0);
  if (row.unit === "share") return `${signed(row.gap * 100, 1)} pp`;
  return signed(row.gap, 3);
}

/* ----------------------------------------------------------- movement table */
function movementTable(a, b, note) {
  const keys = [...new Set([...Object.keys(a.movement || {}),
                            ...Object.keys(b.movement || {})])];
  if (!keys.length) return null;
  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid cmp__grid" }, [
      el("caption", { text: "Cumulative movement over the run. Three separate "
        + "counters, never summed." }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Counter" }),
        el("th", { scope: "col", text: "Attributed to" }),
        el("th", { scope: "col", text: `A · ${a.algorithm}` }),
        el("th", { scope: "col", text: `B · ${b.algorithm}` }),
      ])]),
      el("tbody", {}, keys.map((key) => {
        const meta = a.movement?.[key] || b.movement?.[key];
        return el("tr", {}, [
          el("th", { scope: "row", text: meta.label }),
          el("td", { text: meta.attribution }),
          el("td", { text: count(a.movement?.[key]?.total ?? null) }),
          el("td", { text: count(b.movement?.[key]?.total ?? null) }),
        ]);
      })),
    ]),
    note ? el("p", { class: "cmp__proof", text: note }) : null,
  ]);
}

/* -------------------------------------------------------------- divergence */
function divergenceBlock(divergence) {
  if (!divergence) return null;
  if (!divergence.available) {
    return el("p", { class: "cmp__proof", text: divergence.reason });
  }
  return el("p", { class: "cmp__diverge",
    text: `The lanes first decided differently at step ${count(divergence.step)}: `
          + `A ${divergence.a_moved ? "moved a demand" : "made no TE change"}, `
          + `B ${divergence.b_moved ? "moved a demand" : "made no TE change"}. `
          + divergence.note });
}
