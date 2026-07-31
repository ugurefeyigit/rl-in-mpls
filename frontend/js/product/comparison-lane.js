/* The synchronized comparison lane.
 *
 * It renders a verdict only while the backend can prove both runners share one
 * experiment. When the proof fails, the lane names the fields that broke it and
 * shows no comparative claim — a wrong comparison is worse than no comparison.
 *
 * Lanes are told apart by a letter token and a line style as well as colour.
 */

import { el, unavailable } from "./dom.js";
import { signed } from "./format.js";

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
        el("dd", { text: comparison.proof }),
        el("dt", { text: "Fields that disagree" }),
        el("dd", { text: comparison.mismatched_fields.join(", ") || "—" }),
      ]),
    ]);
  }

  const lanes = comparison.lane_details || comparison.lanes || [];
  return el("div", { class: "cmp" }, [
    el("table", { class: "grid" }, [
      el("caption", { text: `Synchronized on ${comparison.scenario}, seed ${comparison.seed}` }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Lane" }),
        el("th", { scope: "col", text: "Action this interval" }),
        el("th", { scope: "col", text: "Interval reward" }),
        el("th", { scope: "col", text: "Cumulative" }),
      ])]),
      el("tbody", {}, lanes.map((lane, index) => el("tr", {
        dataset: { lane: index === 0 ? "a" : "b" },
      }, [
        el("th", { scope: "row" }, [
          el("span", { class: "cmp__token", text: index === 0 ? "A" : "B" }),
          document.createTextNode(` ${lane.algorithm}`),
        ]),
        el("td", { text: actionText(lane) }),
        el("td", { text: signed(lane.last_decision?.reward, 3) }),
        el("td", { text: signed(lane.cumulative_reward, 2) }),
      ]))),
    ]),
    el("p", { class: "cmp__proof",
      text: `${comparison.reason} Proof: ${comparison.proof}.` }),
    el("p", { class: "cmp__proof",
      text: "Returns are signed simulation scores, so no percentage difference is shown." }),
  ]);
}

function actionText(lane) {
  const decision = lane.last_decision;
  if (!decision) return "—";
  if (decision.moves) {
    return decision.moves.length
      ? decision.moves.map((m) => `${m.demand}→p${m.path_idx}`).join(", ")
      : "No TE change";
  }
  if (decision.action === 0) return "No TE change";
  const decoded = decision.decoded || {};
  return `${decoded.demand || decision.action} → path ${decoded.path_idx ?? "?"}`;
}
