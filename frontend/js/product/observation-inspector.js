/* Observation inspector.
 *
 * 604 (or 586) numbers become readable by grouping them the way the schema
 * groups them, attaching the city or demand each index belongs to, and letting
 * the reader search and filter.
 *
 * The ranked list is called "Changed features" everywhere it appears, and the
 * help text next to it says explicitly that a change is not causal importance
 * and not the policy's reasoning. There is no wording in this file that would
 * let a reader conclude otherwise.
 */

import { el, unavailable } from "./dom.js";
import { count, num } from "./format.js";

export const RANKING_TITLE = "Changed features";
export const RANKING_NOTE =
  "Sorted by absolute change between the prior and current observation. " +
  "This is descriptive change, not causal importance, and not the policy's " +
  "internal reasoning.";

export function renderObservationInspector(state, { search, onSearch, changedOnly, onToggleChanged }) {
  const decision = state.data.decision;
  const schema = state.data.schema;

  if (!decision) return unavailable("Observation", "No decision payload has been read.");
  const observation = decision.observation;
  if (!observation.available) return unavailable("Observation", observation.reason);
  if (!schema) return unavailable("Observation schema", "The schema has not loaded yet.");

  const rows = buildRows(observation, schema);
  const needle = (search || "").trim().toLowerCase();
  const filtered = rows.filter((row) => {
    if (changedOnly && (row.delta === null || row.delta === 0)) return false;
    if (!needle) return true;
    return `${row.feature} ${row.objectLabel} ${row.objectId} ${row.index}`
      .toLowerCase().includes(needle);
  });

  return el("div", { class: "obs" }, [
    el("div", { class: "obs__controls" }, [
      el("label", { class: "filters__legend", for: "obs-search", text: "Search features" }),
      el("input", {
        type: "search", id: "obs-search", class: "field",
        placeholder: "Feature, city, demand ID or offset",
        value: search || "",
        onInput: (event) => onSearch(event.target.value),
      }),
      el("button", {
        type: "button", class: "chip",
        "aria-pressed": changedOnly ? "true" : "false",
        onClick: onToggleChanged, text: "Changed only",
      }),
      el("p", { class: "filters__state", role: "status",
        text: `${count(filtered.length)} of ${count(rows.length)} values shown · ` +
              `${observation.changed_count === null ? "no prior observation"
                 : `${count(observation.changed_count)} changed`}` }),
    ]),

    el("div", { class: "table-scroll obs__table" }, [
      el("table", { class: "grid" }, [
        el("caption", { text: `${schema.observation.version} · ` +
          `${count(schema.observation.dim)} values, feature-major` }),
        el("thead", {}, [el("tr", {}, [
          el("th", { scope: "col", text: "Feature" }),
          el("th", { scope: "col", text: "Object" }),
          el("th", { scope: "col", text: "Now" }),
          el("th", { scope: "col", text: "Prior" }),
          el("th", { scope: "col", text: "Δ" }),
          el("th", { scope: "col", text: "Transform" }),
          el("th", { scope: "col", text: "Offset" }),
        ])]),
        el("tbody", {}, filtered.slice(0, 400).map((row) => el("tr", {}, [
          el("th", { scope: "row", text: row.feature }),
          el("td", { text: row.objectLabel }),
          el("td", { text: num(row.value, 4) }),
          el("td", { text: row.prior === null ? "—" : num(row.prior, 4) }),
          el("td", { text: row.delta === null ? "—" : num(row.delta, 4) }),
          el("td", { text: row.transform }),
          el("td", { text: String(row.index) }),
        ]))),
      ]),
    ]),
    filtered.length > 400
      ? el("p", { class: "filters__state",
          text: `Showing the first 400 of ${count(filtered.length)} matches. ` +
                `Narrow the search to see the rest.` })
      : null,

    el("section", { class: "panel" }, [
      el("h3", { class: "panel__title", text: RANKING_TITLE }),
      observation.changed_feature_ranking.length
        ? el("div", { class: "table-scroll" }, [
            el("table", { class: "grid" }, [
              el("caption", { text: RANKING_NOTE }),
              el("thead", {}, [el("tr", {}, [
                el("th", { scope: "col", text: "Feature" }),
                el("th", { scope: "col", text: "Object" }),
                el("th", { scope: "col", text: "Prior" }),
                el("th", { scope: "col", text: "Now" }),
                el("th", { scope: "col", text: "Δ" }),
              ])]),
              el("tbody", {}, observation.changed_feature_ranking.slice(0, 20)
                .map((entry) => {
                  const row = rows[entry.index] || {};
                  return el("tr", {}, [
                    el("th", { scope: "row", text: row.feature || `index ${entry.index}` }),
                    el("td", { text: row.objectLabel || "—" }),
                    el("td", { text: num(entry.prior, 4) }),
                    el("td", { text: num(entry.current, 4) }),
                    el("td", { text: num(entry.delta, 4) }),
                  ]);
                })),
            ]),
          ])
        : el("p", { class: "tb-empty", text: observation.prior_reason
            || "Nothing changed between the prior and current observation." }),
    ]),
  ]);
}

function buildRows(observation, schema) {
  const axes = schema.axes;
  const rows = new Array(observation.values.length);
  for (const group of schema.observation.groups) {
    for (let offset = group.start; offset < group.end; offset += 1) {
      const position = offset - group.start;
      const axis = group.axis === "dlink" ? axes.dlink
        : (group.axis === "demand" ? axes.demand : null);
      const object = axis ? axis[position] : null;
      rows[offset] = {
        index: offset,
        feature: group.feature,
        group: group.group,
        transform: group.transform_text,
        meaning: group.meaning,
        objectLabel: object ? `${object.label} · ${object.id}` : "global",
        objectId: object ? object.id : "global",
        value: observation.values[offset],
        prior: observation.prior_values ? observation.prior_values[offset] : null,
        delta: observation.prior_values
          ? Number((observation.values[offset] - observation.prior_values[offset]).toFixed(6))
          : null,
      };
    }
  }
  return rows;
}
