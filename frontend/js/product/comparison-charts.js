/* Accessible authored SVG instruments for Exp 2.1.
 * Every plot has visible units, axes, direct A/B labels, keyboard-operable
 * points and a table containing the same values. No curve smoothing invents
 * intermediate values.
 */

import { el, svg, unavailable } from "./dom.js";
import { count, mbps, percent, signed } from "./format.js";

const WIDTH = 960;
const HEIGHT = 320;
const MARGIN = { top: 28, right: 88, bottom: 54, left: 68 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

export function renderLineInstrument({ id, title, note, unit, series, references = [],
                                       domain = null, selectedStep = null, onSelect }) {
  const available = series.filter((item) => item.values?.length);
  if (!available.length) return instrumentUnavailable(title,
    series.find((item) => item.reason)?.reason || "This metric was not recorded.");
  const points = available.flatMap((item) => item.values);
  const xValues = points.map((point) => Number(point.t_min));
  const yValues = points.map((point) => Number(point.value));
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const referenceValues = references.map((reference) => Number(reference.value));
  let yMin = domain?.[0] ?? Math.min(...yValues, ...referenceValues);
  let yMax = domain?.[1] ?? Math.max(...yValues, ...referenceValues);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  if (!domain) {
    const pad = (yMax - yMin) * 0.08;
    yMin -= pad; yMax += pad;
  }
  const x = (value) => MARGIN.left + ((Number(value) - xMin) / Math.max(xMax - xMin, 1)) * PLOT_W;
  const y = (value) => MARGIN.top + (1 - (Number(value) - yMin) / (yMax - yMin)) * PLOT_H;
  const tooltip = el("p", { class: "chart-tooltip", role: "status",
    text: selectedStep === null ? "Focus a point to read its exact value." : `Selected interval · step ${selectedStep}` });
  const plot = svg("svg", { class: "chart-svg", viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    role: "img", "aria-labelledby": `${id}-title ${id}-desc` }, [
    svg("title", { id: `${id}-title`, text: title }),
    svg("desc", { id: `${id}-desc`, text: `${note} Unit: ${unit}. Simulation time in minutes.` }),
  ]);
  drawAxes(plot, { xMin, xMax, yMin, yMax, unit, x, y });
  for (const reference of references) {
    plot.appendChild(svg("line", { class: "chart-reference", x1: MARGIN.left,
      x2: MARGIN.left + PLOT_W, y1: y(reference.value), y2: y(reference.value) }));
    plot.appendChild(svg("text", { class: "chart-reference__label",
      x: MARGIN.left + 6, y: y(reference.value) - 6, text: reference.label }));
  }
  for (const item of available) drawSeries(plot, item, { x, y, unit, tooltip, selectedStep, onSelect });
  return el("section", { class: "chart-instrument", id }, [
    el("div", { class: "chart-instrument__head" }, [
      el("h2", { class: "chart-instrument__title", text: title }),
      el("span", { class: "chart-instrument__unit", text: unit }),
    ]),
    el("p", { class: "chart-instrument__note", text: note }),
    legend(available), plot, tooltip, valuesTable(title, unit, available),
  ]);
}

function drawAxes(plot, { xMin, xMax, yMin, yMax, unit, x, y }) {
  plot.appendChild(svg("line", { class: "chart-axis", x1: MARGIN.left, x2: MARGIN.left,
    y1: MARGIN.top, y2: MARGIN.top + PLOT_H }));
  plot.appendChild(svg("line", { class: "chart-axis", x1: MARGIN.left,
    x2: MARGIN.left + PLOT_W, y1: MARGIN.top + PLOT_H, y2: MARGIN.top + PLOT_H }));
  for (let i = 0; i <= 4; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const py = y(value);
    plot.appendChild(svg("line", { class: "chart-grid", x1: MARGIN.left,
      x2: MARGIN.left + PLOT_W, y1: py, y2: py }));
    plot.appendChild(svg("text", { class: "chart-tick", x: MARGIN.left - 10,
      y: py + 4, "text-anchor": "end", text: formatUnit(unit, value) }));
  }
  for (let i = 0; i <= 4; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / 4;
    plot.appendChild(svg("text", { class: "chart-tick", x: x(value),
      y: MARGIN.top + PLOT_H + 22, "text-anchor": "middle", text: value.toFixed(0) }));
  }
  plot.appendChild(svg("text", { class: "chart-axis__label", x: MARGIN.left + PLOT_W / 2,
    y: HEIGHT - 8, "text-anchor": "middle", text: "Simulation time (minutes)" }));
}

function drawSeries(plot, item, { x, y, unit, tooltip, selectedStep, onSelect }) {
  const path = item.values.map((point, index) =>
    `${index ? "L" : "M"}${x(point.t_min).toFixed(2)} ${y(point.value).toFixed(2)}`).join(" ");
  plot.appendChild(svg("path", { class: `chart-line chart-line--${item.lane} ${item.className || ""}`,
    d: path, fill: "none" }));
  for (const point of item.values) {
    const label = `${item.label}, step ${point.step}, ${point.t_min} minutes: ${formatUnit(unit, point.value)}`;
    const node = marker(item.lane, x(point.t_min), y(point.value), point.step === selectedStep);
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", onSelect ? "button" : "img");
    node.setAttribute("aria-label", label);
    node.appendChild(svg("title", { text: label }));
    const reveal = () => { tooltip.textContent = label; };
    node.addEventListener("mouseenter", reveal);
    node.addEventListener("focus", reveal);
    if (onSelect) {
      node.addEventListener("click", () => onSelect(point.step));
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault(); onSelect(point.step);
        }
      });
    }
    plot.appendChild(node);
  }
  const last = item.values[item.values.length - 1];
  plot.appendChild(svg("text", { class: `chart-direct chart-direct--${item.lane}`,
    x: x(last.t_min) + 10, y: y(last.value) + 4, text: item.direct || item.label }));
}

function marker(lane, x, y, selected) {
  const className = `chart-point chart-point--${lane}${selected ? " chart-point--linked" : ""}`;
  return lane === "a"
    ? svg("circle", { class: className, cx: x, cy: y, r: selected ? 7 : 4 })
    : svg("path", { class: className,
      d: `M${x} ${y - (selected ? 8 : 5)}L${x + (selected ? 8 : 5)} ${y}L${x} ${y + (selected ? 8 : 5)}L${x - (selected ? 8 : 5)} ${y}Z` });
}

function legend(series) {
  return el("ul", { class: "chart-legend", "aria-label": "Series legend" },
    series.map((item) => el("li", { class: `chart-legend__item chart-legend__item--${item.lane}` }, [
      el("span", { class: "chart-legend__mark", "aria-hidden": "true" }),
      el("span", { text: item.label }),
    ])));
}

function valuesTable(title, unit, series) {
  const keyed = new Map();
  for (const item of series) {
    for (const point of item.values) {
      const key = `${point.step}:${point.t_min}`;
      if (!keyed.has(key)) keyed.set(key, { step: point.step, t_min: point.t_min });
      keyed.get(key)[item.key] = point.value;
    }
  }
  const rows = [...keyed.values()].sort((a, b) => a.t_min - b.t_min);
  return el("details", { class: "chart-table" }, [
    el("summary", { text: "Table values" }),
    el("div", { class: "table-scroll" }, [
      el("table", { class: "grid" }, [
        el("caption", { text: `${title}. Same values as the graph; unit: ${unit}.` }),
        el("thead", {}, [el("tr", {}, [
          el("th", { scope: "col", text: "Step" }),
          el("th", { scope: "col", text: "Simulation time (minutes)" }),
          ...series.map((item) => el("th", { scope: "col", text: item.label })),
        ])]),
        el("tbody", {}, rows.map((row) => el("tr", {}, [
          el("th", { scope: "row", text: count(row.step) }),
          el("td", { text: String(row.t_min) }),
          ...series.map((item) => el("td", { text: formatUnit(unit, row[item.key]) })),
        ]))),
      ]),
    ]),
  ]);
}

export function renderDecisionInstrument({ title, a, b, selectedStep, onSelect }) {
  const lanes = [{ lane: "a", run: a }, { lane: "b", run: b }];
  const decisions = lanes.flatMap(({ lane, run }) =>
    (run.series.decisions.values || []).map((value) => ({ ...value, lane })));
  if (!decisions.length) return instrumentUnavailable(title,
    "Per-interval decisions were not recorded for these completed runs.");
  const maxTime = Math.max(...decisions.map((row) => row.t_min), 1);
  const x = (value) => MARGIN.left + (Number(value) / maxTime) * PLOT_W;
  const yLane = { a: 104, b: 202 };
  const tooltip = el("p", { class: "chart-tooltip", role: "status",
    text: selectedStep === null ? "Focus a decision or incident to read its exact value."
      : `Selected interval · step ${selectedStep}` });
  const plot = svg("svg", { class: "chart-svg chart-svg--decisions",
    viewBox: `0 0 ${WIDTH} 280`, role: "img", "aria-label": title });
  for (const lane of lanes) {
    plot.appendChild(svg("line", { class: `decision-rail decision-rail--${lane.lane}`,
      x1: MARGIN.left, x2: MARGIN.left + PLOT_W, y1: yLane[lane.lane], y2: yLane[lane.lane] }));
    plot.appendChild(svg("text", { class: `chart-direct chart-direct--${lane.lane}`,
      x: 18, y: yLane[lane.lane] + 4, text: lane.lane.toUpperCase() }));
  }
  for (const decision of decisions) {
    const py = yLane[decision.lane];
    const node = decision.is_noop
      ? svg("line", { class: `decision-mark decision-mark--noop decision-mark--${decision.lane}`,
          x1: x(decision.t_min) - 3, x2: x(decision.t_min) + 3, y1: py, y2: py })
      : marker(decision.lane, x(decision.t_min), py, decision.step === selectedStep);
    const label = `${decision.lane.toUpperCase()}, step ${decision.step}: ${decision.is_noop ? "no TE change" : decision.accepted ? "accepted TE action" : "rejected request"}`;
    node.setAttribute("tabindex", "0"); node.setAttribute("role", onSelect ? "button" : "img");
    node.setAttribute("aria-label", label); node.appendChild(svg("title", { text: label }));
    const reveal = () => { tooltip.textContent = label; };
    node.addEventListener("mouseenter", reveal); node.addEventListener("focus", reveal);
    if (onSelect) {
      node.addEventListener("click", () => onSelect(decision.step));
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(decision.step); }
      });
    }
    plot.appendChild(node);
  }
  for (const { lane, run } of lanes) {
    for (const event of run.timeline || []) {
      if (!["failure", "recovery", "flap", "reversal"].includes(event.kind)) continue;
      const label = `${lane.toUpperCase()}, step ${event.step}, ${event.t_min} minutes: ${event.title}`;
      const mark = svg("line", { class: `incident-mark incident-mark--${event.kind}`,
        x1: x(event.t_min), x2: x(event.t_min), y1: 64, y2: 238,
        tabindex: "0", role: "img", "aria-label": label });
      mark.appendChild(svg("title", { text: label }));
      const reveal = () => { tooltip.textContent = label; };
      mark.addEventListener("mouseenter", reveal); mark.addEventListener("focus", reveal);
      plot.appendChild(mark);
    }
  }
  plot.appendChild(svg("text", { class: "chart-axis__label", x: MARGIN.left + PLOT_W / 2,
    y: 270, "text-anchor": "middle", text: "Simulation time (minutes)" }));
  const rows = decisions.sort((left, right) => left.step - right.step || left.lane.localeCompare(right.lane));
  const incidentRows = lanes.flatMap(({ lane, run }) => (run.timeline || [])
    .filter((event) => ["failure", "recovery", "flap", "reversal"].includes(event.kind))
    .map((event) => [lane.toUpperCase(), event.step, event.t_min, event.title, "Not applicable"]));
  return el("section", { class: "chart-instrument", id: "decision-timeline" }, [
    el("div", { class: "chart-instrument__head" }, [
      el("h2", { class: "chart-instrument__title", text: title }),
      el("span", { class: "chart-instrument__unit", text: "decision / incident" }),
    ]),
    el("p", { class: "chart-instrument__note",
      text: "Short ticks are no-op stretches. A circles and B diamonds mark requests; vertical rules mark recorded failures, recoveries, reversals and flaps." }),
    legend([{ lane: "a", label: `A · ${a.identity.controller}` },
            { lane: "b", label: `B · ${b.identity.controller}` }]),
    plot, tooltip,
    simpleTable(title, ["Lane", "Step", "Simulation time (minutes)", "Decision", "Moved bandwidth"],
      [...rows.map((row) => [row.lane.toUpperCase(), row.step, row.t_min,
        row.is_noop ? "No TE change" : row.accepted ? "Accepted TE action" : "Rejected request",
        row.moved_mbps === null ? "Unavailable" : mbps(row.moved_mbps)]),
       ...incidentRows].sort((left, right) => left[1] - right[1] || left[0].localeCompare(right[0]))),
  ]);
}

export function renderComponentInstrument({ title, a, b }) {
  const names = [...new Set([...Object.keys(a.reward_components || {}),
    ...Object.keys(b.reward_components || {})])];
  const rows = names.map((name) => ({ name, a: a.reward_components[name]?.total ?? null,
    b: b.reward_components[name]?.total ?? null }));
  const values = rows.flatMap((row) => [row.a, row.b]).filter((value) => value !== null);
  if (!values.length) return instrumentUnavailable(title,
    "The genuine reward components are unavailable for these runs.");
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 0.001);
  const zero = 340;
  const scale = 250 / maxAbs;
  const height = 54 + rows.length * 30;
  const tooltip = el("p", { class: "chart-tooltip", role: "status",
    text: "Focus a component bar to read its exact contribution." });
  const plot = svg("svg", { class: "chart-svg chart-svg--components",
    viewBox: `0 0 960 ${height}`, role: "img", "aria-label": title }, [
    svg("line", { class: "component-zero", x1: zero, x2: zero, y1: 20, y2: height - 20 }),
    svg("text", { class: "chart-reference__label", x: zero + 6, y: 16, text: "Zero reward" }),
  ]);
  rows.forEach((row, index) => {
    const y = 42 + index * 30;
    plot.appendChild(svg("text", { class: "component-label", x: 12, y: y + 5, text: row.name }));
    for (const lane of ["a", "b"]) {
      const value = row[lane];
      if (value === null) continue;
      const width = Math.abs(value) * scale;
      const bar = svg("rect", { class: `component-bar component-bar--${lane}`,
        x: value < 0 ? zero - width : zero, y: y + (lane === "a" ? -9 : 2),
        width, height: 8, tabindex: "0", role: "img",
        "aria-label": `${lane.toUpperCase()} ${row.name}: ${signed(value, 4)} reward contribution` });
      bar.appendChild(svg("title", { text: `${lane.toUpperCase()} ${row.name}: ${signed(value, 4)}` }));
      const reveal = () => { tooltip.textContent = bar.getAttribute("aria-label"); };
      bar.addEventListener("mouseenter", reveal); bar.addEventListener("focus", reveal);
      plot.appendChild(bar);
    }
  });
  return el("section", { class: "chart-instrument", id: "reward-components" }, [
    el("div", { class: "chart-instrument__head" }, [
      el("h2", { class: "chart-instrument__title", text: title }),
      el("span", { class: "chart-instrument__unit", text: "reward contribution" }),
    ]),
    el("p", { class: "chart-instrument__note",
      text: "Diverging totals for every genuine component recorded by each environment; missing terms remain unavailable." }),
    legend([{ lane: "a", label: `A · ${a.identity.controller}` },
            { lane: "b", label: `B · ${b.identity.controller}` }]),
    plot, tooltip,
    simpleTable(title, ["Reward component", "A", "B", "A − B"], rows.map((row) => [
      row.name, signed(row.a, 4), signed(row.b, 4),
      row.a === null || row.b === null ? "Unavailable" : signed(row.a - row.b, 4),
    ])),
  ]);
}

export function renderChurnInstrument({ title, headline }) {
  const counts = headline.filter((row) => ["reroutes", "reversals", "flaps"].includes(row.id));
  const moved = headline.find((row) => row.id === "moved_bandwidth");
  const maxValue = Math.max(...counts.flatMap((row) => [row.a || 0, row.b || 0]), 1);
  const height = 70 + counts.length * 68;
  const tooltip = el("p", { class: "chart-tooltip", role: "status",
    text: "Focus a bar to read its exact count." });
  const plot = svg("svg", { class: "chart-svg chart-svg--churn",
    viewBox: `0 0 960 ${height}`, role: "img", "aria-label": title });
  counts.forEach((row, index) => {
    const y = 48 + index * 68;
    plot.appendChild(svg("text", { class: "component-label", x: 12, y: y + 6, text: row.label }));
    for (const [lane, offset] of [["a", -10], ["b", 6]]) {
      const width = ((row[lane] || 0) / maxValue) * 600;
      const label = `${lane.toUpperCase()} ${row.label}: ${count(row[lane])}`;
      const bar = svg("rect", { class: `churn-bar churn-bar--${lane}`,
        x: 250, y: y + offset, width, height: 12, tabindex: "0", role: "img",
        "aria-label": label });
      bar.appendChild(svg("title", { text: label }));
      const reveal = () => { tooltip.textContent = label; };
      bar.addEventListener("mouseenter", reveal); bar.addEventListener("focus", reveal);
      plot.appendChild(bar);
      plot.appendChild(svg("text", { class: `chart-direct chart-direct--${lane}`,
        x: 260 + width, y: y + offset + 10, text: `${lane.toUpperCase()} ${count(row[lane])}` }));
    }
  });
  return el("section", { class: "chart-instrument", id: "churn-summary" }, [
    el("div", { class: "chart-instrument__head" }, [
      el("h2", { class: "chart-instrument__title", text: title }),
      el("span", { class: "chart-instrument__unit", text: "counts; Mbps separate" }),
    ]),
    el("p", { class: "chart-instrument__note",
      text: "Reroutes, reversals and flaps share a count axis. Moved bandwidth is printed separately so incompatible units never share an axis." }),
    plot, tooltip,
    el("p", { class: "churn-moved", text: moved
      ? `Moved bandwidth · A ${mbps(moved.a)} · B ${mbps(moved.b)} · A − B ${moved.delta === null ? "Unavailable" : `${signed(moved.delta, 0)} Mbps`}`
      : "Moved bandwidth unavailable." }),
    simpleTable(title, ["Measure", "Unit", "A", "B", "A − B"],
      [...counts, ...(moved ? [moved] : [])].map((row) => [row.label, row.unit,
        row.unit === "Mbps" ? mbps(row.a) : count(row.a),
        row.unit === "Mbps" ? mbps(row.b) : count(row.b),
        row.delta === null ? "Unavailable" : row.unit === "Mbps"
          ? `${signed(row.delta, 0)} Mbps` : signed(row.delta, 0)])),
  ]);
}

function simpleTable(caption, headings, rows) {
  return el("details", { class: "chart-table" }, [
    el("summary", { text: "Table values" }),
    el("div", { class: "table-scroll" }, [el("table", { class: "grid" }, [
      el("caption", { text: caption }),
      el("thead", {}, [el("tr", {}, headings.map((heading) =>
        el("th", { scope: "col", text: heading })))]),
      el("tbody", {}, rows.map((row) => el("tr", {}, row.map((value, index) =>
        el(index ? "td" : "th", index ? { text: String(value) }
          : { scope: "row", text: String(value) }))))),
    ])]),
  ]);
}

function instrumentUnavailable(title, reason) {
  return el("section", { class: "chart-instrument" }, [unavailable(title, reason)]);
}

function formatUnit(unit, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "Unavailable";
  if (unit === "percent") return percent(value, 1);
  if (unit === "Mbps") return mbps(value);
  if (unit.includes("demand") || unit.includes("count") || unit.includes("change")) return count(value);
  return signed(value, 3);
}
