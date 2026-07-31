/* The governed study.
 *
 * Final-holdout and development evidence live in separate regions with separate
 * headings, and no chart or aggregate is shared between them. The final result
 * leads with the negative planning finding, because a result that leads with its
 * win and buries its limit is a marketing page, not evidence.
 *
 * Every number arrives from `/api/v2/*`. Nothing scientific is written in this
 * file — a literal here would drift from the frozen record silently.
 */

import { el, tag, unavailable } from "./dom.js";
import { count, num, percent, shortHash, signed } from "./format.js";

export function renderGovernedStudy(state) {
  const evidence = state.data.evidence;
  const outage = evidence.error;
  if (outage) {
    return unavailable("Evidence unavailable", outage);
  }
  if (!evidence.study) {
    return el("p", { class: "tb-empty", text: "Loading the frozen study record…" });
  }

  return el("div", { class: "study" }, [
    finalRegion(state, evidence),
    developmentRegion(state, evidence),
    disclosuresRegion(evidence),
  ]);
}

/* ------------------------------------------------------------ final evidence */
function finalRegion(state, evidence) {
  const holdout = evidence.finalHoldout;
  const scenarios = evidence.finalScenarios;
  const actions = evidence.finalActions;
  const integrity = evidence.finalIntegrity;
  const contracts = state.data.contracts;

  return el("section", { class: "region region--final", "aria-labelledby": "final-heading" }, [
    el("header", { class: "region__head" }, [
      el("div", { class: "stamp", dataset: { kind: "final_holdout_evidence" } }, [
        el("span", { class: "stamp__word", text: "FINAL EVIDENCE" }),
      ]),
      el("h2", { class: "region__title", id: "final-heading",
        text: "One-shot final holdout · seeds 1001–1005" }),
      el("p", { class: "region__sub",
        text: "Evaluated once, after the study closed to selection. Nothing was " +
              "tuned, reselected or retried on these seeds." }),
    ]),

    contracts
      ? el("ul", { class: "findings" }, contracts.final_findings.map((finding) =>
          el("li", { class: "findings__item", text: finding })))
      : null,

    holdout ? holdoutTable(holdout) : loading(),
    scenarios ? scenarioTable(scenarios) : loading(),
    actions ? actionRegion(actions, state) : loading(),
    integrity ? integrityRegion(integrity) : loading(),
  ]);
}

function loading() {
  return el("p", { class: "tb-empty", text: "Reading the frozen artifact…" });
}

function holdoutTable(holdout) {
  const rows = holdout.policies || holdout.results || holdout.returns || [];
  const list = Array.isArray(rows) ? rows : Object.entries(rows).map(
    ([id, value]) => (typeof value === "object" ? { id, ...value } : { id, value }));

  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid" }, [
      el("caption", { text: holdout.grain
        || "Root-averaged: each learner value is the mean of three training-root " +
           "means, each over five holdout seeds." }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Method" }),
        ...columnsOf(list).map((key) => el("th", { scope: "col", text: label(key) })),
      ])]),
      el("tbody", {}, list.map((row) => el("tr", {}, [
        el("th", { scope: "row", text: row.id || row.algorithm || row.policy_id }),
        ...columnsOf(list).map((key) => el("td", { text: cell(row[key]) })),
      ]))),
    ]),
  ]);
}

function scenarioTable(payload) {
  const scenarios = payload.scenarios || {};
  const names = Object.keys(scenarios);
  if (!names.length) return unavailable("Scenarios", "The artifact reports no scenarios.");
  const methods = Object.keys(scenarios[names[0]] || {})
    .filter((k) => typeof scenarios[names[0]][k] === "number");

  return el("div", { class: "table-scroll" }, [
    el("table", { class: "grid" }, [
      el("caption", { text: `${payload.grain} · source ${shortHash(payload.source_sha)}` }),
      el("thead", {}, [el("tr", {}, [
        el("th", { scope: "col", text: "Scenario" }),
        ...methods.map((m) => el("th", { scope: "col", text: label(m) })),
      ])]),
      el("tbody", {}, names.map((name) => el("tr", {}, [
        el("th", { scope: "row", text: name }),
        ...methods.map((m) => el("td", { text: cell(scenarios[name][m]) })),
      ]))),
    ]),
  ]);
}

function actionRegion(actions, state) {
  const noop = actions.noop || {};
  const grains = state.data.contracts?.noop_metrics || {};

  // Each no-op grain gets its own block with its own denominator printed. They
  // are never placed in one column where a reader could average them.
  const blocks = Object.entries(noop).map(([grain, value]) => {
    const meta = grains[grain];
    const rows = (value && typeof value === "object")
      ? Object.entries(value)
      : [[grain, value]];
    return el("div", { class: "panel" }, [
      el("h4", { class: "panel__title", text: meta?.label || label(grain) }),
      el("p", { class: "rew__note",
        text: meta ? `Denominator: ${meta.denominator}. ${meta.description}`
                   : "Denominator not declared for this measure." }),
      el("div", { class: "table-scroll" }, [
        el("table", { class: "grid" }, [
          el("thead", {}, [el("tr", {}, [
            el("th", { scope: "col", text: "Method" }),
            el("th", { scope: "col", text: "Value" }),
          ])]),
          el("tbody", {}, rows.map(([key, entry]) => el("tr", {}, [
            el("th", { scope: "row", text: label(key) }),
            el("td", { text: cell(entry) }),
          ]))),
        ]),
      ]),
    ]);
  });

  return el("div", { class: "panel" }, [
    el("h3", { class: "panel__title", text: "Action and no-op distribution" }),
    ...blocks,
    el("p", { class: "rew__note",
      text: `Distribution source: ${shortHash(actions.source_sha)}` }),
  ]);
}

function integrityRegion(integrity) {
  return el("div", { class: "panel" }, [
    el("h3", { class: "panel__title", text: "Safety and integrity" }),
    el("dl", { class: "facts" }, Object.entries(integrity)
      .filter(([, value]) => typeof value !== "object")
      .flatMap(([key, value]) => [
        el("dt", { text: label(key) }),
        el("dd", { text: cell(value) }),
      ])),
  ]);
}

/* -------------------------------------------------------------- development */
function developmentRegion(state, evidence) {
  const continuity = evidence.developmentContinuity;
  return el("section", { class: "region region--development",
                         "aria-labelledby": "development-heading" }, [
    el("header", { class: "region__head" }, [
      el("div", { class: "stamp", dataset: { kind: "development_evidence" } }, [
        el("span", { class: "stamp__word", text: "DEVELOPMENT — NOT HOLDOUT" }),
      ]),
      el("h2", { class: "region__title", id: "development-heading",
        text: "Selection-stage evidence · seeds 101–105" }),
      el("p", { class: "region__sub",
        text: "The seed-42 pilot and the three-root continuity runs. Checkpoint " +
              "selection happened here, so these numbers cannot support a holdout " +
              "claim and never share a region with final evidence." }),
    ]),
    continuity
      ? el("div", { class: "table-scroll" }, [
          el("table", { class: "grid" }, [
            el("caption", { text: `Continuity summary · source ` +
              `${shortHash(continuity.source_sha)}` }),
            el("thead", {}, [el("tr", {}, [
              el("th", { scope: "col", text: "Measure" }),
              el("th", { scope: "col", text: "Value" }),
            ])]),
            el("tbody", {}, Object.entries(continuity.summary || {})
              .filter(([, value]) => typeof value !== "object")
              .map(([key, value]) => el("tr", {}, [
                el("th", { scope: "row", text: label(key) }),
                el("td", { text: cell(value) }),
              ]))),
          ]),
        ])
      : loading(),
  ]);
}

function disclosuresRegion(evidence) {
  const disclosures = evidence.disclosures;
  if (!disclosures) return null;
  const rows = disclosures.disclosures || [];
  return el("section", { class: "region" }, [
    el("h2", { class: "region__title", text: "Invalidated, superseded and repaired runs" }),
    rows.length
      ? el("div", { class: "table-scroll" }, [
          el("table", { class: "grid" }, [
            el("caption", { text: "Disclosed because a closed study that hides its " +
              "discarded runs is not a closed study." }),
            el("thead", {}, [el("tr", {}, [
              el("th", { scope: "col", text: "Run" }),
              el("th", { scope: "col", text: "Kind" }),
              el("th", { scope: "col", text: "Detail" }),
            ])]),
            el("tbody", {}, rows.map((row) => el("tr", {}, [
              el("th", { scope: "row", text: row.id || row.run || "—" }),
              el("td", { text: row.kind || "—" }),
              el("td", { text: row.reason || row.detail || "—" }),
            ]))),
          ]),
        ])
      : el("p", { class: "tb-empty", text: "No run was invalidated, superseded or repaired." }),
  ]);
}

/* ------------------------------------------------------------ conclusion */
export function renderConclusion(state) {
  const contracts = state.data.contracts;
  const findings = contracts?.final_findings || [];
  return [
    el("p", { class: "prose",
      text: "This is a frozen record, opened over a live run. It is not part of " +
            "the session above and it is never used as a live comparator." }),
    el("ul", { class: "findings" }, findings.map((finding) =>
      el("li", { class: "findings__item", text: finding }))),
    el("p", { class: "prose",
      text: "Open RL Information → Governed Study for the full record, including " +
            "per-scenario results, integrity and provenance." }),
  ];
}

/* ------------------------------------------------------------------ helpers */
function columnsOf(list) {
  const keys = new Set();
  for (const row of list) {
    for (const [key, value] of Object.entries(row)) {
      if (key === "id" || key === "algorithm" || key === "policy_id") continue;
      if (typeof value === "number" || typeof value === "string") keys.add(key);
    }
  }
  return [...keys].slice(0, 6);
}

function label(key) {
  return String(key).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

function cell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Math.abs(value) < 1 && value !== 0 && Number.isFinite(value)) return num(value, 4);
    return num(value, 3);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

export { label as prettyLabel, cell as prettyCell };
