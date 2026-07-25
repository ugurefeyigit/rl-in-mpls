// Presentation label registry, client side.
//
// The server owns the mapping (mplssim/display.py, served at GET /api/display).
// This module caches it once at boot and turns internal IDs into the labels the
// audience sees. Internal IDs (PE1, P5, L11, D2, scenario keys) are NEVER
// renamed — they are the contract with the pretrained model, the configs, the
// tests and the committed result files. City names are display only.

const EMPTY = { cities: {}, scenarios: {}, classes: {}, links: {},
                disclaimer: "", glossary: {} };

let reg = EMPTY;

export async function loadDisplay() {
  const r = await fetch("/api/display");
  if (!r.ok) throw new Error(`/api/display returned ${r.status}`);
  reg = { ...EMPTY, ...(await r.json()) };
  return reg;
}

export const registry = () => reg;
export const disclaimer = () => reg.disclaimer;
export const glossary = () => reg.glossary;

/** "PE1" -> "İstanbul" (falls back to the raw ID if unmapped). */
export const city = (routerId) => reg.cities[routerId] || routerId;

/** ["PE1","P1","P5"] -> "İstanbul → Eskişehir → Kayseri". */
export const pathLabel = (routers) =>
  (routers || []).map(city).join(" → ");

/** Short chain for tight spaces: "İstanbul → … → Erzurum". */
export function pathShort(routers, maxHops = 4) {
  const p = routers || [];
  if (p.length <= maxHops) return pathLabel(p);
  return `${city(p[0])} → … → ${city(p[p.length - 1])}`;
}

/**
 * The transit cities only — "Eskişehir → Kayseri → Sivas". Two candidate
 * routes for the same demand share their endpoints, so the transit chain is
 * what actually distinguishes them in one line of the decision tape.
 */
export function pathVia(routers) {
  const mid = (routers || []).slice(1, -1);
  return mid.length ? mid.map(city).join(" → ") : "a direct link";
}

/** "L11" -> "Ankara–Kayseri link". */
export const linkLabel = (linkId) =>
  (reg.links[linkId] && reg.links[linkId].label) || linkId;

/** "L11" -> "P2–P5, L11" — the technical detail line, never the headline. */
export const linkTechnical = (linkId) =>
  (reg.links[linkId] && reg.links[linkId].technical) || linkId;

/** "Ankara–Kayseri link (P2–P5, L11)" for engineer-facing dropdowns. */
export const linkFull = (linkId) => `${linkLabel(linkId)} (${linkTechnical(linkId)})`;

export const className = (cls) => reg.classes[cls] || cls;

/** "İstanbul → Erzurum video traffic" — the plain-language demand name. */
export const demandLabel = (src, dst, cls) =>
  `${city(src)} → ${city(dst)} ${className(cls)} traffic`;

/** Same, with the internal ID appended for engineer-facing dropdowns. */
export const demandFull = (d) =>
  `${demandLabel(d.src, d.dst, d.class ?? d.cls)} (${d.id})`;

/** Scenario key -> "Guided Operator Demonstration". */
export const scenarioLabel = (key) => reg.scenarios[key] || key;

/** Human name for a controller, used in scoreboards and story text. */
const ALGO_LABELS = {
  rl: "AI Advisor", static: "Fixed routing", greedy: "Traditional controller",
  cspf: "CSPF re-optimizer", random: "Random baseline",
};
export const algoLabel = (a) => ALGO_LABELS[a] || a;

/** Technical name, for the engineering console. */
const ALGO_TECH = {
  rl: "RL (MaskablePPO)", static: "static shortest path", greedy: "greedy (util-aware)",
  cspf: "CSPF periodic reopt", random: "random floor",
};
export const algoTech = (a) => ALGO_TECH[a] || a;

/**
 * Resolve the undirected link ID joining two adjacent routers, using the
 * technical field ("P2–P5, L11") the server already sends. Used to name a
 * congested hop in story text without a second topology lookup.
 */
export function linkBetween(a, z) {
  for (const [id, info] of Object.entries(reg.links)) {
    const t = String(info.technical || "");
    if (t.startsWith(`${a}–${z},`) || t.startsWith(`${z}–${a},`)) return id;
  }
  return null;
}
