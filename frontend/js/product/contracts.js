/* Client mirror of the product contracts.
 *
 * The backend is the authority (`mplssim/product/contracts.py`); these constants
 * exist so the shell can render a source stamp before the first fetch resolves,
 * and so a component that meets an unknown source kind fails loudly instead of
 * defaulting to "live".
 */

export const MODES = [
  { id: "presentation", label: "Presentation", shortcut: "Alt+1" },
  { id: "network", label: "Network Information", shortcut: "Alt+2" },
  { id: "rl", label: "RL Information", shortcut: "Alt+3" },
];

export const MODE_IDS = MODES.map((m) => m.id);

/** Guided Story is a Presentation workflow. It is never a fourth mode. */
export const WORKFLOWS = [
  { id: "guided-story", mode: "presentation", label: "Guided Story" },
];

export const RL_VIEWS = ["decision", "study", "provenance"];

export const SOURCE_KINDS = {
  live_session: {
    label: "LIVE", icon: "live", executes: true, linkTelemetry: true,
    short: "Running simulation",
  },
  recorded_replay: {
    label: "RECORDED", icon: "recorded", executes: false, linkTelemetry: false,
    short: "Immutable recorded trace",
  },
  development_evidence: {
    label: "DEVELOPMENT", icon: "development", executes: false, linkTelemetry: false,
    short: "Selection-stage evidence",
  },
  final_holdout_evidence: {
    label: "FINAL EVIDENCE", icon: "final-evidence", executes: false, linkTelemetry: false,
    short: "Frozen one-shot holdout",
  },
};

export function sourceProfile(kind) {
  const profile = SOURCE_KINDS[kind];
  if (!profile) throw new Error(`unknown source kind: ${kind}`);
  return profile;
}

export const ROUTES = {
  "/": { mode: "network", source: "live_session" },
  "/advanced": { mode: "network", source: "live_session" },
  "/present": { mode: "presentation", source: "live_session" },
  "/study": { mode: "rl", source: "final_holdout_evidence", rlView: "study" },
};

export const MODE_ROUTE = {
  presentation: "/present",
  network: "/advanced",
  rl: "/study",
};

export const KEYS = [
  ["Alt+1 / Alt+2 / Alt+3", "Switch to Presentation, Network Information or RL Information"],
  ["Space", "Play or pause, unless focus is in a form control"],
  ["→", "Next step, or next story beat while Guided Story runs"],
  ["←", "Previous story beat. A live engine is never rewound"],
  ["G", "Open Guided Story from Presentation"],
  ["E", "Explain this moment"],
  ["/", "Focus search in Network or RL Information"],
  ["[ / ]", "Previous or next incident bookmark"],
  ["? ", "Show this list"],
  ["Esc", "Close the top drawer, then leave fullscreen"],
  ["Arrow keys", "Move between cities while the topology has focus"],
  ["Enter", "Open the focused object's inspector"],
];
