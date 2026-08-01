/* The one control panel.
 *
 * Everything needed to configure, start and drive a run lives here, in the
 * order a first-time user needs it: what world, what situation, what seed, who
 * decides, which controller, how fast, then go. Nothing that starts or steers a
 * run lives in the header, a bottom bar or a drawer any more — a control the
 * user cannot find is a control that does not exist.
 *
 * Two rules this module keeps:
 *
 * - a controller with no verified checkpoint is shown as *disabled with the
 *   verification reason*, never hidden and never quietly swapped;
 * - study evidence is a separate, clearly named region. It is a finished record,
 *   not a live simulation choice, so it can never sit beside the model picker.
 */

import { $, el, fill } from "./dom.js";

const SPEEDS = [
  ["1x", "1x · one control interval every 2 s"],
  ["5x", "5x"],
  ["20x", "20x"],
  ["fast", "As fast as possible"],
];

const EXECUTIONS = [
  ["automatic", "Automatic",
   "The controller acts on its own. Each completed decision is explained "
   + "afterwards; there is nothing to approve."],
  ["advisor", "Manual · advisor approval",
   "The run pauses before each proposed action. You approve it or reject it, "
   + "and rejecting applies no TE change."],
];

export function renderControlPanel(state, handlers) {
  const host = $("control-panel");
  const show = state.mode === "presentation" && !state.ui.audienceView;
  host.hidden = !show;
  if (!show) return;

  const session = state.data.snapshot?.session || null;
  const live = state.source.kind === "live_session";

  fill(host, [
    setupSection(state, handlers, session, live),
    runSection(state, handlers, session, live),
    decisionSection(state, handlers, session),
    storySection(state, handlers),
    resultsSection(state, handlers, session),
    evidenceSection(state, handlers),
  ]);
}

/* ------------------------------------------------------------- 1..7 setup */
function setupSection(state, handlers, session, live) {
  const setup = state.setup;
  const capabilities = state.data.capabilities;
  const running = Boolean(session);
  const environments = capabilities?.environments || [];
  const scenarios = state.data.scenarios || {};
  const policies = (capabilities?.live_policies || [])
    .filter((policy) => policy.environment_version === setup.environment);
  const registry = capabilities?.checkpoint_registry;

  const seedProblem = seedError(setup.seed, capabilities);
  const primary = primaryPolicy(policies, setup.policyA);
  const secondary = setup.compare ? primaryPolicy(policies, setup.policyB) : null;
  const blocked = seedProblem
    || (primary && !primary.available ? primary.unavailable_reason : null)
    || (secondary && !secondary.available ? secondary.unavailable_reason : null);

  return el("section", { class: "cp__section", "aria-labelledby": "cp-setup-title" }, [
    el("h2", { class: "cp__title", id: "cp-setup-title", text: "1 · Set up the run" }),

    field("Environment", "cp-environment",
      el("select", {
        id: "cp-environment", disabled: running,
        onChange: (event) => handlers.onSetup({ environment: event.target.value }),
      }, environments.map((environment) => el("option", {
        value: environment.version,
        selected: environment.version === setup.environment,
        text: `${environment.label}${environment.is_default ? " · default" : ""}`
          + ` · ${environment.observation_dim}-value observation`,
      }))),
      environments.find((e) => e.version === setup.environment)?.summary),

    field("Scenario", "cp-scenario",
      el("select", {
        id: "cp-scenario", disabled: running,
        onChange: (event) => handlers.onSetup({ scenario: event.target.value }),
      }, Object.entries(scenarios).map(([id, scenario]) => el("option", {
        value: id, selected: id === setup.scenario,
        text: scenario.display_name || id,
      }))),
      scenarios[setup.scenario]?.description),

    field("Seed", "cp-seed",
      el("input", {
        id: "cp-seed", type: "number", min: "0", step: "1",
        value: String(setup.seed), disabled: running,
        "aria-invalid": seedProblem ? "true" : "false",
        "aria-describedby": "cp-seed-note",
        onInput: (event) => handlers.onSetup({ seed: event.target.value }),
      }),
      seedProblem
        || "Any whole number. 42 is the repository's usual demonstration seed. "
           + "The same seed reproduces the same traffic and the same events.",
      { id: "cp-seed-note", invalid: Boolean(seedProblem) }),

    field("Execution", "cp-execution",
      el("div", { class: "cp__choices", role: "radiogroup",
                  "aria-label": "Execution style" },
        EXECUTIONS.map(([id, label, note]) => el("label", { class: "cp__choice" }, [
          el("input", {
            type: "radio", name: "cp-execution", value: id,
            checked: setup.execution === id, disabled: running,
            onChange: () => handlers.onSetup({ execution: id }),
          }),
          el("span", { class: "cp__choice-label", text: label }),
          el("span", { class: "cp__choice-note", text: note }),
        ]))),
      null),

    field("Controller A", "cp-policy-a",
      el("select", {
        id: "cp-policy-a", disabled: running,
        onChange: (event) => handlers.onSetup({ policyA: event.target.value }),
      }, policies.map((policy) => policyOption(policy, setup.policyA))),
      policyNote(primary)),

    el("div", { class: "cp__field" }, [
      el("label", { class: "cp__checkbox" }, [
        el("input", {
          type: "checkbox", id: "cp-compare", checked: setup.compare,
          disabled: running,
          onChange: (event) => handlers.onSetup({ compare: event.target.checked }),
        }),
        el("span", { text: "Compare two controllers on the same run" }),
      ]),
      setup.compare
        ? el("select", {
            id: "cp-policy-b", "aria-label": "Controller B", disabled: running,
            onChange: (event) => handlers.onSetup({ policyB: event.target.value }),
          }, policies.map((policy) => policyOption(policy, setup.policyB)))
        : null,
      setup.compare
        ? el("p", { class: "cp__note", text: policyNote(secondary)
            || "Both lanes get the same scenario, seed, starting state, traffic, "
               + "failures and interventions. If that cannot be proved, no "
               + "comparison is shown." })
        : null,
    ]),

    setup.environment === "v2" && registry
      ? field("Checkpoint root", "cp-root",
          el("select", {
            id: "cp-root", disabled: running,
            onChange: (event) =>
              handlers.onSetup({ trainingRoot: Number(event.target.value) }),
          }, (registry.training_roots || []).map((root) => el("option", {
            value: String(root), selected: Number(root) === Number(setup.trainingRoot),
            text: `Root ${root}${root === registry.default_training_root ? " · default" : ""}`,
          }))),
          registry.default_training_root_rule)
      : null,

    field("Speed", "cp-speed",
      el("select", {
        id: "cp-speed",
        onChange: (event) => handlers.onSpeed(event.target.value),
      }, SPEEDS.map(([id, label]) => el("option", {
        value: id, selected: id === (session?.speed || setup.speed), text: label,
      }))),
      "Presentation pacing, not real time."),

    el("button", {
      type: "button", class: "cp__primary", id: "cp-start",
      disabled: Boolean(blocked),
      onClick: handlers.onStart,
      text: running ? "Start a new run with these settings" : "Start run",
    }),
    blocked ? el("p", { class: "cp__note cp__note--blocked", role: "status",
                        text: blocked }) : null,
    live ? null : el("p", { class: "cp__note",
      text: "You are looking at a stored record. Starting a run switches back to "
            + "the live simulation." }),
  ]);
}

function policyOption(policy, selected) {
  return el("option", {
    value: policy.id,
    selected: policy.id === selected,
    disabled: !policy.available,
    text: policy.available ? policy.label : `${policy.label} — unavailable`,
  });
}

function policyNote(policy) {
  if (!policy) return null;
  if (!policy.available) return policy.unavailable_reason;
  return `${policy.description} ${policy.output_description || ""}`.trim();
}

function primaryPolicy(policies, id) {
  return policies.find((policy) => policy.id === id) || null;
}

function seedError(seed, capabilities) {
  const value = Number(seed);
  if (seed === "" || seed === null || !Number.isFinite(value)) {
    return "Enter a whole number for the seed.";
  }
  if (!Number.isInteger(value) || value < 0) {
    return "The seed must be a whole number of 0 or more.";
  }
  const blocked = capabilities?.holdout_seeds_blocked_for_live || [];
  if (blocked.includes(value)) {
    return `Seed ${value} is one of the frozen final-holdout seeds. Those are `
      + `reserved for the closed study and cannot be run live.`;
  }
  return null;
}

/* ------------------------------------------------------------- 8 transport */
function runSection(state, handlers, session, live) {
  const running = Boolean(session?.running);
  const awaiting = Boolean(session?.awaiting_decision);
  return el("section", { class: "cp__section", "aria-labelledby": "cp-run-title" }, [
    el("h2", { class: "cp__title", id: "cp-run-title", text: "2 · Run it" }),
    el("div", { class: "cp__transport", role: "group", "aria-label": "Playback" }, [
      el("button", {
        type: "button", class: "cp__primary", id: "btn-playpause",
        disabled: !live || awaiting,
        onClick: handlers.onPlayPause,
        text: !session ? "Start run" : (running ? "Pause" : "Resume"),
      }),
      el("button", { type: "button", class: "ctl", id: "btn-step",
                     disabled: !live || awaiting, onClick: handlers.onStep,
                     text: "Step once" }),
      el("button", { type: "button", class: "ctl", id: "btn-next-event",
                     disabled: !live || !session || awaiting,
                     onClick: handlers.onNextEvent, text: "Skip to next event" }),
    ]),
    el("div", { class: "cp__transport", role: "group", "aria-label": "Reset" }, [
      el("button", { type: "button", class: "ctl", id: "btn-stop",
                     disabled: !session || !running, onClick: handlers.onPause,
                     text: "Stop" }),
      el("button", { type: "button", class: "ctl", id: "btn-reset-run",
                     disabled: !session, onClick: handlers.onResetRun,
                     text: "Reset run" }),
      el("button", { type: "button", class: "ctl ctl--danger", id: "btn-full-reset",
                     onClick: handlers.onFullReset, text: "Full reset" }),
    ]),
    el("p", { class: "cp__note",
      text: "Reset run puts the same scenario, seed and controllers back at step "
            + "zero and keeps the run it replaces. Full reset stops everything and "
            + "returns to this configuration. Neither changes a model or any "
            + "study result." }),
    session
      ? el("p", { class: "cp__status", role: "status",
          text: `${statusWord(session)} · step ${session.step} · `
                + `${session.retained_runs || 0} earlier run(s) kept` })
      : el("p", { class: "cp__status", role: "status",
                  text: "No run yet. Press Start run." }),
  ]);
}

function statusWord(session) {
  if (session.awaiting_decision) return "Waiting for your decision";
  if (session.running) return "Running";
  if (session.done) return "Scenario finished";
  return { idle: "Ready", paused: "Paused", completed: "Finished",
           error: "Error" }[session.state] || session.state;
}

/* ------------------------------------------------------------- 8b decision */
function decisionSection(state, handlers, session) {
  if (!session) return null;
  const advisor = state.data.advisor;
  const pending = Boolean(session.awaiting_decision);
  if (session.execution !== "advisor") {
    return el("section", { class: "cp__section" }, [
      el("h2", { class: "cp__title", text: "3 · Decisions" }),
      el("p", { class: "cp__note",
        text: advisor?.note
          || "The controller acts automatically. The card under the map explains "
             + "the decision it already made; there is nothing to approve." }),
    ]);
  }
  return el("section", { class: "cp__section" }, [
    el("h2", { class: "cp__title", text: "3 · Approve or reject" }),
    el("div", { class: "cp__transport", role: "group", "aria-label": "Decision" }, [
      el("button", { type: "button", class: "ctl ctl--accept", id: "btn-approve",
                     disabled: !pending, onClick: handlers.onApprove,
                     text: "Approve" }),
      el("button", { type: "button", class: "ctl", id: "btn-reject",
                     disabled: !pending, onClick: handlers.onReject,
                     text: "Reject · no TE change" }),
    ]),
    el("p", { class: "cp__note",
      text: pending
        ? "The proposed action is held. Nothing has been applied yet."
        : "Step or resume; the run pauses at the next proposed action." }),
    // The one place per-action approval does not hold. It is stated here, next
    // to the approval controls, rather than only in an API response.
    el("p", { class: "cp__note",
      text: "Skip to next event is the exception: it applies the controller's "
            + "own actions for a stretch of intervals in one gesture. It asks "
            + "you to delegate that stretch first, and records it as one "
            + "delegated batch." }),
    advisor?.delegated_intervals
      ? el("p", { class: "cp__note cp__note--blocked", role: "status",
          text: `${advisor.delegated_intervals} interval(s) in this run were `
                + `delegated, not approved individually.` })
      : null,
  ]);
}

/* --------------------------------------------------------------- 9 story */
function storySection(state, handlers) {
  const story = state.story;
  return el("section", { class: "cp__section", "aria-labelledby": "cp-story-title" }, [
    el("h2", { class: "cp__title", id: "cp-story-title", text: "4 · Guided Story" }),
    el("button", { type: "button", class: "ctl", id: "btn-story-toggle",
                   onClick: handlers.onToggleStory,
                   text: story.active ? "End Guided Story" : "Start Guided Story" }),
    story.active
      ? el("div", { class: "cp__transport", role: "group",
                    "aria-label": "Story pacing" }, [
          el("button", { type: "button", class: "ctl", id: "btn-story-prev",
                         onClick: handlers.onStoryPrevious, text: "Previous" }),
          el("button", { type: "button", class: "ctl", id: "btn-story-next",
                         onClick: handlers.onStoryNext, text: "Next" }),
          el("button", { type: "button", class: "ctl", id: "btn-story-auto",
                         "aria-pressed": story.auto ? "true" : "false",
                         onClick: handlers.onToggleStoryAuto,
                         text: story.auto ? "Pause automatic" : "Play automatically" }),
          el("button", { type: "button", class: "ctl", id: "btn-story-restart",
                         onClick: handlers.onStoryRestart, text: "Restart" }),
        ])
      : null,
    el("p", { class: "cp__status", id: "story-progress", role: "status",
              text: story.active ? `Beat ${(story.reviewBeat ?? story.beat) + 1} of 11`
                                 : "Guided Story is not running." }),
    story.active && story.auto
      ? el("p", { class: "cp__note",
          text: "Automatic playback stops at every recommendation and waits for "
                + "Approve or Reject before continuing." })
      : null,
  ]);
}

/* -------------------------------------------------------------- 5 results */
function resultsSection(state, handlers, session) {
  const results = state.data.results;
  const retained = results?.retained;
  const saved = state.savedRun;
  return el("section", { class: "cp__section", "aria-labelledby": "cp-results-title" }, [
    el("h2", { class: "cp__title", id: "cp-results-title", text: "5 · Results" }),
    el("p", { class: "cp__note",
      text: "The live run, runs kept by Reset run, and the closed study are three "
            + "separate records. They are reported side by side and never "
            + "averaged together." }),
    el("div", { class: "cp__transport", role: "group", "aria-label": "Results" }, [
      el("button", { type: "button", class: "ctl", id: "btn-results-refresh",
                     onClick: handlers.onLoadResults, text: "Refresh results" }),
      el("button", { type: "button", class: "ctl", id: "btn-save-run",
                     disabled: !session || !session.step,
                     onClick: handlers.onSaveRun, text: "Save this run" }),
    ]),
    el("p", { class: "cp__status", id: "results-status", role: "status",
      text: retained
        ? `${retained.session_count} kept in this session · `
          + `${retained.process_count} kept from earlier sessions`
        : "Results have not been read yet." }),
    saved?.saved_run_ids?.length
      ? el("p", { class: "cp__note",
          text: `Saved as run ${saved.saved_run_ids.join(", ")} under the `
                + `${String(saved.environment).toUpperCase()} summary shape. A `
                + `saved live run is a demonstration record, not evidence.` })
      : null,
    el("p", { class: "cp__note",
      text: retained?.lifetime
        || "Reset run archives the run it replaces. Full reset keeps it for this "
           + "server process. A restart drops it: a demonstration number is "
           + "never persisted as a result." }),
  ]);
}

/* ---------------------------------------------------------- study evidence */
function evidenceSection(state, handlers) {
  const sources = (state.data.capabilities?.sources || [])
    .filter((source) => source.group === "study_evidence");
  return el("section", { class: "cp__section cp__section--evidence",
                         "aria-labelledby": "cp-evidence-title" }, [
    el("h2", { class: "cp__title", id: "cp-evidence-title",
               text: "6 · Study evidence and results" }),
    el("p", { class: "cp__note",
      text: "Finished, read-only records of the closed study. "
            + "These are not simulation settings: none of them can be run, "
            + "compared live or chosen as a model." }),
    el("ul", { class: "cp__evidence" }, sources.map((source) => el("li", {}, [
      el("button", {
        type: "button", class: "ctl",
        disabled: !source.available,
        "aria-current": state.source.kind === source.kind ? "true" : "false",
        onClick: () => handlers.onOpenEvidence(source.kind),
        text: source.plain_label || source.label,
      }),
      el("p", { class: "cp__note",
                text: source.available ? source.plain_summary
                                       : source.unavailable_reason }),
    ]))),
    el("button", { type: "button", class: "ctl", id: "btn-conclusion",
                   onClick: handlers.onOpenConclusion,
                   text: "What the study concluded" }),
    el("button", { type: "button", class: "ctl", id: "btn-questions",
                   onClick: handlers.onOpenQuestions, text: "Q&A jumps" }),
  ]);
}

/* ------------------------------------------------------------------ shared */
function field(label, id, control, note, { id: noteId, invalid = false } = {}) {
  return el("div", { class: "cp__field" }, [
    el("label", { class: "cp__label", for: id, text: label }),
    control,
    note ? el("p", {
      class: invalid ? "cp__note cp__note--blocked" : "cp__note",
      id: noteId, text: note,
    }) : null,
  ]);
}
