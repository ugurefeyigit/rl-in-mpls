/* Recorded replay.
 *
 * The traces hold interval aggregates. That is the whole grain, and this surface
 * shows exactly that grain: a table and a scrub control over recorded rows.
 *
 * The stage above it becomes a reference topology with the limitation printed on
 * it. Colouring links from an aggregate `max_util` would be inventing per-link
 * telemetry that was never recorded, and it is the one thing this file exists to
 * prevent.
 */

import { el, unavailable } from "./dom.js";
import { count, mbps, ms, num, percent, signed } from "./format.js";
import { LINK_TELEMETRY_UNAVAILABLE, RECORDED_FIELDS } from "./adapters/recorded-v2.js";

export function renderRecordedTrace(state, { onLoad, onScrub }) {
  const replay = state.data.replay;
  const index = replay?.index;

  if (!index) {
    return el("p", { class: "tb-empty", text: "Reading the recorded-episode catalogue…" });
  }

  const picker = episodePicker(index, replay, onLoad);

  if (!index.available) {
    return el("div", { class: "replay" }, [
      unavailable("Recorded traces are not configured on this machine",
        index.configure_hint),
      el("p", { class: "prose",
        text: `The catalogue still lists all ${count(index.episodes?.length || 0)} ` +
              `recorded episodes; none of them can be loaded here.` }),
      picker,
    ]);
  }

  return el("div", { class: "replay" }, [
    el("p", { class: "prose", text: LINK_TELEMETRY_UNAVAILABLE }),
    picker,
    replay.episode ? episodeView(replay, onScrub)
      : el("p", { class: "tb-empty", text: "Choose an episode to load its recorded steps." }),
  ]);
}

function episodePicker(index, replay, onLoad) {
  const episodes = index.episodes || [];
  const policies = [...new Set(episodes.map((e) => e.policy_id))];
  const scenarios = [...new Set(episodes.map((e) => e.scenario))];
  const seeds = [...new Set(episodes.map((e) => e.seed))];

  const current = {
    policy: replay?.policy_id || policies[0],
    scenario: replay?.scenario || scenarios[0],
    seed: replay?.seed || seeds[0],
  };

  const select = (id, label, options, value) => el("label", { class: "ctl-field" }, [
    el("span", { text: label }),
    el("select", {
      id,
      onChange: (event) => { current[id.replace("replay-", "")] = event.target.value; },
    }, options.map((option) => el("option", {
      value: String(option), selected: String(option) === String(value),
      text: String(option),
    }))),
  ]);

  return el("div", { class: "replay__picker" }, [
    select("replay-policy", "Policy", policies, current.policy),
    select("replay-scenario", "Scenario", scenarios, current.scenario),
    select("replay-seed", "Seed", seeds, current.seed),
    el("button", {
      type: "button", class: "chip",
      disabled: !index.available,
      onClick: () => onLoad(current.policy, current.scenario, Number(current.seed)),
      text: "Load recorded episode",
    }),
  ]);
}

function episodeView(replay, onScrub) {
  const steps = replay.episode.steps || [];
  const cursor = Math.min(replay.currentStep ?? 0, Math.max(0, steps.length - 1));
  const row = steps[cursor];

  return el("div", { class: "replay__body" }, [
    el("div", { class: "replay__controls" }, [
      el("label", { class: "ctl-field" }, [
        el("span", { text: `Recorded step ${count(cursor)} of ${count(steps.length - 1)}` }),
        el("input", {
          type: "range", id: "replay-scrub", min: "0",
          max: String(Math.max(0, steps.length - 1)), value: String(cursor),
          onInput: (event) => onScrub(Number(event.target.value)),
        }),
      ]),
    ]),

    row ? el("dl", { class: "facts" }, RECORDED_FIELDS.flatMap(([key, label, kind]) => {
      if (!(key in row)) return [];
      return [
        el("dt", { text: label }),
        el("dd", { text: formatRecorded(kind, row[key]) }),
      ];
    })) : el("p", { class: "tb-empty", text: "This episode records no steps." }),

    el("div", { class: "table-scroll" }, [
      el("table", { class: "grid" }, [
        el("caption", { text: "Recorded interval aggregates. No per-link utilization " +
          "was recorded, so no link-level topology state exists for these rows." }),
        el("thead", {}, [el("tr", {}, [
          el("th", { scope: "col", text: "Step" }),
          el("th", { scope: "col", text: "Action" }),
          el("th", { scope: "col", text: "Reward" }),
          el("th", { scope: "col", text: "Busiest link" }),
          el("th", { scope: "col", text: "SLA violations" }),
          el("th", { scope: "col", text: "Moved" }),
        ])]),
        el("tbody", {}, steps.slice(
          Math.max(0, cursor - 6), Math.max(0, cursor - 6) + 13,
        ).map((step) => el("tr", {
          "aria-selected": step.step_index === cursor ? "true" : "false",
        }, [
          el("th", { scope: "row", text: String(step.step_index) }),
          el("td", { text: String(step.action) }),
          el("td", { text: signed(step.reward, 3) }),
          el("td", { text: percent(step.max_util, 1) }),
          el("td", { text: count(step.sla_violations) }),
          el("td", { text: mbps(step.moved_mbps) }),
        ]))),
      ]),
    ]),
  ]);
}

function formatRecorded(kind, value) {
  if (value === null || value === undefined) return "—";
  switch (kind) {
    case "share": return percent(value, 2);
    case "ms": return ms(value);
    case "count": return count(value);
    case "mbps": return mbps(value);
    case "bool": return value ? "yes" : "no";
    case "clock": return String(value);
    default: return num(value, 4);
  }
}
