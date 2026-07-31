/* Reward components as a signed waterfall.
 *
 * Components are drawn in the authoritative emission order, with signed bars —
 * failure red is not used for an ordinary negative term, because a movement cost
 * is not a failure. The exact-sum indicator is a first-class readout: if the
 * components stop reconciling with the scalar reward, the reader sees it here
 * before anyone quotes the number.
 *
 * V1 and V2 component sets carry their version label and are never padded to
 * look like each other.
 */

import { el, tag, unavailable } from "./dom.js";
import { count, num, signed } from "./format.js";

export function renderRewardWaterfall(state) {
  const reward = state.data.decision?.reward;
  if (!reward) return unavailable("Reward", "No decision payload has been read.");
  if (!reward.available) return unavailable("Reward", reward.reason);

  const magnitudes = reward.components.map((c) => Math.abs(Number(c.value) || 0));
  const max = Math.max(...magnitudes, 1e-9);

  return el("div", { class: "rew" }, [
    el("div", { class: "rew__head" }, [
      el("h3", { class: "panel__title",
        text: `${reward.environment_version.toUpperCase()} reward · ` +
              `${count(reward.component_count)} components` }),
      reward.exact_sum
        ? tag("Exact sum", "normal")
        : tag("Sum does not reconcile", "failure"),
    ]),

    el("ul", { class: "rew__bars" }, reward.components.map((component) => {
      const value = Number(component.value) || 0;
      const width = (Math.abs(value) / max) * 50;
      return el("li", { class: "rew__row", dataset: { sign: value < 0 ? "neg" : "pos" } }, [
        el("span", { class: "rew__name", text: component.name }),
        el("span", { class: "rew__track", "aria-hidden": "true" }, [
          el("span", {
            class: "rew__fill",
            style: value < 0
              ? `right:50%;width:${width.toFixed(2)}%`
              : `left:50%;width:${width.toFixed(2)}%`,
          }),
        ]),
        el("span", { class: "rew__value", text: signed(value, 4) }),
      ]);
    })),

    el("dl", { class: "facts" }, [
      el("dt", { text: "Component sum" }),
      el("dd", { text: num(reward.component_sum, 6) }),
      el("dt", { text: "Interval reward" }),
      el("dd", { text: num(reward.interval_reward, 6) }),
      el("dt", { text: "Residual" }),
      el("dd", { text: num(reward.residual, 6) }),
      el("dt", { text: "Cumulative reward" }),
      el("dd", { text: signed(reward.cumulative_reward, 3) }),
    ]),

    el("p", { class: "rew__note", text: reward.note }),
    el("p", { class: "rew__note",
      text: "Reward is a simulation score for delivery, congestion, service quality " +
            "and route stability. It is not money and not an industry KPI." }),
  ]);
}
