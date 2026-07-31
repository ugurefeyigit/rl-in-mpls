/* Keyboard reference and Q&A jumps.
 *
 * Every shortcut listed here has a visible control equivalent elsewhere in the
 * shell; none of them is the only way to do anything.
 */

import { $, el, fill } from "./dom.js";
import { KEYS } from "./contracts.js";

export function renderHelp() {
  fill($("help-body"), [
    el("p", { class: "prose",
      text: "Shortcuts are disabled while you are typing in a field. Every one of " +
            "them has a control in the interface as well." }),
    el("dl", { class: "facts" }, KEYS.flatMap(([keys, meaning]) => [
      el("dt", { text: keys }),
      el("dd", { text: meaning }),
    ])),
  ]);
}

export const QUESTIONS = [
  {
    id: "what-is-mpls",
    question: "What is MPLS traffic engineering here?",
    answer: "Explains the selected demand and its label-switched path over the topology.",
    depth: "presentation",
    mode: "network",
  },
  {
    id: "why-this-action",
    question: "Why this action?",
    answer: "Opens the recommendation's measured grounding, then the RL pipeline.",
    depth: "rl",
    mode: "rl",
  },
  {
    id: "is-it-safe",
    question: "Is it safe?",
    answer: "Shows the action mask, the validator result and the protected-class rule.",
    depth: "rl",
    mode: "rl",
  },
  {
    id: "did-planning-help",
    question: "Did temporal planning help?",
    answer: "Opens the final-holdout conclusion with both halves of the finding.",
    conclusion: true,
  },
  {
    id: "how-validated",
    question: "How was this result validated?",
    answer: "Opens final-evidence integrity, roots, seeds, the one-shot workflow and hashes.",
    mode: "rl",
    rlView: "provenance",
    source: "final_holdout_evidence",
  },
];

export function questionDestination(question) {
  return {
    mode: question.mode || (question.conclusion ? "presentation" : "presentation"),
    rlView: question.rlView || (question.mode === "rl" ? "decision" : null),
    source: question.source || null,
    conclusion: Boolean(question.conclusion),
  };
}

export function renderQuestions({ onJump }) {
  fill($("questions-body"), [
    el("ul", { class: "jump-list" }, QUESTIONS.map((jump) => el("li", {}, [
      el("button", { type: "button", class: "jump", onClick: () => onJump(jump) }, [
        el("span", { class: "jump__q", text: jump.question }),
        el("span", { class: "jump__a", text: jump.answer }),
      ]),
    ]))),
  ]);
}
