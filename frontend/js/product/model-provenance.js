/* Model and evidence provenance.
 *
 * Checkpoint hashes, training roots, evaluation stage, seed-ledger references and
 * source SHAs. The live session's checkpoint and the frozen study's checkpoints
 * are listed in separate blocks: a checkpoint being loaded for a demonstration
 * does not make that demonstration evidence.
 */

import { el, unavailable } from "./dom.js";
import { count, shortHash } from "./format.js";

export function renderModelProvenance(state) {
  return el("div", { class: "prov" }, [
    liveBlock(state),
    schemaBlock(state),
    evidenceBlock(state),
  ]);
}

function liveBlock(state) {
  const context = state.context;
  const capabilities = state.data.capabilities;
  const policy = capabilities?.live_policies?.find(
    (p) => p.id === context.policyId && p.environment_version === context.environmentVersion);

  return el("section", { class: "panel" }, [
    el("h3", { class: "panel__title", text: "This session" }),
    context.sessionId
      ? el("dl", { class: "facts" }, [
          el("dt", { text: "Source" }),
          el("dd", { text: "LIVE — a running simulation, not evidence" }),
          el("dt", { text: "Environment" }),
          el("dd", { text: `${context.environmentVersion.toUpperCase()} · ` +
            `${capabilities?.environments?.find((e) => e.version === context.environmentVersion)?.env_class || "—"}` }),
          el("dt", { text: "Observation size" }),
          el("dd", { text: count(state.data.schema?.observation?.dim) }),
          el("dt", { text: "Policy" }),
          el("dd", { text: policy?.label || context.policyId || "—" }),
          el("dt", { text: "Checkpoint" }),
          el("dd", { text: context.checkpointId || "None — this controller is rule-based" }),
          el("dt", { text: "Output semantics" }),
          el("dd", { text: policy?.output_description || "No per-action numbers" }),
          el("dt", { text: "Scenario and seed" }),
          el("dd", { text: `${context.scenario} · seed ${context.seed}` }),
          el("dt", { text: "Session" }),
          el("dd", { text: `${context.sessionId} · generation ${context.generation}` }),
        ])
      : unavailable("This session", "No live session is running."),
  ]);
}

function schemaBlock(state) {
  const schema = state.data.schema;
  if (!schema) return unavailable("Schema", "The schema has not loaded yet.");
  return el("section", { class: "panel" }, [
    el("h3", { class: "panel__title", text: "Schema in use" }),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Observation" }),
      el("dd", { text: `${schema.observation.version} · ${count(schema.observation.dim)} values · ` +
        `${count(schema.observation.groups.length)} groups` }),
      el("dt", { text: "Observation source" }),
      el("dd", { text: schema.observation.source }),
      el("dt", { text: "Action space" }),
      el("dd", { text: `${count(schema.action.count)} · ${schema.action.formula}` }),
      el("dt", { text: "Reward" }),
      el("dd", { text: `${count(schema.reward.components.length)} components · ` +
        `${schema.reward.source}` }),
      el("dt", { text: "Reward order" }),
      el("dd", { text: schema.reward.components.join(", ") }),
    ]),
  ]);
}

function evidenceBlock(state) {
  const provenance = state.data.evidence.finalProvenance;
  const study = state.data.evidence.study;
  if (!provenance || !study) {
    return el("section", { class: "panel" }, [
      el("h3", { class: "panel__title", text: "Governed study provenance" }),
      el("p", { class: "tb-empty", text: "Reading the frozen provenance record…" }),
    ]);
  }

  const checkpoints = provenance.checkpoints || [];
  return el("section", { class: "panel" }, [
    el("div", { class: "stamp", dataset: { kind: "final_holdout_evidence" } }, [
      el("span", { class: "stamp__word", text: "FINAL EVIDENCE" }),
    ]),
    el("h3", { class: "panel__title", text: "Governed study provenance" }),
    el("dl", { class: "facts" }, [
      el("dt", { text: "Environment" }),
      el("dd", { text: `${study.environment} · ${count(study.observation_dim)} values · ` +
        `${count(study.action_count)} actions` }),
      el("dt", { text: "Training roots" }),
      el("dd", { text: study.training_roots.join(", ") }),
      el("dt", { text: "Holdout seeds" }),
      el("dd", { text: study.holdout_seeds.join(", ") }),
      el("dt", { text: "Development seeds" }),
      el("dd", { text: study.continuity_seeds.join(", ") }),
      el("dt", { text: "Evaluation source" }),
      el("dd", { text: study.sources.evaluation }),
      el("dt", { text: "Environment pin" }),
      el("dd", { text: study.sources.environment_pin }),
      el("dt", { text: "Closeout" }),
      el("dd", { text: study.sources.closeout }),
      el("dt", { text: "Artifact" }),
      el("dd", { text: provenance.artifact_path }),
    ]),
    checkpoints.length
      ? el("div", { class: "table-scroll" }, [
          el("table", { class: "grid" }, [
            el("caption", { text: "Frozen checkpoints. None of these is loaded by " +
              "the live session unless a V2 demonstration binding is configured." }),
            el("thead", {}, [el("tr", {}, [
              el("th", { scope: "col", text: "Policy" }),
              ...Object.keys(checkpoints[0]).filter((k) => k !== "policy_id")
                .slice(0, 5).map((k) => el("th", { scope: "col", text: k.replace(/_/g, " ") })),
            ])]),
            el("tbody", {}, checkpoints.map((row) => el("tr", {}, [
              el("th", { scope: "row", text: row.policy_id || "—" }),
              ...Object.keys(checkpoints[0]).filter((k) => k !== "policy_id")
                .slice(0, 5).map((k) => el("td", {
                  text: String(row[k]).length > 16 ? shortHash(row[k]) : String(row[k]),
                })),
            ]))),
          ]),
        ])
      : null,
  ]);
}
