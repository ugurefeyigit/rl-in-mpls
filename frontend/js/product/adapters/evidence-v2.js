/* The frozen V2 evidence adapter.
 *
 * Two source kinds share this transport and nothing else. `development_evidence`
 * and `final_holdout_evidence` are kept apart by `assertStage`, which is called
 * at the boundary: a development payload cannot reach a final-evidence region,
 * because that is precisely how a selection result becomes a holdout claim.
 *
 * Every read is a GET. There is no write path to this API and none is added here.
 */

export const FINAL = "final_holdout_evidence";
export const DEVELOPMENT = "development_evidence";

const STAGE_FOR_KIND = {
  [FINAL]: "final_holdout",
  [DEVELOPMENT]: "development",
};

async function get(path) {
  const response = await fetch(path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body?.detail?.message || `Evidence unavailable (${response.status})`);
    error.status = response.status;
    error.detail = body?.detail;
    error.evidenceOutage = true;
    throw error;
  }
  return body;
}

/** A payload may only render in the region its own stage names. */
export function assertStage(payload, kind) {
  const expected = STAGE_FOR_KIND[kind];
  if (!expected) throw new Error(`not an evidence source kind: ${kind}`);
  if (!payload || payload.stage !== expected) {
    throw new Error(
      `refused to render ${payload?.stage || "unstaged"} evidence in the ${expected} region`);
  }
  return payload;
}

export const api = {
  study: () => get("/api/v2/study"),

  finalHoldout: async () => assertStage(await get("/api/v2/final-holdout"), FINAL),
  finalScenarios: async () => assertStage(await get("/api/v2/final-holdout/scenarios"), FINAL),
  finalRewardComponents: async () =>
    assertStage(await get("/api/v2/final-holdout/reward-components"), FINAL),
  finalActions: async () => assertStage(await get("/api/v2/final-holdout/actions"), FINAL),
  finalIntegrity: async () => assertStage(await get("/api/v2/final-holdout/integrity"), FINAL),
  finalProvenance: async () => assertStage(await get("/api/v2/final-holdout/provenance"), FINAL),

  developmentContinuity: async () =>
    assertStage(await get("/api/v2/development/continuity"), DEVELOPMENT),
  developmentSeed42: async () =>
    assertStage(await get("/api/v2/development/seed42"), DEVELOPMENT),

  disclosures: () => get("/api/v2/disclosures"),
};
