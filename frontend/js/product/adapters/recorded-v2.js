/* Recorded replay of preserved final-holdout episodes.
 *
 * This adapter is a tape player. It has no execution controls, and it refuses a
 * payload that does not declare itself recorded — a live payload reaching a
 * replay surface would put a controller run behind a RECORDED stamp.
 *
 * The traces hold interval aggregates, not per-link utilization. That is not a
 * gap to fill in: `linkTelemetry` is permanently false here, and the stage shows
 * a static reference topology with the limitation printed on it.
 */

export const kind = "recorded_replay";

export const LINK_TELEMETRY_UNAVAILABLE =
  "No per-link utilization was recorded. V2 traces keep interval aggregates " +
  "only, so a link-level topology cannot be replayed.";

export const REFERENCE_TOPOLOGY_NOTE = "REFERENCE TOPOLOGY · NO RECORDED LINK TELEMETRY";

async function get(path) {
  const response = await fetch(path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body?.detail?.message || response.statusText);
    error.status = response.status;
    error.detail = body?.detail;
    throw error;
  }
  return body;
}

/** Refuse anything that is not an immutable recorded trace. */
export function assertRecorded(payload) {
  if (!payload || payload.kind !== "recorded_replay" || payload.live !== false) {
    throw new Error("recorded surface received a payload that is not a recorded trace");
  }
  return payload;
}

export const api = {
  index: async () => assertRecorded(await get("/api/v2/replay/index")),
  episode: async (policyId, scenario, seed) => assertRecorded(
    await get(`/api/v2/replay/episode?policy_id=${encodeURIComponent(policyId)}` +
              `&scenario=${encodeURIComponent(scenario)}&seed=${seed}`)),
};

/** The fields a recorded step genuinely carries, in reading order. */
export const RECORDED_FIELDS = [
  ["clock", "Time", "clock"],
  ["action", "Action", "count"],
  ["action_accepted", "Accepted", "bool"],
  ["valid_action_count", "Valid actions", "count"],
  ["reward", "Interval reward", "num"],
  ["max_util", "Busiest link", "share"],
  ["gross_max_util", "Gross busiest link", "share"],
  ["mean_util", "Mean link load", "share"],
  ["delivered_ratio", "Delivered traffic", "share"],
  ["mean_delay_ms", "Mean demand delay", "ms"],
  ["loss_ratio", "Loss ratio", "share"],
  ["sla_violations", "SLA violations", "count"],
  ["congested_links", "Congested links", "count"],
  ["disconnected_demands", "Disconnected demands", "count"],
  ["moved_mbps", "Moved bandwidth", "mbps"],
  ["frr_changes", "FRR protection moves", "count"],
  ["n_failed_links", "Failed links", "count"],
];
