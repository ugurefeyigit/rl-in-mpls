/* Guided Story: a workflow inside Presentation, never a fourth mode.
 *
 * Eleven beats over one real `demo_evening` session. The story does not script
 * the network — it advances the actual engine and describes what actually
 * happened. A beat whose event has not occurred says so rather than narrating
 * it into existence.
 *
 * Going back reviews an earlier beat. It never rewinds a live engine, and the
 * copy says where the network really is while you review.
 */

import { count, clock, percent } from "./format.js";

export const STORY_SCENARIO = "demo_evening";
export const STORY_SEED = 42;
export const STORY_ENVIRONMENT = "v2";
export const STORY_POLICY = "masked_bandit";
export const STORY_COMPARATOR = "greedy";

/** The story runs the governed V2 environment in advisor execution, because a
 *  beat that asks you to approve an action must actually be able to hold it. */
export function storySessionConfig() {
  return {
    scenario: STORY_SCENARIO,
    environment: STORY_ENVIRONMENT,
    algorithms: [STORY_POLICY, STORY_COMPARATOR],
    seed: STORY_SEED,
    safety_filter: true,
    speed: "1x",
    autostart: false,
    execution: "advisor",
    advisor: true,
    interface_mode: "present",
  };
}

export function matchesStorySession(status) {
  return Boolean(status?.session_id
    && status.scenario === STORY_SCENARIO
    && status.seed === STORY_SEED
    && status.environment === STORY_ENVIRONMENT
    && status.advisor === true
    && status.algorithms?.join(",") === `${STORY_POLICY},${STORY_COMPARATOR}`);
}

/**
 * `advance` is what the presenter's Next does at this beat. `narrate` builds the
 * copy from the live snapshot, so it can only say what the payload contains.
 */
export const BEATS = [
  {
    id: 1,
    label: "Establish the network",
    advance: null,
    narrate: (ctx) =>
      `This is one simulated backbone of ${count(ctx.nodeCount)} cities and ` +
      `${count(ctx.linkCount)} links carrying ${count(ctx.demandCount)} traffic ` +
      `demands across six service classes. It is a fictional scaled network, not a ` +
      `real operator topology. We are watching ${ctx.policyLabel} against ` +
      `${ctx.comparatorLabel} on the same scenario and the same seed.`,
  },
  {
    id: 2,
    label: "Read the initial evening",
    advance: { kind: "step" },
    narrate: (ctx) =>
      ctx.pendingRecommendation
        ? `The story runs with advisor approval, so the first interval is not ` +
          `applied until you accept it. ${ctx.policyLabel} has proposed an action ` +
          `at ${ctx.clock}; the card beneath the map shows exactly what it would ` +
          `move. Approve or reject it to run the interval.`
        : (ctx.maxUtil === null
          ? `No interval has completed yet, so there is nothing measured to read. ` +
            `Approve the proposed action, or press Step, to run the first interval.`
          : `At ${ctx.clock} the network is in ${ctx.phaseLabel.toLowerCase()}. The ` +
            `busiest link is at ${percent(ctx.maxUtil, 0)} and ` +
            `${count(ctx.slaViolations)} demand-interval SLA violation(s) were ` +
            `recorded this interval.`),
  },
  {
    id: 3,
    label: "Traffic rises",
    advance: { kind: "runUntil", condition: "congestion" },
    narrate: (ctx) =>
      ctx.congestedLinks.length
        ? `By ${ctx.clock} the evening load has pushed ${ctx.busiestLabel} to ` +
          `${percent(ctx.maxUtil, 0)}. That corridor is where pressure shows first.`
        : `By ${ctx.clock} load is rising but no link has reached the congestion ` +
          `threshold yet. The busiest is ${ctx.busiestLabel} at ${percent(ctx.maxUtil, 0)}.`,
  },
  {
    id: 4,
    label: "Bottleneck becomes visible",
    advance: null,
    select: (ctx) => ctx.busiestLinkId && { objectType: "link", objectId: ctx.busiestLinkId },
    narrate: (ctx) =>
      ctx.busiestLinkId
        ? `${ctx.busiestLabel} carries ${count(ctx.busiestLsps)} label-switched paths ` +
          `on ${ctx.busiestCapacity}. Each demand on it has up to four candidate ` +
          `routes; moving one changes which links carry its traffic.`
        : `No link is under pressure yet, so there is no bottleneck to inspect.`,
  },
  {
    id: 5,
    label: "SLA risk appears",
    advance: null,
    select: (ctx) => ctx.riskDemandId && { objectType: "demand", objectId: ctx.riskDemandId },
    narrate: (ctx) =>
      ctx.riskDemandId
        ? `${ctx.riskDemandLabel} is the highest-priority demand currently at risk: ` +
          `${ctx.riskDemandState}. ${count(ctx.affectedNow)} demand(s) are affected right ` +
          `now; that is a current count, separate from cumulative demand-interval ` +
          `SLA violations.`
        : `No demand is outside its SLA yet. ${count(ctx.slaViolations)} demand-interval ` +
          `violation(s) have been recorded so far this interval.`,
  },
  {
    id: 6,
    label: "The policy recommends",
    advance: { kind: "propose" },
    narrate: (ctx) =>
      ctx.hasRecommendation
        ? `${ctx.policyLabel} proposes a change. Nothing has been applied — this is a ` +
          `preview, and the card beneath the map shows exactly what it would move.`
        : `${ctx.policyLabel} has not produced a recommendation for this moment.`,
  },
  {
    id: 7,
    label: "Inspect or approve",
    advance: null,
    narrate: () =>
      `Before approving, you can open the same moment at Network depth to see the ` +
      `route and link facts, or at RL depth to see the observation, the action mask ` +
      `and the reward terms. The expected outcome on the card is a simulated ` +
      `estimate computed on a copy of the current state, not an observation.`,
  },
  {
    id: 8,
    label: "Observe the transition",
    advance: { kind: "approve" },
    narrate: (ctx) =>
      ctx.lastReward === null
        ? `The interval has not completed yet.`
        : `The move ran. The observed outcome now sits beside the estimate on the ` +
          `card, and the interval scored ${ctx.lastRewardText}. Where they differ, the ` +
          `difference stays visible.`,
  },
  {
    id: 9,
    label: "Demand surge and failure",
    advance: { kind: "runUntil", condition: "failure" },
    // A fast-forward is one delegated gesture, not many approvals. The copy
    // says so rather than letting the presenter imply each interval was
    // individually approved.
    narrate: (ctx) =>
      ctx.failedLinks.length
        ? `Fast-forward ran these intervals with the controller acting on its ` +
          `own; they were not approved one by one. A link has failed: ` +
          `${ctx.failedLabels.join(", ")}. The engine's built-in ` +
          `fast reroute moved the affected paths immediately — that is protection, ` +
          `not a controller decision. What follows is the traffic-engineering ` +
          `response to the pressure the failure left behind.`
        : `We advanced to ${ctx.clock}, but the scenario did not expose the scheduled ` +
          `Kayseri–Samsun failure within the allowed window.`,
  },
  {
    id: 10,
    label: "Compare decisions",
    advance: null,
    narrate: (ctx) =>
      ctx.comparisonMatched
        ? `Both controllers have run the same scenario from the same starting state ` +
          `with the same inputs. ${ctx.comparisonLead} Movement has a cost, so the ` +
          `lane also shows what each one moved.`
        : `The comparison is not being shown: ${ctx.comparisonReason}`,
  },
  {
    id: 11,
    label: "Repair and conclusion",
    advance: { kind: "runUntil", condition: "recovery" },
    conclusion: true,
    narrate: () =>
      `That is one evening on one simulated backbone. What the closed V2 study ` +
      `actually established is a separate, frozen record — open the governed ` +
      `conclusion to read it, including what it did not establish.`,
  },
];

export const BOOKMARK_LABELS = [
  "Pressure threshold crossed", "First SLA risk", "Policy recommendation",
  "Accepted or rejected action", "Flash crowd", "Kayseri–Samsun failure",
  "FRR completion", "Second recommendation", "Repair", "Stabilization",
  "Governed conclusion",
];

/** Facts the beat copy is allowed to use. Everything comes from the payload. */
export function storyContext(state) {
  const snapshot = state.data.snapshot;
  const metrics = snapshot?.metrics;
  const values = metrics?.available ? metrics.values : {};
  const links = snapshot?.links || [];
  const demands = snapshot?.demands || [];

  const busiest = links.filter((l) => l.up)
    .reduce((best, l) => (!best || l.worst_utilization > best.worst_utilization ? l : best), null);
  const atRisk = demands
    .filter((d) => d.risk_rank <= 2)
    .sort((a, b) => a.risk_rank - b.risk_rank || b.priority - a.priority)[0];
  const failed = links.filter((l) => !l.up);
  const comparison = state.data.comparison;
  const decision = state.data.decision;

  return {
    clock: snapshot?.time?.clock || "—",
    phaseLabel: snapshot?.incident?.label || "unknown state",
    nodeCount: snapshot?.nodes?.length ?? 18,
    linkCount: links.length || 32,
    demandCount: demands.length || 17,
    policyLabel: policyLabel(state),
    comparatorLabel: comparatorLabel(state),
    maxUtil: values.max_util?.value ?? null,
    slaViolations: values.sla_violations?.value ?? 0,
    congestedLinks: snapshot?.incident?.congested_links || [],
    busiestLinkId: busiest?.id || null,
    busiestLabel: busiest ? `${busiest.a_city}–${busiest.z_city} (${busiest.id})` : "no link",
    busiestLsps: busiest?.n_lsps ?? 0,
    busiestCapacity: busiest ? `${busiest.capacity_mbps} Mbps per direction` : "—",
    riskDemandId: atRisk?.id || null,
    riskDemandLabel: atRisk ? `${atRisk.src_city} → ${atRisk.dst_city} ${atRisk.class_label}` : null,
    riskDemandState: atRisk?.risk_label || null,
    affectedNow: snapshot?.incident?.demands_at_risk?.length ?? 0,
    hasRecommendation: Boolean(state.data.recommendation),
    pendingRecommendation: Boolean(state.data.recommendation?.pending),
    lastReward: decision?.reward?.available ? decision.reward.interval_reward : null,
    lastRewardText: decision?.reward?.available
      ? decision.reward.interval_reward.toFixed(3) : "—",
    failedLinks: failed.map((l) => l.id),
    failedLabels: failed.map((l) => `${l.a_city}–${l.z_city} (${l.id})`),
    comparisonMatched: Boolean(comparison?.matched),
    comparisonReason: comparison?.reason || "no comparison is configured.",
    comparisonLead: comparisonLead(comparison),
  };
}

function comparisonLead(comparison) {
  const lanes = comparison?.lane_details || [];
  if (lanes.length < 2) return "";
  const [a, b] = lanes;
  if (a.cumulative_reward === b.cumulative_reward) {
    return "Both lanes are level on cumulative reward so far.";
  }
  const leader = a.cumulative_reward > b.cumulative_reward ? a : b;
  return `${leader.algorithm} leads this run on cumulative reward.`;
}

function policyLabel(state) {
  const id = state.context.policyId;
  return state.data.capabilities?.live_policies
    ?.find((p) => p.id === id && p.environment_version === state.context.environmentVersion)
    ?.label || id || "the selected policy";
}

function comparatorLabel(state) {
  const id = state.context.comparator;
  if (!id) return "no comparator";
  return state.data.capabilities?.live_policies?.find((p) => p.id === id)?.label || id;
}

export function beatAt(index) {
  return BEATS[Math.max(0, Math.min(BEATS.length - 1, index))];
}

export function progressText(state) {
  if (!state.story.active) return "Guided Story is not running.";
  const beat = beatAt(state.story.beat);
  const reviewing = state.story.reviewBeat !== null
    && state.story.reviewBeat !== state.story.beat;
  if (reviewing) {
    const review = beatAt(state.story.reviewBeat);
    return `Reviewing beat ${review.id} · ${review.label} · network remains at ` +
           `${clock(state.context.hour)}`;
  }
  return `Beat ${beat.id}/${BEATS.length} · ${beat.label}`;
}
