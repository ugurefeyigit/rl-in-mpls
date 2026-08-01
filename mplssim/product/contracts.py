"""Typed product contracts shared by the backend and the product shell.

Provenance is a *type*, not a badge. A view has exactly one `SourceKind`, and
what that view is allowed to do — execute a policy, colour a link, state a
scientific conclusion — is a property of the kind, not of the component that
happens to render it. Components that forget a kind fail a contract test rather
than silently presenting a recorded aggregate as live telemetry.

The scientific constants restated here are pinned against their authoritative
sources (`mplssim.evidence.identity`, the environments, the reward config) by
tests/test_product_contracts.py, so a drift fails closed instead of reaching a
presentation slide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mplssim.evidence import identity


# --------------------------------------------------------------- source kinds
class SourceKind(str, Enum):
    """The four record types the product must never blur together."""

    LIVE_SESSION = "live_session"
    RECORDED_REPLAY = "recorded_replay"
    DEVELOPMENT_EVIDENCE = "development_evidence"
    FINAL_HOLDOUT_EVIDENCE = "final_holdout_evidence"


#: A live demonstration driven by a frozen checkpoint is still LIVE. Rendering it
#: as evidence would turn an ad-hoc demo run into a scientific claim.
LIVE_DEMONSTRATION_KIND = SourceKind.LIVE_SESSION
LIVE_DEMONSTRATION_LABEL = "LIVE DEMONSTRATION"


@dataclass(frozen=True)
class SourceProfile:
    kind: SourceKind
    label: str
    #: Ledger-stamp pattern; provenance must survive grayscale and colour-blindness.
    pattern: str
    icon: str
    may_execute_policy: bool
    may_render_link_telemetry: bool
    may_state_conclusions: bool
    link_telemetry_reason: str
    required_fields: tuple[str, ...]
    description: str
    #: Wording for people who have not read the study. The bare ledger stamp is
    #: for the provenance rail; the setup path uses these instead.
    plain_label: str = ""
    plain_summary: str = ""
    #: Live simulation choices belong beside the scenario and model controls.
    #: Evidence never does — it is a finished record, not a thing you can run.
    group: str = "live"


_SOURCE_PROFILES: dict[SourceKind, SourceProfile] = {
    SourceKind.LIVE_SESSION: SourceProfile(
        kind=SourceKind.LIVE_SESSION,
        label="LIVE",
        pattern="solid-rule",
        icon="live",
        may_execute_policy=True,
        may_render_link_telemetry=True,
        may_state_conclusions=False,
        link_telemetry_reason="Link telemetry comes from the running engine.",
        required_fields=("session_id", "generation", "sequence", "step",
                         "environment_version", "scenario", "seed"),
        description="A running or paused simulation session.",
        plain_label="Live simulation",
        plain_summary="A simulation running here, now, on your chosen scenario, "
                      "seed and controller.",
        group="live",
    ),
    SourceKind.RECORDED_REPLAY: SourceProfile(
        kind=SourceKind.RECORDED_REPLAY,
        label="RECORDED",
        pattern="tape-hatch",
        icon="recorded",
        may_execute_policy=False,
        may_render_link_telemetry=False,
        may_state_conclusions=False,
        link_telemetry_reason=(
            "No per-link utilization was recorded. V2 traces keep interval "
            "aggregates only, so a link-level topology cannot be replayed."),
        required_fields=("policy_id", "scenario", "seed", "recorded_step", "stage"),
        description="Playback of an immutable recorded trace. Never a controller run.",
        plain_label="Recorded episode playback",
        plain_summary="Step-by-step playback of an episode that was recorded "
                      "earlier. Nothing is being decided while you watch it.",
        group="study_evidence",
    ),
    SourceKind.DEVELOPMENT_EVIDENCE: SourceProfile(
        kind=SourceKind.DEVELOPMENT_EVIDENCE,
        label="DEVELOPMENT",
        pattern="open-grid",
        icon="development",
        may_execute_policy=False,
        may_render_link_telemetry=False,
        may_state_conclusions=True,
        link_telemetry_reason=(
            "Development evidence is aggregate. It carries no episode topology."),
        required_fields=("stage", "source_sha", "artifact_path"),
        description="Pilot, continuity, learning-curve and checkpoint-selection evidence.",
        plain_label="Pilot and continuity results (before the holdout)",
        plain_summary="Results produced while the study was still being built: "
                      "pilots, learning curves and the checkpoint selection. They "
                      "were all created before the final holdout was opened, so "
                      "they may inform how the study was set up but they are not "
                      "the study's conclusion.",
        group="study_evidence",
    ),
    SourceKind.FINAL_HOLDOUT_EVIDENCE: SourceProfile(
        kind=SourceKind.FINAL_HOLDOUT_EVIDENCE,
        label="FINAL EVIDENCE",
        pattern="double-rule",
        icon="final-evidence",
        may_execute_policy=False,
        may_render_link_telemetry=False,
        may_state_conclusions=True,
        link_telemetry_reason=(
            "Final-holdout evidence is a frozen one-shot result, not a topology state."),
        required_fields=("stage", "source_sha", "artifact_path"),
        description="The untouched one-shot final holdout. Frozen; never a live comparator.",
        plain_label="Final study result (frozen, read-only)",
        plain_summary="The one-shot result on data the study never trained or "
                      "selected on. It ran once, it is frozen, and it can only be "
                      "read. It is never a live comparison and never a model you "
                      "can pick.",
        group="study_evidence",
    ),
}


def source_profile(kind: SourceKind) -> SourceProfile:
    return _SOURCE_PROFILES[SourceKind(kind)]


def may_share_region(a: SourceKind, b: SourceKind) -> bool:
    """Development and final evidence can never occupy one region, chart series
    or aggregate — that is exactly how a selection result becomes a holdout claim."""
    a, b = SourceKind(a), SourceKind(b)
    evidence = {SourceKind.DEVELOPMENT_EVIDENCE, SourceKind.FINAL_HOLDOUT_EVIDENCE}
    if a in evidence and b in evidence:
        return a is b
    return True


# --------------------------------------------------------------------- modes
@dataclass(frozen=True)
class Mode:
    id: str
    label: str
    shortcut: str
    summary: str


PRIMARY_MODES: tuple[Mode, ...] = (
    Mode("presentation", "Presentation", "Alt+1",
         "The whole project at speaking depth, with the topology readable from the room."),
    Mode("network", "Network Information", "Alt+2",
         "The MPLS traffic-engineering operations workspace."),
    Mode("rl", "RL Information", "Alt+3",
         "The inference pipeline and the governed evidence record."),
)

MODE_IDS: tuple[str, ...] = tuple(m.id for m in PRIMARY_MODES)


@dataclass(frozen=True)
class Workflow:
    """A workflow lives *inside* a mode. It is never a fourth primary mode."""

    id: str
    mode: str
    label: str
    summary: str


GUIDED_STORY = Workflow(
    "guided-story", "presentation", "Guided Story",
    "A presenter-paced walk through one real demo_evening session, beat by beat.")

WORKFLOWS: tuple[Workflow, ...] = (GUIDED_STORY,)

#: RL Information secondary views. Also not primary modes.
RL_VIEWS: tuple[str, ...] = ("decision", "study", "provenance")


# -------------------------------------------------------------------- routes
@dataclass(frozen=True)
class RouteContext:
    path: str
    mode: str
    source_kind: SourceKind
    rl_view: str | None = None
    workflow: str | None = None
    note: str = ""


ROUTES: dict[str, RouteContext] = {
    "/": RouteContext("/", "network", SourceKind.LIVE_SESSION,
                      note="Preserves the engineering-console destination."),
    "/advanced": RouteContext("/advanced", "network", SourceKind.LIVE_SESSION,
                              note="Stable alias for `/`; same shell and deep context."),
    "/present": RouteContext("/present", "presentation", SourceKind.LIVE_SESSION,
                             note="Preserves the Presentation Mode destination."),
    "/study": RouteContext("/study", "rl", SourceKind.FINAL_HOLDOUT_EVIDENCE,
                           rl_view="study",
                           note="Preserves the sealed-study destination without "
                                "creating a fourth primary mode."),
}


def route_context(path: str) -> RouteContext:
    try:
        return ROUTES[path]
    except KeyError as exc:  # pragma: no cover - guarded by the route table test
        raise KeyError(f"unregistered product route {path!r}") from exc


# --------------------------------------------------------------- environments
@dataclass(frozen=True)
class EnvironmentProfile:
    version: str
    label: str
    env_class: str
    observation_dim: int
    action_count: int
    reward_components: tuple[str, ...]
    summary: str


#: V1 observation: 5*64 link + 15*17 demand + 11 global = 586. Mirrors
#: mplssim/rl/env.py, which tests/test_product_api.py checks against the live space.
V1_OBSERVATION_DIM = 586
ACTION_COUNT = 69
N_DEMANDS = 17
K_PATHS = 4

#: V1 reward component names, in the order mplssim/rl/reward.py emits them.
V1_REWARD_COMPONENTS: tuple[str, ...] = (
    "delivery", "congestion", "sla", "delay", "loss",
    "reroute", "flap", "invalid", "disconnect",
)

V2_REWARD_COMPONENTS: tuple[str, ...] = identity.REWARD_COMPONENTS

ENVIRONMENTS: dict[str, EnvironmentProfile] = {
    "v1": EnvironmentProfile(
        version="v1", label="V1", env_class="MplsTeEnv",
        observation_dim=V1_OBSERVATION_DIM, action_count=ACTION_COUNT,
        reward_components=V1_REWARD_COMPONENTS,
        summary="The live engineering environment. 586-value observation, "
                "V1 reward terms; not the governed V2 study environment."),
    "v2": EnvironmentProfile(
        version="v2", label="V2", env_class=identity.ENVIRONMENT,
        observation_dim=identity.OBSERVATION_DIM, action_count=identity.ACTION_COUNT,
        reward_components=V2_REWARD_COMPONENTS,
        summary="The governed study environment. 604-value observation and the "
                "exact 12-component reward the closed study reports."),
}


def decode_action(action: int) -> tuple[int | None, int | None]:
    """`0` is no TE change; `1 + 4*d + p` moves demand *d* to candidate path *p*."""
    action = int(action)
    if action == 0:
        return (None, None)
    if not 0 < action < ACTION_COUNT:
        raise ValueError(f"action {action} outside 0..{ACTION_COUNT - 1}")
    d_idx, p_idx = divmod(action - 1, K_PATHS)
    return (d_idx, p_idx)


def encode_action(demand_idx: int, path_idx: int) -> int:
    return 1 + K_PATHS * int(demand_idx) + int(path_idx)


# ------------------------------------------------------------- policy output
class OutputSemantics(str, Enum):
    """What a learner's per-action numbers actually are."""

    PROBABILITIES = "probabilities"
    SCORES = "scores"
    NONE = "none"

    @property
    def label(self) -> str:
        return _OUTPUT_LABELS[self][0]

    @property
    def description(self) -> str:
        return _OUTPUT_LABELS[self][1]

    @property
    def percent(self) -> bool:
        """Only a real normalized distribution may be rendered as a percentage."""
        return self is OutputSemantics.PROBABILITIES


_OUTPUT_LABELS: dict[OutputSemantics, tuple[str, str]] = {
    OutputSemantics.PROBABILITIES: (
        "Action probability",
        "Probability mass over the valid masked action distribution, taken from "
        "the policy itself. Sums to one across valid actions."),
    OutputSemantics.SCORES: (
        "Action score",
        "Immediate-reward estimate per valid action. Unnormalized, may be "
        "negative, and does not sum to one."),
    OutputSemantics.NONE: (
        "Per-action output unavailable",
        "This controller exposes no per-action numbers."),
}


# ---------------------------------------------------------------- no-op grain
@dataclass(frozen=True)
class NoopMetric:
    id: str
    label: str
    denominator: str
    description: str


NOOP_METRICS: dict[str, NoopMetric] = {
    "step_pooled_noop_share": NoopMetric(
        "step_pooled_noop_share", "Step-pooled no-op share",
        "all recorded steps",
        "Action-0 count divided by every recorded step, pooled across episodes."),
    "episode_mean_noop_frequency": NoopMetric(
        "episode_mean_noop_frequency", "Episode-level mean no-op frequency",
        "mean of per-episode fractions",
        "The mean of each episode's own no-op fraction, so short and long "
        "episodes weigh equally."),
}


# --------------------------------------------------------------- vocabulary
#: Phrases the product may never use. Enforced over product copy by
#: tests/test_product_ui.py.
FORBIDDEN_PRODUCT_PHRASES: tuple[str, ...] = (
    "AI Advisor",
    "AI advisor",
    "causal importance",
    "feature importance",
    "the model thinks",
    "the model wants",
    "the model knows",
)

#: The wording the product uses instead.
REQUIRED_VOCABULARY: dict[str, str] = {
    "recommendation": "policy recommendation",
    "changed_features": "changed-feature ranking (descriptive change, not causal importance)",
    "counterfactual": "simulated one-interval estimate from cloned state",
    "bandit_output": "action score / immediate-reward estimate",
    "ppo_output": "action probability",
}


#: The disclaimer that must accompany any network imagery.
TOPOLOGY_DISCLAIMER = (
    "Fictional scaled national backbone for demonstration — not a real operator "
    "topology. Fixed engineering schematic · not geographic.")
