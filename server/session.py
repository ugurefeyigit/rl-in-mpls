"""Live simulation session management with an explicit state machine.

States: idle → running ⇄ paused → completed, with error as a trap state.

Concurrency contract (root-cause fixes documented in docs/DEBUG_AUDIT.md):

- ONE `asyncio.Lock` per session serializes every state mutation: automatic
  ticks, manual steps, pause/resume, interventions, advisor calls and reset.
  The tick loop holds the lock for the duration of one interval computation,
  so "pause" takes effect after at most one in-flight interval.
- Exactly one loop task may exist. `resume` only spawns a task when no live
  task exists; a paused-but-alive task simply continues when the state flips
  back to running.
- A generation counter invalidates loops across reset: a loop captured with
  an old generation exits without broadcasting, so a reset can never be
  followed by a stale tick.
- Interventions never advance the simulated clock; they mutate state under
  the lock and trigger an immediate out-of-band snapshot broadcast.
- Reset rebuilds runners from the SAME stored SessionConfig (scenario,
  algorithms, model tag, seed, safety filter, speed, interface mode).

Wall-clock pacing: speed "1x" is a *presentation* rate (one 5-simulated-minute
control interval every 2 s), not real time.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from mplssim.baselines import make_baseline
from mplssim.factory import engine_config_from_training, make_engine
from mplssim.product import checkpoints_v2
from mplssim.product.live_v2 import EngineV2View
from mplssim.rl.env import MplsTeEnv
from mplssim.rl.reward import compute_reward
from mplssim.sim.engine import SimulationEngine
from server.events import log_event

ROOT = Path(__file__).resolve().parents[1]
SPEED_SECONDS = {"1x": 2.0, "5x": 0.4, "20x": 0.1, "fast": 0.0}

#: The truthful live default. V2 is the governed study environment; V1 remains
#: reachable only by asking for it explicitly, and is never substituted when a
#: V2 artifact is missing.
DEFAULT_ENVIRONMENT = "v2"
ENVIRONMENTS: tuple[str, ...] = ("v1", "v2")

#: Controllers each environment can actually run. `rl` is the V1 generic
#: controller slot driven by a V1 checkpoint tag; V2's learners are named for
#: what they are.
V1_ALGORITHMS: tuple[str, ...] = ("rl", "static", "greedy", "cspf", "random")
V2_ALGORITHMS: tuple[str, ...] = (
    "masked_bandit", "maskable_ppo", "greedy", "cspf", "static")
V2_BASELINES: tuple[str, ...] = ("greedy", "cspf", "static")


def algorithms_for(environment: str) -> tuple[str, ...]:
    if environment == "v1":
        return V1_ALGORITHMS
    if environment == "v2":
        return V2_ALGORITHMS
    raise ValueError(f"environment must be one of {list(ENVIRONMENTS)}")

_MODEL_CACHE: dict[str, Any] = {}


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class SessionError(RuntimeError):
    """Raised for invalid state transitions; mapped to HTTP 409 by the API."""


def load_model(tag: str) -> Any:
    """Load a MaskablePPO checkpoint (best_model.zip, falling back to
    final_model.zip), verify shape compatibility, cache across sessions."""
    if tag in _MODEL_CACHE:
        return _MODEL_CACHE[tag]
    from sb3_contrib import MaskablePPO
    from mplssim.validation import check_model_compatibility
    base = ROOT / "models" / tag
    path = base / "best_model.zip"
    if not path.exists():
        path = base / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"no model found under {base}")
    model = MaskablePPO.load(path, device="cpu")
    check_model_compatibility(model, tag)
    _MODEL_CACHE[tag] = model
    log_event("model_loaded", model_tag=tag, file=path.name)
    return model


def list_checkpoints() -> list[dict[str, Any]]:
    out = []
    models_dir = ROOT / "models"
    if models_dir.exists():
        for tag_dir in sorted(models_dir.iterdir()):
            if tag_dir.is_dir():
                for f in sorted(tag_dir.glob("*.zip")):
                    out.append({
                        "tag": tag_dir.name, "file": f.name,
                        "size_mb": round(f.stat().st_size / 1e6, 2),
                        "modified": time.strftime(
                            "%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                    })
    return out


@dataclass(frozen=True)
class SessionConfig:
    """Everything needed to reconstruct the exact same experiment at t=0."""

    scenario: str
    algorithms: tuple[str, ...]
    seed: int
    model_tag: str | None
    safety_filter: bool
    speed: str
    interface_mode: str = "advanced"   # "advanced" | "present"
    advisor: bool = False              # operator-advisor mode (learner runner only)
    environment: str = DEFAULT_ENVIRONMENT
    #: V2 only: which continuity-selected training root the checkpoints come
    #: from. Never inferred from performance; see checkpoints_v2.DEFAULT_ROOT_RULE.
    training_root: int = checkpoints_v2.DEFAULT_ROOT

    @property
    def execution(self) -> str:
        """`advisor` pauses for approval; `automatic` lets the policy act."""
        return "advisor" if self.advisor else "automatic"


class AlgoRunner:
    """One algorithm bound to one engine within a session."""

    environment_version = "v1"
    checkpoint = None

    @property
    def checkpoint_id(self) -> str | None:
        return self.model_tag if self.algorithm == "rl" else None

    @property
    def output_semantics(self) -> str:
        return "probabilities" if self.algorithm == "rl" else "none"

    def __init__(self, cfg: SessionConfig, algorithm: str) -> None:
        self.algorithm = algorithm
        self.safety_filter = cfg.safety_filter
        self.model = None
        self.model_tag = cfg.model_tag or "ppo_te"
        self.env: MplsTeEnv | None = None
        self.controller = None
        if algorithm == "rl":
            self.model = load_model(cfg.model_tag or "ppo_te")
            self.env = MplsTeEnv(scenario=cfg.scenario, base_seed=cfg.seed,
                                 safety_filter=cfg.safety_filter)
            self.env.reset(options={"episode_seed": cfg.seed})
            self.eng: SimulationEngine = self.env.eng
            self._obs = self.env._obs()
            self._prior_obs = None
        else:
            self.eng = make_engine(cfg.scenario, seed=cfg.seed,
                                   cfg=engine_config_from_training())
            self.controller = make_baseline(algorithm, seed=cfg.seed)
        self.last_decision: dict[str, Any] | None = None
        self.cumulative_reward = 0.0
        # authoritative per-step record: reward + components + interval
        # metrics; exports and the scoreboard are derived from THIS, so
        # displayed and exported values cannot disagree.
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ step
    def step(self, counterfactual: bool = True,
             action_override: int | None = None) -> dict[str, Any]:
        if self.algorithm == "rl":
            decision = self._step_rl(counterfactual, action_override)
        else:
            decision = self._step_baseline()
        interval = self.eng.metrics_history[-1]
        self.history.append({
            "step": interval["step"],
            "t_min": interval["t_min"],
            # full precision: exports must sum EXACTLY to the cumulative reward
            "reward": self._last_raw_reward,
            "components": decision["components"],
            "metrics": {k: v for k, v in interval.items() if k != "failed_links"},
            "n_failed_links": len(interval["failed_links"]),
        })
        return decision

    def _predict(self, mask: np.ndarray) -> tuple[int, np.ndarray | None]:
        action, _ = self.model.predict(self._obs, deterministic=True, action_masks=mask)
        probs = None
        try:
            import torch
            obs_t, _ = self.model.policy.obs_to_tensor(self._obs)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(
                    obs_t, action_masks=mask.reshape(1, -1))
                probs = dist.distribution.probs.cpu().numpy()[0]
        except Exception:
            pass
        return int(action), probs

    def _action_desc(self, a: int) -> str:
        if a == 0:
            return "no-op"
        d_idx, p_idx = divmod(a - 1, self.env.k)
        d = self.eng.demands[d_idx]
        return f"{d.id} -> path {p_idx} ({'>'.join(d.candidate_paths[p_idx])})"

    def evaluate_action_vs_noop(self, action: int) -> dict[str, Any]:
        """Cloned-engine one-interval lookahead: proposed action vs no-op.
        NEVER mutates the real engine (both branches run on deep copies)."""
        keys = ("max_util", "mean_delay_ms", "loss_ratio", "sla_violations",
                "delivered_ratio")
        noop_eng = self.eng.clone()
        noop_metrics = noop_eng.step_interval()
        out: dict[str, Any] = {
            "noop": {k: round(float(noop_metrics[k]), 4) for k in keys}}
        if action > 0:
            act_eng = self.eng.clone()
            d_idx, p_idx = divmod(action - 1, self.env.k)
            ok, reason = act_eng.apply_action(d_idx, p_idx, source="rl")
            act_metrics = act_eng.step_interval()
            out["action"] = {k: round(float(act_metrics[k]), 4) for k in keys}
            out["action_applied"] = ok
            out["action_reason"] = reason
            out["delta_max_util"] = round(
                float(act_metrics["max_util"] - noop_metrics["max_util"]), 4)
        return out

    def _step_rl(self, counterfactual: bool,
                 action_override: int | None) -> dict[str, Any]:
        env = self.env
        mask = env.action_masks()
        probs = None
        if action_override is not None:
            action = int(action_override)
        else:
            action, probs = self._predict(mask)

        cf_metrics = None
        if counterfactual and action != 0:
            cf_eng = self.eng.clone()
            cf_metrics = cf_eng.step_interval()  # what no-op would have done

        before_max_util = float(np.max(self.eng.link_util))
        pre_state = self._pre_action_context(action)
        # Kept so the RL observation inspector can show prior/current/delta
        # without the client having to retain and re-key a 586-value vector.
        self._prior_obs = self._obs
        self._obs, reward, terminated, truncated, info = env.step(action)
        self._last_raw_reward = float(reward)
        self.cumulative_reward += float(reward)

        decision = {
            "algorithm": "rl",
            "step": self.eng.step_count,
            "t_min": self.eng.t_min,
            "action": action,
            "decoded": info["decoded_action"],
            "reward": round(float(reward), 4),
            "components": {k: round(v, 4) for k, v in info["reward_components"].items()},
            "cumulative_reward": round(self.cumulative_reward, 2),
            "mask_valid_actions": int(mask.sum()),
            "done": terminated or truncated,
            "explanation": self._explain(action, info, before_max_util, pre_state),
        }
        if probs is not None:
            top = np.argsort(-probs)[:5]
            # Field names stay semantics-neutral: the same shape carries V1/V2
            # PPO probabilities and V2 bandit immediate-reward estimates, and
            # only `output_semantics` says which one a number is.
            decision["output_value"] = round(float(probs[action]), 4)
            decision["output_semantics"] = "probabilities"
            decision["top_actions"] = [
                {"action": int(a), "value": round(float(probs[a]), 4),
                 "desc": self._action_desc(int(a))}
                for a in top if probs[a] > 1e-4
            ]
        if cf_metrics is not None:
            actual = info["metrics"]
            decision["counterfactual"] = {
                "label": "one-step counterfactual (post-hoc, not used by the agent)",
                "noop": {k: round(float(cf_metrics[k]), 4) for k in
                         ("max_util", "mean_delay_ms", "loss_ratio", "sla_violations")},
                "actual": {k: round(float(actual[k]), 4) for k in
                           ("max_util", "mean_delay_ms", "loss_ratio", "sla_violations")},
                "delta_max_util": round(float(actual["max_util"] - cf_metrics["max_util"]), 4),
            }
        self.last_decision = decision
        return decision

    def _pre_action_context(self, action: int) -> dict[str, Any]:
        if action == 0:
            return {}
        d_idx, p_idx = divmod(action - 1, self.env.k)
        return {
            "old_bottleneck": self.eng.path_bottleneck_util(d_idx, int(self.eng.current_path[d_idx])),
            "new_bottleneck": self.eng.projected_bottleneck_after_move(d_idx, p_idx),
            "volume": float(self.eng.demand_volumes[d_idx]),
        }

    def _explain(self, action: int, info: dict, before_max_util: float,
                 pre: dict[str, Any]) -> str:
        """Engineering interpretation generated from measured values only.
        This is NOT the agent's internal reasoning (see docs/REPORT.md)."""
        mtr = info["metrics"]
        if action == 0:
            if before_max_util < 0.8 and mtr["sla_violations"] == 0:
                return (f"No action. Max utilization {before_max_util:.0%}, no SLA "
                        f"violations — rerouting cost would exceed any benefit.")
            return (f"No action despite max utilization {before_max_util:.0%} and "
                    f"{mtr['sla_violations']} SLA violation(s) — the policy held "
                    f"position (valid alternatives may be masked or unattractive).")
        dec = info["decoded_action"]
        if not dec.get("accepted"):
            return (f"Proposed reroute of {dec['demand']} to path {dec['path_idx']} "
                    f"was REJECTED by the safety filter: {dec['reason']}.")
        d = self.eng.demand_by_id[dec["demand"]]
        return (f"Rerouted {d.cls.name} demand {d.id} ({d.src}→{d.dst}, "
                f"{pre['volume']:.0f} Mbps) from path {dec['from_path']} to "
                f"path {dec['path_idx']}. Old-path bottleneck was "
                f"{pre['old_bottleneck']:.0%}, projected new-path bottleneck "
                f"{pre['new_bottleneck']:.0%}. Network max utilization "
                f"{before_max_util:.0%} → {mtr['max_util']:.0%} over the "
                f"interval (includes demand change; see counterfactual for "
                f"the action's isolated effect).")

    def _step_baseline(self) -> dict[str, Any]:
        eng = self.eng
        moves = self.controller.decide(eng)
        applied = []
        log_mark = len(eng.action_log)
        for d_idx, p_idx in moves:
            ok, reason = eng.apply_action(
                d_idx, p_idx,
                source=self.controller.name if self.controller.name != "random" else "rl")
            applied.append({"demand": eng.demands[d_idx].id, "path_idx": p_idx,
                            "accepted": ok, "reason": reason})
        flapped = any(a.is_flap for a in eng.action_log[log_mark:] if a.accepted)
        interval = eng.step_interval()
        reward, comps = compute_reward(interval, rerouted=bool(applied), flapped=flapped,
                                       invalid=False)
        self._last_raw_reward = float(reward)
        self.cumulative_reward += float(reward)
        decision = {
            "algorithm": self.algorithm,
            "step": eng.step_count,
            "t_min": eng.t_min,
            "moves": applied,
            "reward": round(float(reward), 4),
            "components": {k: round(v, 4) for k, v in comps.items()},
            "cumulative_reward": round(self.cumulative_reward, 2),
            "done": eng.done,
            "explanation": (
                f"{self.algorithm} moved {len(applied)} demand(s)." if applied
                else f"{self.algorithm}: no reroute this interval."),
        }
        self.last_decision = decision
        return decision


class AlgoRunnerV2:
    """One controller bound to one frozen-V2 episode within a session.

    Every V2 runner — learner or baseline — drives a real :class:`MplsTeEnvV2`.
    A baseline only supplies the action integer; the transition, the
    authoritative mask, the validator reason and the twelve reward components
    all come from the governed environment, so a baseline lane and a learner
    lane are scored by exactly the same code.
    """

    environment_version = "v2"

    def __init__(self, cfg: SessionConfig, algorithm: str) -> None:
        from mplssim.experiments.evaluation_v2 import choose_baseline_action
        from mplssim.experiments.v2_factory import make_env_v2

        if algorithm not in V2_ALGORITHMS:
            raise ValueError(
                f"{algorithm!r} cannot run in the V2 environment. Available: "
                f"{', '.join(V2_ALGORITHMS)}.")
        self.algorithm = algorithm
        self.safety_filter = cfg.safety_filter
        self.training_root = int(cfg.training_root)
        self.checkpoint: checkpoints_v2.LoadedCheckpoint | None = None
        self.controller = None
        self._choose_baseline_action = choose_baseline_action

        if algorithm in checkpoints_v2.LEARNER_ALGORITHMS:
            # Fail closed here, before a session exists: an unavailable or
            # incompatible checkpoint must never degrade into a V1 run.
            self.checkpoint = checkpoints_v2.load(algorithm, self.training_root)
        else:
            self.controller = make_baseline(algorithm, seed=cfg.seed)

        self.env = make_env_v2(scenario=cfg.scenario, root_seed=cfg.seed)
        obs, _ = self.env.reset(options={"episode_seed": cfg.seed})
        self.eng = EngineV2View(self.env.eng)
        self._env_obs = obs
        # Only a learner consumes an observation vector. A baseline reads engine
        # state directly, and claiming otherwise would fake an inspector panel.
        self._obs = obs if self.checkpoint is not None else None
        self._prior_obs = None
        self._last_raw_reward = 0.0
        self.model = self.checkpoint.policy if self.checkpoint else None
        self.last_decision: dict[str, Any] | None = None
        self.cumulative_reward = 0.0
        self.history: list[dict[str, Any]] = []

    # -------------------------------------------------------------- identity
    @property
    def checkpoint_id(self) -> str | None:
        return self.checkpoint.entry.id if self.checkpoint else None

    @property
    def output_semantics(self) -> str:
        return self.checkpoint.output_semantics if self.checkpoint else "none"

    def provenance(self) -> dict[str, Any] | None:
        return self.checkpoint.provenance() if self.checkpoint else None

    # ------------------------------------------------------------- decisions
    def _predict(self, mask: np.ndarray) -> tuple[int, list[float | None] | None]:
        if self.checkpoint is not None:
            action = self.checkpoint.predict(self._env_obs, mask)
            return action, self.checkpoint.action_scores(self._env_obs, mask)
        action = self._choose_baseline_action(self.controller, self.eng, mask)
        return int(action), None

    def _action_desc(self, action: int) -> str:
        if action == 0:
            return "no TE change"
        d_idx, p_idx = divmod(action - 1, self.env.k)
        demand = self.eng.demands[d_idx]
        return (f"{demand.id} -> path {p_idx} "
                f"({'>'.join(demand.candidate_paths[p_idx])})")

    def step(self, counterfactual: bool = True,
             action_override: int | None = None) -> dict[str, Any]:
        env = self.env
        mask = env.action_masks()
        scores = None
        if action_override is not None:
            action = int(action_override)
        else:
            action, scores = self._predict(mask)

        before_max_util = float(np.max(self.eng.link_util))
        pre_state = self._pre_action_context(action)
        self._prior_obs = self._obs
        obs, reward, _terminated, truncated, info = env.step(action)
        self._env_obs = obs
        if self.checkpoint is not None:
            self._obs = obs
        self._last_raw_reward = float(reward)
        self.cumulative_reward += float(reward)

        interval = info["metrics"]
        decision: dict[str, Any] = {
            "algorithm": self.algorithm,
            "environment_version": "v2",
            "step": self.eng.step_count,
            "t_min": self.eng.t_min,
            "action": action,
            "decoded": info["decoded_action"],
            "reward": round(float(reward), 4),
            "components": {k: round(float(v), 6)
                           for k, v in info["reward_components"].items()},
            "component_order": list(info["reward_component_order"]),
            "cumulative_reward": round(self.cumulative_reward, 2),
            "mask_valid_actions": int(mask.sum()),
            "done": bool(truncated),
            "accepted_te_changes": int(interval["accepted_te_changes"]),
            "rejected_te_requests": int(interval["rejected_te_requests"]),
            "te_reversals": int(interval["te_reversals"]),
            "frr_changes": int(interval["frr_changes"]),
            "recovery_restorations": int(interval["recovery_restorations"]),
            "explanation": self._explain(action, info, before_max_util, pre_state),
        }
        if scores is not None:
            decision["output_semantics"] = self.output_semantics
            ranked = sorted(
                ((i, v) for i, v in enumerate(scores) if v is not None),
                key=lambda row: -row[1])[:5]
            decision["output_value"] = (
                None if scores[action] is None else round(float(scores[action]), 6))
            decision["top_actions"] = [
                {"action": int(i), "value": round(float(v), 6),
                 "desc": self._action_desc(int(i))} for i, v in ranked]
        self.history.append({
            "step": interval["step"],
            "t_min": interval["t_min"],
            "reward": self._last_raw_reward,
            "components": decision["components"],
            "metrics": {k: v for k, v in interval.items() if k != "failed_links"},
            "n_failed_links": len(interval["failed_links"]),
        })
        self.last_decision = decision
        return decision

    def _pre_action_context(self, action: int) -> dict[str, Any]:
        if action == 0:
            return {}
        d_idx, p_idx = divmod(action - 1, self.env.k)
        return {
            "old_bottleneck": self.eng.path_bottleneck_util(
                d_idx, int(self.eng.current_path[d_idx])),
            "new_bottleneck": self.eng.projected_bottleneck_after_move(d_idx, p_idx),
            "volume": float(self.eng.demand_volumes[d_idx]),
        }

    def _explain(self, action: int, info: dict[str, Any],
                 before_max_util: float, pre: dict[str, Any]) -> str:
        """Engineering interpretation from measured values only.

        This is never the controller's internal reasoning; a bandit's head value
        is an immediate-reward estimate and a PPO probability is a distribution
        mass, and neither is a stated intention.
        """
        metrics = info["metrics"]
        if action == 0:
            if before_max_util < 0.8 and metrics["sla_violations"] == 0:
                return (f"No TE change. Busiest link {before_max_util:.0%}, no SLA "
                        f"violations this interval.")
            return (f"No TE change despite busiest link {before_max_util:.0%} and "
                    f"{metrics['sla_violations']} SLA violation(s). Alternatives "
                    f"may be masked by dwell, a failed link or the protected "
                    f"projected-utilization rule.")
        decoded = info["decoded_action"]
        if not decoded.get("accepted"):
            return (f"TE request for {decoded.get('demand')} to candidate "
                    f"{decoded.get('path_idx')} was rejected by the V2 validator: "
                    f"{decoded.get('reason')}.")
        demand = self.eng.demand_by_id[decoded["demand"]]
        return (f"Moved {demand.cls.name} demand {demand.id} "
                f"({demand.src}→{demand.dst}, {pre['volume']:.0f} Mbps) from "
                f"candidate {decoded['from_path']} to candidate "
                f"{decoded['path_idx']}. Old-path bottleneck {pre['old_bottleneck']:.0%}, "
                f"projected gross bottleneck on the new path "
                f"{pre['new_bottleneck']:.0%}. Busiest link {before_max_util:.0%} → "
                f"{metrics['max_util']:.0%} over the interval, which also includes "
                f"the demand change.")

    # -------------------------------------------------------- counterfactual
    def evaluate_action_vs_noop(self, action: int) -> dict[str, Any]:
        """Cloned-engine one-interval lookahead. Never touches the live engine."""
        keys = ("max_util", "mean_delay_ms", "loss_ratio", "sla_violations",
                "delivered_ratio")
        noop_engine = self.eng.clone()
        noop_metrics = noop_engine.step_interval()
        out: dict[str, Any] = {
            "noop": {k: round(float(noop_metrics[k]), 4) for k in keys}}
        if action > 0:
            action_engine = self.eng.clone()
            d_idx, p_idx = divmod(action - 1, self.env.k)
            accepted, reason = action_engine.apply_action(d_idx, p_idx, source="rl")
            action_metrics = action_engine.step_interval()
            out["action"] = {k: round(float(action_metrics[k]), 4) for k in keys}
            out["action_applied"] = accepted
            out["action_reason"] = reason
            out["delta_max_util"] = round(
                float(action_metrics["max_util"] - noop_metrics["max_util"]), 4)
        return out


def make_runner(cfg: SessionConfig, algorithm: str):
    """The one place an environment version selects a runner implementation."""
    if cfg.environment == "v2":
        return AlgoRunnerV2(cfg, algorithm)
    if cfg.environment == "v1":
        return AlgoRunner(cfg, algorithm)
    raise ValueError(f"environment must be one of {list(ENVIRONMENTS)}")


class SimSession:
    """A live session: one or two AlgoRunners on paired engines, an explicit
    state machine, and single-lock concurrency control."""

    def __init__(self, config: SessionConfig) -> None:
        if not 1 <= len(config.algorithms) <= 2:
            raise ValueError("1 or 2 algorithms")
        if config.speed not in SPEED_SECONDS:
            raise ValueError(f"speed must be one of {list(SPEED_SECONDS)}")
        self.config = config
        # Stable identity for product payloads. A client that holds a prior
        # snapshot must be able to tell "same run, one step later" apart from
        # "the session was reset underneath me" before it renders a delta.
        self.id = uuid4().hex[:12]
        self.sequence = 0
        self.runners = [make_runner(config, a) for a in config.algorithms]
        self.state = SessionState.IDLE
        self.speed = config.speed
        self.error_message: str | None = None
        self._lock = asyncio.Lock()
        self._loop_task: asyncio.Task | None = None
        self._generation = 0
        self.subscribers: list[Callable[[dict], Any]] = []
        # operator-advisor state
        self.pending_proposal: dict[str, Any] | None = None
        self.advisor_history: list[dict[str, Any]] = []
        #: Completed generations kept for later comparison. "Reset run" archives
        #: the run it replaces instead of discarding it.
        self.previous_runs: list[dict[str, Any]] = []
        log_event("session_created", scenario=config.scenario,
                  environment=config.environment,
                  algorithm="+".join(config.algorithms), seed=config.seed,
                  model_tag=config.model_tag, training_root=config.training_root,
                  safety_filter=config.safety_filter,
                  speed=config.speed, execution=config.execution)

    # ---------------------------------------------------------------- status
    @property
    def done(self) -> bool:
        return any(r.eng.done for r in self.runners)

    def _log_ctx(self) -> dict[str, Any]:
        return dict(scenario=self.config.scenario,
                    algorithm="+".join(self.config.algorithms),
                    seed=self.config.seed,
                    step=self.runners[0].eng.step_count,
                    t_min=self.runners[0].eng.t_min)

    @property
    def generation(self) -> int:
        """Bumped by reset. A delta computed across generations is meaningless."""
        return self._generation

    def status(self) -> dict[str, Any]:
        eng = self.runners[0].eng
        return {
            "state": self.state.value,
            "error": self.error_message,
            "session_id": self.id,
            "generation": self._generation,
            "sequence": self.sequence,
            "scenario": self.config.scenario,
            "algorithms": list(self.config.algorithms),
            "seed": self.config.seed,
            "environment": self.config.environment,
            "training_root": (self.config.training_root
                              if self.config.environment == "v2" else None),
            "model_tag": self.config.model_tag,
            "controllers": [
                {"algorithm": r.algorithm,
                 "environment_version": r.environment_version,
                 "checkpoint_id": r.checkpoint_id,
                 "output_semantics": r.output_semantics}
                for r in self.runners
            ],
            "safety_filter": self.config.safety_filter,
            "advisor": self.config.advisor,
            "execution": self.config.execution,
            "interface_mode": self.config.interface_mode,
            "speed": self.speed,
            "running": self.state == SessionState.RUNNING,
            "done": self.done,
            "awaiting_decision": self.pending_proposal is not None,
            "step": eng.step_count,
            "t_min": eng.t_min,
            "hour": round((eng.scenario.start_hour + eng.t_min / 60.0) % 24.0, 3),
            "duration_min": eng.scenario.duration_min,
            "retained_runs": len(self.previous_runs),
        }

    def payload(self, decisions: list[dict] | None = None,
                kind: str = "tick") -> dict[str, Any]:
        # Monotonic per emitted payload: a client can drop an out-of-order or
        # replayed frame instead of rendering it as the present.
        self.sequence += 1
        return {
            "type": kind,
            "status": self.status(),
            "runs": [
                {"algorithm": r.algorithm,
                 "snapshot": r.eng.snapshot(),
                 "decision": d if d is not None else r.last_decision}
                for r, d in zip(self.runners,
                                decisions or [None] * len(self.runners))
            ],
        }

    async def _broadcast(self, payload: dict) -> None:
        for cb in list(self.subscribers):
            try:
                await cb(payload)
            except Exception:
                self.subscribers.remove(cb)

    async def broadcast_snapshot(self, kind: str = "intervention") -> None:
        """Immediate out-of-band snapshot (no time advance)."""
        await self._broadcast(self.payload(kind=kind))

    # ------------------------------------------------------------------ loop
    def _step_all(self) -> dict[str, Any]:
        decisions = [r.step() for r in self.runners]
        return self.payload(decisions)

    async def _loop(self, generation: int) -> None:
        try:
            while True:
                # Advisor execution never advances on its own. Resuming builds
                # the next proposal and stops there; only approve or reject
                # applies anything.
                if self.config.advisor:
                    async with self._lock:
                        if self._generation != generation:
                            return
                        if self.state != SessionState.RUNNING:
                            return
                        if self.pending_proposal is not None:
                            return
                    await self.advisor_propose()
                    return
                async with self._lock:
                    if self._generation != generation:
                        return  # superseded by reset — exit silently
                    if self.state != SessionState.RUNNING:
                        return
                    if self.done:
                        self.state = SessionState.COMPLETED
                        log_event("session_completed", **self._log_ctx())
                        break
                    t0 = time.perf_counter()
                    payload = await asyncio.to_thread(self._step_all)
                    elapsed = time.perf_counter() - t0
                if self._generation != generation:
                    return  # reset raced the release — drop the stale payload
                await self._broadcast(payload)
                delay = SPEED_SECONDS[self.speed] - elapsed
                await asyncio.sleep(max(delay, 0.0))
            await self._broadcast({"type": "status", "status": self.status()})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # trap state; never leave a zombie loop
            self.state = SessionState.ERROR
            self.error_message = f"{type(exc).__name__}: {exc}"
            log_event("session_error", level=40, error=self.error_message,
                      **self._log_ctx())
            await self._broadcast({"type": "status", "status": self.status()})

    # ------------------------------------------------------------- lifecycle
    async def resume(self) -> dict[str, Any]:
        async with self._lock:
            if self.state == SessionState.COMPLETED:
                raise SessionError("scenario finished — reset before resuming")
            if self.state == SessionState.ERROR:
                raise SessionError(f"session is in error state: {self.error_message}")
            if self.pending_proposal is not None:
                raise SessionError("an advisor recommendation is awaiting a decision")
            if self.state == SessionState.RUNNING:
                return self.status()  # idempotent
            self.state = SessionState.RUNNING
            if self._loop_task is None or self._loop_task.done():
                self._loop_task = asyncio.get_running_loop().create_task(
                    self._loop(self._generation))
            log_event("session_resumed", **self._log_ctx())
        return self.status()

    async def pause(self) -> dict[str, Any]:
        async with self._lock:
            if self.state == SessionState.RUNNING:
                self.state = SessionState.PAUSED
                log_event("session_paused", **self._log_ctx())
            # PAUSED/IDLE/COMPLETED: idempotent no-op
        return self.status()

    async def step_manual(self) -> dict[str, Any]:
        # In advisor execution a step is a *request for a recommendation*, not
        # an application of one. The action is proposed and held; approve or
        # reject is what advances the clock.
        if self.config.advisor and self.pending_proposal is None and not self.done:
            await self.advisor_propose()
            return self.payload(kind="advisor")
        async with self._lock:
            if self.state == SessionState.RUNNING:
                raise SessionError("pause the simulation before stepping manually")
            if self.state in (SessionState.COMPLETED,) or self.done:
                raise SessionError("scenario finished — reset to run again")
            if self.pending_proposal is not None:
                raise SessionError("resolve the advisor recommendation first "
                                   "(approve or reject)")
            if self.state == SessionState.IDLE:
                self.state = SessionState.PAUSED
            payload = await asyncio.to_thread(self._step_all)
            if self.done:
                self.state = SessionState.COMPLETED
            log_event("manual_step", **self._log_ctx())
        await self._broadcast(payload)
        return payload

    async def set_speed(self, speed: str) -> dict[str, Any]:
        if speed not in SPEED_SECONDS:
            raise ValueError(f"speed must be one of {list(SPEED_SECONDS)}")
        async with self._lock:
            self.speed = speed
            log_event("speed_changed", speed=speed, **self._log_ctx())
        return self.status()

    def archive(self) -> dict[str, Any] | None:
        """Snapshot the run about to be replaced, so it survives a reset run.

        Holds measured history only — no model state, no evidence — and is never
        promoted to a scientific record.
        """
        if not any(r.history for r in self.runners):
            return None
        return {
            "generation": self._generation,
            "environment": self.config.environment,
            "scenario": self.config.scenario,
            "seed": self.config.seed,
            "training_root": self.config.training_root,
            "steps": max(len(r.history) for r in self.runners),
            "runs": [{
                "algorithm": r.algorithm,
                "checkpoint_id": r.checkpoint_id,
                "cumulative_reward": round(float(r.cumulative_reward), 4),
                "history": list(r.history),
            } for r in self.runners],
        }

    async def reset(self, retain: bool = True) -> dict[str, Any]:
        """Reset run: rebuild the EXACT same experiment at time zero.

        Same SessionConfig — environment, scenario, seed, controllers,
        checkpoint root, safety filter, speed, interface mode. The run being
        replaced is archived when ``retain`` is set, so Part 2's comparison can
        still read it. No model, checkpoint or evidence artifact is touched.
        """
        async with self._lock:
            archived = self.archive() if retain else None
            self._generation += 1
            task = self._loop_task
            self._loop_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            if archived is not None:
                self.previous_runs.append(archived)
            self.runners = [make_runner(self.config, a)
                            for a in self.config.algorithms]
            self.state = SessionState.IDLE
            self.error_message = None
            self.pending_proposal = None
            self.advisor_history = []
            log_event("session_reset", retained=archived is not None,
                      **self._log_ctx())
        await self.broadcast_snapshot(kind="reset")
        return self.status()

    async def shutdown(self) -> None:
        """Full reset: stop the loop and leave no runnable state behind.

        The caller drops the session afterwards; nothing here mutates a model,
        a checkpoint or any evidence artifact.
        """
        async with self._lock:
            self._generation += 1
            self.state = SessionState.IDLE
            self.pending_proposal = None
            task = self._loop_task
            self._loop_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.subscribers.clear()
        log_event("session_shutdown", **self._log_ctx())

    # --------------------------------------------------------- interventions
    async def inject_failure(self, link_id: str) -> dict[str, Any]:
        async with self._lock:
            already = not self.runners[0].eng.link_up.get(link_id, True)
            frr_before = [len(r.eng.action_log) for r in self.runners]
            for r in self.runners:   # paired: both engines in compare mode
                r.eng.inject_failure(link_id)
            frr_moves = [len(r.eng.action_log) - b
                         for r, b in zip(self.runners, frr_before)]
            log_event("link_failed", link=link_id, already_failed=already,
                      frr_reroutes=sum(frr_moves), **self._log_ctx())
        await self.broadcast_snapshot()
        return {"ok": True, "changed": not already, "link": link_id,
                "frr_reroutes": sum(frr_moves),
                "failed_links": [l for l, up in self.runners[0].eng.link_up.items()
                                 if not up]}

    async def recover_link(self, link_id: str) -> dict[str, Any]:
        async with self._lock:
            already = self.runners[0].eng.link_up.get(link_id, True)
            for r in self.runners:
                r.eng.recover_link(link_id)
            log_event("link_recovered", link=link_id, already_up=already,
                      **self._log_ctx())
        await self.broadcast_snapshot()
        return {"ok": True, "changed": not already, "link": link_id,
                "failed_links": [l for l, up in self.runners[0].eng.link_up.items()
                                 if not up]}

    def _require_v1_traffic_override(self) -> None:
        if self.config.environment != "v1":
            from mplssim.product.live_v2 import UNSUPPORTED_INTERVENTION
            raise SessionError(UNSUPPORTED_INTERVENTION)

    async def inject_burst(self, demand_id: str, factor: float,
                           duration_min: float) -> dict[str, Any]:
        self._require_v1_traffic_override()
        async with self._lock:
            for r in self.runners:
                r.eng.inject_burst(demand_id, factor, duration_min)
            log_event("burst_injected", demand=demand_id, factor=factor,
                      duration_min=duration_min, **self._log_ctx())
        await self.broadcast_snapshot()
        return {"ok": True}

    async def set_multiplier(self, factor: float) -> dict[str, Any]:
        self._require_v1_traffic_override()
        async with self._lock:
            for r in self.runners:
                r.eng.manual_multiplier = factor
            log_event("multiplier_set", factor=factor, **self._log_ctx())
        await self.broadcast_snapshot()
        return {"ok": True, "factor": factor}

    # ---------------------------------------------------------- run helpers
    #: Refusal shown when advisor execution is asked to fast-forward without the
    #: caller saying, in the request, that it is delegating those intervals.
    DELEGATION_REQUIRED = (
        "This session runs with advisor approval, so every action is supposed to "
        "be approved individually. A fast-forward cannot do that: it applies the "
        "controller's own actions for a stretch of intervals in one gesture. "
        "Send delegate=true to authorize that stretch explicitly. It is recorded "
        "in the approval ledger as one delegated batch, not as a run of "
        "individual approvals.")

    async def run_until(self, condition: str, max_steps: int = 300,
                        util_threshold: float = 0.9,
                        delegate: bool = False) -> dict[str, Any]:
        """Fast-forward (used by Guided Story's 'Next Event'):
        condition = 'next_event' | 'congestion' | 'failure' | 'recovery' |
        'end'. Steps synchronously
        under the lock; no wall-clock pacing; state must not be RUNNING.

        Under advisor execution the caller must pass ``delegate=True``. Part 1
        left this as a disclosed asymmetry; Part 2 closes it by making the
        delegation an explicit, recorded decision rather than a footnote on a
        response the operator may never read (docs/ADR-003).
        """
        async with self._lock:
            allowed = {"next_event", "congestion", "failure", "recovery", "end"}
            if condition not in allowed:
                raise SessionError(f"unknown run-until condition: {condition}")
            if self.state == SessionState.RUNNING:
                raise SessionError("pause before fast-forwarding")
            if self.done:
                raise SessionError("scenario finished")
            if self.pending_proposal is not None:
                raise SessionError("resolve the advisor recommendation first")
            if self.config.advisor and not delegate:
                raise SessionError(self.DELEGATION_REQUIRED)
            if self.state == SessionState.IDLE:
                self.state = SessionState.PAUSED
            eng = self.runners[0].eng
            failure_seen = any(not up for up in eng.link_up.values())
            next_event_t = None
            if condition == "next_event":
                future = [ev["t_min"] for ev in eng.scenario.events
                          if ev["t_min"] > eng.t_min]
                next_event_t = min(future) if future else None
            steps = 0
            payload = None
            while steps < max_steps and not self.done:
                payload = await asyncio.to_thread(self._step_all)
                steps += 1
                if condition == "next_event":
                    if next_event_t is None or eng.t_min >= next_event_t:
                        break
                elif condition == "congestion":
                    if payload["runs"][0]["snapshot"]["metrics"]["max_util"] >= util_threshold:
                        break
                elif condition == "failure":
                    if any(not up for up in eng.link_up.values()):
                        break
                elif condition == "recovery":
                    failure_seen = failure_seen or any(
                        not up for up in eng.link_up.values()
                    )
                    if failure_seen and all(eng.link_up.values()):
                        break
            if self.done:
                self.state = SessionState.COMPLETED
            # A delegated stretch enters the approval ledger as ONE record, so
            # the history cannot be read as a run of individual approvals. It is
            # the operator's decision, and it is written down as such.
            if self.config.advisor and steps:
                engine = self.runners[0].eng
                start = engine.step_count - steps
                self.advisor_history.append({
                    "id": len(self.advisor_history) + 1,
                    "kind": "delegated_batch",
                    "approved": True,
                    "delegated": True,
                    "condition": condition,
                    "steps": steps,
                    "from_step": int(start),
                    "to_step": int(engine.step_count),
                    # `step`/`t_min` name the interval the stretch ENDED on, so
                    # every ledger record can be placed on the timeline without
                    # the reader having to know which kind it is.
                    "step": int(engine.step_count),
                    "t_min": float(engine.t_min),
                    "policy_id": self.config.algorithms[0],
                    "note": ("The operator delegated this stretch in one "
                             "gesture. The controller's own actions were "
                             "applied; no individual action was approved."),
                })
            log_event("run_until", condition=condition, steps=steps,
                      delegated=bool(self.config.advisor), **self._log_ctx())
        if payload is not None:
            await self._broadcast(payload)
        return {
            "steps": steps,
            "status": self.status(),
            # A fast-forward is the operator delegating a stretch of intervals
            # in one gesture. In advisor execution those intervals are not
            # individually approved, and both the response and the approval
            # ledger say so rather than letting the UI imply that each one was.
            "approval_bypassed": bool(self.config.advisor),
            "delegated": bool(self.config.advisor),
            "note": (f"You delegated {steps} interval(s). The controller's own "
                     f"actions were applied; none of them was approved "
                     f"individually. The approval history records this as one "
                     f"delegated batch."
                     if self.config.advisor else
                     "Fast-forward ran the controller normally."),
        }

    # ---------------------------------------------------------------- advisor
    #: Controllers whose proposal is a genuine *policy recommendation*. A
    #: rule-based baseline has no recommendation to approve; it either runs or
    #: it does not.
    _POLICY_ALGORITHMS: frozenset[str] = frozenset(
        {"rl", *checkpoints_v2.LEARNER_ALGORITHMS})

    def _policy_runner(self):
        for r in self.runners:
            if r.algorithm in self._POLICY_ALGORITHMS:
                return r
        raise SessionError(
            "Advisor approval needs a learned policy in the session. "
            f"{' and '.join(self.config.algorithms)} "
            "propose moves from fixed rules, so there is nothing to approve.")

    async def advisor_propose(self) -> dict[str, Any]:
        """Generate a recommendation WITHOUT mutating the real engine, pause
        the session, and await an operator decision.

        Only reachable in advisor execution. In automatic execution the policy
        already acted, so there is nothing to propose and nothing to approve —
        the completed decision is shown as an explanation instead.
        """
        async with self._lock:
            if not self.config.advisor:
                raise SessionError(
                    "This session runs the policy automatically, so there is no "
                    "proposal awaiting approval. Start a session with "
                    "manual/advisor execution to approve or reject each action.")
            if self.done or self.state == SessionState.COMPLETED:
                raise SessionError("scenario finished")
            if self.pending_proposal is not None:
                return self.pending_proposal  # idempotent
            r = self._policy_runner()
            mask = r.env.action_masks()
            action, outputs = r._predict(mask)
            ok, reason = (True, "no-op") if action == 0 else \
                r.eng.validate_action(*divmod(action - 1, r.env.k), source="rl")
            lookahead = await asyncio.to_thread(r.evaluate_action_vs_noop, action)
            selected_output = None
            if outputs is not None:
                raw = outputs[action] if action < len(outputs) else None
                selected_output = None if raw is None else round(float(raw), 6)
            proposal = {
                "id": len(self.advisor_history) + 1,
                # The ledger holds two kinds of record: a proposal an operator
                # answered, and a stretch an operator delegated. A client that
                # renders one as the other would misreport what was approved.
                "kind": "proposal",
                "proposed_at": time.time(),
                "step": r.eng.step_count,
                "t_min": r.eng.t_min,
                "action": action,
                "is_noop": action == 0,
                "decoded": None,
                "policy_id": r.algorithm,
                "environment_version": r.environment_version,
                "checkpoint_id": r.checkpoint_id,
                # Named by the controller's declared semantics: a bandit head
                # value is an immediate-reward estimate, never a probability.
                "output_semantics": r.output_semantics,
                "output_value": selected_output,
                "safety_ok": ok,
                "safety_reason": reason,
                "lookahead": lookahead,
            }
            if action > 0:
                d_idx, p_idx = divmod(action - 1, r.env.k)
                d = r.eng.demands[d_idx]
                proposal["decoded"] = {
                    "demand": d.id, "demand_idx": d_idx, "path_idx": p_idx,
                    "from_path": int(r.eng.current_path[d_idx]),
                    "from_routers": list(d.candidate_paths[int(r.eng.current_path[d_idx])]),
                    "to_routers": list(d.candidate_paths[p_idx]),
                    "class": d.cls.name,
                    "volume_mbps": round(float(r.eng.demand_volumes[d_idx]), 1),
                    "src": d.src, "dst": d.dst,
                }
            self.pending_proposal = proposal
            if self.state == SessionState.RUNNING:
                self.state = SessionState.PAUSED
            elif self.state == SessionState.IDLE:
                self.state = SessionState.PAUSED
            log_event("advisor_proposal", action=action, safety_ok=ok,
                      **self._log_ctx())
        await self._broadcast({"type": "advisor", "proposal": self.pending_proposal,
                               "status": self.status()})
        return self.pending_proposal

    async def _advisor_decide(self, approve: bool) -> dict[str, Any]:
        async with self._lock:
            if self.pending_proposal is None:
                raise SessionError("no advisor recommendation is pending")
            proposal = self.pending_proposal
            self.pending_proposal = None
            r = self._policy_runner()
            applied_action = int(proposal["action"]) if approve else 0
            decisions = []
            for runner in self.runners:
                if runner is r:
                    decisions.append(runner.step(action_override=applied_action))
                else:
                    decisions.append(runner.step())
            actual = r.history[-1]["metrics"]
            record = {
                **proposal,
                "approved": approve,
                "operator_response_s": round(time.time() - proposal["proposed_at"], 1),
                "applied_action": applied_action,
                "actual": {k: round(float(actual[k]), 4) for k in
                           ("max_util", "mean_delay_ms", "loss_ratio",
                            "sla_violations", "delivered_ratio")},
                "reward": r.history[-1]["reward"],
            }
            self.advisor_history.append(record)
            if self.done:
                self.state = SessionState.COMPLETED
            log_event("advisor_approved" if approve else "advisor_rejected",
                      action=applied_action, **self._log_ctx())
            payload = self.payload(decisions)
        await self._broadcast(payload)
        return record

    async def advisor_approve(self) -> dict[str, Any]:
        return await self._advisor_decide(True)

    async def advisor_reject(self) -> dict[str, Any]:
        return await self._advisor_decide(False)

    def advisor_status(self) -> dict[str, Any]:
        history = self.advisor_history[-20:]
        return {
            "pending": self.pending_proposal,
            "history": history,
            "proposals": [r for r in history if r.get("kind") != "delegated_batch"],
            "delegated_batches": [r for r in history
                                  if r.get("kind") == "delegated_batch"],
            "delegated_intervals": sum(
                int(r.get("steps", 0)) for r in self.advisor_history
                if r.get("kind") == "delegated_batch"),
            "enabled": self.config.advisor,
            "execution": self.config.execution,
            "explanation_only": not self.config.advisor,
            "note": ("Automatic execution: the policy has already acted and the "
                     "card below explains the completed decision."
                     if not self.config.advisor else
                     "Advisor execution: the proposed action is held until you "
                     "approve or reject it."),
        }
