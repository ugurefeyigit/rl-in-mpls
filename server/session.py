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

import numpy as np

from mplssim.baselines import make_baseline
from mplssim.factory import engine_config_from_training, make_engine
from mplssim.rl.env import MplsTeEnv
from mplssim.rl.reward import compute_reward
from mplssim.sim.engine import SimulationEngine
from server.events import log_event

ROOT = Path(__file__).resolve().parents[1]
SPEED_SECONDS = {"1x": 2.0, "5x": 0.4, "20x": 0.1, "fast": 0.0}

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
    advisor: bool = False              # operator-advisor mode (RL runner only)


class AlgoRunner:
    """One algorithm bound to one engine within a session."""

    def __init__(self, cfg: SessionConfig, algorithm: str) -> None:
        self.algorithm = algorithm
        self.safety_filter = cfg.safety_filter
        self.model = None
        self.env: MplsTeEnv | None = None
        self.controller = None
        if algorithm == "rl":
            self.model = load_model(cfg.model_tag or "ppo_te")
            self.env = MplsTeEnv(scenario=cfg.scenario, base_seed=cfg.seed,
                                 safety_filter=cfg.safety_filter)
            self.env.reset(options={"episode_seed": cfg.seed})
            self.eng: SimulationEngine = self.env.eng
            self._obs = self.env._obs()
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
            decision["action_probability"] = round(float(probs[action]), 4)
            decision["top_actions"] = [
                {"action": int(a), "prob": round(float(probs[a]), 4),
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


class SimSession:
    """A live session: one or two AlgoRunners on paired engines, an explicit
    state machine, and single-lock concurrency control."""

    def __init__(self, config: SessionConfig) -> None:
        if not 1 <= len(config.algorithms) <= 2:
            raise ValueError("1 or 2 algorithms")
        if config.speed not in SPEED_SECONDS:
            raise ValueError(f"speed must be one of {list(SPEED_SECONDS)}")
        self.config = config
        self.runners = [AlgoRunner(config, a) for a in config.algorithms]
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
        log_event("session_created", scenario=config.scenario,
                  algorithm="+".join(config.algorithms), seed=config.seed,
                  model_tag=config.model_tag, safety_filter=config.safety_filter,
                  speed=config.speed, advisor=config.advisor)

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

    def status(self) -> dict[str, Any]:
        eng = self.runners[0].eng
        return {
            "state": self.state.value,
            "error": self.error_message,
            "scenario": self.config.scenario,
            "algorithms": list(self.config.algorithms),
            "seed": self.config.seed,
            "model_tag": self.config.model_tag,
            "safety_filter": self.config.safety_filter,
            "advisor": self.config.advisor,
            "interface_mode": self.config.interface_mode,
            "speed": self.speed,
            "running": self.state == SessionState.RUNNING,
            "done": self.done,
            "awaiting_decision": self.pending_proposal is not None,
            "step": eng.step_count,
            "t_min": eng.t_min,
            "hour": round((eng.scenario.start_hour + eng.t_min / 60.0) % 24.0, 3),
            "duration_min": eng.scenario.duration_min,
        }

    def payload(self, decisions: list[dict] | None = None,
                kind: str = "tick") -> dict[str, Any]:
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

    async def reset(self) -> dict[str, Any]:
        """Cancel and await the loop, then rebuild the EXACT same experiment
        (same SessionConfig — scenario, algorithms, model tag, seed, safety
        filter, speed, interface mode) at time zero."""
        async with self._lock:
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
            self.runners = [AlgoRunner(self.config, a) for a in self.config.algorithms]
            self.state = SessionState.IDLE
            self.error_message = None
            self.pending_proposal = None
            self.advisor_history = []
            log_event("session_reset", **self._log_ctx())
        await self.broadcast_snapshot(kind="reset")
        return self.status()

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

    async def inject_burst(self, demand_id: str, factor: float,
                           duration_min: float) -> dict[str, Any]:
        async with self._lock:
            for r in self.runners:
                r.eng.inject_burst(demand_id, factor, duration_min)
            log_event("burst_injected", demand=demand_id, factor=factor,
                      duration_min=duration_min, **self._log_ctx())
        await self.broadcast_snapshot()
        return {"ok": True}

    async def set_multiplier(self, factor: float) -> dict[str, Any]:
        async with self._lock:
            for r in self.runners:
                r.eng.manual_multiplier = factor
            log_event("multiplier_set", factor=factor, **self._log_ctx())
        await self.broadcast_snapshot()
        return {"ok": True, "factor": factor}

    # ---------------------------------------------------------- run helpers
    async def run_until(self, condition: str, max_steps: int = 300,
                        util_threshold: float = 0.9) -> dict[str, Any]:
        """Fast-forward (used by Guided Story's 'Next Event'):
        condition = 'next_event' | 'congestion' | 'end'. Steps synchronously
        under the lock; no wall-clock pacing; state must not be RUNNING."""
        async with self._lock:
            if self.state == SessionState.RUNNING:
                raise SessionError("pause before fast-forwarding")
            if self.done:
                raise SessionError("scenario finished")
            if self.pending_proposal is not None:
                raise SessionError("resolve the advisor recommendation first")
            if self.state == SessionState.IDLE:
                self.state = SessionState.PAUSED
            eng = self.runners[0].eng
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
            if self.done:
                self.state = SessionState.COMPLETED
            log_event("run_until", condition=condition, steps=steps,
                      **self._log_ctx())
        if payload is not None:
            await self._broadcast(payload)
        return {"steps": steps, "status": self.status()}

    # ---------------------------------------------------------------- advisor
    def _rl_runner(self) -> AlgoRunner:
        for r in self.runners:
            if r.algorithm == "rl":
                return r
        raise SessionError("advisor mode requires an RL runner in the session")

    async def advisor_propose(self) -> dict[str, Any]:
        """Generate a recommendation WITHOUT mutating the real engine, pause
        the session, and await an operator decision."""
        async with self._lock:
            if self.done or self.state == SessionState.COMPLETED:
                raise SessionError("scenario finished")
            if self.pending_proposal is not None:
                return self.pending_proposal  # idempotent
            r = self._rl_runner()
            mask = r.env.action_masks()
            action, probs = r._predict(mask)
            ok, reason = (True, "no-op") if action == 0 else \
                r.eng.validate_action(*divmod(action - 1, r.env.k), source="rl")
            lookahead = await asyncio.to_thread(r.evaluate_action_vs_noop, action)
            proposal = {
                "id": len(self.advisor_history) + 1,
                "proposed_at": time.time(),
                "step": r.eng.step_count,
                "t_min": r.eng.t_min,
                "action": action,
                "is_noop": action == 0,
                "decoded": None,
                "action_probability": (round(float(probs[action]), 4)
                                       if probs is not None else None),
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
            r = self._rl_runner()
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
        return {
            "pending": self.pending_proposal,
            "history": self.advisor_history[-20:],
            "enabled": self.config.advisor,
        }
