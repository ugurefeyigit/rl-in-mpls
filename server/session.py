"""Live simulation session management.

A session runs one scenario with one algorithm — or two algorithms
side-by-side on paired engines (identical scenario, seed, traffic and
manual interventions) for comparison mode.

Wall-clock pacing: speed "1x" is a *presentation* rate (one 5-simulated-minute
control interval every 2 s), not real time. "fast" steps as quickly as the
event loop allows.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mplssim.baselines import make_baseline
from mplssim.factory import engine_config_from_training, make_engine
from mplssim.rl.env import MplsTeEnv
from mplssim.rl.reward import compute_reward
from mplssim.sim.engine import SimulationEngine

ROOT = Path(__file__).resolve().parents[1]
SPEED_SECONDS = {"1x": 2.0, "5x": 0.4, "20x": 0.1, "fast": 0.0}

_MODEL_CACHE: dict[str, Any] = {}


def load_model(tag: str) -> Any:
    """Load a trained MaskablePPO checkpoint (models/<tag>/best_model.zip
    falling back to final_model.zip), cached across sessions."""
    if tag in _MODEL_CACHE:
        return _MODEL_CACHE[tag]
    from sb3_contrib import MaskablePPO
    base = ROOT / "models" / tag
    path = base / "best_model.zip"
    if not path.exists():
        path = base / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"no model found under {base}")
    model = MaskablePPO.load(path, device="cpu")
    _MODEL_CACHE[tag] = model
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


class AlgoRunner:
    """One algorithm bound to one engine within a session."""

    def __init__(self, algorithm: str, scenario: str, seed: int,
                 model_tag: str | None, safety_filter: bool) -> None:
        self.algorithm = algorithm
        self.safety_filter = safety_filter
        self.model = None
        self.env: MplsTeEnv | None = None
        self.controller = None
        if algorithm == "rl":
            self.model = load_model(model_tag or "ppo_te")
            self.env = MplsTeEnv(scenario=scenario, base_seed=seed,
                                 safety_filter=safety_filter)
            self.env.reset(options={"episode_seed": seed})
            self.eng: SimulationEngine = self.env.eng
            self._obs = self.env._obs()
        else:
            self.eng = make_engine(scenario, seed=seed, cfg=engine_config_from_training())
            self.controller = make_baseline(algorithm, seed=seed)
        self.last_decision: dict[str, Any] | None = None
        self.cumulative_reward = 0.0

    # ------------------------------------------------------------------ step
    def step(self, counterfactual: bool = True) -> dict[str, Any]:
        if self.algorithm == "rl":
            return self._step_rl(counterfactual)
        return self._step_baseline()

    def _action_desc(self, a: int) -> str:
        if a == 0:
            return "no-op"
        d_idx, p_idx = divmod(a - 1, self.env.k)
        d = self.eng.demands[d_idx]
        return f"{d.id} -> path {p_idx} ({'>'.join(d.candidate_paths[p_idx])})"

    def _step_rl(self, counterfactual: bool) -> dict[str, Any]:
        env = self.env
        mask = env.action_masks()
        action, _ = self.model.predict(self._obs, deterministic=True, action_masks=mask)
        action = int(action)

        # action probabilities under the mask (for the decision panel)
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

        cf_metrics = None
        if counterfactual and action != 0:
            cf_eng = self.eng.clone()
            cf_metrics = cf_eng.step_interval()  # what no-op would have done

        before_max_util = float(np.max(self.eng.link_util))
        pre_state = self._pre_action_context(action)
        self._obs, reward, terminated, truncated, info = env.step(action)
        self.cumulative_reward += reward

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
            "new_bottleneck": self.eng.path_bottleneck_util(d_idx, p_idx),
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
                f"{pre['old_bottleneck']:.0%}, new-path bottleneck "
                f"{pre['new_bottleneck']:.0%}. Network max utilization: "
                f"{before_max_util:.0%} → {mtr['max_util']:.0%}.")

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
        self.cumulative_reward += reward
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
    """A live session: one or two AlgoRunners on paired engines + async pacing."""

    def __init__(self, scenario: str, algorithms: list[str], seed: int,
                 model_tag: str | None = None, safety_filter: bool = True,
                 speed: str = "1x") -> None:
        if not 1 <= len(algorithms) <= 2:
            raise ValueError("1 or 2 algorithms")
        self.scenario = scenario
        self.seed = seed
        self.algorithms = algorithms
        self.speed = speed if speed in SPEED_SECONDS else "1x"
        self.runners = [AlgoRunner(a, scenario, seed, model_tag, safety_filter)
                        for a in algorithms]
        self.running = False
        self._task: asyncio.Task | None = None
        self.subscribers: list[Callable[[dict], Any]] = []

    @property
    def done(self) -> bool:
        return any(r.eng.done for r in self.runners)

    def status(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "algorithms": self.algorithms,
            "seed": self.seed,
            "speed": self.speed,
            "running": self.running,
            "done": self.done,
            "step": self.runners[0].eng.step_count,
            "t_min": self.runners[0].eng.t_min,
            "hour": round((self.runners[0].eng.scenario.start_hour
                           + self.runners[0].eng.t_min / 60.0) % 24.0, 2),
            "duration_min": self.runners[0].eng.scenario.duration_min,
        }

    def payload(self, decisions: list[dict] | None = None) -> dict[str, Any]:
        return {
            "type": "tick",
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

    def step_once(self) -> dict[str, Any]:
        decisions = [r.step() for r in self.runners]
        return self.payload(decisions)

    async def _loop(self) -> None:
        while self.running and not self.done:
            t0 = time.perf_counter()
            payload = await asyncio.to_thread(self.step_once)
            await self._broadcast(payload)
            delay = SPEED_SECONDS[self.speed] - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)  # yield to the event loop
        self.running = False
        await self._broadcast({"type": "status", "status": self.status()})

    def start(self) -> None:
        if self.running or self.done:
            return
        self.running = True
        self._task = asyncio.get_running_loop().create_task(self._loop())

    def pause(self) -> None:
        self.running = False

    def set_speed(self, speed: str) -> None:
        if speed not in SPEED_SECONDS:
            raise ValueError(f"speed must be one of {list(SPEED_SECONDS)}")
        self.speed = speed

    # manual interventions applied to ALL runners (keeps comparison paired)
    def inject_failure(self, link_id: str) -> None:
        for r in self.runners:
            r.eng.inject_failure(link_id)

    def recover_link(self, link_id: str) -> None:
        for r in self.runners:
            r.eng.recover_link(link_id)

    def inject_burst(self, demand_id: str, factor: float, duration_min: float) -> None:
        for r in self.runners:
            r.eng.inject_burst(demand_id, factor, duration_min)

    def set_multiplier(self, factor: float) -> None:
        for r in self.runners:
            r.eng.manual_multiplier = factor
