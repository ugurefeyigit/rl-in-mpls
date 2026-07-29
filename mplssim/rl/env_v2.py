"""Gymnasium environment for MPLS traffic engineering, version 2.

Governing document: docs/RL_ENVIRONMENT_V2_SPEC.md.
This module is additive: :class:`mplssim.rl.env.MplsTeEnv` and every V1
identity are untouched, and no V1 code path reaches this file.

Observation — ``obs-v2.0-notime-604``
-------------------------------------

Flat float32 vector in ``[0,1]``, **feature-major**, dimension
``2*64 + (8 + 5*4)*17 = 604``. Within a block starting at ``b``, element
``b+i`` is directed link ``i`` (topology order) or demand ``i``
(traffic_classes.yaml order)::

      0:64    sat(link input utilization)
     64:128   link up
    128:145   sat(offered / base_mbps)
    145:162   priority / 6
    162:179   protected
    179:196   sat(measured delay / delay SLA)
    196:213   sat(measured loss / loss SLA)
    213:230   min(current path age / 12, 1)
    230:247   TE dwell remaining / 3
    247:264   disconnected
    264:332   current path one-hot          (4 blocks of 17)
    332:400   previous TE path one-hot      (4 blocks of 17, all-zero = none)
    400:468   candidate live                (4 blocks of 17)
    468:536   min(candidate propagation ms / 100, 1)
    536:604   sat(candidate projected gross bottleneck)

``sat(x) = x/(1+x)`` is monotone, bounded and non-clipping, and equals exactly
0.5 at the reference boundary — so 0.5 in the delay/loss health blocks means
"exactly at SLA" and 0.5 in the utilization block means "exactly at capacity",
while 400% utilization still reads differently from 200%. V1's ``clip(u/2,0,1)``
mapped every link at or above 200% to the same 1.0.

Removed relative to V1, and why: link queue delay and link loss (128 features
that are bit-exact functions of link utilization), the global max/mean/std
utilization and prior-interval summaries (derivable duplicates mixing two time
points), and episode progress plus the wall clock (they let a policy memorize
where a fixed scenario's failures are). Time-of-day is available as the
``obs-v2.0-time-606`` ablation, never as the default.

Added relative to V1: per-demand measured delay/loss against their own SLAs,
path age, TE dwell, previous TE path, candidate liveness, candidate propagation
delay, and the *projected* gross bottleneck a move would create — V1 exposed
only each candidate's current bottleneck, which does not answer "what happens if
I move there".

Action — ``action-v2.0-discrete69``
-----------------------------------

``Discrete(69)``: 0 is "no TE change", ``1 + 4*d + p`` requests demand ``d`` onto
candidate ``p``. The numbering is unchanged from V1, but the *candidate paths
those numbers point at* are the V2 role-valid table, so V2 metadata stores the
complete ordered router sequences and a V1 checkpoint can never be loaded here.

Episode: ``terminated`` is always False — temporary full disconnection no longer
ends the episode, so recovery stays observable and every paired controller gets
an identical horizon. ``truncated`` becomes True at scenario duration.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from mplssim.rl.reward_v2 import (
    COMPONENT_ORDER,
    RewardConfigV2,
    compute_reward_v2,
    load_reward_config_v2,
    potential,
    sat,
    utility,
)
from mplssim.sim.engine_v2 import (
    ACTION_VERSION,
    CONFIG_VERSION,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    OBSERVATION_VERSION_TIME,
    REWARD_VERSION,
    SEED_VERSION,
    TRANSITION_VERSION,
    V2_CONFIG_DIR,
    EngineConfigV2,
    SimulationEngineV2,
    load_engine_config_v2,
)

OBSERVATION_CONFIG_PATH = V2_CONFIG_DIR / "rl_observation_v2.yaml"

#: Per-directed-link observation blocks.
LINK_BLOCKS = 2
#: Per-demand observation blocks: 8 scalars + 5 blocks of k candidates.
DEMAND_SCALAR_BLOCKS = 8

#: Sentinel feature values for a candidate slot that does not exist
#: (spec, "Exact default observation schema"). The shipped topology always
#: yields exactly k role-valid candidates, so these only appear in tests.
ABSENT_CANDIDATE_LIVE = 0.0
ABSENT_CANDIDATE_PROPAGATION = 1.0
ABSENT_CANDIDATE_PROJECTED = 1.0

#: Normalization constants named by the spec.
PATH_AGE_NORM_STEPS = 12.0
CANDIDATE_PROP_NORM_MS = 100.0
LOSS_SLA_FLOOR = 1e-6

MAX_WORKER_RANK = 1023
_UINT64_MAX = (1 << 64) - 1


class SeedProtocolError(ValueError):
    """The stride-1024 episode-seed protocol was violated."""


class ObservationSchemaError(ValueError):
    """The built observation layout disagrees with the declared V2 schema."""


def episode_seed_for(root_seed: int, worker_rank: int, episode_index: int,
                     stride: int = 1024) -> int:
    """``root_seed + worker_rank + stride*episode_index`` with fail-closed checks.

    V1 used ``base + 10_000*rank`` per worker and ``+1_000*episode`` per episode,
    which is not injective: with root 42, ``(rank 0, episode 10)`` and
    ``(rank 1, episode 0)`` both produce 10042. Eight workers reused 80 distinct
    seed values within their first 30 episodes. Stride-1024 with rank confined
    to ``[0,1023]`` is injective by construction and, unlike the V1 scheme, a
    worker's own sequence does not depend on how many workers are running.
    """
    if not isinstance(worker_rank, (int, np.integer)) or not (0 <= worker_rank <= MAX_WORKER_RANK):
        raise SeedProtocolError(
            f"worker_rank {worker_rank!r} outside [0,{MAX_WORKER_RANK}]")
    if episode_index < 0:
        raise SeedProtocolError(f"episode_index {episode_index!r} is negative")
    if root_seed < 0:
        raise SeedProtocolError(f"root_seed {root_seed!r} is negative")
    seed = int(root_seed) + int(worker_rank) + int(stride) * int(episode_index)
    if seed > _UINT64_MAX:
        raise SeedProtocolError(
            f"episode seed {seed} overflows uint64 "
            f"(root={root_seed}, rank={worker_rank}, episode={episode_index})")
    return seed


class MplsTeEnvV2(gym.Env):
    """MPLS-TE control environment, V2 semantics. Never a drop-in for V1."""

    metadata = {"render_modes": []}

    #: The seven version identities every V2 artifact must carry.
    VERSIONS: dict[str, str] = {
        "environment": ENVIRONMENT_VERSION,
        "observation": OBSERVATION_VERSION,
        "action": ACTION_VERSION,
        "reward": REWARD_VERSION,
        "transition": TRANSITION_VERSION,
        "config": CONFIG_VERSION,
        "seed_protocol": SEED_VERSION,
    }

    def __init__(
        self,
        scenario: str = "random_day",
        root_seed: int = 0,
        worker_rank: int = 0,
        engine_cfg: EngineConfigV2 | None = None,
        reward_cfg: RewardConfigV2 | None = None,
        include_time_of_day: bool = False,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.root_seed = int(root_seed)
        self.worker_rank = int(worker_rank)
        self.engine_cfg = engine_cfg or load_engine_config_v2()
        self.reward_cfg = reward_cfg or load_reward_config_v2()
        # P2 ablation switch. Default False; enabling it changes the observation
        # identity, which makes a checkpoint from one incompatible with the other.
        self.include_time_of_day = bool(include_time_of_day)
        self._episode_index = 0
        self.episode_seed: int | None = None

        from mplssim.factory import get_scenarios, get_topology, get_traffic_config
        self._topo = get_topology()
        self._traffic_cfg = get_traffic_config()
        self._scenarios = get_scenarios()
        if scenario not in self._scenarios:
            raise KeyError(f"unknown scenario '{scenario}' "
                           f"(have: {sorted(self._scenarios)})")

        self.eng = self._build_engine(scenario, episode_seed_for(
            self.root_seed, self.worker_rank, 0, self.engine_cfg.worker_stride))
        self.k = self.engine_cfg.k_paths
        self.n_demands = self.eng.n_demands
        self.n_dlinks = self.eng.topo.n_dlinks
        self.demand_blocks = DEMAND_SCALAR_BLOCKS + 5 * self.k

        obs_dim = LINK_BLOCKS * self.n_dlinks + self.demand_blocks * self.n_demands
        if self.include_time_of_day:
            obs_dim += 2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(1 + self.n_demands * self.k)
        self._steps_total = (self.eng.scenario.duration_min
                             // self.engine_cfg.control_interval_min)
        self._obs_priority = self.eng._priorities / 6.0
        self._obs_protected = self.eng._protected.astype(np.float32)
        self._validate_observation_schema()

    # ------------------------------------------------------------- identities
    @property
    def observation_version(self) -> str:
        return OBSERVATION_VERSION_TIME if self.include_time_of_day else OBSERVATION_VERSION

    def environment_versions(self) -> dict[str, str]:
        v = dict(self.VERSIONS)
        v["observation"] = self.observation_version
        return v

    def _validate_observation_schema(self) -> None:
        """Assert the built layout against configs/experiments/rl_observation_v2.yaml.

        A silent drift between the code layout and the published schema would
        make every downstream checkpoint claim an identity it does not have, so
        the mismatch fails closed at construction.
        """
        if self.include_time_of_day:
            return  # the ablation carries its own identity and dimension
        raw = yaml.safe_load(OBSERVATION_CONFIG_PATH.read_text(encoding="utf-8"))
        if raw["version"] != OBSERVATION_VERSION:
            raise ObservationSchemaError(
                f"{OBSERVATION_CONFIG_PATH}: version {raw['version']!r} != "
                f"{OBSERVATION_VERSION!r}")
        if int(raw["dim"]) != int(self.observation_space.shape[0]):
            raise ObservationSchemaError(
                f"schema dim {raw['dim']} != built dim {self.observation_space.shape[0]}")
        expected = self._block_offsets()
        declared = [(b["feature"], int(b["start"]), int(b["end"])) for b in raw["blocks"]]
        if declared != expected:
            for got, want in zip(declared, expected):
                if got != want:
                    raise ObservationSchemaError(
                        f"observation block mismatch: schema {got} != built {want}")
            raise ObservationSchemaError(
                f"observation schema declares {len(declared)} blocks, built {len(expected)}")

    def _block_offsets(self) -> list[tuple[str, int, int]]:
        """(feature name, start, end) for every block, in built order."""
        n_dl, n_d, k = self.n_dlinks, self.n_demands, self.k
        names = ["link_input_utilization", "link_up"]
        blocks = [(names[0], 0, n_dl), (names[1], n_dl, 2 * n_dl)]
        rows = ["offered_over_base", "priority", "protected",
                "measured_delay_over_sla", "measured_loss_over_sla",
                "current_path_age_steps", "te_dwell_remaining", "disconnected"]
        for prefix in ("current_path", "previous_te_path"):
            rows += [f"{prefix}_{p}" for p in range(k)]
        rows += [f"candidate_{p}_live" for p in range(k)]
        rows += [f"candidate_{p}_propagation_ms" for p in range(k)]
        rows += [f"candidate_{p}_projected_gross" for p in range(k)]
        base = 2 * n_dl
        for j, name in enumerate(rows):
            blocks.append((name, base + j * n_d, base + (j + 1) * n_d))
        return blocks

    # ------------------------------------------------------------------- gym
    def _build_engine(self, scenario: str, episode_seed: int) -> SimulationEngineV2:
        return SimulationEngineV2(
            topo=self._topo, traffic_cfg=self._traffic_cfg,
            scenario=self._scenarios[scenario], episode_seed=episode_seed,
            cfg=self.engine_cfg,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        ``reset(seed=x)`` restarts this environment's sequence at root ``x``
        (Gym contract). ``options['episode_seed']`` is the explicit deterministic
        override used for evaluation; it does not disturb the published root or
        the episode counter's stride.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.root_seed = int(seed)
            self._episode_index = 0
        opts = options or {}
        if "worker_rank" in opts:
            self.worker_rank = int(opts["worker_rank"])
        episode_seed = episode_seed_for(
            self.root_seed, self.worker_rank, self._episode_index,
            self.engine_cfg.worker_stride)
        if "episode_seed" in opts:
            episode_seed = int(opts["episode_seed"])
        scenario = opts.get("scenario", self.scenario)
        self._episode_index += 1
        self.episode_seed = episode_seed

        self.eng = self._build_engine(scenario, episode_seed)
        self._steps_total = (self.eng.scenario.duration_min
                             // self.engine_cfg.control_interval_min)
        info = {
            "environment_versions": self.environment_versions(),
            "episode_seed": episode_seed,
            "scenario": scenario,
            "worker_rank": self.worker_rank,
            "root_seed": self.root_seed,
            "steps_total": int(self._steps_total),
        }
        return self._obs(), info

    def step(self, action: int):
        eng = self.eng
        # Potential is evaluated on the boundary the agent actually observed,
        # before the action is applied.
        phi_current = potential(utility(eng.boundary_metrics(), self.reward_cfg),
                                self.reward_cfg)

        action = int(action)
        decoded: dict[str, Any] = {"action": action, "type": "noop", "accepted": False}
        accepted = rejected = reversal = False
        volume_share = edge_divergence = 0.0

        if action < 0 or action >= self.action_space.n:
            # Controlled rejection rather than an IndexError: an unmasked or
            # buggy policy must not be able to crash the environment.
            rejected = True
            decoded = {"action": action, "type": "out_of_range", "accepted": False,
                       "reason": f"action outside [0,{self.action_space.n - 1}]"}
        elif action > 0:
            d_idx, p_idx = divmod(action - 1, self.k)
            record = eng.apply_te_action(d_idx, p_idx)
            accepted = bool(record["accepted"])
            rejected = not accepted
            reversal = bool(record["reversal"])
            volume_share = float(record["volume_share"])
            edge_divergence = float(record["edge_divergence"])
            decoded = {
                "action": action, "type": "te_request",
                "demand": eng.demands[d_idx].id if 0 <= d_idx < self.n_demands else None,
                "demand_idx": d_idx, "path_idx": p_idx,
                "from_path": record["from_path"], "accepted": accepted,
                "reason": record["reason"], "reversal": reversal,
                "volume_share": volume_share, "edge_divergence": edge_divergence,
            }

        interval = eng.step_interval()
        phi_next = potential(utility(eng.boundary_metrics(), self.reward_cfg),
                             self.reward_cfg)
        reward, components = compute_reward_v2(
            interval, phi_current, phi_next,
            accepted=accepted, volume_share=volume_share,
            edge_divergence=edge_divergence, reversal=reversal, rejected=rejected,
            cfg=self.reward_cfg,
        )

        info = {
            "environment_versions": self.environment_versions(),
            "episode_seed": self.episode_seed,
            "metrics": interval,
            "reward_components": components,
            "reward_component_order": COMPONENT_ORDER,
            "decoded_action": decoded,
            "action_mask": self.action_masks(),
            "accepted_te_changes": interval["accepted_te_changes"],
            "rejected_te_requests": interval["rejected_te_requests"],
            "te_reversals": interval["te_reversals"],
            "frr_changes": interval["frr_changes"],
            "frr_disconnections": interval["frr_disconnections"],
            "recovery_restorations": interval["recovery_restorations"],
            "flow_solver_iterations_max": interval["flow_solver_iterations_max"],
            "episode_totals": dict(eng.episode_totals),
            "phi_current": phi_current,
            "phi_next": phi_next,
        }
        # V2 never terminates: a temporary full disconnection must stay
        # observable through recovery, and paired controllers need equal horizons.
        return self._obs(), float(reward), False, bool(eng.done), info

    # ------------------------------------------------------------------ masks
    def action_masks(self) -> np.ndarray:
        """Boolean legality mask over the 69 actions (True = allowed).

        Action 0 is always legal. The remainder is the vectorized form of
        :meth:`SimulationEngineV2.validate_te_action`, asserted equal to a
        per-action sweep of that validator in tests/test_env_v2.py.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        mask[1:1 + self.n_demands * self.k] = self.eng.te_action_matrix().ravel()
        return mask

    # -------------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        eng = self.eng
        k, n_d = self.k, self.n_demands

        link = np.empty((LINK_BLOCKS, self.n_dlinks), dtype=np.float32)
        link[0] = eng.link_util / (1.0 + eng.link_util)     # sat(util)
        link[1] = eng._dlink_up

        dm = np.zeros((self.demand_blocks, n_d), dtype=np.float32)
        offered_ratio = eng.demand_offered / eng._base_mbps
        dm[0] = offered_ratio / (1.0 + offered_ratio)
        dm[1] = self._obs_priority
        dm[2] = self._obs_protected

        # Health is measured against each demand's own SLA, so 0.5 is exactly at
        # the SLA line. A disconnected demand reports zero here; connectivity
        # owns that penalty and must not be double-counted as SLA severity.
        connected = ~eng.disconnected
        delay_ratio = np.where(connected, eng.demand_delay / eng._delay_sla, 0.0)
        loss_ratio = np.where(
            connected,
            eng.demand_loss_fraction / np.maximum(eng._loss_sla, LOSS_SLA_FLOOR), 0.0)
        dm[3] = delay_ratio / (1.0 + delay_ratio)
        dm[4] = loss_ratio / (1.0 + loss_ratio)

        dm[5] = np.minimum(eng.path_age_steps / PATH_AGE_NORM_STEPS, 1.0)
        dm[6] = eng.te_dwell_remaining / max(1, self.engine_cfg.minimum_te_dwell_steps)
        dm[7] = eng.disconnected

        rows = np.arange(n_d)
        cur = eng.current_path
        in_range = cur < k
        dm[DEMAND_SCALAR_BLOCKS + cur[in_range], rows[in_range]] = 1.0

        # All-zero previous-TE block means "no reversal target recorded".
        prev = eng.previous_te_path
        has_prev = (prev >= 0) & (prev < k)
        dm[DEMAND_SCALAR_BLOCKS + k + prev[has_prev], rows[has_prev]] = 1.0

        exists = eng._cand_exists                       # (n_demands, k)
        live = eng.candidate_available_matrix()
        base = DEMAND_SCALAR_BLOCKS + 2 * k
        dm[base:base + k] = np.where(exists, live, ABSENT_CANDIDATE_LIVE).T

        prop = np.minimum(eng._cand_prop_ms / CANDIDATE_PROP_NORM_MS, 1.0)
        base += k
        dm[base:base + k] = np.where(exists, prop, ABSENT_CANDIDATE_PROPAGATION).T

        # An absent candidate projects to +inf; sanitize before sat() so the
        # sentinel branch cannot produce inf/inf = nan.
        proj = eng.projected_gross_bottleneck_matrix()
        proj_safe = np.where(exists, proj, 0.0)
        proj_sat = np.where(exists, proj_safe / (1.0 + proj_safe),
                            ABSENT_CANDIDATE_PROJECTED)
        base += k
        dm[base:base + k] = proj_sat.T

        parts = [link.ravel(), dm.ravel()]
        if self.include_time_of_day:
            hour = (eng.scenario.start_hour + eng.t_min / 60.0) % 24.0
            theta = 2.0 * np.pi * hour / 24.0
            parts.append(np.array([0.5 + 0.5 * np.sin(theta),
                                   0.5 + 0.5 * np.cos(theta)], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32, copy=False)


__all__ = [
    "MplsTeEnvV2",
    "SeedProtocolError",
    "ObservationSchemaError",
    "episode_seed_for",
    "sat",
]
