"""V2 schema, candidate table, seed protocol, action mask and metadata.

Covers layers 2, 3, 5, 8 of docs/RL_ENVIRONMENT_V2_TEST_PLAN.md. No learning
algorithm, no smoke training, no server: engines, fixed actions and synthetic
states only.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mplssim.experiments.v2_factory import (
    CONFIG_HASH_ALGORITHM,
    FROZEN_DEFINITION_PATHS,
    IDENTITY_FIELDS,
    PINNED_ENVIRONMENT_COMMIT,
    MetadataMismatchError,
    TrainingPinError,
    assert_training_pin,
    build_environment_metadata,
    canonical_text_sha256,
    config_hashes,
    frozen_definition_drift,
    make_engine_v2,
    make_env_v2,
    sha256_file,
    validate_environment_metadata,
)
from mplssim.factory import get_topology, get_traffic_config
from mplssim.paths.candidates_v2 import (
    CandidatePathError,
    MAX_ENUMERATED_PATHS,
    build_candidate_table,
    generate_candidate_paths_v2,
    hop_cap_for,
    is_loop_free,
    is_role_valid,
    path_admin_cost,
    path_propagation_ms,
    propagation_bound,
)
from mplssim.rl.env_v2 import (
    MplsTeEnvV2,
    ObservationSchemaError,
    SeedProtocolError,
    episode_seed_for,
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
    load_engine_config_v2,
)

K = 4
N_DEMANDS = 17
N_DLINKS = 64
OBS_DIM = 604
N_ACTIONS = 69


@pytest.fixture(scope="module")
def topo():
    return get_topology()


@pytest.fixture(scope="module")
def traffic_cfg():
    return get_traffic_config()


@pytest.fixture()
def env():
    e = make_env_v2(scenario="evening_peak", root_seed=101)
    e.reset(options={"episode_seed": 101})
    return e


# ============================================================ 2. schema/versions
def test_observation_shape_dtype_and_range(env):
    obs, _ = env.reset(options={"episode_seed": 101})
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_observation_dimension_formula(env):
    assert OBS_DIM == 2 * N_DLINKS + (8 + 5 * K) * N_DEMANDS
    assert env.demand_blocks == 28


def test_action_space_is_exactly_69(env):
    assert env.action_space.n == N_ACTIONS
    assert N_ACTIONS == 1 + N_DEMANDS * K


def test_every_declared_block_offset_matches_the_built_layout(env):
    """The published schema and the code layout must not drift apart."""
    import yaml

    from mplssim.rl.env_v2 import OBSERVATION_CONFIG_PATH
    raw = yaml.safe_load(OBSERVATION_CONFIG_PATH.read_text(encoding="utf-8"))
    declared = [(b["feature"], b["start"], b["end"]) for b in raw["blocks"]]
    assert declared == env._block_offsets()
    # spot-check the exact offsets named in the implementation prompt
    named = dict((f, (s, e)) for f, s, e in declared)
    assert named["link_input_utilization"] == (0, 64)
    assert named["link_up"] == (64, 128)
    assert named["offered_over_base"] == (128, 145)
    assert named["measured_delay_over_sla"] == (179, 196)
    assert named["measured_loss_over_sla"] == (196, 213)
    assert named["current_path_age_steps"] == (213, 230)
    assert named["te_dwell_remaining"] == (230, 247)
    assert named["disconnected"] == (247, 264)
    assert named["current_path_0"] == (264, 281)
    assert named["previous_te_path_0"] == (332, 349)
    assert named["candidate_0_live"] == (400, 417)
    assert named["candidate_0_propagation_ms"] == (468, 485)
    assert named["candidate_0_projected_gross"] == (536, 553)
    assert named["candidate_3_projected_gross"] == (587, 604)


def test_schema_drift_fails_closed(env, monkeypatch):
    monkeypatch.setattr(env, "_block_offsets", lambda: [("bogus", 0, 1)])
    with pytest.raises(ObservationSchemaError):
        env._validate_observation_schema()


def test_utilization_transform_is_half_at_capacity_and_monotone_beyond_200pct(env):
    """sat(u) = u/(1+u): 0.5 at capacity, and 400% still differs from 200%."""
    eng = env.eng
    eng.link_util = np.zeros(N_DLINKS)
    eng.link_util[0] = 1.0
    eng.link_util[1] = 2.0
    eng.link_util[2] = 4.0
    obs = env._obs()
    assert obs[0] == pytest.approx(0.5, abs=1e-7)
    assert obs[1] == pytest.approx(2.0 / 3.0, abs=1e-7)
    assert obs[2] == pytest.approx(0.8, abs=1e-7)
    assert obs[1] < obs[2]          # V1's clip(u/2,0,1) made these identical


def test_delay_and_loss_health_transform_is_half_exactly_at_sla(env):
    eng = env.eng
    eng.disconnected[:] = False
    eng.demand_delay = eng._delay_sla.copy()             # exactly at SLA
    eng.demand_loss_fraction = eng._loss_sla.copy()
    obs = env._obs()
    assert np.allclose(obs[179:196], 0.5, atol=1e-6)
    assert np.allclose(obs[196:213], 0.5, atol=1e-6)


def test_disconnected_demand_reports_zero_health_and_flag_one(env):
    eng = env.eng
    eng.disconnected[:] = False
    eng.disconnected[3] = True
    eng.demand_delay[3] = 9999.0
    eng.demand_loss_fraction[3] = 1.0
    obs = env._obs()
    assert obs[179 + 3] == 0.0
    assert obs[196 + 3] == 0.0
    assert obs[247 + 3] == 1.0


def test_all_zero_previous_te_block_means_no_reversal_target(env):
    eng = env.eng
    eng.previous_te_path[:] = -1
    obs = env._obs()
    assert np.all(obs[332:400] == 0.0)
    eng.previous_te_path[5] = 2
    obs = env._obs()
    assert obs[332 + 2 * N_DEMANDS + 5] == 1.0
    assert obs[332:400].sum() == 1.0


def test_absent_candidate_sentinels_are_exact(env):
    """live=0, propagation=1, projected=1 for a candidate slot that does not exist."""
    eng = env.eng
    eng._cand_exists = eng._cand_exists.copy()
    eng._cand_exists[7, 3] = False
    obs = env._obs()
    assert obs[400 + 3 * N_DEMANDS + 7] == 0.0     # candidate_3_live
    assert obs[468 + 3 * N_DEMANDS + 7] == 1.0     # candidate_3_propagation
    assert obs[536 + 3 * N_DEMANDS + 7] == 1.0     # candidate_3_projected


def test_path_age_and_dwell_normalizations(env):
    eng = env.eng
    eng.path_age_steps[:] = 0
    eng.path_age_steps[0] = 6
    eng.path_age_steps[1] = 12
    eng.path_age_steps[2] = 400          # clamped, not wrapped
    eng.te_dwell_remaining[:] = 0
    eng.te_dwell_remaining[0] = 3
    eng.te_dwell_remaining[1] = 1
    obs = env._obs()
    assert obs[213 + 0] == pytest.approx(0.5)
    assert obs[213 + 1] == pytest.approx(1.0)
    assert obs[213 + 2] == pytest.approx(1.0)
    assert obs[230 + 0] == pytest.approx(1.0)
    assert obs[230 + 1] == pytest.approx(1.0 / 3.0)


def test_time_ablation_has_606_features_and_a_distinct_identity():
    e = make_env_v2(scenario="evening_peak", root_seed=1, include_time_of_day=True)
    obs, info = e.reset(options={"episode_seed": 1})
    assert obs.shape == (606,)
    assert e.observation_version == OBSERVATION_VERSION_TIME
    assert info["environment_versions"]["observation"] == OBSERVATION_VERSION_TIME
    assert OBSERVATION_VERSION_TIME != OBSERVATION_VERSION
    default = make_env_v2(scenario="evening_peak", root_seed=1)
    assert default.observation_version == OBSERVATION_VERSION
    assert default.observation_space.shape == (OBS_DIM,)


def test_default_observation_carries_no_clock_or_progress():
    """Two engines at different wall-clock times but identical state agree."""
    a = make_env_v2(scenario="full_day", root_seed=1)
    a.reset(options={"episode_seed": 1})
    b = make_env_v2(scenario="evening_peak", root_seed=1)
    b.reset(options={"episode_seed": 1})
    for eng in (a.eng, b.eng):
        eng.link_util = np.full(N_DLINKS, 0.25)
        eng._dlink_up[:] = True
        eng.demand_offered = eng._base_mbps.copy()
        eng.gross_link_load = np.zeros(N_DLINKS)
        eng.demand_delay = np.zeros(N_DEMANDS)
        eng.demand_loss_fraction = np.zeros(N_DEMANDS)
        eng.disconnected[:] = False
        eng.current_path[:] = 0
        eng.previous_te_path[:] = -1
        eng.path_age_steps[:] = 4
        eng.te_dwell_remaining[:] = 0
    assert a.eng.t_min == b.eng.t_min == 0.0
    assert a.eng.scenario.start_hour != b.eng.scenario.start_hour
    np.testing.assert_array_equal(a._obs(), b._obs())


def test_version_identities_are_the_required_strings(env):
    v = env.environment_versions()
    assert v == {
        "environment": "mpls-te-v2.0.0",
        "observation": "obs-v2.0-notime-604",
        "action": "action-v2.0-discrete69",
        "reward": "reward-v2.0-operational",
        "transition": "transition-v2.0-boundary-right-closed",
        "config": "config-v2.0",
        "seed_protocol": "seed-v2.0-stride1024",
    }
    assert (ENVIRONMENT_VERSION, OBSERVATION_VERSION, ACTION_VERSION, REWARD_VERSION,
            TRANSITION_VERSION, CONFIG_VERSION, SEED_VERSION) == (
        "mpls-te-v2.0.0", "obs-v2.0-notime-604", "action-v2.0-discrete69",
        "reward-v2.0-operational", "transition-v2.0-boundary-right-closed",
        "config-v2.0", "seed-v2.0-stride1024")


# ================================================== 2b. metadata mismatch matrix
def test_metadata_contains_every_required_field(env):
    meta = build_environment_metadata(env)
    for field_name in IDENTITY_FIELDS:
        assert field_name in meta, field_name
    assert meta["observation_dim"] == OBS_DIM
    assert meta["action_dim"] == N_ACTIONS
    assert meta["demand_ids"] == [f"D{i}" for i in range(1, 18)]
    assert set(meta["candidate_paths"]) == set(meta["demand_ids"])
    assert all(len(v) == K for v in meta["candidate_paths"].values())
    assert set(meta["config_hashes"]) == {
        "configs/topology.yaml", "configs/traffic_classes.yaml",
        "configs/scenarios.yaml", "configs/experiments/rl_env_v2.yaml",
        "configs/experiments/rl_observation_v2.yaml",
        "configs/experiments/rl_reward_v2.yaml"}
    assert "git_commit" in meta and "git_dirty" in meta


def test_matching_metadata_validates(env):
    meta = build_environment_metadata(env)
    validate_environment_metadata(meta)


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_each_single_field_mismatch_fails_closed(env, field_name):
    """One corrupted field at a time; every one must be fatal and named."""
    meta = build_environment_metadata(env)
    original = meta[field_name]
    if isinstance(original, str):
        meta[field_name] = original + "-tampered"
    elif isinstance(original, int):
        meta[field_name] = original + 1
    elif isinstance(original, list):
        meta[field_name] = original[:-1]
    elif isinstance(original, dict):
        meta[field_name] = dict(original)
        key = sorted(meta[field_name])[0]
        meta[field_name][key] = "tampered"
    else:
        pytest.skip(f"unhandled type {type(original)}")
    with pytest.raises(MetadataMismatchError) as exc:
        validate_environment_metadata(meta)
    assert field_name in str(exc.value)


def test_missing_metadata_field_fails_closed(env):
    meta = build_environment_metadata(env)
    del meta["candidate_paths"]
    with pytest.raises(MetadataMismatchError, match="candidate_paths"):
        validate_environment_metadata(meta)


def test_v1_style_metadata_never_loads_into_v2():
    """A 586/69 V1 record must be rejected, not padded."""
    v1_like = {
        "environment": "mpls-te-v1", "observation": "obs-v1-586",
        "action": "action-v2.0-discrete69", "observation_dim": 586,
        "action_dim": 69,
    }
    with pytest.raises(MetadataMismatchError):
        validate_environment_metadata(v1_like)


def test_candidate_path_mismatch_message_names_the_demand_and_index(env):
    meta = build_environment_metadata(env)
    meta["candidate_paths"] = copy.deepcopy(meta["candidate_paths"])
    meta["candidate_paths"]["D10"][3] = ["PE4", "P3", "P6", "A1", "PE7", "A2", "PE8"]
    with pytest.raises(MetadataMismatchError) as exc:
        validate_environment_metadata(meta)
    assert "D10 candidate 3" in str(exc.value)


def test_config_hashes_are_stable_and_content_addressed():
    assert config_hashes() == config_hashes()
    assert all(len(h) == 64 for h in config_hashes().values())


# ============================== 2c. configuration identity portability (LF/CRLF)
CONFIG_RELATIVE_PATHS = [
    "topology.yaml", "traffic_classes.yaml", "scenarios.yaml",
    "experiments/rl_env_v2.yaml", "experiments/rl_observation_v2.yaml",
    "experiments/rl_reward_v2.yaml",
]


def _render(raw: bytes, ending: bytes) -> bytes:
    """The same content re-rendered with a chosen line ending."""
    lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return lf if ending == b"\n" else lf.replace(b"\n", ending)


@pytest.mark.parametrize("rel", CONFIG_RELATIVE_PATHS)
def test_lf_and_crlf_renderings_of_a_config_hash_identically(rel, tmp_path):
    """The core portability guarantee, on the six real configuration files."""
    from mplssim.core.topology import CONFIG_DIR
    raw = (CONFIG_DIR / rel).read_bytes()

    lf_file = tmp_path / "lf.yaml"
    crlf_file = tmp_path / "crlf.yaml"
    cr_file = tmp_path / "cr.yaml"
    lf_file.write_bytes(_render(raw, b"\n"))
    crlf_file.write_bytes(_render(raw, b"\r\n"))
    cr_file.write_bytes(_render(raw, b"\r"))

    canonical = canonical_text_sha256(lf_file)
    assert canonical_text_sha256(crlf_file) == canonical
    assert canonical_text_sha256(cr_file) == canonical

    # The test is only meaningful if the raw bytes genuinely differ, i.e. the
    # file actually contains line breaks.
    assert lf_file.read_bytes() != crlf_file.read_bytes()
    assert sha256_file(lf_file) != sha256_file(crlf_file)


def test_the_whole_config_identity_is_line_ending_independent(tmp_path):
    """All six hashes together, as a single identity, survive re-rendering."""
    from mplssim.core.topology import CONFIG_DIR
    identities = {}
    for ending, name in ((b"\n", "lf"), (b"\r\n", "crlf"), (b"\r", "cr")):
        digest = {}
        for rel in CONFIG_RELATIVE_PATHS:
            target = tmp_path / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_render((CONFIG_DIR / rel).read_bytes(), ending))
            digest[f"configs/{rel}"] = canonical_text_sha256(target)
        identities[name] = digest
    assert identities["lf"] == identities["crlf"] == identities["cr"]
    # and that identity is exactly what the live environment publishes
    assert identities["lf"] == config_hashes()


@pytest.mark.parametrize("mutate,label", [
    (lambda s: s.replace("micro_ticks_per_interval: 5",
                         "micro_ticks_per_interval: 6"), "changed value"),
    (lambda s: s.replace("k_paths: 4\n", ""), "dropped line"),
    (lambda s: s + "\nextra_key: 1\n", "added line"),
    (lambda s: s.replace("k_paths: 4", "k_paths:  4"), "added whitespace"),
    (lambda s: s.replace("control_interval_min: 5",
                         "control_interval_min: 5 "), "trailing space"),
    (lambda s: "\n" + s, "leading blank line"),
])
def test_any_real_content_change_still_changes_the_canonical_hash(
        mutate, label, tmp_path):
    """Canonicalization must not mask an actual edit, in any newline form."""
    from mplssim.core.topology import CONFIG_DIR
    original = (CONFIG_DIR / "experiments/rl_env_v2.yaml").read_text(encoding="utf-8")
    baseline = tmp_path / "base.yaml"
    baseline.write_text(original, encoding="utf-8", newline="\n")
    expected = canonical_text_sha256(baseline)

    changed = mutate(original)
    assert changed != original, f"mutation {label!r} did not alter the file"
    for ending, name in (("\n", "lf"), ("\r\n", "crlf")):
        target = tmp_path / f"{name}.yaml"
        target.write_bytes(_render(changed.encode("utf-8"), ending.encode()))
        assert canonical_text_sha256(target) != expected, f"{label} / {name}"


def test_metadata_uses_the_canonical_hash_and_records_the_algorithm(env):
    from mplssim.core.topology import CONFIG_DIR
    meta = build_environment_metadata(env)
    assert meta["config_hash_algorithm"] == CONFIG_HASH_ALGORITHM == "sha256/lf-canonical"
    for rel in CONFIG_RELATIVE_PATHS:
        assert meta["config_hashes"][f"configs/{rel}"] == \
            canonical_text_sha256(CONFIG_DIR / rel)


def test_validation_compares_exactly_one_stored_hash_per_config(env):
    """Fail-closed strictness: a single canonical hash, never a set of accepted ones."""
    meta = build_environment_metadata(env)
    assert "config_hash_algorithm" in IDENTITY_FIELDS
    assert "config_hashes" in IDENTITY_FIELDS
    for value in meta["config_hashes"].values():
        assert isinstance(value, str) and len(value) == 64
    tampered = copy.deepcopy(meta)
    tampered["config_hashes"]["configs/topology.yaml"] = "0" * 64
    with pytest.raises(MetadataMismatchError, match="config_hashes"):
        validate_environment_metadata(tampered)


def test_a_record_written_under_a_different_hash_scheme_fails_closed(env):
    """Raw-byte hashes from the previous scheme must not silently validate."""
    from mplssim.core.topology import CONFIG_DIR
    meta = build_environment_metadata(env)
    legacy = copy.deepcopy(meta)
    legacy["config_hash_algorithm"] = "sha256/raw"
    legacy["config_hashes"] = {f"configs/{rel}": sha256_file(CONFIG_DIR / rel)
                               for rel in CONFIG_RELATIVE_PATHS}
    with pytest.raises(MetadataMismatchError, match="config_hash_algorithm"):
        validate_environment_metadata(legacy)


# ================================ 2d. training pin / frozen definitions
def test_the_pinned_commit_is_the_signed_off_sha():
    assert PINNED_ENVIRONMENT_COMMIT == "dca533b5c6fa9953307d01470c23cac512eb2961"
    assert len(PINNED_ENVIRONMENT_COMMIT) == 40


def test_frozen_definition_set_covers_everything_that_changes_v2_behaviour():
    """Guards against a definition file quietly escaping the freeze."""
    frozen = set(FROZEN_DEFINITION_PATHS)
    # the V2 implementation and its configs
    assert {"mplssim/sim/engine_v2.py", "mplssim/rl/env_v2.py",
            "mplssim/rl/reward_v2.py", "mplssim/paths/candidates_v2.py",
            "configs/experiments/rl_env_v2.yaml",
            "configs/experiments/rl_observation_v2.yaml",
            "configs/experiments/rl_reward_v2.yaml"} <= frozen
    # the shared problem definition V2 reads
    assert {"configs/topology.yaml", "configs/traffic_classes.yaml",
            "configs/scenarios.yaml"} <= frozen
    # shared code V2 behaviour depends on: the analytic curves, offered
    # traffic, admin cost and the topology/demand model
    assert {"mplssim/sim/models.py", "mplssim/traffic/model.py",
            "mplssim/paths/candidates.py", "mplssim/core/model.py",
            "mplssim/core/topology.py", "mplssim/factory.py"} <= frozen
    # every hashed config is frozen
    for rel in CONFIG_RELATIVE_PATHS:
        assert f"configs/{rel}" in frozen
    # the plumbing layer is knowingly excluded, and every listed path is checked
    from mplssim.experiments.v2_factory import UNFROZEN_PLUMBING
    assert UNFROZEN_PLUMBING not in frozen
    assert len(frozen) == len(FROZEN_DEFINITION_PATHS)


def test_definitions_are_currently_frozen_at_the_pinned_commit():
    assert frozen_definition_drift() == {}


def test_assert_training_pin_passes_and_returns_a_usable_record():
    pin = assert_training_pin()
    assert pin["pinned_environment_commit"] == PINNED_ENVIRONMENT_COMMIT
    assert pin["frozen_definitions_verified"] is True
    assert pin["environment_metadata_validated"] is True
    assert set(pin["frozen_definition_paths"]) == set(FROZEN_DEFINITION_PATHS)
    assert "git_commit" in pin


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_the_pin_is_line_ending_independent(ending, monkeypatch):
    """Neither an LF, a CRLF nor a lone-CR checkout may be reported as drift.

    Both renderings are constructed explicitly rather than read from disk, so
    the test means the same thing whichever way this checkout materialized the
    file (a fresh clone here is CRLF, a Linux clone is LF).
    """
    import mplssim.experiments.v2_factory as v2f
    rel = "configs/experiments/rl_reward_v2.yaml"
    target = v2f.REPO_ROOT / rel
    real_read_bytes = Path.read_bytes

    lf = real_read_bytes(target).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert b"\n" in lf, "file must contain line breaks for this test to mean anything"
    rendered = lf if ending == b"\n" else lf.replace(b"\n", ending)
    if ending != b"\n":
        assert rendered != lf, "renderings must genuinely differ in raw bytes"

    def fake_read_bytes(self):
        return rendered if self == target else real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    assert frozen_definition_drift() == {}


@pytest.mark.parametrize("rel", [
    "mplssim/rl/reward_v2.py",
    "configs/experiments/rl_reward_v2.yaml",
    "configs/topology.yaml",
    "mplssim/sim/models.py",
])
def test_the_pin_detects_a_drifted_definition(rel, monkeypatch):
    """Negative control: any real edit to a frozen file must fail closed."""
    import mplssim.experiments.v2_factory as v2f
    target = v2f.REPO_ROOT / rel
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self):
        if self == target:
            return real_read_bytes(self) + b"\n# drifted\n"
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    drift = frozen_definition_drift()
    assert rel in drift, drift
    with pytest.raises(TrainingPinError, match="drifted from the signed-off"):
        assert_training_pin()


def test_the_pin_detects_a_missing_definition(monkeypatch):
    import mplssim.experiments.v2_factory as v2f
    rel = "configs/experiments/rl_env_v2.yaml"
    real_exists = Path.exists

    def fake_exists(self):
        return False if self == v2f.REPO_ROOT / rel else real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert frozen_definition_drift()[rel] == "missing from the working tree"
    with pytest.raises(TrainingPinError):
        assert_training_pin()


def test_an_unknown_pinned_commit_fails_closed_rather_than_passing():
    with pytest.raises(TrainingPinError, match="cannot read"):
        frozen_definition_drift("0" * 40)


def test_v2_training_entry_point_checks_the_pin_before_any_training_work():
    """The guard must run before envs or the model are constructed."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "train.py"
              ).read_text(encoding="utf-8")
    assert "assert_training_pin" in source
    pin_at = source.index("pin = assert_training_pin()")
    assert pin_at < source.index("DummyVecEnv([make_env_v2_worker")
    assert pin_at < source.index("MaskablePPO(")
    assert source.index("model.learn(") < source.rindex("assert_training_pin()")
    assert '"training_pin"' in source


def test_v2_reward_and_engine_configs_are_immutable_dataclasses():
    """Frozen definitions are also frozen objects: no in-run mutation."""
    import dataclasses
    from mplssim.rl.reward_v2 import load_reward_config_v2
    from mplssim.sim.engine_v2 import load_engine_config_v2
    reward_cfg = load_reward_config_v2()
    engine_cfg = load_engine_config_v2()
    assert dataclasses.fields(reward_cfg) and dataclasses.fields(engine_cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reward_cfg.move_fixed = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        engine_cfg.minimum_te_dwell_steps = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        engine_cfg.flow_solver.max_iterations = 1


def test_v2_training_refuses_v1_style_reward_overrides():
    """--zero-weight is a V1 reward ablation and must not touch the V2 reward."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "train.py"
              ).read_text(encoding="utf-8")
    assert 'args.env_version == "v2" and args.zero_weight' in source
    assert "ap.error(" in source


# ============================================================== 3. candidates
def test_exactly_four_candidates_for_every_demand(env):
    for d in env.eng.demands:
        assert len(d.candidate_paths) == K


def test_every_candidate_is_structurally_valid(topo, env):
    for d in env.eng.demands:
        hop_cap = hop_cap_for(topo, d.src, d.dst, env.engine_cfg.max_hop_factor)
        bound = propagation_bound(
            path_propagation_ms(topo, d.candidate_paths[0]),
            env.engine_cfg.candidate_delay_factor,
            env.engine_cfg.candidate_delay_additive_ms)
        for p_idx, routers in enumerate(d.candidate_paths):
            where = f"{d.id}/p{p_idx}"
            assert routers[0] == d.src and routers[-1] == d.dst, where
            assert is_loop_free(routers), where
            assert is_role_valid(topo, routers), where
            for i in range(len(routers) - 1):
                assert (routers[i], routers[i + 1]) in topo.dlink_by_pair, where
            assert len(routers) - 1 <= hop_cap, where
            assert path_propagation_ms(topo, routers) <= bound, where
        assert len(set(d.candidate_paths)) == K, d.id


def test_candidates_are_ascending_in_the_required_key(topo, env):
    for d in env.eng.demands:
        keys = [(path_admin_cost(topo, r), path_propagation_ms(topo, r), r)
                for r in d.candidate_paths]
        assert keys == sorted(keys), d.id


def test_candidate_zero_is_the_role_valid_administrative_shortest(topo, env):
    for d in env.eng.demands:
        costs = [path_admin_cost(topo, r) for r in d.candidate_paths]
        assert costs[0] == min(costs), d.id


def test_no_candidate_transits_a_pe(topo, env):
    """The P0-3 fix: D10/p3 no longer uses PE7 and D16/p3 no longer uses PE3."""
    for d in env.eng.demands:
        for routers in d.candidate_paths:
            transit_roles = {topo.routers[r].role for r in routers[1:-1]}
            assert transit_roles <= {"P", "AGG"}, (d.id, routers)
    d10 = env.eng.demand_by_id["D10"].candidate_paths
    d16 = env.eng.demand_by_id["D16"].candidate_paths
    assert ("PE4", "P3", "P6", "A1", "PE7", "A2", "PE8") not in d10
    assert ("PE4", "P3", "PE3", "P2", "P5", "P8", "PE5") not in d16
    assert all("PE7" not in r[1:-1] for r in d10)
    assert all("PE3" not in r[1:-1] for r in d16)


def test_generation_is_deterministic_and_byte_identical(topo, traffic_cfg):
    a = build_candidate_table(topo, traffic_cfg.demands)
    b = build_candidate_table(topo, traffic_cfg.demands)
    assert a == b
    fresh = {d.id: generate_candidate_paths_v2(topo, d.src, d.dst)
             for d in traffic_cfg.demands}
    assert fresh == a


def test_action_metadata_maps_each_number_to_its_router_sequence(env):
    meta = build_environment_metadata(env)
    for action in range(1, N_ACTIONS):
        d_idx, p_idx = divmod(action - 1, K)
        demand_id = meta["demand_ids"][d_idx]
        assert meta["candidate_paths"][demand_id][p_idx] == \
            list(env.eng.demands[d_idx].candidate_paths[p_idx])


def test_generation_fails_closed_when_four_candidates_are_impossible(topo):
    """A demand whose delay bound admits fewer than k paths must raise."""
    with pytest.raises(CandidatePathError):
        generate_candidate_paths_v2(topo, "PE1", "PE5", k=4,
                                    delay_factor=1.0, delay_additive_ms=0.0)


def test_generation_fails_closed_on_degenerate_endpoints(topo):
    with pytest.raises(CandidatePathError):
        generate_candidate_paths_v2(topo, "PE1", "PE1")


def test_enumeration_cap_is_ample_for_the_shipped_topology(topo, traffic_cfg):
    """Guards the 'tie group exhausted' guarantee, not just 'k found'."""
    assert MAX_ENUMERATED_PATHS >= 200
    for d in traffic_cfg.demands:
        generate_candidate_paths_v2(topo, d.src, d.dst)  # must not raise


# ========================================================== 5. seed protocol
@pytest.mark.parametrize("n_workers", [1, 2, 8, 32])
def test_no_episode_seed_collisions_over_1000_episodes(n_workers):
    n_episodes = math.ceil(1000 / n_workers) + 1
    seeds = [episode_seed_for(42, rank, ep)
             for rank in range(n_workers) for ep in range(n_episodes)]
    assert len(seeds) >= 1000
    assert len(set(seeds)) == len(seeds)


def test_a_workers_sequence_does_not_depend_on_the_worker_count():
    """V1's `base + 10_000*rank` changed a worker's stream with fleet size."""
    for rank in (0, 1, 7, 31):
        seq = [episode_seed_for(42, rank, ep) for ep in range(50)]
        assert seq == [42 + rank + 1024 * ep for ep in range(50)]


def test_the_documented_v1_collision_does_not_recur():
    """(rank 0, ep 10) and (rank 1, ep 0) both mapped to 10042 under V1."""
    assert episode_seed_for(42, 0, 10) != episode_seed_for(42, 1, 0)


@pytest.mark.parametrize("rank", [-1, 1024, 5000])
def test_worker_rank_outside_range_fails(rank):
    with pytest.raises(SeedProtocolError):
        episode_seed_for(42, rank, 0)


def test_uint64_overflow_fails_before_rng_construction():
    with pytest.raises(SeedProtocolError, match="overflow"):
        episode_seed_for((1 << 64) - 1, 1, 1)


def test_negative_root_or_episode_fails():
    with pytest.raises(SeedProtocolError):
        episode_seed_for(-1, 0, 0)
    with pytest.raises(SeedProtocolError):
        episode_seed_for(42, 0, -1)


def test_explicit_evaluation_seed_is_reproduced_exactly():
    e = make_env_v2(scenario="full_day", root_seed=7)
    _, info = e.reset(options={"episode_seed": 12345})
    assert info["episode_seed"] == 12345
    assert e.eng.episode_seed == 12345


def test_gym_seed_restarts_the_documented_sequence():
    e = make_env_v2(scenario="full_day", root_seed=0)
    _, i0 = e.reset(seed=500)
    _, i1 = e.reset()
    _, i2 = e.reset()
    assert [i0["episode_seed"], i1["episode_seed"], i2["episode_seed"]] == \
        [500, 500 + 1024, 500 + 2048]
    _, i3 = e.reset(seed=500)
    assert i3["episode_seed"] == 500          # restart, not continue


def test_scenario_and_ar_streams_are_distinct_children():
    eng = make_engine_v2("random_day", episode_seed=999)
    a = eng.traffic.scenario_rng.bit_generator.state
    b = eng.traffic.ar_rng.bit_generator.state
    assert a["state"] != b["state"]
    expected_scn = np.random.default_rng(np.random.SeedSequence([999, 1]))
    expected_ar = np.random.default_rng(np.random.SeedSequence([999, 2]))
    assert expected_scn.bit_generator.state["state"] != \
        expected_ar.bit_generator.state["state"]


def test_same_seed_and_action_trace_is_bit_reproducible():
    def trace(script):
        e = make_env_v2(scenario="link_failure", root_seed=101)
        e.reset(options={"episode_seed": 101})
        out = []
        for a in script:
            obs, r, _, tr, _ = e.step(a)
            out.append((obs.copy(), r))
            if tr:
                break
        return out

    script = [0, 1, 0, 0, 21, 0, 0, 0, 45, 0] * 3
    for (oa, ra), (ob, rb) in zip(trace(script), trace(script)):
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb


def test_different_seeds_produce_different_traffic():
    a = make_engine_v2("full_day", episode_seed=101)
    b = make_engine_v2("full_day", episode_seed=102)
    a.step_interval()
    b.step_interval()
    assert not np.allclose(a.demand_offered, b.demand_offered)


def test_actions_do_not_perturb_offered_traffic_or_the_ar_stream():
    """The paired-comparison property: routing never consumes traffic randomness."""
    passive = make_env_v2(scenario="evening_peak", root_seed=101)
    active = make_env_v2(scenario="evening_peak", root_seed=101)
    passive.reset(options={"episode_seed": 101})
    active.reset(options={"episode_seed": 101})
    rng = np.random.default_rng(0)
    for _ in range(20):
        legal = np.flatnonzero(active.action_masks())
        active.step(int(rng.choice(legal)))
        passive.step(0)
        np.testing.assert_array_equal(passive.eng.demand_offered,
                                      active.eng.demand_offered)
        assert passive.eng.traffic._rng.bit_generator.state == \
            active.eng.traffic._rng.bit_generator.state


def test_interleaved_environments_equal_isolated_environments():
    def isolated(scenario, seed, n):
        e = make_env_v2(scenario=scenario, root_seed=seed)
        e.reset(options={"episode_seed": seed})
        return [e.step(0)[1] for _ in range(n)]

    a_iso = isolated("full_day", 101, 12)
    b_iso = isolated("evening_peak", 202, 12)
    ea = make_env_v2(scenario="full_day", root_seed=101)
    eb = make_env_v2(scenario="evening_peak", root_seed=202)
    ea.reset(options={"episode_seed": 101})
    eb.reset(options={"episode_seed": 202})
    a_int, b_int = [], []
    for _ in range(12):
        a_int.append(ea.step(0)[1])
        b_int.append(eb.step(0)[1])
    assert a_int == a_iso and b_int == b_iso


def test_fast_clone_shares_no_mutable_state():
    eng = make_engine_v2("link_failure", episode_seed=101)
    for _ in range(4):
        eng.step_interval()
    cl = eng.fast_clone()
    for attr in ("current_path", "disconnected", "_dlink_up", "path_age_steps",
                 "te_dwell_remaining", "previous_te_path", "last_te_step",
                 "gross_link_load", "link_input_load", "link_util", "link_loss",
                 "demand_offered", "demand_delivered", "demand_delay"):
        assert getattr(cl, attr) is not getattr(eng, attr), attr
    assert cl.link_up is not eng.link_up
    assert cl.te_history is not eng.te_history
    assert cl.frr_history is not eng.frr_history
    assert cl.restoration_history is not eng.restoration_history
    assert cl.episode_totals is not eng.episode_totals
    assert cl.traffic is not eng.traffic
    assert cl.traffic._noise is not eng.traffic._noise
    # diverging the clone leaves the original untouched
    for _ in range(3):
        cl.step_interval()
    assert cl.t_min != eng.t_min
    assert not np.allclose(cl.demand_offered, eng.demand_offered)


def test_deep_clone_and_fast_clone_agree_observationally():
    eng = make_engine_v2("flash_crowd", episode_seed=101)
    for _ in range(5):
        eng.step_interval()
    deep, fast = eng.clone(), eng.fast_clone()
    for _ in range(6):
        d, f = deep.step_interval(), fast.step_interval()
        assert d["max_util"] == f["max_util"]
        assert d["delivered_ratio"] == f["delivered_ratio"]
    np.testing.assert_array_equal(deep.link_input_load, fast.link_input_load)


def test_every_shareable_attribute_is_actually_immutable_in_practice():
    """Nothing mutated during a step may appear in the shared-attribute list."""
    def freeze(value):
        """Structural snapshot that compares nested arrays by value."""
        if isinstance(value, np.ndarray):
            return ("array", value.tolist())
        if isinstance(value, (list, tuple)):
            return ("seq", [freeze(v) for v in value])
        if isinstance(value, (dict,)):
            return ("map", {k: freeze(v) for k, v in value.items()})
        if isinstance(value, (str, int, float, bool, type(None), frozenset)):
            return ("scalar", value)
        return ("repr", repr(value))

    eng = make_engine_v2("full_day", episode_seed=101)
    before = {name: freeze(getattr(eng, name))
              for name in eng._SHAREABLE_ATTRS if hasattr(eng, name)}
    for _ in range(3):
        eng.step_interval()
    for name, old in before.items():
        assert old == freeze(getattr(eng, name)), name


# ======================================================= 8. mask and validator
def _mask_from_validator(eng) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True
    for action in range(1, N_ACTIONS):
        d_idx, p_idx = divmod(action - 1, eng.k)
        mask[action] = eng.validate_te_action(d_idx, p_idx)[0]
    return mask


def test_action_zero_is_always_valid(env):
    for _ in range(8):
        assert env.action_masks()[0]
        env.step(0)


def test_mask_equals_a_per_action_validator_sweep_in_many_states():
    """Reset, dwell, single failure, double failure, disconnected and recovered."""
    e = make_env_v2(scenario="ood_double_failure", root_seed=101)
    e.reset(options={"episode_seed": 101})
    np.testing.assert_array_equal(e.action_masks(), _mask_from_validator(e.eng))
    rng = np.random.default_rng(3)
    for _ in range(60):
        legal = np.flatnonzero(e.action_masks())
        _, _, _, trunc, _ = e.step(int(rng.choice(legal)))
        np.testing.assert_array_equal(e.action_masks(), _mask_from_validator(e.eng))
        if trunc:
            break
    # forced synthetic states the trace may not reach on its own
    eng = e.eng
    eng.disconnected[:] = True
    np.testing.assert_array_equal(e.action_masks(), _mask_from_validator(eng))
    eng.disconnected[:] = False
    eng.te_dwell_remaining[:] = 3
    np.testing.assert_array_equal(e.action_masks(), _mask_from_validator(eng))


def test_the_live_current_path_is_invalid_but_a_disconnected_demand_may_reselect(env):
    eng = env.eng
    eng.disconnected[:] = False
    cur = int(eng.current_path[0])
    assert not eng.validate_te_action(0, cur)[0]
    assert "already on this path" in eng.validate_te_action(0, cur)[1]
    eng.disconnected[0] = True
    assert eng.validate_te_action(0, cur)[0]


def test_candidate_on_a_failed_link_is_invalid(env):
    eng = env.eng
    links = eng._cand_links[0][1]
    victim = eng.topo.dlinks[int(links[0])].undirected_id
    eng.set_link_state(victim, False)
    ok, reason = eng.validate_te_action(0, 1)
    assert not ok and "failed link" in reason


@pytest.mark.parametrize("bad", [(-1, 0), (17, 0), (0, -1), (0, 4), (0, 99)])
def test_out_of_range_indices_are_rejected_not_raised(env, bad):
    ok, reason = env.eng.validate_te_action(*bad)
    assert not ok and reason in ("unknown demand", "unknown candidate path")


@pytest.mark.parametrize("action", [-5, 69, 1000])
def test_out_of_range_action_is_a_controlled_rejection(env, action):
    before = env.eng.current_path.copy()
    obs, reward, term, trunc, info = env.step(action)
    assert info["decoded_action"]["type"] == "out_of_range"
    assert info["reward_components"]["invalid"] == pytest.approx(-0.05)
    np.testing.assert_array_equal(env.eng.current_path, before)
    assert obs.shape == (OBS_DIM,)


def test_hard_dwell_blocks_exactly_three_steps_with_boundary_checked():
    """Accept at step s -> blocked at s+1 and s+2 -> legal again at s+3."""
    e = make_env_v2(scenario="full_day", root_seed=101)
    e.reset(options={"episode_seed": 101})
    eng = e.eng
    d_idx = 0
    target = next(p for p in range(K) if eng.validate_te_action(d_idx, p)[0])
    e.step(1 + d_idx * K + target)
    assert int(eng.te_dwell_remaining[d_idx]) == 2          # after interval s
    assert not eng.validate_te_action(d_idx, (target + 1) % K)[0]
    e.step(0)
    assert int(eng.te_dwell_remaining[d_idx]) == 1          # after interval s+1
    assert not eng.validate_te_action(d_idx, (target + 1) % K)[0]
    e.step(0)
    assert int(eng.te_dwell_remaining[d_idx]) == 0          # after interval s+2
    other = next(p for p in range(K)
                 if p != int(eng.current_path[d_idx])
                 and eng.path_available(d_idx, p))
    assert eng.validate_te_action(d_idx, other)[0]


def test_protected_move_at_projected_utilization_exactly_one_is_valid(env):
    """Boundary is inclusive: <= 1.0 passes, anything above fails."""
    eng = env.eng
    d_idx = int(eng._protected_idx[0])
    cur = int(eng.current_path[d_idx])
    p_idx = next(p for p in range(K) if p != cur)
    cur_links = set(eng._cand_links[d_idx][cur].tolist())
    links = eng._cand_links[d_idx][p_idx]
    eng.disconnected[d_idx] = False
    eng.te_dwell_remaining[d_idx] = 0
    vol = float(eng.demand_offered[d_idx])

    # Engineer the gross ledger so every candidate link projects to exactly
    # capacity after the move. The projection removes the demand from its
    # current path first, so a link shared by both paths needs `vol` more in
    # the ledger than one that only the candidate uses.
    eng.gross_link_load = np.zeros(N_DLINKS)
    for li in links:
        li = int(li)
        eng.gross_link_load[li] = eng.capacity[li] - (0.0 if li in cur_links else vol)
    assert eng.projected_gross_bottleneck(d_idx, p_idx) == pytest.approx(1.0, abs=1e-12)
    assert eng.validate_te_action(d_idx, p_idx)[0]

    # Any excess above capacity on a link the candidate alone uses is fatal.
    bump = next(int(li) for li in links if int(li) not in cur_links)
    eng.gross_link_load[bump] += 1e-6
    assert eng.projected_gross_bottleneck(d_idx, p_idx) > 1.0
    assert not eng.validate_te_action(d_idx, p_idx)[0]


def test_unprotected_move_may_exceed_capacity(env):
    eng = env.eng
    d_idx = int(np.flatnonzero(~eng._protected)[0])
    p_idx = next(p for p in range(K) if p != int(eng.current_path[d_idx]))
    eng.te_dwell_remaining[d_idx] = 0
    eng.gross_link_load = eng.capacity * 3.0
    assert eng.projected_gross_bottleneck(d_idx, p_idx) > 1.0
    assert eng.validate_te_action(d_idx, p_idx)[0]


def test_projected_gross_matrix_matches_the_scalar_helper(env):
    eng = env.eng
    for _ in range(3):
        eng.step_interval()
    mat = eng.projected_gross_bottleneck_matrix()
    for d_idx in range(N_DEMANDS):
        for p_idx in range(K):
            assert mat[d_idx, p_idx] == pytest.approx(
                eng.projected_gross_bottleneck(d_idx, p_idx), rel=1e-12)


def test_rejected_request_changes_no_state_at_all():
    e = make_env_v2(scenario="full_day", root_seed=101)
    e.reset(options={"episode_seed": 101})
    eng = e.eng
    d_idx = 0
    cur = int(eng.current_path[d_idx])
    eng.path_age_steps[d_idx] = 5
    before = {
        "path": eng.current_path.copy(), "age": eng.path_age_steps.copy(),
        "dwell": eng.te_dwell_remaining.copy(),
        "prev": eng.previous_te_path.copy(),
        "accepted": eng.episode_totals["accepted_te_changes"],
        "rng": eng.traffic._rng.bit_generator.state,
        "te_history": len(eng.te_history),
    }
    record = eng.apply_te_action(d_idx, cur)          # same live path -> rejected
    assert not record["accepted"]
    np.testing.assert_array_equal(eng.current_path, before["path"])
    np.testing.assert_array_equal(eng.path_age_steps, before["age"])
    np.testing.assert_array_equal(eng.te_dwell_remaining, before["dwell"])
    np.testing.assert_array_equal(eng.previous_te_path, before["prev"])
    assert eng.episode_totals["accepted_te_changes"] == before["accepted"]
    assert eng.traffic._rng.bit_generator.state == before["rng"]
    assert len(eng.te_history) == before["te_history"]
    assert eng.episode_totals["rejected_te_requests"] == 1


def test_one_accepted_action_changes_at_most_one_demand():
    e = make_env_v2(scenario="full_day", root_seed=101)
    e.reset(options={"episode_seed": 101})
    eng = e.eng
    before = eng.current_path.copy()
    legal = [a for a in np.flatnonzero(e.action_masks()) if a != 0]
    e.step(int(legal[0]))
    assert int(np.sum(before != eng.current_path)) <= 1


def test_at_most_one_accepted_te_change_per_interval():
    e = make_env_v2(scenario="full_day", root_seed=101)
    e.reset(options={"episode_seed": 101})
    rng = np.random.default_rng(11)
    for _ in range(40):
        legal = np.flatnonzero(e.action_masks())
        _, _, _, trunc, info = e.step(int(rng.choice(legal)))
        assert info["accepted_te_changes"] <= 1
        assert info["accepted_te_changes"] + info["rejected_te_requests"] <= 1
        if trunc:
            break


# =================================================== configuration fail-closed
def test_engine_rejects_a_wrongly_versioned_config():
    from mplssim.sim.engine_v2 import EngineConfigError, SimulationEngineV2
    cfg = replace(load_engine_config_v2(), version="config-v1.9")
    from mplssim.factory import get_scenarios
    with pytest.raises(EngineConfigError):
        SimulationEngineV2(get_topology(), get_traffic_config(),
                           get_scenarios()["full_day"], 1, cfg)


def test_factory_selection_is_explicit_and_defaults_to_v1():
    from mplssim.factory import make_engine, make_env
    from mplssim.rl.env import MplsTeEnv
    from mplssim.sim.engine import SimulationEngine
    from mplssim.sim.engine_v2 import SimulationEngineV2

    assert isinstance(make_engine("full_day", seed=1), SimulationEngine)
    assert isinstance(make_engine("full_day", seed=1, version="v1"), SimulationEngine)
    assert isinstance(make_engine("full_day", seed=1, version="v2"), SimulationEngineV2)
    assert isinstance(make_env(scenario="full_day"), MplsTeEnv)
    assert isinstance(make_env(version="v2", scenario="full_day"), MplsTeEnvV2)
    with pytest.raises(ValueError):
        make_engine("full_day", seed=1, version="v3")
    with pytest.raises(ValueError):
        make_env(version="v3")
