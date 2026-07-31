"""V1 preservation and classified V1/V2 differences.

Covers layer 1 and the "Identical fixed action traces through V1 and V2"
section of docs/RL_ENVIRONMENT_V2_TEST_PLAN.md.

Every difference asserted here is one the migration plan lists as expected. The
point of the file is the converse: that nothing V1 owns has moved, and that the
differences which do exist are exactly the approved P0/P1 corrections.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mplssim.experiments.v2_factory import make_engine_v2, make_env_v2
from mplssim.factory import get_topology, get_traffic_config, make_engine
from mplssim.rl.env import GLOBAL_FEATURES, LINK_FEATURES, MplsTeEnv
from mplssim.sim.engine import SimulationEngine

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The commit this work is based on. V1 preservation is asserted against it.
AUDITED_BASE_COMMIT = "5e429bcc27277866cdba7cc07087fbb4d4b72a6d"

#: Files this task is permitted to create or touch (implementation prompt,
#: "File ownership"). Anything else appearing in the diff is a violation.
ALLOWED_NEW_FILES = {
    "mplssim/sim/engine_v2.py",
    "mplssim/rl/env_v2.py",
    "mplssim/rl/reward_v2.py",
    "mplssim/paths/candidates_v2.py",
    "mplssim/experiments/v2_factory.py",
    "configs/experiments/rl_env_v2.yaml",
    "configs/experiments/rl_observation_v2.yaml",
    "configs/experiments/rl_reward_v2.yaml",
    "tests/test_env_v2.py",
    "tests/test_reward_v2.py",
    "tests/test_transition_v2.py",
    "tests/test_v1_v2_compatibility.py",
    "scripts/validate_env_v2.py",
    # V2 seed-42 learning comparison tooling (never imported by V1).
    "configs/experiments/learning_v2.yaml",
    "mplssim/experiments/learning_common.py",
    "mplssim/experiments/masked_bandit.py",
    "mplssim/experiments/trainers_v2.py",
    "mplssim/experiments/evaluation_v2.py",
    "scripts/train_v2.py",
    "scripts/evaluate_v2.py",
    "scripts/benchmark_v2.py",
    "scripts/compare_v2.py",
    "scripts/final_holdout_v2.py",
    "tests/test_learning_v2.py",
    "docs/superpowers/specs/2026-07-30-v2-learning-comparison-design.md",
    "docs/superpowers/plans/2026-07-30-v2-learning-comparison.md",
    "NEXT_STAGE_HANDOFF.md",
    # Post-study productization: a read-only presentation layer over the CLOSED
    # V2 evidence. None of these trains, tunes, evaluates a checkpoint, or writes
    # into results/ or runs/ — tests/test_evidence_loader.py and
    # tests/test_evidence_api.py assert the absence of writes to governed paths.
    # V1's models, results, figures, configs and simulation source stay byte
    # identical; the two tests below still enforce that.
    "mplssim/evidence/__init__.py",
    "mplssim/evidence/identity.py",
    "mplssim/evidence/errors.py",
    "mplssim/evidence/loader.py",
    "mplssim/evidence/claims.py",
    "mplssim/evidence/replay.py",
    "server/evidence_api.py",
    "frontend/study.html",
    "frontend/js/study.js",
    "frontend/css/study.css",
    "tests/test_evidence_loader.py",
    "tests/test_evidence_claims.py",
    "tests/test_evidence_replay.py",
    "tests/test_evidence_api.py",
    "tests/test_study_ui.py",
    "docs/V2_EVIDENCE_AUDIT.md",
    "docs/TECHNICAL_DEFENSE.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/V3_RESEARCH_BACKLOG.md",
    "docs/superpowers/plans/2026-07-31-post-study-productization.md",
}
ALLOWED_MODIFIED_FILES = {
    ".gitignore",
    "mplssim/factory.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "tests/test_state_machine.py",
    # Post-study productization: mounting the GET-only evidence router and the
    # /study route, plus documentation that described only V1.
    "server/main.py",
    "README.md",
    "CURRENT_SYSTEM_BASELINE.md",
    "docs/API.md",
    "docs/ARCHITECTURE.md",
    "docs/PRESENTATION_MODE.md",
    "docs/DEMO_SCRIPT.md",
    "docs/UI_ACCEPTANCE_TESTS.md",
}

#: The single directory the implementation prompt designates for validation
#: output ("Validation output may be written only to:
#: results/environment_v2_validation/"). Every other path under results/ and
#: models/ stays protected.
ALLOWED_OUTPUT_PREFIX = "results/environment_v2_validation/"
ALLOWED_LEARNING_OUTPUT_PREFIX = "results/v2_seed42/"
ALLOWED_CONTINUITY_OUTPUT_PREFIX = "results/v2_three_root_continuity/"
ALLOWED_FINAL_HOLDOUT_OUTPUT_PREFIX = "results/v2_final_holdout/"


def _is_allowed(path: str) -> bool:
    return (path in ALLOWED_NEW_FILES or path in ALLOWED_MODIFIED_FILES
            or path.startswith(ALLOWED_OUTPUT_PREFIX)
            or path.startswith(ALLOWED_LEARNING_OUTPUT_PREFIX)
            or path.startswith(ALLOWED_CONTINUITY_OUTPUT_PREFIX)
            or path.startswith(ALLOWED_FINAL_HOLDOUT_OUTPUT_PREFIX))

#: The two V1 candidates that transit a PE. They must still be present in V1.
V1_PE_TRANSIT = {
    "D10": ("PE4", "P3", "P6", "A1", "PE7", "A2", "PE8"),
    "D16": ("PE4", "P3", "PE3", "P2", "P5", "P8", "PE5"),
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _newline_renderings(raw: bytes) -> dict[str, bytes]:
    """The same content rendered with LF and with CRLF line endings."""
    lf = raw.replace(b"\r\n", b"\n")
    return {"raw": raw, "lf": lf, "crlf": lf.replace(b"\n", b"\r\n")}


def content_hash_matches(path: Path, expected: str) -> tuple[bool, str]:
    """Does this text file's *content* hash to ``expected`` in any newline form?

    ``results/v1_manifest.json`` was frozen from one particular working tree:
    its config hashes were taken over LF content while its evaluation-artifact
    hashes were taken over CRLF content. This repository sets
    ``core.autocrlf=true``, so a fresh clone renders every tracked text file
    with CRLF and half of those recorded hashes stop matching -- a checkout
    artefact, not a content change.

    Comparing every newline rendering keeps the check line-ending independent
    without weakening it: two files can only agree here if one is exactly the
    newline transform of the other, which is precisely the difference being
    excused. Any real edit -- a changed digit, a dropped row -- alters the
    normalized bytes and breaks all three forms. Binary artifacts (model
    archives) never go through this helper and stay strictly raw-hashed.
    """
    raw = path.read_bytes()
    for name, candidate in _newline_renderings(raw).items():
        if hashlib.sha256(candidate).hexdigest() == expected:
            return True, name
    return False, "none"


def assert_unchanged_since_base(rel: str) -> None:
    """Git's own normalization-aware check that a tracked file matches the base."""
    diff = _git("diff", "--name-only", AUDITED_BASE_COMMIT, "--", rel).strip()
    assert not diff, f"{rel} differs from the audited base commit"


# ================================================================ 1. V1 shapes
def test_v1_observation_and_action_shapes_are_unchanged():
    env = MplsTeEnv(scenario="evening_peak", base_seed=1)
    obs, _ = env.reset(options={"episode_seed": 1})
    assert env.observation_space.shape == (586,)
    assert obs.shape == (586,)
    assert env.action_space.n == 69
    assert LINK_FEATURES * 64 + (7 + 2 * 4) * 17 + GLOBAL_FEATURES == 586


def test_v1_action_numbering_and_candidate_table_are_unchanged():
    """V1 must still contain the PE-transit candidates the V2 filter removes."""
    eng = make_engine("full_day", seed=101)
    assert isinstance(eng, SimulationEngine)
    for demand_id, routers in V1_PE_TRANSIT.items():
        d = eng.demand_by_id[demand_id]
        assert d.candidate_paths[3] == routers, demand_id
    # action decoding is unchanged
    for action in (1, 5, 37, 40, 61, 64, 68):
        d_idx, p_idx = divmod(action - 1, 4)
        assert 0 <= d_idx < 17 and 0 <= p_idx < 4
    assert eng.demands[9].candidate_paths[3] == V1_PE_TRANSIT["D10"]   # action 40
    assert eng.demands[15].candidate_paths[3] == V1_PE_TRANSIT["D16"]  # action 64


def test_v1_reward_module_is_untouched():
    from mplssim.rl.reward import load_reward_config
    cfg = load_reward_config()
    assert cfg.weights["delivered"] == 1.0
    assert cfg.weights["max_util"] == 1.2
    assert cfg.weights["loss"] == 2.0
    assert cfg.weights["reroute"] == 0.08
    assert cfg.weights["flap"] == 0.25
    assert cfg.util_free_threshold == 0.60


def test_v1_engine_still_terminates_on_full_disconnection():
    """V1 semantics preserved: only V2 runs a fully disconnected episode out."""
    eng = make_engine("full_day", seed=101)
    eng.disconnected[:] = True
    assert eng.all_disconnected is True


# ================================================== 1b. tracked-file immutability
def test_no_v1_tracked_file_changed_relative_to_the_audited_base():
    changed = {line.split("\t", 1)[1].strip() for line in
               _git("diff", "--name-status", AUDITED_BASE_COMMIT).splitlines() if line}
    illegal = {p for p in changed if not _is_allowed(p)}
    assert not illegal, f"files changed outside the allowed list: {sorted(illegal)}"


def test_only_allowed_files_are_new_or_modified():
    # -uall expands untracked directories to individual files, so a new
    # directory cannot hide its contents behind a single entry.
    status = _git("status", "--porcelain", "-uall").splitlines()
    touched = {line[3:].strip().strip('"') for line in status if line.strip()}
    illegal = {p for p in touched if not _is_allowed(p)}
    assert not illegal, f"untracked/modified files outside the allowed list: {sorted(illegal)}"


def test_models_results_figures_and_v1_configs_are_byte_identical_to_the_base():
    protected_prefixes = ("models/", "results/", "server/", "frontend/",
                          "docs/", "runs/")
    protected_exact = {"configs/topology.yaml", "configs/traffic_classes.yaml",
                       "configs/scenarios.yaml", "configs/training.yaml",
                       "configs/reward.yaml", "configs/baselines.yaml",
                       "mplssim/rl/env.py", "mplssim/rl/reward.py",
                       "mplssim/sim/engine.py", "mplssim/sim/models.py",
                       "mplssim/paths/candidates.py", "mplssim/traffic/model.py",
                       "mplssim/core/model.py", "mplssim/core/topology.py",
                       "mplssim/baselines/controllers.py",
                       "mplssim/experiments/runner.py"}
    changed = {line.split("\t", 1)[1].strip() for line in
               _git("diff", "--name-status", AUDITED_BASE_COMMIT).splitlines() if line}
    for path in changed:
        # The designated validation-output directory is the one carve-out; the
        # rest of results/ (v1_manifest.json, eval_*, figures/) and all of
        # models/ stay protected.
        if path.startswith(ALLOWED_OUTPUT_PREFIX):
            continue
        if path.startswith(ALLOWED_LEARNING_OUTPUT_PREFIX):
            continue
        if path.startswith(ALLOWED_CONTINUITY_OUTPUT_PREFIX):
            continue
        if path.startswith(ALLOWED_FINAL_HOLDOUT_OUTPUT_PREFIX):
            continue
        if path in ALLOWED_NEW_FILES or path in ALLOWED_MODIFIED_FILES:
            continue
        assert path not in protected_exact, path
        assert not path.startswith(protected_prefixes), path


def test_the_validation_output_carve_out_touches_nothing_v1_owns():
    """Explicitly: every frozen V1 path is byte-identical to the audited base.

    Stated separately from the prefix rule above so the
    results/environment_v2_validation/ carve-out cannot quietly widen into the
    frozen V1 results or models.
    """
    changed = {line.split("\t", 1)[1].strip() for line in
               _git("diff", "--name-status", AUDITED_BASE_COMMIT).splitlines() if line}
    v1_owned = {p for p in changed
                if (p.startswith("models/")
                    or (p.startswith("results/")
                        and not p.startswith(ALLOWED_OUTPUT_PREFIX)
                        and not p.startswith(ALLOWED_LEARNING_OUTPUT_PREFIX)
                        and not p.startswith(ALLOWED_CONTINUITY_OUTPUT_PREFIX)
                        and not p.startswith(ALLOWED_FINAL_HOLDOUT_OUTPUT_PREFIX)))}
    assert not v1_owned, f"frozen V1 artifacts changed: {sorted(v1_owned)}"
    assert not _git("diff", "--name-only", AUDITED_BASE_COMMIT, "--",
                    "results/v1_manifest.json").strip()
    assert not _git("diff", "--name-only", AUDITED_BASE_COMMIT, "--",
                    "results/figures").strip()


def test_v1_manifest_referenced_configs_still_match_their_recorded_hashes():
    """Frozen V1 config content is unchanged, independent of checkout newlines.

    One documented pre-existing exception: configs/topology.yaml already
    diverged from the frozen v1-original-results manifest between commit
    10e6d59 and the audited base 5e429bc. It is byte-identical to the audited
    base here, so this task did not move it, and correcting the manifest would
    mean editing a frozen V1 artifact.
    """
    manifest = json.loads((REPO_ROOT / "results/v1_manifest.json")
                          .read_text(encoding="utf-8"))
    known_pre_existing = {"configs/topology.yaml"}
    checked = 0
    for key, want in manifest["configs"].items():
        rel = key.replace("\\", "/")
        # Layer 1: git's own normalization-aware comparison against the base.
        assert_unchanged_since_base(rel)
        matched, form = content_hash_matches(REPO_ROOT / rel, want)
        if rel in known_pre_existing:
            assert not matched, (
                f"{rel} now matches the manifest in the {form} newline form; the "
                f"documented pre-existing divergence is gone, so this test needs "
                f"updating")
            continue
        # Layer 2: content matches the frozen manifest in some newline form.
        assert matched, rel
        checked += 1
    assert checked == 5, f"expected 5 verifiable config hashes, checked {checked}"


def test_v1_manifest_referenced_evaluation_artifacts_still_match():
    """The three published evaluation artifacts, line-ending independent."""
    manifest = json.loads((REPO_ROOT / "results/v1_manifest.json")
                          .read_text(encoding="utf-8"))
    artifacts = manifest["evaluation"]["artifacts"]
    assert set(artifacts) == {"eval_summary.csv", "eval_stats.csv",
                              "eval_summary.json"}
    for key, want in artifacts.items():
        rel = f"results/{key}"
        assert_unchanged_since_base(rel)
        matched, _form = content_hash_matches(REPO_ROOT / rel, want)
        assert matched, rel


def test_v1_model_artifacts_are_strictly_raw_hashed():
    """Binaries are never newline-normalized; any byte change is fatal."""
    manifest = json.loads((REPO_ROOT / "results/v1_manifest.json")
                          .read_text(encoding="utf-8"))
    for rel, want in (("models/ppo_te/best_model.zip",
                       manifest["model"]["best_model_sha256"]),
                      ("models/ppo_te/evaluations.npz",
                       manifest["model"]["evaluations_npz_sha256"])):
        assert_unchanged_since_base(rel)
        assert sha256(REPO_ROOT / rel) == want, rel
    assert manifest["model"]["observation_dim"] == 586
    assert manifest["model"]["action_dim"] == 69


def test_newline_agnostic_hash_helper_still_rejects_real_content_changes(tmp_path):
    """Guards the helper itself: only newline form may vary, never content."""
    original = b"alpha,1\nbeta,2\n"
    expected = hashlib.sha256(original).hexdigest()

    same_lf = tmp_path / "lf.csv"
    same_lf.write_bytes(original)
    assert content_hash_matches(same_lf, expected)[0]

    same_crlf = tmp_path / "crlf.csv"
    same_crlf.write_bytes(b"alpha,1\r\nbeta,2\r\n")
    assert content_hash_matches(same_crlf, expected) == (True, "lf")

    for corrupted in (b"alpha,1\nbeta,3\n",        # a changed digit
                      b"alpha,1\n",                 # a dropped row
                      b"alpha,1\nbeta,2\ngamma,3\n",  # an added row
                      b"alpha,1\nbeta,2\n\n",       # an extra blank line
                      b" alpha,1\nbeta,2\n"):       # leading whitespace
        bad = tmp_path / "bad.csv"
        bad.write_bytes(corrupted)
        assert not content_hash_matches(bad, expected)[0], corrupted


def test_importing_v2_does_not_populate_or_mutate_v1_caches():
    from mplssim.paths import candidates_v2
    from mplssim.sim import engine as v1_engine

    v1_before = dict(v1_engine._CANDIDATE_CACHE)
    make_engine_v2("full_day", episode_seed=1)
    assert dict(v1_engine._CANDIDATE_CACHE) == v1_before
    assert candidates_v2._CANDIDATE_CACHE_V2 is not v1_engine._CANDIDATE_CACHE
    # and a V1 engine built afterwards still gets V1 paths
    eng = make_engine("full_day", seed=1)
    assert eng.demand_by_id["D10"].candidate_paths[3] == V1_PE_TRANSIT["D10"]


def test_default_entry_point_selection_remains_v1():
    from mplssim.factory import make_engine as factory_make_engine
    from mplssim.factory import make_env as factory_make_env
    assert isinstance(factory_make_engine("full_day", seed=1), SimulationEngine)
    assert isinstance(factory_make_env(scenario="full_day"), MplsTeEnv)

    for script in ("scripts/train.py", "scripts/evaluate.py"):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert '"--env-version"' in text, script
        assert 'default="v1"' in text, f"{script} must default to V1"


def test_v1_and_v2_are_distinct_classes_with_distinct_shapes():
    from mplssim.rl.env_v2 import MplsTeEnvV2
    v1 = MplsTeEnv(scenario="full_day", base_seed=1)
    v2 = make_env_v2(scenario="full_day", root_seed=1)
    assert not isinstance(v2, MplsTeEnv)
    assert v1.observation_space.shape != v2.observation_space.shape
    assert v1.action_space.n == v2.action_space.n == 69


# ======================================== classified V1 vs V2 differences
FIXED_PAIRS = [("full_day", 101), ("link_failure", 101),
               ("ood_double_failure", 101), ("overload_stress", 103)]


@pytest.mark.parametrize("scenario,seed", FIXED_PAIRS)
def test_reset_offered_traffic_is_byte_identical_between_v1_and_v2(scenario, seed):
    """AR state is exactly zero at t=0, so the traffic model itself is unchanged.

    Any later divergence is the mandated P0-2 child-stream derivation, not a
    change to how offered volume is computed.
    """
    v1 = make_engine(scenario, seed=seed)
    v2 = make_engine_v2(scenario, episode_seed=seed)
    assert v1.t_min == v2.t_min == 0.0
    np.testing.assert_array_equal(v1.demand_volumes, v2.demand_offered)


@pytest.mark.parametrize("scenario,seed", FIXED_PAIRS)
def test_v1_and_v2_run_the_same_number_of_control_intervals(scenario, seed):
    v1 = make_engine(scenario, seed=seed)
    v2 = make_engine_v2(scenario, episode_seed=seed)
    n1 = n2 = 0
    while not v1.done:
        v1.step_interval()
        n1 += 1
    while not v2.done:
        v2.step_interval()
        n2 += 1
    assert n1 == n2
    assert v1.t_min == v2.t_min


def test_ar_stream_divergence_is_the_mandated_child_stream_correction():
    """Documented, expected difference: V1 seeds AR with default_rng(seed)."""
    v1 = make_engine("full_day", seed=101)
    v2 = make_engine_v2("full_day", episode_seed=101)
    v1.step_interval()
    v2.step_interval()
    assert not np.allclose(v1.demand_volumes, v2.demand_offered)
    expected_ar = np.random.default_rng(np.random.SeedSequence([101, 2]))
    fresh = make_engine_v2("full_day", episode_seed=101)
    fresh.traffic.advance_noise()
    reference = np.zeros(17) + 0.0
    reference = 0.9 * reference + fresh.traffic._sigmas * expected_ar.standard_normal(17)
    np.testing.assert_allclose(fresh.traffic._noise, reference, rtol=1e-15)


def test_failure_is_visible_one_control_interval_earlier_in_v2():
    """P0-1: a link scheduled down at 60 is observable at the 60 boundary in V2
    and only at the 65 boundary in V1."""
    v1 = make_engine("link_failure", seed=101)
    v2 = make_engine_v2("link_failure", episode_seed=101)
    first_down = {}
    for name, eng in (("v1", v1), ("v2", v2)):
        while not eng.done:
            eng.step_interval()
            if not eng.link_up["L11"]:
                first_down[name] = eng.t_min
                break
    assert first_down["v2"] == 60.0
    assert first_down["v1"] == 65.0
    assert first_down["v2"] < first_down["v1"]


def test_v2_never_applies_an_event_before_its_configured_time():
    v2 = make_engine_v2("link_failure", episode_seed=101)
    while v2.t_min < 60.0:
        v2.step_interval()
        if v2.t_min < 60.0:
            assert v2.link_up["L11"] is True, (
                f"L11 down at t={v2.t_min}, before its configured time 60")
    assert v2.link_up["L11"] is False


def test_recovery_is_also_one_interval_earlier_in_v2():
    v1 = make_engine("link_failure", seed=101)
    v2 = make_engine_v2("link_failure", episode_seed=101)
    first_up = {}
    for name, eng in (("v1", v1), ("v2", v2)):
        seen_down = False
        while not eng.done:
            eng.step_interval()
            if not eng.link_up["L11"]:
                seen_down = True
            elif seen_down:
                first_up[name] = eng.t_min
                break
    assert first_up["v2"] == 180.0
    assert first_up["v1"] == 185.0


def _sort_key(topo, routers):
    from mplssim.paths.candidates import path_admin_cost as cost
    from mplssim.paths.candidates_v2 import path_propagation_ms as prop
    return (cost(topo, routers), prop(topo, routers), routers)


def test_candidate_table_differences_are_exactly_the_approved_ones():
    """Every V1->V2 candidate change is either the P0-3 PE-transit removal or
    a strict improvement under the mandated ``(admin_cost, propagation_delay,
    router_tuple)`` ordering.

    V1 kept whichever equal-cost path NetworkX happened to enumerate first,
    which the audit flagged as a source of stable action-index bias. V2 selects
    the sort-key-minimal admissible set, so a V1 path is displaced only by one
    that is strictly better on that key.
    """
    topo = get_topology()
    v1 = make_engine("full_day", seed=101)
    v2 = make_engine_v2("full_day", episode_seed=101)

    pe_transit_removals, tie_break_substitutions, reorder_only, identical = [], [], [], []
    for d1, d2 in zip(v1.demands, v2.demands):
        old, new = d1.candidate_paths, d2.candidate_paths
        if old == new:
            identical.append(d1.id)
            continue
        dropped = set(old) - set(new)
        added = set(new) - set(old)
        if not dropped:
            assert not added, d1.id
            reorder_only.append(d1.id)
            continue
        for gone in dropped:
            if d1.id in V1_PE_TRANSIT and gone == V1_PE_TRANSIT[d1.id]:
                pe_transit_removals.append(d1.id)
                continue
            # a non-PE-transit path may only be displaced by a strictly
            # better-keyed replacement
            replacements = sorted("-".join(cand) for cand in added)
            assert any(_sort_key(topo, cand) < _sort_key(topo, gone)
                       for cand in added), (
                f"{d1.id}: dropped {'-'.join(gone)} without a better-keyed "
                f"replacement among {replacements}")
            tie_break_substitutions.append(d1.id)
        for cand in added:
            assert all(topo.routers[r].role in ("P", "AGG") for r in cand[1:-1])

    assert sorted(pe_transit_removals) == ["D10", "D16"]
    assert sorted(set(tie_break_substitutions)) == ["D11", "D13", "D5"]
    assert sorted(reorder_only) == ["D15", "D4", "D7"]
    assert len(identical) == 9


def test_v2_continues_after_all_demands_disconnect_while_v1_would_terminate():
    v1 = make_engine("full_day", seed=101)
    v2 = make_engine_v2("full_day", episode_seed=101)
    v1.disconnected[:] = True
    v2.disconnected[:] = True
    assert v1.all_disconnected and v2.all_disconnected
    env = make_env_v2(scenario="full_day", root_seed=101)
    env.reset(options={"episode_seed": 101})
    env.eng.disconnected[:] = True
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False and truncated is False


def test_downstream_load_differs_only_where_upstream_loss_exists():
    """P1: with no loss anywhere V2's carried ledger equals its gross ledger."""
    v2 = make_engine_v2("overload_stress", episode_seed=103)
    lossy = lossless = 0
    while not v2.done:
        v2.step_interval()
        if np.max(v2.link_loss) == 0.0:
            np.testing.assert_array_equal(v2.link_input_load, v2.gross_link_load)
            lossless += 1
        else:
            assert np.any(v2.link_input_load < v2.gross_link_load - 1e-9)
            lossy += 1
    assert lossy > 0


def test_v1_and_v2_reward_component_sets_are_different_by_design():
    from mplssim.rl.reward import compute_reward
    from mplssim.rl.reward_v2 import COMPONENT_ORDER
    v1_interval = {
        "delivered_ratio": 1.0, "priority_sla_success": 1.0, "max_util": 0.5,
        "util_std": 0.0, "mean_delay_ms": 0.0, "loss_ratio": 0.0,
        "sla_violation_fraction": 0.0, "overload_ratio": 0.0,
        "disconnected_demands": 0, "n_demands": 17,
    }
    _, v1_comp = compute_reward(v1_interval, False, False, False)
    assert set(v1_comp) != set(COMPONENT_ORDER)
    assert "priority_sla" in v1_comp and "priority_sla" not in COMPONENT_ORDER
    assert "protected_disconnect" in COMPONENT_ORDER
    assert "protected_disconnect" not in v1_comp
