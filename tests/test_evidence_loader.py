"""Read-only evidence layer: frozen study identity and fail-closed loading.

The governed V2 study is complete and closed. Everything under `results/v2_*` is an
immutable input. These tests pin the identity of that study and prove the loader
refuses to serve anything that is not it, rather than degrading quietly.
"""

from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import pytest

from mplssim.evidence import errors, identity, loader


# ------------------------------------------------------------------- identity
def test_frozen_identity_matches_the_closed_study():
    assert identity.EVALUATION_SOURCE_SHA == "f7ed0f407c50c5472ecff89f977bc656439a8c49"
    assert identity.SEED42_SOURCE_SHA == "ca64b62fe29e45ab61aa86d642799aec5a4c25e1"
    assert identity.CONTINUATION_SOURCE_SHA == "6a8a4068b98bf9a71dead6e547595b4bbd755689"
    assert identity.SIGNED_OFF_ENV_SHA == "dca533b5c6fa9953307d01470c23cac512eb2961"
    assert identity.APPROVED_ANCESTOR_SHA == "859fdb2e0c5005b4eabd4ac1c3c8e48d2c0e31ac"
    assert identity.TRAINING_ROOTS == (42, 314159, 271828)
    assert identity.EPISODES_PER_POLICY == 35
    assert identity.TOTAL_HOLDOUT_EPISODES == 315
    assert identity.POLICY_COUNT == 9
    assert identity.ENVIRONMENT == "MplsTeEnvV2"
    assert identity.OBSERVATION_DIM == 604
    assert identity.ACTION_COUNT == 69
    assert len(identity.SCENARIOS) == 7
    assert len(identity.REWARD_COMPONENTS) == 12


def test_holdout_seeds_never_overlap_development_seeds():
    """The holdout is only untouched if its seeds were never a continuity seed."""
    assert identity.HOLDOUT_SEEDS == (1001, 1002, 1003, 1004, 1005)
    assert identity.CONTINUITY_SEEDS == (101, 102, 103, 104, 105)
    assert set(identity.HOLDOUT_SEEDS).isdisjoint(identity.CONTINUITY_SEEDS)


def test_scenario_horizons_account_for_every_recorded_step():
    assert identity.STEPS_PER_SEED == 660
    assert identity.STEPS_PER_POLICY == 3300
    assert set(identity.SCENARIO_STEPS) == set(identity.SCENARIOS)


def test_episode_arithmetic_is_self_consistent():
    assert identity.POLICY_COUNT * identity.EPISODES_PER_POLICY == \
        identity.TOTAL_HOLDOUT_EPISODES
    assert len(identity.SCENARIOS) * len(identity.HOLDOUT_SEEDS) == \
        identity.EPISODES_PER_POLICY


# --------------------------------------------------------------- happy loading
def test_final_holdout_loads_the_committed_evidence():
    fh = loader.FinalHoldout.load(loader.default_root())
    assert len(fh.per_root) == 9
    assert int(fh.per_root["episodes"].sum()) == identity.TOTAL_HOLDOUT_EPISODES
    assert len(fh.aggregate) == 5
    assert len(fh.scenario) == 63
    assert len(fh.provenance) == 6
    assert len(fh.actions) == identity.POLICY_COUNT * identity.ACTION_COUNT
    assert len(fh.integrity) == 9
    assert fh.manifest["evaluation_source_sha"] == identity.EVALUATION_SOURCE_SHA


def test_continuity_loads_and_carries_learning_curves():
    cont = loader.Continuity.load(loader.default_root())
    assert not cont.learning_curves.empty
    assert not cont.checkpoint_selection.empty
    assert not cont.per_root.empty


def test_continuity_evidence_contains_no_holdout_seed():
    """Development evidence that touched 1001-1005 would void the holdout."""
    cont = loader.Continuity.load(loader.default_root())
    assert cont.holdout_seeds_touched == ()


def test_seed42_pilot_loads():
    pilot = loader.Seed42.load(loader.default_root())
    assert not pilot.learning_curve.empty
    assert not pilot.comparison.empty
    assert pilot.manifest


# ------------------------------------------------------------------ fail closed
def _copy_holdout(tmp_path: Path) -> Path:
    dst = tmp_path / "v2_final_holdout"
    shutil.copytree(loader.default_root().final_holdout, dst)
    return dst


def test_missing_artifact_directory_fails_closed(tmp_path: Path):
    with pytest.raises(errors.ArtifactMissingError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_missing_column_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    csv = dst / "per_root_metrics.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    drop = lines[0].split(",").index("operational_return_mean")
    csv.write_text("\n".join(
        ",".join(c for i, c in enumerate(line.split(",")) if i != drop)
        for line in lines
    ) + "\n", encoding="utf-8")
    with pytest.raises(errors.SchemaError) as e:
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))
    assert "operational_return_mean" in str(e.value)


def test_wrong_episode_count_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    csv = dst / "per_root_metrics.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    csv.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(errors.IntegrityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_foreign_evaluation_sha_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    man["evaluation_source_sha"] = "0" * 40
    (dst / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(errors.IdentityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_unexpected_holdout_seed_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    man["authorization"]["seeds"] = [101, 102, 103, 104, 105]
    (dst / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(errors.IdentityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_failed_integrity_flag_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    csv = dst / "evaluation_integrity.csv"
    text = csv.read_text(encoding="utf-8")
    csv.write_text(text.replace(",True\n", ",False\n", 1), encoding="utf-8")
    with pytest.raises(errors.IntegrityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_nonzero_safety_counter_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    csv = dst / "evaluation_integrity.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    col = header.index("protected_safety_failures_total")
    row = lines[1].split(",")
    row[col] = "1"
    lines[1] = ",".join(row)
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(errors.IntegrityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_foreign_training_source_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    csv = dst / "checkpoint_provenance.csv"
    text = csv.read_text(encoding="utf-8")
    csv.write_text(text.replace(identity.SEED42_SOURCE_SHA, "b" * 40), encoding="utf-8")
    with pytest.raises(errors.IdentityError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


def test_malformed_manifest_fails_closed(tmp_path: Path):
    dst = _copy_holdout(tmp_path)
    (dst / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(errors.SchemaError):
        loader.FinalHoldout.load(loader.EvidenceRoot(tmp_path))


# ----------------------------------------------------------------- read only
def test_loading_never_opens_a_governed_path_for_writing(monkeypatch):
    """A write anywhere under results/ or runs/ is a release blocker, so the guard
    covers every loader in one pass rather than trusting review."""
    real_open = builtins.open
    guarded = [loader.default_root().results_dir.resolve(),
               (loader.default_root().results_dir.parent / "runs").resolve()]

    def guard(file, mode="r", *a, **kw):
        if not hasattr(file, "read") and any(m in mode for m in "wxa+"):
            p = Path(file).resolve()
            for g in guarded:
                assert g not in p.parents, f"write attempted inside {g}: {p}"
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guard)
    root = loader.default_root()
    loader.FinalHoldout.load(root)
    loader.Continuity.load(root)
    loader.Seed42.load(root)
