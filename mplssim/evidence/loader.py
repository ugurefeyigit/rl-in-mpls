"""Fail-closed readers for the frozen governed V2 evidence.

Every artifact this module touches is immutable. It opens files for reading only;
it never writes into `results/`, `runs/` or an experiment worktree; it never imports
a learner, constructs an environment, or loads a checkpoint.

Validation is deliberately unforgiving. An artifact that is present but does not
match the frozen study identity is more dangerous than a missing one, because it
would render numbers that look authoritative. Anything unexpected raises.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from mplssim.evidence import identity
from mplssim.evidence.errors import (
    ArtifactMissingError, IdentityError, IntegrityError, SchemaError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ locations
@dataclass(frozen=True)
class EvidenceRoot:
    """Where the compact committed evidence lives."""

    results_dir: Path

    @property
    def final_holdout(self) -> Path:
        return self.results_dir / "v2_final_holdout"

    @property
    def continuity(self) -> Path:
        return self.results_dir / "v2_three_root_continuity"

    @property
    def seed42(self) -> Path:
        return self.results_dir / "v2_seed42"


def default_root() -> EvidenceRoot:
    """The repository's own `results/`, overridable for tests and alternate checkouts."""
    override = os.environ.get("V2_EVIDENCE_ROOT", "").strip()
    return EvidenceRoot(Path(override) if override else REPO_ROOT / "results")


# --------------------------------------------------------------- primitives
def load_table(path: Path, required: Sequence[str]) -> pd.DataFrame:
    """Read a frozen CSV and prove it carries every column callers depend on."""
    if not path.is_file():
        raise ArtifactMissingError(f"frozen artifact not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:  # malformed CSV is a schema problem, not a crash
        raise SchemaError(f"{path.name} could not be parsed: {e}") from e
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{path.name} is missing required columns: {missing}")
    return df


def load_json(path: Path, required: Sequence[str]) -> dict:
    """Read a frozen manifest and prove it carries every key callers depend on."""
    if not path.is_file():
        raise ArtifactMissingError(f"frozen artifact not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SchemaError(f"{path.name} could not be parsed: {e}") from e
    if not isinstance(data, dict):
        raise SchemaError(f"{path.name} is not a JSON object")
    missing = [k for k in required if k not in data]
    if missing:
        raise SchemaError(f"{path.name} is missing required keys: {missing}")
    return data


def _truthy(series: pd.Series) -> pd.Series:
    """CSV booleans arrive as `True`/`False` strings or as real bools."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


# --------------------------------------------------- required column contracts
_METRIC_COLUMNS = (
    "policy_id", "algorithm", "training_root", "episodes",
    "operational_return_mean", "operational_return_std",
    "delivered_ratio_mean_mean", "sla_violations_demand_intervals_mean",
    "protected_disconnection_demand_intervals_mean",
    "unprotected_disconnection_demand_intervals_mean",
    "max_utilization_peak_mean", "max_utilization_mean_mean",
    "link_utilization_mean_mean", "congested_link_intervals_mean",
    "overload_ratio_mean_mean", "delay_ms_mean_mean", "delay_ms_max_mean",
    "loss_ratio_mean_mean", "accepted_te_changes_mean", "reroutes_per_hour_mean",
    "te_reversals_mean", "flaps_per_demand_mean", "moved_mbps_total_mean",
    "dwell_active_demand_intervals_mean", "dwell_remaining_mean_mean",
    "rejected_te_requests_mean", "frr_changes_mean", "frr_disconnections_mean",
    "recovery_restorations_mean", "noop_frequency_mean",
    "mean_decision_time_ms_mean", "mean_mask_time_ms_mean", "wall_time_seconds_mean",
)

_INTEGRITY_COUNTERS = (
    "invalid_action_attempts_total", "mask_disagreements_total",
    "reward_mismatches_total", "nonfinite_values_total",
    "solver_convergence_failures_total", "protected_safety_failures_total",
)


# ------------------------------------------------------------- final holdout
@dataclass(frozen=True)
class FinalHoldout:
    """The one-shot final-holdout evidence. Never blended with development stages."""

    aggregate: pd.DataFrame
    per_root: pd.DataFrame
    scenario: pd.DataFrame
    reward_components: pd.DataFrame
    actions: pd.DataFrame
    integrity: pd.DataFrame
    provenance: pd.DataFrame
    manifest: dict
    directory: Path
    stage: str = identity.STAGE_FINAL_HOLDOUT

    @classmethod
    def load(cls, root: EvidenceRoot) -> "FinalHoldout":
        d = root.final_holdout
        if not d.is_dir():
            raise ArtifactMissingError(f"final-holdout evidence directory not found: {d}")

        aggregate = load_table(d / "aggregate_metrics.csv",
                               _METRIC_COLUMNS + ("root_mean_std", "root_count"))
        per_root = load_table(d / "per_root_metrics.csv", _METRIC_COLUMNS)
        scenario = load_table(d / "scenario_metrics.csv", _METRIC_COLUMNS + ("scenario",))
        reward = load_table(
            d / "reward_components.csv",
            ("policy_id", "algorithm", "training_root", "episodes",
             "operational_return_mean", "max_abs_reward_residual")
            + tuple(f"{c}_mean" for c in identity.REWARD_COMPONENTS))
        actions = load_table(d / "action_distribution.csv",
                             ("policy_id", "algorithm", "training_root", "action",
                              "action_type", "count", "frequency"))
        integ = load_table(d / "evaluation_integrity.csv",
                           ("policy_id", "algorithm", "training_root", "episodes",
                            "unique_seeds", "unique_scenarios", "all_episodes_truncated",
                            "any_episode_terminated", "all_reward_component_sums_exact",
                            "all_checks_passed") + _INTEGRITY_COUNTERS)
        prov = load_table(d / "checkpoint_provenance.csv",
                          ("training_root", "algorithm", "checkpoint_transition",
                           "checkpoint_path", "payload_sha256", "sidecar_path",
                           "sidecar_sha256", "training_source_sha", "artifact_worktree",
                           "artifact_worktree_head", "evaluation_source_sha",
                           "evaluation_wall_seconds", "resolved_device", "gpu_name",
                           "peak_gpu_memory_bytes"))
        manifest = load_json(d / "manifest.json",
                             ("status", "evaluation_source_sha", "environment",
                              "authorization", "episodes", "integrity", "runtime",
                              "full_artifact_path"))

        cls._validate_identity(manifest, prov)
        cls._validate_counts(per_root, scenario, actions, integ, prov)
        cls._validate_integrity(integ)

        return cls(aggregate=aggregate, per_root=per_root, scenario=scenario,
                   reward_components=reward, actions=actions, integrity=integ,
                   provenance=prov, manifest=manifest, directory=d)

    # -- validation -----------------------------------------------------------
    @staticmethod
    def _validate_identity(manifest: dict, prov: pd.DataFrame) -> None:
        if manifest["evaluation_source_sha"] != identity.EVALUATION_SOURCE_SHA:
            raise IdentityError(
                f"manifest evaluation source {manifest['evaluation_source_sha']!r} is not "
                f"the frozen {identity.EVALUATION_SOURCE_SHA!r}")

        auth = manifest.get("authorization") or {}
        seeds = set(auth.get("seeds") or ())
        if seeds != set(identity.HOLDOUT_SEEDS):
            raise IdentityError(
                f"holdout seeds {sorted(seeds)} are not the frozen "
                f"{list(identity.HOLDOUT_SEEDS)}")
        scen = set(auth.get("scenarios") or ())
        if scen != set(identity.SCENARIOS):
            raise IdentityError(f"holdout scenarios {sorted(scen)} are not the frozen seven")
        for flag in ("training_performed", "tuning_performed",
                     "checkpoint_selection_performed", "checkpoint_sweep_performed",
                     "holdout_used_for_debugging"):
            if auth.get(flag) is not False:
                raise IdentityError(
                    f"manifest does not assert {flag} is False; the holdout claim is void")

        if set(prov["evaluation_source_sha"]) != {identity.EVALUATION_SOURCE_SHA}:
            raise IdentityError("checkpoint provenance cites a foreign evaluation source")
        unknown = set(prov["training_source_sha"]) - set(identity.TRAINING_SOURCE_SHAS)
        if unknown:
            raise IdentityError(f"checkpoint provenance cites unapproved training sources: {unknown}")
        if not (prov["artifact_worktree_head"] == prov["training_source_sha"]).all():
            raise IdentityError("a checkpoint's worktree head does not match its training source")
        roots = set(int(r) for r in prov["training_root"])
        if roots != set(identity.TRAINING_ROOTS):
            raise IdentityError(f"checkpoint roots {sorted(roots)} are not the frozen roots")

    @staticmethod
    def _validate_counts(per_root, scenario, actions, integ, prov) -> None:
        if len(per_root) != identity.POLICY_COUNT or not per_root["policy_id"].is_unique:
            raise IntegrityError(
                f"expected {identity.POLICY_COUNT} unique policy rows, got {len(per_root)}")
        if set(per_root["episodes"]) != {identity.EPISODES_PER_POLICY}:
            raise IntegrityError(
                f"every policy must have exactly {identity.EPISODES_PER_POLICY} episodes")
        total = int(per_root["episodes"].sum())
        if total != identity.TOTAL_HOLDOUT_EPISODES:
            raise IntegrityError(
                f"expected {identity.TOTAL_HOLDOUT_EPISODES} episodes, found {total}")

        expected_scen_rows = identity.POLICY_COUNT * len(identity.SCENARIOS)
        if len(scenario) != expected_scen_rows:
            raise IntegrityError(
                f"expected {expected_scen_rows} scenario rows, got {len(scenario)}")
        if scenario.duplicated(["policy_id", "scenario"]).any():
            raise IntegrityError("scenario table has duplicate (policy, scenario) keys")
        unknown = set(scenario["scenario"]) - set(identity.SCENARIOS)
        if unknown:
            raise IdentityError(f"scenario table cites unknown scenarios: {unknown}")
        if int(scenario["episodes"].sum()) != identity.TOTAL_HOLDOUT_EPISODES:
            raise IntegrityError("scenario grain does not roll up to 315 episodes")

        expected_action_rows = identity.POLICY_COUNT * identity.ACTION_COUNT
        if len(actions) != expected_action_rows:
            raise IntegrityError(
                f"expected {expected_action_rows} action rows, got {len(actions)}")
        for pid, g in actions.groupby("policy_id"):
            if sorted(int(a) for a in g["action"]) != list(range(identity.ACTION_COUNT)):
                raise IntegrityError(f"{pid} does not cover actions 0..{identity.ACTION_COUNT - 1}")

        if len(integ) != identity.POLICY_COUNT:
            raise IntegrityError(f"expected {identity.POLICY_COUNT} integrity rows")
        if len(prov) != len(identity.TRAINING_ROOTS) * len(identity.LEARNER_ALGORITHMS):
            raise IntegrityError("expected exactly six checkpoint provenance rows")
        if not prov["payload_sha256"].is_unique:
            raise IntegrityError("checkpoint payload hashes are not unique")

    @staticmethod
    def _validate_integrity(integ: pd.DataFrame) -> None:
        if not _truthy(integ["all_checks_passed"]).all():
            raise IntegrityError("an evaluation-integrity row did not pass its own checks")
        if not _truthy(integ["all_episodes_truncated"]).all():
            raise IntegrityError("not every episode reached normal truncation")
        if _truthy(integ["any_episode_terminated"]).any():
            raise IntegrityError("an episode terminated abnormally")
        if not _truthy(integ["all_reward_component_sums_exact"]).all():
            raise IntegrityError("a reward component sum was not exact")
        for col in _INTEGRITY_COUNTERS:
            if (integ[col].astype(float) != 0).any():
                raise IntegrityError(f"{col} is non-zero")
        if set(integ["unique_seeds"].astype(int)) != {len(identity.HOLDOUT_SEEDS)}:
            raise IntegrityError("a policy did not cover all five holdout seeds")
        if set(integ["unique_scenarios"].astype(int)) != {len(identity.SCENARIOS)}:
            raise IntegrityError("a policy did not cover all seven scenarios")


# ------------------------------------------------------- development evidence
@dataclass(frozen=True)
class Continuity:
    """Three-root continuity evidence. Development stage — never a holdout claim."""

    aggregate: pd.DataFrame
    per_root: pd.DataFrame
    scenario: pd.DataFrame
    reward_components: pd.DataFrame
    actions: pd.DataFrame
    integrity: pd.DataFrame
    learning_curves: pd.DataFrame
    checkpoint_selection: pd.DataFrame
    training_summary: pd.DataFrame
    training_integrity: pd.DataFrame
    manifest: dict
    directory: Path
    holdout_seeds_touched: tuple[int, ...] = field(default=())
    stage: str = identity.STAGE_DEVELOPMENT

    @classmethod
    def load(cls, root: EvidenceRoot) -> "Continuity":
        d = root.continuity
        if not d.is_dir():
            raise ArtifactMissingError(f"continuity evidence directory not found: {d}")

        base = ("training_root", "algorithm", "episodes", "operational_return_mean")
        aggregate = load_table(d / "aggregate_metrics.csv",
                               ("algorithm", "roots", "episodes",
                                "root_operational_return_mean",
                                "root_operational_return_std",
                                "episode_operational_return_mean",
                                "episode_operational_return_std"))
        per_root = load_table(d / "comparison_metrics_by_root.csv", base)
        scenario = load_table(d / "scenario_metrics.csv", base + ("scenario",))
        reward = load_table(d / "reward_components.csv", ("training_root", "algorithm"))
        actions = load_table(d / "action_distribution.csv",
                             ("training_root", "algorithm", "action", "count", "frequency"))
        integ = load_table(d / "evaluation_integrity.csv",
                           ("training_root", "algorithm", "episodes", "continuity_seeds",
                            "all_reward_component_sums_exact", "max_abs_reward_sum_error",
                            "all_truncated", "any_terminated", "holdout_accessed"))
        curves = load_table(d / "learning_curves.csv",
                            ("training_root", "algorithm", "checkpoint_transition",
                             "mean_operational_return", "valid", "selected",
                             "payload_sha256"))
        selection = load_table(d / "checkpoint_selection.csv",
                               ("training_root", "algorithm", "checkpoint_transition",
                                "mean_operational_return", "payload_sha256", "valid"))
        summary = load_table(d / "training_summary.csv",
                             ("training_root", "algorithm", "status",
                              "training_source_sha", "aggregate_transitions", "device",
                              "wall_time_seconds", "peak_gpu_memory_bytes"))
        tinteg = load_table(d / "training_integrity.csv",
                            ("training_root", "algorithm", "invalid_action_attempts",
                             "mask_disagreements", "only_intended_root", "seed_collisions"))
        manifest = load_json(d / "manifest.json",
                             ("status", "environment", "training_roots",
                              "continuity_evaluation_seeds", "holdout_seeds",
                              "holdout_accessed", "checkpoint_selection", "failures",
                              "superseded_valid_runs"))

        if manifest["holdout_accessed"] is not False:
            raise IdentityError("continuity manifest does not assert the holdout was untouched")
        if set(manifest["continuity_evaluation_seeds"]) != set(identity.CONTINUITY_SEEDS):
            raise IdentityError("continuity evaluation seeds are not the frozen 101-105")
        if set(int(r) for r in manifest["training_roots"]) != set(identity.TRAINING_ROOTS):
            raise IdentityError("continuity training roots are not the frozen three")
        if _truthy(integ["holdout_accessed"]).any():
            raise IdentityError("a continuity evaluation row records holdout access")

        touched = cls._holdout_seeds_in(integ, curves, per_root)
        if touched:
            raise IdentityError(
                f"continuity evidence references holdout seeds {touched}; the holdout "
                "would not be untouched")

        return cls(aggregate=aggregate, per_root=per_root, scenario=scenario,
                   reward_components=reward, actions=actions, integrity=integ,
                   learning_curves=curves, checkpoint_selection=selection,
                   training_summary=summary, training_integrity=tinteg,
                   manifest=manifest, directory=d, holdout_seeds_touched=())

    @staticmethod
    def _holdout_seeds_in(*frames: pd.DataFrame) -> tuple[int, ...]:
        """Scan every seed-bearing column for a holdout seed."""
        found: set[int] = set()
        holdout = set(identity.HOLDOUT_SEEDS)
        for df in frames:
            for col in df.columns:
                if "seed" not in col.lower():
                    continue
                for value in df[col].astype(str):
                    for token in value.replace("[", " ").replace("]", " ") \
                                      .replace(",", " ").split():
                        if token.isdigit() and int(token) in holdout:
                            found.add(int(token))
        return tuple(sorted(found))


@dataclass(frozen=True)
class Seed42:
    """The single-root pilot. Development stage."""

    comparison: pd.DataFrame
    learning_curve: pd.DataFrame
    ppo_selection: pd.DataFrame
    bandit_selection: pd.DataFrame
    manifest: dict
    directory: Path
    stage: str = identity.STAGE_DEVELOPMENT

    @classmethod
    def load(cls, root: EvidenceRoot) -> "Seed42":
        d = root.seed42
        if not d.is_dir():
            raise ArtifactMissingError(f"seed-42 evidence directory not found: {d}")
        curve_cols = ("algorithm", "checkpoint", "checkpoint_transition",
                      "mean_operational_return", "valid", "payload_sha256")
        comparison = load_table(d / "comparison.csv",
                                ("algorithm", "episodes", "operational_return_mean",
                                 "operational_return_std"))
        curve = load_table(d / "learning_curve.csv", curve_cols)
        ppo = load_table(d / "ppo_checkpoint_selection.csv", curve_cols)
        bandit = load_table(d / "bandit_checkpoint_selection.csv", curve_cols)
        manifest = load_json(d / "manifest.json",
                             ("status", "environment", "final_code_sha", "seed_policy",
                              "checkpoint_selection", "invalidated_or_superseded_runs"))
        if manifest["final_code_sha"] != identity.SEED42_SOURCE_SHA:
            raise IdentityError(
                f"seed-42 manifest cites {manifest['final_code_sha']!r}, not the frozen "
                f"{identity.SEED42_SOURCE_SHA!r}")
        return cls(comparison=comparison, learning_curve=curve, ppo_selection=ppo,
                   bandit_selection=bandit, manifest=manifest, directory=d)
