"""The frozen V2 checkpoints this product may load for a live demonstration.

Six checkpoints exist: one MaskablePPO and one masked contextual bandit per
training root (42, 314159, 271828). They were selected *before* the final
holdout ran, by the preregistered continuity rule recorded in
``results/v2_three_root_continuity/checkpoint_selection.csv``. Nothing in this
module reads, ranks or defaults on final-holdout performance, and nothing here
trains, tunes, evaluates or writes.

Loading is fail-closed at four independent layers, in this order:

1. the artifact root and payload must exist on this machine;
2. the payload and its sidecar must hash to the SHA-256 recorded here, which is
   the same hash the governed provenance record carries;
3. the sidecar must declare V2, the expected algorithm, the expected training
   root, the expected transition and the expected training source commit;
4. the sidecar's stored environment identity must validate against the live V2
   environment through :func:`validate_environment_metadata`, which never pads,
   truncates, reorders or falls back to V1.

Any failure raises :class:`CheckpointUnavailable` carrying a sentence that says
what is wrong. A V1 model is never substituted: V1's 586-value observation and
its different candidate-path table make "adapting" it a different experiment.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the training worktrees holding the frozen checkpoints live. Optional:
#: the default is the main worktree's ``.worktrees`` directory, which is where
#: the governed study wrote them.
ARTIFACT_ROOT_ENV = "V2_LIVE_CHECKPOINTS"

#: Training roots, in the order the study registers them.
TRAINING_ROOTS: tuple[int, ...] = (42, 314159, 271828)

#: Learner algorithms with a bound V2 checkpoint.
LEARNER_ALGORITHMS: tuple[str, ...] = ("masked_bandit", "maskable_ppo")

#: Neutral, documented default. Root 42 is the study's primary *scientific*
#: training root — the one the seed-42 source identity is named for and the
#: first entry of :data:`TRAINING_ROOTS`. It is chosen because of that fixed
#: identity, never because of how any checkpoint scored on the final holdout.
#: Choosing a root by holdout return would let holdout results feed back into a
#: selection decision, which the study's authorization forbids.
DEFAULT_ROOT = 42
DEFAULT_ROOT_RULE = (
    "The default live root is the study's primary seed-42 scientific training "
    "root, fixed by identity and first in the registered root order. It is not "
    "chosen from final-holdout performance."
)

#: Default live controller. The bandit is the study's headline learner, but the
#: default is stated as a product choice, not as a claim re-derived from the
#: holdout at runtime.
DEFAULT_POLICY = "masked_bandit"


class CheckpointUnavailable(RuntimeError):
    """A V2 checkpoint cannot be loaded, with the precise reason why."""


@dataclass(frozen=True)
class CheckpointEntry:
    """One immutable registry row. Every field is part of the identity."""

    training_root: int
    algorithm: str
    transition: int
    worktree: str
    run_directory: str
    payload_name: str
    payload_sha256: str
    sidecar_sha256: str
    training_source_sha: str

    @property
    def id(self) -> str:
        return f"{self.algorithm}-root{self.training_root}-{self.transition}"

    @property
    def relative_path(self) -> str:
        return (f"{self.worktree}/runs/v2/{self.run_directory}/checkpoints/"
                f"{self.payload_name}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "training_root": self.training_root,
            "algorithm": self.algorithm,
            "transition": self.transition,
            "worktree": self.worktree,
            "run_directory": self.run_directory,
            "payload_name": self.payload_name,
            "relative_path": self.relative_path,
            "payload_sha256": self.payload_sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "training_source_sha": self.training_source_sha,
            "selection": "pre-holdout continuity selection",
        }


#: The immutable six-checkpoint continuity registry. Hashes and paths are
#: transcribed from ``results/v2_final_holdout/checkpoint_provenance.csv``;
#: transitions match ``results/v2_three_root_continuity/checkpoint_selection.csv``.
REGISTRY: tuple[CheckpointEntry, ...] = (
    CheckpointEntry(
        training_root=42, algorithm="masked_bandit", transition=250000,
        worktree="seed42", run_directory="seed42_masked_bandit_final",
        payload_name="checkpoint_000250000.pt",
        payload_sha256="c15097700eac518ee259cba67e34e4fba1716881ab3dd912188b55da0c79bf49",
        sidecar_sha256="4c041138f64883c39b5500229cbc55852bac8e164b20eedde0d89a1c9a2e6656",
        training_source_sha="ca64b62fe29e45ab61aa86d642799aec5a4c25e1"),
    CheckpointEntry(
        training_root=42, algorithm="maskable_ppo", transition=250000,
        worktree="seed42", run_directory="seed42_maskable_ppo_final",
        payload_name="checkpoint_000250000.zip",
        payload_sha256="d34cc77ded05b064fa2a39dbe5c5ccc3126c9e6cf85e36c1b507127c987f5676",
        sidecar_sha256="5b82401d32dca4c1bf3c15301709282a6dd3b8cafac1bc5ee184ef3aece2a67e",
        training_source_sha="ca64b62fe29e45ab61aa86d642799aec5a4c25e1"),
    CheckpointEntry(
        training_root=314159, algorithm="masked_bandit", transition=300000,
        worktree="continuity_v2", run_directory="seed314159_masked_bandit_final_r2",
        payload_name="checkpoint_000300000.pt",
        payload_sha256="fd474430e9f5ed60d09d82e3d08390151f54c8c0ca10b5abd98fe11d5d2c8433",
        sidecar_sha256="d3b9aaa9561379cc5f201b0cd1f9fc1e281a20029603279ce38b27ffd8b3d9f0",
        training_source_sha="6a8a4068b98bf9a71dead6e547595b4bbd755689"),
    CheckpointEntry(
        training_root=314159, algorithm="maskable_ppo", transition=350000,
        worktree="continuity_v2", run_directory="seed314159_maskable_ppo_final_r2",
        payload_name="checkpoint_000350000.zip",
        payload_sha256="0af41be78102617b103c3e21ebb0ba26ae251f2626ff50b30c0887fdb1320489",
        sidecar_sha256="431d9702c619712ead29f71d2fe4a898dd5fa98220b7d74339e7b8f14cde75c1",
        training_source_sha="6a8a4068b98bf9a71dead6e547595b4bbd755689"),
    CheckpointEntry(
        training_root=271828, algorithm="masked_bandit", transition=400000,
        worktree="continuity_v2", run_directory="seed271828_masked_bandit_final",
        payload_name="checkpoint_000400000.pt",
        payload_sha256="d9c31430ad4320ae238f6d3aa833614edc120f7411c5a3e99372c85707116e73",
        sidecar_sha256="76fd196ed41452c1452bce59afc03c6b474a597203213dc2f37e89e03b1748ff",
        training_source_sha="6a8a4068b98bf9a71dead6e547595b4bbd755689"),
    CheckpointEntry(
        training_root=271828, algorithm="maskable_ppo", transition=150000,
        worktree="continuity_v2", run_directory="seed271828_maskable_ppo_final",
        payload_name="checkpoint_000150000.zip",
        payload_sha256="40d0f9b7fe92449e6e8bfe2bcb44604ac2a5002c0f2a662dbad6cf70c219fb79",
        sidecar_sha256="6352ac6e38d2228a7f2f5bf5118fbcde2f669e950391cf589f99fe84576efdde",
        training_source_sha="6a8a4068b98bf9a71dead6e547595b4bbd755689"),
)


def entry_for(algorithm: str, training_root: int = DEFAULT_ROOT) -> CheckpointEntry:
    """The single registry row for one algorithm and root, or fail closed."""
    for entry in REGISTRY:
        if entry.algorithm == algorithm and entry.training_root == int(training_root):
            return entry
    raise CheckpointUnavailable(
        f"No frozen V2 checkpoint is registered for algorithm {algorithm!r} at "
        f"training root {training_root}. Registered pairs: "
        + ", ".join(sorted(f"{e.algorithm}@{e.training_root}" for e in REGISTRY))
        + ".")


# ------------------------------------------------------------- artifact root
def _git_main_worktree() -> Path | None:
    """The main worktree root, so a linked worktree still finds `.worktrees`."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    common = Path(result.stdout.strip())
    return common.parent if common.name == ".git" else None


def artifact_root() -> Path | None:
    """Directory holding the training worktrees, or ``None`` if not present.

    ``V2_LIVE_CHECKPOINTS`` wins when set; otherwise the main worktree's
    ``.worktrees`` directory is used, which is where the study wrote them.
    """
    configured = os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_dir() else None
    for base in (_git_main_worktree(), REPO_ROOT, *REPO_ROOT.parents):
        if base is None:
            continue
        candidate = base / ".worktrees"
        if candidate.is_dir():
            return candidate
    return None


def payload_path(entry: CheckpointEntry) -> Path | None:
    root = artifact_root()
    return None if root is None else root / Path(entry.relative_path)


def sidecar_path(entry: CheckpointEntry) -> Path | None:
    payload = payload_path(entry)
    return None if payload is None else Path(f"{payload}.metadata.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------- verifying
def verify(entry: CheckpointEntry) -> dict[str, Any]:
    """Verify one registry row completely. Raises :class:`CheckpointUnavailable`.

    Returns the validated sidecar metadata. Never mutates anything on disk.
    """
    import json

    root = artifact_root()
    if root is None:
        raise CheckpointUnavailable(
            f"No V2 checkpoint artifact root is available. The frozen "
            f"checkpoints live outside Git under the study's training "
            f"worktrees; set {ARTIFACT_ROOT_ENV} to the directory that holds "
            f"'seed42' and 'continuity_v2'.")
    payload = root / Path(entry.relative_path)
    sidecar = Path(f"{payload}.metadata.json")
    if not payload.is_file():
        raise CheckpointUnavailable(
            f"{entry.id}: checkpoint payload {payload} is missing.")
    if not sidecar.is_file():
        raise CheckpointUnavailable(
            f"{entry.id}: checkpoint metadata sidecar {sidecar} is missing.")

    actual_payload = _sha256_file(payload)
    if actual_payload != entry.payload_sha256:
        raise CheckpointUnavailable(
            f"{entry.id}: payload SHA-256 is {actual_payload}, the governed "
            f"registry records {entry.payload_sha256}. Refusing to load a "
            f"checkpoint that is not the one the study froze.")
    actual_sidecar = _sha256_file(sidecar)
    if actual_sidecar != entry.sidecar_sha256:
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar SHA-256 is {actual_sidecar}, the governed "
            f"registry records {entry.sidecar_sha256}.")

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != "v2-learning-checkpoint-v1":
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar format {metadata.get('format')!r} is not "
            f"'v2-learning-checkpoint-v1'.")
    if metadata.get("algorithm") != entry.algorithm:
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar algorithm {metadata.get('algorithm')!r} != "
            f"{entry.algorithm!r}.")
    if int(metadata.get("aggregate_transitions", -1)) != entry.transition:
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar transition "
            f"{metadata.get('aggregate_transitions')} != {entry.transition}.")
    if metadata.get("payload_sha256") != entry.payload_sha256:
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar records a different payload hash than the "
            f"governed registry.")
    run_config = metadata.get("run_config") or {}
    if run_config.get("environment_version") != "v2":
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar run_config is not V2. This product never "
            f"adapts a V1 checkpoint to the V2 environment.")
    if int(run_config.get("root_seed", -1)) != entry.training_root:
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar training root {run_config.get('root_seed')} "
            f"!= {entry.training_root}.")
    source = metadata.get("source") or {}
    if source.get("git_commit") != entry.training_source_sha:
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar training source {source.get('git_commit')!r} "
            f"!= the registered {entry.training_source_sha!r}.")
    if source != (metadata.get("environment_record") or {}).get("source"):
        raise CheckpointUnavailable(
            f"{entry.id}: the sidecar's two source-identity records disagree.")

    from mplssim.experiments.v2_factory import (
        MetadataMismatchError, validate_environment_metadata,
    )
    stored = (metadata.get("environment_record") or {}).get("environment")
    if not isinstance(stored, dict):
        raise CheckpointUnavailable(
            f"{entry.id}: sidecar carries no V2 environment identity record.")
    try:
        validate_environment_metadata(stored)
    except MetadataMismatchError as exc:
        raise CheckpointUnavailable(f"{entry.id}: {exc}") from exc
    return metadata


def availability(entry: CheckpointEntry) -> tuple[bool, str | None]:
    """`(available, reason)` without raising. Used by the capability catalog."""
    try:
        verify(entry)
    except CheckpointUnavailable as exc:
        return (False, str(exc))
    except Exception as exc:  # defensive: a broken artifact is unavailable
        return (False, f"{entry.id}: {type(exc).__name__}: {exc}")
    return (True, None)


def registry_rows() -> list[dict[str, Any]]:
    """Every registry row plus its on-disk availability, for the catalog."""
    rows = []
    for entry in REGISTRY:
        available, reason = availability(entry)
        rows.append({**entry.as_dict(), "available": available,
                     "unavailable_reason": reason})
    return rows


# ------------------------------------------------------------------- loading
class LoadedCheckpoint:
    """A verified frozen learner, bound for *inference only*.

    Holds no optimizer step, writes nothing, and is never fine-tuned. Wrapping
    the two learners here keeps one `predict`/`action_scores` shape so the live
    runner does not branch on algorithm identity at every decision.
    """

    def __init__(self, entry: CheckpointEntry, policy: Any,
                 metadata: dict[str, Any]) -> None:
        self.entry = entry
        self.policy = policy
        self.metadata = metadata

    @property
    def algorithm(self) -> str:
        return self.entry.algorithm

    #: Only MaskablePPO exposes a distribution over actions. The bandit's head
    #: values are immediate-reward estimates and must never be called
    #: probabilities or confidence.
    @property
    def output_semantics(self) -> str:
        return "probabilities" if self.algorithm == "maskable_ppo" else "scores"

    def predict(self, observation: Any, mask: Any) -> int:
        import numpy as np
        predicted = self.policy.predict(
            np.asarray(observation, dtype=np.float32)[None, :],
            np.asarray(mask, dtype=bool)[None, :],
            deterministic=True)
        return int(np.asarray(predicted).reshape(-1)[0])

    def action_scores(self, observation: Any, mask: Any):
        """Per-action outputs in this learner's own semantics, or ``None``.

        MaskablePPO returns masked action probabilities read from the policy
        distribution. The bandit returns its predicted immediate reward per
        action. Masked-out actions are ``None`` in both cases.
        """
        import numpy as np
        import torch

        valid = np.asarray(mask, dtype=bool)
        obs = np.asarray(observation, dtype=np.float32)[None, :]
        if self.algorithm == "maskable_ppo":
            model = self.policy.model
            obs_t, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                distribution = model.policy.get_distribution(
                    obs_t, action_masks=valid.reshape(1, -1))
                values = distribution.distribution.probs.cpu().numpy()[0]
        else:
            network = self.policy.network
            network.eval()
            with torch.no_grad():
                values = network(
                    torch.as_tensor(obs, device=self.policy.device)
                ).cpu().numpy()[0]
        return [float(v) if valid[i] else None for i, v in enumerate(values)]

    def provenance(self) -> dict[str, Any]:
        return {
            **self.entry.as_dict(),
            "output_semantics": self.output_semantics,
            "environment_version": "v2",
            "created_utc": self.metadata.get("created_utc"),
            "inference_only": True,
            "writes_evidence": False,
        }


@functools.lru_cache(maxsize=8)
def _load_cached(algorithm: str, training_root: int, device: str) -> LoadedCheckpoint:
    entry = entry_for(algorithm, training_root)
    metadata = verify(entry)
    payload = payload_path(entry)
    import torch
    torch_device = torch.device(device)
    if algorithm == "masked_bandit":
        from mplssim.experiments.masked_bandit import MaskedContextualBandit
        policy = MaskedContextualBandit.load(payload, device=torch_device)
    else:
        from mplssim.experiments.trainers_v2 import MaskablePpoLearner
        policy = MaskablePpoLearner.load(payload, device=torch_device)
    return LoadedCheckpoint(entry, policy, metadata)


def load(algorithm: str, training_root: int = DEFAULT_ROOT,
         device: str = "cpu") -> LoadedCheckpoint:
    """Verify then load one frozen V2 learner for live inference.

    Cached across sessions: the same verified checkpoint is reused rather than
    re-read, and a failed verification is never cached as a success.
    """
    if algorithm not in LEARNER_ALGORITHMS:
        raise CheckpointUnavailable(
            f"{algorithm!r} is not a V2 learner. Registered learners: "
            f"{', '.join(LEARNER_ALGORITHMS)}.")
    return _load_cached(algorithm, int(training_root), device)


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "CheckpointEntry",
    "CheckpointUnavailable",
    "DEFAULT_POLICY",
    "DEFAULT_ROOT",
    "DEFAULT_ROOT_RULE",
    "LEARNER_ALGORITHMS",
    "LoadedCheckpoint",
    "REGISTRY",
    "TRAINING_ROOTS",
    "artifact_root",
    "availability",
    "entry_for",
    "load",
    "payload_path",
    "registry_rows",
    "sidecar_path",
    "verify",
]
