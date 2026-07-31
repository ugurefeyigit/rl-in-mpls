"""Read-only access to the frozen governed V2 evidence.

The V2 study is complete and closed. This package never trains, never tunes, never
loads a checkpoint for evaluation, never reselects a checkpoint, and never writes
into `results/` or `runs/`. It reads committed compact artifacts, validates them
against the frozen study identity, and fails closed.

Final-holdout evidence and development/continuity evidence are separate types on
purpose: nothing here offers a way to average them together.
"""

from mplssim.evidence import identity
from mplssim.evidence.errors import (
    ArtifactMissingError, EvidenceError, IdentityError, IntegrityError, SchemaError,
)

__all__ = [
    "identity",
    "EvidenceError",
    "ArtifactMissingError",
    "SchemaError",
    "IdentityError",
    "IntegrityError",
]
