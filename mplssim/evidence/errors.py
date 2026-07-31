"""Failure modes of the read-only evidence layer.

Every one is fail-closed: the layer refuses to serve evidence it cannot vouch for
rather than degrading into a plausible-looking partial answer.
"""

from __future__ import annotations


class EvidenceError(Exception):
    """The frozen evidence could not be served as promised."""


class ArtifactMissingError(EvidenceError):
    """A required frozen artifact is absent or unreadable."""


class SchemaError(EvidenceError):
    """An artifact is present but does not carry the columns or fields promised."""


class IdentityError(EvidenceError):
    """An artifact's study identity — SHA, root, seed, scenario — is not the frozen one."""


class IntegrityError(EvidenceError):
    """An artifact's own integrity flags, counts or coverage do not hold."""
