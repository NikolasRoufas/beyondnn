"""Exceptions raised by the schema layer.

Every error is a subclass of :class:`SchemaError`, so callers can catch schema
problems as one family while tests can assert on the specific rule violated.
"""

from __future__ import annotations

__all__ = [
    "DecodeError",
    "EvidenceRuleError",
    "IntegrityError",
    "SchemaError",
    "SchemaTypeError",
    "UnknownLimitationCodeError",
    "UnknownRecordKindError",
    "UnsupportedVersionError",
]


class SchemaError(ValueError):
    """Base class for all schema violations."""


class SchemaTypeError(SchemaError, TypeError):
    """A field value has the wrong runtime type."""


class EvidenceRuleError(SchemaError):
    """An epistemic rule was violated.

    Raised for, among other things: citing generated content as evidence, deriving
    evidence from disallowed parents, supporting a causal claim without causal
    evidence, and assessments whose verdict does not follow from their results.
    """


class UnknownLimitationCodeError(SchemaError):
    """A limitation code is not in the registry."""


class DecodeError(SchemaError):
    """A serialised payload could not be decoded into a valid record."""


class UnknownRecordKindError(DecodeError):
    """The payload names a record kind this version of BeyondNN does not know."""


class UnsupportedVersionError(DecodeError):
    """The payload's schema or record version cannot be read by this version."""


class IntegrityError(DecodeError):
    """The payload's stored id or status does not match its content."""
