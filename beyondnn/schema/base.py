"""The record mechanism: base record, content-derived identity, lineage, registry.

Identity (see ``docs/design/TRACE_SCHEMA_PROPOSAL.md`` §Record identity)
------------------------------------------------------------------------
``record.id == f"{kind}:{h}"`` where ``h`` is the first 32 hex characters (128 bits)
of SHA-256 over the canonical JSON of::

    {"kind": <kind>, "record_version": <int>, "data": <every init field, encoded>}

* Every init field contributes, including ``provenance_id`` and ``derived_from``.
  The same content measured under different provenance is a different record.
* ``id`` itself and ``schema_version`` (the envelope version) do not contribute.
* Tensor *values* never contribute (they are not in records); tensor metadata in
  :class:`~beyondnn.schema.values.TensorRef` does, including summary statistics
  and the optional ``content_digest``.
* Collisions: 128 bits makes accidental collisions negligible; containers must
  still reject two different records with the same id.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar

from ._canonical import digest
from ._types import Checked, Value, encode_fields, field_specs, finalize_fields, require
from .errors import EvidenceRuleError, SchemaError
from .status import CAUSAL_EVIDENCE_STATUSES, EstimandScope, EvidenceStatus, check_derivation
from .values import Estimand

__all__ = [
    "BaseRecord",
    "EvidenceRef",
    "RecordRef",
    "is_record_id",
    "record_kind",
    "register_migration",
    "registered_kinds",
    "verify_ref",
]

_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ID_RE = re.compile(r"^([a-z][a-z0-9_]*):([0-9a-f]{32})$")
_TOKEN_RE = re.compile(r"^\S+$")

_KINDS: dict[str, type[BaseRecord]] = {}
_MIGRATIONS: dict[tuple[str, int], Callable[[dict[str, Any]], dict[str, Any]]] = {}

R = TypeVar("R", bound="BaseRecord")


def is_record_id(value: str, kind: str | None = None) -> bool:
    """Whether ``value`` is a well-formed record id (optionally of ``kind``)."""
    m = _ID_RE.match(value)
    return m is not None and (kind is None or m.group(1) == kind)


def _require_id(record_id: str, kind: str, where: str) -> None:
    require(is_record_id(record_id), f"{where}: malformed record id {record_id!r}")
    require(
        record_id.split(":", 1)[0] == kind,
        f"{where}: id {record_id!r} does not match kind {kind!r}",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordRef(Value):
    """A lineage reference to another record (used in ``derived_from``).

    Carries the parent's status so derivation rules can be checked locally.
    Containers verify refs against the records they point to (:func:`verify_ref`).
    """

    record_id: str
    kind: str
    status: EvidenceStatus | None

    def _validate(self) -> None:
        _require_id(self.record_id, self.kind, "RecordRef")

    @classmethod
    def to(cls, record: BaseRecord) -> RecordRef:
        return cls(record_id=record.id, kind=record.KIND, status=record.status)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceRef(Value):
    """A reference to a record cited as scientific evidence.

    Rules enforced at construction:

    * ``GENERATED`` records can never be evidence.
    * Causal evidence (``INTERVENTIONAL``/``ESTIMATED_CAUSAL``) must state its estimand.
    * ``INTERVENTIONAL`` evidence can never have ``POPULATION`` scope (ADR-013).
    * Non-causal evidence carries no estimand in schema 0.1.
    """

    record_id: str
    kind: str
    status: EvidenceStatus
    estimand: Estimand | None = None

    def _validate(self) -> None:
        _require_id(self.record_id, self.kind, "EvidenceRef")
        if self.status is EvidenceStatus.GENERATED:
            raise EvidenceRuleError(
                f"{self.record_id} is GENERATED content and cannot be cited as evidence"
            )
        if self.status in CAUSAL_EVIDENCE_STATUSES:
            if self.estimand is None:
                raise EvidenceRuleError(
                    f"causal evidence {self.record_id} must state its estimand (ADR-013)"
                )
            if (
                self.status is EvidenceStatus.INTERVENTIONAL
                and self.estimand.scope is EstimandScope.POPULATION
            ):
                raise EvidenceRuleError(
                    f"{self.record_id}: INTERVENTIONAL evidence cannot have POPULATION scope; "
                    "population quantities are ESTIMATED_CAUSAL (ADR-013)"
                )
        elif self.estimand is not None:
            raise EvidenceRuleError(
                f"{self.record_id}: only causal evidence carries an estimand in schema 0.1"
            )

    @classmethod
    def to(cls, record: BaseRecord) -> EvidenceRef:
        status = record.status
        if status is None:
            raise EvidenceRuleError(f"{record.id} is not an evidence record (it has no status)")
        estimand = getattr(record, "estimand", None)
        return cls(record_id=record.id, kind=record.KIND, status=status, estimand=estimand)


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseRecord(Checked):
    """Base class of every record kind. Instantiate only registered subclasses.

    Class-level declarations (set per kind, never per instance):

    ``KIND``, ``RECORD_VERSION``
        set by :func:`record_kind`.
    ``STATUS``
        the evidence status of every record of this kind, or ``None`` for
        non-evidence records (claims, specs, results, assessments, limitations).
        Kinds whose status depends on content override the ``status`` property.
        Status is never a constructor argument and cannot be reassigned.
    ``REQUIRES_PROVENANCE``
        non-evidence kinds that are produced by execution set this to True.
        Evidence records always require provenance.
    """

    KIND: ClassVar[str] = ""
    RECORD_VERSION: ClassVar[int] = 0
    STATUS: ClassVar[EvidenceStatus | None] = None
    REQUIRES_PROVENANCE: ClassVar[bool] = False

    provenance_id: str | None = None
    derived_from: tuple[RecordRef, ...] = ()
    id: str = field(init=False, compare=False, default="")

    def __post_init__(self) -> None:
        cls = type(self)
        if _KINDS.get(cls.KIND) is not cls:
            raise TypeError(f"{cls.__name__} is not a registered record kind")
        finalize_fields(self)
        self._validate()
        self._check_lineage_and_provenance()
        object.__setattr__(self, "id", _compute_id(self))

    @property
    def status(self) -> EvidenceStatus | None:
        return type(self).STATUS

    @property
    def record_id(self) -> str:
        return self.id

    def _check_lineage_and_provenance(self) -> None:
        name = type(self).__name__
        ids = [ref.record_id for ref in self.derived_from]
        require(len(set(ids)) == len(ids), f"{name}.derived_from contains duplicate ids")
        check_derivation(self.status, self.derived_from)
        if self.provenance_id is None:
            if self.status is not None or type(self).REQUIRES_PROVENANCE:
                raise SchemaError(f"{name} requires provenance_id")
        else:
            require(bool(_TOKEN_RE.match(self.provenance_id)), f"{name}.provenance_id invalid")


def _compute_id(record: BaseRecord) -> str:
    payload = {
        "kind": record.KIND,
        "record_version": record.RECORD_VERSION,
        "data": encode_fields(record),
    }
    return f"{record.KIND}:{digest(payload)[:32]}"


def record_kind(kind: str, *, version: int = 1) -> Callable[[type[R]], type[R]]:
    """Register a record class under ``kind`` with ``record_version`` ``version``.

    Apply *above* ``@dataclass(frozen=True, slots=True, kw_only=True)``. Field
    annotations are compiled immediately, so unsupported types fail at import.
    """

    def decorate(cls: type[R]) -> type[R]:
        if not _KIND_RE.match(kind):
            raise ValueError(f"invalid record kind {kind!r}")
        if version < 1:
            raise ValueError("record_version must be >= 1")
        if not (isinstance(cls, type) and issubclass(cls, BaseRecord) and is_dataclass(cls)):
            raise TypeError("record_kind() requires a dataclass subclass of BaseRecord")
        if kind in _KINDS:
            raise ValueError(f"record kind {kind!r} is already registered")
        cls.KIND = kind
        cls.RECORD_VERSION = version
        field_specs(cls)
        _KINDS[kind] = cls
        return cls

    return decorate


def registered_kinds() -> Mapping[str, type[BaseRecord]]:
    """A read-only view of the kind registry."""
    return MappingProxyType(_KINDS)


def register_migration(
    kind: str, from_version: int
) -> Callable[
    [Callable[[dict[str, Any]], dict[str, Any]]], Callable[[dict[str, Any]], dict[str, Any]]
]:
    """Register a function upgrading ``data`` of ``kind`` from ``from_version`` to +1."""

    def decorate(
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        key = (kind, from_version)
        if key in _MIGRATIONS:
            raise ValueError(f"migration {key} is already registered")
        _MIGRATIONS[key] = fn
        return fn

    return decorate


def verify_ref(ref: Value, record: BaseRecord) -> None:
    """Raise :class:`EvidenceRuleError` unless ``ref`` exactly describes ``record``.

    Works for any ref type with a ``to(record)`` constructor. Containers call this
    to check that references carry the referenced record's true attributes.
    """
    to = getattr(type(ref), "to", None)
    if to is None:
        raise TypeError(f"{type(ref).__name__} is not a reference type")
    expected = to(record)
    if expected != ref:
        raise EvidenceRuleError(f"{type(ref).__name__} does not match record {record.id}")
