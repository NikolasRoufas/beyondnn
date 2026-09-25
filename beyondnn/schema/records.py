"""Measurement records of a single execution (schema 0.1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

from ._types import require
from .base import BaseRecord, record_kind, register_migration
from .status import EvidenceStatus
from .values import NamedTensor, Site, TensorRef

__all__ = ["ActivationRecord", "InputRecord", "OutputRecord"]

_SAMPLE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _unique_paths(tensors: tuple[NamedTensor, ...], where: str) -> None:
    paths = [t.path for t in tensors]
    require(len(set(paths)) == len(paths), f"{where}.tensors has duplicate paths")


def _check_pass(pass_index: int | None, where: str) -> None:
    require(pass_index is None or pass_index >= 0, f"{where}.pass_index must be >= 0 or None")


@record_kind("input", version=3)
@dataclass(frozen=True, slots=True, kw_only=True)
class InputRecord(BaseRecord):
    """The root model's input for one root invocation (pass). Status: OBSERVED.

    ``tensors`` are the tensor leaves of the positional/keyword inputs, in traversal
    order; ``display`` is an optional human rendering (e.g. decoded text).
    ``pass_index`` identifies the root invocation; ``None`` only for records
    migrated from record_version 1, which had no pass concept (never invented).
    ``sample_id`` is the exact input identity (``beyondnn.core.samples``; ADR-031):
    ``None`` if the input could not be identified or the record predates version 3.
    """

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.OBSERVED

    tensors: tuple[NamedTensor, ...] = ()
    display: str | None = None
    pass_index: int | None = None
    sample_id: str | None = None

    def _validate(self) -> None:
        _unique_paths(self.tensors, "InputRecord")
        _check_pass(self.pass_index, "InputRecord")
        if self.sample_id is not None:
            require(
                bool(_SAMPLE_RE.match(self.sample_id)),
                "InputRecord.sample_id must be 'sha256:<64 hex>'",
            )


@record_kind("output", version=2)
@dataclass(frozen=True, slots=True, kw_only=True)
class OutputRecord(BaseRecord):
    """The root model's output for one root invocation (pass). Status: OBSERVED."""

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.OBSERVED

    tensors: tuple[NamedTensor, ...] = ()
    pass_index: int | None = None

    def _validate(self) -> None:
        _unique_paths(self.tensors, "OutputRecord")
        _check_pass(self.pass_index, "OutputRecord")


@record_kind("activation")
@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationRecord(BaseRecord):
    """Model state directly observed at a site. Status: MEASURED.

    The status is MEASURED whether the execution was unmodified or intervened;
    the execution context (e.g. which intervention was active) is recorded in
    provenance, not in the status (ADR-017). ``call_index`` is the n-th call of the
    module within one pass (shared modules); ``pass_index`` is the n-th top-level
    forward pass within one recording. A measurement establishes the value only,
    not its meaning, importance, or causal role.
    """

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.MEASURED

    site: Site
    value: TensorRef
    call_index: int = 0
    pass_index: int = 0

    def _validate(self) -> None:
        require(self.call_index >= 0, "ActivationRecord.call_index must be >= 0")
        require(self.pass_index >= 0, "ActivationRecord.pass_index must be >= 0")


def _add_unknown_pass(data: dict[str, Any]) -> dict[str, Any]:
    """record_version 1 had no pass concept: migrate as ``pass_index = None`` (unknown)."""
    if "pass_index" in data:
        raise ValueError("a record_version 1 payload cannot contain pass_index")
    return data | {"pass_index": None}


register_migration("input", 1)(_add_unknown_pass)
register_migration("output", 1)(_add_unknown_pass)


@register_migration("input", 2)
def _input_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """v2 inputs had no exact identity: migrate as ``sample_id = None`` (never invented)."""
    if "sample_id" in data:
        raise ValueError("an input v2 payload cannot contain sample_id")
    return data | {"sample_id": None}
