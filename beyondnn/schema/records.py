"""Measurement records of a single execution (schema 0.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._types import require
from .base import BaseRecord, record_kind
from .status import EvidenceStatus
from .values import NamedTensor, Site, TensorRef

__all__ = ["ActivationRecord", "InputRecord", "OutputRecord"]


def _unique_paths(tensors: tuple[NamedTensor, ...], where: str) -> None:
    paths = [t.path for t in tensors]
    require(len(set(paths)) == len(paths), f"{where}.tensors has duplicate paths")


@record_kind("input")
@dataclass(frozen=True, slots=True, kw_only=True)
class InputRecord(BaseRecord):
    """The model's input for one execution. Status: OBSERVED.

    ``tensors`` are the tensor leaves of the positional/keyword inputs, in pytree
    order; ``display`` is an optional human rendering (e.g. decoded text).
    """

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.OBSERVED

    tensors: tuple[NamedTensor, ...] = ()
    display: str | None = None

    def _validate(self) -> None:
        _unique_paths(self.tensors, "InputRecord")


@record_kind("output")
@dataclass(frozen=True, slots=True, kw_only=True)
class OutputRecord(BaseRecord):
    """The model's output for one execution. Status: OBSERVED."""

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.OBSERVED

    tensors: tuple[NamedTensor, ...] = ()

    def _validate(self) -> None:
        _unique_paths(self.tensors, "OutputRecord")


@record_kind("activation")
@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationRecord(BaseRecord):
    """A value read at a site during an unmodified forward pass. Status: MEASURED.

    ``call_index`` is the n-th call of the module within one pass (shared modules);
    ``pass_index`` is the n-th top-level forward pass within one recording.
    A measurement establishes the value only, not its meaning or importance.
    """

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.MEASURED

    site: Site
    value: TensorRef
    call_index: int = 0
    pass_index: int = 0

    def _validate(self) -> None:
        require(self.call_index >= 0, "ActivationRecord.call_index must be >= 0")
        require(self.pass_index >= 0, "ActivationRecord.pass_index must be >= 0")
