"""Provenance schema: HOW evidence was produced (never what it means).

Pure data, no ``torch`` import. Collection from live models and processes lives
in :mod:`beyondnn.provenance`.

Identity (ADR-019):

* ``ProvenanceRecord.id`` is content-derived (ADR-016) from the model identity,
  environment identity, execution context and method identity only. It names
  a set of reproducible *conditions*; two executions under identical conditions
  share it.
* When something ran (a wall-clock timestamp) is an *occurrence*:
  :class:`ExecutionOccurrence` references the provenance and carries the
  timestamp, so timestamps never change ``provenance_id``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .base import BaseRecord, record_kind, register_migration
from .errors import SchemaError

__all__ = [
    "EnvironmentIdentity",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionOccurrence",
    "FingerprintMethod",
    "MethodIdentity",
    "ModelDeclaration",
    "ModelIdentity",
    "ProvenanceRecord",
    "Randomness",
]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^\S+$")
_DEVICE_RE = re.compile(r"^[a-z][a-z0-9_]*(:[0-9]+)?$")
_METHOD_RE = re.compile(r"^[a-z][a-z0-9_]*(:[A-Za-z0-9_.\-]+)?$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def _token(value: str, where: str) -> None:
    require(bool(_TOKEN_RE.match(value)), f"{where} must be a non-empty token: {value!r}")


def _digest(value: str, where: str) -> None:
    require(bool(_DIGEST_RE.match(value)), f"{where} must be 'sha256:<64 hex>': {value!r}")


class FingerprintMethod(Enum):
    """How a model identity was computed. Only FULL exists; no sampled/partial modes."""

    FULL = "full"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelIdentity(Value):
    """The model state that produced evidence (see ``docs/design`` §Model fingerprint).

    ``structure_digest`` covers module classes and paths, shared-module and tied-
    parameter alias groups, tensor names, roles, dtypes and shapes.
    ``state_digest`` covers the raw bytes of every parameter and persistent
    buffer. Device placement and ``requires_grad`` are not part of model identity.
    """

    model_class: str
    method: FingerprintMethod
    algorithm_version: int
    structure_digest: str
    state_digest: str
    parameter_tensors: int
    parameter_elements: int
    buffer_tensors: int
    buffer_elements: int

    def _validate(self) -> None:
        _token(self.model_class, "ModelIdentity.model_class")
        require(self.algorithm_version >= 1, "ModelIdentity.algorithm_version must be >= 1")
        _digest(self.structure_digest, "ModelIdentity.structure_digest")
        _digest(self.state_digest, "ModelIdentity.state_digest")
        for name in (
            "parameter_tensors",
            "parameter_elements",
            "buffer_tensors",
            "buffer_elements",
        ):
            require(getattr(self, name) >= 0, f"ModelIdentity.{name} must be >= 0")


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentIdentity(Value):
    """Software/platform conditions. Deliberately excludes machine and user identity
    (hostname, username, paths, MAC addresses)."""

    python_implementation: str
    python_version: str
    torch_version: str
    beyondnn_version: str
    platform_system: str
    platform_machine: str

    def _validate(self) -> None:
        for name in (
            "python_implementation",
            "python_version",
            "torch_version",
            "beyondnn_version",
            "platform_system",
            "platform_machine",
        ):
            _token(getattr(self, name), f"EnvironmentIdentity.{name}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Randomness(Value):
    """What is known about randomness. A declared seed and a captured RNG state are
    different facts: a seed does not reproduce a generator that has since advanced.

    ``declared_seed``: the seed the caller says was set (not verified).
    ``rng_state_digest``: SHA-256 of the captured generator state, named by
    ``rng_generator`` (e.g. ``"torch_cpu_default"``). The raw state is not stored.
    """

    declared_seed: int | None = None
    rng_generator: str | None = None
    rng_state_digest: str | None = None

    def _validate(self) -> None:
        require(
            self.declared_seed is not None or self.rng_state_digest is not None,
            "Randomness must declare a seed or a captured RNG state (else use None)",
        )
        require(
            (self.rng_generator is None) == (self.rng_state_digest is None),
            "rng_generator and rng_state_digest must be given together",
        )
        if self.rng_state_digest is not None:
            _digest(self.rng_state_digest, "Randomness.rng_state_digest")
        if self.rng_generator is not None:
            _token(self.rng_generator, "Randomness.rng_generator")


class ExecutionMode(Enum):
    """Whether the model ran unmodified or with an intervention active (ADR-017).

    State measured in either mode is MEASURED; the mode is execution context.
    """

    CLEAN = "clean"
    INTERVENTION = "intervention"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionContext(Value):
    """The conditions of an execution: mode, device, train/eval, grad mode, randomness.

    Invariant: ``intervention_id`` is absent for CLEAN and present for INTERVENTION.
    """

    mode: ExecutionMode
    device: str
    training: bool
    grad_enabled: bool
    intervention_id: str | None = None
    randomness: Randomness | None = None

    def _validate(self) -> None:
        require(bool(_DEVICE_RE.match(self.device)), f"invalid device {self.device!r}")
        if self.mode is ExecutionMode.CLEAN:
            require(self.intervention_id is None, "CLEAN execution cannot have an intervention_id")
        else:
            require(
                self.intervention_id is not None,
                "INTERVENTION execution requires an intervention_id",
            )
        if self.intervention_id is not None:
            _token(self.intervention_id, "ExecutionContext.intervention_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class MethodIdentity(Value):
    """The method that produced evidence, e.g. ``forward_hook`` or
    ``captum:IntegratedGradients``, with its version and canonical parameters."""

    name: str
    version: str
    params: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(bool(_METHOD_RE.match(self.name)), f"invalid method name {self.name!r}")
        _token(self.version, "MethodIdentity.version")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelDeclaration(Value):
    """Model context DECLARED by the caller: recorded as stated, never verified,
    never inferred, and never merged into the automatic :class:`ModelIdentity`.

    Use it for what the FULL fingerprint cannot see, e.g. plain hyperparameters
    (``{"n_heads": 2}``), or the implementation / checkpoint revision
    (``"git:abc123"``, ``"hf:org/model@rev"``, ``"checkpoint:epoch-12"``).
    ``config`` keys are not prescribed. At least one field must be given.
    """

    config: JsonMap = EMPTY_JSON
    implementation_revision: str | None = None
    checkpoint_revision: str | None = None

    def _validate(self) -> None:
        require(
            len(self.config) > 0
            or self.implementation_revision is not None
            or self.checkpoint_revision is not None,
            "ModelDeclaration must declare a config or a revision (else use None)",
        )
        for name in ("implementation_revision", "checkpoint_revision"):
            value = getattr(self, name)
            if value is not None:
                _token(value, f"ModelDeclaration.{name}")


@record_kind("provenance", version=2)
@dataclass(frozen=True, slots=True, kw_only=True)
class ProvenanceRecord(BaseRecord):
    """The reproducible conditions under which evidence was produced.

    Its id is what evidence records put in ``provenance_id``. Every field
    contributes to that id; nothing time- or machine-identity-dependent is stored.
    A provenance record has no provenance of its own and no lineage.

    ``model`` is measured automatically; ``declared_model`` is what the caller
    declared (``None`` if nothing was declared). Record version 2 added
    ``declared_model`` (ADR-022); version-1 payloads migrate with ``None``.
    """

    model: ModelIdentity
    environment: EnvironmentIdentity
    execution: ExecutionContext
    method: MethodIdentity
    declared_model: ModelDeclaration | None = None

    def _validate(self) -> None:
        require(self.provenance_id is None, "a ProvenanceRecord has no provenance_id")
        require(self.derived_from == (), "a ProvenanceRecord has no lineage")


@record_kind("execution_occurrence", version=2)
@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionOccurrence(BaseRecord):
    """One root invocation (pass) under a provenance: observational metadata.

    ``started_at`` is UTC ISO-8601 with a ``Z`` suffix; ``pass_index`` is the root
    pass it belongs to (record_version 2), so two passes starting within the same
    clock resolution are still distinct occurrences. ``pass_index`` is ``None`` only
    for records migrated from record_version 1, where the pass is unknown. Neither
    field is ever part of ``provenance_id``.
    """

    REQUIRES_PROVENANCE: ClassVar[bool] = True

    started_at: str
    pass_index: int | None = None

    def _validate(self) -> None:
        require(
            self.pass_index is None or self.pass_index >= 0,
            "ExecutionOccurrence.pass_index must be >= 0 or None",
        )
        if not _UTC_RE.match(self.started_at):
            raise SchemaError(f"started_at must be UTC ISO-8601 ending in 'Z': {self.started_at!r}")
        try:
            datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaError(f"started_at is not a valid timestamp: {exc}") from None


@register_migration("provenance", 1)
def _provenance_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v1 had no caller declaration: migrate it as ``declared_model = None``.

    The migrated record's id differs from the v1 id (record_version is part of
    identity, ADR-016); references to it must be remapped by the caller (M1.7).
    """
    if "declared_model" in data:
        raise ValueError("a provenance v1 payload cannot contain declared_model")
    return data | {"declared_model": None}


@register_migration("execution_occurrence", 1)
def _occurrence_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v1 occurrences had no pass: migrate as ``pass_index = None`` (unknown, not invented)."""
    if "pass_index" in data:
        raise ValueError("an execution_occurrence v1 payload cannot contain pass_index")
    return data | {"pass_index": None}
