"""Collect provenance from the running process: environment, RNG, records."""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone

import torch
from torch import nn

from beyondnn.schema import (
    EnvironmentIdentity,
    ExecutionContext,
    ExecutionOccurrence,
    MethodIdentity,
    ModelIdentity,
    ProvenanceRecord,
    Randomness,
)

from .fingerprint import fingerprint_model, tensor_bytes

__all__ = ["capture_cpu_rng", "collect_environment", "make_provenance", "record_occurrence"]


def _or_unknown(value: str) -> str:
    return value.replace(" ", "_") if value.strip() else "unknown"


def collect_environment() -> EnvironmentIdentity:
    """Software and platform identity of this process.

    Collects only: Python implementation and version, torch and BeyondNN versions,
    OS family (``platform.system()``) and CPU architecture (``platform.machine()``).
    Never collects hostname, username, paths, or hardware identifiers.
    """
    from beyondnn import __version__

    return EnvironmentIdentity(
        python_implementation=_or_unknown(platform.python_implementation()),
        python_version=_or_unknown(platform.python_version()),
        torch_version=_or_unknown(str(torch.__version__)),
        beyondnn_version=__version__,
        platform_system=_or_unknown(platform.system()),
        platform_machine=_or_unknown(platform.machine()),
    )


def capture_cpu_rng(*, declared_seed: int | None = None) -> Randomness:
    """Capture a digest of torch's default CPU generator state, read-only.

    ``torch.get_rng_state()`` returns a copy, so the generator is neither
    advanced nor reset. Only the SHA-256 of the state is kept. ``declared_seed``
    is recorded separately and is never inferred from the state.
    """
    state = torch.get_rng_state()
    state_digest = "sha256:" + hashlib.sha256(tensor_bytes(state)).hexdigest()
    return Randomness(
        declared_seed=declared_seed,
        rng_generator="torch_cpu_default",
        rng_state_digest=state_digest,
    )


def make_provenance(
    model: nn.Module | ModelIdentity,
    *,
    method: MethodIdentity,
    execution: ExecutionContext,
    environment: EnvironmentIdentity | None = None,
) -> ProvenanceRecord:
    """Build the provenance record for evidence produced by ``method`` on ``model``.

    ``model`` may be a module (fingerprinted now, in full) or a precomputed
    :class:`~beyondnn.schema.ModelIdentity`. The environment is collected from this
    process unless given. Contains no timestamp: see :func:`record_occurrence`.
    """
    identity = model if isinstance(model, ModelIdentity) else fingerprint_model(model)
    return ProvenanceRecord(
        model=identity,
        environment=environment if environment is not None else collect_environment(),
        execution=execution,
        method=method,
    )


def record_occurrence(
    provenance: ProvenanceRecord, *, started_at: datetime | None = None
) -> ExecutionOccurrence:
    """Record that an execution under ``provenance`` happened at ``started_at`` (UTC now)."""
    when = started_at if started_at is not None else datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return ExecutionOccurrence(provenance_id=provenance.id, started_at=stamp)
