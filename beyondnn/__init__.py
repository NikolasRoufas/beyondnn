"""BeyondNN: an interpretability evidence framework for PyTorch.

Implemented: the trace schema (:mod:`beyondnn.schema`), provenance
(:mod:`beyondnn.provenance`), and trace recording: :func:`trace`,
:func:`recording`, :class:`TraceResult`.

``import beyondnn`` does not import torch; the tracing names are loaded lazily on
first use, so :mod:`beyondnn.schema` stays usable without torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import schema
from .schema import EstimandScope, EvidenceStatus, Outcome, Relation, Verdict

if TYPE_CHECKING:
    from .core.trace import TraceResult, recording, trace

__version__ = "0.0.0.dev0"

__all__ = [
    "EstimandScope",
    "EvidenceStatus",
    "Outcome",
    "Relation",
    "TraceResult",
    "Verdict",
    "__version__",
    "recording",
    "schema",
    "trace",
]

_LAZY = {
    "trace": "beyondnn.core.trace",
    "recording": "beyondnn.core.trace",
    "TraceResult": "beyondnn.core.trace",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'beyondnn' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
