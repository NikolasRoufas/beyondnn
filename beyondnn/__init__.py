"""BeyondNN: an interpretability evidence framework for PyTorch.

Implemented: the trace schema (:mod:`beyondnn.schema`), provenance
(:mod:`beyondnn.provenance`), trace recording (:func:`trace`,
:func:`recording`, :class:`TraceResult`), persistence (``TraceResult.save``,
:func:`load_trace`), and :func:`instrument` (``handle.explain`` returns
INPUT -> WHY -> OUTPUT, where Phase-1 WHY is measured internal evidence, not a
causal or attributed explanation).

``import beyondnn`` does not import torch; the tracing names are loaded lazily on
first use, so :mod:`beyondnn.schema` stays usable without torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import schema
from .schema import EstimandScope, EvidenceStatus, Outcome, Relation, Verdict

if TYPE_CHECKING:
    from .core.persistence import load_trace
    from .core.trace import TraceResult, recording, trace
    from .explain import instrument
    from .interventions import intervene

__version__ = "0.0.0.dev0"

__all__ = [
    "EstimandScope",
    "EvidenceStatus",
    "Outcome",
    "Relation",
    "TraceResult",
    "Verdict",
    "__version__",
    "instrument",
    "intervene",
    "interventions",
    "load_trace",
    "recording",
    "schema",
    "trace",
]

_LAZY = {
    "trace": "beyondnn.core.trace",
    "recording": "beyondnn.core.trace",
    "TraceResult": "beyondnn.core.trace",
    "load_trace": "beyondnn.core.persistence",
    "instrument": "beyondnn.explain",
    "intervene": "beyondnn.interventions",
}


def __getattr__(name: str) -> Any:
    if name == "interventions":
        import importlib

        module = importlib.import_module("beyondnn.interventions")
        globals()[name] = module
        return module
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'beyondnn' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
