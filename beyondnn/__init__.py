"""BeyondNN: an interpretability evidence framework for PyTorch.

Only the schema foundation (milestone M1.1) exists. Records and serialisation
live in :mod:`beyondnn.schema`; the top level re-exports the status vocabularies.
See ``docs/roadmap/PHASE_1_PLAN.md`` for what comes next.
"""

from . import schema
from .schema import EstimandScope, EvidenceStatus, Outcome, Relation, Verdict

__version__ = "0.0.0.dev0"

__all__ = [
    "EstimandScope",
    "EvidenceStatus",
    "Outcome",
    "Relation",
    "Verdict",
    "__version__",
    "schema",
]
