"""Explanation interface: ``instrument()``, evidence composition, and the
INPUT -> [TARGET] -> WHY -> OUTPUT view.

Measured-only (Phase 1): WHY answers "what internal evidence was measured while
this output was produced?". With composed evidence (Phase 4, :func:`compose`), WHY
presents OBSERVED, MEASURED, ATTRIBUTED and INTERVENTIONAL evidence, declared claims
and their tests and assessments in separate sections, without combining them, and
states what was not evaluated (faithfulness, concepts).
"""

from .bundle import (
    CompositionError,
    EvidenceBundle,
    EvidenceIntegrityError,
    ModelMismatchError,
    SampleMismatchError,
    TargetMismatchError,
    UnsupportedScopeError,
)
from .handle import Instrumented, instrument
from .response import EXPLANATION_LIMITATIONS, ExplainResponse, Why, compose
from .views import AttributionView, ClaimView, Coverage, InterventionView

__all__ = [
    "EXPLANATION_LIMITATIONS",
    "AttributionView",
    "ClaimView",
    "CompositionError",
    "Coverage",
    "EvidenceBundle",
    "EvidenceIntegrityError",
    "ExplainResponse",
    "Instrumented",
    "InterventionView",
    "ModelMismatchError",
    "SampleMismatchError",
    "TargetMismatchError",
    "UnsupportedScopeError",
    "Why",
    "compose",
    "instrument",
]
