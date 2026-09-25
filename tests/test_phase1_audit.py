"""M1.10 audit guards: what Phase 1 can and cannot produce."""

from __future__ import annotations

import re
from pathlib import Path

import beyondnn
from beyondnn.schema import EvidenceStatus
from beyondnn.schema.base import BaseRecord, registered_kinds

ROOT = Path(__file__).resolve().parents[1] / "beyondnn"
STRONGER = ("ATTRIBUTED", "INTERVENTIONAL", "ESTIMATED_CAUSAL", "VALIDATED_CONCEPT", "GENERATED")


def test_no_registered_kind_can_carry_a_stronger_status() -> None:
    for kind, cls in registered_kinds().items():
        assert cls.STATUS in (EvidenceStatus.OBSERVED, EvidenceStatus.MEASURED, None), kind
        # no kind derives its status from content in Phase 1
        assert cls.status is BaseRecord.status, kind


def test_execution_layers_never_name_a_stronger_status() -> None:
    pattern = re.compile(r"EvidenceStatus\.(" + "|".join(STRONGER) + r")\b")
    for package in ("core", "explain", "provenance", "_testing"):
        for path in (ROOT / package).rglob("*.py"):
            assert not pattern.search(path.read_text()), path


def test_top_level_names_are_exactly_the_audited_surface() -> None:
    assert sorted(beyondnn.__all__) == sorted(
        [
            "EstimandScope",
            "EvidenceStatus",
            "Outcome",
            "Relation",
            "TraceResult",
            "Verdict",
            "__version__",
            "instrument",
            "load_trace",
            "recording",
            "schema",
            "trace",
        ]
    )
    for name in beyondnn.__all__:
        assert getattr(beyondnn, name) is not None
