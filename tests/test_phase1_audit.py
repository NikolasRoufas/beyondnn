"""Audit guards (M1.10, updated for Phases 2 and 3): which evidence statuses can exist."""

from __future__ import annotations

import re
from pathlib import Path

import beyondnn
from beyondnn.schema import EvidenceStatus
from beyondnn.schema.base import BaseRecord, registered_kinds

ROOT = Path(__file__).resolve().parents[1] / "beyondnn"
STRONGER = ("ATTRIBUTED", "INTERVENTIONAL", "ESTIMATED_CAUSAL", "VALIDATED_CONCEPT", "GENERATED")


def test_only_causal_effects_carry_causal_status_and_nothing_stronger_exists() -> None:
    attributed = {"attribution", "attribution_reduction"}
    for kind, cls in registered_kinds().items():
        if kind in attributed:
            assert cls.STATUS is EvidenceStatus.ATTRIBUTED, kind
        else:
            assert cls.STATUS in (EvidenceStatus.OBSERVED, EvidenceStatus.MEASURED, None), kind
        if kind == "causal_effect":
            assert cls.status is not BaseRecord.status  # derived from estimand/estimator
        else:
            assert cls.status is BaseRecord.status, kind


def test_execution_layers_never_name_a_stronger_status() -> None:
    pattern = re.compile(r"EvidenceStatus\.(" + "|".join(STRONGER) + r")\b")
    for package in ("core", "provenance", "_testing"):
        for path in (ROOT / package).rglob("*.py"):
            assert not pattern.search(path.read_text()), path


EVIDENCE_KINDS = (
    "InputRecord",
    "OutputRecord",
    "ActivationRecord",
    "AttributionRecord",
    "AttributionReduction",
    "CausalEffect",
    "InterventionRecord",
    "ProvenanceRecord",
    "ClaimTestResult",
    "Claim",
)


def test_the_presentation_layer_never_constructs_evidence() -> None:
    # Phase 4 (ADR-031): explain/ groups records by status (so it may name statuses),
    # but it only reads records; it never builds evidence, claims, or test results.
    constructor = re.compile(r"(?<![\w.])(" + "|".join(EVIDENCE_KINDS) + r")\(")
    forged = re.compile(r"\.(for_claim|to)\(|_add\(|_add_tensor\(")
    for path in (ROOT / "explain").rglob("*.py"):
        text = path.read_text()
        assert not constructor.search(text), (path, constructor.search(text))
        assert not forged.search(text), (path, forged.search(text))


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
            "attribute",
            "attribution",
            "compose",
            "faithfulness",
            "instrument",
            "intervene",
            "interventions",
            "load_trace",
            "recording",
            "schema",
            "trace",
        ]
    )
    for name in beyondnn.__all__:
        assert getattr(beyondnn, name) is not None
