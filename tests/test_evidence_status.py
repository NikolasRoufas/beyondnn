"""Epistemic status: not a hierarchy, explicit derivation rules, no upgrades."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import cast

import pytest

from beyondnn.schema import (
    ALLOWED_PARENT_STATUSES,
    CAUSAL_EVIDENCE_STATUSES,
    Estimand,
    EvidenceRef,
    EvidenceRuleError,
    EvidenceStatus,
    OutputRecord,
    RecordRef,
    SchemaError,
)
from beyondnn.schema.status import check_derivation

S = EvidenceStatus


def _ref(status: S | None, n: int = 0) -> RecordRef:
    return RecordRef(record_id=f"thing:{n:032x}", kind="thing", status=status)


def test_status_values_are_exactly_the_agreed_set() -> None:
    assert {s.value for s in S} == {
        "observed",
        "measured",
        "attributed",
        "interventional",
        "estimated_causal",
        "validated_concept",
        "generated",
    }


@pytest.mark.parametrize(("a", "b"), list(itertools.permutations(S, 2))[:12])
def test_statuses_are_not_ordered(a: S, b: S) -> None:
    with pytest.raises(TypeError):
        _ = a < b  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = a >= b  # type: ignore[operator]


def test_statuses_are_not_strings() -> None:
    assert cast(object, S.MEASURED) != "measured"
    assert not isinstance(S.MEASURED, str)


def test_derivation_table_covers_every_status() -> None:
    assert set(ALLOWED_PARENT_STATUSES) == set(S)


def test_only_generated_may_derive_from_generated() -> None:
    for child in S:
        if child is S.GENERATED:
            check_derivation(child, [_ref(S.GENERATED)])
        else:
            with pytest.raises(EvidenceRuleError, match="cannot derive from generated"):
                check_derivation(child, [_ref(S.GENERATED)])


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (S.OBSERVED, S.MEASURED),
        (S.MEASURED, S.ATTRIBUTED),
        (S.MEASURED, S.INTERVENTIONAL),
        (S.INTERVENTIONAL, S.ATTRIBUTED),
        (S.INTERVENTIONAL, S.ESTIMATED_CAUSAL),
        (S.ATTRIBUTED, S.INTERVENTIONAL),
        (S.VALIDATED_CONCEPT, S.ATTRIBUTED),
    ],
)
def test_forbidden_derivations(child: S, parent: S) -> None:
    with pytest.raises(EvidenceRuleError):
        check_derivation(child, [_ref(parent)])


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (S.MEASURED, S.OBSERVED),
        (S.ATTRIBUTED, S.MEASURED),
        (S.INTERVENTIONAL, S.MEASURED),
        (S.ESTIMATED_CAUSAL, S.INTERVENTIONAL),
        (S.ESTIMATED_CAUSAL, S.ATTRIBUTED),
        (S.GENERATED, S.ESTIMATED_CAUSAL),
    ],
)
def test_allowed_derivations(child: S, parent: S) -> None:
    check_derivation(child, [_ref(parent)])


def test_evidence_cannot_derive_from_non_evidence_records() -> None:
    with pytest.raises(EvidenceRuleError, match="non-evidence"):
        check_derivation(S.MEASURED, [_ref(None)])


def test_non_evidence_records_may_derive_from_anything() -> None:
    check_derivation(None, [_ref(s, i) for i, s in enumerate(S)] + [_ref(None, 99)])


def test_records_enforce_derivation_rules(mk: SimpleNamespace) -> None:
    mk.activation(parents=(RecordRef.to(mk.input()),))
    measured = RecordRef.to(mk.activation())
    with pytest.raises(EvidenceRuleError):
        OutputRecord(provenance_id=mk.PROV, derived_from=(measured,))
    with pytest.raises(EvidenceRuleError, match="cannot derive from generated"):
        mk.activation(parents=(_ref(S.GENERATED),))


def test_generated_claims_may_exist_but_are_not_evidence(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    # Claims are hypotheses; they may be derived from generated text...
    import dataclasses

    dataclasses.replace(claim, derived_from=(_ref(S.GENERATED),))
    # ...but a claim is not an evidence record and cannot be cited as evidence.
    with pytest.raises(EvidenceRuleError, match="not an evidence record"):
        EvidenceRef.to(claim)


def test_duplicate_lineage_is_rejected(mk: SimpleNamespace) -> None:
    parent = RecordRef.to(mk.input())
    with pytest.raises(SchemaError, match="duplicate"):
        mk.activation(parents=(parent, parent))


# ----------------------------------------------------------------- generated evidence


def test_generated_content_cannot_be_cited_as_evidence() -> None:
    with pytest.raises(EvidenceRuleError, match="GENERATED content"):
        EvidenceRef(record_id="summary:" + "0" * 32, kind="summary", status=S.GENERATED)


def test_evidence_refs_are_built_from_actual_records(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    ref = EvidenceRef.to(rec)
    assert (ref.record_id, ref.kind, ref.status, ref.estimand) == (
        rec.id,
        "activation",
        S.MEASURED,
        None,
    )


def test_causal_evidence_must_state_its_estimand() -> None:
    for status in CAUSAL_EVIDENCE_STATUSES:
        with pytest.raises(EvidenceRuleError, match="estimand"):
            EvidenceRef(record_id="causal_effect:" + "0" * 32, kind="causal_effect", status=status)


def test_non_causal_evidence_carries_no_estimand(mk: SimpleNamespace) -> None:
    with pytest.raises(EvidenceRuleError, match="only causal evidence"):
        EvidenceRef(
            record_id=mk.activation().id,
            kind="activation",
            status=S.MEASURED,
            estimand=Estimand.instance("x0"),
        )


@pytest.mark.parametrize(
    "record_id", ["activation:xyz", "activation:" + "A" * 32, "input:" + "0" * 32, "noid"]
)
def test_evidence_refs_validate_id_shape_and_kind(record_id: str) -> None:
    with pytest.raises(SchemaError):
        EvidenceRef(record_id=record_id, kind="activation", status=S.MEASURED)


def test_record_refs_carry_the_parent_status(mk: SimpleNamespace) -> None:
    ref = RecordRef.to(mk.input())
    assert ref.status is S.OBSERVED
    assert ref.kind == "input"
