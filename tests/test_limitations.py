"""The limitation registry and TraceLimitation records."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from beyondnn.schema import (
    LIMITATIONS,
    SchemaError,
    Severity,
    TraceLimitation,
    UnknownLimitationCodeError,
    from_json,
    to_json,
)


def test_registry_contains_only_codes_needed_now() -> None:
    assert set(LIMITATIONS) == {
        "FUNCTIONAL_OPS_UNOBSERVED",
        "PARTIAL_SITE_COVERAGE",
        "SELECTED_SITE_NOT_EXECUTED",
        "NON_TENSOR_LEAVES_IGNORED",
        "ZERO_ABLATION_MAY_BE_OOD",
        "CONSTANT_REPLACEMENT_MAY_BE_OOD",
        "PATCH_SOURCE_CONTEXT_DIFFERS",
        "CUSTOM_METRIC_UNVERIFIED",
        "NO_ATTRIBUTION",
        "NO_CAUSAL_EVIDENCE",
        "NO_CLAIMS_TESTED",
    }


def test_every_code_is_fully_defined_without_duplicate_meanings() -> None:
    meanings = set()
    for code, definition in LIMITATIONS.items():
        assert definition.code == code
        assert code.isupper()
        assert " " not in code
        assert definition.meaning.strip()
        assert definition.emitted_when.strip()
        assert isinstance(definition.severity, Severity)
        meanings.add(definition.meaning)
    assert len(meanings) == len(LIMITATIONS)


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        LIMITATIONS["NEW"] = LIMITATIONS["NO_ATTRIBUTION"]  # type: ignore[index]


def test_unknown_codes_are_rejected() -> None:
    with pytest.raises(UnknownLimitationCodeError):
        TraceLimitation(code="MADE_UP")
    with pytest.raises(UnknownLimitationCodeError):
        TraceLimitation(code="no_attribution")


def test_limitation_scoping_and_properties(mk: SimpleNamespace) -> None:
    a, b = mk.activation("a"), mk.activation("b")
    lim = TraceLimitation(code="PARTIAL_SITE_COVERAGE", applies_to=(b.id, a.id), detail="2 of 9")
    assert lim.applies_to == tuple(sorted((a.id, b.id)))
    assert lim.severity is Severity.WARNING
    assert lim.definition is LIMITATIONS["PARTIAL_SITE_COVERAGE"]
    assert lim.status is None
    assert from_json(to_json(lim)) == lim


def test_limitation_scope_validation(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    with pytest.raises(SchemaError):
        TraceLimitation(code="NO_ATTRIBUTION", applies_to=("not-an-id",))
    with pytest.raises(SchemaError):
        TraceLimitation(code="NO_ATTRIBUTION", applies_to=(rec.id, rec.id))
