"""M1.6 gate: caller-declared model context in provenance (ADR-022)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch

from beyondnn._testing.models import TinyTransformer
from beyondnn.provenance import collect_environment, fingerprint_model, make_provenance
from beyondnn.schema import (
    DecodeError,
    ExecutionContext,
    ExecutionMode,
    IntegrityError,
    JsonMap,
    MethodIdentity,
    ModelDeclaration,
    ProvenanceRecord,
    SchemaError,
    SchemaTypeError,
    UnsupportedVersionError,
    from_dict,
    from_json,
    to_dict,
    to_json,
)
from beyondnn.schema.codec import _expected_id

HOOK = MethodIdentity(name="forward_hook", version="1")
CLEAN = ExecutionContext(mode=ExecutionMode.CLEAN, device="cpu", training=False, grad_enabled=True)
ENV = collect_environment()


def _prov(model: Any, declared: ModelDeclaration | None = None) -> ProvenanceRecord:
    return make_provenance(
        model, method=HOOK, execution=CLEAN, environment=ENV, declared_model=declared
    )


def _config(**kw: Any) -> ModelDeclaration:
    return ModelDeclaration(config=JsonMap(kw))


# ----------------------------------------------------------------- the M1.3 regression


def test_declared_config_separates_models_the_fingerprint_cannot() -> None:
    a, b = TinyTransformer(n_heads=2, seed=0), TinyTransformer(n_heads=4, seed=0)
    ia, ib = fingerprint_model(a), fingerprint_model(b)
    assert ia == ib  # FULL v1 limitation, unchanged
    assert _prov(ia).id == _prov(ib).id  # without a declaration they are indistinguishable
    pa, pb = _prov(ia, _config(n_heads=2)), _prov(ib, _config(n_heads=4))
    assert pa.id != pb.id  # the gate: declared context separates them
    assert pa.model == pb.model  # the automatic identity is not altered by declarations


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"implementation_revision": "git:abc123"}, {"implementation_revision": "git:def456"}),
        (
            {"checkpoint_revision": "checkpoint:epoch-12"},
            {"checkpoint_revision": "checkpoint:epoch-13"},
        ),
        (
            {"config": JsonMap({"n_layers": 2})},
            {"config": JsonMap({"n_layers": 2}), "checkpoint_revision": "hf:org/m@r1"},
        ),
    ],
)
def test_each_declared_field_contributes_to_provenance(
    first: dict[str, Any], second: dict[str, Any]
) -> None:
    identity = fingerprint_model(TinyTransformer())
    assert (
        _prov(identity, ModelDeclaration(**first)).id
        != _prov(identity, ModelDeclaration(**second)).id
    )


def test_absent_declaration_leaves_provenance_unchanged() -> None:
    identity = fingerprint_model(TinyTransformer())
    implicit = make_provenance(identity, method=HOOK, execution=CLEAN, environment=ENV)
    explicit_none = _prov(identity, None)
    assert implicit.id == explicit_none.id
    assert implicit.declared_model is None
    assert _prov(identity, _config(n_heads=2)).id != implicit.id


def test_declarations_are_inspectable_as_declared() -> None:
    prov = _prov(TinyTransformer(n_heads=2), _config(n_heads=2, variant="tiny"))
    assert prov.declared_model is not None
    assert prov.declared_model.config["n_heads"] == 2
    env = to_dict(prov)
    assert env["data"]["declared_model"]["config"] == {"n_heads": 2, "variant": "tiny"}
    assert "declared_model" not in env["data"]["model"]  # kept apart from the measured identity


def test_nothing_is_inferred_from_the_model() -> None:
    model = TinyTransformer(n_heads=4)
    prov = _prov(model)
    assert prov.declared_model is None  # n_heads is never scraped from attributes


# ----------------------------------------------------------------- validation and canonicality


def test_declaration_validation() -> None:
    with pytest.raises(SchemaError, match="must declare"):
        ModelDeclaration()
    for bad in ("", "has space", " git:abc"):
        with pytest.raises(SchemaError):
            ModelDeclaration(implementation_revision=bad)
        with pytest.raises(SchemaError):
            ModelDeclaration(checkpoint_revision=bad)
    with pytest.raises(SchemaTypeError):
        ModelDeclaration(config={"layer": torch.nn.Linear(2, 2)})  # type: ignore[arg-type]
    with pytest.raises(SchemaTypeError):
        ModelDeclaration(config={"lr": float("nan")})  # type: ignore[arg-type]


def test_declarations_are_canonical_and_immutable() -> None:
    a = ModelDeclaration(config=JsonMap({"b": 1, "a": [1, 2]}))
    b = ModelDeclaration(config=JsonMap({"a": [1, 2], "b": 1}))
    identity = fingerprint_model(TinyTransformer())
    assert a == b
    assert _prov(identity, a).id == _prov(identity, b).id
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.implementation_revision = "git:x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        a.config["b"] = 2  # type: ignore[index]


# ----------------------------------------------------------------- versioning


def test_provenance_record_version_is_2() -> None:
    assert ProvenanceRecord.RECORD_VERSION == 2
    assert to_dict(_prov(TinyTransformer()))["record_version"] == 2


def test_round_trip_with_declaration() -> None:
    prov = _prov(
        TinyTransformer(),
        ModelDeclaration(
            config=JsonMap({"n_heads": 2}),
            implementation_revision="git:abc123",
            checkpoint_revision="custom:v4",
        ),
    )
    assert from_json(to_json(prov)) == prov


def test_version_1_payload_migrates_to_version_2() -> None:
    current = _prov(TinyTransformer())
    v1 = to_dict(current)
    v1["record_version"] = 1
    del v1["data"]["declared_model"]
    v1["id"] = _expected_id("provenance", 1, v1["data"])  # a valid v1 id
    migrated = from_dict(v1)
    assert migrated.id != v1["id"]  # ids change on migration (ADR-016)
    assert isinstance(migrated, ProvenanceRecord)
    assert migrated.declared_model is None
    assert migrated == current
    assert migrated.RECORD_VERSION == 2


def test_version_1_payload_with_declaration_is_rejected() -> None:
    v1 = to_dict(_prov(TinyTransformer()))
    v1["record_version"] = 1
    v1["id"] = _expected_id("provenance", 1, v1["data"])
    with pytest.raises(DecodeError, match="migrating provenance from record_version 1"):
        from_dict(v1)


def test_future_provenance_version_is_rejected() -> None:
    v3 = to_dict(_prov(TinyTransformer()))
    v3["record_version"] = 3
    with pytest.raises(UnsupportedVersionError):
        from_dict(v3)


def test_tampered_id_on_an_old_version_payload_is_detected() -> None:
    v1 = to_dict(_prov(TinyTransformer()))
    v1["record_version"] = 1
    del v1["data"]["declared_model"]
    v1["id"] = "provenance:" + "0" * 32  # not the v1 id of this content
    with pytest.raises(IntegrityError):
        from_dict(v1)
