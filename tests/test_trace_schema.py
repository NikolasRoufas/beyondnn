"""Record mechanism: immutability, slots, construction rules, runtime types, registry."""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from beyondnn.schema import (
    ActivationRecord,
    EvidenceStatus,
    InputRecord,
    JsonMap,
    NamedTensor,
    OutputRecord,
    SchemaError,
    SchemaTypeError,
    Site,
    SiteIO,
    Subject,
    TargetSpec,
    TensorRef,
    TensorStats,
)
from beyondnn.schema.base import BaseRecord, registered_kinds

# ----------------------------------------------------------------- immutability


def test_records_are_frozen(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.call_index = 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.id = "activation:" + "0" * 32


def test_records_use_slots(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    assert not hasattr(rec, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(rec, "note", "x")


def test_values_are_frozen_and_slotted(mk: SimpleNamespace) -> None:
    ref = mk.tref()
    assert not hasattr(ref, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.dtype = "float64"


def test_json_maps_are_deeply_immutable() -> None:
    m = JsonMap({"a": [1, {"b": 2}]})
    assert m["a"] == (1, JsonMap({"b": 2}))
    with pytest.raises(TypeError):
        m["a"] = 3  # type: ignore[index]
    assert isinstance(m["a"], tuple)
    plain = m.to_plain()
    plain["a"].append(9)
    assert m["a"] == (1, JsonMap({"b": 2}))


def test_json_maps_cannot_be_reinitialised(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    params = claim.target.params
    with pytest.raises(TypeError, match="immutable"):
        params.__init__({"class": 99})
    assert params == {"class": 2}


def test_records_are_keyword_only(mk: SimpleNamespace) -> None:
    with pytest.raises(TypeError):
        cast(Any, ActivationRecord)(Site(module="a"), mk.tref())


def test_status_is_not_a_constructor_argument(mk: SimpleNamespace) -> None:
    with pytest.raises(TypeError):
        ActivationRecord(  # type: ignore[call-arg]
            site=Site(module="a"),
            value=mk.tref(),
            provenance_id=mk.PROV,
            status=EvidenceStatus.INTERVENTIONAL,
        )


def test_status_cannot_be_reassigned(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    # ``status`` is a read-only property; frozen + slots makes assignment fail
    # (with TypeError on some CPython versions).
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError, TypeError)):
        rec.status = EvidenceStatus.INTERVENTIONAL
    with pytest.raises(TypeError):
        dataclasses.replace(rec, status=EvidenceStatus.INTERVENTIONAL)
    assert rec.status is EvidenceStatus.MEASURED


def test_replace_recomputes_identity(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    other = dataclasses.replace(rec, call_index=1)
    assert other.call_index == 1
    assert other.id != rec.id
    assert dataclasses.replace(rec).id == rec.id


def test_equal_content_means_equal_and_hashable(mk: SimpleNamespace) -> None:
    a, b = mk.activation(), mk.activation()
    assert a == b
    assert len({a, b}) == 1


def test_base_record_and_unregistered_subclasses_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="not a registered"):
        BaseRecord(provenance_id="p")

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Stray(BaseRecord):
        x: int

    with pytest.raises(TypeError, match="not a registered"):
        Stray(x=1)


# ----------------------------------------------------------------- statuses per kind


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (InputRecord, EvidenceStatus.OBSERVED),
        (OutputRecord, EvidenceStatus.OBSERVED),
        (ActivationRecord, EvidenceStatus.MEASURED),
    ],
)
def test_measurement_kinds_have_fixed_status(cls: type[BaseRecord], status: EvidenceStatus) -> None:
    assert cls.STATUS is status


def test_non_evidence_kinds_have_no_status(mk: SimpleNamespace) -> None:
    assert mk.claim().status is None
    assert mk.spec().status is None


def test_evidence_records_require_provenance(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaError, match="requires provenance_id"):
        ActivationRecord(site=Site(module="a"), value=mk.tref())
    with pytest.raises(SchemaError, match="requires provenance_id"):
        InputRecord()


@pytest.mark.parametrize("bad", ["", "has space", "tab\tin"])
def test_provenance_id_must_be_a_token(mk: SimpleNamespace, bad: str) -> None:
    with pytest.raises(SchemaError):
        mk.activation(prov=bad)


def test_claims_do_not_require_provenance(mk: SimpleNamespace) -> None:
    assert mk.claim().provenance_id is None


# ----------------------------------------------------------------- runtime types


def test_tuple_fields_reject_lists(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaTypeError, match=r"TensorRef\.shape"):
        TensorRef(shape=[2, 3], dtype="float32", device="cpu")  # type: ignore[arg-type]


def test_int_fields_reject_bool_and_str(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaTypeError):
        mk.activation(call_index=True)
    with pytest.raises(SchemaTypeError):
        mk.activation(call_index="1")


def test_float_fields_accept_ints_and_normalise() -> None:
    stats = TensorStats(numel=1, mean=0, std=0, min=0, max=0, l2_norm=0)
    assert isinstance(stats.mean, float)
    with pytest.raises(SchemaTypeError):
        TensorStats(numel=1, mean=True, std=0, min=0, max=0, l2_norm=0)


def test_enum_fields_reject_raw_strings() -> None:
    with pytest.raises(SchemaTypeError):
        Site(module="a", io="output")  # type: ignore[arg-type]


def test_nested_value_fields_are_type_checked(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaTypeError):
        ActivationRecord(site="layers.0", value=mk.tref(), provenance_id=mk.PROV)  # type: ignore[arg-type]


def test_json_fields_accept_mappings_and_reject_non_json() -> None:
    assert isinstance(TargetSpec(metric="logit", params={"class": 1}).params, JsonMap)  # type: ignore[arg-type]
    with pytest.raises(SchemaTypeError):
        TargetSpec(metric="logit", params={"x": object()})  # type: ignore[arg-type]
    with pytest.raises(SchemaTypeError):
        TargetSpec(metric="logit", params={"x": math.nan})  # type: ignore[arg-type]
    with pytest.raises(SchemaTypeError):
        TargetSpec(metric="logit", params={1: "x"})  # type: ignore[arg-type]


# ----------------------------------------------------------------- value validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (-1,)},
        {"dtype": "Float32"},
        {"dtype": ""},
        {"device": "cuda:x"},
        {"device": "GPU"},
        {"storage_key": ""},
        {"content_digest": "md5:abc"},
        {"content_digest": "sha256:" + "0" * 63},
    ],
)
def test_tensor_ref_rejects_invalid_metadata(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {"shape": (2,), "dtype": "float32", "device": "cpu"}
    with pytest.raises(SchemaError):
        TensorRef(**(base | kwargs))


def test_tensor_ref_accepts_valid_metadata() -> None:
    ref = TensorRef(
        shape=(0, 4),
        dtype="bfloat16",
        device="cuda:1",
        storage_key="act/1",
        content_digest="sha256:" + "f" * 64,
    )
    assert ref.numel == 0
    assert TensorRef(shape=(), dtype="float32", device="mps").numel == 1


def test_tensor_ref_stats_must_match_shape() -> None:
    stats = TensorStats(numel=5, mean=0, std=0, min=0, max=0, l2_norm=0)
    with pytest.raises(SchemaError, match="does not match shape"):
        TensorRef(shape=(2, 3), dtype="float32", device="cpu", stats=stats)


def test_tensor_stats_validation() -> None:
    ok: dict[str, Any] = {
        "numel": 1,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 1.0,
        "l2_norm": 1.0,
    }
    TensorStats(**(ok | {"mean": math.nan, "std": math.nan, "min": -math.inf}))
    for bad in ({"std": -1.0}, {"l2_norm": -0.5}, {"min": 2.0}, {"numel": -1}):
        with pytest.raises(SchemaError):
            TensorStats(**(ok | bad))


@pytest.mark.parametrize("module", ["", "layers", "layers.0.attn", "blocks.3.mlp-out"])
def test_site_accepts_module_paths(module: str) -> None:
    assert Site(module=module).io is SiteIO.OUTPUT


@pytest.mark.parametrize("module", ["layers..0", ".layers", "layers.", "a b", "a.\tb"])
def test_site_rejects_malformed_module_paths(module: str) -> None:
    with pytest.raises(SchemaError):
        Site(module=module)


@pytest.mark.parametrize("metric", ["Logit", "", "custom:", "custom:a b", "1logit"])
def test_target_spec_rejects_malformed_metrics(metric: str) -> None:
    with pytest.raises(SchemaError):
        TargetSpec(metric=metric)


def test_subject_units_are_canonicalised_and_validated() -> None:
    site = Site(module="layers.1")
    assert Subject(site=site, units=(9, 2, 5)).units == (2, 5, 9)
    for bad in ((), (1, 1), (-1,)):
        with pytest.raises(SchemaError):
            Subject(site=site, units=bad)


def test_measurement_records_reject_duplicate_tensor_paths(mk: SimpleNamespace) -> None:
    t = NamedTensor(path="x", ref=mk.tref())
    with pytest.raises(SchemaError, match="duplicate paths"):
        InputRecord(tensors=(t, t), provenance_id=mk.PROV)
    with pytest.raises(SchemaError):
        NamedTensor(path="", ref=mk.tref())


def test_activation_indices_must_be_non_negative(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaError):
        mk.activation(call_index=-1)
    with pytest.raises(SchemaError):
        mk.activation(pass_index=-1)


# ----------------------------------------------------------------- registry


def test_registered_kinds_are_exactly_schema_0_1() -> None:
    assert set(registered_kinds()) == {
        "input",
        "output",
        "activation",
        "limitation",
        "claim",
        "claim_test_spec",
        "claim_test_result",
        "assessment",
        "provenance",
        "execution_occurrence",
        "intervention",
        "causal_effect",
    }


def test_registry_view_is_read_only() -> None:
    with pytest.raises(TypeError):
        registered_kinds()["x"] = InputRecord  # type: ignore[index]


def test_duplicate_kind_registration_fails(scratch_registry: Any) -> None:
    with pytest.raises(ValueError, match="already registered"):

        @scratch_registry("activation")
        @dataclass(frozen=True, slots=True, kw_only=True)
        class Again(BaseRecord):
            pass


def test_unsupported_annotations_fail_at_registration(scratch_registry: Any) -> None:
    with pytest.raises(TypeError, match="unsupported schema annotation"):

        @scratch_registry("bad_list")
        @dataclass(frozen=True, slots=True, kw_only=True)
        class BadList(BaseRecord):
            items: list[int]


@pytest.mark.parametrize("kind", ["Upper", "1x", "with-dash", ""])
def test_invalid_kind_names_are_rejected(scratch_registry: Any, kind: str) -> None:
    with pytest.raises(ValueError, match="invalid record kind"):

        @scratch_registry(kind)
        @dataclass(frozen=True, slots=True, kw_only=True)
        class K(BaseRecord):
            pass


def test_schema_does_not_import_torch() -> None:
    code = "import sys, beyondnn, beyondnn.schema; print('torch' in sys.modules)"
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root
    )
    assert out.stdout.strip() == "False"
