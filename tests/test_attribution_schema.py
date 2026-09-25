"""Phase 3 schema: AttributionMethodSpec, AttributionRecord, AttributionReduction (ADR-030)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from beyondnn.schema import (
    AttributionBaseline,
    AttributionMethodSpec,
    AttributionRecord,
    AttributionReduction,
    BaselineKind,
    EvidenceStatus,
    InputRecord,
    JsonMap,
    MetricDeclaration,
    MetricSpec,
    OutputRecord,
    RecordRef,
    SchemaError,
    Site,
    SiteIO,
    TensorRef,
    from_json,
    to_json,
)

DIGEST = "sha256:" + "c" * 64
VALUE = TensorRef(
    shape=(1, 2), dtype="float32", device="cpu", storage_key=DIGEST, content_digest=DIGEST
)
PROV = "prov:x"
IG = AttributionMethodSpec(
    name="integrated_gradients",
    implementation="beyondnn",
    implementation_version="1",
    params=JsonMap({"n_steps": 4, "rule": "riemann_middle"}),
)
GRAD = AttributionMethodSpec(name="gradient", implementation="beyondnn", implementation_version="1")
SEL = MetricSpec(name="select", builtin=True, params=JsonMap({"index": [0, 0]}))
INPUT_SITE = Site(module="", io=SiteIO.INPUT, output_path="args[0]")


def _lineage(kind: str = "input") -> tuple[RecordRef, ...]:
    out = RecordRef.to(OutputRecord(pass_index=0, provenance_id=PROV))
    if kind == "input":
        return (out, RecordRef.to(InputRecord(pass_index=0, provenance_id=PROV)))
    return (out,)


def _record(**kw: Any) -> AttributionRecord:
    base: dict[str, Any] = {
        "method": GRAD,
        "target": SEL,
        "site": INPUT_SITE,
        "pass_index": 0,
        "sample_id": "sha256:x",
        "value": VALUE,
        "target_value": 1.0,
        "provenance_id": PROV,
        "derived_from": _lineage(),
    }
    return AttributionRecord(**(base | kw))


def _ig(**kw: Any) -> AttributionRecord:
    diag = {"baseline_target_value": 0.25, "attribution_sum": 0.5, "completeness_delta": -0.25}
    return _record(
        method=IG,
        baseline=AttributionBaseline(kind=BaselineKind.ZERO),
        diagnostics=JsonMap(diag),
        **kw,
    )


def test_status_is_attributed_by_kind_and_cannot_be_supplied() -> None:
    assert _record().status is EvidenceStatus.ATTRIBUTED
    assert AttributionRecord.STATUS is EvidenceStatus.ATTRIBUTED
    with pytest.raises(TypeError):
        _record(status=EvidenceStatus.MEASURED)
    assert from_json(to_json(_ig())) == _ig()


def test_method_spec_validation() -> None:
    with pytest.raises(SchemaError, match="unknown attribution method"):
        AttributionMethodSpec(name="shap", implementation="beyondnn", implementation_version="1")
    with pytest.raises(SchemaError, match="n_steps"):
        dataclasses.replace(IG, params=JsonMap({"rule": "riemann_middle"}))
    with pytest.raises(SchemaError, match="not a beyondnn rule"):
        dataclasses.replace(IG, params=JsonMap({"n_steps": 4, "rule": "gausslegendre"}))
    with pytest.raises(SchemaError, match="integration"):
        dataclasses.replace(GRAD, params=JsonMap({"n_steps": 4}))
    captum = dataclasses.replace(
        IG, implementation="captum", params=JsonMap({"n_steps": 4, "rule": "gausslegendre"})
    )
    assert captum.implementation == "captum"


def test_ig_requires_baseline_and_consistent_completeness_diagnostics() -> None:
    _ig()
    with pytest.raises(SchemaError, match="baseline"):
        _record(method=IG, diagnostics=_ig().diagnostics)
    with pytest.raises(SchemaError, match="diagnostics"):
        _record(method=IG, baseline=AttributionBaseline(kind=BaselineKind.ZERO))
    wrong = {"baseline_target_value": 0.25, "attribution_sum": 0.5, "completeness_delta": 0.0}
    with pytest.raises(SchemaError, match="completeness_delta"):
        _record(
            method=IG,
            baseline=AttributionBaseline(kind=BaselineKind.ZERO),
            diagnostics=JsonMap(wrong),
        )
    with pytest.raises(SchemaError, match="no baseline"):
        _record(baseline=AttributionBaseline(kind=BaselineKind.ZERO))


def test_sites_targets_and_lineage() -> None:
    with pytest.raises(SchemaError, match="positional input leaf"):
        _record(site=Site(module="", io=SiteIO.OUTPUT))
    with pytest.raises(SchemaError, match="OUTPUTs only"):
        _record(site=Site(module="a", io=SiteIO.INPUT), derived_from=_lineage("activation"))
    with pytest.raises(SchemaError, match="exactly the reference output"):
        _record(derived_from=_lineage("none"))
    custom = MetricSpec(
        name="custom:m", builtin=False, declaration=MetricDeclaration(implementation_revision="1")
    )
    with pytest.raises(SchemaError, match="built-in"):
        _record(target=custom)
    with pytest.raises(SchemaError, match="retained"):
        _record(value=TensorRef(shape=(1, 2), dtype="float32", device="cpu"))
    with pytest.raises(SchemaError, match="INPUT_TENSOR"):
        _ig().__class__(
            **{
                f.name: getattr(_ig(), f.name)
                for f in dataclasses.fields(AttributionRecord)
                if f.init and f.name != "baseline"
            },
            baseline=AttributionBaseline(
                kind=BaselineKind.INPUT_TENSOR, value=VALUE, input_path="args[0]"
            ),
        )


def test_baseline_validation() -> None:
    with pytest.raises(SchemaError, match="no tensor"):
        AttributionBaseline(kind=BaselineKind.ZERO, value=VALUE)
    with pytest.raises(SchemaError, match="needs its tensor"):
        AttributionBaseline(kind=BaselineKind.TENSOR)
    with pytest.raises(SchemaError, match="names the positional input"):
        AttributionBaseline(kind=BaselineKind.INPUT_TENSOR, value=VALUE)
    with pytest.raises(SchemaError, match="only INPUT_TENSOR"):
        AttributionBaseline(kind=BaselineKind.TENSOR, value=VALUE, input_path="args[0]")


def test_reduction_validation() -> None:
    parent = (RecordRef.to(_record()),)
    scalar = TensorRef(
        shape=(), dtype="float64", device="cpu", storage_key=DIGEST, content_digest=DIGEST
    )
    ok = AttributionReduction(
        reduction="sum",
        dims=(0, 1),
        value=scalar,
        scalar=1.0,
        provenance_id=PROV,
        derived_from=parent,
    )
    assert ok.status is EvidenceStatus.ATTRIBUTED
    with pytest.raises(SchemaError, match="unknown reduction"):
        dataclasses.replace(ok, reduction="mean")
    with pytest.raises(SchemaError, match="sorted"):
        dataclasses.replace(ok, dims=(1, 0))
    with pytest.raises(SchemaError, match="scalar is set iff"):
        dataclasses.replace(ok, scalar=None)
    with pytest.raises(SchemaError, match="exactly one attribution"):
        dataclasses.replace(ok, derived_from=())
