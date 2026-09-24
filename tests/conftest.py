"""Shared builders for schema tests.

Tests use the ``mk`` fixture (a namespace of builder functions) because
``--import-mode=importlib`` does not allow importing helper modules from tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from beyondnn.schema import (
    ActivationRecord,
    ClaimSource,
    ClaimSourceKind,
    Estimand,
    EvidenceRef,
    EvidenceStatus,
    InputRecord,
    JsonMap,
    NamedTensor,
    OutputRecord,
    Relation,
    Site,
    Subject,
    TargetSpec,
    TensorRef,
    TensorStats,
)
from beyondnn.schema import base as schema_base
from beyondnn.schema.base import BaseRecord, record_kind
from beyondnn.schema.claims import Claim, ClaimTestSpec

PROV = "prov:run-1"

# A syntactically valid id of a kind that does not exist in schema 0.1. Evidence
# references are container-verified; locally only their shape is checked.
FAKE_EFFECT_ID = "causal_effect:" + "a" * 32


def _tref(shape: tuple[int, ...] = (2, 3), *, stats: bool = True) -> TensorRef:
    numel = 1
    for d in shape:
        numel *= d
    return TensorRef(
        shape=shape,
        dtype="float32",
        device="cpu",
        stats=TensorStats(numel=numel, mean=0.5, std=0.25, min=-1.0, max=2.0, l2_norm=3.5)
        if stats
        else None,
    )


def _input(prov: str = PROV, display: str | None = None) -> InputRecord:
    return InputRecord(
        tensors=(NamedTensor(path="args[0]", ref=_tref()),), display=display, provenance_id=prov
    )


def _output(prov: str = PROV) -> OutputRecord:
    return OutputRecord(tensors=(NamedTensor(path="out", ref=_tref((2,))),), provenance_id=prov)


def _activation(
    module: str = "layers.0", *, prov: str = PROV, parents: tuple[Any, ...] = (), **kw: Any
) -> ActivationRecord:
    return ActivationRecord(
        site=Site(module=module),
        value=_tref(),
        provenance_id=prov,
        derived_from=parents,
        **kw,
    )


def _claim(
    relation: Relation = Relation.NECESSARY_FOR,
    estimand: Estimand | None = None,
    statement: str = "layers.1 unit 4 is necessary for class 2",
) -> Claim:
    return Claim(
        statement=statement,
        relation=relation,
        subject=Subject(site=Site(module="layers.1"), units=(4,)),
        target=TargetSpec(metric="logit", params=JsonMap({"class": 2})),
        estimand=estimand or Estimand.instance("x0"),
        source=ClaimSource(kind=ClaimSourceKind.USER),
    )


def _spec(
    protocol: str = "ablation_necessity",
    relations: tuple[Relation, ...] = (Relation.NECESSARY_FOR,),
    criteria: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> ClaimTestSpec:
    return ClaimTestSpec(
        protocol=protocol,
        protocol_version=1,
        applicable_relations=relations,
        criteria=JsonMap(criteria if criteria is not None else {"direction": "decrease"}),
        params=JsonMap(params or {"op": "mean_ablation"}),
    )


def _causal_ref(
    estimand: Estimand | None = None,
    status: EvidenceStatus = EvidenceStatus.INTERVENTIONAL,
    record_id: str = FAKE_EFFECT_ID,
) -> EvidenceRef:
    return EvidenceRef(
        record_id=record_id,
        kind="causal_effect",
        status=status,
        estimand=estimand or Estimand.instance("x0"),
    )


@pytest.fixture
def mk() -> SimpleNamespace:
    return SimpleNamespace(
        tref=_tref,
        input=_input,
        output=_output,
        activation=_activation,
        claim=_claim,
        spec=_spec,
        causal_ref=_causal_ref,
        PROV=PROV,
        FAKE_EFFECT_ID=FAKE_EFFECT_ID,
    )


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """Register test-only record kinds without leaking them into other tests."""
    monkeypatch.setattr(schema_base, "_KINDS", dict(schema_base._KINDS))
    monkeypatch.setattr(schema_base, "_MIGRATIONS", dict(schema_base._MIGRATIONS))
    return record_kind


@pytest.fixture
def fake_effect_kind(scratch_registry: Callable[..., Any]) -> type[BaseRecord]:
    """A test-only causal-effect kind whose status is derived from its estimand.

    Mirrors the ADR-013 rule: exact effects on an instance or finite sample are
    INTERVENTIONAL; population quantities are ESTIMATED_CAUSAL.
    """

    @scratch_registry("fake_effect")
    @dataclass(frozen=True, slots=True, kw_only=True)
    class FakeEffect(BaseRecord):
        STATUS: ClassVar[EvidenceStatus | None] = None
        estimand: Estimand
        effect: float

        @property
        def status(self) -> EvidenceStatus | None:
            from beyondnn.schema import EstimandScope

            if self.estimand.scope is EstimandScope.POPULATION:
                return EvidenceStatus.ESTIMATED_CAUSAL
            return EvidenceStatus.INTERVENTIONAL

    return FakeEffect
