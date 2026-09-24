"""Deterministic, content-derived record identity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from beyondnn.schema import (
    JsonMap,
    RecordRef,
    Site,
    Subject,
    TargetSpec,
    TensorStats,
    to_dict,
)
from beyondnn.schema._canonical import canonical_json

ID_RE = re.compile(r"^[a-z][a-z0-9_]*:[0-9a-f]{32}$")

# Golden ids pin the identity algorithm. If one of these changes, identity changed:
# that is a schema-breaking change and needs an ADR, not a test update.
GOLDEN_ACTIVATION_ID = "activation:bf06f02771845673d43895ed844a18fd"
GOLDEN_CLAIM_ID = "claim:5924afd11aa7363a3a2780a43b6169a7"


def test_ids_are_well_formed_and_prefixed_by_kind(mk: SimpleNamespace) -> None:
    for rec in (mk.input(), mk.output(), mk.activation(), mk.claim(), mk.spec()):
        assert ID_RE.match(rec.id)
        assert rec.id.split(":")[0] == rec.KIND


def test_identical_content_gives_identical_ids(mk: SimpleNamespace) -> None:
    assert mk.activation().id == mk.activation().id
    assert mk.claim().id == mk.claim().id


def test_id_is_sha256_of_canonical_kind_version_and_data(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    envelope = to_dict(rec)
    payload = {"kind": "activation", "record_version": 1, "data": envelope["data"]}
    expected = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:32]
    assert rec.id == f"activation:{expected}"


def test_schema_version_does_not_contribute_to_identity(mk: SimpleNamespace) -> None:
    rec = mk.activation()
    data = to_dict(rec)["data"]
    payload = {"kind": "activation", "record_version": 1, "data": data}
    assert "schema_version" not in canonical_json(payload)


def test_golden_ids(mk: SimpleNamespace) -> None:
    assert mk.activation().id == GOLDEN_ACTIVATION_ID
    assert mk.claim().id == GOLDEN_CLAIM_ID


def test_ids_are_stable_across_processes_and_hash_seeds(mk: SimpleNamespace) -> None:
    code = (
        "from beyondnn.schema import *\n"
        "c = Claim(statement='s', relation=Relation.ENCODES,"
        " subject=Subject(site=Site(module='m'), units=(3, 1)),"
        " target=TargetSpec(metric='logit', params={'b': 1, 'a': [1, 2.5]}),"
        " estimand=Estimand.instance('x0'), source=ClaimSource(kind=ClaimSourceKind.USER))\n"
        "print(c.id)"
    )
    root = Path(__file__).resolve().parents[1]
    ids = set()
    for seed in ("0", "1", "12345"):
        env = os.environ | {"PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
            env=env,
        )
        ids.add(out.stdout.strip())
    assert len(ids) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"call_index": 1},
        {"pass_index": 2},
        {"module": "layers.9"},
        {"prov": "prov:run-2"},
    ],
)
def test_every_field_contributes_to_identity(
    mk: SimpleNamespace, change: dict[str, object]
) -> None:
    kwargs: dict[str, object] = {}
    module = str(change.pop("module", "layers.0"))
    if "prov" in change:
        kwargs["prov"] = change.pop("prov")
    assert mk.activation(module, **kwargs, **change).id != mk.activation().id


def test_provenance_change_changes_identity(mk: SimpleNamespace) -> None:
    assert mk.input(prov="prov:a").id != mk.input(prov="prov:b").id


def test_lineage_contributes_to_identity(mk: SimpleNamespace) -> None:
    parent = RecordRef.to(mk.input())
    assert mk.activation(parents=(parent,)).id != mk.activation().id


def test_tensor_metadata_contributes_but_not_tensor_bytes(mk: SimpleNamespace) -> None:
    import dataclasses

    rec = mk.activation()
    stats = TensorStats(numel=6, mean=0.51, std=0.25, min=-1.0, max=2.0, l2_norm=3.5)
    changed = dataclasses.replace(rec, value=dataclasses.replace(rec.value, stats=stats))
    assert changed.id != rec.id
    digest = "sha256:" + "1" * 64
    with_digest = dataclasses.replace(
        rec, value=dataclasses.replace(rec.value, content_digest=digest)
    )
    assert with_digest.id != rec.id


def test_json_key_order_does_not_matter(mk: SimpleNamespace) -> None:
    a = TargetSpec(metric="logit", params=JsonMap({"a": 1, "b": 2}))
    b = TargetSpec(metric="logit", params=JsonMap({"b": 2, "a": 1}))
    assert a == b
    assert mk.claim().id == mk.claim().id


def test_json_types_are_distinguished(mk: SimpleNamespace) -> None:
    ids = {mk.spec(criteria={"k": v}).id for v in (1, 1.0, True, "1")}
    assert len(ids) == 4


def test_int_and_float_typed_fields_normalise_before_hashing() -> None:
    a = TensorStats(numel=1, mean=0, std=0, min=0, max=0, l2_norm=0)
    b = TensorStats(numel=1, mean=0.0, std=0.0, min=0.0, max=0.0, l2_norm=0.0)
    assert a == b


def test_canonical_unit_order_gives_same_identity(mk: SimpleNamespace) -> None:
    site = Site(module="layers.1")
    assert Subject(site=site, units=(3, 1, 2)) == Subject(site=site, units=(1, 2, 3))


def test_id_does_not_participate_in_equality_or_hash(mk: SimpleNamespace) -> None:
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(mk.activation())}
    assert fields["id"].compare is False
    assert fields["id"].init is False


def test_canonical_json_is_compact_sorted_and_utf8() -> None:
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json({"x": float("nan")})
    assert json.loads(canonical_json({"f": 0.1})) == {"f": 0.1}


def test_negative_zero_does_not_change_identity(mk: SimpleNamespace) -> None:
    # -0.0 == 0.0, so equal records must also have equal ids.
    def with_mean(mean: float) -> str:
        rec = mk.activation()
        stats = TensorStats(numel=6, mean=mean, std=0.25, min=-1.0, max=2.0, l2_norm=3.5)
        value = dataclasses.replace(rec.value, stats=stats)
        return str(dataclasses.replace(rec, value=value).id)

    assert with_mean(-0.0) == with_mean(0.0)
    assert mk.spec(criteria={"k": -0.0}).id == mk.spec(criteria={"k": 0.0}).id


def test_formally_equivalent_claims_can_have_different_ids(mk: SimpleNamespace) -> None:
    # Record identity is not semantic equivalence (ADR-016): later phases must
    # compare the formal structure, never ids, to decide equivalence.
    a = mk.claim(statement="unit 4 is necessary for class 2")
    b = mk.claim(statement="class 2 needs unit 4")
    assert a.id != b.id
    formal = ("subject", "relation", "target", "estimand")
    assert all(getattr(a, f) == getattr(b, f) for f in formal)
