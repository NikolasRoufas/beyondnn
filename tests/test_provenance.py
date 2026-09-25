"""Provenance schema and collection: identity, execution context, privacy, time."""

from __future__ import annotations

import dataclasses
import getpass
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from beyondnn.provenance import (
    capture_cpu_rng,
    collect_environment,
    fingerprint_model,
    make_provenance,
    record_occurrence,
)
from beyondnn.schema import (
    ActivationRecord,
    EvidenceStatus,
    ExecutionContext,
    ExecutionMode,
    ExecutionOccurrence,
    JsonMap,
    MethodIdentity,
    ModelIdentity,
    ProvenanceRecord,
    Randomness,
    RecordRef,
    SchemaError,
    Site,
    TensorRef,
    from_json,
    to_json,
)

HOOK = MethodIdentity(name="forward_hook", version="1")


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


def _clean(**kw: Any) -> ExecutionContext:
    base: dict[str, Any] = {
        "mode": ExecutionMode.CLEAN,
        "device": "cpu",
        "training": False,
        "grad_enabled": True,
    }
    return ExecutionContext(**(base | kw))


def _prov(model: nn.Module | None = None, **kw: Any) -> ProvenanceRecord:
    args: dict[str, Any] = {"method": HOOK, "execution": _clean()} | kw
    return make_provenance(model if model is not None else _model(), **args)


# ----------------------------------------------------------------- execution context


def test_clean_execution_cannot_have_an_intervention_id() -> None:
    with pytest.raises(SchemaError, match="CLEAN"):
        _clean(intervention_id="intervention:abc")


def test_intervention_execution_requires_an_intervention_id() -> None:
    with pytest.raises(SchemaError, match="requires an intervention_id"):
        _clean(mode=ExecutionMode.INTERVENTION)
    ctx = _clean(mode=ExecutionMode.INTERVENTION, intervention_id="intervention:abc")
    assert ctx.intervention_id == "intervention:abc"


@pytest.mark.parametrize("device", ["", "CPU", "cuda:x", "gpu 0"])
def test_execution_device_is_validated(device: str) -> None:
    with pytest.raises(SchemaError):
        _clean(device=device)


def test_measured_state_under_intervention_differs_only_in_provenance() -> None:
    # ADR-017: the activation stays MEASURED; the intervention lives in provenance.
    model = _model()
    identity = fingerprint_model(model)
    clean = make_provenance(identity, method=HOOK, execution=_clean())
    intervened = make_provenance(
        identity,
        method=HOOK,
        execution=_clean(mode=ExecutionMode.INTERVENTION, intervention_id="intervention:abl-1"),
    )
    ref = TensorRef(shape=(1, 4), dtype="float32", device="cpu")
    a = ActivationRecord(site=Site(module="0"), value=ref, provenance_id=clean.id)
    b = ActivationRecord(site=Site(module="0"), value=ref, provenance_id=intervened.id)
    assert a.status is b.status is EvidenceStatus.MEASURED
    assert clean.id != intervened.id
    assert a.id != b.id


# ----------------------------------------------------------------- randomness


def test_randomness_distinguishes_declared_seed_from_captured_state() -> None:
    declared = Randomness(declared_seed=7)
    captured = capture_cpu_rng()
    assert declared.rng_state_digest is None
    assert captured.declared_seed is None
    assert captured.rng_generator == "torch_cpu_default"
    both = capture_cpu_rng(declared_seed=7)
    assert both.declared_seed == 7
    assert both.rng_state_digest is not None


def test_randomness_validation() -> None:
    with pytest.raises(SchemaError):
        Randomness()
    with pytest.raises(SchemaError):
        Randomness(rng_state_digest="sha256:" + "0" * 64)  # generator missing
    with pytest.raises(SchemaError):
        Randomness(rng_generator="g", rng_state_digest="md5:1")


def test_rng_capture_is_read_only_and_tracks_state() -> None:
    torch.manual_seed(5)
    first = capture_cpu_rng()
    again = capture_cpu_rng()
    assert first == again  # reading does not advance the generator
    draw = torch.rand(3)
    torch.manual_seed(5)
    assert torch.equal(torch.rand(3), draw)
    assert capture_cpu_rng() != first  # the generator has advanced


def test_raw_rng_state_is_not_stored() -> None:
    text = to_json(_prov(execution=_clean(randomness=capture_cpu_rng())))
    assert len(text) < 3000  # the raw CPU generator state alone is ~5 KB


# ----------------------------------------------------------------- method identity


def test_method_identity_validation_and_canonical_params() -> None:
    a = MethodIdentity(name="captum:IntegratedGradients", version="0.9.0", params={"n": 50, "b": 0})  # type: ignore[arg-type]
    b = MethodIdentity(
        name="captum:IntegratedGradients", version="0.9.0", params=JsonMap({"b": 0, "n": 50})
    )
    assert a == b
    for bad in ({"name": "Forward Hook"}, {"name": ""}, {"version": ""}, {"version": "1 0"}):
        kwargs: dict[str, Any] = {"name": "forward_hook", "version": "1"} | bad
        with pytest.raises(SchemaError):
            MethodIdentity(**kwargs)


# ----------------------------------------------------------------- provenance identity


def test_provenance_id_is_deterministic_and_occurrence_time_does_not_enter_it() -> None:
    p1, p2 = _prov(), _prov()
    assert p1.id == p2.id
    t0 = datetime(2026, 9, 25, 3, 0, tzinfo=timezone.utc)
    o1 = record_occurrence(p1, started_at=t0)
    o2 = record_occurrence(p2, started_at=t0 + timedelta(hours=5))
    assert o1.provenance_id == o2.provenance_id == p1.id
    assert o1.id != o2.id  # occurrence identity is not provenance identity


def test_provenance_changes_with_method_parameters() -> None:
    a = _prov(method=MethodIdentity(name="zero_ablation", version="1", params={"k": 1}))  # type: ignore[arg-type]
    b = _prov(method=MethodIdentity(name="zero_ablation", version="1", params={"k": 2}))  # type: ignore[arg-type]
    assert a.id != b.id


def test_provenance_changes_with_model_state() -> None:
    model = _model()
    before = _prov(model)
    with torch.no_grad():
        model.get_parameter("0.bias")[0] += 1.0
    assert _prov(model).id != before.id


@pytest.mark.parametrize(
    "change",
    [
        {"training": True},
        {"grad_enabled": False},
        {"device": "mps"},
        {"randomness": Randomness(declared_seed=1)},
    ],
)
def test_provenance_changes_with_execution_conditions(change: dict[str, Any]) -> None:
    assert _prov(execution=_clean(**change)).id != _prov().id


def test_provenance_changes_with_environment() -> None:
    env = collect_environment()
    other = dataclasses.replace(env, torch_version="0.0.0")
    assert _prov(environment=env).id != _prov(environment=other).id


def test_provenance_accepts_a_precomputed_identity() -> None:
    model = _model()
    identity = fingerprint_model(model)
    assert make_provenance(identity, method=HOOK, execution=_clean()).id == _prov(model).id


def test_provenance_records_have_no_provenance_or_lineage() -> None:
    p = _prov()
    with pytest.raises(SchemaError):
        dataclasses.replace(p, provenance_id="provenance:" + "0" * 32)
    with pytest.raises(SchemaError):
        dataclasses.replace(p, derived_from=(RecordRef.to(p),))
    assert p.status is None


def test_provenance_round_trips() -> None:
    p = _prov(execution=_clean(randomness=capture_cpu_rng(declared_seed=3)))
    assert from_json(to_json(p)) == p
    o = record_occurrence(p)
    assert from_json(to_json(o)) == o


# ----------------------------------------------------------------- occurrences


@pytest.mark.parametrize(
    "stamp",
    ["2026-09-25T03:00:00", "2026-09-25T03:00:00+03:00", "2026-13-25T03:00:00Z", "yesterday"],
)
def test_occurrence_timestamps_must_be_utc_iso(stamp: str) -> None:
    with pytest.raises(SchemaError):
        ExecutionOccurrence(provenance_id=_prov().id, started_at=stamp)


def test_occurrence_requires_provenance_and_aware_times() -> None:
    with pytest.raises(SchemaError):
        ExecutionOccurrence(started_at="2026-09-25T03:00:00Z")
    with pytest.raises(ValueError, match="timezone-aware"):
        record_occurrence(_prov(), started_at=datetime(2026, 9, 25))


def test_occurrence_normalises_to_utc() -> None:
    local = datetime(2026, 9, 25, 6, 0, tzinfo=timezone(timedelta(hours=3)))
    assert record_occurrence(_prov(), started_at=local).started_at == "2026-09-25T03:00:00.000000Z"


# ----------------------------------------------------------------- immutability


def test_provenance_values_are_immutable() -> None:
    p = _prov()
    for obj, field in (
        (p, "model"),
        (p.model, "state_digest"),
        (p.environment, "torch_version"),
        (p.execution, "mode"),
        (p.method, "name"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, None)
    with pytest.raises(TypeError):
        p.method.params["x"] = 1  # type: ignore[index]


def test_model_identity_validation() -> None:
    ok = fingerprint_model(_model())
    for bad in (
        {"state_digest": "sha256:xyz"},
        {"structure_digest": "abc"},
        {"algorithm_version": 0},
        {"parameter_tensors": -1},
        {"model_class": "has space"},
    ):
        with pytest.raises(SchemaError):
            dataclasses.replace(ok, **bad)
    assert isinstance(ok, ModelIdentity)


# ----------------------------------------------------------------- privacy


def _sensitive_strings() -> list[str]:
    values = {
        getpass.getuser(),
        socket.gethostname(),
        os.path.expanduser("~"),
        os.getcwd(),
        str(Path(__file__).resolve().parent),
    }
    return [v for v in values if v and len(v) >= 3]


def test_collected_provenance_contains_no_user_host_or_path() -> None:
    p = _prov(execution=_clean(randomness=capture_cpu_rng(declared_seed=1)))
    text = to_json(p) + to_json(record_occurrence(p))
    for value in _sensitive_strings():
        assert value not in text, f"provenance leaked {value!r}"


def test_environment_fields_are_exactly_the_documented_set() -> None:
    env = collect_environment()
    assert {f.name for f in dataclasses.fields(env)} == {
        "python_implementation",
        "python_version",
        "torch_version",
        "beyondnn_version",
        "platform_system",
        "platform_machine",
    }
    assert env.torch_version == str(torch.__version__)


def test_occurrences_of_different_passes_never_collide() -> None:
    p = _prov()
    stamp = datetime(2026, 9, 25, 3, 0, tzinfo=timezone.utc)
    a = record_occurrence(p, started_at=stamp, pass_index=0)
    b = record_occurrence(p, started_at=stamp, pass_index=1)
    assert a.provenance_id == b.provenance_id
    assert a.started_at == b.started_at
    assert a.id != b.id
    with pytest.raises(SchemaError):
        record_occurrence(p, pass_index=-1)


def test_occurrence_v1_migrates_with_unknown_pass() -> None:
    from beyondnn.schema import from_dict, to_dict
    from beyondnn.schema.codec import _expected_id

    env = to_dict(record_occurrence(_prov(), pass_index=3))
    env["record_version"] = 1
    del env["data"]["pass_index"]
    env["id"] = _expected_id("execution_occurrence", 1, env["data"])
    migrated = from_dict(env)
    assert isinstance(migrated, ExecutionOccurrence)
    assert migrated.pass_index is None
