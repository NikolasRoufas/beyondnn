"""M1.7: trace persistence, migration with reference remapping, corruption handling."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

import beyondnn as bnn
from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer
from beyondnn.core.persistence import TracePersistenceError
from beyondnn.core.trace import TraceIntegrityError, TraceResult
from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import (
    ActivationRecord,
    InputRecord,
    JsonMap,
    ModelDeclaration,
    ProvenanceRecord,
    TraceLimitation,
    to_dict,
    to_json,
)
from beyondnn.schema.codec import _expected_id


def _tok() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 3, 0, 31], [2, 2, 7, 30, 4, 8]])


def assert_equivalent(a: TraceResult, b: TraceResult) -> None:
    """Scientific equivalence: ids, canonical content, config, tensors (never ``==``)."""
    assert [r.id for r in a.records] == [r.id for r in b.records]
    assert [to_json(r) for r in a.records] == [to_json(r) for r in b.records]
    assert a.config == b.config
    assert a._tensors.keys() == b._tensors.keys()
    for key in a._tensors:  # bitwise (NaN-safe), with dtype and shape
        assert a._tensors[key].dtype == b._tensors[key].dtype
        assert a._tensors[key].shape == b._tensors[key].shape
        assert tensor_bytes(a._tensors[key]) == tensor_bytes(b._tensors[key])


def roundtrip(t: TraceResult, tmp_path: Path, name: str = "trace") -> TraceResult:
    t.save(tmp_path / name)
    return bnn.load_trace(tmp_path / name)


# ----------------------------------------------------------------- round trips


@pytest.mark.parametrize("retention", ["none", "summary", "cpu"])
def test_round_trip_for_every_retention(retention: str, tmp_path: Path) -> None:
    t = bnn.trace(
        TinyTransformer(), _tok(), sites=["**"], input_sites=["blocks.*"], retention=retention
    )
    loaded = roundtrip(t, tmp_path)
    assert_equivalent(t, loaded)
    assert (tmp_path / "trace" / "tensors.pt").exists() == (retention == "cpu")


def test_round_trip_multi_pass_structured_declared_and_limitations(tmp_path: Path) -> None:
    model = TinyCNN().train()
    with bnn.recording(
        model,
        sites=["stem", "classifier", "block"],
        retention="cpu",
        declared_model=ModelDeclaration(config=JsonMap({"width": 4}), checkpoint_revision="ckpt:0"),
    ) as ctx:
        model(torch.ones(2, 1, 8, 8), return_features=True)
        model(torch.zeros(2, 1, 8, 8), return_features=True)
    t = ctx.result
    assert t.passes == 2
    assert len(t.provenance) == 2
    loaded = roundtrip(t, tmp_path)
    assert_equivalent(t, loaded)
    assert loaded.provenance[0].declared_model is not None
    assert [lim.code for lim in loaded.limitations] == [lim.code for lim in t.limitations]
    assert tensor_bytes(loaded.tensor(loaded.activations[0])) == tensor_bytes(
        t.tensor(t.activations[0])
    )


def test_nan_values_survive_round_trip(tmp_path: Path) -> None:
    class Nan(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * float("nan")

    t = bnn.trace(Nan(), torch.ones(2), retention="cpu")
    loaded = roundtrip(t, tmp_path)
    assert_equivalent(t, loaded)  # compared by id and canonical JSON, not ==
    stats = loaded.output.tensors[0].ref.stats
    assert stats is not None
    assert stats.mean != stats.mean  # NaN


# ----------------------------------------------------------------- format and saving


def test_json_is_deterministic_and_has_no_paths(tmp_path: Path) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4), sites=["shared"], retention="cpu")
    t.save(tmp_path / "a")
    t.save(tmp_path / "b")
    text = (tmp_path / "a" / "trace.json").read_text()
    assert text == (tmp_path / "b" / "trace.json").read_text()
    for forbidden in (str(tmp_path), os.path.expanduser("~"), os.getcwd()):
        assert forbidden not in text
    document = json.loads(text)
    assert set(document) == {
        "format",
        "format_version",
        "schema_version",
        "config",
        "records",
        "tensors",
    }
    assert document["tensors"] == sorted(t._tensors)


def test_save_never_overwrites(tmp_path: Path) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4))
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        t.save(tmp_path / "exists")
    with pytest.raises(FileNotFoundError):
        t.save(tmp_path / "missing_parent" / "trace")


def test_failed_save_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4), retention="cpu")

    def broken_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        t.save(tmp_path / "trace")
    assert list(tmp_path.iterdir()) == []  # neither the target nor a partial directory


# ----------------------------------------------------------------- migration and remapping


def _downgrade(document: dict[str, Any], kinds: set[str]) -> dict[str, Any]:
    """Rewrite a current trace document as an old one: records of ``kinds`` go back
    to record_version 1, and every reference and id is rewritten consistently, the
    way an older BeyondNN would have written it."""
    old_ids: dict[str, str] = {}
    out = []
    for env in document["records"]:
        env = json.loads(json.dumps(env))
        text = json.dumps(env["data"])
        for new, old in old_ids.items():
            text = text.replace(new, old)
        env["data"] = json.loads(text)
        if env["kind"] in kinds:
            env["record_version"] = 1
            env["data"].pop("declared_model" if env["kind"] == "provenance" else "pass_index")
        new_id = env["id"]
        env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
        if env["id"] != new_id:
            old_ids[new_id] = env["id"]
        out.append(env)
    return document | {"records": out}


def _write(directory: Path, document: dict[str, Any]) -> None:
    (directory / "trace.json").write_text(json.dumps(document, sort_keys=True, indent=2))


def test_old_provenance_migrates_and_every_reference_is_remapped(tmp_path: Path) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4), sites=["shared"])
    t.save(tmp_path / "trace")
    document = json.loads((tmp_path / "trace" / "trace.json").read_text())
    act = next(e for e in document["records"] if e["kind"] == "activation")
    limitation = to_dict(TraceLimitation(code="NO_ATTRIBUTION", applies_to=(act["id"],)))
    document["records"].append(limitation)
    old = _downgrade(document, {"provenance"})
    (prov_env,) = [e for e in old["records"] if e["kind"] == "provenance"]
    assert prov_env["id"] != t.provenance[0].id  # really an old id
    _write(tmp_path / "trace", old)

    loaded = bnn.load_trace(tmp_path / "trace")
    (prov,) = loaded.provenance
    assert prov.id == t.provenance[0].id  # migrated to v2: the current id again
    assert prov.RECORD_VERSION == 2
    for record in loaded.records:
        if record.provenance_id is not None:
            assert record.provenance_id == prov.id
    assert [r.id for r in loaded.records][: len(t.records)] == [r.id for r in t.records]
    (extra,) = [lim for lim in loaded.limitations if lim.code == "NO_ATTRIBUTION"]
    assert extra.applies_to == (loaded.activations[0].id,)


def test_cascading_id_changes_are_remapped(tmp_path: Path) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4), sites=["shared"])
    t.save(tmp_path / "trace")
    document = json.loads((tmp_path / "trace" / "trace.json").read_text())
    _write(
        tmp_path / "trace",
        _downgrade(document, {"provenance", "input", "output", "execution_occurrence"}),
    )
    loaded = bnn.load_trace(tmp_path / "trace")
    (inp,) = loaded.inputs
    assert inp.pass_index is None  # v1 had no pass concept: unknown, not invented
    assert inp.id != t.input.id
    for record in (*loaded.activations, loaded.output):
        assert record.derived_from[0].record_id == inp.id
    (occurrence,) = loaded.occurrences
    assert occurrence.pass_index is None
    assert occurrence.provenance_id == loaded.provenance[0].id


def test_tampered_old_record_fails_before_migration(tmp_path: Path) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4))
    t.save(tmp_path / "trace")
    document = _downgrade(
        json.loads((tmp_path / "trace" / "trace.json").read_text()), {"provenance"}
    )
    document["records"][0]["data"]["method"]["version"] = "2"  # id left as-is
    _write(tmp_path / "trace", document)
    with pytest.raises(TracePersistenceError, match="does not match content"):
        bnn.load_trace(tmp_path / "trace")


# ----------------------------------------------------------------- corruption


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4), sites=["shared", "head"], retention="cpu")
    t.save(tmp_path / "trace")
    return tmp_path / "trace"


def _edit(directory: Path, change: Callable[[dict[str, Any]], Any]) -> None:
    document = json.loads((directory / "trace.json").read_text())
    change(document)
    _write(directory, document)


def _drop_kind(kind: str) -> Callable[[dict[str, Any]], Any]:
    def change(document: dict[str, Any]) -> None:
        document["records"] = [e for e in document["records"] if e["kind"] != kind]

    return change


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda d: d["records"][2].update(kind="causal_effect"), "unknown record kind"),
        (lambda d: d.update(schema_version="0.2"), "schema_version"),
        (lambda d: d["records"][0].update(schema_version="9.0"), "schema_version"),
        (lambda d: d["records"][3].update(id="activation:" + "0" * 32), "does not match content"),
        (_drop_kind("input"), "not in trace"),
        (_drop_kind("provenance"), "does not name a ProvenanceRecord"),
        (lambda d: d["records"].reverse(), "trace integrity"),
        (lambda d: d.update(tensor_file="../../etc/passwd"), "exactly the keys"),
        (lambda d: d.update(format="other"), "not a BeyondNN trace"),
        (lambda d: d.update(format_version=2), "format_version"),
        (lambda d: d.update(tensors=[*d["tensors"][::-1], "x"]), "sorted and unique"),
        (
            lambda d: d["records"][3]["data"]["value"].update(storage_key="../../etc/passwd"),
            "does not match content",
        ),
        (lambda d: d["config"].update(retention="device"), "retention"),
    ],
)
def test_corrupted_trace_documents_are_rejected(
    saved: Path, change: Callable[[dict[str, Any]], Any], match: str
) -> None:
    _edit(saved, change)
    with pytest.raises(TracePersistenceError, match=match):
        bnn.load_trace(saved)


def test_missing_or_malformed_trace_json(saved: Path) -> None:
    (saved / "trace.json").write_text("{not json")
    with pytest.raises(TracePersistenceError, match="not valid JSON"):
        bnn.load_trace(saved)
    (saved / "trace.json").unlink()
    with pytest.raises(TracePersistenceError, match="missing"):
        bnn.load_trace(saved)
    with pytest.raises(TracePersistenceError, match="not a trace directory"):
        bnn.load_trace(saved / "nope")


def test_nan_literal_in_json_is_rejected(saved: Path) -> None:
    text = (
        (saved / "trace.json").read_text().replace('"format_version": 1', '"format_version": NaN')
    )
    (saved / "trace.json").write_text(text)
    with pytest.raises(TracePersistenceError, match="non-standard JSON constant"):
        bnn.load_trace(saved)


def _sidecar(directory: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(directory / "tensors.pt", weights_only=True)
    assert isinstance(loaded, dict)
    return loaded


def _resave(directory: Path, tensors: dict[str, Any]) -> None:
    (directory / "tensors.pt").unlink()
    torch.save(tensors, directory / "tensors.pt")


def test_missing_sidecar_is_rejected(saved: Path) -> None:
    (saved / "tensors.pt").unlink()
    with pytest.raises(TracePersistenceError, match=r"tensors\.pt is missing"):
        bnn.load_trace(saved)


def test_missing_tensor_key_is_rejected(saved: Path) -> None:
    tensors = _sidecar(saved)
    tensors.pop(next(iter(tensors)))
    _resave(saved, tensors)
    with pytest.raises(TracePersistenceError, match="keys do not match"):
        bnn.load_trace(saved)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t: t.reshape(-1).flip(0).reshape(1, -1), "shape"),
        (lambda t: t.double(), "dtype"),
        (lambda t: t + 1, "content digest"),
    ],
)
def test_incompatible_tensors_are_rejected(saved: Path, mutate: Any, match: str) -> None:
    tensors = _sidecar(saved)
    key = next(iter(tensors))
    tensors[key] = mutate(tensors[key])
    _resave(saved, tensors)
    with pytest.raises(TracePersistenceError, match=match):
        bnn.load_trace(saved)


class Payload:
    def __reduce__(self) -> Any:
        return (os.system, ("echo pwned",))


def test_pickled_objects_in_the_sidecar_are_never_executed(saved: Path) -> None:
    tensors: dict[str, Any] = dict(_sidecar(saved))
    tensors[next(iter(tensors))] = Payload()
    _resave(saved, tensors)
    with pytest.raises(TracePersistenceError, match="could not be loaded safely"):
        bnn.load_trace(saved)


def test_non_tensor_sidecar_values_are_rejected(saved: Path) -> None:
    tensors: dict[str, Any] = dict(_sidecar(saved))
    tensors[next(iter(tensors))] = [1, 2, 3]
    _resave(saved, tensors)
    with pytest.raises(TracePersistenceError, match="not a plain tensor"):
        bnn.load_trace(saved)


def test_unexpected_sidecar_is_rejected(tmp_path: Path) -> None:
    t = bnn.trace(TinyMLP(), torch.ones(2, 4))  # summary: no tensors
    t.save(tmp_path / "trace")
    torch.save({}, tmp_path / "trace" / "tensors.pt")
    with pytest.raises(TracePersistenceError, match=r"unexpected tensors\.pt"):
        bnn.load_trace(tmp_path / "trace")


def test_symlinked_trace_files_are_refused(saved: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere.pt"
    (saved / "tensors.pt").rename(elsewhere)
    (saved / "tensors.pt").symlink_to(elsewhere)
    with pytest.raises(TracePersistenceError, match="symbolic link"):
        bnn.load_trace(saved)


def test_loaded_traces_are_read_only_and_fully_typed(saved: Path) -> None:
    loaded = bnn.load_trace(saved)
    assert isinstance(loaded.input, InputRecord)
    assert all(isinstance(a, ActivationRecord) for a in loaded.activations)
    assert isinstance(loaded.origin(loaded.activations[0]), ProvenanceRecord)
    with pytest.raises(TraceIntegrityError, match="finalised"):
        loaded._add(TraceLimitation(code="NO_ATTRIBUTION"))
