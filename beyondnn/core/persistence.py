"""Trace persistence: ``TraceResult.save(path)`` and ``load_trace(path)`` (M1.7).

Directory format (both file names are fixed; nothing in the JSON is a path)::

    <path>/trace.json   deterministic JSON (sorted keys, 2-space indent, UTF-8)
    <path>/tensors.pt   only if tensors were retained: a plain {storage_key: Tensor}
                        dict, read with torch.load(weights_only=True)

``trace.json``::

    {"format": "beyondnn.trace", "format_version": 1, "schema_version": "0.1",
     "config": {...}, "records": [<record envelopes in execution order>],
     "tensors": [<sorted storage keys>]}

Save guarantee: everything is written into a sibling temporary directory
(``trace.json`` last) which is then renamed to ``path`` in one step, so ``path``
either does not exist or is complete. An existing ``path`` is never overwritten.

Load validates everything again: document keys and versions, every record (ids,
invariants), every reference and provenance link (via ``TraceResult``), and every
tensor's presence, type, dtype, shape, and content digest. Old record versions
are migrated; when that changes an id, every later reference to it (``derived_from``,
evidence/claim/spec/result refs, ``applies_to``, ``provenance_id``) is rewritten
deterministically before those records are decoded. Records whose data was
rewritten are first verified against their stored ids. Nothing unknown is dropped:
unknown keys, kinds, or versions fail the load.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import SCHEMA_VERSION, BaseRecord, SchemaError, TensorRef, to_dict
from beyondnn.schema.codec import _check_schema_version, _decode, verify_stored_id

from .trace import TraceConfig, TraceError, TraceIntegrityError, TraceResult

__all__ = ["FORMAT", "FORMAT_VERSION", "TracePersistenceError", "load_trace", "save_trace"]

FORMAT = "beyondnn.trace"
FORMAT_VERSION = 1
TRACE_JSON = "trace.json"
TENSOR_FILE = "tensors.pt"
_DOC_KEYS = frozenset(
    {"format", "format_version", "schema_version", "config", "records", "tensors"}
)


class TracePersistenceError(TraceError):
    """A trace could not be saved or loaded safely."""


# ------------------------------------------------------------------ save


def save_trace(trace: TraceResult, path: str | os.PathLike[str]) -> None:
    """Save ``trace`` to the new directory ``path`` (atomically; never overwrites)."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"parent directory {target.parent} does not exist")
    keys = sorted(trace._tensors)
    document = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config": trace.config.to_json(),
        "records": [to_dict(r) for r in trace.records],
        "tensors": keys,
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    )
    try:
        if keys:
            with open(staging / TENSOR_FILE, "wb") as fh:
                torch.save({k: trace._tensors[k] for k in keys}, fh)
                fh.flush()
                os.fsync(fh.fileno())
        with open(staging / TRACE_JSON, "w", encoding="utf-8") as fh:  # written last
            fh.write(text + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ------------------------------------------------------------------ load


def _reject_constant(name: str) -> Any:
    raise TracePersistenceError(f"non-standard JSON constant {name} in trace.json")


def _owned_file(directory: Path, name: str) -> Path:
    candidate = directory / name
    if candidate.is_symlink():
        raise TracePersistenceError(f"{name} is a symbolic link; refusing to follow it")
    return candidate


def _remap(value: Any, id_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return id_map.get(value, value)
    if isinstance(value, list):
        return [_remap(v, id_map) for v in value]
    if isinstance(value, dict):
        return {k: _remap(v, id_map) for k, v in value.items()}
    return value


def _decode_record(envelope: Any, id_map: dict[str, str]) -> BaseRecord:
    if not isinstance(envelope, dict) or "data" not in envelope:
        raise TracePersistenceError("a record entry is not a record envelope")
    remapped = _remap(envelope["data"], id_map)
    needs_rewrite = remapped != envelope["data"]
    try:
        if not needs_rewrite:
            record = _decode(envelope)  # full checks, incl. stored id (also pre-migration)
        else:
            verify_stored_id(envelope)  # the original, as stored, must be intact
            rewritten = copy.deepcopy(envelope)
            rewritten["data"] = remapped
            record = _decode(rewritten, check_stored_id=False)
    except SchemaError as exc:
        raise TracePersistenceError(f"invalid record {envelope.get('id')!r}: {exc}") from exc
    if record.id != envelope.get("id"):
        id_map[envelope["id"]] = record.id  # migration/remap changed the id
    return record


def _load_tensors(directory: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    sidecar = _owned_file(directory, TENSOR_FILE)
    if not keys:
        if sidecar.exists():
            raise TracePersistenceError("unexpected tensors.pt: the trace declares no tensors")
        return {}
    if not sidecar.is_file():
        raise TracePersistenceError("tensors.pt is missing but the trace declares tensors")
    try:
        loaded = torch.load(sidecar, weights_only=True, map_location="cpu")
    except Exception as exc:
        raise TracePersistenceError(f"tensors.pt could not be loaded safely: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TracePersistenceError("tensors.pt must contain a plain mapping of tensors")
    if set(loaded) != set(keys):
        raise TracePersistenceError(
            f"tensors.pt keys do not match the trace (missing {sorted(set(keys) - set(loaded))}, "
            f"unexpected {sorted(set(loaded) - set(keys))})"
        )
    for key, value in loaded.items():
        if type(value) is not torch.Tensor:
            raise TracePersistenceError(f"tensors.pt entry {key!r} is not a plain tensor")
    return loaded


def _check_tensor(ref: TensorRef, tensor: torch.Tensor) -> None:
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype != ref.dtype:
        raise TracePersistenceError(
            f"tensor {ref.storage_key}: dtype {dtype} != recorded {ref.dtype}"
        )
    if tuple(tensor.shape) != ref.shape:
        raise TracePersistenceError(
            f"tensor {ref.storage_key}: shape {tuple(tensor.shape)} != recorded {ref.shape}"
        )
    if ref.content_digest is not None:
        actual = "sha256:" + hashlib.sha256(tensor_bytes(tensor)).hexdigest()
        if actual != ref.content_digest:
            raise TracePersistenceError(f"tensor {ref.storage_key}: content digest mismatch")


def load_trace(path: str | os.PathLike[str]) -> TraceResult:
    """Load and fully re-validate a trace saved with ``TraceResult.save``."""
    directory = Path(path)
    if not directory.is_dir():
        raise TracePersistenceError(f"{directory} is not a trace directory")
    document_path = _owned_file(directory, TRACE_JSON)
    if not document_path.is_file():
        raise TracePersistenceError(f"{TRACE_JSON} is missing (incomplete or not a trace)")
    try:
        document = json.loads(
            document_path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TracePersistenceError(f"{TRACE_JSON} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != _DOC_KEYS:
        raise TracePersistenceError(f"{TRACE_JSON} must have exactly the keys {sorted(_DOC_KEYS)}")
    if document["format"] != FORMAT:
        raise TracePersistenceError(f"not a BeyondNN trace (format {document['format']!r})")
    if document["format_version"] != FORMAT_VERSION:
        raise TracePersistenceError(
            f"unsupported trace format_version {document['format_version']!r}"
        )
    try:
        _check_schema_version(document["schema_version"])
    except SchemaError as exc:
        raise TracePersistenceError(str(exc)) from exc
    records, keys = document["records"], document["tensors"]
    if not isinstance(records, list):
        raise TracePersistenceError("records must be a list")
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise TracePersistenceError("tensors must be a list of storage keys")
    if keys != sorted(set(keys)):
        raise TracePersistenceError("tensor keys must be sorted and unique")

    try:
        config = TraceConfig.from_json(document["config"])
    except TraceIntegrityError as exc:
        raise TracePersistenceError(str(exc)) from exc
    trace = TraceResult(config)
    id_map: dict[str, str] = {}
    try:
        for envelope in records:
            trace._add(_decode_record(envelope, id_map))
    except TraceIntegrityError as exc:
        raise TracePersistenceError(f"trace integrity check failed: {exc}") from exc

    tensors = _load_tensors(directory, keys)
    for ref in trace._tensor_refs():
        if ref.storage_key is not None:
            if ref.storage_key not in tensors:
                raise TracePersistenceError(f"tensor {ref.storage_key} is missing from tensors.pt")
            _check_tensor(ref, tensors[ref.storage_key])
    for key, tensor in tensors.items():
        trace._add_tensor(key, tensor)
    try:
        trace._seal()
    except TraceIntegrityError as exc:
        raise TracePersistenceError(str(exc)) from exc
    return trace
