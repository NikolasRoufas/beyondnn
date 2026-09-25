"""Serialisation envelope and strict, versioned decoding (ADR-014).

Envelope (every key required, no others allowed)::

    {
      "schema_version": "0.1",
      "kind": "activation",
      "record_version": 1,
      "id": "activation:<32 hex>",
      "status": "measured" | null,
      "data": {<every init field, encoded>}
    }

Decoding rejects unknown kinds, unknown or missing fields, a newer
``record_version``, an incompatible ``schema_version``, and payloads whose stored
``id`` or ``status`` do not match their content. Older record versions are read
only through registered migrations. Tensor data is never part of the envelope;
sidecar storage arrives in M1.7.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ._canonical import canonical_json
from ._types import decode_fields, encode_fields
from .base import BaseRecord, migration_for, registered_kinds
from .errors import (
    DecodeError,
    IntegrityError,
    SchemaError,
    UnknownRecordKindError,
    UnsupportedVersionError,
)

__all__ = ["SCHEMA_VERSION", "from_dict", "from_json", "to_dict", "to_json"]

SCHEMA_VERSION = "0.1"

_ENVELOPE_KEYS = frozenset({"schema_version", "kind", "record_version", "id", "status", "data"})
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def to_dict(record: BaseRecord) -> dict[str, Any]:
    """Encode ``record`` as a JSON-compatible envelope."""
    status = record.status
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": record.KIND,
        "record_version": record.RECORD_VERSION,
        "id": record.id,
        "status": None if status is None else status.value,
        "data": encode_fields(record),
    }


def to_json(record: BaseRecord) -> str:
    """Canonical JSON of the envelope (sorted keys, compact, deterministic)."""
    return canonical_json(to_dict(record))


def _check_schema_version(version: Any) -> None:
    if not isinstance(version, str) or not (m := _VERSION_RE.match(version)):
        raise DecodeError(f"invalid schema_version {version!r}")
    ours = _VERSION_RE.match(SCHEMA_VERSION)
    assert ours is not None
    major, minor = int(m.group(1)), int(m.group(2))
    our_major, our_minor = int(ours.group(1)), int(ours.group(2))
    # Before 1.0 every minor version may break compatibility, so require equality.
    compatible = (
        version == SCHEMA_VERSION if our_major == 0 else (major == our_major and minor <= our_minor)
    )
    if not compatible:
        raise UnsupportedVersionError(
            f"schema_version {version} cannot be read by this BeyondNN (schema {SCHEMA_VERSION})"
        )


def from_dict(payload: Any) -> BaseRecord:
    """Decode an envelope produced by :func:`to_dict`. Strict; see module docstring."""
    if not isinstance(payload, dict):
        raise DecodeError(f"record envelope must be a JSON object, got {type(payload).__name__}")
    keys = payload.keys()
    if keys != _ENVELOPE_KEYS:
        raise DecodeError(
            f"record envelope keys must be exactly {sorted(_ENVELOPE_KEYS)}; "
            f"missing {sorted(_ENVELOPE_KEYS - keys)}, unknown {sorted(keys - _ENVELOPE_KEYS)}"
        )
    _check_schema_version(payload["schema_version"])

    kind = payload["kind"]
    cls = registered_kinds().get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise UnknownRecordKindError(
            f"unknown record kind {kind!r}; it may come from a newer BeyondNN version"
        )

    version = payload["record_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise DecodeError(f"invalid record_version {version!r}")
    if version > cls.RECORD_VERSION:
        raise UnsupportedVersionError(
            f"{kind} record_version {version} is newer than supported ({cls.RECORD_VERSION})"
        )
    data = payload["data"]
    while version < cls.RECORD_VERSION:
        migrate = migration_for(kind, version)
        if migrate is None:
            raise UnsupportedVersionError(
                f"no migration registered for {kind} record_version {version}"
            )
        try:
            data = migrate(data)
        except DecodeError:
            raise
        except Exception as exc:
            raise DecodeError(
                f"migrating {kind} from record_version {version} failed: {exc}"
            ) from exc
        version += 1

    try:
        record = cls(**decode_fields(cls, data, "$.data"))
    except DecodeError:
        raise
    except SchemaError as exc:
        raise DecodeError(f"invalid {kind} record: {exc}") from exc

    migrated = payload["record_version"] != cls.RECORD_VERSION
    if not migrated and payload["id"] != record.id:
        raise IntegrityError(
            f"stored id {payload['id']!r} does not match content (computed {record.id!r})"
        )
    status = record.status
    if payload["status"] != (None if status is None else status.value):
        raise IntegrityError(
            f"stored status {payload['status']!r} does not match kind {kind!r} "
            f"(status {None if status is None else status.value!r})"
        )
    return record


def _reject_constant(name: str) -> Any:
    raise DecodeError(f"non-standard JSON constant {name} is not allowed")


def from_json(text: str) -> BaseRecord:
    """Decode canonical (or any standard) JSON produced by :func:`to_json`."""
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise DecodeError(f"invalid JSON: {exc}") from exc
    return from_dict(payload)
