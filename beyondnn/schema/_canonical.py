"""Canonical JSON, stable digests, and an immutable JSON mapping.

Record identity and criteria fingerprints are SHA-256 digests of *canonical JSON*:

* keys sorted, separators ``(",", ":")``, UTF-8, ``ensure_ascii=False``;
* floats rendered with Python's shortest round-trip ``repr`` (via :mod:`json`);
* non-finite floats are rejected here (typed float fields are encoded as the
  strings ``"NaN"``, ``"Infinity"``, ``"-Infinity"`` *before* reaching this module);
* ``True`` and ``1`` are distinct, as are ``1`` and ``1.0``.

Python's process-randomised :func:`hash` is never used for identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from typing import Any, TypeAlias, Union

from .errors import SchemaTypeError

__all__ = ["EMPTY_JSON", "JsonMap", "JsonValue", "canonical_json", "digest"]

JsonValue: TypeAlias = Union[bool, int, float, str, tuple["JsonValue", ...], "JsonMap", None]


def canonical_json(obj: Any) -> str:
    """Serialise plain JSON data (dict/list/str/int/float/bool/None) canonically."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(obj: Any) -> str:
    """Hex SHA-256 of the canonical JSON of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _freeze(value: Any, path: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaTypeError(f"{path}: non-finite float {value!r} is not valid JSON data")
        return value
    if isinstance(value, JsonMap):
        return value
    if isinstance(value, Mapping):
        return JsonMap(value, _path=path)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v, f"{path}[{i}]") for i, v in enumerate(value))
    raise SchemaTypeError(f"{path}: {type(value).__name__} is not JSON data")


def _thaw(value: JsonValue) -> Any:
    if isinstance(value, JsonMap):
        return value.to_plain()
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


class JsonMap(Mapping[str, JsonValue]):
    """An immutable, hashable, deeply frozen mapping of JSON data.

    Lists become tuples and nested mappings become ``JsonMap``. Keys must be
    strings. Equality is mapping equality; the hash is derived from canonical JSON.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[str, Any] | None = None, *, _path: str = "$") -> None:
        items: dict[str, JsonValue] = {}
        for key, value in (data or {}).items():
            if not isinstance(key, str):
                raise SchemaTypeError(f"{_path}: JSON object keys must be str, got {key!r}")
            items[key] = _freeze(value, f"{_path}.{key}")
        self._data = items
        self._hash: int | None = None

    def __getitem__(self, key: str) -> JsonValue:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(canonical_json(self.to_plain()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, JsonMap):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self.to_plain() == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"JsonMap({self.to_plain()!r})"

    def to_plain(self) -> dict[str, Any]:
        """Return a fresh, mutable ``dict`` with lists in place of tuples."""
        return {k: _thaw(v) for k, v in self._data.items()}

    def digest(self) -> str:
        """Stable hex SHA-256 of this mapping's canonical JSON."""
        return digest(self.to_plain())


#: The shared empty ``JsonMap`` (safe as a dataclass default: it is immutable).
EMPTY_JSON = JsonMap()
