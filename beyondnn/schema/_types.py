"""Runtime type checking and field-level encoding for schema dataclasses.

The schema supports a deliberately small annotation vocabulary:

``str``, ``int``, ``float``, ``bool``, ``Enum`` subclasses, ``X | None``,
``tuple[X, ...]``, :class:`JsonMap`, and :class:`Value` subclasses.

Any other annotation on a schema dataclass raises ``TypeError`` when the class is
first compiled, so an unsupported field can never reach serialisation silently.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from collections.abc import Mapping
from enum import Enum
from typing import Any

from ._canonical import JsonMap
from .errors import DecodeError, SchemaError, SchemaTypeError

__all__ = [
    "Checked",
    "FieldSpec",
    "Value",
    "decode_fields",
    "encode_fields",
    "field_specs",
    "finalize_fields",
    "require",
]

_NONFINITE_ENCODE = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
_NONFINITE_DECODE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


class _Type:
    name: str = "?"

    def check(self, value: Any, path: str) -> Any:  # returns the (normalised) value
        raise NotImplementedError

    def encode(self, value: Any) -> Any:
        return value

    def decode(self, data: Any, path: str) -> Any:
        return self.check(data, path)

    def _fail(self, value: Any, path: str) -> SchemaTypeError:
        return SchemaTypeError(f"{path}: expected {self.name}, got {type(value).__name__}")


class _Str(_Type):
    name = "str"

    def check(self, value: Any, path: str) -> Any:
        if not isinstance(value, str):
            raise self._fail(value, path)
        return value


class _Int(_Type):
    name = "int"

    def check(self, value: Any, path: str) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._fail(value, path)
        return value


class _Bool(_Type):
    name = "bool"

    def check(self, value: Any, path: str) -> Any:
        if not isinstance(value, bool):
            raise self._fail(value, path)
        return value


class _Float(_Type):
    """Floats. Ints are accepted and normalised to float. Non-finite values are
    allowed and encoded as the strings ``"NaN"``, ``"Infinity"``, ``"-Infinity"``."""

    name = "float"

    def check(self, value: Any, path: str) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._fail(value, path)
        value = float(value)
        return 0.0 if value == 0.0 else value  # -0.0 == 0.0: one encoding, one id

    def encode(self, value: Any) -> Any:
        return value if math.isfinite(value) else _NONFINITE_ENCODE[repr(value)]

    def decode(self, data: Any, path: str) -> Any:
        if isinstance(data, str):
            if data not in _NONFINITE_DECODE:
                raise DecodeError(f"{path}: invalid float string {data!r}")
            return _NONFINITE_DECODE[data]
        return self.check(data, path)


class _EnumT(_Type):
    def __init__(self, cls: type[Enum]) -> None:
        self.cls = cls
        self.name = cls.__name__

    def check(self, value: Any, path: str) -> Any:
        if not isinstance(value, self.cls):
            raise self._fail(value, path)
        return value

    def encode(self, value: Any) -> Any:
        return value.value

    def decode(self, data: Any, path: str) -> Any:
        try:
            return self.cls(data)
        except ValueError:
            raise DecodeError(f"{path}: {data!r} is not a valid {self.name}") from None


class _Optional(_Type):
    def __init__(self, inner: _Type) -> None:
        self.inner = inner
        self.name = f"{inner.name} | None"

    def check(self, value: Any, path: str) -> Any:
        return None if value is None else self.inner.check(value, path)

    def encode(self, value: Any) -> Any:
        return None if value is None else self.inner.encode(value)

    def decode(self, data: Any, path: str) -> Any:
        return None if data is None else self.inner.decode(data, path)


class _Tuple(_Type):
    def __init__(self, inner: _Type) -> None:
        self.inner = inner
        self.name = f"tuple[{inner.name}, ...]"

    def check(self, value: Any, path: str) -> Any:
        if not isinstance(value, tuple):
            raise self._fail(value, path)
        return tuple(self.inner.check(v, f"{path}[{i}]") for i, v in enumerate(value))

    def encode(self, value: Any) -> Any:
        return [self.inner.encode(v) for v in value]

    def decode(self, data: Any, path: str) -> Any:
        if not isinstance(data, list):
            raise DecodeError(f"{path}: expected a JSON array, got {type(data).__name__}")
        return tuple(self.inner.decode(v, f"{path}[{i}]") for i, v in enumerate(data))


class _JsonMapT(_Type):
    name = "JsonMap"

    def check(self, value: Any, path: str) -> Any:
        if isinstance(value, JsonMap):
            return value
        if isinstance(value, Mapping):
            return JsonMap(value, _path=path)
        raise self._fail(value, path)

    def encode(self, value: Any) -> Any:
        return value.to_plain()

    def decode(self, data: Any, path: str) -> Any:
        if not isinstance(data, dict):
            raise DecodeError(f"{path}: expected a JSON object, got {type(data).__name__}")
        return JsonMap(data, _path=path)


class _ValueT(_Type):
    def __init__(self, cls: type[Value]) -> None:
        self.cls = cls
        self.name = cls.__name__

    def check(self, value: Any, path: str) -> Any:
        if not isinstance(value, self.cls):
            raise self._fail(value, path)
        return value

    def encode(self, value: Any) -> Any:
        return encode_fields(value)

    def decode(self, data: Any, path: str) -> Any:
        return self.cls(**decode_fields(self.cls, data, path))


_PRIMITIVES: dict[Any, _Type] = {str: _Str(), int: _Int(), float: _Float(), bool: _Bool()}


def _compile(tp: Any, where: str) -> _Type:
    if tp in _PRIMITIVES:
        return _PRIMITIVES[tp]
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(args) == 2 and len(non_none) == 1:
            return _Optional(_compile(non_none[0], where))
    elif origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return _Tuple(_compile(args[0], where))
    elif isinstance(tp, type) and origin is None:  # 3.10: list[int] passes isinstance(type)
        if issubclass(tp, Enum):
            return _EnumT(tp)
        if issubclass(tp, JsonMap):
            return _JsonMapT()
        if issubclass(tp, Value):
            return _ValueT(tp)
    raise TypeError(f"{where}: unsupported schema annotation {tp!r}")


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    name: str
    type: _Type


_SPECS: dict[type, tuple[FieldSpec, ...]] = {}


def field_specs(cls: type) -> tuple[FieldSpec, ...]:
    """The init fields of a schema dataclass with compiled types (cached)."""
    specs = _SPECS.get(cls)
    if specs is None:
        hints = typing.get_type_hints(cls)
        specs = tuple(
            FieldSpec(f.name, _compile(hints[f.name], f"{cls.__name__}.{f.name}"))
            for f in dataclasses.fields(cls)
            if f.init
        )
        _SPECS[cls] = specs
    return specs


def encode_fields(obj: Any) -> dict[str, Any]:
    """Encode a schema dataclass's init fields as plain JSON data."""
    return {s.name: s.type.encode(getattr(obj, s.name)) for s in field_specs(type(obj))}


def decode_fields(cls: type, data: Any, path: str) -> dict[str, Any]:
    """Decode plain JSON data into constructor kwargs for ``cls``.

    Strict: every init field must be present and no unknown field is allowed.
    """
    if not isinstance(data, dict):
        raise DecodeError(f"{path}: expected a JSON object, got {type(data).__name__}")
    specs = field_specs(cls)
    names = {s.name for s in specs}
    missing = sorted(names - data.keys())
    unknown = sorted(data.keys() - names)
    if missing:
        raise DecodeError(f"{path}: missing field(s) {missing} for {cls.__name__}")
    if unknown:
        raise DecodeError(f"{path}: unknown field(s) {unknown} for {cls.__name__}")
    return {s.name: s.type.decode(data[s.name], f"{path}.{s.name}") for s in specs}


def finalize_fields(obj: Any) -> None:
    """Type-check every init field of ``obj``, writing back normalised values."""
    cls = type(obj)
    for spec in field_specs(cls):
        value = getattr(obj, spec.name)
        normalised = spec.type.check(value, f"{cls.__name__}.{spec.name}")
        if normalised is not value:
            object.__setattr__(obj, spec.name, normalised)


class Checked:
    """Mixin for schema dataclasses: type-check, then run ``_validate``."""

    __slots__ = ()

    def __post_init__(self) -> None:
        finalize_fields(self)
        self._validate()

    def _validate(self) -> None:
        """Local invariants. Override in subclasses; raise a SchemaError subclass."""


class Value(Checked):
    """Base for immutable value types embedded in records (not records themselves)."""

    __slots__ = ()


def require(condition: bool, message: str, error: type[SchemaError] = SchemaError) -> None:
    """Raise ``error(message)`` unless ``condition`` holds."""
    if not condition:
        raise error(message)
