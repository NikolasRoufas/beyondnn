"""Exact identity of one model input sample (ADR-028, ADR-031).

``sample_id(*inputs, model_kwargs=...)`` is SHA-256 over the dtypes, shapes, and
bytes of every tensor plus JSON scalars, in positional then keyword order. It is
the identity used by intervention estimands, attribution records, claims, and
(since ADR-031) the ``sample_id`` of every traced ``InputRecord``. Two inputs have
the same id iff they have the same structure, dtypes, shapes, and bytes.

Privacy: the digest can confirm a guessed low-entropy input; it is not an
anonymisation mechanism.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from beyondnn.provenance.fingerprint import tensor_bytes

__all__ = ["SampleIdentityError", "sample_id"]


class SampleIdentityError(ValueError):
    """The input contains a value that has no deterministic identity."""


def _digest_update(h: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)})
        h.update(b"T" + header.encode() + tensor_bytes(value))
    elif value is None or isinstance(value, (bool, int, float, str)):
        h.update(b"J" + json.dumps(value, allow_nan=False).encode())
    elif isinstance(value, (tuple, list)):
        h.update(b"[%d" % len(value))
        for item in value:
            _digest_update(h, item)
    elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
        h.update(b"{%d" % len(value))
        for key in sorted(value):
            h.update(json.dumps(key).encode())
            _digest_update(h, value[key])
    else:
        raise SampleIdentityError(
            f"cannot identify an input sample containing {type(value).__name__}"
        )


def sample_id(*inputs: Any, model_kwargs: dict[str, Any] | None = None) -> str:
    """Deterministic identity of one input sample (see module docstring)."""
    h = hashlib.sha256(b"beyondnn.sample/v1")
    try:
        _digest_update(h, list(inputs))
        _digest_update(h, dict(model_kwargs or {}))
    except ValueError as exc:  # json: non-finite floats
        if isinstance(exc, SampleIdentityError):
            raise
        raise SampleIdentityError(f"cannot identify an input sample: {exc}") from None
    return "sha256:" + h.hexdigest()
