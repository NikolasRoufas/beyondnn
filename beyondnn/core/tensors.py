"""Tensor-tree walking and retention for traces (M1.6). Internal.

Leaf paths are deterministic strings, compatible with ``Site.output_path`` and
``NamedTensor.path``:

* a bare tensor: ``""``
* sequence element: ``"[0]"``, ``"[1][0]"``
* mapping entry with a ``str`` key: ``'["logits"]'`` (JSON-quoted key)
* mapping entry with an ``int`` key: ``"[3]"``
* root inputs/outputs are prefixed: ``"args[0]"``, ``'kwargs["mask"]'``,
  ``"output"``, ``'output["logits"]'`` (activation leaves are not prefixed,
  except module INPUT leaves: ``"args[0]"``)

Mappings are walked in their own (insertion) order; sequences in index order.
``None`` leaves are skipped silently. Other non-tensor leaves (numbers, strings,
objects, sets, non-str/int keys) are never stringified or pickled: they are only
counted, so the trace can report ``NON_TENSOR_LEAVES_IGNORED``.

Retention (per tensor leaf; the live tensor is never retained):

* ``"none"``: shape, dtype, original device only;
* ``"summary"`` (default): plus scalar summary statistics (float64, computed
  without autograd; ``None`` for complex tensors);
* ``"cpu"``: plus a detached, contiguous CPU clone stored under a content key
  ``sha256:<hex>`` (identical contents are stored once) and recorded as the
  ``TensorRef.content_digest``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import TensorRef, TensorStats

__all__ = ["RETENTIONS", "TensorLeaf", "tensor_ref", "walk"]

RETENTIONS = ("summary", "cpu", "none")


@dataclass(frozen=True)
class TensorLeaf:
    path: str
    tensor: torch.Tensor


def walk(value: Any, prefix: str = "") -> tuple[list[TensorLeaf], int]:
    """Return the tensor leaves of ``value`` in deterministic order and the number of
    ignored non-tensor leaves."""
    leaves: list[TensorLeaf] = []
    ignored = 0

    def visit(node: Any, path: str) -> None:
        nonlocal ignored
        if isinstance(node, torch.Tensor):
            leaves.append(TensorLeaf(path, node))
        elif node is None:
            return
        elif isinstance(node, (tuple, list)):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")
        elif isinstance(node, Mapping):
            for key, item in node.items():
                if isinstance(key, str):
                    visit(item, f"{path}[{json.dumps(key)}]")
                elif isinstance(key, int) and not isinstance(key, bool):
                    visit(item, f"{path}[{key}]")
                else:
                    ignored += 1
        else:
            ignored += 1

    visit(value, prefix)
    return leaves, ignored


def _stats(t: torch.Tensor) -> TensorStats | None:
    if t.is_complex():
        return None
    with torch.no_grad():
        x = t.detach().to(torch.float64)
        n = x.numel()
        if n == 0:
            nan = math.nan
            return TensorStats(numel=0, mean=nan, std=nan, min=nan, max=nan, l2_norm=0.0)
        return TensorStats(
            numel=n,
            mean=x.mean().item(),
            std=x.std(correction=0).item(),
            min=x.min().item(),
            max=x.max().item(),
            l2_norm=torch.linalg.vector_norm(x).item(),
        )


def tensor_ref(t: torch.Tensor, retention: str, store: dict[str, torch.Tensor]) -> TensorRef:
    """Describe ``t`` under ``retention``; for ``"cpu"`` also put its copy in ``store``."""
    if retention not in RETENTIONS:
        raise ValueError(f"retention must be one of {RETENTIONS}, got {retention!r}")
    shape = tuple(int(d) for d in t.shape)
    dtype = str(t.dtype).removeprefix("torch.")
    device = str(t.device)
    if retention == "none":
        return TensorRef(shape=shape, dtype=dtype, device=device)
    stats = _stats(t)
    if retention == "summary":
        return TensorRef(shape=shape, dtype=dtype, device=device, stats=stats)
    copy = t.detach().to("cpu").clone(memory_format=torch.contiguous_format)
    digest = "sha256:" + hashlib.sha256(tensor_bytes(copy)).hexdigest()
    store.setdefault(digest, copy)
    return TensorRef(
        shape=shape,
        dtype=dtype,
        device=device,
        stats=stats,
        storage_key=digest,
        content_digest=digest,
    )
