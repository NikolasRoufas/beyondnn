"""FULL model fingerprint (algorithm version 1).

"FULL" means: the full *supported* PyTorch module topology and registered tensor
state under BeyondNN's v1 fingerprint specification. It is NOT a cryptographic
identity of every behaviour the Python object could exhibit: Python code is not
hashed, so changing ``forward`` (e.g. ``x + 1`` to ``x + 2``) while keeping the
qualified class name, module tree, parameters and buffers leaves the fingerprint
unchanged. It is artifact identity, not semantic equivalence.

The fingerprint is two SHA-256 digests over canonical, device-independent
descriptions of the model:

Structure digest
    canonical JSON of: every module path (``named_modules(remove_duplicate=False)``,
    so shared modules appear under each path) with its fully qualified class;
    groups of paths that are the same module object; every parameter (name, role,
    dtype, shape) and every registered non-``None`` buffer (name, role,
    ``persistent`` flag, dtype, shape); and groups of tensor names that are the
    same tensor object (tied parameters). Groups are expressed by sorted names,
    never by object ids or addresses.

State digest
    for each distinct parameter or buffer (persistent AND non-persistent: a
    non-persistent buffer is only excluded from ``state_dict``, it can still take
    part in ``forward``), ordered by its representative name (the smallest name in
    its alias group): a length-prefixed canonical JSON header (name, role, dtype,
    shape, nbytes, and ``persistent`` for buffers) followed by the tensor's raw
    bytes, read from a contiguous CPU copy with lazy conjugate/negative bits
    resolved. Values are hashed bitwise and never converted between dtypes.

Excluded by design: device placement, ``requires_grad``, ``None`` parameter or
buffer slots, Python source code, Python object ids, and arbitrary module
attributes. Class paths are included, so moving a class to another module changes
the structure digest even if behaviour is unchanged. The algorithm is read-only:
it does not modify the model or any RNG. No result is cached, because in-place
mutation could not be detected reliably.

Unsupported (raises :class:`FingerprintError` instead of hashing partially):
tensor subclasses other than ``Parameter`` (including lazy/uninitialised
parameters), meta, sparse and other non-strided layouts, quantized and nested
tensors, modules with custom ``get_extra_state``, and big-endian hosts.
"""

from __future__ import annotations

import ctypes
import hashlib
import sys
from dataclasses import dataclass

import torch
from torch import nn

from beyondnn.schema import FingerprintMethod, ModelIdentity
from beyondnn.schema._canonical import canonical_json, digest

__all__ = ["ALGORITHM_VERSION", "FingerprintError", "fingerprint_model", "tensor_bytes"]

ALGORITHM_VERSION = 1


class FingerprintError(ValueError):
    """The model contains something the FULL fingerprint cannot represent exactly."""


@dataclass
class _Slot:
    name: str
    role: str  # "parameter" | "buffer"
    persistent: bool
    tensor: torch.Tensor


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _non_persistent_buffer_names(module: nn.Module) -> frozenset[str]:
    """Names of ``module``'s own buffers registered with ``persistent=False``.

    The single place that reads PyTorch's private ``_non_persistent_buffers_set``:
    torch has no public per-buffer persistence API, and deriving it from
    ``state_dict()`` would be subject to user state-dict hooks. Compatibility with
    the tested torch versions is checked against ``state_dict()`` keys in the tests.
    """
    names = getattr(module, "_non_persistent_buffers_set", None)
    if not isinstance(names, set):
        raise FingerprintError(
            "unsupported torch version: nn.Module._non_persistent_buffers_set is unavailable, "
            "so buffer persistence cannot be determined"
        )
    return frozenset(names)


def _check_supported(slot: _Slot) -> None:
    t = slot.tensor
    problem = None
    if type(t) not in (torch.Tensor, nn.Parameter):
        problem = f"tensor subclass {type(t).__name__}"
    elif t.is_meta:
        problem = "meta tensor (no data)"
    elif t.layout != torch.strided:
        problem = f"layout {t.layout}"
    elif t.is_quantized:
        problem = "quantized tensor"
    elif t.is_nested:
        problem = "nested tensor"
    if problem:
        raise FingerprintError(f"cannot fingerprint {slot.role} {slot.name!r}: {problem}")


def tensor_bytes(t: torch.Tensor) -> bytes:
    """The raw bytes of ``t``'s values in row-major order, as a CPU copy.

    Lazy conjugate/negative bits are resolved first. The bytes are read from a
    fresh contiguous copy with ``ctypes.string_at``; the address is used only to
    read and never enters any digest. (``bytes(untyped_storage())`` gives the
    same bytes but iterates in Python: ~3.7 s per 4 MB.)
    """
    plain = t.detach().resolve_conj().resolve_neg().to("cpu")
    plain = plain.clone(memory_format=torch.contiguous_format)
    nbytes = plain.numel() * plain.element_size()
    if plain.storage_offset() != 0 or plain.untyped_storage().nbytes() != nbytes:
        raise FingerprintError("internal error: contiguous copy does not own exact storage")
    if nbytes == 0:
        return b""
    return ctypes.string_at(plain.data_ptr(), nbytes)


def _collect(model: nn.Module) -> tuple[list[dict[str, str]], list[list[str]], list[_Slot]]:
    modules: list[dict[str, str]] = []
    module_objects: dict[int, list[str]] = {}
    slots: list[_Slot] = []
    for path, module in model.named_modules(remove_duplicate=False):
        if type(module).get_extra_state is not nn.Module.get_extra_state:
            raise FingerprintError(
                f"module {path or '<root>'!r} defines get_extra_state; extra state is not "
                "covered by the FULL fingerprint"
            )
        modules.append({"path": path, "class": _qualified(type(module))})
        # id() only groups objects within this call; it never enters a digest.
        module_objects.setdefault(id(module), []).append(path)
        non_persistent = _non_persistent_buffer_names(module)
        for name, param in module.named_parameters(recurse=False, remove_duplicate=False):
            slots.append(_Slot(_join(path, name), "parameter", True, param))
        for name, buf in module.named_buffers(recurse=False, remove_duplicate=False):
            slots.append(_Slot(_join(path, name), "buffer", name not in non_persistent, buf))
    module_groups = [sorted(g) for g in module_objects.values() if len(g) > 1]
    return modules, sorted(module_groups), slots


def _describe(name: str, slot: _Slot) -> dict[str, object]:
    entry: dict[str, object] = {"name": name, "role": slot.role}
    if slot.role == "buffer":
        entry["persistent"] = slot.persistent
    entry["dtype"] = _dtype(slot.tensor)
    entry["shape"] = list(slot.tensor.shape)
    return entry


def fingerprint_model(model: nn.Module) -> ModelIdentity:
    """Compute the FULL identity of ``model``'s current structure and state.

    Recomputed on every call (no cache). Raises :class:`FingerprintError` for
    anything the algorithm cannot represent exactly.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("fingerprint_model() requires a torch.nn.Module")
    if sys.byteorder != "little":
        raise FingerprintError("the FULL fingerprint (v1) is defined for little-endian hosts only")

    modules, module_groups, slots = _collect(model)

    groups: dict[int, list[_Slot]] = {}
    for slot in slots:
        groups.setdefault(id(slot.tensor), []).append(slot)
    for members in groups.values():
        if len({(s.role, s.persistent) for s in members}) > 1:
            names = sorted(s.name for s in members)
            raise FingerprintError(f"one tensor is registered with different roles: {names}")

    tensors: list[dict[str, object]] = []
    for slot in slots:
        _check_supported(slot)
        tensors.append(_describe(slot.name, slot))
    tensors.sort(key=lambda entry: str(entry["name"]))
    tensor_groups = sorted(
        sorted(s.name for s in members) for members in groups.values() if len(members) > 1
    )
    structure = {
        "algorithm": "beyondnn.model_structure",
        "version": ALGORITHM_VERSION,
        "modules": sorted(modules, key=lambda m: m["path"]),
        "module_alias_groups": module_groups,
        "tensors": tensors,
        "tensor_alias_groups": tensor_groups,
    }

    unique = sorted(
        ((min(s.name for s in members), members[0]) for members in groups.values()),
        key=lambda pair: pair[0],
    )
    state = hashlib.sha256(b"beyondnn.model_state/v%d\n" % ALGORITHM_VERSION)
    counts = {"parameter": [0, 0], "buffer": [0, 0]}
    for representative, slot in unique:
        data = tensor_bytes(slot.tensor)
        header = canonical_json(_describe(representative, slot) | {"nbytes": len(data)}).encode(
            "utf-8"
        )
        state.update(len(header).to_bytes(8, "little"))
        state.update(header)
        state.update(data)
        counts[slot.role][0] += 1
        counts[slot.role][1] += slot.tensor.numel()

    return ModelIdentity(
        model_class=_qualified(type(model)),
        method=FingerprintMethod.FULL,
        algorithm_version=ALGORITHM_VERSION,
        structure_digest="sha256:" + digest(structure),
        state_digest="sha256:" + state.hexdigest(),
        parameter_tensors=counts["parameter"][0],
        parameter_elements=counts["parameter"][1],
        buffer_tensors=counts["buffer"][0],
        buffer_elements=counts["buffer"][1],
    )
