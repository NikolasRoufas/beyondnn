"""FULL model fingerprint: determinism, mutation, metadata, structure, aliasing."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

from beyondnn.provenance import ALGORITHM_VERSION, FingerprintError, fingerprint_model
from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import FingerprintMethod, ModelIdentity

ROOT = Path(__file__).resolve().parents[1]


class Net(nn.Module):
    """A small test-local model (not the M1.3 model package)."""

    def __init__(self, seed: int = 0, *, tie: bool = False, copy_tied: bool = False) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Linear(4, 4, bias=False)
        self.body = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.head = nn.Linear(4, 4, bias=False)
        self.norm = nn.BatchNorm1d(4)
        if tie:
            self.head.weight = self.embed.weight
        elif copy_tied:
            self.head.weight = nn.Parameter(self.embed.weight.detach().clone())


def fp(model: nn.Module) -> ModelIdentity:
    return fingerprint_model(model)


# ----------------------------------------------------------------- determinism


def test_repeated_calls_are_identical() -> None:
    m = Net()
    assert fp(m) == fp(m)


def test_separately_built_identical_models_match() -> None:
    assert fp(Net(seed=3)) == fp(Net(seed=3))
    assert fp(Net(seed=3)).state_digest != fp(Net(seed=4)).state_digest


_SUBPROCESS = """
import torch
from torch import nn
from beyondnn.provenance import fingerprint_model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Linear(4, 4, bias=False)
        self.body = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.head = nn.Linear(4, 4, bias=False)
        self.head.weight = self.embed.weight
        self.norm = nn.BatchNorm1d(4)

i = fingerprint_model(Net())
print(i.structure_digest, i.state_digest)
"""


def test_fingerprint_is_stable_across_processes_and_hash_seeds() -> None:
    results = set()
    for seed in ("0", "1", "4242"):
        env = os.environ | {"PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
            env=env,
        )
        results.add(out.stdout.strip())
    assert len(results) == 1


def test_identity_fields() -> None:
    i = fp(Net())
    assert i.method is FingerprintMethod.FULL
    assert i.algorithm_version == ALGORITHM_VERSION == 1
    assert i.model_class.endswith("test_fingerprint.Net")
    # embed.weight 16, body 32+8+32+4 (4 tensors), head.weight 16, norm weight+bias 8
    assert (i.parameter_tensors, i.parameter_elements) == (8, 116)
    # BatchNorm persistent buffers: running_mean, running_var, num_batches_tracked
    assert (i.buffer_tensors, i.buffer_elements) == (3, 9)


# ----------------------------------------------------------------- mutation


def test_parameter_change_changes_state_and_restoring_restores_it() -> None:
    m = Net()
    before = fp(m)
    with torch.no_grad():
        weight = m.get_parameter("body.0.weight")
        original = weight[0, 0].item()
        weight[0, 0] += 1.0
    changed = fp(m)
    assert changed.state_digest != before.state_digest
    assert changed.structure_digest == before.structure_digest
    with torch.no_grad():
        weight[0, 0] = original
    assert fp(m) == before


def test_single_bit_change_is_detected() -> None:
    m = Net()
    before = fp(m).state_digest
    with torch.no_grad():
        w = m.get_parameter("body.2.bias")
        w[0] = torch.nextafter(w[0], torch.tensor(float("inf")))
    assert fp(m).state_digest != before


def test_persistent_buffer_change_changes_state() -> None:
    m = Net()
    before = fp(m).state_digest
    m.get_buffer("norm.running_mean").add_(0.5)
    assert fp(m).state_digest != before
    before = fp(m).state_digest
    m.get_buffer("norm.num_batches_tracked").add_(1)
    assert fp(m).state_digest != before


def test_requires_grad_and_mode_are_not_model_identity() -> None:
    m = Net()
    before = fp(m)
    m.embed.weight.requires_grad_(False)
    m.eval()
    assert fp(m) == before


# ----------------------------------------------------------------- metadata


def test_dtype_change_with_equal_values_changes_identity() -> None:
    a = nn.Linear(2, 2)
    with torch.no_grad():
        a.weight.fill_(1.0)
        a.bias.fill_(0.0)
    b = copy.deepcopy(a).double()
    ia, ib = fp(a), fp(b)
    assert ia.structure_digest != ib.structure_digest
    assert ia.state_digest != ib.state_digest


def test_float32_and_float64_never_share_state_digest_for_equal_values() -> None:
    # Same numeric values and names, different dtypes: headers and bytes differ.
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", torch.tensor([1.0, 2.0], dtype=torch.float32))
    b.register_buffer("x", torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert fp(a).state_digest != fp(b).state_digest


def test_shape_change_with_same_bytes_changes_identity() -> None:
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", torch.arange(6.0).reshape(2, 3))
    b.register_buffer("x", torch.arange(6.0).reshape(3, 2))
    assert fp(a).structure_digest != fp(b).structure_digest
    assert fp(a).state_digest != fp(b).state_digest


def test_tensor_name_matters() -> None:
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", torch.ones(2))
    b.register_buffer("y", torch.ones(2))
    assert fp(a).state_digest != fp(b).state_digest


def test_parameter_and_buffer_roles_are_distinguished() -> None:
    a, b = nn.Module(), nn.Module()
    a.register_parameter("x", nn.Parameter(torch.ones(2)))
    b.register_buffer("x", torch.ones(2))
    assert fp(a).structure_digest != fp(b).structure_digest
    assert fp(a).state_digest != fp(b).state_digest


def test_negative_zero_is_distinct_bitwise_state() -> None:
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", torch.tensor([0.0]))
    b.register_buffer("x", torch.tensor([-0.0]))
    assert fp(a).state_digest != fp(b).state_digest


def test_non_contiguous_and_conjugate_views_hash_their_values() -> None:
    base = torch.arange(12.0).reshape(3, 4)
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", base.t())  # non-contiguous
    b.register_buffer("x", base.t().contiguous())
    assert fp(a).state_digest == fp(b).state_digest
    z = torch.tensor([1 + 2j, 3 - 1j])
    c, d = nn.Module(), nn.Module()
    c.register_buffer("z", z.conj())  # lazy conjugate bit
    d.register_buffer("z", z.conj().resolve_conj())
    assert fp(c).state_digest == fp(d).state_digest


# ----------------------------------------------------------------- structure


class Wrapped(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(2, 2)


class OtherClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(2, 2)


def test_same_values_different_module_structure_differs() -> None:
    torch.manual_seed(0)
    flat = nn.Linear(2, 2)
    nested = Wrapped()
    nested.inner.load_state_dict(flat.state_dict())
    assert fp(flat).structure_digest != fp(nested).structure_digest


def test_module_class_is_structural() -> None:
    a, b = Wrapped(), OtherClass()
    b.load_state_dict(a.state_dict())
    ia, ib = fp(a), fp(b)
    assert ia.structure_digest != ib.structure_digest
    assert ia.state_digest == ib.state_digest  # same names, dtypes, shapes, values


def test_parameterless_module_changes_structure() -> None:
    torch.manual_seed(0)
    a = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    b = nn.Sequential(nn.Linear(2, 2), nn.Tanh())
    b.load_state_dict(a.state_dict())
    assert fp(a).structure_digest != fp(b).structure_digest
    assert fp(a).state_digest == fp(b).state_digest


# ----------------------------------------------------------------- aliasing


def test_tied_vs_equal_copied_weights_differ_structurally() -> None:
    tied, copied = Net(tie=True), Net(copy_tied=True)
    assert torch.equal(tied.head.weight, copied.head.weight)
    it, ic = fp(tied), fp(copied)
    assert it.structure_digest != ic.structure_digest
    assert it.parameter_tensors == ic.parameter_tensors - 1  # tied weight counted once


def test_tied_parameter_is_hashed_once_under_a_stable_name() -> None:
    tied = Net(tie=True)
    before = fp(tied)
    with torch.no_grad():
        tied.head.weight.add_(1.0)  # also changes embed.weight: same object
    after = fp(tied)
    assert after.state_digest != before.state_digest
    assert after.structure_digest == before.structure_digest


def test_shared_module_vs_equal_copy_differ_structurally() -> None:
    torch.manual_seed(0)
    block = nn.Linear(2, 2)
    shared = nn.Sequential(block, block)
    copied = nn.Sequential(block, copy.deepcopy(block))
    assert fp(shared).structure_digest != fp(copied).structure_digest


def test_alias_topology_not_object_identity_determines_the_digest() -> None:
    # Two independently built tied models: different objects, same topology.
    assert fp(Net(tie=True)) == fp(Net(tie=True))


# ----------------------------------------------------------------- unsupported


class Holder(nn.Module):
    def __init__(self, tensor: torch.Tensor, *, as_param: bool = False) -> None:
        super().__init__()
        if as_param:
            self.p = nn.Parameter(tensor, requires_grad=False)
        else:
            self.register_buffer("b", tensor)


def _quantized() -> torch.Tensor:
    # torch >= 2.14 deprecates creating quantized tensors (UserWarning). This only
    # silences test *setup*; the fingerprint must still reject the tensor.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"torch\.quantize_per_tensor", category=UserWarning
        )
        return torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.quint8)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Holder(torch.empty(2, device="meta")), "meta"),
        (lambda: Holder(torch.ones(3).to_sparse()), "layout"),
        (lambda: Holder(_quantized()), "quantized"),
        (lambda: nn.LazyLinear(3), "tensor subclass"),
    ],
)
def test_unsupported_tensors_fail_clearly(make: object, match: str) -> None:
    model = make()  # type: ignore[operator]
    with pytest.raises(FingerprintError, match=match):
        fingerprint_model(model)


def test_extra_state_fails_clearly() -> None:
    class WithExtra(nn.Module):
        def get_extra_state(self) -> object:
            return {"vocab": ["a", "b"]}

        def set_extra_state(self, state: object) -> None:
            pass

    with pytest.raises(FingerprintError, match="get_extra_state"):
        fingerprint_model(nn.Sequential(WithExtra()))


def test_tensor_subclass_fails_clearly() -> None:
    class Tagged(torch.Tensor):
        pass

    m = nn.Module()
    m.register_buffer("t", torch.ones(2).as_subclass(Tagged))
    with pytest.raises(FingerprintError, match="tensor subclass"):
        fingerprint_model(m)


def test_rejects_non_modules() -> None:
    with pytest.raises(TypeError):
        fingerprint_model(torch.ones(2))  # type: ignore[arg-type]


# ----------------------------------------------------------------- read-only


def test_fingerprinting_does_not_modify_model_or_rng() -> None:
    m = Net()
    state = {k: v.clone() for k, v in m.state_dict().items()}
    grads = {n: p.requires_grad for n, p in m.named_parameters()}
    training = m.training
    torch.manual_seed(123)
    fingerprint_model(m)
    after_draw = torch.rand(4)
    torch.manual_seed(123)
    expected = torch.rand(4)
    assert torch.equal(after_draw, expected)
    assert all(torch.equal(state[k], v) for k, v in m.state_dict().items())
    assert grads == {n: p.requires_grad for n, p in m.named_parameters()}
    assert m.training == training


def test_no_cache_in_place_mutation_is_always_seen() -> None:
    m = nn.Linear(2, 2)
    seen = {fp(m).state_digest}
    for _ in range(3):
        with torch.no_grad():
            m.weight.add_(1)
        seen.add(fp(m).state_digest)
    assert len(seen) == 4


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.int64,
        torch.uint8,
        torch.bool,
        torch.complex64,
        torch.float8_e4m3fn,
    ],
)
def test_fast_byte_path_equals_reference_storage_bytes(dtype: torch.dtype) -> None:
    t = (torch.arange(7) % 3).to(dtype)
    reference = bytes(t.clone().untyped_storage())  # slow but unambiguous
    assert tensor_bytes(t) == reference
    assert tensor_bytes(t[::2]) == bytes(t[::2].contiguous().untyped_storage())
    assert tensor_bytes(t[:0]) == b""


def test_golden_state_digest() -> None:
    # Pins algorithm v1 end to end (as corrected before release to hash all buffers,
    # ADR-020). Any further change needs an ADR and an algorithm_version bump.
    m = nn.Module()
    m.register_buffer("x", torch.tensor([1.0, -2.5, 0.0], dtype=torch.float32))
    m.register_parameter("w", nn.Parameter(torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)))
    m.register_buffer("n", torch.tensor([7], dtype=torch.int64), persistent=False)
    i = fingerprint_model(m)
    assert i.state_digest == (
        "sha256:490358f91867e94c8c1051528c820e4746ec876c4f9d39b386ead5a41a2135c6"
    )
    assert i.structure_digest == (
        "sha256:acdb1dcec4282dec661fe05052c21348ac135beb93ef6bce6cd3dcb5e84bb1e1"
    )


def test_state_digest_matches_the_documented_algorithm() -> None:
    # Independent re-implementation from the docstring, using the slow reference bytes.
    import hashlib
    import json

    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    h = hashlib.sha256(b"beyondnn.model_state/v1\n")
    m.register_buffer("scratch", torch.tensor([3.0, 4.0]), persistent=False)
    persistent_names = set(m.state_dict())
    entries = [(n, "parameter", t) for n, t in m.named_parameters()] + [
        (n, "buffer", t) for n, t in m.named_buffers()
    ]
    for name, role, t in sorted(entries, key=lambda e: e[0]):
        data = bytes(t.detach().clone().untyped_storage())
        entry: dict[str, object] = {"name": name, "role": role}
        if role == "buffer":
            entry["persistent"] = name in persistent_names
        header = json.dumps(
            entry
            | {
                "dtype": str(t.dtype).removeprefix("torch."),
                "shape": list(t.shape),
                "nbytes": len(data),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        h.update(len(header).to_bytes(8, "little") + header + data)
    assert fingerprint_model(m).state_digest == "sha256:" + h.hexdigest()


def test_identical_bytes_with_different_dtype_differ_in_state() -> None:
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", torch.zeros(2, dtype=torch.int32))
    b.register_buffer("x", torch.zeros(2, dtype=torch.float32))  # same 8 zero bytes
    assert tensor_bytes(a.get_buffer("x")) == tensor_bytes(b.get_buffer("x"))
    assert fp(a).state_digest != fp(b).state_digest


def test_shared_parameterless_module_is_structural() -> None:
    # Only the module alias groups can see this: there are no tensors to tie.
    act = nn.ReLU()
    torch.manual_seed(0)
    shared = nn.Sequential(nn.Linear(2, 2), act, act)
    separate = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.ReLU())
    separate.load_state_dict(shared.state_dict())
    assert fp(shared).structure_digest != fp(separate).structure_digest
    assert fp(shared).state_digest == fp(separate).state_digest


# ----------------------------------------------------------------- non-persistent buffers


class UsesNonPersistent(nn.Module):
    """Test-local: a non-persistent buffer that takes part in forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.register_buffer("offset", torch.tensor([0.5, -1.0, 2.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(x) + self.get_buffer("offset")
        return out


def _nonpersistent_model() -> UsesNonPersistent:
    torch.manual_seed(0)
    return UsesNonPersistent()


def test_same_non_persistent_buffer_gives_same_fingerprint() -> None:
    assert fp(_nonpersistent_model()) == fp(_nonpersistent_model())


def test_non_persistent_buffer_change_changes_state_and_restore_returns_it() -> None:
    m = _nonpersistent_model()
    offset = m.get_buffer("offset")
    assert "offset" not in m.state_dict()  # really non-persistent
    before = fp(m)
    original = offset.clone()
    offset[1] = 7.0
    changed = fp(m)
    assert changed.state_digest != before.state_digest
    assert changed.structure_digest == before.structure_digest
    offset.copy_(original)
    assert fp(m) == before


def test_behaviour_change_through_non_persistent_buffer_changes_identity() -> None:
    # Regression guard: output changes, so identity must change too.
    m = _nonpersistent_model()
    x = torch.ones(2, 3)
    out_before, id_before = m(x), fp(m)
    m.get_buffer("offset").add_(1.0)
    assert not torch.equal(m(x), out_before)
    assert fp(m).state_digest != id_before.state_digest


def test_buffer_persistence_is_structural() -> None:
    a, b = nn.Module(), nn.Module()
    a.register_buffer("x", torch.ones(2), persistent=True)
    b.register_buffer("x", torch.ones(2), persistent=False)
    ia, ib = fp(a), fp(b)
    assert ia.structure_digest != ib.structure_digest
    assert ia.state_digest != ib.state_digest  # persistence is in the state header too


def test_non_persistent_buffers_are_counted_and_checked() -> None:
    m = _nonpersistent_model()
    i = fp(m)
    assert (i.buffer_tensors, i.buffer_elements) == (1, 3)
    m.register_buffer("scratch", torch.empty(2, device="meta"), persistent=False)
    with pytest.raises(FingerprintError, match="meta"):
        fingerprint_model(m)


def test_none_buffer_slots_are_excluded() -> None:
    a, b = nn.Linear(2, 2), nn.Linear(2, 2)
    b.load_state_dict(a.state_dict())
    b.register_buffer("unset", None)
    assert fp(a) == fp(b)


def test_persistence_helper_agrees_with_state_dict() -> None:
    # Compatibility check for the private torch field on the tested torch versions.
    from beyondnn.provenance.fingerprint import _non_persistent_buffer_names

    m = nn.Sequential(nn.BatchNorm1d(2), _nonpersistent_model())
    m.register_buffer("p", torch.zeros(1))
    m.register_buffer("q", torch.zeros(1), persistent=False)
    reported = set()
    for path, module in m.named_modules():
        for name in _non_persistent_buffer_names(module):
            reported.add(f"{path}.{name}" if path else name)
    all_buffers = {n for n, _ in m.named_buffers()}
    assert reported == all_buffers - set(m.state_dict())
    assert reported == {"q", "1.offset"}


def test_missing_persistence_field_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    m = nn.Linear(2, 2)
    monkeypatch.delattr(m, "_non_persistent_buffers_set")
    with pytest.raises(FingerprintError, match="buffer persistence cannot be determined"):
        fingerprint_model(m)
