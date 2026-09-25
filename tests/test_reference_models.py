"""M1.3: deterministic tiny reference models (internal testing namespace)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer, seeded_init
from beyondnn.provenance import fingerprint_model


@dataclass(frozen=True)
class Ref:
    name: str
    build: Callable[..., nn.Module]
    make_input: Callable[[], torch.Tensor]
    out_shape: tuple[int, ...]
    params: int
    paths: tuple[str, ...]


def _x_mlp() -> torch.Tensor:
    return torch.linspace(-1, 1, 8).reshape(2, 4)


def _x_cnn() -> torch.Tensor:
    return torch.linspace(-1, 1, 2 * 64).reshape(2, 1, 8, 8)


def _x_tok() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 3, 0, 31], [2, 2, 7, 30, 4, 8]])


REFS = [
    Ref("mlp", TinyMLP, _x_mlp, (2, 3), 139, ("projection", "shared", "shared.linear", "head")),
    Ref(
        "cnn",
        TinyCNN,
        _x_cnn,
        (2, 4),
        396,
        (
            "stem",
            "stem.conv",
            "stem.norm",
            "pool",
            "block",
            "block.conv",
            "block.norm",
            "gap",
            "classifier",
        ),
    ),
    Ref(
        "transformer",
        TinyTransformer,
        _x_tok,
        (2, 6, 32),
        5120,
        (
            "token_embedding",
            "position_embedding",
            "blocks.0",
            "blocks.0.ln_attn",
            "blocks.0.attn",
            "blocks.0.attn.qkv",
            "blocks.0.attn.out",
            "blocks.0.ln_mlp",
            "blocks.0.mlp",
            "blocks.0.mlp.fc_in",
            "blocks.0.mlp.fc_out",
            "blocks.1",
            "ln_final",
            "lm_head",
        ),
    ),
]
by_name = pytest.mark.parametrize("ref", REFS, ids=[r.name for r in REFS])


def _state_equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _out(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    assert isinstance(y, torch.Tensor)
    return y


# ----------------------------------------------------------------- construction


@by_name
def test_parameter_counts_are_exact_and_tiny(ref: Ref) -> None:
    model = ref.build()
    assert sum(p.numel() for p in model.parameters()) == ref.params
    assert ref.params < 20_000


@by_name
def test_critical_module_paths_exist(ref: Ref) -> None:
    paths = {name for name, _ in ref.build().named_modules()}
    assert set(ref.paths) <= paths


@by_name
def test_forward_shape_and_finiteness(ref: Ref) -> None:
    for mode in ("train", "eval"):
        model = ref.build()
        getattr(model, mode)()
        y = _out(model, ref.make_input())
        assert tuple(y.shape) == ref.out_shape
        assert torch.isfinite(y).all()


@by_name
def test_backward_gives_finite_gradients(ref: Ref) -> None:
    model = ref.build().train()
    loss = _out(model, ref.make_input()).square().mean()
    torch.autograd.backward(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ----------------------------------------------------------------- determinism / RNG


@by_name
def test_same_seed_same_model_outputs_and_fingerprint(ref: Ref) -> None:
    a, b = ref.build(seed=7).eval(), ref.build(seed=7).eval()
    assert _state_equal(a, b)
    assert torch.equal(_out(a, ref.make_input()), _out(b, ref.make_input()))
    assert fingerprint_model(a) == fingerprint_model(b)


@by_name
def test_different_seed_same_structure_different_state(ref: Ref) -> None:
    a, b = ref.build(seed=1).eval(), ref.build(seed=2).eval()
    fa, fb = fingerprint_model(a), fingerprint_model(b)
    assert fa.structure_digest == fb.structure_digest
    assert fa.state_digest != fb.state_digest
    assert not torch.equal(_out(a, ref.make_input()), _out(b, ref.make_input()))


@by_name
def test_construction_preserves_global_rng(ref: Ref) -> None:
    torch.manual_seed(2026)
    before = torch.get_rng_state().clone()
    ref.build(seed=123)
    assert torch.equal(before, torch.get_rng_state())


def test_rng_is_restored_even_if_construction_fails() -> None:
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="divisible"):
        TinyTransformer(d_model=15, n_heads=2)
    assert torch.equal(before, torch.get_rng_state())


def test_seeded_init_is_isolated_and_reproducible() -> None:
    before = torch.get_rng_state().clone()
    with seeded_init(5):
        first = torch.rand(3)
    with seeded_init(5):
        second = torch.rand(3)
    assert torch.equal(first, second)
    assert torch.equal(before, torch.get_rng_state())


# ----------------------------------------------------------------- state_dict / modes / dtype


@by_name
def test_state_dict_round_trip(ref: Ref) -> None:
    source, target = ref.build(seed=1).eval(), ref.build(seed=2).eval()
    target.load_state_dict(source.state_dict())
    x = ref.make_input()
    assert torch.equal(_out(source, x), _out(target, x))
    assert fingerprint_model(source) == fingerprint_model(target)


def test_train_and_eval_differ_for_batchnorm_and_are_distinguishable() -> None:
    model = TinyCNN()
    x = _x_cnn()
    eval_out = _out(model.eval(), x)
    before = fingerprint_model(model)
    train_out = _out(model.train(), x)  # batch statistics + running-stat update
    assert not torch.allclose(train_out, eval_out)
    assert fingerprint_model(model).state_digest != before.state_digest  # running stats moved


@by_name
def test_float64_conversion_works(ref: Ref) -> None:
    model = ref.build().to(dtype=torch.float64).eval()
    x = ref.make_input()
    if x.is_floating_point():
        x = x.double()
    y = _out(model, x)
    assert y.dtype == torch.float64
    assert torch.isfinite(y).all()


# ----------------------------------------------------------------- TinyMLP contracts


def test_mlp_shared_module_is_called_twice_per_forward() -> None:
    model = TinyMLP()
    calls: list[nn.Module] = []
    handle = model.shared.register_forward_hook(lambda m, i, o: calls.append(m))
    try:
        model(_x_mlp())
    finally:
        handle.remove()
    assert calls == [model.shared, model.shared]
    assert [n for n, m in model.named_modules() if m is model.shared] == ["shared"]


def test_mlp_keyword_only_scale_changes_the_result() -> None:
    model = TinyMLP()
    x = _x_mlp()
    assert torch.equal(model(x), model(x, scale=1.0))
    assert not torch.allclose(model(x), model(x, scale=2.0))
    with pytest.raises(TypeError):
        model(x, 2.0)


# ----------------------------------------------------------------- TinyCNN contracts


def test_cnn_structured_tuple_output() -> None:
    model = TinyCNN().eval()
    x = _x_cnn()
    out = model(x, return_features=True)
    assert isinstance(out, tuple)
    logits, features = out
    assert tuple(features.shape) == (2, 8)
    assert torch.equal(logits, _out(model, x))


def test_cnn_owns_batchnorm_buffers() -> None:
    # Only our own module paths are pinned, not torch's internal buffer names.
    buffers = [n for n, _ in TinyCNN().named_buffers()]
    assert any(n.startswith("stem.norm.") for n in buffers)
    assert any(n.startswith("block.norm.") for n in buffers)


# ----------------------------------------------------------------- TinyTransformer contracts


def test_transformer_lm_head_is_tied_to_token_embedding() -> None:
    model = TinyTransformer()
    assert model.lm_head.weight is model.token_embedding.weight


def test_transformer_tie_survives_load_and_dtype_conversion() -> None:
    target = TinyTransformer(seed=2)
    target.load_state_dict(TinyTransformer(seed=1).state_dict())
    assert target.lm_head.weight is target.token_embedding.weight
    target = target.to(dtype=torch.float64)
    assert target.lm_head.weight is target.token_embedding.weight
    assert target.get_buffer("position_ids").dtype == torch.int64
    assert target.get_buffer("causal_mask").dtype == torch.bool


def test_transformer_tie_vs_equal_copy_differs_in_structure() -> None:
    tied = TinyTransformer()
    copied = TinyTransformer()
    copied.lm_head.weight = nn.Parameter(copied.token_embedding.weight.detach().clone())
    x = _x_tok()
    assert torch.equal(_out(tied, x), _out(copied, x))  # same function...
    ft, fc = fingerprint_model(tied), fingerprint_model(copied)
    assert ft.structure_digest != fc.structure_digest  # ...different model
    assert ft.parameter_tensors == fc.parameter_tensors - 1


def test_transformer_structured_dict_output() -> None:
    model = TinyTransformer()
    out = model(_x_tok(), return_dict=True)
    assert isinstance(out, dict)
    assert set(out) == {"logits", "hidden_states"}
    assert tuple(out["hidden_states"].shape) == (2, 6, 16)
    assert torch.equal(out["logits"], _out(model, _x_tok()))


def test_transformer_non_persistent_buffers_are_real() -> None:
    model = TinyTransformer()
    assert {"position_ids", "causal_mask"} <= {n for n, _ in model.named_buffers()}
    assert not {"position_ids", "causal_mask"} & set(model.state_dict())


def test_transformer_causal_mask_participates_and_is_fingerprinted() -> None:
    model = TinyTransformer()
    x = _x_tok()
    # Causality: changing the last token leaves earlier positions unchanged.
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32
    assert torch.equal(_out(model, x)[:, :-1], _out(model, changed)[:, :-1])
    before_out, before_id = _out(model, x), fingerprint_model(model)
    model.get_buffer("causal_mask").fill_(True)  # disable causality
    assert not torch.allclose(_out(model, x), before_out)
    after = fingerprint_model(model)
    assert after.state_digest != before_id.state_digest
    assert after.structure_digest == before_id.structure_digest


def test_transformer_position_ids_participate() -> None:
    model = TinyTransformer()
    before = _out(model, _x_tok())
    model.get_buffer("position_ids").copy_(torch.arange(8).flip(0))
    assert not torch.allclose(_out(model, _x_tok()), before)


def test_transformer_input_validation() -> None:
    model = TinyTransformer()
    with pytest.raises(TypeError, match="integer"):
        model(torch.zeros(1, 3))
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.zeros(1, 9, dtype=torch.int64))


# ----------------------------------------------------------------- namespace


def test_reference_models_are_not_public_api() -> None:
    import beyondnn

    for name in ("TinyMLP", "TinyCNN", "TinyTransformer", "models", "_testing"):
        assert name not in beyondnn.__all__
    assert not hasattr(beyondnn, "TinyMLP")


def test_known_limitation_plain_hyperparameters_are_not_fingerprinted() -> None:
    # Documents a KNOWN limitation of FULL v1 (ADR-020: Python attributes and code
    # are not hashed). n_heads is a plain attribute, parameter shapes are identical,
    # so identity is equal although the computed function differs. If this test
    # starts failing, the fingerprint gained attribute coverage: update the ADR.
    a, b = TinyTransformer(n_heads=2), TinyTransformer(n_heads=4)
    assert fingerprint_model(a) == fingerprint_model(b)
    assert not torch.allclose(_out(a, _x_tok()), _out(b, _x_tok()))
