"""M1.4: deterministic, strict module site resolution. The tables document the language."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer
from beyondnn.core.sites import (
    InvalidPatternError,
    ResolvedSite,
    SiteResolutionError,
    UnmatchedPatternError,
    parse_pattern,
    resolve_sites,
)
from beyondnn.schema import Site, SiteIO


class _C(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d = nn.Linear(2, 2)


class _B(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = _C()


class Tree(nn.Module):
    """Paths in traversal order: '', a, a.0, a.1, b, b.c, b.c.d, pool."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.b = _B()
        self.pool = nn.ReLU()


ALL = ["a", "a.0", "a.1", "b", "b.c", "b.c.d", "pool"]


def paths(model: nn.Module, patterns: list[str], **kw: Any) -> list[str]:
    return [r.path for r in resolve_sites(model, patterns, **kw)]


# ----------------------------------------------------------------- the language


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        # exact paths
        ("a", ["a"]),
        ("a.0", ["a.0"]),  # numeric segment
        ("b.c.d", ["b.c.d"]),
        # * = exactly one segment
        ("*", ["a", "b", "pool"]),  # never the root
        ("a.*", ["a.0", "a.1"]),  # not 'a' itself, not deeper
        ("b.*", ["b.c"]),  # not b.c.d
        ("*.c", ["b.c"]),
        ("*.*", ["a.0", "a.1", "b.c"]),
        ("b.*.d", ["b.c.d"]),
        # ** = zero or more segments
        ("**", ALL),  # every named descendant, never the root
        ("a.**", ["a", "a.0", "a.1"]),  # zero segments includes 'a'
        ("b.**", ["b", "b.c", "b.c.d"]),
        ("**.d", ["b.c.d"]),
        ("**.0", ["a.0"]),
        ("**.c.**", ["b.c", "b.c.d"]),
        ("b.**.d", ["b.c.d"]),  # zero or more between
        ("**.*", ALL),
        # root
        ("", [""]),
    ],
)
def test_pattern_semantics(pattern: str, expected: list[str]) -> None:
    assert paths(Tree(), [pattern]) == expected


@pytest.mark.parametrize(
    "pattern",
    [
        "a..b",
        ".a",
        "a.",
        ".",
        "***",
        "block*",
        "*block",
        "a.***.b",
        "a*.b",
        "**.**",
        "a.**.**.b",
        "a b",
        " a",
        "a.\tb",
    ],
)
def test_invalid_patterns_are_rejected(pattern: str) -> None:
    with pytest.raises(InvalidPatternError):
        parse_pattern(pattern)
    with pytest.raises(InvalidPatternError):
        resolve_sites(Tree(), [pattern])


def test_parse_pattern_segments() -> None:
    assert parse_pattern("") == ()
    assert parse_pattern("blocks.*.attn") == ("blocks", "*", "attn")
    assert parse_pattern("**.fc_in") == ("**", "fc_in")
    with pytest.raises(InvalidPatternError):
        parse_pattern(3)  # type: ignore[arg-type]


def test_wildcards_never_select_the_root() -> None:
    for pattern in ("*", "**", "**.*"):
        assert "" not in paths(Tree(), [pattern])
    assert paths(Tree(), ["", "*"]) == ["", "a", "b", "pool"]


def test_star_never_crosses_a_separator() -> None:
    assert paths(Tree(), ["*"]) == ["a", "b", "pool"]
    assert "b.c.d" not in paths(Tree(), ["b.*"])
    assert "a" not in paths(Tree(), ["a.*"])


# ----------------------------------------------------------------- strictness


def test_unmatched_pattern_raises_and_names_it() -> None:
    with pytest.raises(UnmatchedPatternError) as info:
        resolve_sites(Tree(), ["a.*", "b.zzz"])
    assert info.value.patterns == ("b.zzz",)
    assert "'b.zzz' matched no module" in str(info.value)


def test_every_unmatched_pattern_is_reported() -> None:
    with pytest.raises(UnmatchedPatternError) as info:
        resolve_sites(Tree(), ["nope", "a", "b.*.x"])
    assert info.value.patterns == ("nope", "b.*.x")


def test_unmatched_error_suggests_nearby_paths() -> None:
    with pytest.raises(UnmatchedPatternError, match=r"similar paths: .*blocks\.0\.attn"):
        resolve_sites(TinyTransformer(), ["blocks.0.atn"])


def test_invalid_patterns_are_reported_before_matching() -> None:
    with pytest.raises(InvalidPatternError) as info:
        resolve_sites(Tree(), ["a", "a..b", "block*"])
    assert "a..b" in str(info.value)
    assert "block*" in str(info.value)


@pytest.mark.parametrize("bad", ["a", ""])
def test_a_bare_string_is_not_a_pattern_list(bad: str) -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        resolve_sites(Tree(), bad)


def test_empty_pattern_list_is_rejected() -> None:
    with pytest.raises(SiteResolutionError, match="no patterns"):
        resolve_sites(Tree(), [])


def test_error_hierarchy() -> None:
    assert issubclass(InvalidPatternError, SiteResolutionError)
    assert issubclass(UnmatchedPatternError, SiteResolutionError)
    assert not issubclass(InvalidPatternError, UnmatchedPatternError)


def test_argument_types() -> None:
    with pytest.raises(TypeError):
        resolve_sites(torch.ones(2), ["a"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve_sites(Tree(), ["a"], io="output")  # type: ignore[arg-type]


# ----------------------------------------------------------------- ordering and dedup


def test_order_is_traversal_order_not_pattern_order() -> None:
    assert paths(Tree(), ["pool", "a.0", "b.c"]) == ["a.0", "b.c", "pool"]
    m = TinyTransformer()
    assert paths(m, ["blocks.*", "lm_head"]) == paths(m, ["lm_head", "blocks.*"])


def test_overlapping_patterns_yield_each_path_once() -> None:
    m = TinyTransformer()
    got = paths(m, ["blocks.0.attn", "blocks.*.attn", "**.attn"])
    assert got == ["blocks.0.attn", "blocks.1.attn"]
    assert paths(Tree(), ["a", "a", "*", "**"]) == ALL


# ----------------------------------------------------------------- bindings


def test_resolved_sites_bind_schema_sites_to_live_modules() -> None:
    model = Tree()
    (resolved,) = resolve_sites(model, ["b.c.d"])
    assert isinstance(resolved, ResolvedSite)
    assert resolved.site == Site(module="b.c.d", io=SiteIO.OUTPUT)
    assert resolved.module is model.b.c.d
    assert resolved.path == "b.c.d"
    assert resolved.aliases == ()


def test_input_and_output_sites_are_distinct() -> None:
    model = Tree()
    (out,) = resolve_sites(model, ["pool"])
    (inp,) = resolve_sites(model, ["pool"], io=SiteIO.INPUT)
    assert out.module is inp.module
    assert out.site != inp.site
    assert inp.site.io is SiteIO.INPUT


def test_root_binding() -> None:
    model = Tree()
    (root,) = resolve_sites(model, [""])
    assert root.module is model
    assert root.site == Site(module="")


class Aliased(nn.Module):
    """Test-local: one module object registered under two names."""

    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(2, 2)
        self.left = shared
        self.right = shared
        self.other = nn.Linear(2, 2)


def test_alias_paths_are_distinct_sites_of_one_module() -> None:
    model = Aliased()
    left, right = resolve_sites(model, ["left", "right"])
    assert (left.path, right.path) == ("left", "right")
    assert left.module is right.module is model.left
    assert left.site != right.site
    assert left.aliases == ("right",)
    assert right.aliases == ("left",)


def test_wildcards_select_every_alias_path() -> None:
    model = Aliased()
    assert paths(model, ["*"]) == ["left", "right", "other"]
    assert paths(model, ["**"]) == ["left", "right", "other"]


def test_container_and_leaf_modules_are_both_selectable() -> None:
    assert paths(Tree(), ["b", "b.c", "b.c.d"]) == ["b", "b.c", "b.c.d"]


def test_unrepresentable_module_names_fail_clearly() -> None:
    model = nn.Module()
    model.add_module("has space", nn.ReLU())
    model.add_module("ok", nn.ReLU())
    assert paths(model, ["ok"]) == ["ok"]
    with pytest.raises(SiteResolutionError, match="cannot be a site"):
        resolve_sites(model, ["*"])


# ----------------------------------------------------------------- reference models


def test_tiny_mlp_paths_and_shared_module_resolve_once() -> None:
    m = TinyMLP()
    assert paths(m, ["projection", "shared", "shared.linear", "head"]) == [
        "projection",
        "shared",
        "shared.linear",
        "head",
    ]
    # 'shared' is CALLED twice per forward, but it is one path: one site.
    assert paths(m, ["shared", "*", "**"]).count("shared") == 1


def test_tiny_cnn_paths_including_parameterless_modules() -> None:
    m = TinyCNN()
    expected = ["stem", "stem.conv", "stem.norm", "pool", "block", "block.conv", "block.norm"]
    assert paths(m, [*expected, "gap", "classifier"]) == [*expected, "gap", "classifier"]
    for name in ("pool", "gap"):
        (site,) = resolve_sites(m, [name])
        assert sum(p.numel() for p in site.module.parameters()) == 0
    assert paths(m, ["*"]) == ["stem", "pool", "block", "gap", "classifier"]


def test_tiny_transformer_patterns() -> None:
    m = TinyTransformer()
    assert paths(m, ["blocks.*"]) == ["blocks.0", "blocks.1"]
    assert paths(m, ["blocks.*.attn"]) == ["blocks.0.attn", "blocks.1.attn"]
    assert paths(m, ["**.attn"]) == ["blocks.0.attn", "blocks.1.attn"]
    assert paths(m, ["blocks.**.fc_in"]) == ["blocks.0.mlp.fc_in", "blocks.1.mlp.fc_in"]
    assert paths(m, ["token_embedding", "lm_head"]) == ["token_embedding", "lm_head"]


def test_tied_parameters_do_not_merge_module_sites() -> None:
    m = TinyTransformer()
    emb, head = resolve_sites(m, ["token_embedding", "lm_head"])
    assert emb.module is not head.module
    assert emb.module.weight is head.module.weight
    assert emb.aliases == head.aliases == ()


# ----------------------------------------------------------------- no execution, no hooks


class Exploding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("forward must not run during resolution")


def _hook_counts(model: nn.Module) -> list[int]:
    return [
        len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
        for m in model.modules()
    ]


def test_resolution_runs_nothing_and_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resolution must not register hooks")

    for name in (
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
    ):
        monkeypatch.setattr(nn.Module, name, forbidden)
    monkeypatch.setattr(nn.modules.module, "register_module_forward_hook", forbidden)

    model = Exploding()
    torch.manual_seed(11)
    rng = torch.get_rng_state().clone()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    hooks = _hook_counts(model)

    resolved = resolve_sites(model, ["**", ""])

    assert [r.path for r in resolved] == ["", "inner", "inner.0", "inner.1"]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert _hook_counts(model) == hooks
    assert model.get_buffer("inner.1.num_batches_tracked").item() == 0


def test_resolution_is_repeatable_on_tiny_models() -> None:
    for build in (TinyMLP, TinyCNN, TinyTransformer):
        m = build()
        assert paths(m, ["**"]) == paths(m, ["**"])
        assert paths(m, ["**"]) == [n for n, _ in m.named_modules(remove_duplicate=False) if n]
