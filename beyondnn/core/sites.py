"""Deterministic, strict resolution of module-path patterns to module sites (M1.4).

Pattern language (wildcards operate on dot-separated path *segments*, never on
characters):

=================  ==========================================================
Pattern            Meaning
=================  ==========================================================
``blocks.0.attn``  exactly that path
``blocks.*``       ``*`` = exactly one segment (``blocks.0``, ``blocks.1``)
``blocks.**``      ``**`` = zero or more segments (``blocks``, ``blocks.0``,
                   ``blocks.0.attn``, ...)
``**.attn``        any path whose last segment is ``attn``
``**``             every named descendant module (never the root)
``""``             the root module, and only the root
=================  ==========================================================

Rules:

* A segment is either a literal (non-empty, no ``*``, no whitespace), ``*``, or
  ``**``. Partial wildcards (``block*``), ``***``, empty segments (``a..b``,
  ``.a``, ``a.``) and consecutive ``**`` segments are invalid.
* The root module (path ``""``) is selected only by the pattern ``""``. No
  wildcard ever selects it, so a wildcard can never hook the whole model.
* Resolution is strict: every pattern must match at least one module, or
  :class:`UnmatchedPatternError` is raised (nothing is returned).
* Results follow the model's module traversal order
  (``named_modules(remove_duplicate=False)``), independent of pattern order, and
  each path appears once however many patterns select it.
* Sites are paths, not objects: a module registered under two names (aliases)
  yields two sites. Repeated *calls* of one path during forward are a separate
  concern (call indices, M1.5/M1.6) and are not handled here.
* Resolution never runs forward, installs hooks, reads tensors, or touches RNG.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from torch import nn

from beyondnn.schema import SchemaError, Site, SiteIO

__all__ = [
    "InvalidPatternError",
    "ResolvedSite",
    "SiteResolutionError",
    "UnmatchedPatternError",
    "parse_pattern",
    "resolve_sites",
]

_STAR = "*"
_GLOBSTAR = "**"
_MAX_SUGGESTIONS = 5


class SiteResolutionError(ValueError):
    """Base class for site-resolution failures."""


class InvalidPatternError(SiteResolutionError):
    """A pattern is not valid in the path-pattern language."""


class UnmatchedPatternError(SiteResolutionError):
    """One or more valid patterns matched no module. ``patterns`` lists them."""

    def __init__(self, message: str, patterns: tuple[str, ...]) -> None:
        super().__init__(message)
        self.patterns = patterns


@dataclass(frozen=True, slots=True)
class ResolvedSite:
    """A schema :class:`Site` bound to the live module at that path (not serialisable).

    ``aliases`` are the other paths under which the same module object is
    registered, in traversal order (empty for ordinary modules).
    """

    site: Site
    module: nn.Module
    aliases: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return self.site.module


def parse_pattern(pattern: str) -> tuple[str, ...]:
    """Validate ``pattern`` and return its segments (``()`` for the root pattern ``""``)."""
    if not isinstance(pattern, str):
        raise InvalidPatternError(f"pattern must be a str, got {type(pattern).__name__}")
    if pattern == "":
        return ()
    segments = tuple(pattern.split("."))
    for segment in segments:
        if segment == "":
            raise InvalidPatternError(f"empty path segment in pattern {pattern!r}")
        if any(ch.isspace() for ch in segment):
            raise InvalidPatternError(f"whitespace in pattern {pattern!r}")
        if "*" in segment and segment not in (_STAR, _GLOBSTAR):
            raise InvalidPatternError(
                f"wildcards must be a whole segment ('*' or '**'), got {segment!r} "
                f"in pattern {pattern!r}"
            )
    for first, second in pairwise(segments):
        if first == second == _GLOBSTAR:
            raise InvalidPatternError(f"consecutive '**' segments in pattern {pattern!r}")
    return segments


def _matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """Segment matcher. ``pattern`` must be non-empty and ``path`` non-root."""
    if not pattern:
        return not path
    head, rest = pattern[0], pattern[1:]
    if head == _GLOBSTAR:
        return any(_matches(rest, path[i:]) for i in range(len(path) + 1))
    if not path:
        return False
    return (head == _STAR or head == path[0]) and _matches(rest, path[1:])


def _pattern_matches(pattern: tuple[str, ...], path: str) -> bool:
    if path == "":
        return pattern == ()  # only the explicit root pattern selects the root
    if pattern == ():
        return False
    return _matches(pattern, tuple(path.split(".")))


def _suggestions(pattern: str, paths: Sequence[str]) -> list[str]:
    named = [p for p in paths if p]
    return difflib.get_close_matches(pattern, named, n=_MAX_SUGGESTIONS, cutoff=0.5)


def resolve_sites(
    model: nn.Module, patterns: Sequence[str], *, io: SiteIO = SiteIO.OUTPUT
) -> tuple[ResolvedSite, ...]:
    """Resolve ``patterns`` against ``model``'s module tree.

    Returns one :class:`ResolvedSite` per selected path, in module traversal
    order, each with ``Site(module=path, io=io)``. Raises
    :class:`InvalidPatternError` or :class:`UnmatchedPatternError`; never returns
    a partial selection.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("resolve_sites() requires a torch.nn.Module")
    if isinstance(patterns, str):
        raise TypeError("patterns must be a sequence of strings, e.g. ['blocks.*'], not a str")
    if not isinstance(io, SiteIO):
        raise TypeError("io must be a SiteIO")
    patterns = tuple(patterns)
    if not patterns:
        raise SiteResolutionError("no patterns given; select at least one module path")

    errors: list[str] = []
    parsed: list[tuple[str, ...]] = []
    for pattern in patterns:
        try:
            parsed.append(parse_pattern(pattern))
        except InvalidPatternError as exc:
            errors.append(str(exc))
    if errors:
        raise InvalidPatternError("; ".join(errors))

    # One traversal; aliases preserved (a module under two names appears twice).
    entries = list(model.named_modules(remove_duplicate=False))
    paths = [path for path, _ in entries]
    by_object: dict[int, list[str]] = {}
    for path, module in entries:
        by_object.setdefault(id(module), []).append(path)  # grouping only

    hit = [False] * len(patterns)
    selected: list[ResolvedSite] = []
    for path, module in entries:
        matched = False
        for index, segments in enumerate(parsed):
            if _pattern_matches(segments, path):
                hit[index] = True
                matched = True
        if not matched:
            continue
        try:
            site = Site(module=path, io=io)
        except SchemaError as exc:
            raise SiteResolutionError(f"module path {path!r} cannot be a site: {exc}") from None
        aliases = tuple(p for p in by_object[id(module)] if p != path)
        selected.append(ResolvedSite(site=site, module=module, aliases=aliases))

    unmatched = tuple(p for p, ok in zip(patterns, hit, strict=True) if not ok)
    if unmatched:
        details = []
        for pattern in unmatched:
            near = _suggestions(pattern, paths)
            hint = f" (similar paths: {', '.join(near)})" if near else ""
            details.append(f"{pattern!r} matched no module{hint}")
        raise UnmatchedPatternError(
            f"{'; '.join(details)}. The model has {len(paths)} module paths "
            "(root is selected only by '').",
            unmatched,
        )
    return tuple(selected)
