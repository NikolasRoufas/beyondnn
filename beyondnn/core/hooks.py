"""Safe, scoped observation of module executions (M1.5). Internal.

:class:`HookSession` installs hooks on ``__enter__`` and removes exactly the hooks
it created on ``__exit__`` (also after errors). It emits ephemeral
:class:`HookEvent` objects to a caller-supplied sink and keeps no reference to
observed values. It never modifies arguments or outputs: every hook returns
``None``, and values are passed on without detaching, cloning, or moving them.

What a hook event proves: *this module object executed and this value crossed
its boundary*. It does not prove which Python alias reached the object, what a
value means, or that the module caused any prediction.

Hook layout (per session, per physical module object):

* every observed module: one pre-hook (claims the invocation's call index; emits
  INPUT if selected) and one ``always_call`` cleanup forward hook (releases the
  invocation; never raises, never emits);
* modules observed at OUTPUT: one more ordinary forward hook, which runs only when
  the call succeeded and emits OUTPUT;
* the root model: one prepended pass-start pre-hook and one ``always_call``
  pass-end forward hook for pass bookkeeping (never emit).

Selecting a module through several patterns, paths, or both INPUT and OUTPUT
never installs duplicate hooks.

Indices:

* ``pass_index``: the n-th invocation of the root model during the session. A
  root invocation that raises still consumes its index. A root that calls itself
  re-entrantly does not start a new pass. Executions outside any root invocation
  (e.g. ``model.block(x)`` called directly) get ``pass_index == -1``.
* ``call_index``: the n-th invocation of that physical module within the pass
  (reset at every new pass). It is claimed by the pre-hook, so an invocation
  that raises still consumes it, and the INPUT and OUTPUT events of one
  invocation share it (a per-module stack keeps recursive calls paired). Calls
  outside a root pass are counted separately and never reset.

Aliases: a module object registered under several paths cannot be attributed to
one path by hooks. By default (``AliasPolicy.REFUSE``) selecting such a module
raises :class:`AliasSiteAmbiguityError`. ``AliasPolicy.GROUP`` observes it once
per execution with ``paths`` listing every alias and ``path_specific=False``.

Not supported in v0: changing the module tree while a session is active,
concurrent use from several threads (not thread-safe), ``torch.compile``.
Pre-hooks and forward hooks registered by the user earlier run before BeyondNN's
(except the prepended pass-start hook), so observed values are those after any
user hook modifications.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType, TracebackType
from typing import Any

from torch import nn
from torch.utils.hooks import RemovableHandle

from beyondnn.schema import Site, SiteIO

from .sites import SiteResolutionError, resolve_sites

__all__ = [
    "AliasPolicy",
    "AliasSiteAmbiguityError",
    "HookEvent",
    "HookSession",
    "HookSessionError",
]

OUT_OF_PASS = -1


class HookSessionError(RuntimeError):
    """Misuse of a hook session (e.g. entering twice)."""


class AliasSiteAmbiguityError(SiteResolutionError):
    """A selected module is registered under several paths; hooks cannot tell which."""


class AliasPolicy(Enum):
    REFUSE = "refuse"
    GROUP = "group"


@dataclass(frozen=True, slots=True, eq=False)
class HookEvent:
    """One observed crossing of a module boundary. Ephemeral: not a record, no id.

    ``args``/``kwargs`` are set for INPUT (``kwargs`` is a read-only view),
    ``output`` for OUTPUT. Values are the live objects PyTorch passed.
    """

    io: SiteIO
    paths: tuple[str, ...]
    path_specific: bool
    pass_index: int
    call_index: int
    module: nn.Module
    args: tuple[Any, ...] | None = None
    kwargs: Mapping[str, Any] | None = None
    output: Any = None

    @property
    def site(self) -> Site:
        """The schema site. Only defined for path-specific events."""
        if not self.path_specific:
            raise AliasSiteAmbiguityError(
                f"execution of an aliased module cannot be attributed to one of {self.paths}"
            )
        return Site(module=self.paths[0], io=self.io)

    @property
    def in_root_pass(self) -> bool:
        return self.pass_index != OUT_OF_PASS


@dataclass
class _Target:
    module: nn.Module
    paths: tuple[str, ...]
    path_specific: bool
    observe_input: bool = False
    observe_output: bool = False
    stack: list[tuple[int, int]] = field(default_factory=list)  # (pass, call) per open call


class HookSession:
    """Observe selected module executions of ``model`` inside a ``with`` block.

    ``outputs`` / ``inputs`` are site patterns (see :mod:`beyondnn.core.sites`),
    resolved against the live model on entry. ``sink`` receives each
    :class:`HookEvent` synchronously; if it raises, the exception propagates out
    of the model call and bookkeeping stays consistent. Single use.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        sink: Callable[[HookEvent], None],
        outputs: Sequence[str] = (),
        inputs: Sequence[str] = (),
        alias_policy: AliasPolicy = AliasPolicy.REFUSE,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("HookSession requires a torch.nn.Module")
        if not callable(sink):
            raise TypeError("sink must be callable")
        for name, value in (("outputs", outputs), ("inputs", inputs)):
            if isinstance(value, str):
                raise TypeError(f"{name} must be a sequence of patterns, not a str")
        if not outputs and not inputs:
            raise SiteResolutionError("select at least one input or output site")
        if not isinstance(alias_policy, AliasPolicy):
            raise TypeError("alias_policy must be an AliasPolicy")
        self._model = model
        self._sink = sink
        self._outputs = tuple(outputs)
        self._inputs = tuple(inputs)
        self._alias_policy = alias_policy
        self._handles: list[RemovableHandle] = []
        self._state = "new"
        self._next_pass = 0
        self._current_pass = OUT_OF_PASS
        self._root_depth = 0
        self._pass_calls: dict[int, int] = {}
        self._out_of_pass_calls: dict[int, int] = {}

    # ------------------------------------------------------------ lifecycle

    @property
    def active(self) -> bool:
        return self._state == "active"

    @property
    def installed_hooks(self) -> int:
        return len(self._handles)

    def __enter__(self) -> HookSession:
        if self._state != "new":
            raise HookSessionError("a HookSession can be entered only once")
        targets = self._plan()
        self._state = "active"
        try:
            self._install(targets)
        except BaseException:
            self._remove_all()
            self._state = "closed"
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._remove_all()
        self._state = "closed"

    def _remove_all(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    # ------------------------------------------------------------ planning

    def _plan(self) -> list[_Target]:
        all_paths: dict[int, list[str]] = {}
        for path, module in self._model.named_modules(remove_duplicate=False):
            all_paths.setdefault(id(module), []).append(path)  # grouping only

        targets: dict[int, _Target] = {}
        for patterns, io in ((self._inputs, SiteIO.INPUT), (self._outputs, SiteIO.OUTPUT)):
            if not patterns:
                continue
            for resolved in resolve_sites(self._model, patterns, io=io):
                paths = tuple(all_paths[id(resolved.module)])
                aliased = len(paths) > 1
                if aliased and self._alias_policy is AliasPolicy.REFUSE:
                    raise AliasSiteAmbiguityError(
                        f"site {resolved.path!r} is a module registered under several paths "
                        f"{paths}; hooks attach to the object and cannot tell which path was "
                        "used. Select a non-aliased module, or observe the alias group."
                    )
                target = targets.get(id(resolved.module))
                if target is None:
                    target = _Target(
                        module=resolved.module,
                        paths=paths if aliased else (resolved.path,),
                        path_specific=not aliased,
                    )
                    targets[id(resolved.module)] = target
                if io is SiteIO.INPUT:
                    target.observe_input = True
                else:
                    target.observe_output = True
        return list(targets.values())

    def _install(self, targets: list[_Target]) -> None:
        root = self._model
        self._handles.append(
            root.register_forward_pre_hook(self._pass_start, with_kwargs=True, prepend=True)
        )
        for target in targets:
            self._handles.append(
                target.module.register_forward_pre_hook(self._pre_hook(target), with_kwargs=True)
            )
            if target.observe_output:
                self._handles.append(
                    target.module.register_forward_hook(self._output_hook(target), with_kwargs=True)
                )
            self._handles.append(
                target.module.register_forward_hook(
                    self._cleanup_hook(target), with_kwargs=True, always_call=True
                )
            )
        self._handles.append(
            root.register_forward_hook(self._pass_end, with_kwargs=True, always_call=True)
        )

    # ------------------------------------------------------------ bookkeeping hooks

    def _pass_start(self, module: nn.Module, args: Any, kwargs: Any) -> None:
        if self._root_depth == 0:
            self._current_pass = self._next_pass
            self._next_pass += 1
            self._pass_calls.clear()
        self._root_depth += 1

    def _pass_end(self, module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
        # always_call: must never raise.
        if self._root_depth > 0:
            self._root_depth -= 1
            if self._root_depth == 0:
                self._current_pass = OUT_OF_PASS

    def _claim(self, target: _Target) -> tuple[int, int]:
        key = id(target.module)
        counts = self._out_of_pass_calls if self._root_depth == 0 else self._pass_calls
        call_index = counts.get(key, 0)
        counts[key] = call_index + 1
        return (self._current_pass if self._root_depth else OUT_OF_PASS), call_index

    # ------------------------------------------------------------ observation hooks

    def _pre_hook(self, target: _Target) -> Callable[..., None]:
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            pass_index, call_index = self._claim(target)
            target.stack.append((pass_index, call_index))
            if target.observe_input:
                self._sink(
                    HookEvent(
                        io=SiteIO.INPUT,
                        paths=target.paths,
                        path_specific=target.path_specific,
                        pass_index=pass_index,
                        call_index=call_index,
                        module=module,
                        args=args,
                        kwargs=MappingProxyType(kwargs),
                    )
                )

        return hook

    def _output_hook(self, target: _Target) -> Callable[..., None]:
        def hook(module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
            pass_index, call_index = target.stack[-1]
            self._sink(
                HookEvent(
                    io=SiteIO.OUTPUT,
                    paths=target.paths,
                    path_specific=target.path_specific,
                    pass_index=pass_index,
                    call_index=call_index,
                    module=module,
                    output=output,
                )
            )

        return hook

    def _cleanup_hook(self, target: _Target) -> Callable[..., None]:
        def hook(module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
            # always_call: runs once per invocation, on success or failure; never raises.
            if target.stack:
                target.stack.pop()

        return hook
