"""TD: Tracr programs with decoys (Phase 7.75; docs/PHASE_7_75_PLAN.md §3).

Every program has a categorical output ``out`` computed by a ``SequenceMap`` whose function is
written in the program source. Each intermediate variable is labelled here, from the source,
as ``used`` (the output function depends on it, injectively) or ``decoy`` (the output function
reads it but provably ignores it). Ground truth therefore comes from the program text, never
from an intervention on any model. Components (heads, MLPs) are mapped to variables by the
residual dimensions their output weights write (structural, weight-based; ``td_bench.py``).

Runs in the InterpBench environment (Tracr fork + circuits-benchmark).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase7_5" / "artifacts" / "cb"))

from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase  # noqa: E402
from tracr.rasp import rasp  # noqa: E402


def shift(sop: rasp.SOp, offset: int, name: str) -> rasp.SOp:
    """The value of ``sop`` ``offset`` positions back (one attention head)."""
    select = rasp.Select(rasp.indices, rasp.indices, lambda k, q, o=offset: k == q - o)
    return rasp.Aggregate(select, sop, default=None).named(name)


def first(x: Any, y: Any) -> Any:
    return x


def second(x: Any, y: Any) -> Any:
    return y


def pair(x: Any, y: Any) -> Any:
    return f"{x}{y}"


@dataclass(frozen=True)
class Program:
    name: str
    build: Callable[[], tuple[rasp.SOp, dict[str, str]]]  # -> (output, {variable: used|decoy})
    vocab: frozenset[str]
    max_seq_len: int  # including BOS
    description: str


def _copy(u: int, d: int) -> Callable[[], tuple[rasp.SOp, dict[str, str]]]:
    def build() -> tuple[rasp.SOp, dict[str, str]]:
        x = shift(rasp.tokens, u, "x_used")
        y = shift(rasp.tokens, d, "y_decoy")
        return rasp.SequenceMap(first, x, y).named("out"), {"x_used": "used", "y_decoy": "decoy"}

    return build


def _copy_swap(u: int, d: int) -> Callable[[], tuple[rasp.SOp, dict[str, str]]]:
    def build() -> tuple[rasp.SOp, dict[str, str]]:
        y = shift(rasp.tokens, d, "y_decoy")
        x = shift(rasp.tokens, u, "x_used")
        return rasp.SequenceMap(second, y, x).named("out"), {"x_used": "used", "y_decoy": "decoy"}

    return build


def _both(u: int, v: int) -> Callable[[], tuple[rasp.SOp, dict[str, str]]]:
    def build() -> tuple[rasp.SOp, dict[str, str]]:
        x = shift(rasp.tokens, u, "x_used")
        z = shift(rasp.tokens, v, "z_used")
        return rasp.SequenceMap(pair, x, z).named("out"), {"x_used": "used", "z_used": "used"}

    return build


def _map_decoy(u: int, members: frozenset[str]) -> Callable[[], tuple[rasp.SOp, dict[str, str]]]:
    def build() -> tuple[rasp.SOp, dict[str, str]]:
        x = shift(rasp.tokens, u, "x_used")
        m = rasp.Map(lambda t, s=members: t in s, rasp.tokens).named("m_decoy")
        return rasp.SequenceMap(first, x, m).named("out"), {"x_used": "used", "m_decoy": "decoy"}

    return build


def _correlated(u: int) -> Callable[[], tuple[rasp.SOp, dict[str, str]]]:
    """The decoy carries exactly the same information as the used variable (a copy of the
    same token, via a separate token copy), but the output ignores it."""

    def build() -> tuple[rasp.SOp, dict[str, str]]:
        copy = rasp.Map(lambda t: t, rasp.tokens).named("tok_copy")
        x = shift(rasp.tokens, u, "x_used")
        y = shift(copy, u, "y_decoy")
        roles = {"x_used": "used", "y_decoy": "decoy", "tok_copy": "decoy"}
        return rasp.SequenceMap(first, x, y).named("out"), roles

    return build


def _deep() -> Callable[[], tuple[rasp.SOp, dict[str, str]]]:
    """Two layers: a used chain (a -> m -> c -> out) with a decoy at each layer."""

    def build() -> tuple[rasp.SOp, dict[str, str]]:
        a = shift(rasp.tokens, 1, "a_used")
        b = shift(rasp.tokens, 2, "b_decoy")
        m = rasp.SequenceMap(first, a, b).named("m_used")
        c = shift(m, 1, "c_used")
        d = shift(rasp.tokens, 3, "d_decoy")
        roles = {
            "a_used": "used",
            "b_decoy": "decoy",
            "m_used": "used",
            "c_used": "used",
            "d_decoy": "decoy",
        }
        return rasp.SequenceMap(first, c, d).named("out"), roles

    return build


V5 = frozenset("abcde")
V6 = frozenset("abcdef")

#: Frozen split (docs/PHASE_7_75_PLAN.md): development programs may be inspected and run
#: before the freeze; held-out programs are run only after it.
PROGRAMS: dict[str, Program] = {
    # development
    "dev_copy_1_2": Program("dev_copy_1_2", _copy(1, 2), V5, 7, "out = x(t-1); decoy y(t-2)"),
    "dev_map_decoy": Program(
        "dev_map_decoy", _map_decoy(1, frozenset("ab")), V5, 7, "out = x(t-1); decoy MLP m"
    ),
    # held-out
    "ho_copy_2_1": Program("ho_copy_2_1", _copy(2, 1), V5, 7, "out = x(t-2); decoy y(t-1)"),
    "ho_swap_1_3": Program(
        "ho_swap_1_3", _copy_swap(1, 3), V5, 7, "out = x(t-1) (2nd arg); decoy y(t-3)"
    ),
    "ho_both_1_2": Program("ho_both_1_2", _both(1, 2), V5, 7, "out = (x(t-1), z(t-2)); both used"),
    "ho_map_decoy_2": Program(
        "ho_map_decoy_2", _map_decoy(2, frozenset("ace")), V6, 8, "out = x(t-2); decoy MLP m"
    ),
    "ho_deep": Program("ho_deep", _deep(), V5, 8, "two-layer used chain; decoys at both layers"),
    "ho_corr_1": Program(
        "ho_corr_1", _correlated(1), V5, 7, "decoy = identical copy of x, ignored"
    ),
    "ho_corr_2": Program(
        "ho_corr_2", _correlated(2), V6, 8, "decoy = identical copy of x, ignored"
    ),
    "ho_copy_1_3": Program("ho_copy_1_3", _copy(1, 3), V6, 9, "out = x(t-1); decoy y(t-3)"),
}
DEV = tuple(p for p in PROGRAMS if p.startswith("dev_"))
HELDOUT = tuple(p for p in PROGRAMS if p.startswith("ho_"))


class TDCase(TracrBenchmarkCase):
    """A circuits-benchmark case wrapper around one TD program (compilation only)."""

    def __init__(self, program: Program) -> None:
        super().__init__()
        self.program = program
        self._roles: dict[str, str] | None = None

    def get_name(self) -> str:
        return self.program.name

    def get_program(self) -> rasp.SOp:
        out, roles = self.program.build()
        self._roles = roles
        return out

    @property
    def roles(self) -> dict[str, str]:
        if self._roles is None:
            self.get_tracr_output()
        assert self._roles is not None
        return self._roles

    def get_vocab(self) -> set[str]:
        return set(self.program.vocab)

    def get_max_seq_len(self) -> int:
        return self.program.max_seq_len

    def get_task_description(self) -> str:
        return self.program.description

    def get_correct_output_for_input(self, inp: Any) -> Any:
        """The RASP interpreter's output (the program's own semantics, no model)."""
        out, _ = self.program.build()
        return rasp.evaluate(out, list(inp))
