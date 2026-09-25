"""``instrument()``: a lightweight BeyondNN handle referencing the original model (M1.8).

The handle is not an ``nn.Module``, does not register, copy, wrap, patch, or hook
the model, and holds no state besides the reference. Every method delegates to
the one evidence-capture pipeline (``beyondnn.trace`` / ``beyondnn.recording``).
Use ``handle.model`` for the model itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from beyondnn.core.trace import Recording, TraceResult
from beyondnn.core.trace import recording as _recording
from beyondnn.core.trace import trace as _trace
from beyondnn.schema import ModelDeclaration, Randomness

from .response import ExplainResponse

__all__ = ["Instrumented", "instrument"]


@dataclass(frozen=True, slots=True, eq=False)
class Instrumented:
    """A handle referencing ``model``. See :func:`instrument`."""

    model: nn.Module

    def trace(
        self,
        *inputs: Any,
        sites: Sequence[str] = (),
        input_sites: Sequence[str] = (),
        retention: str = "summary",
        declared_model: ModelDeclaration | None = None,
        randomness: Randomness | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> TraceResult:
        """Exactly ``beyondnn.trace(self.model, ...)``."""
        return _trace(
            self.model,
            *inputs,
            sites=sites,
            input_sites=input_sites,
            retention=retention,
            declared_model=declared_model,
            randomness=randomness,
            model_kwargs=model_kwargs,
        )

    def recording(
        self,
        *,
        sites: Sequence[str] = (),
        input_sites: Sequence[str] = (),
        retention: str = "summary",
        declared_model: ModelDeclaration | None = None,
        randomness: Randomness | None = None,
    ) -> Recording:
        """Exactly ``beyondnn.recording(self.model, ...)``."""
        return _recording(
            self.model,
            sites=sites,
            input_sites=input_sites,
            retention=retention,
            declared_model=declared_model,
            randomness=randomness,
        )

    def explain(
        self,
        *inputs: Any,
        sites: Sequence[str] = (),
        input_sites: Sequence[str] = (),
        retention: str = "summary",
        declared_model: ModelDeclaration | None = None,
        randomness: Randomness | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> ExplainResponse:
        """One trace of ``model(*inputs, **model_kwargs)`` presented as INPUT -> WHY -> OUTPUT.

        Phase-1 WHY is measured internal evidence, not a causal or attributed
        explanation. There is no ``target`` argument: target-specific explanation
        needs attribution/interventions (later phases).
        """
        return ExplainResponse(
            self.trace(
                *inputs,
                sites=sites,
                input_sites=input_sites,
                retention=retention,
                declared_model=declared_model,
                randomness=randomness,
                model_kwargs=model_kwargs,
            )
        )

    def __repr__(self) -> str:
        return f"Instrumented(model={type(self.model).__name__})"


def instrument(model: nn.Module) -> Instrumented:
    """Return a lightweight BeyondNN handle referencing the original ``model``.

    Nothing about ``model`` changes: no hooks, no wrapping, no copies.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("instrument() requires a torch.nn.Module")
    return Instrumented(model)
