"""Phase-3 attribution: method-relative scores for a scalar target (ATTRIBUTED evidence).

An attribution answers "under method A (configuration C, baseline B), for target T
on input X, what score was assigned to the elements of S?". It does not answer
what caused the output: that needs interventions (:mod:`beyondnn.interventions`).
A high attribution is not necessity, sufficiency, meaning, or "the reason".

* Methods: :func:`gradient`, :func:`input_x_gradient`, :func:`integrated_gradients`
  (native reference implementations) and :mod:`beyondnn.attribution.captum`
  (Captum adapters, optional dependency).
* What is attributed: :func:`input` (a floating-point model input) or :func:`layer`
  (one leaf of one call of a module output; e.g. an embedding output for token
  models: that is attribution to embedding dimensions, not to token ids).
* Targets: Phase-2 built-in metrics (``interventions.metrics.select``,
  ``difference``, ``mean``); exactly one element, never an implicit sum.
* Reductions: only explicit (:func:`reduce`), recorded as their own records.
"""

from .claims import ATTRIBUTION_POLICY, evaluate_claim, make_claim, threshold_spec
from .native import AttributionError, DiscreteInputError, RepeatedCallError
from .runner import (
    AttributionResult,
    AutogradStateError,
    StatefulAttributionError,
    StochasticAttributionError,
    attribute,
)
from .spec import (
    At,
    Baseline,
    Method,
    Reduce,
    baseline,
    gradient,
    input,
    input_baseline,
    input_x_gradient,
    integrated_gradients,
    layer,
    reduce,
    zero_baseline,
)

__all__ = [
    "ATTRIBUTION_POLICY",
    "At",
    "AttributionError",
    "AttributionResult",
    "AutogradStateError",
    "Baseline",
    "DiscreteInputError",
    "Method",
    "Reduce",
    "RepeatedCallError",
    "StatefulAttributionError",
    "StochasticAttributionError",
    "attribute",
    "baseline",
    "evaluate_claim",
    "gradient",
    "input",
    "input_baseline",
    "input_x_gradient",
    "integrated_gradients",
    "layer",
    "make_claim",
    "reduce",
    "threshold_spec",
    "zero_baseline",
]
