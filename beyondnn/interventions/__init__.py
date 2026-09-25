"""Phase-2 causal interventions: controlled baseline-vs-intervention comparisons.

An intervention replaces one tensor leaf of one call of a module's output (zero
ablation, a constant, or an activation patched in from a source execution). The
effect on a scalar metric, ``intervention_value - baseline_value``, is recorded as
an INTERVENTIONAL ``CausalEffect`` scoped to exactly the input(s) compared.
Activations recorded during the intervened pass remain MEASURED. An effect is not
a claim of necessity or sufficiency: claims are decided only by declared tests
(``intervention_threshold``), and sufficiency cannot be assessed yet.
"""

from . import metrics
from .claims import (
    INTERVENTION_POLICY,
    INTERVENTION_THRESHOLD,
    PROTOCOLS,
    check_policy,
    evaluate_claim,
    threshold_spec,
)
from .runner import (
    ComparisonFamily,
    InterventionError,
    InterventionNotAppliedError,
    InterventionResult,
    StatefulComparisonError,
    StochasticComparisonError,
    compare_family,
    intervene,
    intervene_sample,
    make_claim,
    sample_id,
)
from .spec import Intervention, constant, constant_input, patch, zero, zero_input

__all__ = [
    "INTERVENTION_POLICY",
    "INTERVENTION_THRESHOLD",
    "PROTOCOLS",
    "ComparisonFamily",
    "Intervention",
    "InterventionError",
    "InterventionNotAppliedError",
    "InterventionResult",
    "StatefulComparisonError",
    "StochasticComparisonError",
    "check_policy",
    "compare_family",
    "constant",
    "constant_input",
    "evaluate_claim",
    "intervene",
    "intervene_sample",
    "make_claim",
    "metrics",
    "patch",
    "sample_id",
    "threshold_spec",
    "zero",
    "zero_input",
]
