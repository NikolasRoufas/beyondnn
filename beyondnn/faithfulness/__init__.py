"""Phase-5 faithfulness tests: specific interpretability claims made testable under
explicit, declared perturbation protocols (ADR-033).

Not a global faithfulness score (ADR-007): every quantity stays separate.

* Claim tests: :func:`comprehensiveness` (remove the selection: does the target
  drop?) and :func:`sufficiency` (retain only the selection within its site: is the
  target preserved?), run with :func:`run` (one input) or :func:`run_dataset`
  (a declared sample set, with counterexample and paired-control summaries).
* Selections: :func:`top_k` / :func:`ranking` of an attribution, or declared
  :func:`units`; matched random :func:`controls` with a recorded seed.
* Curves: :func:`curve` (removal = deletion / progressive ablation; retention =
  insertion / progressive retention); the full curve is always kept.
* Stability under user-declared transformations and method diagnostics:
  :mod:`.stability` / :mod:`.diagnostics`.

Every perturbation is a Phase-2 intervention (INTERVENTIONAL effects); results are
``ClaimTestResult`` / ``ProtocolResult`` records in the same trace. A passed protocol
supports only the claim it tested, under its declared replacement and scope.
"""

from .claims import COMPREHENSIVENESS_POLICY, SUFFICIENCY_POLICY, evaluate
from .runner import (
    CurveResult,
    DatasetResult,
    FaithfulnessResult,
    PerturbationNotAppliedError,
    curve,
    run,
    run_dataset,
)
from .spec import (
    Controls,
    FaithfulnessError,
    Replacement,
    Selection,
    SelectionMismatchError,
    SelectionRule,
    TestTemplate,
    comprehensiveness,
    controls,
    fixed,
    ranking,
    replacement,
    selector,
    sufficiency,
    top_k,
    units,
    zero,
)
from .stability import (
    DiagnosticResult,
    Transformation,
    baseline_sensitivity,
    ig_step_sensitivity,
    method_agreement,
    stability,
    transformation,
)

__all__ = [
    "COMPREHENSIVENESS_POLICY",
    "SUFFICIENCY_POLICY",
    "Controls",
    "CurveResult",
    "DatasetResult",
    "DiagnosticResult",
    "FaithfulnessError",
    "FaithfulnessResult",
    "PerturbationNotAppliedError",
    "Replacement",
    "Selection",
    "SelectionMismatchError",
    "SelectionRule",
    "TestTemplate",
    "Transformation",
    "baseline_sensitivity",
    "comprehensiveness",
    "controls",
    "curve",
    "evaluate",
    "fixed",
    "ig_step_sensitivity",
    "method_agreement",
    "ranking",
    "replacement",
    "run",
    "run_dataset",
    "selector",
    "stability",
    "sufficiency",
    "top_k",
    "transformation",
    "units",
    "zero",
]
