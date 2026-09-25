# BeyondNN

> **Status: pre-alpha, Phase 4 (structured WHY / evidence synthesis) implemented (local only, not released).**
>
> Implemented:
> - the trace schema, provenance, trace recording and persistence;
> - a minimal INPUT → WHY → OUTPUT view (WHY = measured evidence);
> - controlled activation interventions with INTERVENTIONAL effect records;
> - gradient, input × gradient, and Integrated Gradients attribution (native, and through Captum) with ATTRIBUTED records;
> - a structured WHY that composes measured, attributed, and interventional evidence and declared claim tests without merging them.
>
> Faithfulness evaluation, sufficiency testing, and concepts are **not implemented**. See [`docs/PHASE_1_REPORT.md`](docs/PHASE_1_REPORT.md), [`docs/PHASE_2_REPORT.md`](docs/PHASE_2_REPORT.md), [`docs/PHASE_3_REPORT.md`](docs/PHASE_3_REPORT.md), and [`docs/PHASE_4_REPORT.md`](docs/PHASE_4_REPORT.md).

BeyondNN is an interpretability evidence framework for PyTorch.

It turns claims about neural-network computation into structured, provenance-aware, testable objects. Today it provides:
- structured traces;
- provenance;
- explicit epistemic status;
- controlled interventions whose effects are recorded as INTERVENTIONAL evidence;
- method-relative attributions recorded as ATTRIBUTED evidence (never as causes);
- threshold claim tests that must be declared before they run;
- one structured WHY view that keeps each kind of evidence separate.

BeyondNN does not assume that an attribution, a probe, a generated explanation, or a readable feature is automatically a faithful explanation of model computation. The schema labels every result with how it was obtained:
- observed or measured;
- attributed;
- interventional or estimated causal;
- validated concept;
- generated.

**Today BeyondNN produces observed, measured, attributed, and (from controlled interventions) interventional evidence only.** The other statuses exist in the schema for later phases.

BeyondNN builds on PyTorch and is meant to work *alongside* Captum, nnsight, TransformerLens, and SAELens, not to replace them.

## What works today (pre-alpha, local only)

Trace recording and a minimal INPUT → WHY → OUTPUT view are implemented, with structured, provenance-bearing, validated records.

```python
# runnable example (executed by tests/test_readme.py)
import torch
from torch import nn

import beyondnn as bnn


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


torch.manual_seed(0)
model = TinyNet()
x = torch.randn(3, 4)

handle = bnn.instrument(model)          # a handle referencing the original model; nothing is modified
response = handle.explain(x, sites=["encoder.*", "head"])

print(response.input)                   # OBSERVED root input
print(response.why.activations)         # MEASURED internal states, in execution order
print(response.why.limitations)         # what this does NOT cover, as structured records
print(response.output)                  # OBSERVED root output
print(response.render())                # deterministic text view of the same records

trace = bnn.trace(model, x, sites=["head"], retention="cpu")   # the underlying evidence
print(trace.activation("head"), trace.origin(trace.activation("head")).model)
```

> **In Phase 1, `WHY` is measured internal evidence, not a causal or attributed explanation.** It answers
> "what internal evidence was measured while this output was produced?". It does not answer "which
> internal state caused the output?". No activation is ranked, called important, or treated as a reason.
> Every explanation carries the `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE` and `NO_CLAIMS_TESTED` limitations.

- **Recording several passes:** `bnn.recording(model, sites=[...])` records several forward passes in a `with` block. `ctx.result` is only available after a clean exit.
- **Retention:** `summary` (default: metadata and summary statistics), `cpu` (detached CPU copies), or `none`.
- **Honest limits:** every trace states its limits. For example, `FUNCTIONAL_OPS_UNOBSERVED`: module hooks cannot see functional operations or residual additions.
- **Persistence:** `trace.save("run1/")` and `bnn.load_trace("run1/")` persist traces as `trace.json` plus an optional `tensors.pt`. The sidecar is only ever read with `weights_only=True`, and everything is re-validated on load.
- **Refusals:** public tracing refuses models that carry forward hooks not installed by BeyondNN, and models whose tensors span several devices. Only CPU is verified in Phase 1.

## Controlled interventions (Phase 2)

```python
# runnable example (executed by tests/test_readme.py)
import torch
from torch import nn

import beyondnn as bnn

iv = bnn.interventions


class TwoPaths(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)
        self.b = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)


torch.manual_seed(0)
model = TwoPaths().eval()                  # comparisons are refused in training mode
x = torch.tensor([[1.0, 2.0]])

result = bnn.intervene(model, x, intervention=iv.zero("a"), metric=iv.metrics.select([0, 0]))
print(result.baseline_value, result.intervention_value, result.value)   # value = intervention - baseline
print(result.effect.status, result.effect.estimand.scope)               # INTERVENTIONAL, INSTANCE
print([lim.code for lim in result.limitations])                         # e.g. ZERO_ABLATION_MAY_BE_OOD
```

- **What it runs:** `bnn.intervene` records a CLEAN baseline pass and an INTERVENTION pass of the same input in one trace. Activations in the intervened pass stay **MEASURED**; only the metric difference is **INTERVENTIONAL**.
- **Scope of the effect:** it is scoped to exactly the input(s) compared. It is not a claim that `a` is necessary in general: a redundant path can make ablation look small, and other inputs can behave differently.
- **Claims:** they are decided only by a threshold test declared in advance (`iv.threshold_spec`, `iv.make_claim`, `claims=[...]`). Sufficiency cannot be assessed yet.
- **Refusals:** comparisons are refused if randomness is consumed, the model state changes between passes, or the intervention does not apply.

## Attribution (Phase 3)

```python
# runnable example (executed by tests/test_readme.py)
import torch
from torch import nn

import beyondnn as bnn

A, iv = bnn.attribution, bnn.interventions

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).eval()
x = torch.tensor([[1.0, -2.0]])

result = bnn.attribute(
    model,
    x,
    target=iv.metrics.select([0, 0]),                  # one explicit scalar; outputs are never summed
    method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64),  # the baseline is a choice
)
print(result.value)                  # raw attribution tensor, same shape as x
print(result.record.status)          # EvidenceStatus.ATTRIBUTED (never interventional)
print(result.completeness_delta)     # sum(attr) - (F(x) - F(baseline)): a numerical diagnostic
print([lim.code for lim in result.limitations])   # ATTRIBUTION_BASELINE_ASSUMPTION, ...
```

- **What it answers:** "under method A (configuration C, baseline B), for target T on this input, what score was assigned to each input element?" It does not answer what caused the output.
- **Attribution is not necessity:** on a model with two redundant paths, one path receives substantial attribution, yet ablating it leaves the output unchanged (see the Phase 3 report). Attribution cannot support NECESSARY_FOR or SUFFICIENT_FOR claims; only ATTRIBUTED_TO, under a declared `attribution_threshold` test.
- **Other methods:** `A.gradient()` and `A.input_x_gradient()` are separate methods. Captum implementations are available through `beyondnn.attribution.captum` (optional: `pip install beyondnn[captum]`, Captum 0.9.x).
- **Token models:** integer token ids are refused. `at=A.layer("token_embedding")` attributes to embedding dimensions per position, not to token ids. Any per-token score comes only from an explicit `reductions=[A.reduce("sum", (-1,))]`.
- **Refusals:** training mode; foreign forward or backward hooks; alias paths; RNG use; and any change to model state, gradients, or caller tensors.

## The full progression: one structured WHY (Phase 4)

```python
# runnable example (executed by tests/test_readme.py)
import torch
from torch import nn

import beyondnn as bnn
from beyondnn.schema import InterventionOperation, Relation

A, iv = bnn.attribution, bnn.interventions


class TwoEqualPaths(nn.Module):
    """y = p(x) + q(x), where p and q both compute x0: two redundant paths."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Linear(2, 1, bias=False)
        self.q = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.p.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.q.weight.copy_(torch.tensor([[1.0, 0.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x) + self.q(x)


handle = bnn.instrument(TwoEqualPaths().eval())
x = torch.tensor([[3.0, 5.0]])
target = iv.metrics.select([0, 0])

# 1. Measure (OBSERVED + MEASURED).
trace = handle.trace(x, sites=["p", "q"])

# 2. Attribute, with a claim declared before running (ATTRIBUTED).
ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
credit = A.make_claim(A.layer("p"), target, x, statement="p receives attribution for y")
attr = handle.attribute(
    x, target=target, method=ig, at=A.layer("p"),
    claims=[(credit, A.threshold_spec(ig, at=A.layer("p"), min_abs_attribution=2.0))],
)

# 3. Intervene, with a causal claim declared before running (INTERVENTIONAL).
necessary = iv.make_claim(iv.zero("p"), target, Relation.NECESSARY_FOR, x,
                          statement="p is necessary for y")
effect = handle.intervene(
    x, intervention=iv.zero("p"), metric=target,
    claims=[(necessary, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=6.0))],
)

# 4. Compose (runs nothing; refuses evidence about another model, input, or target).
response = bnn.compose(trace, attributions=[attr], interventions=[effect],
                       policies=[A.ATTRIBUTION_POLICY, iv.INTERVENTION_POLICY])
print(response.render())
for claim in response.why.claims:
    print(claim.claim.statement, "->", [a.verdict.value for a in claim.assessments])
print(response.why.coverage.faithfulness_evaluated)          # False: never evaluated here
```

- **Measured evidence** says what was observed internally.
- **Attribution** says what a method assigned credit to (here, `p` receives 3.0).
- **An intervention** says what changed under a controlled manipulation (zeroing `p` changes `y` by −3.0, yet `y` stays 3.0: by construction, `q` computes the same value).
- **Claims** say what was explicitly tested: "p ATTRIBUTED_TO y" is *supported*; "p NECESSARY_FOR y" is *contradicted*.

**WHY keeps these statements separate.** It never combines them into one score and never generates a claim. It states what was not evaluated: faithfulness, comprehensiveness, sufficiency, and concepts. Evidence about a different model, declared model, input, or target is refused, not merged.

## Still proposed (not implemented)

Faithfulness and sufficiency protocols (Phase 5), and concepts (Phase 6).

There is no single "explanation confidence" percentage. BeyondNN reports component evidence until an aggregate has been validated.

## Documentation

Start at [`docs/README.md`](docs/README.md). Key documents:

- [Architecture proposal](docs/design/ARCHITECTURE_PROPOSAL.md)
- [Trace schema](docs/design/TRACE_SCHEMA_PROPOSAL.md)
- [What "interpretable" means here](docs/design/INTERPRETABILITY_DEFINITION.md)
- [Ecosystem audit](docs/research/ECOSYSTEM_AUDIT.md) and [differentiation](docs/research/DIFFERENTIATION.md)
- [Architecture decisions](docs/decisions/ARCHITECTURE_DECISIONS.md)

## License

[Apache-2.0](LICENSE)
