# BeyondNN

> **Status: pre-alpha, Phase 2 (causal interventions) implemented (local only, not released).**
>
> Implemented:
> - the trace schema, provenance, trace recording and persistence;
> - a minimal INPUT → WHY → OUTPUT view (WHY = measured evidence);
> - controlled activation interventions with INTERVENTIONAL effect records.
>
> Attribution, concepts, and sufficiency testing are **not implemented**. See [`docs/PHASE_1_REPORT.md`](docs/PHASE_1_REPORT.md) and [`docs/PHASE_2_REPORT.md`](docs/PHASE_2_REPORT.md).

BeyondNN is an interpretability evidence framework for PyTorch.

It turns claims about neural-network computation into structured, provenance-aware, testable objects. Today it provides:
- structured traces;
- provenance;
- explicit epistemic status;
- controlled interventions whose effects are recorded as INTERVENTIONAL evidence;
- threshold claim tests that must be declared before they run.

BeyondNN does not assume that an attribution, a probe, a generated explanation, or a readable feature is automatically a faithful explanation of model computation. The schema labels every result with how it was obtained:
- observed or measured;
- attributed;
- interventional or estimated causal;
- validated concept;
- generated.

**Today BeyondNN produces observed, measured, and (from controlled interventions) interventional evidence only.** The other statuses exist in the schema for later phases.

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

## Still proposed (not implemented)

```python
bnn.attribute(model, x, method=...)   # Phase 3+: attribution (not implemented)
```

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
