# BeyondNN

> **Status: pre-alpha, Phase 1 complete (local only, not released).** Implemented:
> - the trace schema, provenance, and trace recording and persistence;
> - a minimal INPUT → WHY → OUTPUT view whose WHY is *measured evidence only*.
>
> Interventions, attribution, causal tests, concepts, and claim testing are **not implemented**. See
> [`docs/PHASE_1_REPORT.md`](docs/PHASE_1_REPORT.md).

BeyondNN is an interpretability evidence framework for PyTorch.

Its goal is to turn claims about neural-network computation into structured, provenance-aware, testable
objects. Today it provides:
- structured traces;
- provenance;
- explicit epistemic status.

Interventions and claim testing are planned for Phase 2+.

BeyondNN does not assume that an attribution, a probe, a generated explanation, or a readable feature is
automatically a faithful explanation of model computation. The schema labels every result with how it was
obtained:
- observed or measured;
- attributed;
- interventional or estimated causal;
- validated concept;
- generated.

**Phase 1 produces only *observed* and *measured* evidence.** The other statuses exist in the schema for
later phases.

The schema already models claims as explicit records that must pass declared tests before they are
reported as supported. No test runners exist yet.

BeyondNN builds on PyTorch and is meant to work *alongside* Captum, nnsight, TransformerLens, and SAELens,
not to replace them.

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

## Still proposed (not implemented)

```python
result = bnn.test_claim(model, claim, inputs=..., tests=[...])   # Phase 2+: interventions, causal tests
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
