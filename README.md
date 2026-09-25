# BeyondNN

> **Status: pre-alpha.** Phase 1 is in progress: the trace schema, provenance, and trace recording work.
> Explanations, interventions, and claim testing are not implemented yet. See [`docs/roadmap/PHASE_1_PLAN.md`](docs/roadmap/PHASE_1_PLAN.md).

BeyondNN is an interpretability evidence framework for PyTorch.

It turns claims about neural-network computation into structured, provenance-aware, testable objects.
It provides:
- structured traces;
- provenance;
- interventions;
- claim testing;
- explicit epistemic status for interpretability results.

BeyondNN does not assume that an attribution, a probe, a generated explanation, or a readable feature is
automatically a faithful explanation of model computation. Every result is labelled with how it was
obtained:
- observed or measured;
- attributed;
- interventional or estimated causal;
- validated concept;
- generated.

Claims about the model become explicit records that must survive declared tests before they are
reported as supported. It also records what remains unknown.

BeyondNN builds on PyTorch and is meant to work *alongside* Captum, nnsight, TransformerLens, and SAELens,
not to replace them.

## What works today (pre-alpha, local only)

Trace recording is implemented: structured, provenance-bearing, and validated.

```python
import torch
import beyondnn as bnn

trace = bnn.trace(
    model,
    x,
    sites=["blocks.*.attn"],      # module outputs; input_sites=[...] for module inputs
)

for activation in trace.activations:      # MEASURED records, in execution order
    print(activation.site, activation.pass_index, activation.call_index, activation.value.shape)

print(trace.input, trace.output)          # OBSERVED root input/output
print(trace.origin(trace.activations[0])) # model fingerprint + environment + execution conditions
print(trace.limitations)                  # what the trace does NOT cover
```

- `bnn.recording(model, sites=[...])` records several forward passes in a `with` block. `ctx.result` is only available after a clean exit.
- Retention is `summary` (default: metadata and summary statistics), `cpu` (detached CPU copies), or `none`.
- Every trace states its limits. For example, `FUNCTIONAL_OPS_UNOBSERVED`: module hooks cannot see functional operations.

## Still proposed (not implemented)

```python
response = bnn.instrument(model).explain(x, target=target)   # M1.8
result = bnn.test_claim(model, claim, inputs=..., tests=[...])  # Phase 2
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
