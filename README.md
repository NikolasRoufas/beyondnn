# BeyondNN

> **Status: pre-alpha, design phase.** There is no implementation yet. Every API below is a **proposal**
> and none of it is importable. See [`docs/roadmap/PHASE_1_PLAN.md`](docs/roadmap/PHASE_1_PLAN.md).

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

## Proposed API (not implemented)

```python
import beyondnn as bnn

trace = bnn.trace(model, x)                 # measured internal states, with provenance

response = bnn.instrument(model).explain(   # the model itself is never modified
    x,
    target=target,                          # an explanation is always *of* something
)

print(response.input)                       # observed
print(response.why)                         # structured evidence, claims, assessments, limitations
print(response.output)                      # observed
```

Testing a claim (proposed; Phase 2):

```python
claim = bnn.claims.necessary(subject=bnn.site("layers.1", index=(..., 41)),
                             target=bnn.targets.Logit(2), scope=eval_inputs, min_effect=0.1)
result = bnn.test_claim(model, claim, inputs=eval_inputs,
                        tests=[bnn.tests.Ablation(op=bnn.MeanAblation(ref=train_inputs)),
                               bnn.tests.RandomBaseline(n=50)])
result.assessment.verdict                   # SUPPORTED / CONTRADICTED / MIXED / INCONCLUSIVE / UNTESTED
```

There is no single "explanation confidence" percentage. BeyondNN reports component evidence (effect
versus random baseline, stability, method agreement, and so on) until an aggregate has been validated.

## Documentation

Start at [`docs/README.md`](docs/README.md). Key documents:

- [Architecture proposal](docs/design/ARCHITECTURE_PROPOSAL.md)
- [Trace schema](docs/design/TRACE_SCHEMA_PROPOSAL.md)
- [What "interpretable" means here](docs/design/INTERPRETABILITY_DEFINITION.md)
- [Ecosystem audit](docs/research/ECOSYSTEM_AUDIT.md) and [differentiation](docs/research/DIFFERENTIATION.md)
- [Architecture decisions](docs/decisions/ARCHITECTURE_DECISIONS.md)

## License

[Apache-2.0](LICENSE)
