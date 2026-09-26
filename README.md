# BeyondNN

> **Status: pre-alpha, Phase 6 (concepts and concept validation) implemented (local only, not released).**
>
> Implemented:
> - the trace schema, provenance, trace recording and persistence;
> - a minimal INPUT → WHY → OUTPUT view (WHY = measured evidence);
> - controlled activation interventions with INTERVENTIONAL effect records;
> - gradient, input × gradient, and Integrated Gradients attribution (native, and through Captum) with ATTRIBUTED records;
> - a structured WHY that composes measured, attributed, and interventional evidence and declared claim tests without merging them;
> - faithfulness *protocols* (comprehensiveness, sufficiency, removal/retention curves, stability, counterexamples) with matched random controls. There is no faithfulness score;
> - concepts: features (neurons, directions, SAE latents), proposals, and controlled validation that keeps *decodable* (ENCODES) and *used* (intervention) separate. There is no concept score.
>
> See [`docs/PHASE_1_REPORT.md`](docs/PHASE_1_REPORT.md), [`docs/PHASE_2_REPORT.md`](docs/PHASE_2_REPORT.md), [`docs/PHASE_3_REPORT.md`](docs/PHASE_3_REPORT.md), [`docs/PHASE_4_REPORT.md`](docs/PHASE_4_REPORT.md), and [`docs/PHASE_5_REPORT.md`](docs/PHASE_5_REPORT.md), [`docs/PHASE_5_5_REPORT.md`](docs/PHASE_5_5_REPORT.md), and [`docs/PHASE_6_REPORT.md`](docs/PHASE_6_REPORT.md).

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
- **Claims:** they are decided only by a threshold test declared in advance (`iv.threshold_spec`, `iv.make_claim`, `claims=[...]`). Sufficiency can be assessed only by the Phase-5 `sufficiency` protocol, and only in its declared, site-relative sense.
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

## Faithfulness tests (Phase 5)

```python
# runnable example (executed by tests/test_readme.py)
import torch
from torch import nn

import beyondnn as bnn

A, F, iv = bnn.attribution, bnn.faithfulness, bnn.interventions


class Saturated(nn.Module):
    """y = tanh(4 x0) + 0.2 x1: x0 dominates y at x0 = 3, but its gradient is ~0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(4 * x[:, 0]) + 0.2 * x[:, 1]).unsqueeze(1)


model, x = Saturated().eval(), torch.tensor([[3.0, 1.0]])
target = iv.metrics.select([0, 0])
gradient = A.attribute(model, x, target=target, method=A.gradient())
integrated = A.attribute(model, x, target=target,
                         method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64))

# The same declared test for both rankings: does removing the top-1 unit drop y by >= 0.5?
test = F.comprehensiveness(target=target, min_drop=0.5, replacement=F.zero(),
                           statement="the top-ranked input unit is necessary for y")
for name, attribution in (("gradient", gradient), ("integrated gradients", integrated)):
    result = F.run(model, x, test=test, selection=F.top_k(attribution, k=1))
    print(name, result.claim.subject.units, result.outcome.value, round(result.drop, 3))
# gradient (1,) contradicts 0.2               <- a plausible ranking that misses x0
# integrated gradients (0,) supports 1.0
```

- **What a result means:** "the claim passed `comprehensiveness/v1` under zero replacement on this input". It does not mean "the explanation is faithful".
- **Where the results go:** into the structured WHY (`bnn.compose(..., faithfulness=[...])`), next to the attribution and intervention evidence, with their limitations, controls, and the list of protocols that were **not** run.
- **Protocol documentation:** [`docs/protocols/`](docs/protocols/README.md).

## Concepts (Phase 6)

```python
# runnable example (executed by tests/test_readme.py)
import torch

import beyondnn as bnn
from beyondnn._testing.concept_models import ConceptToy, concept_inputs

C, iv = bnn.concepts, bnn.interventions
model, x = ConceptToy().eval(), concept_inputs(200, seed=0)  # output uses h0 = x0; h1 = x1 is unused
splits = ["train"] * 100 + ["val"] * 40 + ["test"] * 60
data = C.dataset([x[i : i + 1] for i in range(200)], (x[:, 1] > 0).long().tolist(), splits,
                 name="toy x1", label_source="x1 > 0")
concept = C.propose(C.neuron("hidden", 1), label="x1 is positive", definition="x1 > 0")  # PROPOSED
encoding = C.encoding_test(model, concept, data, criteria=C.encoding_criteria(min_fraction_below=0.95),
                           controls=[C.random_neurons(50, seed=1), C.label_permutation(50, seed=2)])
use = C.use_test(model, concept, data, target=iv.metrics.select([0, 0]), relation="decreases",
                 intervention=C.remove(C.zero()),  # the intervention is always declared
                 controls=[C.random_neurons(20, seed=3)],
                 criteria=C.use_criteria(min_change=0.25, min_fraction_beyond_controls=0.9))
validation = C.validate(concept, encoding=encoding, use=[use])
print(encoding.outcome.value, use.outcome.value, validation.semantic_status.value)
# supports contradicts proposed_concept     <- decodable, but not used: never "validated"
```

- **What VALIDATED_CONCEPT would mean:** encoding *and* use claims supported above declared controls, with counterexamples recorded, *within the recorded scope* (checkpoint, site, dataset split, intervention, target). It never means "the model understands C".
- **Where the results go:** into the structured WHY (`bnn.compose(trace, concepts=[validation])`), as dataset-scoped context with the encoding and use outcomes on separate lines.
- **Documentation:** [`docs/concepts/`](docs/concepts/README.md); the flagship example is `examples/phase6_concepts.py`.

## Audits (Phase 7)

```python
# runnable example (executed by tests/test_readme.py)
import torch

import beyondnn as bnn
from beyondnn._testing.causal_models import Redundant
from beyondnn.core.samples import sample_id
from beyondnn.schema import InterventionOperation, Relation, Site, Subject

A, iv, AU = bnn.attribution, bnn.interventions, bnn.audits
model, x = Redundant().eval(), torch.tensor([[3.0, 5.0]])  # y = p(x) + q(x), with p = q = x0
target = iv.metrics.select([0, 0])
ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
credit = A.make_claim(A.layer("p"), target, x, statement="p receives attribution for y")
attribution = A.attribute(model, x, target=target, method=ig, at=A.layer("p"),
                          claims=[(credit, A.threshold_spec(ig, at=A.layer("p"), min_abs_attribution=2.0))])
necessary = iv.make_claim(iv.zero("p"), target, Relation.NECESSARY_FOR, x, statement="p is necessary for y")
effect = bnn.intervene(model, x, intervention=iv.zero("p"), metric=target,
                       claims=[(necessary, iv.threshold_spec(operation=InterventionOperation.ZERO,
                                                             min_effect=6.0))])

plan = AU.plan(  # declared before auditing; every field is explicit
    name="redundant_path", checkpoint=AU.checkpoint_of(model), declared_model=None,
    samples=[sample_id(x)], datasets=[], concepts=[], naive_auroc=None,
    claims=[AU.claim("p_necessary", statement="p is necessary for y", relation="necessary_for",
                     target=target, scope="instance", requirement="intervention",
                     subject=Subject(site=Site(module="p")))],
    requirements=[AU.requirement("intervention", policy=iv.INTERVENTION_POLICY, controls=False)],
    counterexamples=AU.counterexample_rule(max_counterexample_fraction=None,
                                           max_false_positive_rate=None, max_false_negative_rate=None))
for evidence in ([attribution], [attribution, effect]):
    audited = bnn.audit(evidence, plan=plan).claim("p_necessary")
    print(dict(audited.distribution), sorted(f.code for f in audited.findings))
# {'unsupported': 1} ['attribution_is_not_intervention']       <- attribution only
# {'contradicted': 1} ['attribution_intervention_disagree', 'counterexamples_present']
```

- **What an audit is:** a deterministic, model-free classification of the declared claims from recorded, **re-derived**, in-scope evidence. The standings are SUPPORTED, CONTRADICTED, MIXED, ASSUMPTION_SENSITIVE, INCONCLUSIVE, UNSUPPORTED and NOT_EVALUATED, each with the findings behind it.
- **What an audit never does:** compute a score, resolve a contradiction, or say that an explanation is trustworthy. Missing evidence is NOT_EVALUATED.
- **What it reports:**
  - structural overclaims (attribution → causal, decodable → used, generated → validated, instance → population, one replacement / k / threshold / null → universal);
  - sensitivity to each assumption;
  - per-sample distributions with counterexample identities;
  - evidence that is excluded for provenance or scope.
- **Persistence:** audits run on saved traces too (`bnn.audit([path, ...], plan=plan)`), and `bnn.audits.verify_report` re-derives a stored report.
- **WHY integration:** `bnn.compose(trace, audit=report)` adds an AUDIT section.
- **Documentation:** [`docs/audit/`](docs/audit/README.md).

## Still proposed (not implemented)

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
