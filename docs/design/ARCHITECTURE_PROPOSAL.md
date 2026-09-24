# Architecture Proposal

Status: **design, pending implementation.** Resolved decisions are recorded in
[`../decisions/ARCHITECTURE_DECISIONS.md`](../decisions/ARCHITECTURE_DECISIONS.md) and referenced here as ADR-NNN.
Anything not backed by an ADR is still a proposal. A `TraceResult` is single-execution evidence; multi-input work belongs in a Phase 2 `Study` (ADR-015).

## Positioning (ADR-005)

> BeyondNN is an interpretability evidence framework for PyTorch that turns claims about neural-network computation into structured, provenance-aware, testable objects.

Existing tools *produce* activations, attributions, interventions and features. BeyondNN *records*:
- what those results mean, and what epistemic status they have;
- where they came from;
- which claim they bear on;
- which tests were run, and what survived;
- what remains unknown.

## Conceptual pipeline

```
 Measurement      exact or method-relative records produced by running the model
     │            (OBSERVED / MEASURED / ATTRIBUTED / INTERVENTIONAL / ESTIMATED_CAUSAL)
     ▼
 Claim            an immutable, typed proposition about the computation
     │            (subject, relation, target, scope, expectation, source)
     ▼
 Test             a pre-declared procedure plus decision criteria, applicable to certain relations
     │            (ClaimTestSpec: kind, params, criteria, spec_hash)
     ▼
 Evidence         the outcome of running one test on one claim, with the measurements it produced
     │            (ClaimTestResult: SUPPORTS / CONTRADICTS / INCONCLUSIVE / NOT_APPLICABLE / ERRORED)
     ▼
 Assessment       a derived, recomputable verdict under a versioned policy, plus component scores
     │            (UNTESTED / SUPPORTED / CONTRADICTED / MIXED / INCONCLUSIVE; never a probability)
     ▼
 Presentation     Why / render() / summary(): views over the above, never a source of truth
```

The names are conceptual. The schema names are given in `TRACE_SCHEMA_PROPOSAL.md`.

**Separation rules:**
1. **Measurements know nothing about claims.** A trace is valid with zero claims.
2. **A claim has no status field.** Status is a function of (claim, results, policy), computed by an Assessment. Storing status on the claim would allow stale or unsupported status. (This is a deliberate deviation from the `Claim(status=…)` sketch.)
3. **A test declares, before running, which relations it can bear on and what criteria decide its outcome.** `spec_hash` is computed at spec construction. Results reference it, so changing the criteria after seeing results produces a different, detectable hash.
4. **An assessment is reproducible:** `assess(claim, results, policy)` is pure. A saved assessment can be recomputed and compared on load.
5. **Presentation may only cite.** Rendered and generated text carries `supporting_records`, and nothing may cite generated text as support.

## How the design makes the four classic mistakes hard

| Mistake | Mechanism | Where enforced |
|---|---|---|
| Correlation presented as causation | Only `INTERVENTIONAL` / `ESTIMATED_CAUSAL` measurements can make a causal-relation test (`NECESSARY_FOR`, `SUFFICIENT_FOR`, `INCREASES`, `DECREASES`) return `SUPPORTS`. Other runners return `NOT_APPLICABLE` for those relations. | test registry (Phase 2) plus `ClaimTestResult` invariant (Phase 1 schema) |
| Probe success presented as model use | Detection/probe tests apply only to `ENCODES`. An `ENCODES` claim never implies `NECESSARY_FOR`. `VALIDATED_CONCEPT` requires both detection and causal tests (ADR-009). | registry, `Concept` invariant |
| Generated labels presented as validated concepts | Labels enter as `PROPOSED_CONCEPT` with a `GENERATED` source. Only `validate()` can produce `VALIDATED_CONCEPT`. | `Concept.__post_init__` |
| Saliency presented as faithfulness | Attribution supports only `ATTRIBUTED_TO`. Faithfulness is a *test result* (comprehensiveness, sufficiency, random baseline) about a claim, never a property of a map. | registry; the renderer uses status-specific verbs |

The renderer also uses **status-bound vocabulary**:
- "caused a change of" appears only for `INTERVENTIONAL`;
- "estimated effect" for `ESTIMATED_CAUSAL`;
- "scored" for `ATTRIBUTED`;
- "measured" for `MEASURED`.

A unit test lints the renderer templates against this table.

## Layering

```
┌───────────────────────────────────────────────────────────────┐
│ explain/   Why, rendering, summaries (presentation only)      │  Phase 4
│ audit/     audit(model, dataset) → AuditReport                │  Phase 7
│ concepts/  Concept lifecycle + validation protocols           │  Phase 6
│ claims/    test runners, registry, assessment policies        │  Phase 2 (+5)
├───────────────────────────────────────────────────────────────┤
│ attribution/  gradient, input×grad (+ Captum adapter)         │  Phase 3
│ causal/       measure_effect, CausalEffect, metric interface  │  Phase 2
│ interventions/ InterventionSpec + operations                  │  Phase 2
├───────────────────────────────────────────────────────────────┤
│ core/    site resolution, hook engine, trace(), recording(),  │  Phase 1
│          instrument(), targets                                │
│ schema/  all record types (incl. Claim/Test/Result/Assessment │  Phase 1
│          as pure data), status enums, provenance, codec       │
├───────────────────────────────────────────────────────────────┤
│ PyTorch                                                       │
└───────────────────────────────────────────────────────────────┘
          adapters/ (optional extras): captum, nnsight, sae_lens, transformer_lens
```

**Rules:**
- Dependencies point downward only.
- `schema` imports only the stdlib, plus `torch` for `TensorRef` construction helpers.
- `core` imports only `torch` + stdlib + `schema`.
- Every analysis layer produces records into a `TraceResult` and never returns bare tensors as its primary output. Tensors remain accessible through records.
- No architecture-specific code in `schema` or `core` (ADR-010).

"Metrics" (comprehensiveness, sufficiency, stability, random baseline) are **claim tests** in this design, not a separate layer. That removes the earlier `metrics/` package: each faithfulness metric is a `ClaimTestSpec` kind with a runner.

## Design decisions

### D1. Schema mechanism: frozen stdlib dataclasses (ADR-001, accepted)
- Frozen `slots` dataclasses plus a hand-written codec keyed by `kind` and `schema_version`.
- Invariants live in `__post_init__`.
- A JSON Schema can be generated later for documentation.

### D2. Tensors are never embedded in JSON
- Records hold a `TensorRef`: shape, dtype, device, optional summary stats, and an optional storage key. The in-memory tensor is a runtime-only field excluded from the codec.
- `trace.save(path)` writes `trace.json` plus an optional `tensors.pt`: a plain `dict[str, Tensor]`, loaded with `torch.load(weights_only=True)` so no pickled code executes.
- `safetensors` is an optional alternative later.
- JSON round-trips all semantic information. Tensors round-trip only if retained.

### D3. Hook engine: eager, context-scoped, no persistent hooks
- Hooks exist **only** inside a `recording()` block (`trace()` is built on it). At rest, the model has zero BeyondNN hooks.
- A single `HookSession` owns every registration through an `ExitStack`. Removal happens in `finally`.
- Forward hooks use `always_call=True`.
- **Nesting:** nested sessions are allowed. Each owns its handles, and closing out of order raises *after* cleanup.
- **Leak detection:** an autouse test fixture snapshots `_forward_hooks`, `_forward_pre_hooks`, and `_backward_hooks` on every module of every test model, then asserts they are identical afterwards.
- **Shared modules:** record the canonical path (first in `named_modules()`) plus `aliases`, and a per-module `call_index`. Multiple forward passes inside one `recording()` are distinguished by `pass_index`.
- **Tuple, dict, and dataclass outputs:** flattened with a small internal pytree walker (not the private `torch.utils._pytree`). Each tensor leaf gets an `output_path`.
- Functional ops inside `forward` are invisible. Every trace emits `FUNCTIONAL_OPS_UNOBSERVED`. There is no `torch.fx` capture in v0.

### D4. `instrument()` is a pass-through handle (ADR-002, accepted)

| Option | `isinstance(nn.Module)` | `state_dict` keys unchanged | Mutates user object | Outcome |
|---|---|---|---|---|
| Wrapping `nn.Module` | ✓ | ✗ | ✗ | rejected |
| Swap `__class__` to a mixin subclass | ✓ | ✓ | ✓ | rejected |
| **Pass-through handle** | ✗ | ✓ | ✗ | **accepted (sugar)** |
| **Free functions** | ✓ | ✓ | ✗ | **accepted (primitive)** |

### D5. `explain()` requires a target
- `explain(x, target=…)`. The target is a `Metric` (class logit/prob, logit difference, token log-prob, loss, custom).
- An omitted target on a classifier defaults to argmax, and records `DEFAULT_TARGET_ARGMAX` plus `TargetSpec.defaulted=True`.

### D6. Metrics are callables with a declared signature
```python
class Metric(Protocol):
    name: str
    def spec(self) -> TargetSpec: ...                       # serialisable description
    def __call__(self, output: Any, *, inputs: Any) -> torch.Tensor: ...  # shape [batch]
```
Built-ins: `Logit(cls)`, `Prob(cls)`, `LogitDiff(a, b)`, `LogProb(token_id, position)`, `Loss(fn, y)`, `Output(path)`.
- Custom metrics have a `name` and return a `TargetSpec` with `metric="custom:<name>"`.
- Custom metrics are not reconstructible from JSON. That is recorded, not hidden.

### D7. Interventions are data (Phase 2)
- `InterventionSpec(site=Site(...), op=ZeroAblation() | MeanAblation(ref_id) | Constant(v) | Patch(source_trace_id, source_record_id) | Scale(k) | FeatureDelta(basis_id, idx, delta))`.
- Applying a spec yields an `InterventionRecord`. Measuring the effect yields `CausalEffect` records referencing it.

### D8. Feature bases are explicit objects (Phase 6)
- `FeatureBasis`: `id`, `kind` (`neuron | direction | sae | native`), `encode`, `decode`.
- Feature interventions are encode → modify → decode + reconstruction error. The error norm is recorded, and a `BASIS_RECONSTRUCTION_ERROR` limitation is raised above a threshold.

### D9. Semantic status transitions are enforced in code (ADR-009, accepted)
- `SemanticStatus`: `UNLABELED_FEATURE → PROPOSED_CONCEPT → VALIDATED_CONCEPT | REJECTED_CONCEPT`.
- Only `concepts.validate()` produces the last two, with attached `ValidationResult`s from a protocol that includes detection with counterexamples **and** causal relevance vs random-direction controls.

### D10. Mode B deferred; extension point preserved (ADR-008, accepted)
- Phase 1 site resolution reads an optional `__bnn_sites__` mapping on modules. It is ignored if absent, and it is the only Mode B artifact in Phase 1.
- Native concept slots, when they exist, start at `PROPOSED_CONCEPT`.

### D11. Scaling path, without building for it now
- Retention per site: `"none" | "summary" | "cpu" | "device"`. The default is `"summary"`.
- `TensorRef.storage_key` can point to external storage later.
- Glob site selection (`"blocks.*.mlp"`).
- A `Backend` seam is internal only in Phase 1: `EagerHookBackend`. A later candidate is `NNsightBackend` for remote or large models.
- Eager mode only. `torch.compile` is documented as unsupported in v0.

### D12. Claims are immutable data; testing is a free function (ADR-012, accepted)
- `bnn.test_claim(model, claim, inputs, tests=[...]) -> ClaimEvaluation`.
- There is no `claim.test(...)` method, because claims are schema records and must not hold model references or runner logic.
- A thin `ClaimEvaluation` result object groups `results`, `assessment`, and the produced `records`.

### D13. Test classes avoid the `Test*` prefix
- pytest collects classes whose names start with `Test` when they are imported into test modules.
- Schema and runner names are therefore `ClaimTestSpec`, `ClaimTestResult`, `Outcome`, `Verdict`. They are never `TestSpec` / `TestResult`.

## Mode A vs Mode B

| | Mode A `instrument(model)` | Mode B `InterpretableModule` (deferred) |
|---|---|---|
| Model changes | none | model authored with BeyondNN |
| Sites | inferred from the module tree | declared by the author (`__bnn_sites__`) |
| Concepts | only via external bases or probes | declared slots, still `PROPOSED_CONCEPT` until validated |
| Coverage guarantee | module boundaries only | declared sites are exact |
| Phase | 1+ | after the claim-testing layer is mature, if justified by evidence |

---

## Phase 1 API (final proposal)

Everything below is **proposed, not implemented.**

```python
import torch
import beyondnn as bnn
from beyondnn.models import TinyMLP

model = TinyMLP(d_in=4, d_hidden=16, n_classes=2, seed=0).eval()
x = torch.randn(8, 4)

# 1. Eager one-shot trace (ADR-003). No hooks remain afterwards.
tr = bnn.trace(model, x, sites=["layers.*"], retain="cpu")  # retain: "none"|"summary"(default)|"cpu"|"device"
tr.activations                    # ActivationView (read-only mapping keyed by site key)
rec = tr.activation("layers.1")   # ActivationRecord(site=Site("layers.1","output"), value=TensorRef((8,16),...))
tr.tensor(rec.value)              # torch.Tensor; raises TensorNotRetainedError under retain="summary"
tr.origin(rec)                    # ProvenanceTree: forward_hook @ layers.1, versions, model fingerprint
tr.limitations                    # [FUNCTIONAL_OPS_UNOBSERVED, PARTIAL_SITE_COVERAGE]
tr.save("run1/")                  # run1/trace.json (+ run1/tensors.pt if anything retained)
tr2 = bnn.TraceResult.load("run1/", model=model)   # MODEL_MISMATCH limitation if fingerprint differs

# 2. Context form (ADR-003).
with bnn.recording(model, sites=["layers.*"]) as ctx:
    logits = model(x)
    loss = logits.logsumexp(-1).mean()
tr3 = ctx.result                  # finalised on exit, hooks removed even if the block raised

# Re-used modules and repeated passes are distinguished explicitly.
tr3.activation("block", call_index=1, pass_index=0)

# 3. Pass-through handle (ADR-002). Does not modify the model.
ix = bnn.instrument(model)
ix(x)                             # identical to model(x); no hooks installed
ix.module is model                # True
ix.eval() is ix                   # train/eval/to/cpu return the handle
tr4 = ix.trace(x, sites=["layers.*"])

# 4. explain(): in Phase 1 it reports only what was measured, and says what is missing.
resp = ix.explain(x[:1], target=bnn.targets.Logit(1))
resp.input                        # InputRecord  (OBSERVED)
resp.output                       # OutputRecord (OBSERVED), target_value=2.31
resp.why.activations              # MEASURED records
resp.why.attributions             # ()   Phase 3
resp.why.causal_effects           # ()   Phase 2
resp.why.claims                   # ()   Phase 2+
resp.why.assessments              # ()
resp.why.limitations              # NO_ATTRIBUTION, NO_CAUSAL_EVIDENCE, NO_CLAIMS_TESTED, FUNCTIONAL_OPS_UNOBSERVED
print(resp.why.render())
```

Phase 1 render (deterministic template, no LLM):

```text
INPUT   [observed]  tensor (1, 4)
WHY
  Measured internal states:
    layers.0 output   shape (1, 16)   mean 0.12   l2 3.41
    layers.1 output   shape (1, 2)
  Input evidence:     none computed        [NO_ATTRIBUTION]
  Causal evidence:    none computed        [NO_CAUSAL_EVIDENCE]
  Claims:             none tested          [NO_CLAIMS_TESTED]
  Limitations:
    - Computation between module boundaries is not observed.
OUTPUT  [observed]  logit[1] = 2.31
```

### Claim API (orientation; schema in Phase 1, runners from Phase 2)

```python
claim = bnn.claims.necessary(
    subject=bnn.site("layers.1", index=(..., 41)),     # neuron 41 of layers.1 output
    target=bnn.targets.Logit(2),
    scope=bnn.inputs_ref(X_eval, name="eval-512"),
    min_effect=0.10, relative=True,                    # expectation: ≥10% drop
    statement="layers.1 unit 41 contributes to class 2",
    source=bnn.ClaimSource.user(),
)
ev = bnn.test_claim(
    model, claim, inputs=X_eval,
    tests=[
        bnn.tests.Ablation(op=bnn.MeanAblation(ref=X_train)),
        bnn.tests.RandomBaseline(n=50, match="same_site_count", alpha=0.05),
        bnn.tests.Stability(perturb=my_invariance_fn, n=20),  # Phase 5; user-declared invariance
    ],
)
ev.results       # (ClaimTestResult(outcome=SUPPORTS, statistics={"effect": -0.31, ...}), ...)
ev.assessment    # Assessment(verdict=SUPPORTED, components={"effect_vs_random_z": 4.1, ...},
                 #            required_but_missing=(), policy="default/v1")
```

## Repository structure

Items marked `†` exist now. Everything else is created by the Phase 1 milestone named in `docs/roadmap/PHASE_1_PLAN.md`.

```
beyondnn/                         (repo root)
  README.md† LICENSE† CONTRIBUTING.md† CODE_OF_CONDUCT.md† SECURITY.md† CHANGELOG.md†
  pyproject.toml† .gitignore† .pre-commit-config.yaml†
  .github/workflows/ci.yml†
  beyondnn/                       (flat layout, ADR-011)
    __init__.py† py.typed†
    schema/      status.py records.py provenance.py limitations.py claims.py tensors.py codec.py   M1.1–M1.2, M1.7
    core/        sites.py hooks.py trace.py instrument.py targets.py                              M1.4–M1.8
    explain/     why.py render.py                                                                 M1.8
    models/      tiny_mlp.py tiny_cnn.py tiny_transformer.py                                      M1.3
    (interventions/ causal/ attribution/ claims/ concepts/ audit/ adapters/ — later phases)
  tests/†        test_package.py† + Phase 1 suites
  benchmarks/    bench_trace_overhead.py                                                          M1.9
  examples/      01_trace_mlp.py 02_trace_cnn.py 03_trace_transformer.py                          M1.8
  docs/†         README.md design/ research/ roadmap/ experiments/ decisions/
```

The schema's claim records live in `schema/claims.py` (pure data). The later `beyondnn/claims/` package holds runners and policies.
