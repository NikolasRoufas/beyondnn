# Architecture Proposal

Status: **design; implementation in progress.** Milestone M1.1 (schema) is implemented, see `TRACE_SCHEMA_PROPOSAL.md` Part A. Resolved decisions are recorded in
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

The names are conceptual. In schema 0.1 (implemented in M1.1):
- **Measurement** is the evidence records (`InputRecord`, `OutputRecord`, `ActivationRecord`; later kinds in later schema versions);
- **Claim** is `Claim`;
- **Test** is `ClaimTestSpec`;
- **Evidence** is `ClaimTestResult`, citing `EvidenceRef`s;
- **Assessment** is `Assessment`, under an explicit `AssessmentPolicy` (ADR-018).

Presentation is not implemented yet.

**Separation rules:**
1. **Measurements know nothing about claims.** A trace is valid with zero claims.
2. **A claim has no status field.** Status is a function of (claim, results, policy), computed by an Assessment. Storing status on the claim would allow stale or unsupported status. (This is a deliberate deviation from the `Claim(status=…)` sketch.)
3. **A test declares, before running, which relations it can bear on and what criteria decide its outcome.** The spec's content-derived id (ADR-016) covers its criteria (`criteria_digest` fingerprints them alone). Results reference the spec id, so changing the criteria after seeing results produces a different spec that the results do not refer to.
4. **An assessment is reproducible:** `assess(claim, results, policy)` is pure. A saved assessment can be recomputed and compared on load.
5. **Presentation may only cite.** Rendered and generated text carries `supporting_records`, and nothing may cite generated text as support.

## How the design makes the four classic mistakes hard

| Mistake | Mechanism | Where enforced |
|---|---|---|
| Correlation presented as causation | A `SUPPORTS`/`CONTRADICTS` result on a causal relation must cite `INTERVENTIONAL`/`ESTIMATED_CAUSAL` evidence about the claim's estimand. Finite-sample evidence cannot decide a population claim (ADR-013). A causal claim cannot be assessed without a policy naming protocols (ADR-018). | `ClaimTestResult` and `derive_verdict` (implemented, M1.1); protocol registry (Phase 2) |
| Probe success presented as model use | Detection/probe tests apply only to `ENCODES`. An `ENCODES` claim never implies `NECESSARY_FOR`. `VALIDATED_CONCEPT` requires both detection and causal tests (ADR-009). | registry, `Concept` invariant |
| Generated labels presented as validated concepts | Labels enter as `PROPOSED_CONCEPT` with a `GENERATED` source. Only `validate()` can produce `VALIDATED_CONCEPT`. GENERATED records can never be cited as evidence, and nothing but GENERATED derives from them. | `EvidenceRef` and derivation rules (implemented, M1.1); `Concept` (Phase 6) |
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

**Package layout as implemented:**
- `beyondnn/schema/` holds pure data and imports no torch.
- `beyondnn/provenance/` (M1.2) holds torch-dependent collection: model fingerprints, environment, RNG, provenance records. It depends on `schema`, never the reverse.
- `beyondnn/_testing/` (M1.3) holds **internal, unstable** reference models for BeyondNN's own tests. It is not exported from `beyondnn`, and not BeyondNN model architectures. The `beyondnn.models` namespace is reserved for possible real architectures later.
- `beyondnn/core/` (M1.4, M1.5) is the torch-dependent execution layer: site resolution (`sites.py`) and hook sessions (`hooks.py`). It depends on `schema` (and later `provenance`), never the reverse.
- Future tracing (`core/`) depends on both.
- The top-level `beyondnn` package imports only `schema`, so `import beyondnn` stays torch-free until tracing exists.

**Rules:**
- Dependencies point downward only.
- `schema` imports only the stdlib. Building a `TensorRef` from a live tensor belongs to a torch-dependent layer (planned for M1.6 tracing), never to `schema`. A subprocess test asserts that `import beyondnn.schema` does not load torch.
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
- Custom metrics have a `name` and a caller-declared `implementation_revision`/`config` (ADR-029). Their `TargetSpec` is `metric="custom:<name>"` with the declared identity in its params.
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

### Module site resolution (M1.4, `beyondnn/core/sites.py`)

`resolve_sites(model, patterns, *, io=SiteIO.OUTPUT) -> tuple[ResolvedSite, ...]` is internal to `beyondnn.core` and not exported from `beyondnn`. It binds schema `Site`s to live modules. It never runs forward, installs hooks, reads tensors, or touches RNG.

**Wildcards operate on dot-separated path segments, not characters.**

| Pattern | Meaning |
|---|---|
| `blocks.0.attn` | exact path |
| `blocks.*` | `*` is exactly one segment (`blocks.0`, `blocks.1`; not `blocks`, not `blocks.0.attn`) |
| `blocks.**` | `**` is zero or more segments (`blocks`, `blocks.0`, `blocks.0.attn`, …) |
| `**.attn` | `attn` at any depth |
| `**` | every named descendant module |
| `""` | the root module, and only the root |

Grammar:
- A pattern is `""`, or segments joined by `.`.
- A segment is a literal (non-empty, no `*`, no whitespace), `*`, or `**`.
- **Invalid:** partial wildcards (`block*`, `*block`), `***`, empty segments (`a..b`, `.a`, `a.`), consecutive `**`, and whitespace.
- **No** regex, character classes, brace expansion, or escaping.

Semantics:
- **Root.** The root is selected **only** by `""`. Wildcards (`*`, `**`) never select it, which prevents accidental whole-model hooks.
- **Strict.** Every pattern must match at least one module. Otherwise `UnmatchedPatternError` lists all unmatched patterns with up to 5 similar paths, and nothing is returned. Invalid patterns raise `InvalidPatternError` before any matching. Both subclass `SiteResolutionError`.
- **Order.** Results follow `named_modules(remove_duplicate=False)` traversal order, independent of pattern order.
- **Dedup by exact path.** A path selected by several patterns appears once.
- **Paths, not objects.** A module registered under two names (aliases) yields two sites; `ResolvedSite.aliases` lists the other paths. Tied *parameters* never merge module sites.
- **Not filtered** by leaf status, parameter count, or buffer count. Containers and parameterless modules (`pool`, `gap`) are ordinary sites.
- **Alias paths vs repeated calls.** Aliases are several paths to one object, and M1.4 handles them. Repeated calls are one path invoked several times per forward (e.g. `TinyMLP.shared`); they are call indices, handled in M1.5/M1.6.
- **Structured outputs.** `Site.output_path` is not inferred here. It needs produced tensors (M1.5/M1.6).

### Hook sessions (M1.5, `beyondnn/core/hooks.py`)

`HookSession` is the internal execution primitive under the future `trace()`/`recording()`. It emits ephemeral `HookEvent`s to a sink, and never stores or modifies values. The semantics are in ADR-021:
- physical-module hook dedup;
- `pass_index` and `call_index` rules, including failure consumption and `-1` for out-of-pass calls;
- stack pairing of INPUT and OUTPUT;
- default refusal of aliased modules, with an internal `GROUP` mode;
- guaranteed removal of only BeyondNN's hooks on every exit path.

A hook event proves only that *this module object executed and this value crossed its boundary*.

### Interventions (Phase 2, `beyondnn/interventions/`, ADR-028)

```python
import beyondnn as bnn
iv = bnn.interventions
result = bnn.intervene(model.eval(), x, intervention=iv.zero("a"), metric=iv.metrics.select([0, 0]))
result.value            # intervention_value - baseline_value (INTERVENTIONAL CausalEffect, INSTANCE)
result.trace            # one trace: CLEAN baseline pass, INTERVENTION pass (activations stay MEASURED)
```

- **Dependencies:** schema ← provenance ← core ← interventions.
- **Operations:** zero, constant, and patch (from a source pass in the same recording).
- **Refusals:** training mode, RNG consumption, model-state drift between the paired passes, and interventions that did not apply.
- **Claims:** decided only by the declared `intervention_threshold` protocol. Sufficiency cannot be assessed.

### Faithfulness tests (Phase 5, `beyondnn/faithfulness/`, ADR-032, ADR-033)

```python
test = F.comprehensiveness(target=m, min_drop=1.0, statement="...", controls=F.controls(200, seed=0))
result = F.run(model, x, test=test, selection=F.top_k(attr, k=2))   # a ClaimTestResult + raw effects
curve = F.curve(model, x, ranking=F.ranking(attr), target=m, mode="remove")
bnn.compose(trace, attributions=[attr], faithfulness=[result, curve])
```

- **Orchestration only.** Perturbations are Phase-2 interventions: unit-level removal and retention and model-input sites (ADR-032), run in one recording per test (`compare_family`). Selections come from Phase-3 attributions; presentation is Phase 4.
- **Records.**
  - Raw measurements are INTERVENTIONAL effects.
  - `comprehensiveness` and `sufficiency` are claim tests (`ClaimTestResult`).
  - Curves, stability, counterexamples, paired controls, and method diagnostics are `ProtocolResult`s.
  - `EvidenceSelection` records what was tested.
  - There is no new status and no score.
- **Controls and statistics.** Matched random selections from a seeded local generator; descriptive fractions, Monte-Carlo p, and a paired sign-flip test at dataset level.

### Evidence synthesis (Phase 4, `beyondnn/explain/`, ADR-031)

```python
trace = handle.trace(x, sites=["p", "q"])                     # OBSERVED + MEASURED
attr = handle.attribute(x, target=m, method=..., at=...)      # ATTRIBUTED (explicit)
effect = handle.intervene(x, intervention=..., metric=m)      # INTERVENTIONAL (explicit)
response = bnn.compose(trace, attributions=[attr], interventions=[effect],
                       policies=[A.ATTRIBUTION_POLICY, iv.INTERVENTION_POLICY])
response.why.by_status(EvidenceStatus.ATTRIBUTED)   # original records, never merged
print(response.render())
```

- **Methods run explicitly; composition only reads.** The `EvidenceBundle` validates one explanation context: model, declaration, exact sample, scope, and target. It re-derives recorded claim results and refuses anything incompatible.
- **What `Why` shows:** status sections, declared claims with every test and assessment, the limitations union, `Coverage` (not confidence), and unanswered questions.

### Attribution (Phase 3, `beyondnn/attribution/`, ADR-030)

```python
import beyondnn as bnn
A, iv = bnn.attribution, bnn.interventions
r = bnn.attribute(model.eval(), x, target=iv.metrics.select([0, 3]),
                  method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64))
r.value                 # raw attribution tensor (shape of x); r.record is ATTRIBUTED
r.completeness_delta    # IG numerical diagnostic, not a confidence score
bnn.attribute(lm.eval(), ids, target=..., method=..., at=A.layer("token_embedding"),
              reductions=[A.reduce("sum", (-1,))])   # embedding dims -> explicit per-position score
```

- **Dependencies:** schema ← provenance ← core ← interventions (metrics, `sample_id`) ← attribution. Captum is an optional adapter, imported only when a Captum method is used.
- **Execution:**
  - gradient passes run outside tracing, on detached clones, with `torch.autograd.grad` only;
  - one CLEAN traced reference pass then provides provenance and the observed/measured records the attribution derives from.
- **Refusals:**
  - training mode;
  - foreign forward, backward, or tensor gradient hooks;
  - alias paths;
  - an undeclared or ambiguous call;
  - integer inputs for input attribution;
  - RNG consumption;
  - a target that is not reproducible;
  - any change to the model, gradients, caller tensors, or hooks (parameter gradients are restored first).
- **Claims:** `attribution_threshold` justifies only ATTRIBUTED_TO.
- **Explain:** `explain()` never runs attribution. `Why.attributions` shows attribution records when a trace has them (Phase 4 synthesises views).

## Mode A vs Mode B

| | Mode A `instrument(model)` | Mode B `InterpretableModule` (deferred) |
|---|---|---|
| Model changes | none | model authored with BeyondNN |
| Sites | inferred from the module tree | declared by the author (`__bnn_sites__`) |
| Concepts | only via external bases or probes | declared slots, still `PROPOSED_CONCEPT` until validated |
| Coverage guarantee | module boundaries only | declared sites are exact |
| Phase | 1+ | after the claim-testing layer is mature, if justified by evidence |

---

## Implemented API (M1.6)

```python
import torch
import beyondnn as bnn
from beyondnn._testing.models import TinyTransformer   # internal reference model

model = TinyTransformer()
tokens = torch.tensor([[1, 5, 9, 3]])

trace = bnn.trace(model, tokens, sites=["blocks.*.attn"])          # outputs; input_sites=[...] for inputs
trace.input, trace.output               # OBSERVED root records
trace.activations                       # MEASURED, execution order, one per tensor leaf
trace.activation("blocks.0.attn", output_path="[0]")               # strict lookup
trace.origin(trace.activations[0])      # ProvenanceRecord (model state at the start of the pass)
trace.limitations                       # e.g. FUNCTIONAL_OPS_UNOBSERVED, PARTIAL_SITE_COVERAGE

with bnn.recording(model, sites=["lm_head"], retention="cpu") as ctx:
    logits = model(tokens)              # the live output stays the caller's
    model(tokens)                       # second pass
ctx.result.activation("lm_head", pass_index=1)
```

M1.8 adds `handle = bnn.instrument(model)` and `response = handle.explain(x, sites=[...])` (ADR-026):
- `response.input` and `response.output` are OBSERVED;
- `response.why.activations` are MEASURED;
- `response.why.limitations` include `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE`, `NO_CLAIMS_TESTED`.

**Phase-1 WHY is measured internal evidence, not a causal or attributed explanation.**

The `Phase 1 API (final proposal)` below is kept for history. Where it differs from the above, the above is authoritative: there is no `target=` and no attribute-delegating handle.

## Phase 1 API (final proposal)

Everything below is **proposed, not implemented.**

```python
import torch
import beyondnn as bnn
from beyondnn._testing.models import TinyMLP  # internal reference model (M1.3), not public API

model = TinyMLP(d_in=4, d_hidden=8, d_out=3, seed=0).eval()
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
    _testing/    models.py (TinyMLP, TinyCNN, TinyTransformer; internal, unstable)                  M1.3 ✓
    (interventions/ causal/ attribution/ claims/ concepts/ audit/ adapters/ — later phases)
  tests/†        test_package.py† + Phase 1 suites
  benchmarks/    bench_trace_overhead.py                                                          M1.9
  examples/      01_trace_mlp.py 02_trace_cnn.py 03_trace_transformer.py                          M1.8
  docs/†         README.md design/ research/ roadmap/ experiments/ decisions/
```

The schema's claim records live in `schema/claims.py` (pure data). The later `beyondnn/claims/` package holds runners and policies.
