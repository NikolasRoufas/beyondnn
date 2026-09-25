# Roadmap

**Rules for advancing:**
- Phases advance only through a written go/no-go in `docs/roadmap/PHASE_<N>_REPORT.md`.
- Each report covers: what was implemented, tests, known limitations, performance, failures, design concerns, and a recommendation for the next phase.
- Decisions made along the way are appended to `docs/decisions/ARCHITECTURE_DECISIONS.md`.
- Experiments go in `docs/experiments/EXPERIMENT_LOG.md`, and negative results stay.

The conceptual pipeline every phase serves (see `docs/design/ARCHITECTURE_PROPOSAL.md`):

```
Measurement → Claim → Test → Evidence → Assessment → WHY presentation
```

| Phase | Adds to the pipeline |
|---|---|
| 1 | Measurement (observe/measure), plus the *types* for every later stage |
| 2 | Interventional measurement, the first claim tests, the Assessment policy |
| 3 | Attribution measurement plus `ATTRIBUTED_TO` claims |
| 4 | WHY presentation |
| 5 | More claim tests (faithfulness suite) |
| 6 | Concepts, validated via `ENCODES` + causal claims |
| 7 | Audit (runs claim tests over datasets) |
| 8 | External comparison |

## Phase 1: Foundations

Detailed plan: [`PHASE_1_PLAN.md`](PHASE_1_PLAN.md) (milestones M1.0–M1.10).

**Goal:** a trace schema and hook engine that researchers can trust.
- Hooks never leak.
- Results serialise losslessly.
- Every record type through `Assessment` is defined and round-trips.
- The same test suite passes on an MLP, a CNN, and a tiny transformer.

**Out of scope:** interventions, attribution, test runners, concepts logic, adapters, GPU-specific code, visualisation, `torch.compile`, and Mode B.

## Phase 2: Interventions, causal effects, first claim tests
- **`Study` container (ADR-015):** multiple `TraceResult`s, multi-input claim test results, dataset-level assessments, population estimates, and cross-input statistics.
- The `CausalEffect` record, designed around `Estimand` (ADR-013). Finite-sample aggregates reference their individual per-input effects. Population estimates are a separate `ESTIMATED_CAUSAL` step.
- `InterventionSpec` operations: zero, mean (explicit reference set), constant, patch-from-trace, scale, and neuron-basis feature delta.
- A `bnn.intervene(model, spec)` context, with the same cleanup test matrix as hooks.
- `Metric` evaluation, `measure_effect`, and `CausalEffect` (single-input exact, and aggregate with bootstrap CI).
- Distribution checks leading to `OFF_DISTRIBUTION_INTERVENTION`.
- **Claims:** `bnn.test_claim`, the runner registry with relation compatibility, the tests `ablation_necessity/v1`, `random_baseline/v1`, `sufficiency_patch/v1`, and `scaling_direction/v1`, and assessment policy `default/v1` with its own documentation page.
- **First ground-truth suite:** synthetic models where A is causal, B is correlated but unused, and C is noise.
  - Built with hand-set weights (exact ground truth) and also trained (does training preserve it?).
  - Expected results are written *before* running, and outcomes are logged whatever they are.

## Phase 3: Attribution
- Native gradient and input × gradient.
- A Captum adapter (IG, FeatureAblation, Occlusion, LayerIntegratedGradients). It must be numerically equal to direct Captum calls.
- `EvidenceSpan` derivation. `ATTRIBUTED_TO` claims and the `method_agreement/v1` test.
- **Experiment:** do top-attributed units pass `ablation_necessity` more often than random units on the Phase 2 suite? Logged, not assumed.

## Phase 4: WHY
- `Why` as a view and `render()` with status-bound vocabulary, including a lint test on templates.
- A template `summary()` with `supporting_records`. An optional LLM summariser goes behind an extra and is always `GENERATED`.
- `explain(x, target, plan)`: proposes candidate claims (`ClaimSource.method`), runs the plan's tests, and assesses.

## Phase 5: Faithfulness tests
- New test kinds: `comprehensiveness/v1`, `sufficiency/v1` (input and internal), `stability/v1` (user-declared invariances only), and `counterexample/v1`.
- Assessment components are exposed. There is no aggregate score (ADR-007).
- Every test gets a documentation page covering its formal definition, implementation, interpretation, and limitations.
- Also Research Question 8: can any aggregate predict held-out outcomes on ground-truth models?

## Phase 6: Concepts
- `FeatureBasis` (neuron, direction; SAE via adapter).
- The `SemanticStatus` lifecycle, and `concepts.validate()` implemented as testing an `ENCODES` claim plus a causal claim, with counterexamples and random-direction controls.

## Phase 7: Audit
- `bnn.audit(model, dataset, plan) -> AuditReport` with `metadata` (versions, seed, device, config, methods).
- It uses only tests and metrics that already exist with documentation.

## Phase 8: Benchmark
- Capability and correctness vs nnsight, Captum, TransformerLens, and pyvene on the tiny models, plus runtime and memory.
- Ease of use is assessed through task scripts and reported qualitatively.

## Hard gates

- **Before M1.6 (trace containers):** resolve a caller-declared model configuration / implementation revision in provenance, so that behaviourally different models with the same FULL v1 fingerprint cannot silently share provenance identity when the distinction is known (see the M1.3 finding below and `PHASE_1_PLAN.md` §M1.6). FULL v1 itself stays unchanged; the `n_heads` case is an accepted, documented limitation of automatic fingerprinting.

## Open investigation items

- **Implementation revision in provenance** (from the M1.2 review). The FULL fingerprint does not hash Python code (ADR-020). Investigate an optional, caller-supplied `implementation_revision` / code revision / repository commit / model revision for reproducibility. Automatic source hashing is out of scope. There is no schema change until implementation evidence shows it is needed.
  - **Evidence from M1.3:** `TinyTransformer(n_heads=2)` and `n_heads=4` with the same seed have identical FULL fingerprints but compute different functions, because `n_heads` is a plain Python attribute and parameter shapes are unchanged. Plain constructor hyperparameters are therefore also outside v1 identity. Candidate remedies: a caller-declared model config or revision in provenance, or an opt-in declared-hyperparameter hook. This needs a decision before evidence from differently configured models is compared.

## Release blockers (`BLOCKS_PUBLIC_RELEASE`)

These don't block local development. BeyondNN must not be published (public repo, PyPI) until every item is resolved.

| ID | Item | Where | Resolution needed |
|---|---|---|---|
| RB-1 | Code of Conduct enforcement contact is `[INSERT CONTACT METHOD]` | `CODE_OF_CONDUCT.md` | The project owner chooses a public project contact. Never use a personal address without explicit consent. |
| RB-2 | Security contact is `[INSERT SECURITY CONTACT]` | `SECURITY.md` | Same as RB-1, or enable GitHub private vulnerability reporting. |
| RB-3 | CI has never run on GitHub | `.github/workflows/ci.yml` | Push to a private remote first, and confirm the matrix is green. |

## Explicitly not planned
Large model training, dashboards, transformer-only APIs, auto-labelled features presented as concepts, and a global confidence scalar (without a superseding ADR).
