# Why does BeyondNN need to exist?

Status: the conclusion of this analysis was accepted as ADR-005 (`../decisions/ARCHITECTURE_DECISIONS.md`).

Short answer: **BeyondNN only needs to exist as an evidence-accounting and claim-testing layer. It does not need to exist as another tracing or intervention engine.**

## The hypothesis, tested item by item

The hypothesis is that BeyondNN's value comes from combining ten things under one INPUT → WHY → OUTPUT interface. Here is who already does each one.

| # | Component | Already done well by | BeyondNN novelty if built alone |
|---|---|---|---|
| 1 | Standardised structured traces | Partly: TL `ActivationCache`, circuit-tracer graph JSON. Neither is architecture-agnostic or carries epistemic metadata. | **Real** |
| 2 | Input attribution | Captum (definitively), Inseq | None. Adapt. |
| 3 | Internal activations | PyTorch hooks, nnsight, TL | None |
| 4 | Features | SAELens, circuit-tracer | None for the methods; small for "feature relative to a basis" as a typed object |
| 5 | Concepts | TCAV (Captum), CBM papers | Small. The value is conservative *status* handling, not the methods. |
| 6 | Interventions | nnsight, pyvene, TL | None as an engine. Small for "intervention spec as a serialisable, provenance-bearing record". |
| 7 | Causal validation | pyvene (causal abstraction), circuit-tracer, patching folklore | Partial. Nobody *attaches* validation results to the claims they support. |
| 8 | Explanation confidence | Nobody has a validated one | Real, but **scientifically unsolved**. Could easily become fake. No global score in v0 (ADR-007); per-claim component evidence instead. |
| 9 | Explanation limitations | circuit-tracer error nodes (partial) | **Real** |
| 10 | Interpretability evaluation | Quantus (input maps), MIB, InterpBench, AxBench | Partial. Evaluating *internal* claims across architectures is thin. |

**Verdict:** items 2, 3, 4, and 6 are commodity. If BeyondNN leads with them, it is "nnsight + Captum with more boilerplate", and researchers will not switch. The combination is only differentiated if the unifying object is something the other tools lack. That object is a **typed, serialisable, provenance-bearing record of claims and the evidence for them, where every item is labelled with how it was obtained.**

## Where BeyondNN is actually differentiated

1. **Epistemic typing.** No existing tool distinguishes, at the data-structure level, "I read this activation" from "a gradient estimated this" from "an intervention measured this" from "a model generated this sentence". This is small but real, and cheap to build.
2. **Provenance on every record.** Captum, TL, and nnsight return bare tensors. Reproducing a published patching result today means re-reading the paper's code. A record that carries method, target, baseline, seed, versions, and parent records is useful even for people who never use the rest of BeyondNN.
3. **Claims as testable objects.** This is specified in `../design/TRACE_SCHEMA_PROPOSAL.md` (ADR-012, proposed) and is the strongest idea in this document.
   - "Feature 41 matters" becomes a `Claim(subject=…, relation=NECESSARY_FOR, target=…, scope=…, expectation=…)`.
   - Declared `ClaimTestSpec`s are run against it (ablation, random-baseline comparison, stability, counterexamples).
   - Its standing is a derived `Assessment`.
   - The schema itself refuses to let a non-interventional test support a causal relation.

   No existing tool models the *claim*, rather than the *score*.
4. **Limitations as first-class output.** The error-node idea from circuit-tracer, generalised: every explanation lists coverage gaps, off-distribution interventions, unvalidated labels, and generated text.
5. **Architecture-agnostic faithfulness tests of internal claims.** Quantus covers input maps. MIB and InterpBench cover transformers. A test harness that checks "is this internal component causally necessary, and more so than random components?" for MLPs, CNNs, and transformers alike is thinly served.

## Where it overlaps too much (and what to do)

| Overlap | Decision |
|---|---|
| Tracing and interventions vs nnsight/pyvene/TL | Build a **minimal eager hook engine**, because a trustworthy core has to have zero-magic cleanup semantics and must work with no optional deps. Keep it small (target < 1k LOC). Add an nnsight backend later for scale and remote execution. Never compete on features. |
| Attribution vs Captum | Native gradient and input×gradient only. Everything else goes through the Captum adapter. |
| Input-faithfulness metrics vs Quantus | Implement only comprehensiveness and sufficiency (needed as internal-feature metrics anyway) and random baselines. Add a Quantus adapter for pixel-flipping-style metrics if users ask for it. |
| Ground-truth benchmarks vs Tracr/InterpBench/OpenXAI | Frame ours as the *framework's own correctness tests*. Reuse Tracr/InterpBench models in an adapter when useful. |
| Concepts vs TCAV/CBMs | Reuse TCAV through Captum. BeyondNN owns the concept *lifecycle and status rules*, not detection methods. |
| `InterpretableModule` (Mode B) vs CBM papers | Defer. Concept leakage means a named bottleneck does not guarantee meaning. See `../design/ARCHITECTURE_PROPOSAL.md` §D10 and ADR-008. |

## The honest risk

A "schema + glue" library succeeds only if the schema is adopted, and schemas are adopted when they save people work: reproducibility, reporting, comparison. The plan should therefore make the trace format useful on its own, for example `bnn.from_captum(...)` and `bnn.from_nnsight(...)` converters that let someone wrap *existing* results in provenance. It should not require users to move their whole workflow into BeyondNN.

## Positioning (accepted, ADR-005)

> BeyondNN is an interpretability evidence framework for PyTorch that turns claims about neural-network computation into structured, provenance-aware, testable objects.

Existing tools produce activations, attributions, interventions, and features. BeyondNN records:
- what those results mean;
- where they came from, and what epistemic status they have;
- what claim they support, and what tests were performed;
- what evidence survived, and what remains unknown.

It is **not** a replacement for Captum, nnsight, TransformerLens, or SAELens. It consumes their outputs through adapters where that adds real functionality.

Superseded working tagline (kept for history): *"a PyTorch library for making interpretability claims explicit, reproducible, and testable. It records what was observed, measured, attributed, intervened on, and generated, and it checks whether an explanation survives causal tests."*

## Naming note

"BeyondNN" suggests replacing neural networks, but the framework explicitly builds on them. The name is available on PyPI (checked 2026-09-24). Consider whether the README tagline needs to defuse the implication. This is not blocking.
