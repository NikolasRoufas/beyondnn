# Phase 2 Gate Report: Causal Interventions

- **Date:** 2026-09-25
- **Baseline:** Phase 1 `34cfa89`.
- **Plan:** `docs/PHASE_2_PLAN.md` (`b6889c4`, written before implementation, including the expected ground-truth effects).
- **Decisions:** ADR-028.
- **Repository:** local only, never pushed.

## 1. Executive summary

Phase 2 moves BeyondNN from OBSERVED/MEASURED evidence to genuine **INTERVENTIONAL** evidence. It does so through controlled, paired comparisons: a CLEAN baseline pass and an INTERVENTION pass of the same input, in one recording, with refusal of anything that would confound the comparison. The resulting `CausalEffect` records the metric difference (`intervention − baseline`) over an explicit estimand (one input, or exactly a given finite sample).

Activations recorded under intervention remain MEASURED. Claims are decided only by a threshold test declared in advance. Sufficiency cannot be assessed.

**Gate: GO WITH EXPLICIT LIMITATIONS** (§11).

## 2. What was implemented

| Piece | Details |
|---|---|
| Schema | `InterventionRecord` (module-output leaf, call index, ZERO/CONSTANT/PATCH, retained digest-checked values, source `RecordRef`); `MetricSpec`; `CausalEffect` (derived status, sign convention, lineage rules); container check that INTERVENTION provenance resolves to its spec |
| Runtime | `bnn.intervene`, `bnn.interventions.{zero, constant, patch, intervene_sample, metrics, make_claim, threshold_spec, sample_id, INTERVENTION_POLICY, PROTOCOLS, check_policy, InterventionResult}`, `handle.intervene` |
| Hooks | `HookSession.owned_forward_hook` (BeyondNN-owned, recognised by the foreign-hook check, runs before output observation), `current_invocation` |
| Claims | `intervention_threshold` v1 protocol plus registry. NECESSARY_FOR, DECREASES, and INCREASES only |
| Ground truth | `beyondnn/_testing/causal_models.py`: additive, gated, redundant, interaction |
| Limitations | `ZERO_ABLATION_MAY_BE_OOD`, `CONSTANT_REPLACEMENT_MAY_BE_OOD`, `PATCH_SOURCE_CONTEXT_DIFFERS`, `CUSTOM_METRIC_UNVERIFIED` |
| Benchmark | `benchmarks/bench_interventions.py` |

## 3. Semantics

- **Intervention spec.** `InterventionRecord(site=Site(module, OUTPUT, output_path), call_index, operation, constant | value, source)`. Its content-derived id is the `intervention_id` in `ExecutionContext`. It cannot be ambiguous: exactly one module path (aliases refused), one tensor leaf, and one call.
  - **Zero ablation:** `zeros_like(leaf)`. Always carries `ZERO_ABLATION_MAY_BE_OOD`; zero is never presented as neutral.
  - **Constant:** a finite scalar (filled with `full_like`), or a tensor of **exactly** the leaf's shape and dtype. There is no broadcasting and no dtype conversion. The tensor is retained and digest-checked. Carries `CONSTANT_REPLACEMENT_MAY_BE_OOD`.
  - **Patch:** the source runs as a CLEAN pass *in the same recording*. The captured leaf is retained, its content digest recorded, and the spec references the MEASURED source `ActivationRecord` (so source site, pass, call, and leaf are all verifiable). A source input that differs from the target gives `PATCH_SOURCE_CONTEXT_DIFFERS`. Raw unexplained tensors are CONSTANT, never PATCH.
- **Effect.** `effect = intervention_value − baseline_value`, for every metric.
  - **INSTANCE:** exact, derived from the two paired `OutputRecord`s. Its estimand is `sample_id` = SHA-256 of the input tensors and JSON arguments.
  - **FINITE_SAMPLE:** the mean over exactly `n` instance effects. Also INTERVENTIONAL, because it claims nothing beyond those samples.
  - **ESTIMATED_CAUSAL:** only POPULATION with a non-exact estimator. **Never produced in Phase 2** (verified).
- **Provenance.**
  - The baseline pass is CLEAN with no `intervention_id`.
  - The intervention pass is INTERVENTION with `intervention_id`, and its model identity is equal to the baseline's.
  - The effect has its own provenance (method `intervention_effect`).
  - Everything is in one trace and persists with the M1.7 machinery.
- **Pairing (refused, never reported, with `StatefulComparisonError` / `StochasticComparisonError` / `InterventionNotAppliedError`):**
  - any module in training mode;
  - CPU RNG state changed by either pass;
  - model-state fingerprint, execution conditions (training/grad/device/randomness), or inputs (in-place modification) differing between passes;
  - the target call not run exactly once.

  Both passes run under `no_grad`.
- **Claims.** `intervention_threshold` v1: `min_effect` is declared before running.
  - NECESSARY_FOR / DECREASES: SUPPORTS iff `effect ≤ −min_effect`, otherwise CONTRADICTS.
  - INCREASES: SUPPORTS iff `effect ≥ min_effect`.
  - Any mismatch of site/leaf, metric, estimand, or operation gives NOT_APPLICABLE.
  - Only precisely scoped claims can be supported ("under zero ablation of S on input X, M decreased by at least T").
  - SUFFICIENT_FOR has no protocol and cannot be assessed. `check_policy` rejects policies that use a protocol for a relation it does not justify.

## 4. Ground-truth results (pre-registered; float-exact)

| Model | Intervention | Expected | Observed |
|---|---|---|---|
| Additive (3, 5) | zero `a` | −3 | −3 |
| Additive (−2, 0.5) | zero `a` | 2 | 2 |
| Additive (3, 5) | constant 10 | 7 | 7 |
| Additive (3, 5) | patch from (7, 1) | 4 | 4 |
| Additive (3, 5) | identical-source patch / own value | 0 | 0 |
| Gated (3, 5) | zero gate | −15 (output 0) | −15 (0) |
| Redundant (3, 5) | zero `p` | −3, output stays 3 | −3, 3 |
| Interaction (3, 0) / (3, 5) | zero `a` | 0 / −15 | 0 / −15 |
| Additive sample {1, 2, 6} | zero `a` | mean −3 | −3 |

On the redundant model, "p is necessary with `min_effect = 6`" was **CONTRADICTED**. An ablation effect did not become a necessity claim.

## 5. Validation

| Check | Result |
|---|---|
| Python 3.14.3 / torch 2.12 | **699 passed** (`-W error`, 0 skipped) |
| Python 3.10 / torch 2.14 | 699 passed |
| Python 3.12 / torch 2.14 | 699 passed |
| ruff / ruff format / mypy `--strict` | clean |
| sdist + wheel | built |
| Clean venv install of the wheel (Python 3.12, torch 2.14, `-W error`, outside the checkout) | Phase 1 smoke, Phase 2 patch + save/load smoke, and both README examples: OK |

## 6. Mutation tests (all caught)

| Mutation | Tests that fail |
|---|---|
| intervention hook leaks | 34 |
| baseline runs under intervention mode | 2 |
| intervention pass marked CLEAN | 2 |
| intervened activation labelled INTERVENTIONAL | 23 |
| CausalEffect labelled MEASURED | 7 |
| effect sign reversed | 28 |
| patch source value ignored | 4 |
| zero ablation not applied | 11 |
| stochastic comparison accepted | 2 |
| state-mutating passes paired | 2 |
| causal ClaimTestResult without CausalEffect | 10 |
| GENERATED evidence as causal support | 2 |
| input mutated in place accepted | 1 |
| execution-condition drift accepted | 1 |

Three mutations were initially caught by one test only (baseline mode, stochastic, stateful), and each received an independent second test. Two confounding paths were **found during the review** and fixed before the gate: in-place input mutation and execution-condition drift.

## 7. Benchmark (this machine only; characterisation, no thresholds)

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64, 4 threads; warm-up 3, 30 iterations; median / p90 ms.

| model | site | forward | baseline trace | zero-ablation comparison | patch comparison |
|---|---|---|---|---|---|
| TinyMLP | shared | 0.023 / 0.023 | 0.669 / 0.752 | 1.460 / 1.502 | 2.034 / 2.092 |
| TinyTransformer | blocks.0.attn | 0.251 / 0.261 | 1.311 / 1.352 | 2.661 / 2.698 | 3.810 / 3.843 |

A comparison costs about two traced passes (three for patching) plus effect bookkeeping. This is linear in passes and not pathological, so there was no optimisation.

## 8. Known limitations

- Module **output** interventions only. No input, parameter, attention-head, or functional-op interventions.
- One leaf, one call per intervention. There are no multi-site interventions.
- Patching is only from a source pass in the same comparison, not from external saved traces. Finite-sample comparisons support ZERO/CONSTANT only.
- Effects are INSTANCE or FINITE_SAMPLE only. There are no population estimates or confidence intervals.
- Stochastic models (anything consuming CPU RNG) and stateful or training-mode models are refused rather than handled. Only the CPU generator is checked.
- Scalar metrics only. Custom metrics are unverifiable (flagged).
- The claim protocol is threshold-only. SUFFICIENT_FOR is not assessable.
- Out-of-distribution concerns for zero and constant replacement are flagged, not quantified.
- All Phase 1 limitations carry over (CPU-only, module-boundary coverage, no code identity, no user hooks during tracing).

## 9. Scientific risks

1. **An INTERVENTIONAL effect is exact for *this* intervention on *these* inputs.** Readers may still generalise it to necessity or mechanism. The redundant and interaction tests, the estimand, and the limitations are the safeguards.
2. **Zero/constant ablations are off-distribution by construction.** Mean ablation and resampling baselines are future work.
3. **Refusing stochastic models excludes dropout-at-inference and sampling-based models** until controlled RNG replay is designed.

## 10. Engineering risks

- The single-recording design keeps every comparison in one trace. A large multi-sample comparison therefore produces a single large trace.
- The fingerprint is taken once per pass, so a comparison costs a full fingerprint two or three times.
- Several private torch internals are relied on (inherited from Phase 1).

## 11. Gate

**GO WITH EXPLICIT LIMITATIONS.**

| Criterion | Status |
|---|---|
| Analytical ground-truth effects reproduced | ✅ (all, float-exact) |
| No leaked intervention hooks | ✅ (normal, forward error, replacement error, metric error, BaseException) |
| Baseline/intervention provenance correct | ✅ |
| Effect status semantics correct | ✅ (INTERVENTIONAL only for effects; activations MEASURED; no ESTIMATED_CAUSAL produced) |
| Controlled pairing correct | ✅ (training, RNG, state, execution, input mutation, application all checked) |
| Persistence round trip correct | ✅ (spec, patch tensor, effect, claims, limitations; tampered patch tensor rejected) |
| Claim integration cannot fabricate causal support | ✅ (schema requires a CausalEffect about the claim's estimand; declared thresholds; mismatch gives NOT_APPLICABLE; sufficiency unassessable) |
| Limitations explicit | ✅ |
| All validation green | ✅ |

There is **no known silent intervention misattribution.** The limits in §8 are scope boundaries that produce refusals or flags, not silent errors.

**Before Phase 3:** owner review of this report and ADR-028. **Public release** remains blocked by RB-1/RB-2 (contact placeholders) and RB-3 (CI never run). Nothing has been pushed or published.
