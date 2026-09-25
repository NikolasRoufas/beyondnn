# Phase 3 Gate Report: Attribution

- **Date:** 2026-09-25
- **Baseline:** Phase 2 closed at `50aed1a`.
- **Hardening before Phase 3:** `3f3ff27` (ADR-029).
- **Plan:** `docs/PHASE_3_PLAN.md` (`dd1bffc`, written before any attribution code, with the analytic expectations and tolerances).
- **Decisions:** ADR-030.
- **Repository:** local only, never pushed.

## 1. Executive summary

Phase 3 adds genuine **ATTRIBUTED** evidence. An attribution record answers:

> under method A (configuration C, baseline B), for scalar target T on input X, what score was assigned to the elements of tensor S?

- It does not answer what caused the output.
- The status is fixed by record kind, is never MEASURED or INTERVENTIONAL, and nothing merges an attribution with a `CausalEffect`.
- Every analytic ground truth was reproduced, and native IG agrees with Captum IG under equivalent settings.
- Attribution cannot support NECESSARY_FOR or SUFFICIENT_FOR.

**Gate: GO WITH EXPLICIT LIMITATIONS** (§12).

## 2. Phase-2 hardening (before Phase 3): caller-metric identity (ADR-029)

- **Problem:** caller metrics were identified by a friendly name only.
- **Fix:**
  - `metrics.custom(name, fn, *, implementation_revision, config=None)`. The revision is required, so an unversioned caller metric is **refused** (option A): a limitation could not remove the ambiguity.
  - The declaration is a `MetricDeclaration`, recorded as DECLARED (never verified), kept apart from anything measured, as with `ModelDeclaration`.
  - No function serialisation, source scraping, bytecode hashing, or repository lookup.
  - `MetricSpec.target()` is the single metric-to-claim-target mapping and includes the declaration. A claim about revision A is NOT_APPLICABLE to evidence from revision B.
  - `causal_effect` record version 2. A v1 effect on a caller metric fails to load rather than gaining an invented revision.
- **Regression tests:** two functions with the same name, a different revision or config, claims, persistence, and migration.

## 3. What was implemented

| Piece | Details |
|---|---|
| Schema | `AttributionMethodSpec`, `AttributionBaseline` (ZERO / TENSOR / INPUT_TENSOR), `AttributionRecord` (kind `attribution`), `AttributionReduction` (kind `attribution_reduction`), both ATTRIBUTED by kind. New container checks (site/call/pass, shape/dtype, baseline shape, model identity, reduction shape). |
| Targets | Phase-2 built-in metrics with a differentiable form (`Metric.tensor`): `select`, `difference`, `mean`. Exactly one element; caller metrics refused as attribution targets. |
| Native methods | `gradient`, `input_x_gradient`, `integrated_gradients(baseline=..., n_steps, rule)` with rules `riemann_left/right/middle`, `trapezoid` |
| Attributed tensor | `input(i)` (floating point only) or `layer(path, call_index, output_path)` (one leaf of one call of a module output) |
| Captum adapter | `beyondnn.attribution.captum`: `saliency()` (abs=False), `input_x_gradient()`, `integrated_gradients()` (input: IntegratedGradients; layer: LayerIntegratedGradients). Optional extra `beyondnn[captum]` (`captum>=0.9,<0.10`). |
| Reductions | explicit `reduce("sum" / "abs_sum" / "l2", dims)` only, each its own record |
| Claims | central registry `beyondnn.protocols`; `attribution_threshold` v1 (ATTRIBUTED_TO only); `ATTRIBUTION_POLICY`, `threshold_spec`, `make_claim`, `evaluate_claim` |
| Limitations | `ATTRIBUTION_BASELINE_ASSUMPTION`, `ATTRIBUTION_NUMERICAL_APPROXIMATION`, `LAYER_ATTRIBUTION_PARTIAL_COVERAGE`, `DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION` |
| API | `bnn.attribute`, `bnn.attribution`, `handle.attribute`; `Why.attributions` (explain never runs attribution) |
| Ground truth | `beyondnn/_testing/attribution_models.py`: Linear, Irrelevant, Product, Saturating, Twice, Aliased |
| Benchmark | `benchmarks/bench_attribution.py` |

## 4. Semantics

- **Record.** `AttributionRecord(method, target, site, call_index, pass_index, sample_id, baseline, value, target_value, diagnostics)`.
  - **Identity** includes every scientific choice: the method name, implementation and version, params (n_steps, rule, Captum class and fixed settings), the target `MetricSpec`, the baseline kind and tensor digest, the site, leaf, call, and pass, the sample, and the attribution tensor's digest.
  - **Lineage:** the reference pass's `OutputRecord`, plus the `InputRecord` (input site) or the MEASURED `ActivationRecord` of exactly that call.
  - `sample_id` is deliberately not an `Estimand`: attribution is not causal evidence.
- **Target.** One element of the output, selected by a built-in metric. A raw tensor, `None`, an index selecting several elements, or a caller metric is refused. Outputs are never summed.
- **Gradient:** `∂T/∂x` via `torch.autograd.grad` on a detached clone. It is exactly 0 where the target does not depend on the input.
- **Input × gradient:** `x ⊙ ∂T/∂x`, recorded as its own method.
- **IG:** `(x − x') ⊙ Σ w_k ∇T(x' + α_k(x − x'))`.
  - Gradients are accumulated in float64; the result is stored in the attributed dtype.
  - The baseline is required (zero only when asked for) and its shape and dtype must match exactly.
  - `completeness_delta = Σ attr − (T(x) − T(x'))` is recorded as a diagnostic. For layer IG, `T(x')` replaces the layer output with its baseline.
  - The result is never called exact.
- **Internal attribution:**
  - A temporary hook replaces exactly call `k`'s leaf with the evaluation point, so the gradient is with respect to that call only.
  - The number of calls must match across all passes, including the traced one.
  - The container rejects a record whose activation reference is another call.
  - Alias paths are refused before any computation.
  - Functional operations (residual additions, `F.gelu`) have no sites and are not attributable (`LAYER_ATTRIBUTION_PARTIAL_COVERAGE`).
- **Tokens:**
  - Integer inputs are refused for input attribution (`DiscreteInputError`, pointing to layer attribution).
  - `layer("token_embedding")` attributes to embedding dimensions per position. It carries `DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION`.
  - A per-token score exists only through an explicit reduction.
  - Layer IG baselines are either layer-space tensors or `input_baseline(ids)` (the layer activation on baseline ids, which is Captum's convention).
- **Autograd and state guarantees** (checked before and after every run; refused, never reported):
  - model fingerprint and train/eval flags;
  - parameter `.grad` (identity and values; restored before refusing);
  - caller tensors (values, `requires_grad`, `.grad`);
  - hook counts and CPU RNG.
  - Foreign forward, backward, and parameter-gradient hooks are refused up front.
  - The traced reference pass must reproduce the target value and must not modify inputs.
- **Captum integration:**
  - Captum receives detached clones, `internal_batch_size=1`, and batch-1 inputs, so it evaluates exactly the tensors the native code does.
  - Every setting is recorded, including the class actually used (`LayerIntegratedGradients` for layers).
  - Refused:
    - unverified Captum versions;
    - repeated layer calls (Captum hooks every call);
    - layer-space baselines for Captum LayerIG;
    - non-IG Captum layer methods.
  - Importing BeyondNN never imports Captum.

## 5. Ground-truth results (pre-registered in PHASE_3_PLAN.md)

| Model | Point | gradient | input×gradient | IG (zero baseline, riemann_middle) |
|---|---|---|---|---|
| Linear `2x0+3x1` | (3, 5) | [2, 3] | [6, 15] | [6, 15], delta 0 (every rule) |
| Irrelevant `4x0` | (3, 5) | [4, **0**] | [12, **0**] | [12, **0**] |
| Product `x0·x1` | (3, 5) | [5, 3] | [15, 15] | [7.5, 7.5] (exact for middle/trapezoid). Baseline (1, 2): [7, 6], Σ = 13 = 15 − 2 |
| Product | (3, 0) | [0, 3] | [0, 0] | [0, 0] |
| Saturating `tanh x0 + x1` | (3, 1) | [0.0098660, 1] | [0.029598, 1] | [0.9950565, 1] |

- **Saturating IG error vs tanh 3 (declared bound 2.25/n² + 1e-5):**

  | n | observed error | bound |
  |---|---|---|
  | 16 | 2.9e-5 | 8.8e-3 |
  | 64 | 1.8e-6 | 5.5e-4 |
  | 256 | 1.5e-7 | 3.4e-5 |

- **One-sided rules on Product:** completeness delta ∓15/n exactly (left: −3.75 at n = 4, −1.5 at n = 10; right: +).
- **Repeated call (Twice):** the gradient at call 0 is 2 and at call 1 is 1, as predicted.
- **Negative example (Phase-2 Redundant model):**
  - input × gradient and IG at `p` are both 3.0, and "p ATTRIBUTED_TO y ≥ 2" is SUPPORTED;
  - yet zero-ablating `p` leaves the output at 3, and "p NECESSARY_FOR y, min_effect 6" is CONTRADICTED;
  - a `CausalEffect` cannot derive from the attribution;
  - an attribution record cannot be cited for NECESSARY_FOR;
  - a policy mapping NECESSARY_FOR to `attribution_threshold` is rejected.

  **High attribution is not necessity.**

## 6. Native / Captum cross-check (Captum 0.9.0)

| Comparison | Max abs difference | Declared tolerance |
|---|---|---|
| IG, rules riemann_left/right/middle, n = 16, on Product, Saturating, TinyMLP, TinyCNN | 1.2e-7 | rtol 1e-4, atol 1e-5 |
| gradient vs `Saliency(abs=False)`; input×gradient vs `InputXGradient` | 1.2e-7 | atol 1e-6 |
| layer IG at `TinyTransformer.token_embedding`, input baseline = zero ids | 1.2e-7 | rtol 1e-4, atol 1e-5 |

**Documented convention differences (not "fixed"):**
- Captum `riemann_trapezoid` weights sum to (n−1)/n. On Product with n = 10 it gives 6.75 per coordinate, versus the native `trapezoid` 7.5.
- Captum `Saliency` defaults to `abs=True`.
- Captum sets `requires_grad` on its inputs.
- Captum's layer baselines are in input space.
- **Configuration mismatch is visible:** Captum IG at n = 2 differs from native n = 64, agrees with native n = 2, and records `n_steps = 2`.

## 7. Persistence and security

- Attribution and reduction records, retained attribution and baseline tensors, claims, and limitations round-trip through `trace.json` and `tensors.pt`. There are no new files.
- A tampered attribution tensor is rejected on load (content digest).
- Loading remains `weights_only=True`, and no callables are serialised.
- The new record kinds needed no migrations. `causal_effect` v1 → v2 has one (ADR-029).

## 8. Validation

| Check | Result |
|---|---|
| Python 3.10 / 3.12 / 3.14, **without Captum** (`-W error -rsxX`) | 770 passed, 1 skipped (the Captum cross-check module, reported) |
| Python 3.10 / 3.12 / 3.14, **with Captum 0.9.0** | **791 passed**, 0 skipped |
| ruff, ruff format, mypy `--strict` (with and without Captum installed) | clean |
| sdist + wheel | built |
| Clean venv (Python 3.12, torch 2.14), wheel only | Phase 1, 2, and 3 smokes and all 3 README examples pass. Without Captum: native path works, `CaptumUnavailableError` is clear, Captum is never imported. With Captum: the Captum path matches native. |

Torch versions: 2.14 on 3.10/3.12, 2.12 on 3.14.

## 9. Mutation tests (all caught; scratch copies; with Captum installed)

| Mutation | Tests that fail |
|---|---|
| AttributionRecord labelled MEASURED | 5 |
| AttributionRecord labelled INTERVENTIONAL | 16 |
| target ignored | 12 |
| baseline ignored | 2 |
| IG sign reversed | 22 |
| IG missing the (x − baseline) factor | 23 |
| input×gradient returns the raw gradient | 11 |
| integer token ids treated as differentiable | 2 |
| caller tensor autograd state mutated (no detach/clone) | 18 |
| caller-tensor autograd guard removed | 2 |
| parameter gradients accumulated (`.backward()`) | 35 |
| parameter `.grad` not restored | 2 |
| model state change accepted | 2 |
| repeated call attributed to the wrong call | 2 |
| alias ambiguity accepted (runner check) | 2 |
| alias ambiguity accepted (runner and trace checks) | 7 |
| native/Captum configuration mismatch hidden | 11 |
| attribution accepted as causal evidence | 7 |
| attribution protocol justifies NECESSARY_FOR | 2 |
| method identity reduced to its friendly name (claims) | 2 |
| caller-metric identity reduced to its friendly name | 2 |

Notes on the mutation work:
- The first run exposed two survivors, and neither was a real defect.
  - The first caller-tensor mutation did not create a broken state, because the input is already detached. It was replaced by a real one, and an independent guard test was added.
  - The runner's alias check is backed by the trace-level refusal. The test now requires the early, attribution-specific refusal, and a both-checks-removed variant was added.
- Seven mutations were caught by one test only; each got an independent second test.
- Custom attribution methods do not exist (deferred), so "custom attribution identity by friendly name" is covered by the method-identity mutation and the ADR-029 caller-metric mutation.

## 10. Benchmark (this machine; characterisation only)

Environment: Python 3.14.3, torch 2.12.0, Captum 0.9.0, Darwin arm64, 4 threads; warm-up 3, 30 iterations; median / p90 ms. IG: zero baseline (input ids 0 for TinyTransformer), rule riemann_middle.

| model | attributed | forward | trace | gradient | input x grad | IG 16 | IG 64 | Captum IG 16 | Captum IG 64 |
|---|---|---|---|---|---|---|---|---|---|
| Product (analytic) | input | 0.004 / 0.004 | 0.333 / 0.341 | 0.648 / 0.669 | 0.643 / 0.663 | 1.450 / 1.489 | 3.570 / 3.643 | 1.866 / 1.911 | 5.074 / 5.175 |
| TinyMLP | input | 0.020 / 0.020 | 0.456 / 0.471 | 0.957 / 1.000 | 0.954 / 0.988 | 2.297 / 2.354 | 6.190 / 7.189 | 2.823 / 3.139 | 7.646 / 8.343 |
| TinyCNN | input | 0.071 / 0.084 | 0.683 / 0.717 | 1.532 / 1.563 | 1.553 / 1.690 | 5.232 / 6.373 | 14.969 / 15.911 | 5.505 / 5.582 | 16.934 / 17.389 |
| TinyTransformer | token_embedding | 0.185 / 0.199 | 1.188 / 1.223 | 3.128 / 3.188 | 3.140 / 3.473 | 14.511 / 16.530 | 40.464 / 41.236 | 14.014 / 14.239 | 44.319 / 49.170 |

- Every attribution includes two full model fingerprints, the guards, one traced reference pass, and record building.
- IG cost is linear in `n_steps` (one forward and backward per step).
- Captum at batch size 1 costs about the same as native. There was no optimisation.

## 11. Limitations

- Instance-level only: one input, one scalar target, and no FINITE_SAMPLE/POPULATION attribution.
- Module-output (layer) attribution only. There is no module-input or functional-op attribution, and a layer attribution does not account for paths that bypass the layer.
- Built-in targets only; caller targets and custom attribution methods are deferred.
- Native IG rules are Riemann/trapezoid only (Gauss-Legendre is available through Captum).
- The Captum adapter needs batch-1 inputs, single-call layers for LayerIG, and Captum 0.9.x.
- Only CPU and the CPU RNG are verified, and stochastic models are refused, not handled.
- Not built from the roadmap's Phase-3 list:
  - FeatureAblation/Occlusion adapters;
  - `EvidenceSpan`;
  - the `method_agreement/v1` test;
  - the "top-attributed units vs random units under ablation" experiment.
- There are no faithfulness scores and no global explanation score (by design).

## 12. Gate

**GO WITH EXPLICIT LIMITATIONS.**

| Criterion | Status |
|---|---|
| Analytical gradients reproduced | ✅ exact |
| Analytical input×gradient reproduced | ✅ exact |
| Analytical IG reproduced within declared tolerance | ✅ (exact where the rule is exact; within the declared bound elsewhere) |
| Native/Captum cross-check | ✅ (max 1.2e-7) |
| Target identity explicit | ✅ |
| Baseline identity explicit | ✅ |
| AttributionRecord status semantics | ✅ (ATTRIBUTED by kind; no upgrading) |
| No autograd state leakage | ✅ |
| No silent attribution to ambiguous repeated/alias sites | ✅ (per-call targeting plus container check; aliases and Captum repeated calls refused) |
| Persistence | ✅ |
| Causal claims cannot be supported by attribution alone | ✅ (schema, registry, policy, and protocol) |
| Limitations explicit | ✅ |
| All validation green | ✅ |

- **No known silent attribution misassignment.**
- The limitations in §11 are scope boundaries that are refused or flagged.
- **Before Phase 4:** owner review of this report, ADR-029, and ADR-030.
- **Public release** remains blocked by RB-1/RB-2 (contact placeholders) and RB-3 (CI never run). Nothing has been pushed.
