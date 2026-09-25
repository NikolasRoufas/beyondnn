# Phase 3 Plan: Attribution

- **Status:** plan, written before any attribution code. Expected values below were derived analytically before any experiment ran.
- **Baseline:** Phase 2 closed at `50aed1a` (gate: GO WITH EXPLICIT LIMITATIONS), followed by the Phase-2 hardening commit `3f3ff27` (ADR-029, declared caller-metric identity). 704 tests green.
- **Central question:** given a specific scalar model outcome, how does a chosen attribution method assign credit or sensitivity to inputs or internal components?
- **What attribution does not answer:** what caused the output. That requires intervention evidence (Phase 2).

## Principles (fixed before code)

1. **ATTRIBUTED is its own status.** An `AttributionRecord` is ATTRIBUTED by kind, never supplied by the caller. It is never MEASURED and never INTERVENTIONAL. Nothing merges an attribution with a `CausalEffect` into a stronger status (the Phase-1 derivation table already forbids ATTRIBUTED ↔ INTERVENTIONAL lineage).
2. **An attribution is always relative** to a method, its configuration, a target, a baseline (where the method has one), one input, and one attributed tensor (an input leaf, or one leaf of one call of a module output).
   - All of these are part of the record's identity.
   - Presentation choices are not part of its identity.
3. **The target is a scalar or nothing.**
   - Targets reuse the Phase-2 `Metric` / `MetricSpec` language: built-in `select` (one logit), `difference` (logit difference), and `mean` (explicit mean reduction).
   - A target that does not reduce to exactly one element is refused. Outputs are never silently summed.
   - Caller (custom) targets are deferred for attribution.
4. **Raw attribution tensors are not scores.** A record keeps the full attribution tensor, which has the attributed tensor's shape. Any reduction (for example, to a per-token score) is a separate, explicit `AttributionReduction` record that names its operation and dimensions.
5. **No autograd side effects.** Attribution never:
   - modifies caller tensors (their values, `requires_grad`, or `.grad`);
   - modifies parameter `.grad`, parameters, buffers, or train/eval flags;
   - consumes RNG;
   - leaves hooks behind.

   Every one of these is checked before and after the run. Any change is refused, and parameter gradients are restored before the error is raised.
6. **No silent misassignment.** Alias module paths are refused. Repeated module calls are addressed by `call_index`, and the container verifies that the referenced activation has the same site, call, and pass. A library that cannot target one call (Captum layer methods) is refused when the layer runs more than once.
7. **Discrete inputs are not differentiable.** Integer token ids are refused for input attribution. The legitimate path is layer attribution at the embedding output, which gives attribution to embedding dimensions of each position. It is not attribution to token ids.
8. **Captum is reused, not rebuilt.**
   - BeyondNN's native gradient, input×gradient, and IG are small reference implementations used for validation.
   - Captum (optional extra, 0.9.x verified) provides the mature implementations through an adapter.
   - The adapter records Captum's version and every setting.
   - Convention differences are documented and tested.
9. **Custom attribution methods are deferred.** No caller-supplied attribution code is accepted in Phase 3.

## Scope

| In scope | Out of scope |
|---|---|
| input attribution (float inputs); module-output (layer) attribution; gradient, input×gradient, Integrated Gradients (native); Captum IntegratedGradients, Saliency, InputXGradient, LayerIntegratedGradients adapters; explicit scalar targets; attribution records, reductions, provenance, diagnostics, limitations; `attribution_threshold` claim protocol (ATTRIBUTED_TO only); ground-truth models; benchmark | concepts, SAEs, probes, feature naming, natural-language explanations, faithfulness/comprehensiveness scores, sufficiency, Phase-4 evidence synthesis, custom attribution methods, attribution to module inputs, multi-sample (FINITE_SAMPLE/POPULATION) attribution, large models |

## Design

| Piece | Where | Notes |
|---|---|---|
| `AttributionMethodSpec` | `schema/attribution.py` | `name` (`gradient` / `input_x_gradient` / `integrated_gradients`), `implementation` (`beyondnn` / `captum`), `implementation_version`, `params` (IG: `n_steps`, `rule`; Captum: its method name and fixed settings) |
| `AttributionBaseline` | same | ZERO (in the attributed tensor's space), TENSOR (explicit, in the attributed tensor's space), INPUT_TENSOR (layer attribution only: a model-input baseline whose layer activation is the layer-space baseline, matching Captum's LayerIG convention). Tensors are retained and carry a digest. |
| `AttributionRecord` (kind `attribution`, status ATTRIBUTED) | same | `method`, `target: MetricSpec`, `site` (root `INPUT` leaf `args[i]`, or module `OUTPUT` leaf), `call_index`, `pass_index`, `sample_id`, `baseline`, `value` (retained raw tensor), `target_value`, `diagnostics`. Lineage: the reference pass's `OutputRecord` plus the attributed `InputRecord` or `ActivationRecord`. |
| `AttributionReduction` (kind `attribution_reduction`, ATTRIBUTED) | same | `reduction` ∈ {sum, abs_sum, l2}, explicit `dims`, retained reduced `value`, `scalar` iff fully reduced. Derives from exactly one attribution. |
| Container checks | `core/trace.py` | Attribution and reduction tensors are retained tensors. The attributed site, call, and pass match the referenced record. Reduction shapes are consistent. |
| Runtime | `beyondnn/attribution/` | `attribute()` (also `bnn.attribute`, `handle.attribute`); `gradient()`, `input_x_gradient()`, `integrated_gradients(baseline=..., n_steps, rule)`; `zero_baseline()`, `baseline(t)`, `input_baseline(t)`; `input(i)`, `layer(path, call_index, output_path)`; `reduce()`; `captum.*` adapters; claims |
| Execution | runtime | Gradient passes run first, outside tracing, under `enable_grad`, on detached clones (`torch.autograd.grad` only). A single CLEAN traced reference pass under `no_grad` then provides provenance, the observed input/output, and the MEASURED activation of the attributed site. The target value is checked to be the same in the gradient passes and the traced pass. |

## Numerical semantics of IG

For a path `z(α) = x' + α(x − x')`:

`IG(x) = (x − x') ⊙ Σ_k w_k ∇F(z(α_k))`

| Rule | Nodes and weights |
|---|---|
| `riemann_left` | `α_k = k/n` |
| `riemann_right` | `α_k = (k+1)/n` |
| `riemann_middle` | `α_k = (k+½)/n`, all with `w = 1/n` |
| `trapezoid` | `α_k = k/(n−1)`, `w = 1/(n−1)` with the two ends halved (`n ≥ 2`) |

- Gradients are accumulated in float64. The stored attribution has the attributed tensor's dtype.
- **Completeness diagnostic:** `completeness_delta = Σ attribution − (F(x) − F(x'))`. For layer IG, `F(x')` is the target with the layer output replaced by the layer baseline.
- The delta is recorded as method-specific diagnostic metadata. It is not a confidence score.

## Ground-truth models and expected results (written before running)

`x = (x0, x1)` has shape (1, 2), and the target is `select([0, 0])`.

| Model | F | Point | gradient | input×gradient | IG (baseline) |
|---|---|---|---|---|---|
| Linear | `2·x0 + 3·x1` | (3, 5) | [2, 3] | [6, 15] | zero: [6, 15] exactly for every rule (constant integrand); Σ = 21 = F(x) − F(0) |
| Irrelevant | `4·x0` | (3, 5) | [4, 0] | [12, 0] | zero: [12, 0]; attribution to x1 is exactly 0 |
| Product | `x0·x1` | (3, 5) | [5, 3] | [15, 15] | zero: [7.5, 7.5]. Baseline (1, 2): [(x0−b0)(x1+b1)/2, (x1−b1)(x0+b0)/2] = [7, 6]; Σ = 13 = 15 − 2. Exact for `riemann_middle` and `trapezoid` (linear integrand). |
| Product | same | (3, 0) | [0, 3] | [0, 0] | zero: [0, 0] (context dependence) |
| Saturating | `tanh(x0) + x1` | (3, 1) | [sech²3 ≈ 0.0098660, 1] | [≈ 0.029598, 1] | zero: [tanh 3 ≈ 0.9950548, 1]. Raw gradient is near 0 where IG is near 1: saturation. |

- **Declared tolerance for the saturating IG** (from the midpoint-rule error bound `max|g''|/(24n²)` with `g(α) = x0·sech²(α·x0)`, `|g''| ≤ 2·x0³ = 54`): `|IG₀ − tanh 3| ≤ 2.25/n² + 1e-5`, where 1e-5 covers float32 rounding. This is checked at n = 16, 64, 256. No monotonic-convergence claim is made beyond the bound.
- **Declared tolerances elsewhere:**
  - linear, irrelevant, and product expectations: `atol = 1e-5` (float32 forward/backward);
  - native vs Captum under equivalent settings (same rule ∈ {riemann_left, riemann_right, riemann_middle}, same n, same baseline): `rtol = 1e-4, atol = 1e-5`, because native accumulates in float64 while Captum accumulates in float32;
  - native gradient / input×gradient vs Captum Saliency(`abs=False`) / InputXGradient: `atol = 1e-6`.
- **Known convention differences** (from inspecting Captum 0.9.0 before implementation):
  - Captum `riemann_trapezoid` uses step weights `1/n` with halved ends, so they sum to `(n−1)/n`, not 1. On the product model its IG is `x0·x1·(1/2 − 1/(2n))` per coordinate. BeyondNN's `trapezoid` uses `1/(n−1)` and is exact there. These settings are therefore **not** equivalent and are not cross-checked as such.
  - Captum `Saliency` defaults to `abs=True`. The adapter fixes `abs=False` so that it is a raw gradient.
  - Captum sets `requires_grad` on the tensors it is given (with a warning), so the adapter gives it a detached clone.
  - Captum treats dim 0 as the batch. The adapter uses `internal_batch_size=1` and requires a leading dimension of 1, so that each forward sees one interpolation point exactly like the native implementation.
- **Repeated call** (`Twice`: `y = lin(lin(x))`, scalar weight 2, x = 3): the gradient at call 0's output is 2, and at call 1's output is 1. Captum layer attribution is refused, because the layer runs twice.
- **Redundant paths** (Phase-2 `Redundant`: `y = p(x) + q(x)`, `p = q = x0`, x = (3, 5)):
  - layer input×gradient at `p` = 3, and layer IG (zero layer baseline) at `p` = 3: substantial attribution;
  - yet zero-ablating `p` leaves the output at 3 (Phase-2 effect −3, and "p necessary with min_effect 6" is CONTRADICTED).
  - High attribution does not mean necessity. An attribution record cannot even be cited for NECESSARY_FOR (schema refusal).
- **TinyTransformer:** input attribution to token ids is refused (`DiscreteInputError`). Layer attribution at `token_embedding` gives a (1, T, d) tensor. A per-token score only exists through an explicit `AttributionReduction(sum, dims=(2,))`.

## Claims

- **Protocol:** `attribution_threshold` v1 justifies **ATTRIBUTED_TO only**.
- **Declared before running:** the method (spec plus baseline identity), the call index, and `min_abs_attribution`.
- **Score:** `Σ |attribution|` over the claim subject (all elements, or the listed last-dimension units).
- **Outcome:**
  - SUPPORTS iff the score ≥ `min_abs_attribution`, else CONTRADICTS;
  - NOT_APPLICABLE on any mismatch of method, baseline, site, call, target, or input.
- **Meaning:** it is a statement about the attribution method's output, not about causation. The existing `ClaimTestResult` rule already refuses non-causal evidence for causal relations.

## Milestones

1. Plan and ADR-030.
2. Schema: records, container checks, limitations, migrations (none needed: new kinds only).
3. Native runtime: targets, gradient, input×gradient, IG, layer/call targeting, state guards, ground truth.
4. Captum adapter and cross-checks (with and without Captum installed).
5. Claims, explain hook, benchmark, report.

Each milestone ends green on Python 3.10, 3.12, and 3.14, with mutation checks.
