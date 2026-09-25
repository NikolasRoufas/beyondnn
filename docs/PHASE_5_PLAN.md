# Phase 5 Plan: Faithfulness Tests

- **Status:** plan, written and committed before any Phase-5 code. Every expected value below was derived analytically before any experiment ran.
- **Baseline** (verified 2026-09-25):
  - HEAD `39499da`, clean tree.
  - Python 3.10.16 and 3.12.13 (torch 2.14.0), and 3.14.3 (torch 2.12.0).
  - Without Captum: 808 passed plus 1 reported skip. With Captum 0.9.0: 829 passed.
  - ruff, format, and mypy `--strict` (with and without Captum) clean; sdist and wheel build.
  - There is no `beyondnn/causal/` package: the causal machinery is `beyondnn/interventions/`.

## 1. Scientific objective

- **Goal:** make specific interpretability claims testable under explicit, declared perturbation protocols, and keep every raw measurement, control, statistic, and conclusion separately inspectable.
- **Not the goal:** a global "faithfulness score" (ADR-007 stays binding).
- **Premise to falsify.** Does attaching explicit protocols to claims actually separate explanations that track known causal structure from plausible-looking non-causal ones? Two designed experiments test this (§17, scenarios B and I), and one checks that metrics can disagree (scenario K). A framework that only passes on easy monotone examples would fail this phase.

## 2. Scope

**In scope:**
- `comprehensiveness/v1` and `sufficiency/v1` claim-test protocols, input-level and internal-level;
- removal and retention curves;
- matched random controls with Monte-Carlo comparisons;
- `stability/v1` under user-declared transformations;
- `counterexample/v1` over declared sample sets;
- sample-level results with dataset-level summaries;
- method diagnostics (IG step sensitivity, baseline sensitivity, method agreement);
- a ground-truth suite;
- composition into the Phase-4 WHY;
- protocol documentation, a benchmark, and mutation tests.

## 3. Non-goals

- A global or aggregate faithfulness score, and no default public aggregate. RQ8 is deferred (§15).
- ROAR-style retraining.
- Pixel-flipping or region-perturbation reimplementations (Quantus covers input maps). A Quantus adapter would require heavy dependencies (opencv, scikit-image, pandas, scikit-learn), and Quantus does not evaluate internal claims.
- Captum `infidelity`/`sensitivity_max` adapters. They evaluate input maps under random perturbations; they are candidates for a later adapter, not needed to test claims.
- Concepts, probes as evidence, and semantic labels (Phase 6).
- Large models, GPU, network downloads.
- Universally quantified claims as first-class claims: "for every sample in S" is represented as a counterexample protocol result over instance claims (§7).

## 4. Terminology

| Term | Meaning |
|---|---|
| **site** | a positional model input leaf `args[i]` (input level), or one leaf of one call of a module OUTPUT (internal level) |
| **unit** | an index along the **last dimension** of the site tensor. Selection is supported only when all other dimensions have size 1, so units are elements. Otherwise it is refused: per-position selection needs an explicit reduction (Phase 3) and is out of scope. |
| **selection** | an ordered set of units: a ranking and a size `k`. Its source is an attribution record (rank by \|score\| or signed score, ties broken by lower index), a declared set, or a seeded random draw. Recorded as an `EvidenceSelection`. |
| **replacement** | the declared value that replaced units: zero, or an explicit tensor of the site's exact shape (for example per-unit means computed by the caller). It is never a neutral "absence". |
| **remove(S)** | units in S are replaced; everything else is untouched |
| **retain(S)** | units **not** in S are replaced, *within the site*. For internal sites, computation that bypasses the site is untouched. |
| **drop** | `F(x) − F(x')` for the declared scalar target F. This is ERASER's orientation: comprehensiveness = F(x) − F(x∖S), sufficiency = F(x) − F(S only). |
| **control** | a matched random selection: the same site, the same size k, uniform without replacement, from a seeded local `torch.Generator` (never the global RNG) |

- **Why four curve names are two modes.** "Deletion" and "progressive ablation" are both remove(ranking[:k]); "insertion" and "progressive retention" are both retain(ranking[:k]).
- They differ only in level (input vs internal), which is recorded by the site. So there are exactly two curve protocols: `removal_curve/v1` and `retention_curve/v1`.

## 5. Proposed API (`bnn.faithfulness`, alias `F`)

```python
sel = F.top_k(attribution_result, k=2, by="abs")              # or F.units(site, units), F.random(...)
test = F.comprehensiveness(target=m, replacement=F.zero(), controls=F.controls(n=200, seed=0),
                           min_drop=1.0, min_fraction_below=None, relation=Relation.NECESSARY_FOR,
                           statement="the selected units are necessary for the target")
result = F.run(model, x, test, selection=sel)                  # one sample
dataset = F.run_dataset(model, samples, test, selector=F.top_k_by(method, k=2))  # many samples
curve = F.curve(model, x, ranking=F.ranking(attr), target=m, mode="remove", replacement=F.zero(),
                controls=F.controls(n=50, seed=0))
stab = F.stability(model, x, transformation=F.transformation("swap01", fn, implementation_revision="v1"),
                   method=A.input_x_gradient(), target=m, k=1, tolerances=...)
diag = F.ig_step_sensitivity(...), F.baseline_sensitivity(...), F.method_agreement(a, b, k=...)
response = bnn.compose(trace, attributions=[...], faithfulness=[result], policies=[...])
```

- Methods run explicitly. Composition only reads (Phase 4).
- Each result object holds its trace (a single recording for all perturbations of one run) and any attribution results it used.

## 6. Schema changes

1. **`InterventionRecord` v2** (migration: `units = None`, `retain = False`).
   - New fields: `units: tuple[int, ...] | None` (sorted, unique, ≥ 0, non-empty) and `retain: bool` (requires `units`).
   - The site may now also be a root positional input leaf (`module=""`, `io=INPUT`, `args[i]`; ZERO or CONSTANT only).
   - This extends the Phase-2 engine instead of duplicating it. Every perturbation is still a paired baseline/intervention comparison with the same refusals, and its effect is still an INTERVENTIONAL `CausalEffect`.
2. **Several comparisons in one recording.** One baseline pass per sample, then one intervention pass per perturbation. This is internal Phase-2 machinery, reused by faithfulness runs, so that N controls do not create N traces.
3. **New non-evidence record kinds** (status `None`, like `InterventionRecord`):
   - `evidence_selection`: what was selected and how (source, rule, full ranking, scores used, k, seed).
   - `protocol_result`: the output of a non-claim diagnostic protocol (curves, stability, counterexample summaries, method diagnostics, dataset summaries). It holds:
     - declared `params` and `criteria`;
     - derived `measurements` (numbers computed from the referenced records);
     - per-aspect `outcomes` (PASS, FAIL, NOT_APPLICABLE, INDETERMINATE; never a single number);
     - sample ids and target.
4. **No new `EvidenceStatus`.** The raw measurements are INTERVENTIONAL effects (and OBSERVED outputs, ATTRIBUTED rankings). Test results and protocol results are not evidence.
5. **Protocol registry.**
   - `comprehensiveness` v1 justifies NECESSARY_FOR and DECREASES.
   - `sufficiency` v1 justifies SUFFICIENT_FOR, **the first protocol that can decide SUFFICIENT_FOR**, scoped to its declared retention and replacement.
   - Diagnostic protocols go in a separate registry and never decide claims.
6. **Coverage.** `faithfulness_evaluated` becomes true only when a composed bundle contains a faithfulness claim result or diagnostic, and `faithfulness_protocols` names exactly which ones. `NOT_EVALUATED` then lists the protocols that were not run. `concepts_validated` stays `Literal[False]`.

## 7. Protocols

| Protocol | Quantity | SUPPORTS / PASS iff (all declared before running) | Otherwise |
|---|---|---|---|
| `comprehensiveness/v1` (claim test) | `drop = F(x) − F(remove(S))`; controls: drops of N matched random sets | `drop ≥ min_drop` and, if declared, `fraction_below ≥ min_fraction_below` | CONTRADICTS. INCONCLUSIVE if the perturbation is a no-op (replacement equals the original at every selected position). |
| `sufficiency/v1` (claim test) | `drop = F(x) − F(retain(S))`; controls likewise | `drop ≤ max_drop` and, if declared, the fraction of controls with a *larger* drop ≥ `min_fraction_above` | CONTRADICTS; INCONCLUSIVE for a no-op. Refused if S covers all units (vacuous). |
| `removal_curve/v1`, `retention_curve/v1` (diagnostic) | drops at declared k values along a ranking, the same for N random rankings | no pass/fail by default; optional `aopc_mean_drop` (mean over the declared points, including k = 0; normalisation recorded) | the full curve is always stored |
| `stability/v1` (diagnostic) | four separate aspects: prediction \|F(g(x)) − F(x)\|; ranking Spearman ρ (on the recorded ordinal rankings, ties broken by lower index; raw-score Spearman is undefined under ties); top-k Jaccard; claim-outcome equality | each aspect against its own declared tolerance | each aspect separately FAIL, or NOT_APPLICABLE if not requested. Never combined. |
| `counterexample/v1` (diagnostic over a sample set) | the instance claim test on each declared sample | lists every CONTRADICTS as a counterexample, with the number of samples where the claim held | a counterexample weakens *that claim template under its declared scope*. It is never dropped, and it never invalidates the method. |
| `ig_step_sensitivity/v1` | max \|IG_n − IG_2n\| and completeness deltas | `≤ tol` | FAIL |
| `baseline_sensitivity/v1` | Spearman ρ and top-k Jaccard of IG under two declared baselines | against declared minima | FAIL |
| `method_agreement/v1` | Spearman ρ and top-k Jaccard between two attributions (same sample, site, and target, or refused) | against declared minima | FAIL |

**Statistics** (pure Python/torch; no numpy or scipy dependency):
- `fraction_below`, `fraction_tied`, and `fraction_above` of the control drops relative to the observed drop;
- a one-sided Monte-Carlo p = (1 + #{control ≥ observed}) / (N + 1), recorded with N and the seed and described as "the probability that a matched random set does at least as well", never as "significant";
- dataset level: per-sample paired differences (observed drop − mean control drop), their mean and median, and a seeded sign-flip permutation p.

**No threshold turns a statistic into a faithfulness verdict.** Only declared criteria produce outcomes, and assessments come only from declared policies.

## 8. Baselines (replacement values)

- Zero, or an explicit tensor of the site's exact shape and dtype.
- The value is recorded in the `InterventionRecord` (digest), and the OOD limitations of Phase 2 apply (`ZERO_ABLATION_MAY_BE_OOD` / `CONSTANT_REPLACEMENT_MAY_BE_OOD`).
- Mean/resample ablation is expressed by the caller computing the explicit tensor. RQ9 shows why the choice matters.

## 9. Ground-truth tasks (`beyondnn/_testing/faithfulness_models.py`)

Inputs have shape (1, d) and the target is `select([0, 0])`.

| Task | F | Ground truth (fixed before running any method) |
|---|---|---|
| `Weighted8` | `8x0 + 4x1 + 2x2 + x3` (x4..x7 unused) | causal importance order 0 > 1 > 2 > 3; units 4–7 have no effect |
| `ProxyDistractor` | `x0 + x1` (x2 unused); declared data distribution has `x2 = x0 + x1` | x2 is perfectly correlated with the output but causally unused |
| `RedundantMax` | `max(x0, x1)` | redundant causal features; each is alone sufficient at x0 = x1 (two sufficient mechanisms) |
| `EqualSum4` | `x0 + x1 + x2 + x3` | every unit is relevant; no single unit is sufficient |
| `ProbeReadable` | hidden `h = (x0, x1)` (identity map), `y = 2·h0` | `h1` is exactly readable (h1 = x1, MEASURED) but unused: internal unit 1 has no causal effect |
| `SaturatingPlus` | `tanh(4x0) + 0.2x1` | x0 is causally dominant at x0 = 3, but its gradient ≈ 0 (saturated) |
| `Interaction` (Phase 2) | `a·b` | internal `a` matters except when b = 0 |
| `Additive` (Phase 2) | `x0 + x1` | used for the declared swap invariance |

## 10. Invariance requirements (stability)

- The transformation is declared: name, `implementation_revision`, config, and an optional `unit_map` (a permutation of units, to compare rankings across positions).
- The function is never serialised, scraped, or hashed; the declaration is recorded as declared (`DECLARED_TRANSFORMATION_UNVERIFIED`).
- **Refused:**
  - a transformed input of a different shape or dtype;
  - a `unit_map` that is not a permutation;
  - a transformation that returns the identical input (a no-op declared as an invariance test);
  - a transformation that modifies its input in place.

## 11. Failure and refusal conditions (explicit errors, never silent)

- **Selection:**
  - an empty selection;
  - `k` greater than the number of units;
  - selection tensors with non-singleton leading dimensions;
  - a selection or attribution from another sample, model, target, site, or call (a sample mismatch is scenario G);
  - an attribution whose ranking does not re-derive from its tensor.
- **Tests:**
  - sufficiency with S covering all units;
  - controls when k equals the number of units (degenerate null);
  - a replacement of the wrong shape or dtype;
  - an unsupported target (Phase-3 target rules);
  - a non-finite metric value.
- **Perturbations:** the perturbation was not applied (verified against the recorded input identity or the retained activation). The Phase-2 refusals also carry over: training mode, RNG use, state drift, in-place input modification, and an intervention that did not apply exactly once.
- **Composition:**
  - composing a dataset-level result into an instance-level WHY;
  - composing with a reference trace that lacks `sample_id` (scenario J);
  - unregistered protocols, and forged results (re-derived).

## 12. Numerical checks

- Every expected value in §17 must be reproduced to `atol = 1e-5` (float32 forward), except where exact integers are stated.
- Control statistics use fixed seeds. The Monte-Carlo expectations in §17 are stated with tolerances derived from the binomial standard deviation (±3σ), written here before running.

## 13. Mutation tests (written before running)

1. Skip the selection sample-id check.
2. Rank by ascending instead of descending score.
3. Compare against controls of a different size.
4. Controls drawn from the global RNG, or with an ignored seed.
5. Perturbation not applied (the hook skips unit replacement).
6. Retain mode replaces S instead of its complement.
7. Reuse the target of another result (target check skipped).
8. Limitations dropped in dataset summaries.
9. Incompatible metrics averaged (dataset summary ignoring target mismatch).
10. Dataset p-value computed unpaired.
11. `faithfulness_evaluated` true without a faithfulness result.
12. Sufficiency protocol allowed to decide NECESSARY_FOR.
13. A counterexample dropped from the summary.
14. A no-op perturbation reported as CONTRADICTS instead of INCONCLUSIVE.
15. Stability aspects collapsed (one FAIL hidden by a PASS).
16. A forged comprehensiveness result accepted in composition.
17. Curve points mis-ordered or the k = 0 anchor dropped.
18. The `aopc` normalisation differs from the recorded one.

Every guard should be caught by at least two tests where practical.

## 14. Compatibility requirements

- Phase 1–4 APIs, results, and traces unchanged in meaning.
- `InterventionRecord` v1 payloads migrate with units `None` and retain `False` (no invented data).
- All Phase 1–4 tests pass unmodified, except inventory tests (kinds, codes, API).
- Captum stays optional.
- The Phase-4 measured-only render is unchanged. Only when faithfulness results are composed does a FAITHFULNESS section appear.

## 15. Known risks

- **Distribution shift.** Removal and retention inputs are off-distribution (Hase et al. 2021; Hooker et al. 2019). BeyondNN records the replacement and the OOD limitation, but cannot remove the confound.
- **Mask-shape leakage** (Rong et al. 2022) is not addressed. Input-level results on images inherit it.
- **Metric fragility.** Circuit and ablation faithfulness scores are method-sensitive (Miller et al. 2024). RQ9 measures this on a toy model.
- **Small-d resolution.** With few units, matched random sets frequently coincide with the selection, so even a perfect selection cannot reach a small p (scenario A reports this rather than hiding it).
- **Site-relative internal sufficiency.** Paths that bypass a site stay intact, so internal sufficiency is only sufficiency *at that site*.
- **RQ8** (any predictive aggregate) is **deferred with reason.** The suite has about 8 tasks, far too few for a calibration/held-out split that could justify an aggregate; an attempt would be over-fitted by construction. It stays an open question (negative decision logged).

## 16. Exit criteria

- All §17 expectations met, or deviations explained.
- Every mutation caught.
- 3 Pythons × with/without Captum green; ruff, format, and mypy clean; the build works; the clean wheel works.
- Protocol docs written.
- No global score; no faithfulness verdict beyond declared protocol outcomes.
- The Phase-5 report is written with a gate.

## 17. Pre-registered scenario outcomes

All scenarios use zero replacement unless stated otherwise.

| # | Scenario | Setup | Expected |
|---|---|---|---|
| A | True causal selection | `Weighted8`, x = ones; IG top-2 = {0, 1}; controls N = 200, seed 0 | Comprehensiveness drop = 12 exactly; `fraction_below` ≈ 27/28 = 0.964 (±0.04); p ≈ 1/28 = 0.036 (±0.03). Sufficiency (retain {0, 1}): drop = 3 exactly; fraction of controls with a larger drop ≈ 27/28. |
| B | Distractor, two plausible methods | `ProxyDistractor`, x = (3, 1, 4). IG top-1 = {x0} (IG = [3, 1, 0]). Correlation-saliency top-1 (\|corr(xi, y)\| over the declared dataset, where x2 = x0 + x1, so corr(x2, y) = 1) = {x2}. | IG selection: drop 3 → SUPPORTS min_drop 1. Correlation selection: drop 0 → CONTRADICTS. Only comprehensiveness exposes the difference; both rankings look plausible. |
| C | Redundant causal features | `RedundantMax`, x = (3, 3) | Remove {x0}: drop 0 → CONTRADICTS min_drop 1, although x0 is causal. Remove {x0, x1}: drop 3 → SUPPORTS. Necessity ≠ causal relevance. |
| D | Insufficient set | `EqualSum4`, x = ones | Retain {x0}: drop 3 → sufficiency CONTRADICTS (max_drop 0.5). Remove {x0}: drop 1 → comprehensiveness SUPPORTS (min_drop 0.5). |
| E | Declared invariance | `Additive`, x = (3, 5), g = swap the values of x0 and x1 (declared, revision v1); input×gradient, k = 1 | Prediction: \|8 − 8\| = 0 → PASS. Ranking ρ = −1 → FAIL (min 0.5). Top-1 Jaccard 0 → FAIL. Reported as three separate outcomes. |
| F | Counterexample | `Interaction`, internal site `a` units (0,), NECESSARY_FOR, min_drop 1; samples (3,5), (2,2), (1,3), (3,0) | Drops 15, 4, 3, 0 → SUPPORTS ×3 and CONTRADICTS at (3, 0). Recorded as a counterexample; "held on 3 of 4 declared samples". Nothing discarded. |
| G | Wrong-sample evidence | Selection from an attribution on (3, 5), tested on (1, 1) | Refused (`SampleMismatchError`). Composing a faithfulness result about another sample into a WHY is also refused. |
| H | Random method | `Weighted8`: 40 seeded random rankings, top-2 each, vs N = 200 matched controls | Mean over methods of (fraction_below + ½·fraction_tied) in [0.40, 0.60]. IG's selection: ≥ 0.95. |
| I | Misleading attribution | `SaturatingPlus`, x = (3, 1): gradient = [4·sech²(12) ≈ 6.0e-10, 0.2] → top-1 {x1}; IG = [tanh 12 ≈ 1.0, 0.2] → top-1 {x0} | Gradient selection drop 0.2 → CONTRADICTS min_drop 0.5. IG selection drop ≈ 1.0 → SUPPORTS. `method_agreement` ρ = −1 → FAIL. The disagreement is shown, not resolved. |
| J | Migrated pre-ADR-031 trace | `InputRecord.sample_id = None` as reference | Measured-only WHY still works. Composing a faithfulness result → refused (no exact sample identity). |
| K | Metrics disagree | `RedundantMax`, x = (3, 3), declared S = {x0} | Comprehensiveness CONTRADICTS (drop 0), sufficiency SUPPORTS (drop 0 ≤ 0.5). Both shown; no aggregation. |
| L | Probe-readable but unused | `ProbeReadable`, x = (3, 5), internal site | h1 = 5 = x1 (MEASURED, readable). Remove unit 1: drop 0; remove unit 0: drop 6. Internal sufficiency: retain {0} drop 0 (SUPPORTS, with a site-relative limitation); retain {1} drop 6. |
| M | Curves | `Weighted8`, x = ones, IG ranking (0..7, ties by index), k = 0..8 | Removal drops 0, 8, 12, 14, 15, 15, 15, 15, 15; `aopc_mean_drop` = 109/9 = 12.1111. Retention drops 15, 7, 3, 1, 0, 0, 0, 0, 0; `aopc_mean_drop` = 26/9 = 2.8889. |
| N | Method diagnostics | Product `x0·x1`, x = (3, 5): IG zero baseline [7.5, 7.5] vs baseline (2, 1) [3, 10] | Rankings [0, 1] vs [1, 0]: ρ = −1 → baseline_sensitivity FAIL (min 0.5). IG step sensitivity on Saturating (n = 16 vs 32): max diff ≤ 2.25/16² + 1e-5 → PASS. |
| RQ9 | Replacement choice | `Weighted8` with x = (1, 1, 1, 1, 5, 5, 5, 5) and caller "mean" replacement m = (1, 1, 1, 1, 5, 5, 5, 5) vs zero | Zero: single-unit drops (8, 4, 2, 1, 0, 0, 0, 0); mean replacement: all drops 0 (the replacement equals x, a no-op, so INCONCLUSIVE). Shows that the replacement choice alone can erase the signal. Logged, not "fixed". |
