# Phase 5.5 API Review: BeyondNN on realistic trained models

- **Scope:** every public-API interaction in `experiments/phase5_5/` on:
  - **A:** a trained MLP on breast_cancer, input and hidden sites;
  - **B:** a trained CNN on digits, pixels and conv channels;
  - **C:** BERT-tiny fine-tuned on SST-2, token positions of the word embeddings.
- **Classes** (plan §15): BUG / SCIENTIFIC DESIGN ISSUE / API ERGONOMICS / MISSING CAPABILITY / DOCUMENTATION ISSUE / EXPECTED LIMITATION.
- **Rule of the stress test:**
  - The experiments use only public BeyondNN APIs.
  - The one exception is `beyondnn.faithfulness.stats` (an importable, tested submodule not listed in `beyondnn.faithfulness.__all__`), used for `rank_order`, `jaccard`, `quantile` and `sign_flip_p`; see F-15.
  - Helpers named `_x` in the experiment scripts are experiment-local, not BeyondNN internals.
  - No BeyondNN private attribute was read or written.

## Summary

| # | Finding | Class | Status |
|---|---|---|---|
| F-1 | Pixel, channel and token units were not expressible (last-axis rule) | MISSING CAPABILITY | **fixed** (ADR-034) |
| F-2 | Count-matched controls confound "selected" with "perturbed more" | SCIENTIFIC DESIGN ISSUE | **fixed** (ADR-035, opt-in) |
| F-3 | Deterministic CNN refused as stochastic (grad vs no-grad kernels, small margin) | BUG | **fixed** (ADR-036) |
| F-4 | Curves over declared units dropped `unit_axes` from their selection record | BUG (in the ADR-034 change) | **fixed** (`2e4e662`) |
| F-5 | Declared reduction dropped from the record for single-element units | BUG (in the ADR-034 change) | **fixed** (`a5768ec`) |
| F-23 | Magnitude-matched results at a conv site refused: exact-equality re-derivation vs a 1-ulp different pass | BUG (in ADR-035 verification) | **fixed** (ADR-038, `39bdd75`) |
| F-6 | Wrong-shaped declared input selections were refused only after every pass ran | API ERGONOMICS | **fixed for input sites**; internal sites still after the passes |
| F-7 | `run_dataset` has one target and one criterion for all samples | MISSING CAPABILITY | open, bounded (per-sample `run` + experiment-side aggregation) |
| F-8 | `run_dataset` has no `model_kwargs` | MISSING CAPABILITY | open, bounded |
| F-9 | No selection source for intervention-derived (ablation) rankings | MISSING CAPABILITY | open, bounded (DECLARED selection; provenance link lost) |
| F-10 | `curve` refuses declared rankings (ablation/random curves) | MISSING CAPABILITY | open, bounded (workaround via `compare_family`) |
| F-11 | Method diagnostics and `stability` took no `model_kwargs` | MISSING CAPABILITY | **fixed** (ADR-037, `3c7650b`) |
| F-12 | Method diagnostics take one `reduce` for both attributions | API ERGONOMICS | open |
| F-13 | Two sign conventions: `CausalEffect.effect` = perturbed − clean, faithfulness `drop` = clean − perturbed | DOCUMENTATION ISSUE | open (documented here) |
| F-14 | Gradient-ranked hidden units can be dead ReLUs; zero removal is then a no-op | EXPECTED LIMITATION | reported (INCONCLUSIVE, never CONTRADICTS) |
| F-15 | `faithfulness.stats` helpers needed for analysis but not exported | API ERGONOMICS | open |
| F-16 | Captum `LayerIntegratedGradients` cannot express layer-space baselines | EXPECTED LIMITATION | documented in the adapter; native IG used for C |
| F-17 | `faithfulness.units` docstring described only last-axis units | DOCUMENTATION ISSUE | **fixed** (`071f05b`) |
| F-18 | Result statistics are typed `JsonValue`, so typed code needs casts | API ERGONOMICS | open, minor |
| F-19 | Margin targets need the caller to fix predicted/runner-up classes from a clean pass | EXPECTED LIMITATION | by design (ADR-030: explicit targets) |
| F-20 | Units are axis-aligned grids only (no superpixels, spans, or word pieces merged into words) | EXPECTED LIMITATION | documented in ADR-034 |
| F-21 | A SUPPORTS outcome does not mean "better than random" unless `min_fraction_below` is declared | SCIENTIFIC DESIGN ISSUE | open, bounded (control statistics are recorded; policies must opt in) |
| F-22 | `comprehensiveness`/`sufficiency` default to the zero replacement when none is given | API ERGONOMICS / SCIENTIFIC DESIGN ISSUE | open (recorded, with OOD limitation); decision for the owner |

## Findings

### F-1: pixel, channel and token units were not expressible (MISSING CAPABILITY, fixed)

- **Observed:** the pre-change probe (`results/abstraction_probe.json`, produced before any framework change) found three failures:
  - `F.top_k` refused B's `(1, 1, 8, 8)` pixel attribution and `(1, 16, 8, 8)` channel attribution;
  - it refused C's `(1, T, 128)` token attribution;
  - a declared pixel selection was refused.
- Interventions could replace only positions along the last axis, so "token t, all hidden dimensions" required reshaping the model.
- **Fix:** declared `unit_axes` (ADR-034). `abstraction_probe_after_adr034.json` shows every B and C step succeeding with declared axes, and that implicit aggregation is still refused.

### F-2: controls confound selection with perturbation size (SCIENTIFIC DESIGN ISSUE, fixed)

- **Observed:** count-matched controls answer "better than a random set of the same size". Input × gradient and IG favour units with large |x − b|, so a selection can beat count-matched controls because it perturbs more.
- **Fix:** `controls(match="magnitude")` (ADR-035). The magnitudes are recorded and re-derived. The quantitative effect on conclusions is in the report (H1b).

### F-3: false stochasticity refusal (BUG, fixed)

- **Observed:** B sample 2 raised `StochasticAttributionError`. The deterministic CNN's grad-enabled forward differs from the no-grad forward by one float32 rounding unit of the logits, and the margin target (a logit difference, 0.84) amplifies that to a 1.6e-6 relative error.
- **Fix:** tolerance scaled to the output precision (ADR-036), with a regression test. Real drift is still refused.

### F-4 and F-5: two defects in the ADR-034 change itself (BUG, fixed)

- **F-4:** `curve` rebuilt its ranking `Selection` positionally and silently dropped `unit_axes`/`unit_reduction`. The recorded curve therefore misstated its units, and composition refused it.
- **F-5:** `ranking(..., reduce="l2")` over one-channel pixels (one element per unit) scored with `l2` (|g|) but recorded `unit_reduction=None`, so verification re-derived signed scores.
- In both cases **composition caught the misstatement**: the re-derivation design worked as intended. Regression tests fail before and pass after each fix. Mutation checks cover both unit fields.

### F-23: magnitude re-derivation by exact equality (BUG, fixed)

- **Observed:** composition refused all 15 magnitude-matched results of B's first sample at `relu2`. The magnitudes are computed from a separate traced pass before the family runs. That pass's conv activation differed from the family's clean pass by one float32 ulp, and verification compared the magnitudes exactly.
- **Fix:** ADR-038 compares within a rounding-scaled tolerance, while the controls are still re-drawn exactly from the recorded magnitudes. The regression test fails before the fix.
- The re-run of B reproduced every result row, and all 105 first-sample `relu2` results composed.

### F-6: late refusal of mis-shaped selections (API ERGONOMICS, partly fixed)

- **Before:** a declared 64-unit pixel selection ran all perturbation passes and was refused afterwards.
- **Now:** input sites are checked before any pass. Internal sites are checked against the recorded activation; without magnitude controls that still happens after the passes, because the activation shape is unknown until a forward pass runs.
- **Bounded:** the refusal is correct and states the fix (declare `unit_axes`).

### F-7: `run_dataset` has one target and one criterion (MISSING CAPABILITY, open)

- Realistic evaluation targets the *predicted* class margin, which differs per sample, with a criterion relative to that sample's margin (plan §8: t × margin). `run_dataset` takes one `TestTemplate`: one target metric and one absolute `min_drop`.
- **Workaround used:** one `F.run` per sample. The `counterexample`/`paired_control` summaries are then computed in experiment code (`analysis.py`), not recorded as `ProtocolResult`s.
- **Consequence:** the dataset-level summaries of Phase 5.5 are experiment outputs, not verifiable BeyondNN records.
- **Candidate for later:** per-sample target and criterion rules (a target *rule* such as "margin of the clean prediction", and relative criteria).

### F-8: `run_dataset` has no `model_kwargs` (MISSING CAPABILITY, open)

- `run_dataset` runs every sample without keyword inputs. For BERT and single unpadded sentences the default `attention_mask`/`token_type_ids` coincide with the tokenizer's, so a run would compute the same outputs. It cannot express models that need keyword inputs, or padded inputs.
- Its per-sample identity would also differ from that of attributions computed with `model_kwargs` (the kwargs are part of `sample_id`), so such evidence would be refused as "about another input".
- **Bounded:** per-sample `F.run` takes `model_kwargs`. Not exercised further because F-7 already rules out `run_dataset` for these experiments.

### F-9: ablation rankings are recorded as DECLARED selections (MISSING CAPABILITY, open)

- The ablation ranking comes from INTERVENTIONAL single-unit effects, but the only way to test it is `F.units(...)` (source DECLARED). The resulting `EvidenceSelection` does not cite the effects that produced the ranking.
- **Bounded:** the experiment stores the ablation drops next to each result. A selection source "intervention" (citing a comparison family) is a candidate for later.

### F-10: `curve` refuses declared rankings (MISSING CAPABILITY, open)

- `curve(ranking=F.units(...))` raises `TypeError`, so the ablation ranking has no library curve.
- **Workaround:** a single `compare_family` with the same interventions (`diagnostics.py`). The random reference comes from the curve's own random-ranking controls (plan §18).

### F-11: diagnostics without `model_kwargs` (MISSING CAPABILITY, fixed)

- **Observed:** `method_agreement`, `baseline_sensitivity`, `ig_step_sensitivity` and `stability` took a single tensor `x` and no `model_kwargs`. For model C the attributions' `sample_id` includes the kwargs, so all 160 diagnostics (40 sentences × 4) were refused with `SelectionMismatchError: the attributions are not about this input`.
- **Fix:** ADR-037 adds `model_kwargs` to all four, with regression tests. The C diagnostics were re-run after the fix.

### F-12: one reduction for two attributions (API ERGONOMICS, open)

- The plan scores gradient units by `l2` and IG/input × gradient units by `sum`. A method diagnostic takes one `reduce`, so a "gradient-l2 vs IG-sum" agreement cannot be recorded. The library diagnostics used `sum` for all methods (plan §18).

### F-13: two sign conventions (DOCUMENTATION ISSUE, open)

- `CausalEffect.effect` = perturbed − clean (Phase 2). The faithfulness `drop` = clean − perturbed (Phase 5). Both are documented where they are defined, but experiment code that mixes `compare_family` with `F.run` must negate one of them (`diagnostics.py`, `run_faithfulness.py`).

### F-14: dead ReLU units in gradient rankings (EXPECTED LIMITATION)

- For A's hidden layer, the gradient with respect to a ReLU *output* can be large for a unit whose activation is 0. Removing it (zero replacement) is an exact no-op.
- BeyondNN records such results as INCONCLUSIVE with `no_op=True`, never as CONTRADICTS, which is the intended behaviour. Counts are in the report.

### F-15: `faithfulness.stats` is not exported (API ERGONOMICS, open)

- Realistic analyses need the same ranking, Jaccard, quantile and sign-flip helpers the protocols use. They are importable and tested, but not in `__all__` and not documented as public.

### F-16: Captum LayerIG baselines (EXPECTED LIMITATION)

- `LayerIntegratedGradients` takes baselines in model-input space (token ids). The pre-registered layer-space zero baseline for C therefore used native IG (plan §18). This is already stated in the adapter docstring.

### F-17: `units` docstring (DOCUMENTATION ISSUE, fixed)

- The docstring said `n_units` "is the size of the last dimension". It now describes the declared-axes case.

### F-18: statistics typing (API ERGONOMICS, minor)

- `FaithfulnessResult.statistics` is a `JsonMap`, so typed callers need casts for `fraction_below`, `control_drops`, and similar fields. A typed accessor would help. Not changed.

### F-19: margin targets (EXPECTED LIMITATION, by design)

- `metrics.difference([0, pred], [0, runner])` has to be built from a clean pass by the caller. This is intentional (ADR-030: explicit targets), and the experiment fixes the target before any perturbation.

### F-20: grid units only (EXPECTED LIMITATION)

- Declared units are axis-aligned grids. Superpixels, contiguous spans, or merging word pieces into words need either a mask-set abstraction or caller-side model wrappers. Not needed for A/B/C; recorded in ADR-034.

### F-21: SUPPORTS without beating random controls (SCIENTIFIC DESIGN ISSUE, open)

- By design (ADR-033; request §18), a claim test's outcome comes only from its declared criterion. Control statistics are recorded next to it and do not change it unless `min_fraction_below`/`min_fraction_above` is declared.
- **Observed on realistic models:** at the primary threshold, the seeded **random** "method" received comprehensiveness SUPPORTS **218 times**: A 41 + 16, B 85 + 44, C 32 over the five sites. Across all methods, 51 SUPPORTS outcomes had a selection no better than the median matched control (superiority ≤ 0.5; 0–2.7% of SUPPORTS per site).
- Each such record states its `fraction_below` and Monte-Carlo p, so nothing is hidden. But a reader who looks only at `outcome` would conclude "the selected units are necessary" for a random set.
- **Recommendation:** Phase 6 policies (concept validation) should declare the control criterion explicitly. The WHY rendering could show the control fractions next to every faithfulness outcome.
- **Not changed in this phase:** changing the semantics of `comprehensiveness/v1` would alter Phase-5 records.

### F-22: implicit zero replacement (API ERGONOMICS / SCIENTIFIC DESIGN ISSUE, open)

- `comprehensiveness(...)` and `sufficiency(...)` use `zero()` when `replacement` is omitted. The replacement is recorded in the spec, and `ZERO_ABLATION_MAY_BE_OOD` is attached, but H4 shows that 28–49% of A/B comprehensiveness outcomes change between replacements.
- A silent default therefore selects one of several materially different experiments.
- **Options** (owner decision; not changed here because it breaks the Phase-5 API and README): make `replacement` required, or keep the default and warn.

## What worked without friction

- **Transformer outputs:** Hugging Face `ModelOutput` targets via leaf paths (`'["logits"]'`), `model_kwargs` in `attribute`/`run`/`curve`, and hooks on `bert.embeddings.word_embeddings`.
- **Captum:** the adapter at A/B inputs, including its fixed `internal_batch_size=1`.
- **Replacements:** explicit tensor replacements of the exact site shape (train means, resampled rows, [MASK]/[PAD] embeddings).
- **Verification:** composition re-derived every test result, selection and curve on the first sample of each setting. The only failures were F-4 and F-5, which were real record misstatements.
- **Out-of-distribution limitations:** the OOD limitation codes (`ZERO_ABLATION_MAY_BE_OOD`, `CONSTANT_REPLACEMENT_MAY_BE_OOD`) and `SITE_RELATIVE_SUFFICIENCY` appeared where expected.
