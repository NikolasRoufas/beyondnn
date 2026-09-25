# Phase 5.5 Report: Realistic Faithfulness Validation

- **Gate:** **READY FOR PHASE 6 WITH EXPLICIT LIMITATIONS** (§40).
- **Principle:** the phase set out to falsify BeyondNN's premise on realistic trained models, not to confirm it.
- **Pre-registration:** `docs/PHASE_5_5_PLAN.md`. Its §18 logs every correction and analysis-rule clarification made before any aggregate was computed.
- **Raw results:**
  - `experiments/phase5_5/results/`: gzipped JSON of every claim-test run, curve and diagnostic, with sample ids and result ids;
  - `analysis.json` / `analysis.md` for the derived tables.
- **Terms used below:**
  - **Superiority** = fraction_below + ½ · fraction_tied against the matched random controls. For sufficiency it is fraction_above + ½ · fraction_tied.
  - **Drop** = F(x) − F(perturbed), where F is the margin: logit[pred] − logit[runner-up], fixed from the clean pass.
  - **t** is the outcome threshold as a fraction of that margin.

## 1. Starting commit

`28d4026`, "Add Phase 5 faithfulness benchmark and gate report". It was verified not to be `39499da`, with a clean tree.

## 2. Ending commit

The commit that adds this report. The hash is given in the final summary; `git log` is authoritative. Every commit of the phase is listed in §43.

## 3. Baseline validation

Run at `28d4026` before any change:
- **Tests:**
  - Python 3.10.16 and 3.12.13 (torch 2.14.0), and 3.14.3 (torch 2.12.0);
  - without Captum: 877 passed plus 1 reported skip; with Captum 0.9.0: 898 passed.
- **Checks:**
  - ruff, format and mypy `--strict` (with and without Captum) clean;
  - sdist and wheel build;
  - the Phase 1–5 clean-wheel smokes and all 5 README examples pass, with and without Captum.

## 4. Literature reviewed

`docs/research/PHASE_5_5_LITERATURE.md`: 27 sources. [V] marks sources verified in this or the previous session; [P] marks prior knowledge, flagged as such.

| Group | Sources |
|---|---|
| Removal metrics | ERASER (DeYoung 2020), Samek 2017, RISE (Petsiuk 2018) |
| Remove-and-retrain and OOD | ROAR (Hooker 2019), ROAD (Rong 2022), Blücher 2024, Hase 2021 |
| Infidelity and aggregation | Yeh 2019, Bhatt 2020 |
| Sanity checks | Adebayo 2018, Kindermans 2019, Tomsett 2020 |
| Meta-evaluation and disagreement | MetaQuantus (Hedström 2023), Krishna 2022 |
| Ground truth | Zhou 2022; InterpBench, MIB, Tracr [P] |
| Attention and IG | Jain & Wallace [P], Sundararajan IG [P], Sturmfels baselines [P] |
| Activation patching | Zhang & Nanda 2024, Heimersheim & Nanda 2024, Li & Janson 2024, Miller 2024 |
| Circuits | Wang IOI [P], causal scrubbing [P] |
| Implementations | Captum FeatureAblation / infidelity |

**What the literature implied for Phase 5.5:**
- The removal value is part of the experiment.
- Removal creates off-distribution inputs.
- Evaluation metrics disagree with each other.
- Random baselines are essential.
- Mask shape leaks class information (ROAD).
- Ground truth exists only in constructed settings.

## 5. Selected models and datasets, and justification

| ID | Model | Dataset | Why |
|---|---|---|---|
| A | MLP 30→64→32→2 (ReLU), trained here | scikit-learn `load_breast_cancer`: 569 × 30 standardised features, bundled with scikit-learn | Real tabular classifier with vector units; strongly correlated features (redundancy) |
| B | CNN: Conv(1,8,3)–ReLU–Conv(8,16,3)–ReLU–MaxPool–Linear(256,10), trained here | scikit-learn `load_digits`: 1797 × 8 × 8 | Real image classifier; pixels and conv channels; CPU-sized |
| C | `M-FAC/bert-tiny-finetuned-sst2` at revision `41ad6709ec46b414749b37daf49cf5ca1c7dba7c`, 4.39M params, pretrained externally | GLUE SST-2 validation (872 sentences), `nyu-mll/glue` at revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | Real fine-tuned transformer; token positions; CPU-sized; pinned revisions |

- No model was hand-constructed. No large model was built.

## 6. Training and checkpoint details

| | A | B | C |
|---|---|---|---|
| Training | Adam lr 1e-3, cross-entropy, full batch, 300 epochs, torch seed 0, final epoch | Adam lr 1e-3, cross-entropy, batch 64, 30 epochs, seed 0, local shuffling generator seed 0 | external (fine-tuned by M-FAC) |
| Split | stratified 60/20/20 (`train_test_split`, random_state 0, twice): 341/114/114 | same rule: 1078/359/360 | SST-2 validation |
| Accuracy (train/val/test) | 0.997 / 0.956 / 0.974 | 0.985 / 0.972 / 0.964 | 368 of the sentences with 8–24 tokens are classified correctly (full accuracy not needed) |
| Checkpoint digest (BeyondNN state digest) | `sha256:4c19a8dafa3c0b38219220528303b3f195487a131b3e12a64c703203946b1c24` | `sha256:d799d1e89efd31f2df0f9d3b815257e80e1e375e91bbc9d6206f9d46dc36ed53` | `sha256:e2b44a0891e8b46f810a7b65d5535a29668789b88924f888cd8657a7f7b826e0` |

- A and B are deterministic, CPU-retrainable, and cached under `experiments/phase5_5/artifacts/` (git-ignored). The SST-2 parquet's sha256 (`a1371f3b…`) is recorded in every C result.

## 7. Held-out sample-selection procedure

Plan §5, applied without deviation:
- **A:** the correctly classified test rows (111 of 114), a `torch.Generator().manual_seed(1234)` permutation, the first 60.
- **B:** the correctly classified test images (347 of 360), the same rule, 60 images.
- **C:** validation sentences with 8–24 tokens (including [CLS]/[SEP]) that are classified correctly: 368 eligible. Seeded permutation (1234), first 40. Sentence lengths are 10–24 tokens.

All sample ids are stored in each result file. No sample was dropped or replaced. Case studies below are labelled as such and were found by fixed rules (§22), not chosen by hand.

## 8. Attribution methods

| Method | Implementation | Unit score |
|---|---|---|
| gradient | BeyondNN native | `l2` over within-unit elements (identity for vector units) |
| input × gradient | native | `sum` |
| IG, n = 32, `riemann_middle`, zero baseline | Captum 0.9 adapter at the A and B inputs; native at internal sites and for C | `sum` |
| ablation ranking | BeyondNN interventions: single-unit removal drops under the same replacement as the test (INTERVENTIONAL, circular at k = 1) | the drop |
| random | seeded permutation (seed 30 000 + index) | — |

- **Ranking:** by |score|, ties to the lower index.
- **Plan correction (§18):** for C, IG is native. Captum's LayerIG accepts only input-space baselines and cannot express the pre-registered layer-space zero baseline.
- **Numerical check:** IG at 32 vs 64 steps gives median top-10% Jaccard 1.0 on every site. The median max |Δ| is 0.013, 0.004, 0.018, 0.0, and 4e-5.

## 9. Replacement strategies

Every replacement is out of distribution to some degree and carries the corresponding limitation code. The resample seed is 20 000 + index.

| Site | r1 | r2 | r3 |
|---|---|---|---|
| A input (standardised) | zero (= training mean) | a resampled training row | the training minimum per feature |
| A hidden (`net.1`, 64) | zero | the training-mean activation | a resampled training activation |
| B pixels (64) | zero (black) | the training-mean image | a resampled training image |
| B channels (`relu2`, 16) | zero | the training-mean map | a resampled training map |
| C tokens (word-embedding output) | zero embedding | the [MASK] embedding at every position | the [PAD] embedding |

## 10. Control strategies

Each control is described by its seed, number, sampling scheme, matching rule and null.

| Control | Seed | N | Sampling | Matching | Null ("the selection is no different from…") |
|---|---|---|---|---|---|
| Count-matched (Phase 5) | 10 000 + index | 50 | uniform without replacement within a draw; independent draws | same site, same k | …a random set of the same size at the same site |
| Magnitude-matched (**new, ADR-035**) | 10 000 + index | 50 | per selected unit, a uniform draw from its own stratum | same site and k; 4 equal-count strata of ‖x_u − b_u‖₂ | …a random set perturbed by similar amounts |
| Random-ranking "method" | 30 000 + index | 1 per sample | seeded permutation | — | calibration of superiority (H7) |
| Curve controls | 10 000 + index | 20 | random permutations | same points | …a random ordering |

- **Considered and rejected:**
  - *Attribution-score-band matching:* circular, because it conditions on the method under test.
  - *Activation-magnitude matching:* for zero replacement it is identical to perturbation-magnitude matching, and for other replacements the perturbation (‖x − b‖) is the relevant quantity. So it is subsumed.
  - *Contiguous (RISE-like) masks:* image-specific, not general across A, B and C.
- **Statistics:** fractions below/tied/above and a one-sided Monte-Carlo p = (1 + b)/(N + 1) (minimum 1/51). Paired per-sample differences use a seeded sign-flip test (5 000 draws, seed 0). The word "significant" is not used. Statistics are kept separate from outcomes (see also §38 and API review F-21).

## 11. k policy

- p ∈ {1, 5, 10, 20}% with k = max(1, ceil(p · n)); duplicates collapse. The same k is used for every method, and all k are reported.
- **k values:**
  - A input: 1, 2, 3, 6;
  - A hidden: 1, 4, 7, 13;
  - B pixels: 1, 4, 7, 13;
  - B channels: 1, 2, 4;
  - C: 1 to 5 (varies with sentence length).

## 12. Threshold policy

- **Primary:** t = 0.5.
  - Comprehensiveness SUPPORTS iff drop ≥ t · margin.
  - Sufficiency SUPPORTS iff drop ≤ t · margin.
  - A no-op perturbation is INCONCLUSIVE.
- **Grid:** t ∈ {0.1, 0.25, 0.5, 0.75, 0.9}, derived from the stored raw drops. Raw drops are always reported.

## 13. RQ1 result: do attribution-selected units beat matched random controls?

Median superiority against count-matched controls, comprehensiveness, r1 (IQR in parentheses):

| site (k at p = 10%) | gradient | IxG | IG | ablation | random |
|---|---|---|---|---|---|
| A input (3) | 0.96 (0.74–1.00) | 1.00 (0.99–1.00) | 1.00 (0.98–1.00) | 1.00 (0.95–1.00) | 0.46 (0.26–0.74) |
| A hidden (7) | 0.93 (0.59–1.00) | 1.00 | 1.00 | 1.00 | 0.54 (0.26–0.70) |
| B pixels (7) | 0.67 (0.06–0.94) | 0.98 (0.63–1.00) | 0.99 (0.71–1.00) | 1.00 (0.64–1.00) | 0.46 (0.26–0.76) |
| B channels (2) | 0.69 (0.36–0.91) | 0.99 (0.88–1.00) | 0.99 (0.88–1.00) | 0.99 (0.88–1.00) | 0.45 (0.21–0.69) |
| C tokens (2) | 0.84 (0.21–0.98) | 0.90 (0.23–0.99) | 0.94 (0.80–0.99) | 0.92 (0.75–0.98) | 0.48 (0.18–0.70) |

**H1 held at every site.** Four qualifications stop this from being read as "attributions are faithful":

1. **Relative is not absolute.** At t = 0.5, IG comprehensiveness SUPPORTS for 6/60 (A input), 19/60 (A hidden), 42/60 (B pixels), 21/60 (B channels) and 16/40 (C). IG's top 10% beats random sets but usually does not remove half the margin. Paired against the random method (p = 10%, r1), IG's median extra drop is 2.83, 4.02, 4.28, 2.18 and 1.10 margin units; win rates are 0.92, 1.00, 0.80, 0.87 and 0.72; the sign-flip p ranges from 0.0002 to 0.003.
2. **Circularity.** Under zero replacement in these piecewise-linear networks, input × gradient with a zero baseline is the first-order effect of zeroing a unit. At A hidden and B channels, the IxG, IG and ablation selections were *identical* (top-k Jaccard 1; paired differences exactly 0). Their agreement with interventions under r1 is largely by construction.
3. **Stronger controls.** They lower the pixel result (§26).
4. **Other replacements.** Under r2/r3, IG's superiority stays at 0.82–1.00 at every A/B site except A input under the training minimum (0.69). C is 0.94 under every replacement.

## 14. RQ2 result: which methods better predict intervention-sensitive units? (scoped; no universal ranking)

- **H2** (k = 1: ablation ≥ IG ≥ IxG ≥ gradient):
  - Under r1 it held at 4 of 5 sites. The exception is A input, where IxG (median drop 1.71) beat IG (1.69).
  - Under r2/r3 it failed at A input and at B pixels.
  - At k = 1, ablation is circular.
- **Scoped findings:**
  - *Under zero replacement on B pixels*, IG's top-10% pixels had larger drops than gradient's (median difference 3.16, win rate 0.87, sign-flip p 0.0002). They also had larger drops than input × gradient's (0.34, 0.58, 0.002).
  - *Under the mean-image replacement on B pixels*, gradient beat IG (median superiority 0.98 vs 0.91). Gradient ranks black background pixels. Zeroing a black pixel is a no-op; replacing it with the mean image is not.
  - *At A hidden under zero replacement*, gradient was near or below chance at k = 1 (median drop 0.0; 39 gradient selections consisted only of dead ReLU units, an exact no-op → INCONCLUSIVE). IG, IxG and ablation coincided.
  - *On C tokens under all three replacements*, IG had the highest median superiority (0.94). Its paired advantage over gradient was small (median difference 0.0, win rate 0.42, p 0.009); most selections coincide at k = 2.
- **Conclusion:** no method is "best". Rankings depend on the replacement and the site.

## 15. RQ3 result: how often do comprehensiveness and sufficiency disagree?

**H3 held.** Share of decided (sample, method, k, replacement) cells where one SUPPORTS and the other CONTRADICTS:

| site | share | not necessary but sufficient | necessary but not sufficient | excluded (no-op) |
|---|---|---|---|---|
| A input | 51.7% | 1431 | 428 | 3 |
| A hidden | 24.1% | 679 | 164 | 101 |
| B pixels | 36.5% | 520 | 752 | 112 |
| B channels | 23.6% | 351 | 283 | 9 |
| C tokens | 48.9% | 972 | 3 | 0 |

The pooled share is 36.6%.
- **C is a clear redundancy pattern:**
  - Retaining only IG's top-10% tokens (the others' embeddings zeroed) loses a median 5.6% of the margin; random retention loses 97%.
  - Removing the same tokens rarely halves the margin.
  - The same sufficiency test on random selections mostly CONTRADICTS (28/40), so sufficiency is informative here and not trivially satisfied.

## 16. RQ4 result: how sensitive are conclusions to the replacement?

- **H4:**
  - The share of (sample, method, k) comprehensiveness outcomes that change across the three replacements is 32.7% (A input), 27.5% (A hidden), 49.2% (B pixels), 31.4% (B channels) and 5.0% (C).
  - Median method orderings reverse between replacements at every site (31, 2, 20, 8 and 8 reversed pairs).
  - **H4 held for A and B, and not for C (5% < 10%).** The three token replacements (zero, [MASK], [PAD] embeddings) behave alike for BERT-tiny.
- **The same IG selection changes outcome with the replacement** in 88, 71, 131, 73 and 7 (sample, k) cases.
  - *Case study*, found by rule as the first such case: A input, sample 327, k = 2. The drop is 2.77 (CONTRADICTS) under the train-mean replacement, 6.76 (SUPPORTS) under a resampled row, and −0.66 (CONTRADICTS, the margin *increases*) under the training minimum.

## 17. RQ5 result: sensitivity to k and threshold

- **H5 (t ∈ {0.25, 0.5, 0.75}):** the share of cells whose outcome changes is 25.5%, 32.5%, 28.1%, 34.2% and 16.5%. **It held for A and B, and not for C.**
- **Share of comprehensiveness SUPPORTS by threshold** (all cells, count controls):

| t | A input | A hidden | B pixels | B channels | C |
|---|---|---|---|---|---|
| 0.10 | 0.59 | 0.61 | 0.69 | 0.75 | 0.55 |
| 0.25 | 0.38 | 0.37 | 0.59 | 0.50 | 0.43 |
| 0.50 | 0.18 | 0.14 | 0.42 | 0.23 | 0.30 |
| 0.75 | 0.09 | 0.04 | 0.30 | 0.11 | 0.21 |
| 0.90 | 0.07 | 0.02 | 0.24 | 0.07 | 0.17 |

- **k:**
  - IG's superiority is flat in k (0.94–1.00).
  - Gradient's rises with k at B pixels (0.43 → 0.86 from 1% to 20%) and at A hidden (0.33 at 1% to 0.93 at 10%, then 0.84 at 20%).
  - Magnitude-matched superiority *falls* with k at B pixels (IG 0.96 → 0.75; IxG 0.94 → 0.69).
- **Conclusion:** "SUPPORTS" depends as much on t as on the method. The raw drops are the robust quantity.

## 18. RQ6 result: does attribution disagreement correspond to faithfulness disagreement?

- **H6 predicted a weak negative association** (Spearman ρ between top-k Jaccard and |Δ comprehensiveness drop| < 0 with |ρ| < 0.5). **Not held:** ρ = −0.76, −0.60, −0.56, −0.76 and −0.86 (attribution methods; −0.63 to −0.89 with ablation included). The association is strong, but partly mechanical: identical sets (Jaccard 1) give Δ = 0 exactly.
- **Per method pair** (p = 10%, r1; median Spearman of full rankings / median top-k Jaccard / median comprehensiveness drop difference):

| pair | A input | A hidden | B pixels | B channels | C |
|---|---|---|---|---|---|
| gradient vs IG | 0.74 / 0.20 / −1.16 | 0.26 / 0.17 / −3.19 | 0.27 / 0.17 / −3.16 | 0.31 / 0.00 / −1.41 | 0.55 / 0.50 / 0.00 |
| IxG vs IG | 0.98 / 1.00 / 0 | 1.00 / 1.00 / 0 | 0.99 / 0.75 / −0.34 | 1.00 / 1.00 / 0 | 0.69 / 0.75 / 0 |
| IG vs ablation | 0.97 / 1.00 / 0 | 1.00 / 1.00 / 0 | 0.99 / 0.75 / 0 | 1.00 / 1.00 / 0 | 0.71 / 1.00 / 0 |

- High agreement is not evidence of correctness (§13, point 2).
- **Agreement vs stability (B pixels):** in 17 of 60 samples, IG and ablation agree (top-10% Jaccard ≥ 0.5) but IG's stability under a 1-pixel roll is poor (top-k Jaccard ≤ 0.3). The Spearman between agreement and stability is 0.10.

## 19. RQ7 result: consistency across held-out samples

- **H7 held:** the random method's median superiority is 0.45–0.54 at every site, so the superiority scale is calibrated.
- The IQRs in §13 show that IG's median of ≈ 1 coexists with a lower quartile of 0.71 (B pixels) and 0.80 (C).
- **Every site has samples where IG's top-k removal *increases* the margin.** *Case studies* found by rule (the lowest IG superiority, r1):
  - A input, sample 81, k = 1: drop −3.63 on a margin of 1.06; superiority 0.0.
  - B pixels, sample 95, k = 1: −3.28 on 2.16.
  - B channels, sample 245, k = 1: −1.72 on 1.59.
  - C, sample 565, k = 2: −0.12 on 5.60.
- **Correlates** (descriptive only; no cause inferred): IG comprehensiveness failures at p = 10%, r1 have *larger* median margins than successes at A (12.0 vs 7.1 input; 14.7 vs 8.8 hidden) and B (6.7 vs 5.8 pixels; 7.4 vs 5.2 channels). In C they do not (4.0 vs 4.1). IG completeness deltas are small everywhere (median |δ| ≤ 0.025), so there is no evidence that IG saturation explains the failures.

## 20. RQ8 result: do the abstractions work without architecture-specific hacks?

- **H8 held.** Before any change, the unchanged Phase-5 API (`results/abstraction_probe.json`):
  - expressed A;
  - refused B's pixels and channels and C's token positions ("non-last dimensions");
  - refused a declared 64-pixel selection only after running every pass.
- **Tensor-shape / unit abstraction review** (request §25):

| Evidence unit | Phase 5 (last-axis index) | After ADR-034 (declared axis grid) | Notes |
|---|---|---|---|
| tabular features | yes | yes | A input |
| neurons | yes | yes | A hidden |
| channels | no | **yes**, `unit_axes=(1,)` | B `relu2` (a 16 × 8 × 8 map per unit) |
| pixels / positions | no | **yes**, `unit_axes=(2, 3)` | B input (every channel of a pixel together) |
| tokens / sequence positions | no | **yes**, `unit_axes=(1,)` | C word embeddings (128 dims per token) |
| regions / superpixels, spans, words from word pieces | no | **no** | Needs mask sets; not required by A/B/C (ADR-034 consequences) |
| attention heads | no | only if a hooked module outputs a head axis | Heads usually live inside a reshaped hidden dimension; not tested |
| directions (feature bases) | no | **no** | Projections, not axis grids: the Phase-6 `FeatureBasis` scope |

- **Conclusion:**
  - "unit = final-dimension index" was insufficient for realistic image and sequence models.
  - The minimum generalisation is declared axis-grid units, with an explicit per-unit reduction (ADR-034). Its migrations keep every Phase-5 record meaning.
  - With it, all three settings ran through the same public API with no model wrappers (`abstraction_probe_after_adr034.json`).
  - Directions and irregular regions remain unsupported.

## 21. RQ9 result: does BeyondNN reveal disagreement a single score would hide?

- **H9 held:**
  - Share of (method, k, replacement, test) cells whose samples include both SUPPORTS and CONTRADICTS: 98% (A input), 68% (A hidden), 100% (B pixels), 100% (B channels), 95% (C).
  - A majority-outcome aggregate would hide 25%, 21%, 29%, 20% and 22% of the per-sample results in those cells.
- **Collapsed by any single score** (§§15–18):
  - comprehensiveness/sufficiency disagreement: 24–52%;
  - removal/retention curve disagreement on method pairs: 18%, 13%, 9%, 0% and 33%;
  - replacement reversals.
- BeyondNN keeps each as a separate record (ADR-007 unchanged).

## 22. Naturally occurring counterexamples

All were found by fixed rules in `analysis.py` over the declared samples. Each is a *case study*, first by sample order or extreme by rule.

| Pattern | Where | Case study | Count |
|---|---|---|---|
| Attribution top-k whose removal is an exact no-op | A hidden (dead ReLUs, gradient); B pixels (black pixels, gradient) | A hidden, sample 327, unit 16 | 39; 22 |
| Removing IG's top unit *raises* the margin | every site | B pixels, sample 95: drop −3.28 on a margin of 2.16 | lowest superiority 0.0–0.01 at every site |
| All three attribution methods agree on the top-k set, and the claim fails | every site | B pixels, sample 1361, pixel 42: drop −4.24 on a margin of 0.84 | first found per site |
| Beats count controls, not magnitude controls | B pixels, B channels, C | B channels, sample 245, IxG k = 4: superiority 0.98 → 0.20 | §26 |
| Outcome flips with the replacement | every site | A input, sample 327 (§16) | 88 / 71 / 131 / 73 / 7 |
| Sufficient but not necessary | C (972 cells) | §15 | — |

- **Discarded as outliers:** none.

## 23. Metric-disagreement cases

- **Comprehensiveness vs sufficiency:** §15 (24–52% of cells).
- **Removal vs retention curves** (r1): method pairs ranked differently by removal AOPC and retention AOPC in 17.7%, 13.0%, 9.2%, 0.0% and 32.9% of decided pairs. The highest is C.
- **High agreement but poor stability:** 17 of 60 B samples (§18).
- **Replacement reversal of a conclusion:** §16; for B pixels, "gradient is near chance" (r1) becomes "gradient is best" (r2).

## 24. Attribution-method disagreement cases

- **Gradient vs the rest:** median top-10% Jaccard 0.00–0.20 at A and B, and 0.50 in C. Gradient's selections lose 1.2–3.2 margin units of comprehensiveness against IG at A and B.
- **IxG vs IG:** agree almost perfectly at A and B channels. At B pixels they agree at 0.75 (IG removes 0.34 more; win rate 0.58).
- **The library's own `method_agreement`** (with one shared `sum` reduction; API review F-12) gives the same picture: gradient–IG top-k Jaccard 0.17–0.33, IxG–IG 0.75–1.0.
- No universal method ranking is claimed.

## 25. Replacement-sensitive cases

- **B pixels:**
  - At p = 10%, IG's comprehensiveness SUPPORTS share is 70% under zero, 42% under the mean image and 45% under a resampled image.
  - Gradient's superiority moves from 0.67 to 0.98.
- **A hidden:** IG's SUPPORTS share is 32% (zero), 3% (training mean) and 15% (resample).
- **A input:** the training-minimum replacement gives negative median drops for gradient at k = 1 (−0.66).
- **C:** replacement-insensitive by comparison (5% of outcomes change).

## 26. Stronger-control findings

The magnitude-matched control is **implemented and justified** (ADR-035; the confound: "large perturbations change the output regardless of the method").

| site | IxG count → magnitude | IG count → magnitude | ablation | gradient |
|---|---|---|---|---|
| A input | 1.00 → 0.99 | 1.00 → 0.99 | 1.00 → 0.99 | 0.96 → 0.98 |
| A hidden | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 0.93 → 0.96 |
| B pixels | 0.98 → 0.83 | 0.99 → 0.85 | 1.00 → 0.96 | 0.67 → 0.71 |
| B channels | 0.99 → 0.96 | 0.99 → 0.96 | 0.99 → 0.96 | 0.69 → 0.72 |
| C tokens | 0.90 → 0.82 | 0.94 → 0.84 | 0.92 → 0.83 | 0.84 → 0.81 |

*(median superiority, comprehensiveness, r1, p = 10%)*

- **B pixels at p = 20%:** IG 0.94 → 0.75 and IxG 0.90 → 0.69. *Case studies*:
  - IG, sample 493, k = 13: 0.94 → 0.42;
  - IxG on B channels, sample 245: 0.98 → 0.20.
- **H1b** (magnitude controls hurt IxG more than IG) **mostly did not hold**. It held only at A input and B pixels, each by ≤ 0.01. IxG and IG are affected alike, and at C, IG slightly more.
- **Did stronger controls change conclusions?** On pixels, yes quantitatively: a quarter of the apparent advantage at large k is explained by perturbation size. Qualitatively, no site's IG or ablation selection fell to chance (median ≥ 0.75).
- **The random method is unaffected** (0.45–0.54), as expected under both nulls.

## 27. MLP findings (model A)

- **Selections beat random controls:** IG/IxG/ablation at superiority ≈ 1.00 for both sites and all k, robust to magnitude matching.
- **Few absolute SUPPORTS:** 10% (input) and 32% (hidden) at t = 0.5, consistent with correlated features and redundancy.
- **Comprehensiveness and sufficiency disagree** in 52% of input cells, mostly "not necessary but sufficient".
- **Dead ReLU units:** gradient selects them (39 no-op selections); BeyondNN reports INCONCLUSIVE, not CONTRADICTS.
- **The training-minimum replacement** makes gradient's selections *raise* the margin (median −0.66 at k = 1), and 88 IG selections flip outcome with the replacement.

## 28. CNN findings (model B)

- **Pixels and channels were expressible only after ADR-034.**
- **Pixel results:**
  - the most replacement-sensitive site (49% of outcomes change);
  - the most weakened by magnitude-matched controls;
  - gradient's pixel rankings are driven by zero-valued background, which makes it look bad under zero replacement and best under the mean image.
- **Channel results:**
  - IG, IxG and ablation coincide under r1;
  - removal and retention curves agree on every method pair.
- **Stability under a 1-pixel circular roll:**
  - median prediction change 2.56 logits, more than half the margin in 45% of samples;
  - full-ranking ρ 0.95 but top-k Jaccard 0.40;
  - claim outcome unchanged in 77%.
  - The four aspects are reported separately; none is collapsed.
- **Realistic runs found three framework bugs here** (ADR-036, ADR-038, and the reduction-record fix) and one expressivity gap (ADR-034).

## 29. Transformer findings (model C)

- **Token positions of the word-embedding output** were expressible after ADR-034, with keyword inputs throughout.
- **Diagnostics** were not expressible until ADR-037.
- **Selections and replacements:**
  - Every attribution top-1 was a word token, never [CLS]/[SEP]; the plan's predicted failure mode did not occur.
  - IG's top-2 tokens beat count controls (0.94), and less clearly magnitude-matched ones (0.84).
  - The replacement matters little (5%), because the three token replacements behave alike.
  - Threshold sensitivity is the lowest (17%).
- **Redundancy:** the dominant structure is "sufficient but not necessary". One or two sentiment words retain the margin, and removing them rarely halves it.
- **Removal and retention curves disagree most here** (33% of method pairs).
- **Cost:** 0.51 s per claim test with N = 50 (§33).

## 30. API problems discovered

`docs/PHASE_5_5_API_REVIEW.md` has 23 findings.

- **BUG, all fixed with regression tests:** F-3 (false stochasticity refusal), F-4 (curve records dropped unit axes), F-5 (declared reduction dropped), F-23 (exact-equality magnitude check).
- **MISSING CAPABILITY:**
  - fixed: F-1 (unit axes), F-11 (keyword inputs in diagnostics);
  - open: F-7 (`run_dataset` has one target and one criterion), F-8 (no kwargs in `run_dataset`), F-9 (no selection source for intervention rankings), F-10 (curves refuse declared rankings).
- **SCIENTIFIC DESIGN ISSUE:**
  - fixed: F-2 (controls confounded with perturbation size);
  - open: F-21 (SUPPORTS without beating random controls: 218 SUPPORTS for the random method), F-22 (implicit zero replacement).
- **API ERGONOMICS:** F-6 (partly fixed), F-12, F-15, F-18.
- **DOCUMENTATION ISSUE:** F-13 (two sign conventions), F-17 (fixed).
- **EXPECTED LIMITATION:** F-14, F-16, F-19, F-20.

## 31. Framework changes made

Each change comes from an observed realistic failure and has an ADR with alternatives, a regression test that fails before the fix, the full suite re-run, and backward compatibility checked.

| Change | Commit | ADR | Backward compatibility |
|---|---|---|---|
| Declared unit axes; explicit per-unit reduction; early input check | `622150d` | ADR-034 (supersedes ADR-033's "Units" bullet only) | `unit_axes=None` keeps the last-axis meaning. Migrations: intervention v2→v3, claim v1→v2, selection v1→v2. The **golden claim id changes** (claim v2 adds `subject.unit_axes`; documented in `tests/test_record_identity.py`). Other Phase-5 ids are unchanged. |
| Magnitude-matched controls | `622150d` | ADR-035 | Opt-in; count controls remain the default |
| Reproducibility tolerance scaled to output precision | `4862125` | ADR-036 | Accepts more deterministic models; real drift is still refused |
| Curve selection keeps unit fields | `2e4e662` | ADR-034 fix | — |
| Declared reduction recorded for single-element units | `a5768ec` | ADR-034 fix | — |
| `units` docstring | `071f05b` | — | — |
| `model_kwargs` in the diagnostics | `3c7650b` | ADR-037 | Additive; diagnostic ids are unchanged |
| Magnitude re-derivation within rounding tolerance | `39bdd75` | ADR-038 (refines ADR-035's verification) | Controls are still re-drawn exactly |

- The base install is still `torch` only. The experiment dependencies live in `experiments/phase5_5/requirements.txt`.

## 32. Changes considered but rejected

- **Flattening or wrapping models in user code for pixels and tokens:** architecture-specific, loses the site, and changes the fingerprint (ADR-034).
- **Arbitrary per-unit masks:** not auditable, and they allow overlapping units.
- **Named dimensions.**
- **Implicit reductions.**
- **Attribution-score-band controls:** circular.
- **RISE-like contiguous masks:** image-only.
- **ROAR retraining:** out of scope and changes the model under test.
- **ROAD imputation:** a different confound, covered by r1–r3.
- **Grad-enabled reference pass, user-settable tolerance, or a fixed 1e-4 tolerance** (ADR-036).
- **Computing magnitudes from the family's own pass** (ADR-038).
- **Making `replacement` required** (F-22): breaks the Phase-5 API; left to the owner.
- **Changing the `comprehensiveness/v1` semantics to require control superiority** (F-21): would alter Phase-5 records; recommended as a Phase-6 policy instead.
- **Per-sample targets and relative criteria in `run_dataset`** (F-7): a design change larger than the phase's fixes; deferred.
- **A selection source "intervention"** (F-9): deferred.
- **A global faithfulness or explanation score:** rejected (ADR-007).

## 33. Performance measurements

**Setup:**
- Idle machine, one torch thread; median of 5 repeats (3 for claim tests, 1 for control scaling) after a warm-up.
- macOS arm64 (8 cores, 16 GB); Python 3.12.13, torch 2.14.0, Captum 0.9.0; BeyondNN `8a2a536`, whose code equals `39bdd75`.
- Inputs: A (1, 30); B (1, 1, 8, 8); C (1, T) token ids, first declared sentence.

| | A input | A hidden | B pixels | B channels | C tokens |
|---|---|---|---|---|---|
| plain forward | 0.012 ms | — | 0.027 ms | — | 0.28 ms |
| trace | 0.46 ms (6 records) | 0.55 ms (7) | 0.50 ms (6) | 0.57 ms (7) | 10.0 ms (7) |
| gradient / IxG / IG-32 | 0.94 / 0.89 / 3.9 ms | 1.13 / 1.12 / 3.5 ms | 1.05 / 0.99 / 5.5 ms | 1.21 / 1.16 / 4.2 ms | 29 / 29 / 48 ms |
| one intervention | 1.0 ms (14 records) | 1.3 ms | 1.1 ms | 1.3 ms | 19.5 ms |
| comprehensiveness, N = 50 count | 31.7 ms (344 records, 324 KB saved) | 38.5 ms (414) | 33.1 ms (362) | 38.9 ms (363) | 508 ms (360; 483 KB) |
| same, N = 50 magnitude | 31.6 ms | 39.1 ms | 32.7 ms | 38.9 ms | 518 ms |
| controls N = 10 / 50 / 200 (single run) | 7.4 / 80.7* / 135 ms | 8.8 / 39.5 / 177 ms | 7.7 / 33.6 / 142 ms | 8.8 / 38.5 / 176 ms | 120 / 509 / 2043 ms |
| composition with verification | 7.8 ms | 9.9 ms | 8.3 ms | 8.8 ms | 19.3 ms |

\* A single-shot outlier; the repeated median at N = 50 is 31.7 ms.

- **Dataset runtime, full grid** (3–4 concurrent runs on 8 cores): A 29 min (16,800 claim tests), B 19 min (14,700), C 75 min (4,655).
- The median per sample and site is 8–18 s for A/B and 112 s for C.
- Cost is linear in the number of perturbation passes. The overhead over a plain forward pass is large for tiny models (the MLP forward is 12 µs) and small in relative terms for BERT.
- These are microbenchmarks of these models on this machine, not universal figures.

## 34. Reproducibility information

**Recorded in every result file:**
- dataset and version:
  - the scikit-learn bundled datasets, version 1.9.1 (the plan said 1.9.0; logged in §18);
  - the SST-2 parquet at `bcdcba79…`, sha256 `a1371f3b…`;
- model and checkpoint digests (§6); the BERT repo and revision;
- training seed 0; evaluation seeds: selection 1234, controls 10 000 + index, resample 20 000 + index, random ranking 30 000 + index, curve controls 10 000 + index;
- sample ids;
- the target definition (the margin, via `metrics.difference`, fixed from the clean pass);
- attribution configurations, including the IG rule, steps and baseline, and the Captum version;
- k values;
- the intervention strategy per replacement, with tensor digests in the spec;
- the control count and strategy;
- thresholds (the primary one in the spec; the grid in the analysis);
- hardware and package versions;
- the git commit.

**Determinism evidence:**
- B's full re-run reproduced all 14,700 rows, including content-derived result ids.
- C's first 3 sentences re-run at HEAD reproduced all 315 rows.
- To re-run: `uv run --no-project --python 3.12 --with-requirements experiments/phase5_5/requirements.txt --with-editable . python experiments/phase5_5/<script>.py`.

## 35. Mutation and adversarial checks

- **Scope:** 20 mutations of the Phase 5.5 code, run on scratch copies of the whole suite. **All caught.**

| # | Mutation | Tests failing |
|---|---|---|
| U1 | wrong image/token unit mapping (reversed unit index) | 5 |
| U2 | within-unit reduction over the wrong axes | 2 |
| U3 | intervention ignores `unit_axes` | 11 |
| U4 | implicit reduction allowed | 1 |
| U5 | no early input unit check | 1 |
| U6 | evaluator ignores the claim's `unit_axes` | 1 |
| U7 | evaluator ignores the controls' `unit_axes` | 1 |
| U8 | verifier ignores the selection's reduction | 3 |
| U9 | diagnostics drop the unit parameters | 1 |
| M1 | strata ignore magnitude | 1 |
| M2 | control draws ignore strata | 2 |
| M3 | magnitudes not re-derived | 2 |
| M4 | magnitudes computed with `abs_sum` instead of `l2` | 2 |
| F1 | curve selection drops its unit fields | 1 |
| F2 | reduction not recorded for single-element units | 1 |
| F3 | tolerance ignores output precision | 1 |
| F4 | diagnostics drop `model_kwargs` when anchoring | 2 |
| F5 | diagnostic identity check ignores `model_kwargs` | 1 |
| F6 | magnitude check exact again | 1 |
| F7 | magnitude check unbounded | 2 |

- **Mutation U7 initially survived.** No test covered a control whose `unit_axes` differ from the selection's. A test was added before the final run.
- **Adversarial realism:** composition re-derived every first-sample result at every site (630 claim tests plus curves and diagnostics). It caught three real record misstatements during the phase: F-4, F-5, and F-23 (a false alarm caused by rounding).

## 36. Final test counts

| Environment | Result |
|---|---|
| Python 3.10.16 / 3.12.13 / 3.14.3 (torch 2.14.0), without Captum | **905 passed, 1 skipped** (the Captum cross-check module, reported) |
| same, with Captum 0.9.0 | **926 passed**, 0 skipped |

- The baseline was 877 + 1 / 898. There are 28 new tests: `tests/test_unit_axes.py` (25), `tests/test_diagnostics_kwargs.py` (2), and one tolerance test in `tests/test_attribution.py`.
- One existing golden id was updated deliberately, with a comment (the claim v2 schema).
- There are no other Phase 1–5 regressions.
- Python 3.14 now resolves torch 2.14.0 (the baseline had 2.12.0).

## 37. Lint, type and build status

- **Code checks:** ruff check and ruff format are clean (101 files, including `experiments/`). mypy `--strict` is clean with and without Captum (89 source files).
- **Build:** sdist and wheel build (`uv build`).
- **Fresh wheel install** (Python 3.12), without and with Captum:
  - the Phase 1–5 smokes pass;
  - a new Phase-5.5 smoke passes (conv pixel units, magnitude-matched controls, channel curves, diagnostics with keyword inputs, composition);
  - all 5 README examples pass.

## 38. Remaining scientific weaknesses

1. **Circularity under zero replacement.** In piecewise-linear nets, IxG/IG agreement with zero ablation is partly mathematical (§13). The non-circular evidence is the r2/r3 results and the magnitude-matched controls.
2. **Every replacement is off-distribution.** No replacement is "correct"; conclusions are replacement-relative (H4). ROAR and ROAD were not done.
3. **Relative vs absolute.** High superiority coexists with few absolute SUPPORTS, and absolute outcomes depend strongly on t (H5).
4. **SUPPORTS without controls.** A claim-test SUPPORTS can be issued for a selection no better than random controls, unless a control criterion is declared (F-21: 218 random-method SUPPORTS).
5. **Scale.** Three small models, CPU-sized, one seed per trained model; there is no training-seed variation, and C is a single external checkpoint. The results are scoped to these models and data.
6. **Pre-registration errors.** H1b and H6 were wrong, and the [CLS]/[SEP] failure mode did not occur. These are recorded, not reinterpreted.
7. **Mask-shape leakage** (ROAD) was not measured for pixels.
8. **Stability** was tested only for B, and only with one declared transformation.
9. **Statistics.** Monte-Carlo p has a floor of 1/51 with N = 50. The statistics are descriptive, with no multiplicity correction; none are used as verdicts.

## 39. Remaining engineering limitations

- **Dataset runs** (F-7, F-8): one target and one criterion only, and no keyword inputs. Realistic dataset-level summaries were therefore computed in experiment code, not stored as verifiable `ProtocolResult`s. Per-sample results *are* verifiable records.
- **Ablation rankings** (F-9, F-10): no provenance-bearing selection source, and no library curves along declared rankings.
- **Minor gaps:** one reduction per diagnostic (F-12); `faithfulness.stats` not exported (F-15); untyped statistics (F-18); two sign conventions (F-13); an implicit zero replacement (F-22).
- **Units:** grid units only; directions and regions are unsupported (F-20). Internal-site shape errors are detected only after the passes (F-6).
- **Cost:** about 0.5 s per claim test with 50 controls on BERT-tiny. Larger models would need batching of perturbation passes, which does not exist yet.
- **Release blockers RB-1, RB-2 and RB-3 remain:** CoC and security contact placeholders, and CI never run. Nothing was pushed.

## 40. Exact gate decision and reasoning

**READY FOR PHASE 6 WITH EXPLICIT LIMITATIONS.**

**Why not NO-GO:**
- No record misstating what was done survived.
- The three misstatements the realistic runs produced (F-4, F-5, and the F-23 false alarm) were refused by composition, never accepted silently. They were then fixed with regression tests.
- The evidence structure for all three settings is represented without breaking Phase 1–5 semantics: legacy meaning is preserved, and migrations are tested.

**Why not REQUIRES CHANGES:**
- Every substantive framework problem exposed in this phase was fixed in-phase with an ADR, a regression test and a re-run: the four bugs, unit axes, keyword inputs in diagnostics, and magnitude controls.
- The open items (F-7 to F-10, F-12, F-15, F-18, F-21, F-22) are capability or ergonomic gaps. Each has a documented, verifiable workaround (per-sample records) or is an owner decision. None makes a record wrong.

**Why not READY without limitations:**
- The premise holds only in a scoped form.
- Attribution-selected units reliably beat random controls, including stronger ones. But whether a claim is SUPPORTED depends heavily on the replacement (up to 49% of outcomes), the threshold (17–34%) and the protocol (24–52% comprehensiveness/sufficiency disagreement).
- Several apparent agreements are circular under zero ablation.
- A single outcome field can be misleading unless it is read with its control statistics (F-21).

**Summary:** BeyondNN's value in these experiments was that it made that dependence explicit, per sample and per protocol.

## 41. What Phase 6 may assume

- Units may be declared over any axis grid of a site (features, neurons, channels, pixels, token positions) at input or internal sites. Interventions, selections, claims, curves and diagnostics carry and verify them.
- Claim tests, curves and diagnostics accept keyword model inputs.
- Magnitude-matched controls exist and are verifiable.
- Records are content-addressed and reproducible across runs on CPU (§34). Composition re-derives results and refuses misstated units, reductions, magnitudes, samples and targets.
- Realistic trained models (MLP, CNN, BERT-tiny) run through the public API without wrappers.

## 42. What Phase 6 must NOT assume

- That attribution agreement, or high superiority, implies a claim is SUPPORTED, or vice versa.
- That any replacement is neutral, or that conclusions transfer between replacements.
- That a SUPPORTS outcome implies "better than random" unless the test declared a control criterion (F-21). Concept-validation policies must declare it.
- That `run_dataset` can express per-sample targets or criteria, keyword inputs, or intervention-derived rankings (F-7 to F-10).
- That units can be directions or irregular regions. Concept directions need Phase 6's own `FeatureBasis` interventions (F-20).
- That zero-ablation agreement with IxG/IG is independent evidence in piecewise-linear models.
- That these results transfer to larger models, other datasets or other seeds.

## 43. Commits

Phase 5.5 commits after `28d4026`, in order:

1. `23c0047` Add the Phase 5.5 literature review and pre-registered validation plan
2. `7e29fe9` Experiment infrastructure and pre-change abstraction probe
3. `622150d` Declared unit axes (ADR-034) and magnitude-matched controls (ADR-035)
4. `4862125` Scale the attribution reproducibility tolerance to output precision (ADR-036)
5. `2e4e662` Keep declared unit axes on curve selection records (ADR-034 fix)
6. `a5768ec` Record a declared unit reduction even for single-element units (ADR-034 fix)
7. `efa0712` Plan §18: corrections and analysis-rule clarifications (before analysis)
8. `071f05b` Document `n_units` under declared unit axes
9. `3c7650b` Accept `model_kwargs` in faithfulness diagnostics (ADR-037)
10. `39bdd75` Verify control magnitudes within rounding tolerance (ADR-038)
11. `9fbb43d` MLP experiment (model A) and post-ADR-034 abstraction probe
12. `8a2a536` CNN experiment (model B)
13. `db6bbd6` Transformer experiment (model C)
14. `fb55137` Analysis (H1–H9) and performance measurements
15. `a04ff13` API review, changelog/roadmap updates, plan §18 run provenance
16. `fe9c074` Experiment-log entry
17. `95b793b` Correct experiment-log counts
18. This report

All commits are local. Nothing was pushed or published, and no commit has a co-author trailer.

## 44. Git status

Clean after the report commit. The ignored `experiments/phase5_5/artifacts/` holds the cached checkpoints, the SST-2 parquet, run logs and a pre-ADR-038 copy of B's results.

## 45. Any uncommitted work

None.
