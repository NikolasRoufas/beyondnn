# Manuscript-to-repository consistency (TNNLS revision, Stage 1, §3.E)

## Document checked

| field | value |
|---|---|
| file | `TNNLS___BeyondNN__Auditing_Interpretability_Claims_with_Provenance__Controls__and_Counterevidence.pdf` (supplied by the author on 2026-11-03; not committed, because the repository is public) |
| SHA-256 | `0471417fcfae81692d523e00873b9c118bfaaef95b5614946e0011e072a1fa60` |
| pages | 16 (IEEE TNNLS two-column format), 57 references |
| PDF creation time | 2026-11-03 16:57 EET (pdfTeX 1.40.29) |
| LaTeX source / supplementary appendices A–D | **not supplied** |

**Identity question for the authors:** the research plan refers to a 19-page manuscript. This file has 16 pages, is already in TNNLS format, and was compiled on 2026-11-03. Its wording ("this revision", "the draft", "source bundle", "checkout") suggests it is itself an earlier revision. That revision appears to have been prepared without access to the repository. The authors should confirm:
- which file is the *original* version to preserve;
- whether this PDF is the starting point for `paper_tnnls/`.

All row IDs below refer to `MANUSCRIPT_RESULT_VERIFICATION.csv`.

## 1. Numbers: almost all verified

Every quantitative claim checked against the frozen artifacts matches **except one description (R08b)**. Exact recomputation covers:
- the running example (sample 423: margin 3.97, drop 2.66, τ = 1.99, 5/8/4 alternatives, 12/13/2 stress tests, content-token drop −0.54);
- 47/52 and the per-site counts;
- 302 held-out samples;
- the 35% median contradicting share, 26/52 and 5/52;
- gradient 37/60 on CNN pixels;
- 30/40, 10/11, 6/40, 3/11 and the 5/3/3 transitions;
- 0/400, 591/760, 146/183, 19 used / 10 decoy components, 28–38 of 40;
- mean 586/760, zero 400/400 and 760/760;
- 479/560, 425/431, 120 no-op instances;
- InterpBench 942/942, 0/8,840, 1,150/8,840 [12.3, 13.7] and 1,571/1,960;
- the concept and calibration claims;
- e-SNLI 0.34, +0.149, 10/7/3, 5 of 7 in 17;
- 26,904 audited results.

These are recomputed by `recalculate_stage1.py → manuscript_checks` and the other sections.

## 2. Contradictions with the repository

| row | location | manuscript says | repository shows | fix |
|---|---|---|---|---|
| R08b | §VI-A b; §VI-E a, c; Fig 4a | "0 of 160 exact-copy decoy cases"; Fig 4a labels the 160 as a subset of exact-copy decoys | only `y_decoy` (80 instances) is an exact copy of the used variable; `tok_copy` (80) copies the token stream | "0 of 160 decoy tests in the two correlated-decoy programs, including 0 of 80 exact copies of the used variable" |
| M19 | §V-E | operations "zero, mean, constant, patch, scale" | `InterventionOperation` = zero, constant, patch, direction; mean/resample are declared replacement tensors | correct the list |
| M02 | Fig 2 caption | "inherited summaries; input-level records and the extraction script are absent" | per-sample record for index 423 is in `central775_D_heldout_{all,content_tokens}.json`; script `central775.py` | cite the files |
| M21 | §II-F | Captum adapter and TransformerLens implementation "not independently verified in the source bundle inspected for this revision" | Captum adapter present (21 tests pass, Captum 0.9.0); TransformerLens 1.19.0 interchange in `td_bench.py` (1160/1160 recomputed) | state as verified |
| M22 | §VI-A | per-sample records, frozen plans, implementation and scripts are absent; "counts … not independently reproduced" | all present; counts recalculated; the MLP site was reproduced by a full re-run | rewrite the setup paragraph; cite the repository and frozen commits |
| M23 | §VI-A d | "raw timing records are absent" | `performance75.json`, `fd9_performance.csv`, `bench_audit_*.json` | cite, with the single-run caveat |
| M24 | §VI-B a | per-configuration records and analysis script "absent from this revision checkout" | scripts present. Per-sample role × outcome counts present. Per-configuration axis values are *not* stored for the central sites. | keep the conclusion (no fixed-selection decomposition yet); correct the reason |
| M25 | §VI-H a; §VII d | package, tests, evidence, protocols and scripts not included; release claims unverified | all present; 1080 tests pass; v0.1.0 on PyPI | rewrite. The true residual limitation is that audit traces and reports are not published |
| M26 | §VI-H c | the cross-hardware tie "could not be inspected" | documented in `docs/REPRODUCIBILITY.md` and the Phase 8 report | cite |
| M27 | §VII c | raw records of the two-fine-tune OOD comparison are absent | `fd8_token_ood.csv`, `nlp_probes_{D,E}.json` | cite |

## 3. Draft and revision-process language (Prof. Kanavos asked for its removal)

| location | phrase |
|---|---|
| §II-F | "the draft describes …", "source bundle inspected for this revision", "see the reproducibility note in Supplementary Appendix D" |
| Fig 2 caption | "Values are inherited summaries; … absent" |
| §IV-G | "The manuscript's Wilson intervals …" (self-reference; say "The Wilson intervals reported here …") |
| §VI-A | "The main artifact contains …"; "the missing files are inventoried in Supplementary Appendix D" |
| §VI-A d | "the raw timing records are absent"; "The missing records prevent checking …" |
| §VI-B a | "absent from this revision checkout" |
| §VI-H a | "The supplied aggregate table reports …"; "could not be verified in this revision" |
| §VI-H c | "The draft reports …"; "from the current source bundle"; "in this revision" |
| §VII c | "its raw records are absent" |
| §VII d | "This checkout lacks the package, tests, …" |

## 4. Omissions

1. **Code availability.** The manuscript never gives the repository URL (`https://github.com/NikolasRoufas/beyondnn`) or the frozen commits. Prof. Kanavos asked for the URL to be included.
2. **Re-audit of seen samples.** The 47/52 is reported without stating that the same 52 samples gave 49/52 under the Phase 7.5 audit, and that ADR-053/054 were adopted after that outcome was seen (R03). §VI-A d acknowledges only in general terms that earlier results informed later revisions.
3. **AI-assistance disclosure.** There is no acknowledgments section. IEEE policy requires disclosure of AI-generated content; check the exact current wording before Stage 5.
4. **Related work.** Two directly relevant 2026 papers are not cited:
   - Certified Circuits ([arXiv:2602.22968](https://arxiv.org/abs/2602.22968));
   - Lin & Liu's disclosure norm for identification assumptions ([arXiv:2605.08012](https://arxiv.org/abs/2605.08012)), which is cited in the repository's own literature notes.

   Already cited: Joshi et al., Lan et al., Méloux et al., CIF.
5. **The standing/finding distinction** is stated correctly (§IV-G, Fig 3 caption: "The primary standing remains SUPPORTED"). Keep this wording everywhere 47/52 appears.

## 5. References checked

- [35] Lan et al., ACL 2026, pp. 3516–3540: verified ([ACL Anthology 2026.acl-long.159](https://aclanthology.org/2026.acl-long.159/)).
- [31] Joshi et al. and [39] Méloux et al.: ICML 2026 acceptance confirmed ([poster 67225](https://icml.cc/virtual/2026/poster/67225), [poster 63205](https://icml.cc/virtual/2026/poster/63205)). The PMLR volume 306 page numbers could not be confirmed (M28).
- [5] CIF, UAI 2026, PMLR 337: a proceedings page exists ([asiaee26d](https://proceedings.mlr.press/v337/asiaee26d.html)); pages not checked.
- [18] M-FAC is cited for the BERT-tiny checkpoint. This is justified: the model is `M-FAC/bert-tiny-finetuned-sst2` (`experiments/phase5_5/common.py`).
- A full reference audit (every entry against its primary source) belongs in Stage 5.

## 6. What the manuscript already does well

The manuscript already adopts most of the scoping the audit would require:
- standing versus finding;
- k changes the subject;
- an at-least-one count, not a rate;
- structural labels are not per-input necessity;
- InterpBench is consistency only;
- unclustered intervals;
- one checkpoint per model;
- protocols not uniformly blind;
- no "first" claims.

The Stage 1 weaknesses that remain are therefore mostly *evidence gaps* (W1–W3, W5, W6, W8), not wording problems. The exceptions are the contradictions in §2 and the language in §3.
