# BeyondNN TNNLS research revision: executive summary

- **Date:** 2026-11-03.
- **Branch:** `tnnls-revision` (local; not pushed).
- **Plan:** Prof. Kanavos's six-stage plan, carried out end to end.
- **Final adversarial recommendation:** **REQUIRES SUBSTANTIAL ADDITIONAL RESEARCH** (for TNNLS). For IEEE TAI, only targeted revision would be needed.

## What was done

| stage | status | main outputs |
|---|---|---|
| 1 Evidence audit | **completed** | 66 claims checked: 42 verified from artifacts, 12 summary-only, 2 unverified, 10 contradicted (8 false "files absent" statements, the "0/160 exact-copy" wording, the intervention-operation list). The MLP experiment was re-run exactly. |
| 2 Configuration sensitivity | **completed** | Exact re-execution of the frozen design for all four models (0 per-sample differences). Fixed-selection decomposition of 47/52. Confirmation on never-inspected samples. |
| 3 Comparison and independent validation | **completed** (with negative results) | Source-verified capability matrix; fault-injection comparison on identical measurements; InterpBench re-analysed against specification labels |
| 4 Robustness, cost, reproducibility | **completed** for MLP/CNN; BERT models single-checkpoint | 5 seeds per trained model; per-phase cost; per-test overhead; save/reload/re-audit; scaling benchmark |
| 5 Manuscript | **completed** (reconstructed source) | `paper_tnnls/main.pdf`: 11 pages, IEEEtran; 6 figures, 6 tables; all empirical tables and figures generated from result JSON |
| 6 Adversarial review | **completed** (two rounds) | `ADVERSARIAL_REVIEW.md` |

## Strongest findings

1. **The 47/52 headline splits in two.** In an exact re-execution:
   - **23** of the 47 reversals keep the selected units fixed: a declared alternative *replacement* (19) or *control rule* (12) reverses the conclusion about the same units.
   - **24** arise only from changing the selection size k, i.e. from changing the claim's subject.
   - The halved-threshold alternative can never reverse a supported claim, and it reversed none.
   - The split is model-dependent. On CNN pixels the reversals are mostly fixed-selection (10/13); on the token models they come mostly from changes of k (BERT-tiny 10/13, BERT-base 7/11).
   - On never-inspected samples, all five pre-stated predictions hold.
2. **Exact reproducibility.**
   - Re-running the frozen design from source on Linux reproduces every per-sample audit output of all four models.
   - Saved evidence reloads and re-audits identically.
   - Every tampered trace is refused.
   - Retraining, by contrast, is *not* bit-reproducible across platforms: different checkpoint digests on Linux and macOS.
3. **Trained-network reference labels.** On 18 SIIT-trained InterpBench transformers, scored against specification labels that use no intervention on the trained network:
   - the primary test supports 942/942 specification-necessary instances and 18/8,858 specification-unnecessary ones;
   - all 18 are nodes the trained network demonstrably uses.

   The trained network deviates from its specification, and the audit reports the network.
4. **Checkpoint robustness.** Across five independent checkpoints per trained model:
   - IG support exceeds random support at every site;
   - most supports carry an alternative reversal (75–100%);
   - CNN-pixel fixed-selection reversal shares are 60–83%.

   Absolute counts vary (CNN pixels 12–22 of 60), so a single-checkpoint count is not a model-class rate.

## Most important negative results

- **On identical measurements with seven injected faults, a careful conventional checker detected everything the audit detected**, provided its author wrote a check for each fault (54 lines; the audit's plan is 98 lines). The audit's advantages are:
  - checks by default, reusable across studies;
  - correct handling of unattainable-control tests, where the hand-written rule mis-scored one;
  - refusal of tampered evidence.

  It is not superior detection.
- **The evidence layer costs about 23× a plain PyTorch implementation per test** on a 30-feature MLP. Audit + verify are about a third of total time on small models, and 0.1% on BERT-base. Evidence takes 33–39 MB per MLP sample.
- **One baseline implementation error was found and fixed:** the first version ranked IG scores by signed value. The failed run is kept on record.

## Biggest remaining weaknesses

1. Added value over careful practice is undemonstrated in real use; this needs a re-audit of published claims or a user study.
2. There is no independent mechanism truth for the trained central models.
3. TNNLS scope fit: a methodology and tooling contribution whose properties hold by construction.

## Compute

- **Server:** `interstellar`.
- **CPU:** 22.7 timed CPU-hours (one thread per process; at most 40 concurrent).
- **GPU:** 0 GPU-hours.
- **Storage:** 6.5 GB on `/data`.
- **Disk state:** the root filesystem remained 100% full throughout; nothing was written there.

## Decisions for Prof. Kanavos

1. TNNLS versus TAI.
2. Whether to fund the real-world re-audit study or a user study (the main TNNLS gap).
3. Review of the AI-assistance disclosure, and of the reconstructed LaTeX in place of the missing original source.
