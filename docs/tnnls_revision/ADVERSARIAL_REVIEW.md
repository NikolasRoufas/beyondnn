# Adversarial TNNLS review (Stage 6)

- **Reviewed:** the **revised** manuscript `paper_tnnls/main.pdf` (reconstructed IEEEtran source), not the original.
- **Stance:** an independent, skeptical reviewer.
- **Rounds:** Round 1 found the issues listed in §1. Round 2 (§2) re-reviews after corrections. The final recommendation is in §3.

**Severity:**
- **CRITICAL:** likely to cause rejection.
- **MAJOR:** must be addressed.
- **MINOR:** should be fixed.

## 1. Round 1

### R1-C1 (CRITICAL): TNNLS scope and type of contribution

- **Location:** §I, contributions; §III-F; §IV.
- **Problem:**
  - There is no learning method, architecture or learning-theoretic result.
  - The contribution is an audit semantics plus software.
  - The six propositions hold "by construction"; they restate rules rather than prove non-obvious properties.
- **Why vulnerable:** TNNLS asks for "novel contributions of archival nature in developing theories and/or innovative applications of neural networks and learning systems", not of incremental nature. An editor can desk-reject a tooling paper.
- **Evidence required:** none that experiments can supply. This is a framing and venue decision.
- **Correction:**
  - State the propositions as *properties of the semantics* (done).
  - Show that they are checked: tests, the τ × 0.5 consistency check, and role invariance on 47 samples (done).
  - Make the neural-network specificity explicit: unit eligibility, sites, replacements in activation space, checkpoints (done in §III).
  - **The venue decision remains with the authors** (TNNLS_SUITABILITY.md).
- **Status after revision:** **unresolved risk** (venue).

### R1-C2 (CRITICAL → MAJOR after revision): added value over a careful conventional workflow

- **Location:** §VI-C, Table IV.
- **Problem:**
  - The comparison shows *no* detection advantage over a careful checker.
  - The audit's advantage is that its checks are built in and reusable, but its plan has *more* lines of code than the checker.
  - A reviewer will ask why a community should adopt a new representation for checks it can write itself.
- **Why vulnerable:** the paper's central motivation ("conclusions move", "make it checkable") needs evidence that the representation changes outcomes in practice.
- **Evidence required:** real-world usage, either (a) a re-audit of *published* interpretability claims with their released artifacts, counting the claims whose stated conclusion the declared evidence does not support, or (b) a user study.
- **Correction made:**
  - The paper now reports the negative result plainly.
  - It identifies the two concrete semantic differences: attainability, and the refusal of tampered evidence.
  - It states the burden honestly.
  - No new experiment can be run in this revision without external artifacts or participants.
- **Status:** **MAJOR, partially addressed;** remains the strongest reviewer objection.

### R1-M1 (MAJOR): the 47/52 headline was a re-audit of already-seen samples

- **Location:** abstract; §VI-A; §V.
- **Problem:** the samples had been audited before the current rules were adopted (49/52), so the headline is not a fresh held-out estimate.
- **Correction:**
  - Disclosed in §V and §VII.
  - **New confirmatory run on fresh, never-inspected samples** (protocol P3; CNN and BERT-tiny) with pre-stated predictions F-i–F-v. Results in §2.
- **Status:** see §2.

### R1-M2 (MAJOR): no independent mechanism truth for trained central models

- **Location:** §VI-D; §VII.
- **Problem:**
  - The trained-network reference labels come from SIIT-trained InterpBench models: post-hoc and interchange-trained.
  - The MLP, CNN and BERT models have no reference mechanism.
- **Correction:** stated explicitly, and the independence argument is scoped precisely (label independent of the audited *evaluation*, not of the *intervention family* used in training).
- **Status:** **unresolved,** honestly bounded.

### R1-M3 (MAJOR): fault set and checker written by the authors

- **Location:** §VI-C.
- **Problem:** detection rates over a designed fault set are not error rates. The checker was written knowing the faults, which favours the checker, not the audit.
- **Correction:** stated in §VI-C and §VII. A blind fault set from a third party would strengthen it.
- **Status:** **disclosed, unresolved.**

### R1-M4 (MAJOR): statistical treatment

- **Location:** §V; all tables.
- **Problem:**
  - Intervals are unclustered.
  - Pooled counts (41/52, 0/400, 591/760) ignore nesting.
  - There is no inference beyond descriptive counts.
- **Correction:**
  - Per-site and per-checkpoint reporting.
  - Between-checkpoint ranges (Table V).
  - The text never presents intervals as population statements.
- **Status:** **acceptable for a descriptive audit paper;** a reviewer may still ask for cluster-robust intervals.

### R1-M5 (MAJOR): cost of the evidence layer

- **Location:** §VI-G; Table VI.
- **Problem:**
  - About 23× per-test overhead on a 30-feature MLP.
  - Audit + verify account for about 37% of the total on small models.
  - 33–39 MB of evidence per MLP sample.
- **Correction:** all reported, without minimising them.
- **Status:** **addressed (disclosed).**

### R1-m1 (MINOR): AI-disclosure placeholder

- **Location:** Acknowledgment.
- **Problem:** the bracketed note "for author review" must not be submitted.
- **Status:** **author action required.**

### R1-m2 (MINOR): two references with unconfirmed proceedings details

- **Location:** [Joshi 2026], [Méloux 2026].
- **Problem:** ICML 2026 acceptance is confirmed, but PMLR volume and pages are not. The bib now omits the unconfirmed volume and pages.
- **Status:** **author check** before submission.

### R1-m3 (MINOR): Table I judgments

- **Location:** Table I.
- **Problem:** cells such as "partial" need sources.
- **Correction:** BASELINE_CAPABILITY_MATRIX.md lists the sources. A supplementary version can be included.
- **Status:** **addressed in the repository.**

### R1-m4 (MINOR): the BERT-base decomposition depended on a long-running job

- **Location:** Table III.
- **Problem:** at the time of round 1 the BERT-base re-run was incomplete, so totals covered 41 of 52 supports.
- **Status:** see §2.

## 2. Round 2 (after corrections and the final runs)

### Resolved or reduced since round 1

| issue | action | status |
|---|---|---|
| R1-M1: re-audit of seen samples | Protocol P3, run on never-inspected samples (CNN 60, BERT-tiny 40). All five pre-stated predictions held: CNN pixels 16 of 19 reversals with the units fixed; BERT-tiny 5 of 6 only through k; τ × 0.5 reverses none. Reported in §VI-A and Table III (bottom block). | **reduced to MINOR**: BERT-base and MLP have no fresh-sample confirmation (cost; exhausted pool) |
| R1-m4: BERT-base decomposition incomplete | Completed. BERT-base reproduces exactly (0 per-sample differences); 4 of 11 reversals with the units fixed, 7 only through k. Totals: 23 fixed, 24 k-only, 47 reversed of 52. | **resolved** |
| new R2-1: "every sample tested under 45 configurations" | One sample had 11 alternatives, because two k values coincide. Text now says "up to 45 … fewer when two values of k coincide". | **resolved** |
| new R2-2: "no reversal required a combination of axes" overstated the scope | Now restricted to fixed-selection reversals, which is what the data support (0 multi-axis-only fixed reversals). | **resolved** |
| new R2-3: wording audit | Searched for "first", "novel", "prove", "guarantee", "significant", "outperform", "superior". Every remaining "never" or "always" is a rule-level statement that holds by construction. No unsupported superlatives. | **resolved** |
| new R2-4: abstract length | Reduced to about 228 words (IEEE: 150–250). | **resolved** |
| new R2-5: fonts | XeTeX lacked IEEE Times; the build now uses T1 and newtx (TeX Gyre Termes), all fonts embedded. | **resolved** |

### Still open after round 2

| severity | issue | why it remains | what would close it |
|---|---|---|---|
| **CRITICAL** | R1-C1: TNNLS scope; a methodology/tooling paper with by-construction properties | Not an evidence gap; a venue judgement | Author decision; TAI is the lower-risk venue |
| **MAJOR** | R1-C2: added value over a careful conventional workflow | The comparison is honest but partly negative: equal detection, more lines of code; differences only in defaults, attainability and tamper refusal | Re-audit of published claims with released artifacts, or a user study |
| **MAJOR** | R1-M2: no independent mechanism truth for the trained central models | SIIT specification labels are post hoc and interchange-trained; Tracr is tiny and compiled | A non-trivial trained model with a construction-fixed mechanism |
| **MAJOR** | R1-M3: author-designed faults and checker | Inherent to this design | A third-party blind fault set |
| MINOR | R1-M4: unclustered statistics | Descriptive design; ranges reported | Cluster bootstrap |
| MINOR | R1-M5: overhead | Disclosed: 23× per test on a tiny MLP, 0.1% of total on BERT-base | Leaner record retention |
| MINOR | R1-m1: AI-disclosure placeholder; R1-m2: two references | Author action | Author review |
| MINOR | Reconstructed source | The original LaTeX was unavailable | Authors merge the revision into their source |

### Reviewer's summary judgement

- **Rigour:** the revised paper is scientifically careful and internally consistent. Every number traces to an artifact, the main headline is now decomposed and confirmed on fresh samples, negative results are kept, and the claims are scoped.
- **The remaining weaknesses are not editorial:**
  - the incremental value over a careful practitioner is not demonstrated;
  - independent truth on trained central models is absent;
  - the contribution type is a poor match for TNNLS's emphasis on learning methods and theory.

## 3. Final recommendation

**REQUIRES SUBSTANTIAL ADDITIONAL RESEARCH** (for TNNLS).

**Required before a TNNLS submission:**
1. Evidence that the audit changes conclusions in real practice: a re-audit of published interpretability claims from their released artifacts, or a user study.
2. Independent mechanism validation on at least one non-compiled trained network whose reference is not interchange-defined, or an explicit decision to scope the paper without it.
3. A venue decision.

**If the authors retarget to IEEE Transactions on Artificial Intelligence**, whose scope explicitly includes explainable AI, the same manuscript would need only targeted revision:
- the author items (AI disclosure, two references, merging into the original source);
- the published evidence bundle.

No acceptance probability is given.
