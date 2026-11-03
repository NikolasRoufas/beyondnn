# Scientific weaknesses (TNNLS revision, Stage 1)

- **Stance:** a skeptical TNNLS reviewer reading the evidence, not the prose.
- **Inputs:** the repository at `c9e43ad`, `MANUSCRIPT_RESULT_VERIFICATION.csv` (R-numbers) and `recalculation.json`.
- **Manuscript:** not available. Weaknesses that concern the manuscript text are marked *(manuscript unchecked)*.
- **Severity:**
  - **CRITICAL:** likely to sink a TNNLS submission on its own.
  - **MAJOR:** a reviewer will require it to be fixed.
  - **MINOR:** should be fixed.

## Four layers kept separate

The plan asks that these never be conflated.

| layer | what the repository shows |
|---|---|
| software correctness | strong: 1080 tests, mutation logs, strict typing, CI |
| evidence admissibility | shown by construction: the type rules refuse attribution → causal, decodable → used, generated → validated |
| causal validity of individual tests | depends on the replacement. Shown only where truth is independent (compiled Tracr TD). On InterpBench, truth shares the PRIMARY semantics |
| mechanism identification | **not shown for any trained network** |

## CRITICAL

### W1. The added value of the audit layer over simply running the PRIMARY intervention is not demonstrated

**Evidence:**
- On TD, the audit's PRIMARY outcome agrees with an independent TransformerLens interchange on 1160/1160 decisive instances (R04/R05).
- Hence 0/400 decoys and 591/760 used components are exactly what a researcher would get by running resample patching alone.
- On InterpBench, PRIMARY = the benchmark's own semantics (R12).
- The TD and InterpBench results therefore validate the *intervention choice*. They do not validate the claim-audit abstraction.

**Why it matters:** the paper's contribution is the audit (typed claims, admissibility, provenance, roles, findings). TNNLS reviewers will ask what error a conventional, careful workflow (Captum + patching + logging) would make that BeyondNN prevents. That question has no empirical answer yet.

**What is needed:** a fair comparative study (Stage 3, E3.2) on the same models, samples and interventions. It should count errors that are detected, missed or falsely flagged by each workflow, including a strong baseline that a careful researcher would actually write.

### W2. The headline 47/52 combines subject-changing and fixed-subject sensitivity, and depends on author-declared alternatives

**Evidence:**
- ALTERNATIVE k = 5%/20% changes *which units* are claimed, so it changes the claim's subject. Only replacement and null are fixed-subject alternatives (`central75.py` roles; plan §6).
- Each supported sample was compared against 17 alternative configurations (one against 11).
- A sample counts as "reversed" if **any** of them contradicts it.
- Median 6 of 17 contradict; only 1 of the 47 is reversed by exactly one alternative (`alternative_multiplicity`).
- The per-sample `reversal_axes` field merges ALTERNATIVE and STRESS_TEST findings, so the axes behind the 47 cannot be recovered from the artifacts (R01).
- 12/47 have only k-involving recorded explanations.
- All 47 keep the audit standing SUPPORTED. "Reversed" means the audit attached a QUALIFYING `alternative_reverses` finding, not that the standing changed. The manuscript wording must reflect this.

**Why it matters:**
- "Conclusions change under reasonable alternatives" is only as strong as the alternatives are reasonable.
- With 17 alternatives and an "any" rule, some reversal is expected whenever a result is near its decision boundary.
- There is no reference rate: how often does the same rule reverse when there is no real sensitivity?
- The ledger already concedes that "a different declared set could reverse fewer".

**What is needed** (Stage 2):
- a fixed-selection decomposition of replacement, null and threshold effects, with k reported separately as a change of subject;
- the share of alternatives that reverse, not just "any";
- a reference distribution, for example the reversal rate of R-selections or of synthetic results with known zero sensitivity.

### W3. Independent mechanism truth exists only for tiny compiled programs, where it is close to trivial for the PRIMARY intervention

**Evidence:**
- TD: 8 compiled Tracr programs, 1–2 layers, at most 3 heads, one-hot residual stream (R04–R09).
- Unused components have *exactly* zero functional influence, so resample patching contradicts them by construction.
- Zero ablation supports all 400 decoys because it pushes a one-hot residual off-distribution (R07).
- InterpBench is circular (R12).
- No trained network has independent reference truth.

**Why it matters:** TNNLS is a neural-networks-and-learning journal. Validation that never touches a *trained* model with independently known structure is the most likely target for "insufficient validation".

**What is needed:** a trained model with planted or independently specified mechanisms (Stage 3, E3.3), evaluated with held-out examples, with truth not defined by the audited intervention.

### W4. Novelty is narrow, and recent work is closer than the repository's literature review records

The repository's literature notes (September 2026) do not cite three directly relevant works:

- **Lan et al., "Make Mechanistic Interpretability Auditable"** ([arXiv:2606.00033](https://arxiv.org/abs/2606.00033), ACL 2026). Proposes "source-based auditing systems that track arguments which claims depend on", plus guidelines. A position paper, not an implementation.
- **Anani et al., "Certified Circuits"** ([arXiv:2602.22968](https://arxiv.org/abs/2602.22968)). Stability certificates for circuit-inclusion decisions under dataset perturbation.
- **Méloux et al., "Mechanistic Interpretability as Statistical Estimation"** ([arXiv:2510.00845](https://arxiv.org/abs/2510.00845)). Structural variance of EAP-IG; recommends routine stability reporting.

Already cited and overlapping:

- **Joshi et al. 2026** ([arXiv:2602.16698](https://arxiv.org/abs/2602.16698), ICML 2026 position). The claim/evidence rung framework.
- **Lin & Liu 2026** ([arXiv:2605.08012](https://arxiv.org/abs/2605.08012)). A disclosure norm: name the identification strategy, stress-test at least one assumption, report how conclusions change. This is close to BeyondNN's ALTERNATIVE / STRESS_TEST roles.
- **Asiaee, CIF** (UAI 2026, [arXiv:2607.08349](https://arxiv.org/abs/2607.08349)). Anytime-valid statistical certification of interventional claims.

**What remains distinctive** (as implemented, after checking current Captum 0.9.0, Quantus 0.6.0, nnsight 0.8.2, pyvene 0.1.8 and TransformerLens 4.3.0 releases):
- typed claim records with re-derived standing;
- provenance-bound refusal to compose evidence across checkpoints, samples or scopes;
- a declared-role audit producing typed findings without a global score;
- re-derivation of saved results.

**Risk:**
- "First" or "no prior work" claims would be false or unverifiable.
- The contribution reads as an *operationalisation* of others' position papers, which a reviewer may judge as engineering.

**What is needed:**
- a current, source-verified capability matrix (Stage 3, E3.1);
- framing as "the first implemented …" only where the matrix supports it, ideally avoiding "first" altogether;
- formal properties (W6) that are not just restatements of the position papers.

### W5. TNNLS scope fit is uncertain

**Evidence:**
- The method makes no learning-algorithm or architecture contribution.
- It is an evaluation and auditing layer for interpretability evidence.
- TAI explicitly lists explainable AI in its scope (per Prof. Kanavos's own check).

**Why it matters:** editorial desk rejection is a real risk for a tool/framework paper at TNNLS unless the contribution is framed as methodology for *evaluating claims about learned neural representations*. That framing needs formal properties and trained-network evidence.

**What is needed:**
- Re-assess after Stage 3. This is a decision for the authors, not something an experiment fixes.
- Verify the current official TNNLS scope text before Stage 5.

## MAJOR

### W6. No formal semantics or stated properties

**Evidence:**
- Audit rules exist as code and ADRs (precedence in `beyondnn/audits/engine.py::_standing`; disagreement explanation in `_disagreement`).
- No definitions, propositions or proofs are stated.
- Some properties hold by construction and could be stated and proved. Two examples:
  1. Under the role-based design (ADR-048, Phase 7.5+), the standing is decided by PRIMARY configurations, and ALTERNATIVE / STRESS_TEST configurations only add typed findings. All 47 "reversed" samples keep the standing SUPPORTED, with a QUALIFYING `alternative_reverses` finding. This invariance must be stated and checked against `engine.py`, which has precedence subtleties.
  2. A causal relation never reaches SUPPORTED from evidence whose status is below INTERVENTIONAL.

**Why it matters:** TNNLS expects formal clarity. Without it, the work reads as software documentation.

**What is needed:** a formal specification of claim, evidence, admissibility, roles and standing, with a small set of proved properties, each linked to a test (Stage 5; groundwork in E2.0).

### W7. Held-out samples were re-audited after their outcomes were seen

**Evidence:**
- The 7.75 central numbers (47/52, D1 11/40, D2 6/40) re-audit the Phase 7.5 held-out samples.
- They use rules (ADR-053/054) designed after the 7.5 outcomes were known (the plan states the D [SEP] result was known).
- The value changed from 49/52 to 47/52 (R03).

**Why it matters:** these are no longer confirmatory held-out estimates.

**What is needed:**
- Label them as re-audits in the paper.
- For any confirmatory claim, use fresh held-out samples. The pools are large enough for A, B and C; for D this is costly.

### W8. Statistical dependence is not modelled

**Evidence:**
- Pooled counts (47/52, 591/760, 0/400, 1571/1571) treat instances as independent.
- In fact samples are nested in sites, sites in checkpoints (A and B have two sites each), and instances in components (40 per component in TD).
- Wilson intervals over instances overstate precision.
- There is no between-checkpoint variance: one checkpoint per model (R34).

**What is needed:**
- report per-cluster results and cluster-aware intervals (cluster bootstrap over programs/components; per-site reporting);
- at least one genuine checkpoint replication (Stage 4, E4.1).

### W9. Estimand mismatch in TD "false negatives"

**Evidence:**
- 169/760 known-used instances are CONTRADICTED.
- In every one, the TransformerLens interchange also did not flip (R05): a single resample counterfactual left the used value unchanged.
- "Used by the program" is not the same as "necessary for this sample under this counterfactual".

**Why it matters:**
- Calling these false negatives understates the method, while 77.8% sensitivity invites a "low power" objection.
- The correct framing is a precise estimand statement.
- The fix is a multi-counterfactual necessity estimand, which the repository already lists as a desirable future method.

### W10. The comparison of computational overhead is missing

**Evidence:**
- `fd9_performance.csv` reports single runs of the BeyondNN workflow on one Apple machine.
- For example, D took 2313 s of evidence generation versus 12.5 s of audit build for 2 samples (R32).
- There is no equivalent workflow *without* the audit layer and no repeated timings.

**What is needed:**
- A matched with/without comparison and repeated timings (Stage 4, E4.2).
- The existing numbers already suggest the audit is cheap relative to evidence generation, but that is not yet a measured comparison.

### W11. The research-scale deterministic re-audit is not reproducible from public artifacts

**Evidence:**
- `verify_report: passed` is recorded, but the traces and reports are not published (R30).
- Not repeatable from the repository (REPRODUCIBILITY_AUDIT §3).

**What is needed:** a published evidence bundle (P0 in the plan).

### W12. Scale

**Evidence:**
- The largest model is BERT-base (110M), run on CPU with 40–60 samples per site.
- There is no modern decoder-only LM, no vision transformer, and no GPU execution path validated ("GPU and other devices are not verified", `docs/REPRODUCIBILITY.md`).

**Note:** scaling up is useful only if it tests a claim (Stage 4). Size for its own sake is NOT JUSTIFIED (plan §9).

## MINOR

- **W13. Overstated decoy description (R08b).** The 0/160 decoys include 80 copies of the token stream, not of the used variable.
- **W14. Concept validation evidence is tiny** (4 positives, 8 negatives, compiled programs). Its programs do not discriminate the relative and absolute rules (R18). The Phase 7.5 known positive was rejected (R19). It should stay a qualified secondary claim.
- **W15. The NLP case study is thin for its claims.**
  - One annotator; n = 40.
  - N1 (IG ≥ human necessity) failed.
  - There is a hypothesis-only shortcut (0.456 empty-premise accuracy, R26).
  - It is acceptable as a qualified case study only.
- **W16. The design error behind ADR-054 was found post hoc.** The count-matched null criterion was unattainable, and this was discovered only after it rejected 98.6% of necessary heads. Reviewers may ask what other criteria have not been checked for attainability. A systematic attainability check of every control criterion would answer this (cheap; artifact-only).
- **W17. All scenarios, ground-truth programs and alternatives were written by the authors.** This is unavoidable but must be stated. An external known-mechanism benchmark helps (W3).
- **W18. AI-assistance disclosure** *(manuscript unchecked)*.
  - Substantial AI assistance was used in developing the software and documents. The first three commits carry an AI co-author trailer, and this revision is AI-assisted.
  - IEEE policy requires disclosure of AI-generated content in the acknowledgments. Verify the exact current text before Stage 5.
- **W19. Manuscript consistency unchecked** *(manuscript unavailable)*. The plan reports that the PDF states files were unavailable or unverifiable, and that it contains draft/revision-process language. Every script and result file behind R01–R32 is present in the public repository, so such statements are probably wrong. This needs confirming against the actual text.

## Most important, in one line each

- **Reproducibility issue:** the two headline versions (49/52 vs 47/52) and the absence of published traces, which make the research-scale re-audit unrepeatable.
- **Scientific weakness:** the audit's value over a careful conventional workflow is undemonstrated (W1), and its independent validation never touches a trained network (W3).
- **Novelty risk:** recent position papers (Joshi 2026; Lin & Liu 2026; Lan et al. 2026) already state the principles. BeyondNN's distinct claim is the implemented, provenance-bound audit, and it must be argued as such, without "first" claims.
