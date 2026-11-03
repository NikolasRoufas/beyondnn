# TNNLS suitability assessment (Stage 5)

- **Sources:** the TNNLS Information for Authors on ScholarOne. The PDF metadata is dated 2012, so the current rules must be re-checked on the journal site before submission. IEEE Author Guidelines for AI-generated text, open.ieee.org, read 2026-11-03.

## Formal requirements

| requirement | source | revised manuscript |
|---|---|---|
| full paper: novel contribution of archival nature in developing theories and/or innovative applications of neural networks and learning systems; "should not be of incremental nature" | TNNLS IFA | see "Scope fit" below |
| ≤ 13 formatted pages, IEEE two-column (overlength charge beyond 13) | TNNLS IFA | **checked:** page count reported in the final report (reconstructed IEEEtran build) |
| single-spaced, double-column, standard IEEE format, numbered pages; PDF | TNNLS IFA | IEEEtran `journal` class |
| informative abstract including the important results | TNNLS IFA | abstract states the main numbers and the limits |
| figures legible at column width; text must not refer to colour | TNNLS IFA | vector PDF at 3.45 in. Bar categories are distinguished by grey level and hatching as well as colour, and captions do not refer to colour |
| references in IEEE style | TNNLS IFA | IEEEtran BibTeX style, numbered by first citation |
| AI-generated content disclosed in the acknowledgments, naming the system, the sections and the level of use; AI cannot be an author | IEEE guidelines | an acknowledgment draft is included and **marked for author review**; no AI author |

## Scope fit

**For TNNLS:**
- the subject is claims about the computation of trained neural networks (MLP, CNN, transformers);
- the contribution includes a formal semantics with properties, and an evaluation across architectures and independently trained checkpoints;
- the paper engages directly with the mechanistic-interpretability and evaluation literature that TNNLS publishes.

**Against TNNLS:**
- there is no new learning algorithm, architecture or theoretical result about learning;
- the core contribution is an evaluation and audit methodology implemented as software;
- the propositions are "by construction" properties of a rule system, not deep theorems;
- the strongest new empirical result (RQ3) is partly negative: a careful conventional checker matches the audit's detections;
- an editor may judge the paper as tooling and suggest a venue focused on explainable AI.

**Assessment:**
- **Fit is plausible but at risk.** The revision strengthens the TNNLS case (formal framework, trained-network reference labels, checkpoint replication, cost), but it cannot change the paper's type.
- **TAI** (IEEE Transactions on Artificial Intelligence) explicitly lists explainable AI in its scope and is a lower-risk fit for the same content.
- The decision belongs to the authors.
- **Recommendation:** keep TNNLS as the first target only if the authors accept a substantial desk-rejection risk. Otherwise retarget to TAI without changing the science.
