# Method diagnostics: `method_agreement`, `baseline_sensitivity`, `ig_step_sensitivity` (v1)

These are diagnostics of attribution *methods*. They are not causal tests and not faithfulness evidence for a claim. Each needs its attributions to have the same sample, site, call, and target, or it is refused.

| Diagnostic | Quantity | Interpretation | Cannot support |
|---|---|---|---|
| `method_agreement` | Spearman ρ of the ordinal rankings; top-k Jaccard | whether two methods rank the same units alike | correctness: two methods can agree and both miss the causal structure; agreement is not faithfulness |
| `baseline_sensitivity` | as above, for IG under two declared baselines (all else equal) | how much the IG ranking depends on the baseline choice (scenario N: ρ = −1 on `x0·x1`) | which baseline is right |
| `ig_step_sensitivity` | max \|IG_n − IG_m\| and both completeness deltas | whether the numerical integration has converged | anything about faithfulness; a converged IG can still be misleading |

- **Outcomes:** PASS/FAIL only against declared criteria (`min_rank_correlation`, `min_topk_jaccard`, `max_abs_difference`); INDETERMINATE otherwise.
- **Output:** `ProtocolResult` in an anchor trace (one clean pass for provenance). The measurements carry both attribution record ids; composition re-derives every number from the attribution tensors.
- **References:** Sundararajan et al. 2017 (IG, completeness); Sturmfels et al. 2020 (baselines); Adebayo et al. 2018.
