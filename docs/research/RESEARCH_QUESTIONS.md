# Research Questions

Each question lists what would count as an answer, so the questions can actually be closed.

1. **Can important internal states be identified reproducibly?**
   Answer: rank overlap (top-k Jaccard / Kendall τ) of important sites across seeds, data subsamples, and methods, on ground-truth models.
2. **Do interventions on reported features reliably affect outputs?**
   Answer: the effect of reported-important sites vs random sites of matched size (distribution plus test), across ablation types (zero / mean / resample).
3. **Are causal features distinguishable from merely correlated ones?**
   Answer: classification accuracy of A/B/C on the synthetic suite, for each method: attribution alone, probing alone, intervention.
   Expected result: probing fails on B, and intervention succeeds. The expectation must be *tested*, not assumed.
4. **Do interpretable representations reduce capability?**
   Answer: accuracy / loss vs sparsity or bottleneck width, at matched parameters (Mode B, later).
5. **How stable are explanations across equivalent inputs?**
   Answer: stability metrics under user-declared invariances. The definition of "equivalent" is always user-supplied and recorded.
6. **Can concept labels be validated rather than merely generated?**
   Answer: the fraction of auto-proposed labels that pass `detection+causal/v1`, and agreement with human judgement on a sample.
7. **Can one trace representation support multiple architectures?**
   Answer: the same test suite passes on MLP / CNN / transformer without architecture-specific branches in `core`.
8. **Is any aggregate "explanation confidence" predictive?**
   Answer: does an aggregate of component scores predict held-out intervention outcomes, or ground-truth correctness on synthetic models, better than its best single component? If not, it is never shipped.
9. **How much does the ablation baseline change conclusions?**
   Answer: the rank correlation of site importance under zero vs mean vs resample ablation.
