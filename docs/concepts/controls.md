# Controls (plan §11, §14; request §16–§17)

**A concept may not become VALIDATED_CONCEPT without a declared control criterion.** Specs refuse to be built without one, and the criterion is part of the decision, never merely reported.

| Control | Test | Null ("the feature is no better than …") | Matching |
|---|---|---|---|
| `random_directions(n, seed=, distribution="isotropic")` | encoding, use | … a random unit direction at the same site | same site, axis and pooling; unit norm; sign-free scoring |
| `random_directions(n, seed=, distribution="covariance")` | encoding, use | … a random direction drawn from the train-split activation covariance (inside the data subspace) | as above; Σ^½ is the symmetric root, so it is basis-independent |
| `random_neurons(n, seed=)` | encoding (neurons), use (neurons) | … another unit of the same axis | same site and axis; sign-free scoring |
| `label_permutation(n, seed=)` | encoding | … the same feature against permuted held-out labels | same feature, same readout sign |

- **Seeds:** local generators only; the global RNG is never used.
- **What is recorded:** seed, n, distribution, normalisation and matching rule.
- **What composition regenerates:**
  - random neurons and isotropic directions, exactly;
  - covariance directions, from the recorded train activations, within a rounding tolerance (ADR-038 argument).

**Not matched:**
- concept correlation (it would make the null meaningless);
- sparsity;
- attribution score (circular).

**Which null matters** (Phase-6 realistic runs, R7):
- The covariance-matched null is often much harder to beat than the isotropic one. A high AUROC can fail it (for example MLP K1: AUROC 0.92, contradicted).
- The isotropic null is reported only as sensitivity.
