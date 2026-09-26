# Features (ADR-040, ADR-041)

A **feature** is a structural coordinate of one module-output leaf. It says *where* and *how*, never *what it means*.

| Basis | Declared by | Activation on one input | Intervention |
|---|---|---|---|
| neuron | `concepts.neuron(site, index, axis=1, pooling="none")` | the value at `index` of `axis` | unit replacement at every other position (`unit_axes=(axis,)`) |
| direction | `concepts.direction(site, v, axis=1, pooling="none")` | ⟨x, v/‖v‖⟩ along `axis` | DIRECTION removal or retention (below) |
| SAE latent | `concepts.sae_feature(model, site, encoder=, decoder=, b_enc=, b_dec=, latent=, checkpoint=)` | relu(⟨x − b_dec, W_enc[:, i]⟩ + b_enc[i]) | DIRECTION along the normalised decoder row (*projection*, not SAE-native ablation) |

- **Pooling.** `none` requires every non-batch, non-feature axis to have size 1. `mean` averages over them (for example the spatial positions of a conv channel map).
- **Identity.** The exact vector is part of a feature's identity (a content digest), so a changed norm is a different feature. An SAE feature also records the SAE checkpoint, its latent index, and the caller-reported reconstruction statistics.
- **Checkpoint binding.** Fitted, searched and SAE features record the model state digest they were derived on. Using them with another checkpoint is refused.

## Discovery (outside validation)

- `fit_direction(model, data, site=, method="mean_difference")`: the difference of the class means of the pooled site vectors on the **train** split only.
- `search_neurons(model, data, site=)`: the index with the highest sign-free train AUROC. The candidate count, criterion and sign are recorded.

A fitted or searched feature can be evaluated only on the concept dataset it was derived on, so its test split is guaranteed disjoint from its train split. Composition re-derives the fit or search from the kept fitting recording.

## Directions are not units

A direction is a vector, not an index. `interventions.direction(site, v, axis=, reference=, retain=)` builds the Phase-2 DIRECTION operation (`InterventionRecord` v4), with v̂ = v/‖v‖ and reference r (zeros, or a tensor of the leaf's exact shape):

- **Removal** (`retain=False`): x′ = x − ⟨x − r, v̂⟩ v̂. The coordinate along v̂ becomes the reference's.
- **Retention** (`retain=True`): x′ = r + ⟨x − r, v̂⟩ v̂. Only the coordinate along v̂ is kept.

**Addition and scaling (steering) are not implemented.** They push activations off-distribution by construction, and conflate influence with use.
