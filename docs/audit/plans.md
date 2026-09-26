# Writing an audit plan

A plan is written **before** the audit runs, and ideally before the tests do. It is a record: its id covers everything it declares, and every report embeds it.

```python
import beyondnn as bnn
AU, F = bnn.audits, bnn.faithfulness

plan = AU.plan(
    name="my_audit",
    checkpoint=AU.checkpoint_of(model),        # the FULL state digest
    declared_model=None,                       # or the ModelDeclaration the evidence used
    samples=[...],                             # sample ids instance evidence must be about
    datasets=[...],                            # concept_dataset ids for concept evidence
    claims=[
        AU.claim(
            "ig_necessary",
            statement="the IG top-k units are necessary for the margin",
            relation="necessary_for",
            target=None,
            sample_targets={sample: target, ...},  # or target=<one metric> for all samples
            scope="instance",                      # per-sample: one standing per declared sample
            requirement="necessity_v1",
            selection=AU.selection("net.1", method="integrated_gradients", k=None),
            invariant_over=[AU.invariance("replacement", min_values=3),
                            AU.invariance("k", min_values=2),
                            AU.invariance("null", min_values=2)],
        ),
    ],
    requirements=[
        AU.requirement("necessity_v1", policy=F.COMPREHENSIVENESS_POLICY, controls=True,
                       alternatives=[AU.alternative("comprehensiveness", "min_drop", factor=0.5)]),
    ],
    concepts=[AU.concept(concept_id, policy=bnn.concepts.POLICY_V1,
                         invariant_over=[AU.invariance("null", min_values=2)])],
    counterexamples=AU.counterexample_rule(max_counterexample_fraction=None,
                                           max_false_positive_rate=0.1,
                                           max_false_negative_rate=0.1),
    naive_auroc=0.6,
)
```

## Fields

- **`subject` / `selection`.** A claim is about a declared subject (`Subject`: site, units, unit axes, feature) **or** about the units a method selects on each sample (`selection(site, method, k)`). `method` is an attribution method name, or `"declared"` for declared or random unit sets. Selection claims need the attribution traces, so the selection method can be verified.
- **`scope`:**
  - `"instance"`: audited on every plan sample;
  - `"finite_sample"`: about exactly `sample_set`, an `Estimand.sample_id`;
  - `"population"`: about `population`. No BeyondNN protocol yields population evidence yet, so such claims are UNSUPPORTED (`narrower_estimand`) or NOT_EVALUATED.
- **`target` / `sample_targets`.** One declared target, or one per plan sample, for example a margin fixed from each sample's clean pass.
- **`invariant_over`.** The assumption axes the claim asserts it holds across. Fewer tested values than declared makes the claim UNSUPPORTED (`invariance_untested`).
- **`requirement`:**
  - `policy`: a registry-checked `AssessmentPolicy` that decides the verdict;
  - `controls`: a SUPPORTS result must have a control criterion that enters its decision;
  - `alternatives`: criteria under which recorded results are re-evaluated (labelled; never part of a verdict). Not available for `concept_encoding`: record a second encoding test instead.
- **`concepts`:** concepts asserted to be VALIDATED_CONCEPT under a `ConceptPolicy`, with invariances over null (encoding tests), replacement (use tests) or dataset (validations).
- **`counterexamples`:** caps. `None` means "no cap declared", and the report says so.
- **`naive_auroc`:** if declared, an encoding test that its controls reject, although its AUROC reaches this value, is flagged `controls_defeat_encoding`.

## Refusals

A plan is refused if:
- names are duplicated;
- a claim names an unknown requirement;
- a causal claim's requirement names no protocol for its relation;
- `sample_targets` do not cover exactly the plan samples;
- a policy uses unregistered or unjustified protocols;
- alternatives are declared for `concept_encoding`.
