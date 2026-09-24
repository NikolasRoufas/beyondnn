# Ecosystem Audit

Date: 2026-09-24. Versions were checked against PyPI and official docs on that date.
Literature points (method papers, benchmarks) come from prior knowledge and are cited
by author/year. Verify them before quoting them in a paper.

| Tool | Version checked | Release |
|---|---|---|
| PyTorch | 2.14.0 | current stable |
| Captum | 0.9.0 | 2026-04-17 |
| TransformerLens | 4.0.0 | 2026-09-21 |
| nnsight | 0.7.0 (0.8.0rc1 pre-release) | 2026-05-05 |
| SAELens | 6.51.2 | 2026-09-24 |
| pyvene | 0.1.8 | 2025-05-26 (no release in 16 months) |
| Quantus | 0.6.0 | 2025-07-21 |
| circuit-tracer | on PyPI | active |
| Inseq | 0.4.x on PyPI (0.7 announced) | slow cadence |

---

## 1. PyTorch (substrate, not a competitor)

- **Purpose:** tensors, autograd, `nn.Module`, device backends, compilation, distribution.
- **Relevant primitives:**
  - `register_forward_pre_hook(hook, *, prepend, with_kwargs)` can rewrite inputs.
  - `register_forward_hook(hook, *, prepend, with_kwargs, always_call)` can replace outputs by returning a value. `always_call=True` makes the hook run even when `forward` raises.
  - `register_full_backward_hook` gives gradients w.r.t. module inputs/outputs.
  - `RemovableHandle.remove()`, `named_modules()` (deduplicates shared modules by default), `torch.func` (functional transforms, `vmap`, `jvp`).
  - Global module hooks: `torch.nn.modules.module.register_module_forward_hook`.
- **Limitations for us:**
  - Hooks only see module boundaries. Functional ops inside `forward` (`F.relu(x)`, `x + y`, attention softmax inside an SDPA kernel) are invisible unless wrapped in a module.
  - Hooks interact badly with `torch.compile`: they cause graph breaks or recompiles, and their behaviour is not documented on the `nn.Module` page.
  - In-place ops can mutate a tensor after a hook has captured it by reference.
- **BeyondNN should reuse:** everything. Hooks are the instrumentation mechanism. Tensors, dtypes, and devices come from PyTorch. Sidecar serialisation uses `torch.save`/`torch.load(weights_only=True)`.
- **Should not duplicate:** anything. No custom tensor type, no custom autograd, no module re-implementation.
- **Design consequence:** BeyondNN's "coverage" is bounded by module granularity. This has to be surfaced as a `TraceLimitation` (`MISSING_LAYER_COVERAGE` / `FUNCTIONAL_OPS_UNOBSERVED`), not hidden.

## 2. Captum

- **Purpose:** the mature, general PyTorch attribution library.
- **Abstractions:** `Attribution` classes in three families: primary (input) attribution (Saliency, InputXGradient, IntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Occlusion, FeatureAblation, FeaturePermutation, ShapleyValueSampling, KernelShap, Lime); `Layer*` attribution; `Neuron*` attribution. Also `LLMAttribution` / `LLMGradientAttribution` wrappers for generative LMs, TCAV (concepts), TracIn influence, and the metrics `infidelity` and `sensitivity_max`.
- **Supported models:** any PyTorch callable. Architecture-agnostic.
- **Tracing:** none as a first-class object. Layer methods hook one layer at a time internally.
- **Intervention:** implicit only (perturbation baselines inside FeatureAblation and Occlusion). There is no user-facing intervention context.
- **Attribution:** best in class. Also the reference implementation for IG convergence delta.
- **Causal:** perturbation-based attribution is interventional on *inputs*. There is no internal-state causal API.
- **Concepts:** TCAV (Kim et al. 2018), including statistical significance testing against random concepts. This is the best existing precedent for "validated concept" semantics.
- **Strengths:** mature, tested, broad method coverage, architecture-agnostic.
- **Limitations:** returns raw tensors. No provenance, no epistemic labelling, no unified record type, and no notion of "explanation" beyond a tensor of scores. Its faithfulness metrics are only infidelity and sensitivity.
- **Reuse:** all attribution algorithms beyond trivial gradients, TCAV, and possibly `infidelity`. Use it through an **adapter that converts outputs into `AttributionRecord`s with provenance**.
- **Don't duplicate:** IG, DeepLift, SHAP variants, occlusion, LIME. The only attribution BeyondNN should own natively is plain gradient and input×gradient (a few lines each), so the base install is useful without Captum and there is a reference to test the adapter against.

## 3. TransformerLens (4.0)

- **Purpose:** mechanistic interpretability of generative LMs.
- **Abstractions:** `TransformerBridge` (new standard in 4.0, wraps HF weights, 140+ architecture families), `HookPoint`, `ActivationCache`, `run_with_cache`, `run_with_hooks`, and standardised hook names (`blocks.{l}.attn.hook_z`, …). Legacy `HookedTransformer.from_pretrained` was removed in 4.0.
- **Supported models:** transformers and, experimentally, Mamba/SSM. **Not** CNNs, MLPs, or arbitrary modules.
- **Tracing:** excellent for supported LMs. Canonical names make results comparable across models.
- **Intervention:** hook functions that return modified activations. Patching utilities.
- **Attribution:** logit lens, direct logit attribution, and attribution patching idioms (mostly in tutorials and user code).
- **Causal:** activation/path patching patterns. It is the de facto standard for circuit work.
- **Concepts:** none. SAE integration moved to SAELens.
- **Strengths:** canonical activation names, huge community, many tutorials.
- **Limitations:** transformer-only. Results are tensors and caches with no epistemic metadata. Its major API churned in 4.0.
- **Reuse:** an optional adapter that maps TL hook names into BeyondNN module paths, so BeyondNN traces over TL models get canonical names.
- **Don't duplicate:** transformer-specific hook naming, model loading, weight processing, logit lens.

## 4. nnsight (0.7)

- **Purpose:** read and write the internals of any PyTorch model. It also supports remote execution on NDIF for very large models, and vLLM since 0.7.
- **Abstractions:** `NNsight(module)` / `LanguageModel`, `with model.trace(inp):` context, `Envoy` proxies for `.input`/`.output`, `.save()`, in-trace interventions, multi-invoke batching, generation-step iteration.
- **Supported models:** any `nn.Module` locally, with transformer/LLM conveniences.
- **Tracing / intervention:** the most general and powerful intervention engine in the ecosystem. It covers setting, patching, gradients, cross-prompt patching, and generation.
- **Causal:** a strong substrate for patching experiments. It provides no causal *bookkeeping*.
- **Concepts / attribution:** none built in.
- **Strengths:** architecture-agnostic, handles large models and remote execution, expressive.
- **Limitations:**
  - The deferred/threaded execution model is harder to reason about and debug than eager hooks.
  - Values must be `.save()`d explicitly.
  - Nothing is recorded about *what claim* an intervention supports.
  - There is no evaluation layer.
- **Reuse:** an optional backend for executing traces and interventions on models too big for eager in-process hooks, especially remote. That is the strongest scaling path BeyondNN has.
- **Don't duplicate:** remote execution, generation-time intervention scheduling, proxy-graph machinery. **This is the tool BeyondNN overlaps with most on tracing and intervention.**

## 5. SAELens (6.x)

- **Purpose:** train, load, and analyse sparse autoencoders (and transcoder variants).
- **Abstractions:** `SAE` classes (standard, gated, TopK, JumpReLU, …), `encode`/`decode`, a pretrained registry, a training runner, `HookedSAETransformer` (TL integration), and evals.
- **Supported models:** SAEs trained mostly on transformer LMs. `encode`/`decode` accepts activations from any source.
- **Concepts:** feature dashboards and auto-interp labels exist in the wider ecosystem (Neuronpedia). Those labels are *proposed* concepts in our terminology.
- **Strengths:** the standard for SAE work, with a large pretrained catalogue.
- **Limitations:** LM-centric. Feature labels usually come from LLM auto-interpretation and are not validated.
- **Reuse:** an optional adapter that exposes an SAE as a `FeatureBasis`. BeyondNN can then express "feature 41 at layer 5" as `(basis=sae_id, index=41)` and intervene on it.
- **Don't duplicate:** SAE architectures, training, pretrained registries.

## 6. pyvene (Stanford)

- **Purpose:** declarative, serialisable interventions on any PyTorch model, including trainable interventions (DAS / distributed alignment search, interchange interventions, low-rank rotated subspaces).
- **Abstractions:** `IntervenableModel`, `IntervenableConfig`, intervention classes, dict-serialisable configs.
- **Supported models:** any PyTorch module (RNNs, CNNs, ResNets, Mamba, transformers).
- **Causal:** the only library built around **causal abstraction** (Geiger et al.). Interchange-intervention accuracy is a principled causal-hypothesis test.
- **Strengths:** serialisable interventions, rigorous causal-abstraction methodology.
- **Limitations:** its last release was May 2025, so maintenance risk. The API is heavy, and there is no attribution or provenance layer.
- **Reuse:** conceptually, its "intervention as serialisable data" design, which BeyondNN should copy as a principle. Possibly a DAS adapter later.
- **Don't duplicate:** trainable interventions and DAS, unless pyvene becomes unmaintained and those features prove necessary.
- **Overlap:** high. BeyondNN's `InterventionSpec` design is close to pyvene configs.

## 7. circuit-tracer (Anthropic / decoderesearch)

- **Purpose:** attribution graphs over transcoder / cross-layer transcoder features for LMs, plus visualisation and feature interventions.
- **Causal:** computes direct effects between features in a linearised replacement model, then validates them with interventions. This is the closest existing thing to the "INPUT → features → OUTPUT with causal evidence" picture in the BeyondNN pitch, **for LMs with pretrained transcoders.**
- **Limitations:** needs pretrained transcoders and is LM-specific. The graph describes a *replacement* model, and its error nodes quantify how much is unexplained.
- **Lesson for BeyondNN:** first-class "error node" / unexplained-residual accounting is the right way to show what an explanation does not cover. Copy that idea into `TraceLimitation` and into the metrics.
- **Don't duplicate:** attribution-graph computation. Possible adapter later.

## 8. Quantus

- **Purpose:** evaluation of explanations. It has 35+ metrics in six categories: faithfulness (pixel flipping, region perturbation, faithfulness correlation, ROAD, infidelity, monotonicity), robustness, localisation, complexity, randomisation (Adebayo sanity checks), and axiomatic.
- **Scope:** input-attribution maps, mostly images. PyTorch and TF.
- **Overlap:** **high with BeyondNN's Phase 5 plan** for input-level faithfulness. BeyondNN should not reimplement pixel-flipping-style metrics without a reason.
- **Gap:** Quantus evaluates *input* saliency maps. It does not evaluate claims about *internal* features or components, which is where BeyondNN can add value.

## 9. Inseq

- **Purpose:** attribution for sequence-generation models (HF), wrapping Captum with step-wise generation and contrastive attribution.
- **Lesson:** a thin, well-typed layer over Captum can be genuinely useful. It is also a warning: a narrow scope (seq2seq) plus a slow release cadence limits adoption.

## 10. Concept-based methods (no dominant library)

- **Concept Bottleneck Models** (Koh et al. 2020) and many variants (CEM, stochastic CBMs, label-free CBMs). Implementations are scattered per paper. "PyTorch, Explain!" (`torch_explain`) is the closest to a library.
- **Known weakness:** concept *leakage*. Bottleneck activations encode information beyond the named concept, so the downstream head can use it (Margeloiu et al. 2021; Mahinpei et al. 2021). **A bottleneck named "wing colour" is not guaranteed to mean or only carry wing colour.** This directly affects the Mode B design.
- **TCAV / CAVs:** a linear direction plus a significance test against random directions. It establishes *decodability plus directional sensitivity*, not causal use.
- **Probing literature:** high probe accuracy does not imply use (Hewitt & Liang 2019 control tasks; Belinkov 2022 review; Ravichander et al. 2021; "Causality ≠ Decodability", 2025). Amnesic probing / INLP (Elazar et al. 2021) and causal probing are the relevant corrections.

## 11. Activation patching / causal tracing methodology

- **Causal tracing / ROME** (Meng et al. 2022): corrupt the input with Gaussian noise, then restore clean states. It is sensitive to the noise level.
- **Best practices** (Zhang & Nanda 2023; Heimersheim & Nanda 2024):
  - Prefer symmetric token replacement over Gaussian noise.
  - Prefer logit difference over probability.
  - Report several metrics.
  - Denoising and noising answer *different* questions (sufficiency-like vs necessity-like).
- **Attribution patching** (Nanda 2023; Kramár et al. 2024 AtP*): a gradient linearisation of patching. It is an *estimate* of an interventional quantity, not the quantity itself.
- **Causal abstraction** (Geiger et al. 2021–2024): the most rigorous framework for claims of the form "component C implements variable V".
- **Position (2026):** "Mechanistic interpretability must disclose identification assumptions for causal claims" (arXiv 2605.08012). This supports BeyondNN's premise that causal claims need explicit assumptions attached.
- **Ablation choice matters:** zero ablation is often off-distribution. Mean and resample ablation are more defensible. Every effect is relative to its ablation baseline.

## 12. Ground-truth benchmarks (important prior art for BeyondNN's benchmark plan)

- **Tracr** (Lindner et al. 2023): compiles RASP programs into transformers with known circuits.
- **InterpBench** (Gupta et al. 2024): semi-synthetic transformers trained to implement known circuits.
- **MIB – Mechanistic Interpretability Benchmark** (Mueller et al. 2025): circuit localisation and causal variable localisation.
- **OpenXAI** (Agarwal et al. 2022): synthetic data with ground-truth feature importance for attribution evaluation.
- **AxBench** (Wu et al. 2025): concept detection and steering. Notably, simple baselines often beat SAEs.
- **RAVEL** (Huang et al. 2024): evaluates disentangling of entity attributes.

**Consequence:** the planned "A causal / B correlated / C noise" suite is a good idea, but it is not new as a category. It must be framed as a *regression and test suite for the framework's own claims*, not as a novel benchmark contribution. It should also borrow Tracr/InterpBench-style models where possible.

## 13. Explanation evaluation literature (for Phase 5)

- **ERASER** (DeYoung et al. 2020): comprehensiveness and sufficiency for token rationales.
- **ROAR** (Hooker et al. 2019): remove-and-retrain. Removal without retraining confounds the result with distribution shift.
- **Sanity checks** (Adebayo et al. 2018): model and label randomisation tests. Many saliency methods fail them.
- **Infidelity / sensitivity** (Yeh et al. 2019).
- **ROAD** (Rong et al. 2022): noisy linear imputation to reduce masking-induced distribution shift.
- **Consistent lesson:** every perturbation-based faithfulness metric depends on the choice of perturbation and baseline, and different metrics often disagree. Rankings of methods are metric-dependent (Tomsett et al. 2020, "Sanity checks for saliency metrics").

---

## Summary matrix

| Capability | PyTorch | Captum | TL 4 | nnsight | SAELens | pyvene | circuit-tracer | Quantus |
|---|---|---|---|---|---|---|---|---|
| Any `nn.Module` | ✓ | ✓ | ✗ | ✓ | n/a | ✓ | ✗ | ✓ |
| Activation capture | hooks | internal | ✓✓ | ✓✓ | – | ✓ | ✓ | – |
| Interventions | hooks | implicit | ✓✓ | ✓✓ | via TL | ✓✓ | features | – |
| Input attribution | autograd | ✓✓ | partial | manual | – | – | ✓ | uses Captum |
| Internal causal effects | – | – | patterns | substrate | – | ✓✓ (abstraction) | ✓✓ (graph) | – |
| Concepts | – | TCAV | – | – | features | DAS | features | – |
| Explanation evaluation | – | 2 metrics | – | – | SAE evals | IIA | error nodes | ✓✓ |
| Epistemic status labels | – | – | – | – | – | – | partial (error nodes) | – |
| Provenance records | – | – | – | – | – | config ser. | – | – |
| Unified serialisable evidence schema | – | – | – | – | – | interventions only | graph JSON | – |
