"""Feature definitions (ADR-040): neurons, directions, SAE latents, and train-only
discovery (``fit_direction``, ``search_neurons``).

A feature is structural: where it lives, how its activation is computed, and how it
was obtained. It carries no meaning (UNLABELED_FEATURE) until a concept is proposed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from beyondnn.core.trace import TraceResult, recording
from beyondnn.provenance import fingerprint_model
from beyondnn.schema import (
    ConceptDataset,
    FeatureBasis,
    FeatureRecord,
    FeatureSource,
    JsonMap,
    ModelDeclaration,
    SAEIdentity,
    SemanticStatus,
    Site,
)

from ._core import (
    ConceptError,
    auroc,
    new_store,
    pooled_vectors,
    sample_set_id,
)
from .data import ConceptData

__all__ = [
    "Feature",
    "direction",
    "fit_direction",
    "neuron",
    "record_site",
    "sae_feature",
    "search_neurons",
]


@dataclass(frozen=True, slots=True, eq=False)
class Feature:
    """A feature record with the store holding it (and its tensors). ``fit_trace`` is the
    recording of train activations a fitted/searched feature was derived from (kept so
    composition can re-derive the feature). ``expected_sign`` is the declared direction
    of association with a concept (+1: concept -> higher activation)."""

    record: FeatureRecord
    store: TraceResult
    fit_trace: TraceResult | None = None

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def semantic_status(self) -> SemanticStatus:
        return SemanticStatus.UNLABELED_FEATURE

    @property
    def expected_sign(self) -> int:
        sign = self.record.source.params.get("sign", 1)
        assert sign in (1, -1)
        return int(sign)  # type: ignore[arg-type]

    def vector(self) -> torch.Tensor:
        """The (unnormalised) direction vector of a direction or SAE feature."""
        if self.record.direction is None:
            raise ConceptError("a neuron feature has no direction vector")
        return self.store.tensor(self.record.direction)


def _site(site: str, output_path: str) -> Site:
    if not isinstance(site, str) or not site:
        raise ConceptError("a feature lives at a named (non-root) module output")
    return Site(module=site, output_path=output_path)


def _check_axis(axis: int, pooling: str) -> None:
    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 1:
        raise ConceptError("axis is a non-batch axis of the leaf (int >= 1)")
    if pooling not in ("none", "mean"):
        raise ConceptError("pooling is 'none' or 'mean'")


def _build(record_kwargs: dict[str, Any], tensors: Sequence[torch.Tensor]) -> Feature:
    from beyondnn.interventions.runner import _retained_ref

    store = new_store()
    refs = []
    for t in tensors:
        ref = _retained_ref(t, "cpu")
        store._add_tensor(ref.storage_key or "", t)
        refs.append(ref)
    extra = _bind(record_kwargs, refs)
    record = FeatureRecord(**record_kwargs, **extra)
    store._add(record)
    store._seal()
    return Feature(record, store)


def _bind(kwargs: dict[str, Any], refs: list[Any]) -> dict[str, Any]:
    basis = kwargs["basis"]
    if basis is FeatureBasis.NEURON:
        return {}
    if basis is FeatureBasis.DIRECTION:
        return {"direction": refs[0]}
    sae = kwargs.pop("_sae")
    return {"direction": refs[0], "sae": SAEIdentity(encoder=refs[1], b_dec=refs[2], **sae)}


def _vector(v: torch.Tensor, what: str) -> torch.Tensor:
    if not isinstance(v, torch.Tensor) or v.dim() != 1 or not v.is_floating_point():
        raise ConceptError(f"{what} must be a 1-D floating-point tensor")
    if not bool(torch.isfinite(v).all()) or float(v.norm()) == 0.0:
        raise ConceptError(f"{what} must be finite and non-zero")
    return v.detach().to("cpu", torch.float32).clone(memory_format=torch.contiguous_format)


def neuron(
    site: str,
    index: int,
    *,
    axis: int = 1,
    pooling: str = "none",
    call_index: int = 0,
    output_path: str = "",
) -> Feature:
    """A declared neuron: ``index`` along ``axis`` of ``site``'s output."""
    _check_axis(axis, pooling)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ConceptError("a neuron index is an int >= 0")
    return _build(
        {
            "basis": FeatureBasis.NEURON,
            "site": _site(site, output_path),
            "call_index": call_index,
            "axis": axis,
            "pooling": pooling,
            "index": index,
            "source": FeatureSource(kind="declared"),
        },
        (),
    )


def direction(
    site: str,
    vector: torch.Tensor,
    *,
    axis: int = 1,
    pooling: str = "none",
    call_index: int = 0,
    output_path: str = "",
) -> Feature:
    """A declared direction along ``axis`` (activation ``<x, v/|v|>``). The exact vector
    (not only its direction) is its identity: a changed norm is a different feature."""
    _check_axis(axis, pooling)
    return _build(
        {
            "basis": FeatureBasis.DIRECTION,
            "site": _site(site, output_path),
            "call_index": call_index,
            "axis": axis,
            "pooling": pooling,
            "source": FeatureSource(kind="declared"),
        },
        (_vector(vector, "a direction"),),
    )


def sae_feature(
    model: nn.Module,
    site: str,
    *,
    encoder: torch.Tensor,
    decoder: torch.Tensor,
    b_enc: torch.Tensor,
    b_dec: torch.Tensor,
    latent: int,
    checkpoint: str,
    axis: int = 1,
    pooling: str = "none",
    reconstruction: dict[str, Any] | None = None,
    search: dict[str, Any] | None = None,
    data: ConceptData | None = None,
    call_index: int = 0,
    output_path: str = "",
) -> Feature:
    """Adapter for one latent of an external SAE (ADR-041): ``encoder`` (d, d_sae),
    ``decoder`` (d_sae, d), ``b_enc`` (d_sae,), ``b_dec`` (d,); activation
    ``relu(<x - b_dec, encoder[:, latent]> + b_enc[latent])``; interventions act along
    the normalised decoder row. No SAE library is imported. ``search`` (with ``data``)
    records how the latent was chosen on the train split (candidate count, criterion)."""
    _check_axis(axis, pooling)
    if encoder.dim() != 2 or decoder.dim() != 2 or encoder.shape != decoder.T.shape:
        raise ConceptError("encoder is (d, d_sae) and decoder is (d_sae, d)")
    d, d_sae = encoder.shape
    if b_enc.shape != (d_sae,) or b_dec.shape != (d,):
        raise ConceptError("b_enc is (d_sae,) and b_dec is (d,)")
    if not 0 <= latent < d_sae:
        raise ConceptError(f"latent {latent} out of range for {d_sae} latents")
    if search is None:
        source = FeatureSource(kind="sae")
    else:
        if data is None:
            raise ConceptError("a searched SAE latent names the concept dataset it was chosen on")
        source = FeatureSource(
            kind="search",
            method=str(search.get("method", "train_auroc")),
            params=JsonMap(dict(search)),
            dataset=data.record.id,
            split="train",
        )
    return _build(
        {
            "basis": FeatureBasis.SAE,
            "site": _site(site, output_path),
            "call_index": call_index,
            "axis": axis,
            "pooling": pooling,
            "source": source,
            "model_state_digest": fingerprint_model(model).state_digest,
            "_sae": {
                "checkpoint": checkpoint,
                "latent": latent,
                "d_sae": d_sae,
                "activation": "relu",
                "b_enc": float(b_enc[latent]),
                "reconstruction": JsonMap(dict(reconstruction or {})),
            },
        },
        (
            _vector(decoder[latent], "the decoder row"),
            _vector(encoder[:, latent], "the encoder column"),
            b_dec.detach().to("cpu", torch.float32).clone(memory_format=torch.contiguous_format),
        ),
    )


# ------------------------------------------------------------------ recording activations


def record_site(
    model: nn.Module,
    data: ConceptData,
    indices: Sequence[int],
    site: Site,
    *,
    declared_model: ModelDeclaration | None = None,
) -> TraceResult:
    """One clean pass per selected dataset sample, retaining ``site`` (MEASURED)."""
    if model.training:
        raise ConceptError("put the model in eval mode (training-mode passes are not paired)")
    with (
        torch.no_grad(),
        recording(
            model, sites=[site.module], retention="cpu", declared_model=declared_model
        ) as ctx,
    ):
        for i in indices:
            model(*data.inputs[i], **data.kwargs[i])
    return ctx.result


def site_tensors(
    trace: TraceResult,
    data: ConceptData | ConceptDataset,
    indices: Sequence[int],
    site: Site,
    call_index: int,
) -> list[tuple[Any, torch.Tensor]]:
    """(ActivationRecord, tensor) for each selected sample, matched by exact sample id."""
    record = data if isinstance(data, ConceptDataset) else data.record
    by_sample: dict[str, int] = {}
    for inp in trace.inputs:
        if inp.sample_id is None or inp.pass_index is None:
            raise ConceptError("a recorded input has no identity; its sample cannot be matched")
        by_sample.setdefault(inp.sample_id, inp.pass_index)
    out = []
    for i in indices:
        sample = record.samples[i]
        if sample not in by_sample:
            raise ConceptError(f"sample {sample} was not recorded")
        act = trace.activation(
            site.module,
            output_path=site.output_path,
            pass_index=by_sample[sample],
            call_index=call_index,
        )
        if act.value.storage_key is None:
            raise ConceptError("the site activation was not retained")
        out.append((act, trace.tensor(act)))
    return out


def _train_vectors(
    model: nn.Module,
    data: ConceptData,
    site: Site,
    axis: int,
    pooling: str,
    call_index: int,
    declared_model: ModelDeclaration | None,
) -> tuple[TraceResult, list[int], torch.Tensor, list[int]]:
    train = list(data.record.indices("train"))
    if not train:
        raise ConceptError("the concept dataset has no train split")
    probe = FeatureRecord(
        basis=FeatureBasis.NEURON,
        site=site,
        call_index=call_index,
        axis=axis,
        pooling=pooling,
        index=0,
        source=FeatureSource(kind="declared"),
    )
    trace = record_site(model, data, train, site, declared_model=declared_model)
    rows = [pooled_vectors(t, probe) for _, t in site_tensors(trace, data, train, site, call_index)]
    labels = [data.record.labels[i] for i in train]
    if len(set(labels)) < 2:
        raise ConceptError("the train split needs both concept and non-concept samples")
    return trace, train, torch.stack(rows), labels


def fit_direction(
    model: nn.Module,
    data: ConceptData,
    *,
    site: str,
    axis: int = 1,
    pooling: str = "none",
    method: str = "mean_difference",
    call_index: int = 0,
    output_path: str = "",
    declared_model: ModelDeclaration | None = None,
) -> Feature:
    """Discovery (outside validation): the difference of the concept and non-concept
    means of the pooled site vectors on the **train split only** (plan §8). The
    fitting recording is kept so the direction can be re-derived."""
    _check_axis(axis, pooling)
    if method != "mean_difference":
        raise ConceptError("only 'mean_difference' fitting exists (plan §8)")
    where = _site(site, output_path)
    trace, train, vectors, labels = _train_vectors(
        model, data, where, axis, pooling, call_index, declared_model
    )
    y = torch.tensor(labels, dtype=torch.bool)
    v = vectors[y].mean(dim=0) - vectors[~y].mean(dim=0)
    feature = _build(
        {
            "basis": FeatureBasis.DIRECTION,
            "site": where,
            "call_index": call_index,
            "axis": axis,
            "pooling": pooling,
            "source": FeatureSource(
                kind="fit",
                method=method,
                params=JsonMap(
                    {
                        "train_set": sample_set_id([data.record.samples[i] for i in train]),
                        "n_train": len(train),
                        "n_concept": int(y.sum()),
                        "sign": 1,
                    }
                ),
                dataset=data.record.id,
                split="train",
            ),
            "model_state_digest": fingerprint_model(model).state_digest,
        },
        (_vector(v.float(), "the fitted direction"),),
    )
    return Feature(feature.record, feature.store, trace)


def search_neurons(
    model: nn.Module,
    data: ConceptData,
    *,
    site: str,
    axis: int = 1,
    pooling: str = "none",
    call_index: int = 0,
    output_path: str = "",
    declared_model: ModelDeclaration | None = None,
) -> Feature:
    """Discovery: the neuron with the highest sign-free train-split AUROC among every
    index of ``axis`` (ties -> lower index). The candidate count, criterion and chosen
    sign are recorded (request §29)."""
    _check_axis(axis, pooling)
    where = _site(site, output_path)
    trace, train, vectors, labels = _train_vectors(
        model, data, where, axis, pooling, call_index, declared_model
    )
    best: tuple[float, int, int] | None = None
    for j in range(vectors.shape[1]):
        a = auroc(vectors[:, j].tolist(), labels)
        score, sign = (a, 1) if a >= 1 - a else (1 - a, -1)
        if best is None or score > best[0]:
            best = (score, j, sign)
    assert best is not None
    score, index, sign = best
    feature = _build(
        {
            "basis": FeatureBasis.NEURON,
            "site": where,
            "call_index": call_index,
            "axis": axis,
            "pooling": pooling,
            "index": index,
            "source": FeatureSource(
                kind="search",
                method="train_auroc",
                params=JsonMap(
                    {
                        "candidate_count": int(vectors.shape[1]),
                        "criterion": "max sign-free train AUROC, ties to the lower index",
                        "train_auroc": score,
                        "train_set": sample_set_id([data.record.samples[i] for i in train]),
                        "sign": sign,
                    }
                ),
                dataset=data.record.id,
                split="train",
            ),
            "model_state_digest": fingerprint_model(model).state_digest,
        },
        (),
    )
    return Feature(feature.record, feature.store, trace)
