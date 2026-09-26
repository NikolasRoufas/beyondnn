"""Shared concept machinery: feature activation rules, AUROC, controls, record copying.

Everything here is deterministic and used identically by the runners and by
verification (``concepts.verify``), so a recorded statistic can be re-derived.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from beyondnn.core.trace import TraceConfig, TraceResult
from beyondnn.schema import (
    BaseRecord,
    FeatureBasis,
    FeatureRecord,
    ProvenanceRecord,
    TensorRef,
)

__all__ = [
    "ConceptError",
    "Control",
    "auroc",
    "copy_record",
    "feature_values",
    "new_store",
    "pooled_vectors",
    "random_directions_for",
    "random_indices_for",
    "sample_set_id",
    "superiority",
]


class ConceptError(ValueError):
    """A concept operation was refused (missing controls, leakage, mismatched scope, ...)."""


# ------------------------------------------------------------------ identities


def sample_set_id(ids: Sequence[str]) -> str:
    """The identity of an ordered sample set (the Phase-2 finite-sample convention)."""
    return "sample:" + hashlib.sha256("\n".join(ids).encode()).hexdigest()


def new_store() -> TraceResult:
    """An empty trace for records not produced by an execution (features, datasets,
    proposals, validations). Sealed by the caller."""
    return TraceResult(TraceConfig((), (), "cpu"))


def copy_record(target: TraceResult, source: TraceResult, record: BaseRecord) -> BaseRecord:
    """Add ``record`` (with its provenance and retained tensors) from ``source`` into
    ``target``; records are content-addressed, so the id is unchanged."""
    if record.provenance_id is not None and record.provenance_id not in {
        r.id for r in target.records
    }:
        provenance = source.get(record.provenance_id)
        assert isinstance(provenance, ProvenanceRecord)
        target._add(provenance)
    retained = getattr(record, "retained_tensors", None)
    if retained is not None:
        for ref in retained():
            assert ref.storage_key is not None
            target._add_tensor(ref.storage_key, source.tensor(ref))
    return target._add(record)


def tensor_of(trace: TraceResult, ref: TensorRef | None) -> torch.Tensor:
    if ref is None or ref.storage_key is None:
        raise ConceptError("a feature tensor was not retained")
    return trace.tensor(ref)


# ------------------------------------------------------------------ activation rules


def _move_axis(x: torch.Tensor, axis: int) -> torch.Tensor:
    """(batch=1, ...) leaf -> (positions, d): the feature axis last, other non-batch axes
    flattened into positions."""
    if x.dim() < 2 or x.shape[0] != 1:
        raise ConceptError(f"a feature site must have batch size 1, got shape {tuple(x.shape)}")
    if axis >= x.dim():
        raise ConceptError(f"feature axis {axis} does not exist for shape {tuple(x.shape)}")
    moved = x.detach().double().movedim(axis, -1)[0]
    return moved.reshape(-1, moved.shape[-1])


def pooled_vectors(activation: torch.Tensor, feature: FeatureRecord) -> torch.Tensor:
    """The per-input feature-axis vector (pooled over the other non-batch axes)."""
    rows = _move_axis(activation, feature.axis)
    if feature.pooling == "none":
        if rows.shape[0] != 1:
            raise ConceptError(
                f"pooling 'none' needs every non-feature axis to have size 1 "
                f"(shape {tuple(activation.shape)}); declare pooling='mean'"
            )
        return rows[0]
    return rows.mean(dim=0)


def feature_values(activation: torch.Tensor, feature: FeatureRecord, store: TraceResult) -> float:
    """The feature's scalar activation on one input (ADR-040 activation rules)."""
    rows = _move_axis(activation, feature.axis)
    if feature.pooling == "none" and rows.shape[0] != 1:
        raise ConceptError(
            f"pooling 'none' needs every non-feature axis to have size 1 "
            f"(shape {tuple(activation.shape)}); declare pooling='mean'"
        )
    if feature.basis is FeatureBasis.NEURON:
        assert feature.index is not None
        if feature.index >= rows.shape[1]:
            raise ConceptError(
                f"neuron {feature.index} is out of range for axis size {rows.shape[1]}"
            )
        per_position = rows[:, feature.index]
    elif feature.basis is FeatureBasis.DIRECTION:
        v = _unit(tensor_of(store, feature.direction), rows.shape[1])
        per_position = rows @ v
    else:
        assert feature.sae is not None
        enc = tensor_of(store, feature.sae.encoder).double()
        b_dec = tensor_of(store, feature.sae.b_dec).double()
        if enc.shape[0] != rows.shape[1]:
            raise ConceptError("the SAE encoder does not fit the feature axis")
        per_position = torch.relu((rows - b_dec) @ enc + feature.sae.b_enc)
    return float(per_position.mean())


def _unit(v: torch.Tensor, d: int) -> torch.Tensor:
    if v.dim() != 1 or v.shape[0] != d:
        raise ConceptError(f"a direction of shape {tuple(v.shape)} does not fit axis size {d}")
    v = v.double()
    out: torch.Tensor = v / v.norm()
    return out


# ------------------------------------------------------------------ statistics


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve (Mann-Whitney U / (P N), ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        raise ConceptError("AUROC needs both concept and non-concept samples in the split")
    ranked = sorted((s, i) for i, s in enumerate([*pos, *neg]))
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(ranks[: len(pos)])
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def superiority(
    observed: float, controls: Sequence[float], *, larger: bool = True
) -> dict[str, float]:
    """Fractions of controls strictly beaten / tied by ``observed`` (``larger``: a larger
    observed value beats a smaller control)."""
    n = len(controls)
    if n == 0:
        raise ConceptError("no control values")
    below = sum(1 for c in controls if (c < observed if larger else c > observed))
    tied = sum(1 for c in controls if c == observed)
    return {
        "fraction_beaten": below / n,
        "fraction_tied": tied / n,
        "superiority": (below + 0.5 * tied) / n,
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def q(p: float) -> float:
        pos = p * (len(ordered) - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {"median": q(0.5), "q05": q(0.05), "q95": q(0.95), "max": ordered[-1]}


# ------------------------------------------------------------------ controls


@dataclass(frozen=True, slots=True)
class Control:
    """A declared null (request §11/§17). ``kind``: ``random_directions`` (``distribution``
    ``isotropic`` or ``covariance``), ``random_neurons`` or ``label_permutation``."""

    kind: str
    n: int
    seed: int
    distribution: str | None = None

    def identity(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "n": self.n, "seed": self.seed}
        if self.kind == "random_directions":
            out |= {
                "distribution": self.distribution,
                "normalisation": "unit_l2",
                "matching": "same site, axis and pooling; unit norm; sign-free scoring",
            }
        elif self.kind == "random_neurons":
            out |= {"matching": "other indices of the same axis, uniform; sign-free scoring"}
        else:
            out |= {"matching": "the observed feature against permuted held-out labels"}
        return out


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def random_directions_for(
    control: Control, d: int, train_vectors: torch.Tensor | None
) -> list[torch.Tensor]:
    """Seeded random unit directions (float64, local generator). ``covariance`` draws
    v ~ N(0, Σ_train) with Σ the covariance of the train-split pooled vectors."""
    g = _generator(control.seed)
    z = torch.randn(control.n, d, generator=g, dtype=torch.float64)
    if control.distribution == "covariance":
        if train_vectors is None or train_vectors.shape[0] < 2:
            raise ConceptError("covariance-matched directions need train-split activations")
        z = z @ covariance_root(train_vectors)
    elif control.distribution != "isotropic":
        raise ConceptError("random directions are 'isotropic' or 'covariance'")
    norms = z.norm(dim=1, keepdim=True)
    if bool((norms == 0).any()):
        raise ConceptError("a random direction had zero norm (degenerate covariance)")
    return list(z / norms)


def covariance_root(train_vectors: torch.Tensor) -> torch.Tensor:
    """The symmetric square root Σ^½ = V Λ^½ Vᵀ of the train-split covariance (unique,
    so it does not depend on the eigenvector basis of repeated eigenvalues)."""
    centred = train_vectors - train_vectors.mean(dim=0, keepdim=True)
    cov = centred.T @ centred / (train_vectors.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov)
    root: torch.Tensor = (eigvec * eigval.clamp(min=0.0).sqrt()) @ eigvec.T
    return root


def random_indices_for(control: Control, d: int, exclude: int) -> list[int]:
    """Seeded random other indices of the same axis (with replacement across draws)."""
    if d < 2:
        raise ConceptError("random neurons need at least two units on the axis")
    g = _generator(control.seed)
    draws = torch.randint(0, d - 1, (control.n,), generator=g).tolist()
    return [i if i < exclude else i + 1 for i in draws]


def permutations_for(control: Control, n: int) -> list[list[int]]:
    g = _generator(control.seed)
    return [torch.randperm(n, generator=g).tolist() for _ in range(control.n)]
