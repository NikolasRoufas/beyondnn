"""Immutable value types embedded in records.

Value types have no identity of their own; they contribute to the identity of
the record that contains them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .errors import SchemaError
from .status import EstimandScope

__all__ = [
    "ClaimSource",
    "ClaimSourceKind",
    "Estimand",
    "NamedTensor",
    "Site",
    "SiteIO",
    "Subject",
    "TargetSpec",
    "TensorRef",
    "TensorStats",
]

_DTYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DEVICE_RE = re.compile(r"^[a-z][a-z0-9_]*(:[0-9]+)?$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODULE_SEGMENT_RE = re.compile(r"^[^.\s]+$")
_METRIC_RE = re.compile(r"^[a-z][a-z0-9_]*(:[A-Za-z0-9_.\-]+)?$")
_TOKEN_RE = re.compile(r"^\S+$")


def _nonneg_or_nan(x: float) -> bool:
    return math.isnan(x) or x >= 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorStats(Value):
    """Summary statistics of a tensor, computed at capture time.

    Non-finite values are permitted (a tensor may contain NaN/Inf) and survive
    serialisation.
    """

    numel: int
    mean: float
    std: float
    min: float
    max: float
    l2_norm: float

    def _validate(self) -> None:
        require(self.numel >= 0, "TensorStats.numel must be >= 0")
        require(_nonneg_or_nan(self.std), "TensorStats.std must be >= 0 or NaN")
        require(_nonneg_or_nan(self.l2_norm), "TensorStats.l2_norm must be >= 0 or NaN")
        if math.isfinite(self.min) and math.isfinite(self.max):
            require(self.min <= self.max, "TensorStats.min must be <= max")


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorRef(Value):
    """Metadata about a tensor. Never contains tensor data.

    ``storage_key`` names where the values were retained (resolved by the
    containing trace; ``None`` means the values were not retained).
    ``content_digest`` optionally identifies the exact bytes (``"sha256:<hex>"``).
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    stats: TensorStats | None = None
    storage_key: str | None = None
    content_digest: str | None = None

    def _validate(self) -> None:
        require(all(d >= 0 for d in self.shape), f"TensorRef.shape must be >= 0: {self.shape}")
        require(bool(_DTYPE_RE.match(self.dtype)), f"TensorRef.dtype invalid: {self.dtype!r}")
        require(bool(_DEVICE_RE.match(self.device)), f"TensorRef.device invalid: {self.device!r}")
        if self.stats is not None:
            require(
                self.stats.numel == self.numel,
                f"TensorRef.stats.numel={self.stats.numel} does not match shape {self.shape}",
            )
        if self.storage_key is not None:
            require(bool(_TOKEN_RE.match(self.storage_key)), "TensorRef.storage_key invalid")
        if self.content_digest is not None:
            require(
                bool(_DIGEST_RE.match(self.content_digest)),
                "TensorRef.content_digest must be 'sha256:<64 hex>'",
            )

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedTensor(Value):
    """A tensor leaf of a (possibly nested) input or output, with its pytree path."""

    path: str
    ref: TensorRef

    def _validate(self) -> None:
        require(bool(self.path), "NamedTensor.path must be non-empty")


class SiteIO(Enum):
    """Whether a site is a module's input (pre-hook) or output (forward hook)."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True, kw_only=True)
class Site(Value):
    """A location in the module tree.

    ``module`` is a dotted module path (``""`` is the root model); ``output_path``
    selects a tensor leaf inside tuple/dict inputs or outputs (``""`` = the value itself).
    Time (which call, which pass) is not part of a site.
    """

    module: str
    io: SiteIO = SiteIO.OUTPUT
    output_path: str = ""

    def _validate(self) -> None:
        if self.module:
            segments = self.module.split(".")
            require(
                all(_MODULE_SEGMENT_RE.match(s) for s in segments),
                f"Site.module is not a valid module path: {self.module!r}",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetSpec(Value):
    """What an output-level quantity is measured on, e.g. ``logit`` of class 3.

    ``metric`` is a registered metric name or ``custom:<name>``; ``params`` are its
    JSON parameters.
    """

    metric: str
    params: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(bool(_METRIC_RE.match(self.metric)), f"TargetSpec.metric invalid: {self.metric!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Estimand(Value):
    """What a quantity or claim is about (ADR-013).

    INSTANCE:
        ``sample_id`` identifies the single input; ``n == 1``.
    FINITE_SAMPLE:
        exactly the ``n`` inputs identified by ``sample_id``, summarised by
        ``aggregation`` (e.g. ``"mean"``). Makes no claim beyond them, whatever ``n`` is.
    POPULATION:
        the distribution named by ``population``, summarised by ``aggregation``.
        ``n`` (optional) is the number of measured inputs an estimate used, and
        ``sample_id`` (optional) identifies them.

    Scope is never inferred from ``n``.
    """

    scope: EstimandScope
    sample_id: str | None = None
    n: int | None = None
    aggregation: str | None = None
    population: str | None = None

    def _validate(self) -> None:
        scope = self.scope
        if scope is EstimandScope.INSTANCE:
            require(self.sample_id is not None, "INSTANCE estimand requires sample_id")
            require(self.n == 1, "INSTANCE estimand requires n == 1")
            require(self.aggregation is None, "INSTANCE estimand has no aggregation")
            require(self.population is None, "INSTANCE estimand has no population")
        elif scope is EstimandScope.FINITE_SAMPLE:
            require(self.sample_id is not None, "FINITE_SAMPLE estimand requires sample_id")
            require(self.n is not None and self.n >= 1, "FINITE_SAMPLE estimand requires n >= 1")
            require(self.aggregation is not None, "FINITE_SAMPLE estimand requires aggregation")
            require(self.population is None, "FINITE_SAMPLE estimand has no population")
        else:
            require(self.population is not None, "POPULATION estimand requires population")
            require(self.aggregation is not None, "POPULATION estimand requires aggregation")
            require(self.n is None or self.n >= 1, "POPULATION estimand n must be >= 1")
        for name in ("sample_id", "aggregation", "population"):
            value = getattr(self, name)
            if value is not None and not _TOKEN_RE.match(value):
                raise SchemaError(f"Estimand.{name} must be a non-empty token: {value!r}")

    @classmethod
    def instance(cls, sample_id: str) -> Estimand:
        return cls(scope=EstimandScope.INSTANCE, sample_id=sample_id, n=1)

    @classmethod
    def finite_sample(cls, sample_id: str, n: int, aggregation: str) -> Estimand:
        return cls(
            scope=EstimandScope.FINITE_SAMPLE, sample_id=sample_id, n=n, aggregation=aggregation
        )

    @classmethod
    def population_of(
        cls,
        population: str,
        aggregation: str,
        *,
        n: int | None = None,
        sample_id: str | None = None,
    ) -> Estimand:
        return cls(
            scope=EstimandScope.POPULATION,
            population=population,
            aggregation=aggregation,
            n=n,
            sample_id=sample_id,
        )

    def covers(self, evidence: Estimand) -> bool:
        """Whether evidence with estimand ``evidence`` is about exactly this estimand.

        INSTANCE and FINITE_SAMPLE require the same sample (and ``n``/aggregation);
        POPULATION requires the same population and aggregation.
        """
        if self.scope is not evidence.scope:
            return False
        if self.scope is EstimandScope.POPULATION:
            return (self.population, self.aggregation) == (
                evidence.population,
                evidence.aggregation,
            )
        return (self.sample_id, self.n, self.aggregation) == (
            evidence.sample_id,
            evidence.n,
            evidence.aggregation,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Subject(Value):
    """What a claim is about: a site, optionally restricted to units.

    ``units`` are indices along the last dimension of the site's tensor, stored
    sorted; ``None`` means the whole site. Feature/concept subjects arrive with
    their record types in later schema versions (ADR-014).
    """

    site: Site
    units: tuple[int, ...] | None = None

    def _validate(self) -> None:
        if self.units is not None:
            require(len(self.units) > 0, "Subject.units must be non-empty or None")
            require(all(u >= 0 for u in self.units), "Subject.units must be >= 0")
            require(len(set(self.units)) == len(self.units), "Subject.units has duplicates")
            object.__setattr__(self, "units", tuple(sorted(self.units)))


class ClaimSourceKind(Enum):
    """Who proposed a claim. The source is never evidence for the claim."""

    USER = "user"
    METHOD = "method"
    GENERATED = "generated"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimSource(Value):
    kind: ClaimSourceKind
    detail: str | None = None
