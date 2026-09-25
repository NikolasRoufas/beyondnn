"""Provenance collection: model fingerprints, environment, RNG, provenance records.

Depends on ``torch`` and on :mod:`beyondnn.schema` (never the reverse). The data
types themselves live in :mod:`beyondnn.schema`.
"""

from .collect import capture_cpu_rng, collect_environment, make_provenance, record_occurrence
from .fingerprint import ALGORITHM_VERSION, FingerprintError, fingerprint_model

__all__ = [
    "ALGORITHM_VERSION",
    "FingerprintError",
    "capture_cpu_rng",
    "collect_environment",
    "fingerprint_model",
    "make_provenance",
    "record_occurrence",
]
