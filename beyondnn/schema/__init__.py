"""The BeyondNN trace schema (schema version 0.1).

Pure data: immutable records with content-derived ids, explicit epistemic
status, lineage, and strict versioned serialisation. Importing this package does
not import ``torch``.

Stable in 0.1 (ADR-014): the record mechanism and envelope, ``InputRecord``,
``OutputRecord``, ``ActivationRecord``, ``TensorRef``, ``TraceLimitation``, and the
claim types (ADR-012), and the provenance types (M1.2, ADR-019). Later-phase record
kinds arrive in later schema versions.
"""

from ._canonical import JsonMap
from .base import BaseRecord, EvidenceRef, RecordRef, verify_ref
from .claims import (
    Assessment,
    AssessmentPolicy,
    Claim,
    ClaimRef,
    ClaimTestResult,
    ClaimTestSpec,
    PolicyRequirement,
    ResultRef,
    SpecRef,
    derive_verdict,
)
from .codec import SCHEMA_VERSION, from_dict, from_json, to_dict, to_json
from .errors import (
    DecodeError,
    EvidenceRuleError,
    IntegrityError,
    SchemaError,
    SchemaTypeError,
    UnknownLimitationCodeError,
    UnknownRecordKindError,
    UnsupportedVersionError,
)
from .limitations import LIMITATIONS, LimitationDef, Severity, TraceLimitation
from .provenance import (
    EnvironmentIdentity,
    ExecutionContext,
    ExecutionMode,
    ExecutionOccurrence,
    FingerprintMethod,
    MethodIdentity,
    ModelIdentity,
    ProvenanceRecord,
    Randomness,
)
from .records import ActivationRecord, InputRecord, OutputRecord
from .status import (
    ALLOWED_PARENT_STATUSES,
    CAUSAL_EVIDENCE_STATUSES,
    CAUSAL_RELATIONS,
    EstimandScope,
    EvidenceStatus,
    Outcome,
    Relation,
    Verdict,
)
from .values import (
    ClaimSource,
    ClaimSourceKind,
    Estimand,
    NamedTensor,
    Site,
    SiteIO,
    Subject,
    TargetSpec,
    TensorRef,
    TensorStats,
)

__all__ = [
    "ALLOWED_PARENT_STATUSES",
    "CAUSAL_EVIDENCE_STATUSES",
    "CAUSAL_RELATIONS",
    "LIMITATIONS",
    "SCHEMA_VERSION",
    "ActivationRecord",
    "Assessment",
    "AssessmentPolicy",
    "BaseRecord",
    "Claim",
    "ClaimRef",
    "ClaimSource",
    "ClaimSourceKind",
    "ClaimTestResult",
    "ClaimTestSpec",
    "DecodeError",
    "EnvironmentIdentity",
    "Estimand",
    "EstimandScope",
    "EvidenceRef",
    "EvidenceRuleError",
    "EvidenceStatus",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionOccurrence",
    "FingerprintMethod",
    "InputRecord",
    "IntegrityError",
    "JsonMap",
    "LimitationDef",
    "MethodIdentity",
    "ModelIdentity",
    "NamedTensor",
    "Outcome",
    "OutputRecord",
    "PolicyRequirement",
    "ProvenanceRecord",
    "Randomness",
    "RecordRef",
    "Relation",
    "ResultRef",
    "SchemaError",
    "SchemaTypeError",
    "Severity",
    "Site",
    "SiteIO",
    "SpecRef",
    "Subject",
    "TargetSpec",
    "TensorRef",
    "TensorStats",
    "TraceLimitation",
    "UnknownLimitationCodeError",
    "UnknownRecordKindError",
    "UnsupportedVersionError",
    "Verdict",
    "derive_verdict",
    "from_dict",
    "from_json",
    "to_dict",
    "to_json",
    "verify_ref",
]
