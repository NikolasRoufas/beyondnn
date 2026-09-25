import re
import subprocess
import sys
from pathlib import Path

import beyondnn


def test_version_is_pep440_dev_string() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", beyondnn.__version__)


def test_top_level_api_is_the_reviewed_surface() -> None:
    # Guards against internal or later-phase APIs becoming public before review.
    assert sorted(beyondnn.__all__) == sorted(
        [
            "EstimandScope",
            "EvidenceStatus",
            "Outcome",
            "Relation",
            "TraceResult",
            "Verdict",
            "__version__",
            "load_trace",
            "recording",
            "schema",
            "trace",
        ]
    )
    for name in (
        "instrument",
        "explain",
        "test_claim",
        "Study",
        "HookSession",
        "HookEvent",
        "resolve_sites",
        "walk",
        "tensor_ref",
    ):
        assert not hasattr(beyondnn, name)


def test_import_beyondnn_does_not_import_torch() -> None:
    code = "import sys, beyondnn, beyondnn.schema; print('torch' in sys.modules)"
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root
    )
    assert out.stdout.strip() == "False"


def test_tracing_names_load_lazily() -> None:
    from beyondnn.core.trace import TraceResult, recording, trace

    assert beyondnn.trace is trace
    assert beyondnn.recording is recording
    assert beyondnn.TraceResult is TraceResult
