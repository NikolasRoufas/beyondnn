import re

import beyondnn


def test_version_is_pep440_dev_string() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", beyondnn.__version__)


def test_top_level_api_is_only_the_m1_1_surface() -> None:
    # Guards against later-phase APIs becoming importable before they are reviewed.
    assert sorted(beyondnn.__all__) == sorted(
        [
            "EstimandScope",
            "EvidenceStatus",
            "Outcome",
            "Relation",
            "Verdict",
            "__version__",
            "schema",
        ]
    )
    for name in ("trace", "recording", "instrument", "explain", "test_claim", "Study"):
        assert not hasattr(beyondnn, name)
