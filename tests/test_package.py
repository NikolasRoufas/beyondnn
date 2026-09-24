import re

import beyondnn


def test_version_is_pep440_dev_string() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", beyondnn.__version__)


def test_no_public_api_before_phase_1() -> None:
    # Guard against implementation landing before the architecture review.
    assert beyondnn.__all__ == ["__version__"]
