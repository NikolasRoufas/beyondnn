"""Phase-1 explanation interface: ``instrument()`` and the INPUT -> WHY -> OUTPUT view.

Phase-1 WHY answers "what internal evidence was measured while this output was
produced?" It does not answer "which internal state caused the output?".
"""

from .handle import Instrumented, instrument
from .response import EXPLANATION_LIMITATIONS, ExplainResponse, Why

__all__ = ["EXPLANATION_LIMITATIONS", "ExplainResponse", "Instrumented", "Why", "instrument"]
