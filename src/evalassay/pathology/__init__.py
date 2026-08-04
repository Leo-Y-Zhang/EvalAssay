"""Model-free detectors for defects in the benchmark itself.

Nothing in this package loads or calls a model. These measurements describe the
corpus, so they cost only arithmetic and can be run by someone with no compute
at all - and a benchmark whose answer key leaks through surface form is
defective regardless of which model is pointed at it.

Findings from this layer therefore sit outside the Shapley decomposition and
never move the assayed score. They are a statement about the ruler, not about
the thing being measured.
"""

from __future__ import annotations

from evalassay.pathology.base import (
    Detector,
    RawFinding,
    tokenise,
    wilson_interval,
)
from evalassay.pathology.choices_only import ChoicesOnly
from evalassay.pathology.longest_answer import LongestAnswer
from evalassay.pathology.near_duplicate import (
    JACCARD_THRESHOLD,
    NearDuplicate,
    contradiction_groups,
    jaccard,
    shingles,
)
from evalassay.pathology.position_skew import PositionSkew, total_variation
from evalassay.pathology.runner import (
    PathologyReport,
    default_detectors,
    run_all,
)

__all__ = [
    "JACCARD_THRESHOLD",
    "ChoicesOnly",
    "Detector",
    "LongestAnswer",
    "NearDuplicate",
    "PathologyReport",
    "PositionSkew",
    "RawFinding",
    "contradiction_groups",
    "default_detectors",
    "jaccard",
    "run_all",
    "shingles",
    "tokenise",
    "total_variation",
    "wilson_interval",
]
