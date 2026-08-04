"""Statistical machinery for the audit.

The submodules are deliberately independent of anything to do with language
models: they take per-item correctness vectors and return estimates. That
separation is what lets the calibration harness point the same statistics at a
synthetic model whose artifacts are known exactly.
"""

from __future__ import annotations

from evalassay.stats.decision import Decision, GateConfig, decide
from evalassay.stats.multiplicity import holm_bonferroni
from evalassay.stats.paired import (
    McNemarResult,
    bca_ci,
    bootstrap_counts,
    mcnemar_exact,
    minimum_detectable_effect,
    percentile_ci,
)
from evalassay.stats.shapley import (
    MAX_PLAYERS,
    coalition_accuracies,
    shapley_bootstrap,
    shapley_jackknife,
    shapley_operator,
    shapley_shares,
    total_drop,
)

__all__ = [
    "MAX_PLAYERS",
    "Decision",
    "GateConfig",
    "McNemarResult",
    "bca_ci",
    "bootstrap_counts",
    "coalition_accuracies",
    "decide",
    "holm_bonferroni",
    "mcnemar_exact",
    "minimum_detectable_effect",
    "percentile_ci",
    "shapley_bootstrap",
    "shapley_jackknife",
    "shapley_operator",
    "shapley_shares",
    "total_drop",
]
