"""Scoring backends and the interface they share.

Three backends implement one protocol:

- :class:`~evalassay.score.oracle.OracleScorer` is a simulated model with
  artifacts dialled in at known magnitudes. It is the instrument's reference
  standard, and it needs no compute at all.
- :class:`~evalassay.score.local.LocalScorer` scores options by exact
  log-likelihood under a locally loaded model. Deterministic.
- :class:`~evalassay.score.api.ApiScorer` asks a hosted model to choose. Not
  deterministic, and it says so.

No backend may read ``answer_index``. The contract is enforced by a test, not
left to good intentions.
"""

from __future__ import annotations

from evalassay.score.base import (
    ScoreCache,
    Scorer,
    break_ties,
    is_correct,
    predict,
)
from evalassay.score.oracle import OracleScorer, OracleSpec

__all__ = [
    "OracleScorer",
    "OracleSpec",
    "ScoreCache",
    "Scorer",
    "break_ties",
    "is_correct",
    "predict",
]
