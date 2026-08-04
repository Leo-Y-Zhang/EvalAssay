"""Paired interventions whose joint effect the audit decomposes.

The interventions are the players in the cooperative game solved in
:mod:`evalassay.stats.shapley`. Everything here is deterministic given the run
seed, and every intervention preserves the option count, so chance accuracy is
identical in every condition.
"""

from __future__ import annotations

from evalassay.intervene.base import (
    CANONICAL_ORDER,
    Intervention,
    apply_coalition,
    intervention_rng,
    order_key,
)
from evalassay.intervene.interventions import (
    REFRAME_PREFIX,
    HideQuestion,
    NeutralReframing,
    PermuteOptions,
    StrongerDistractor,
    distractor_pool,
)

__all__ = [
    "CANONICAL_ORDER",
    "REFRAME_PREFIX",
    "HideQuestion",
    "Intervention",
    "NeutralReframing",
    "PermuteOptions",
    "StrongerDistractor",
    "apply_coalition",
    "distractor_pool",
    "intervention_rng",
    "order_key",
]
