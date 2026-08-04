"""EvalAssay: measure how much of a reported benchmark score is capability.

An assay is the test that determines how much of an ore is actually the metal.
This package does the same to a leaderboard number: it applies controlled,
paired interventions to a multiple-choice benchmark, attributes the resulting
accuracy loss among named artifacts using Shapley values, and refuses to report
any attribution that does not clear a pre-registered significance threshold.

The headline output is *purity*: the fraction of a reported score that survives
the audit.
"""

from __future__ import annotations

from evalassay.types import (
    SCHEMA_VERSION,
    AuditReport,
    Component,
    Estimate,
    Finding,
    Item,
    ItemSet,
    RunManifest,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AuditReport",
    "Component",
    "Estimate",
    "Finding",
    "Item",
    "ItemSet",
    "RunManifest",
    "Verdict",
    "__version__",
]
