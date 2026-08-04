"""Rendering an audit for a person and for a program.

The text form is for reading; the JSON form is for checking. The JSON carries
strictly more than the text - unadjusted p-values, refusal reasons, the full
manifest - so a reader who disagrees with the gate's thresholds can re-apply
their own without re-running anything.
"""

from __future__ import annotations

from evalassay.report.compare import (
    Comparison,
    NotComparableError,
    compare,
    render_comparison,
)
from evalassay.report.render import render, render_blind, render_decomposition, render_findings
from evalassay.report.serialise import from_json, report_from_dict, report_to_dict, to_json

__all__ = [
    "Comparison",
    "NotComparableError",
    "compare",
    "from_json",
    "render",
    "render_blind",
    "render_comparison",
    "render_decomposition",
    "render_findings",
    "report_from_dict",
    "report_to_dict",
    "to_json",
]
