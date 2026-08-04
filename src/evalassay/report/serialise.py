"""Serialise an audit to JSON that another program can check.

The JSON carries everything the text report shows and everything it does not:
unadjusted and adjusted p-values, intervals, the refusal reason for every
quantity the gate declined, and the full manifest. A reader who disagrees with
the gate's thresholds can therefore re-apply their own without re-running the
audit.

Verdicts are written out as words rather than as a boolean. A field named
``established`` invites being read as ``not established means refuted``, and the
distinction between "we looked and found nothing" and "we could not look" is the
one this project is most concerned with keeping.
"""

from __future__ import annotations

import json
import math
from typing import Any

from evalassay.types import AuditReport, Component, Estimate, Finding, RunManifest


def _number(value: float) -> float | None:
    """Convert a float to something JSON can represent.

    Args:
        value: The value.

    Returns:
        The value, or ``None`` where it is not a finite number. JSON has no
        NaN, and emitting the non-standard literal would break strict parsers.
    """
    return None if math.isnan(value) or math.isinf(value) else value


def estimate_to_dict(estimate: Estimate) -> dict[str, Any]:
    """Serialise an estimate.

    Args:
        estimate: The estimate.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "point": _number(estimate.point),
        "ci_low": _number(estimate.ci_low),
        "ci_high": _number(estimate.ci_high),
        "p_value": _number(estimate.p_value),
        "n": estimate.n,
        "method": estimate.method,
    }


def component_to_dict(component: Component) -> dict[str, Any]:
    """Serialise one artifact share.

    Args:
        component: The component.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "name": component.name,
        "description": component.description,
        "verdict": component.verdict.value,
        "estimate": estimate_to_dict(component.estimate),
        "adjusted_p": _number(component.adjusted_p),
        "minimum_detectable_effect": _number(component.mde),
        "attributed_points": _number(component.attributed_points),
    }


def finding_to_dict(finding: Finding) -> dict[str, Any]:
    """Serialise one model-free corpus finding.

    Args:
        finding: The finding.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "detector": finding.detector,
        "description": finding.description,
        "verdict": finding.verdict.value,
        "estimate": estimate_to_dict(finding.estimate),
        "adjusted_p": _number(finding.adjusted_p),
        "minimum_detectable_effect": _number(finding.mde),
        "detail": finding.detail,
    }


def manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    """Serialise the run manifest.

    Args:
        manifest: The manifest.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "schema_version": manifest.schema_version,
        "corpus_name": manifest.corpus_name,
        "corpus_hash": manifest.corpus_hash,
        "n_items": manifest.n_items,
        "scorer_id": manifest.scorer_id,
        "scorer_deterministic": manifest.scorer_deterministic,
        "config_hash": manifest.config_hash,
        "seed": manifest.seed,
        "alpha": manifest.alpha,
        "power": manifest.power,
        "bootstrap_draws": manifest.bootstrap_draws,
        "evalassay_version": manifest.evalassay_version,
        "library_versions": dict(manifest.library_versions),
    }


def report_to_dict(report: AuditReport) -> dict[str, Any]:
    """Serialise a whole audit.

    Args:
        report: The audit result.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "manifest": manifest_to_dict(report.manifest),
        "reported_score": _number(report.reported_score),
        "chance_accuracy": _number(report.chance_accuracy),
        "total_drop": _number(report.total_drop),
        "attributed_points": _number(report.attributed_points),
        "assayed_score": _number(report.assayed_score),
        "purity": _number(report.purity),
        "components": [component_to_dict(c) for c in report.components],
        "blind_accuracy": (
            estimate_to_dict(report.blind_accuracy) if report.blind_accuracy else None
        ),
        "findings": [finding_to_dict(f) for f in report.findings],
    }


def to_json(report: AuditReport, *, indent: int = 2) -> str:
    """Serialise an audit to a JSON string.

    Keys are sorted so two runs of the same audit produce byte-identical output,
    which is what makes the reproducibility claim checkable with ``diff``.

    Args:
        report: The audit result.
        indent: Indentation, or ``0`` for the compact form.

    Returns:
        The JSON text.
    """
    return json.dumps(
        report_to_dict(report),
        indent=indent if indent > 0 else None,
        sort_keys=True,
        ensure_ascii=True,
    )
