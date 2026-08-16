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

from evalassay.types import (
    AuditReport,
    Component,
    Estimate,
    Finding,
    RunManifest,
    Verdict,
)


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
        "skipped_detectors": list(report.skipped_detectors),
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


# ---------------------------------------------------------------------------
# Reading a report back
# ---------------------------------------------------------------------------
#
# The inverse of the writers above, so a saved report can be re-opened and
# compared against another without re-running the audit. Keeping the pair
# exercised by a round-trip test also proves the JSON is lossless, which is what
# entitles the format to be called machine-readable rather than merely printable.


def _float(value: object) -> float:
    """Read a number that may have been written as null.

    Args:
        value: The serialised value.

    Returns:
        The number, or ``nan`` where it was undefined.
    """
    return math.nan if value is None else float(value)  # type: ignore[arg-type]


def estimate_from_dict(data: dict[str, Any]) -> Estimate:
    """Rebuild an estimate.

    Args:
        data: A serialised estimate.

    Returns:
        The estimate.
    """
    return Estimate(
        point=_float(data["point"]),
        ci_low=_float(data["ci_low"]),
        ci_high=_float(data["ci_high"]),
        p_value=_float(data["p_value"]),
        n=int(data["n"]),
        method=str(data["method"]),
    )


def manifest_from_dict(data: dict[str, Any]) -> RunManifest:
    """Rebuild a run manifest.

    Args:
        data: A serialised manifest.

    Returns:
        The manifest.
    """
    return RunManifest(
        schema_version=str(data["schema_version"]),
        corpus_name=str(data["corpus_name"]),
        corpus_hash=str(data["corpus_hash"]),
        n_items=int(data["n_items"]),
        scorer_id=str(data["scorer_id"]),
        scorer_deterministic=bool(data["scorer_deterministic"]),
        config_hash=str(data["config_hash"]),
        seed=int(data["seed"]),
        alpha=float(data["alpha"]),
        power=float(data["power"]),
        bootstrap_draws=int(data["bootstrap_draws"]),
        evalassay_version=str(data["evalassay_version"]),
        library_versions=tuple(
            sorted((str(k), str(v)) for k, v in data["library_versions"].items())
        ),
    )


def report_from_dict(data: dict[str, Any]) -> AuditReport:
    """Rebuild a whole audit from its serialised form.

    Args:
        data: A serialised report.

    Returns:
        The report.
    """
    components = tuple(
        Component(
            name=str(c["name"]),
            description=str(c["description"]),
            estimate=estimate_from_dict(c["estimate"]),
            verdict=Verdict(c["verdict"]),
            adjusted_p=_float(c["adjusted_p"]),
            mde=_float(c["minimum_detectable_effect"]),
        )
        for c in data["components"]
    )
    findings = tuple(
        Finding(
            detector=str(f["detector"]),
            description=str(f["description"]),
            estimate=estimate_from_dict(f["estimate"]),
            verdict=Verdict(f["verdict"]),
            adjusted_p=_float(f["adjusted_p"]),
            mde=_float(f["minimum_detectable_effect"]),
            detail=str(f["detail"]),
        )
        for f in data["findings"]
    )
    blind = data.get("blind_accuracy")
    return AuditReport(
        manifest=manifest_from_dict(data["manifest"]),
        reported_score=_float(data["reported_score"]),
        total_drop=_float(data["total_drop"]),
        components=components,
        findings=findings,
        chance_accuracy=_float(data["chance_accuracy"]),
        blind_accuracy=estimate_from_dict(blind) if blind else None,
        # Read with a default so reports written before this field existed still
        # open. Absent is not the same as empty, but a report that predates the
        # field carries no answer either way and refusing to load it would lose
        # everything else it does say.
        skipped_detectors=tuple(str(name) for name in data.get("skipped_detectors", ())),
    )


def from_json(text: str) -> AuditReport:
    """Rebuild an audit from JSON text.

    Args:
        text: Serialised report.

    Returns:
        The report.
    """
    parsed: dict[str, Any] = json.loads(text)
    return report_from_dict(parsed)
