"""Render an audit as text a person can read and check.

Two rules govern the layout.

**A number that was not established is never printed as a number.** It shows a
dash, the reason the gate refused it, and the smallest effect the run could have
detected. Printing an unestablished estimate in the same column as an
established one, differently styled, invites exactly the misreading this tool
exists to prevent.

**Provenance sits above the result, not in an appendix.** The corpus hash, the
model identity, the seed and the thresholds are what make the number checkable,
so they are the first thing on the page.
"""

from __future__ import annotations

import math
from typing import Final

from evalassay.types import AuditReport, Verdict

WIDTH: Final = 78
"""Fixed width, so a report pasted into a document keeps its shape."""

RULE: Final = "-" * WIDTH

BLIND_EXCESS_TO_REMARK_ON: Final = 0.02
"""Blind accuracy this far above chance earns a sentence of interpretation.

Below it the excess is within the noise of a few hundred items, and adding a
confident sentence to a number that small would be exactly the over-reading
this tool exists to discourage.
"""


def _percent(value: float) -> str:
    """Format a proportion as a percentage.

    Args:
        value: A proportion.

    Returns:
        The formatted percentage, or ``n/a`` if undefined.
    """
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _short(digest: str, keep: int = 12) -> str:
    """Abbreviate a digest for display, keeping the prefix.

    Args:
        digest: A prefixed hex digest.
        keep: How many hex characters to show.

    Returns:
        The abbreviated digest.
    """
    prefix, _, body = digest.partition(":")
    return f"{prefix}:{body[:keep]}" if body else digest


def render_header(report: AuditReport) -> list[str]:
    """Render the provenance block.

    Args:
        report: The audit result.

    Returns:
        Lines of text.
    """
    manifest = report.manifest
    determinism = "deterministic" if manifest.scorer_deterministic else "NOT deterministic"
    lines = [
        "EvalAssay report",
        "=" * WIDTH,
        f"corpus   {manifest.corpus_name}",
        f"         {_short(manifest.corpus_hash)}  {manifest.n_items} items",
        f"model    {manifest.scorer_id}  ({determinism})",
        f"config   alpha={manifest.alpha:g}  power={manifest.power:g}  "
        f"bootstrap={manifest.bootstrap_draws}  seed={manifest.seed}",
        f"         {_short(manifest.config_hash)}",
        f"version  evalassay {manifest.evalassay_version}  "
        + "  ".join(f"{name} {value}" for name, value in manifest.library_versions),
    ]
    if not manifest.scorer_deterministic:
        lines.append("         NOTE: this backend cannot promise an identical rerun, so the")
        lines.append("         manifest below does not certify reproducibility.")
    return lines


def render_decomposition(report: AuditReport) -> list[str]:
    """Render the score, the artifact shares, and the assayed capability.

    Args:
        report: The audit result.

    Returns:
        Lines of text.
    """
    lines = [
        "",
        f"{'Reported score':<44}{report.reported_score:>10.4f}",
        f"{'Chance (uniform guessing)':<44}{report.chance_accuracy:>10.4f}",
        RULE,
    ]

    for component in report.components:
        if component.verdict is Verdict.ESTABLISHED:
            estimate = component.estimate
            lines.append(
                f"  {component.name:<26}{-estimate.point:>9.4f}  "
                f"[{estimate.ci_low:.4f}, {estimate.ci_high:.4f}]  charged"
            )
        else:
            lines.append(
                f"  {component.name:<26}{'-':>9}  not established (MDE {component.mde:.4f})"
            )
            reason = component.description.partition("not established: ")[2]
            if reason:
                lines.append(f"  {'':<26}{'':>9}  {reason}")

    lines.extend(
        [
            RULE,
            f"{'  Assayed capability':<44}{report.assayed_score:>10.4f}",
            f"{'  Purity (share of the score that survived)':<44}{_percent(report.purity):>10}",
        ]
    )
    return lines


def render_blind(report: AuditReport) -> list[str]:
    """Render the hidden-question diagnostic.

    Args:
        report: The audit result.

    Returns:
        Lines of text, empty if the diagnostic was not run.
    """
    blind = report.blind_accuracy
    if blind is None:
        return []

    excess = blind.point - report.chance_accuracy
    lines = [
        "",
        "Blind accuracy - the question removed entirely",
        RULE,
        f"  {'accuracy with no question':<26}{blind.point:>9.4f}  "
        f"[{blind.ci_low:.4f}, {blind.ci_high:.4f}]",
        f"  {'above chance by':<26}{excess:>9.4f}",
    ]

    # The interpretation is only stated when the interval clears chance. A point
    # estimate above chance with an interval straddling it is exactly the
    # over-reading this tool objects to, and printing the sentence anyway would
    # have the report commit the error it exists to detect.
    clears_chance = blind.ci_low > report.chance_accuracy
    if clears_chance and excess > BLIND_EXCESS_TO_REMARK_ON:
        lines.append("  On those items the model is not answering the question, because there")
        lines.append("  is no question. This is a floor, not part of the decomposition.")
    elif excess > BLIND_EXCESS_TO_REMARK_ON:
        lines.append("  The interval includes chance, so this is not established: the model")
        lines.append("  may or may not be scoring above chance without the question.")
    return lines


def render_findings(report: AuditReport) -> list[str]:
    """Render the model-free corpus findings.

    Args:
        report: The audit result.

    Returns:
        Lines of text, empty if the layer did not run.
    """
    if not report.findings:
        return []

    lines = ["", "Benchmark defects - measured without any model", RULE]
    for finding in report.findings:
        if finding.verdict is Verdict.ESTABLISHED:
            estimate = finding.estimate
            lines.append(
                f"  {finding.detector:<26}{estimate.point:>9.4f}  "
                f"[{estimate.ci_low:.4f}, {estimate.ci_high:.4f}]"
            )
            lines.append(f"  {'':<26}{'':>9}  {finding.detail}")
        else:
            lines.append(
                f"  {finding.detector:<26}{'-':>9}  not established (MDE {finding.mde:.4f})"
            )
    lines.append("")
    lines.append("  These describe the benchmark, not the model, so they do not move the")
    lines.append("  assayed score above.")
    return lines


def render(report: AuditReport) -> str:
    """Render a complete audit report as text.

    Args:
        report: The audit result.

    Returns:
        The report.
    """
    lines: list[str] = []
    lines.extend(render_header(report))
    lines.extend(render_decomposition(report))
    lines.extend(render_blind(report))
    lines.extend(render_findings(report))
    lines.append("")
    return "\n".join(lines)
