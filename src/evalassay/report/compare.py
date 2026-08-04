"""Compare two audits, and refuse when they are not comparable.

Two reported scores are worth subtracting only when everything except the one
thing under study was held fixed. That is easy to say and easy to get wrong: a
score difference between two runs on subtly different item sets, or under
different thresholds, is a number about the difference in setup rather than
about the model.

The manifests exist precisely so that the check can be mechanical. This module
refuses to compare two runs whose corpus content hash or configuration hash
disagree, and names which one differs, rather than producing a plausible
difference that means nothing.

The intended use is measuring how much of a score is presentation rather than
capability: audit one model on one corpus twice, changing only how the question
is put, and compare.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalassay.types import AuditReport, Verdict


class NotComparableError(ValueError):
    """Raised when two audits differ in something that must have been held fixed."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """The difference between two audits of the same corpus.

    Attributes:
        baseline_id: Scorer identity of the run treated as the baseline.
        variant_id: Scorer identity of the run compared against it.
        corpus_name: The corpus both were run on.
        n_items: Item count both consumed.
        reported_delta: Variant reported score minus baseline reported score.
        assayed_delta: The same difference in assayed capability.
        baseline_reported: The baseline's reported score.
        variant_reported: The variant's reported score.
        baseline_assayed: The baseline's assayed capability.
        variant_assayed: The variant's assayed capability.
        charged_only_in_baseline: Artifacts established in the baseline alone.
        charged_only_in_variant: Artifacts established in the variant alone.
    """

    baseline_id: str
    variant_id: str
    corpus_name: str
    n_items: int
    reported_delta: float
    assayed_delta: float
    baseline_reported: float
    variant_reported: float
    baseline_assayed: float
    variant_assayed: float
    charged_only_in_baseline: tuple[str, ...]
    charged_only_in_variant: tuple[str, ...]


def _charged(report: AuditReport) -> set[str]:
    """Names of the artifacts a report established.

    Args:
        report: An audit result.

    Returns:
        The established component names.
    """
    return {c.name for c in report.components if c.verdict is Verdict.ESTABLISHED}


def compare(baseline: AuditReport, variant: AuditReport) -> Comparison:
    """Difference two audits that held everything but one factor fixed.

    Args:
        baseline: The run to subtract from.
        variant: The run to compare against it.

    Returns:
        The comparison.

    Raises:
        NotComparableError: If the two runs consumed different corpora or ran
            under different thresholds. The difference would then be a statement
            about the setup rather than about the factor under study, and the
            manifests are carried precisely so this can be caught rather than
            published.
    """
    if baseline.manifest.corpus_hash != variant.manifest.corpus_hash:
        raise NotComparableError(
            "the two runs consumed different corpora "
            f"({baseline.manifest.corpus_hash} against {variant.manifest.corpus_hash}); "
            "any difference would be about the items, not about the model"
        )
    if baseline.manifest.config_hash != variant.manifest.config_hash:
        raise NotComparableError(
            "the two runs used different thresholds "
            f"({baseline.manifest.config_hash} against {variant.manifest.config_hash}); "
            "hold the configuration fixed before differencing them"
        )
    if baseline.manifest.scorer_id == variant.manifest.scorer_id:
        raise NotComparableError(
            f"both runs used the same scorer ({baseline.manifest.scorer_id}); "
            "there is nothing to compare"
        )

    baseline_charged = _charged(baseline)
    variant_charged = _charged(variant)

    return Comparison(
        baseline_id=baseline.manifest.scorer_id,
        variant_id=variant.manifest.scorer_id,
        corpus_name=baseline.manifest.corpus_name,
        n_items=baseline.manifest.n_items,
        reported_delta=variant.reported_score - baseline.reported_score,
        assayed_delta=variant.assayed_score - baseline.assayed_score,
        baseline_reported=baseline.reported_score,
        variant_reported=variant.reported_score,
        baseline_assayed=baseline.assayed_score,
        variant_assayed=variant.assayed_score,
        charged_only_in_baseline=tuple(sorted(baseline_charged - variant_charged)),
        charged_only_in_variant=tuple(sorted(variant_charged - baseline_charged)),
    )


def render_comparison(comparison: Comparison) -> str:
    """Render a comparison as text.

    Args:
        comparison: The comparison.

    Returns:
        The rendered text.
    """
    width = 78
    rule = "-" * width
    lines = [
        "EvalAssay comparison",
        "=" * width,
        f"corpus   {comparison.corpus_name}  ({comparison.n_items} items, identical in both)",
        f"baseline {comparison.baseline_id}",
        f"variant  {comparison.variant_id}",
        "",
        f"{'':<28}{'baseline':>12}{'variant':>12}{'difference':>14}",
        rule,
        f"{'reported score':<28}{comparison.baseline_reported:>12.4f}"
        f"{comparison.variant_reported:>12.4f}{comparison.reported_delta:>+14.4f}",
        f"{'assayed capability':<28}{comparison.baseline_assayed:>12.4f}"
        f"{comparison.variant_assayed:>12.4f}{comparison.assayed_delta:>+14.4f}",
        rule,
    ]

    if comparison.charged_only_in_baseline:
        lines.append(
            f"  charged only in baseline: {', '.join(comparison.charged_only_in_baseline)}"
        )
    if comparison.charged_only_in_variant:
        lines.append(f"  charged only in variant:  {', '.join(comparison.charged_only_in_variant)}")
    if not (comparison.charged_only_in_baseline or comparison.charged_only_in_variant):
        lines.append("  the same artifacts were established in both runs")

    lines.append("")
    lines.append("  The corpus and thresholds were identical, so the difference is")
    lines.append("  attributable to whatever distinguishes the two scorers above.")
    lines.append("")
    return "\n".join(lines)
