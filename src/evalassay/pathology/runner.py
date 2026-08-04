"""Run the detector family and apply family-wise correction across it.

Detectors do not judge themselves. Each returns a raw effect and an unadjusted
p-value; this module corrects across the whole family and applies the
default-deny gate, so the significance threshold accounts for how many questions
were actually asked.

Each detector is given a generator derived from the run seed *and its own name*.
That means adding a detector, or skipping one, cannot change the random draws
any other detector makes - so a result stays reproducible as the family grows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evalassay.pathology.base import Detector, RawFinding
from evalassay.pathology.choices_only import ChoicesOnly
from evalassay.pathology.longest_answer import LongestAnswer
from evalassay.pathology.near_duplicate import NearDuplicate, deduplicated
from evalassay.pathology.position_skew import PositionSkew
from evalassay.stats.decision import GateConfig, decide
from evalassay.stats.multiplicity import holm_bonferroni
from evalassay.types import Finding, ItemSet, Verdict

_SEED_MODULUS = 2**32


def default_detectors() -> tuple[Detector, ...]:
    """The detector family run unless a caller supplies its own.

    Returns:
        One instance of each detector, in a fixed order.
    """
    family: tuple[Detector, ...] = (
        PositionSkew(),
        LongestAnswer(),
        ChoicesOnly(),
        NearDuplicate(),
    )
    return family


@dataclass(frozen=True, slots=True)
class PathologyReport:
    """Findings from the model-free layer, plus what could not be measured.

    Attributes:
        findings: One finding per detector that ran, gated and corrected.
        skipped: Names of detectors that declined to run, usually because the
            corpus was too small. Recorded rather than dropped, because a
            question that was never asked must not read as a question that was
            asked and answered negatively.
    """

    findings: tuple[Finding, ...]
    skipped: tuple[str, ...]
    duplicates_removed: int = 0
    """Exact repeats withheld from the detectors that assume independent items."""

    @property
    def established(self) -> tuple[Finding, ...]:
        """Findings that cleared the default-deny gate."""
        return tuple(f for f in self.findings if f.verdict is Verdict.ESTABLISHED)


def _detector_rng(seed: int, name: str) -> np.random.Generator:
    """Derive a detector's generator from the run seed and the detector name.

    Args:
        seed: Run seed.
        name: Detector name.

    Returns:
        A seeded generator unique to this detector and run.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return np.random.default_rng((seed + offset) % _SEED_MODULUS)


def run_all(
    item_set: ItemSet,
    seed: int,
    config: GateConfig | None = None,
    detectors: Sequence[Detector] | None = None,
) -> PathologyReport:
    """Run every detector and gate the results as one family.

    Args:
        item_set: The corpus to inspect.
        seed: Run seed.
        config: Pre-registered thresholds. Defaults to :class:`GateConfig`.
        detectors: Detector family. Defaults to :func:`default_detectors`.

    Returns:
        The gated findings and the names of any detectors that declined to run.
    """
    gate = config if config is not None else GateConfig()
    family = tuple(detectors) if detectors is not None else default_detectors()

    # Exact repeats are not second observations. A detector whose inference
    # treats items as independent draws is handed the deduplicated corpus, or
    # its effective sample size would be smaller than its nominal one and every
    # test built on it would be anti-conservative. Measured, not theoretical: a
    # corpus with a tenth of its items duplicated made the position-skew test
    # report a significant result on a key that was in fact uniform.
    unique, duplicates_removed = deduplicated(item_set)

    raw: list[RawFinding] = []
    skipped: list[str] = []
    for detector in family:
        target = unique if detector.assumes_independent_items else item_set
        result = detector.run(target, _detector_rng(seed, detector.name))
        if result is None:
            skipped.append(detector.name)
        else:
            raw.append(result)

    if not raw:
        return PathologyReport(
            findings=(), skipped=tuple(skipped), duplicates_removed=duplicates_removed
        )

    adjusted = holm_bonferroni([finding.estimate.p_value for finding in raw])
    findings = tuple(
        Finding(
            detector=finding.detector,
            description=finding.description,
            estimate=finding.estimate,
            verdict=decide(finding.estimate, adjusted_p, gate).verdict,
            adjusted_p=adjusted_p,
            detail=_with_reason(finding, adjusted_p, gate),
        )
        for finding, adjusted_p in zip(raw, adjusted, strict=True)
    )
    return PathologyReport(
        findings=findings, skipped=tuple(skipped), duplicates_removed=duplicates_removed
    )


def _with_reason(finding: RawFinding, adjusted_p: float, gate: GateConfig) -> str:
    """Append the gate's refusal reason to a finding's detail.

    Args:
        finding: The raw finding.
        adjusted_p: Family-wise adjusted p-value.
        gate: Pre-registered thresholds.

    Returns:
        The detail text, with the reason appended when the gate refused.
    """
    decision = decide(finding.estimate, adjusted_p, gate)
    if decision.established:
        return finding.detail
    if not finding.detail:
        return f"not established: {decision.reason}"
    return f"{finding.detail}; not established: {decision.reason}"
