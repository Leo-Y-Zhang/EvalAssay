"""Value types shared across the audit pipeline.

Every type here is frozen. An audit claims to be a pure function of its inputs,
and :mod:`evalassay.report.manifest` can only honour that claim if the values
flowing through the pipeline cannot be mutated behind its back.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

MIN_CHOICES: Final = 2
"""Fewest choices an item may carry and still be a multiple-choice item."""

SCHEMA_VERSION: Final = "1"
"""Bumped whenever the on-disk report or manifest layout changes incompatibly."""


class Verdict(enum.StrEnum):
    """Outcome of the default-deny gate for a single measured quantity.

    The gate exists because a decomposition that always prints four numbers is
    indistinguishable from one that prints four numbers at random. A quantity
    that does not clear its pre-registered threshold is reported as
    ``NOT_ESTABLISHED`` and contributes nothing to the assayed score.
    """

    ESTABLISHED = "established"
    NOT_ESTABLISHED = "not_established"


@dataclass(frozen=True, slots=True)
class Item:
    """One multiple-choice benchmark item.

    Attributes:
        item_id: Stable identifier, unique within an :class:`ItemSet`.
        question: The question stem. May be empty: the hide-question
            intervention deliberately produces items with no stem, to measure
            how much of a score survives when the question is removed entirely.
        choices: The answer options, in presentation order.
        answer_index: Index into ``choices`` of the single correct option.
        subject: Optional grouping label (for example an MMLU subject).
    """

    item_id: str
    question: str
    choices: tuple[str, ...]
    answer_index: int
    subject: str = ""

    def __post_init__(self) -> None:
        """Reject items that cannot be scored.

        Raises:
            ValueError: If the identifier is blank, there are too few choices,
                or the answer index does not point at a choice.
        """
        if not self.item_id:
            raise ValueError("item_id must be a non-empty string")
        if len(self.choices) < MIN_CHOICES:
            raise ValueError(
                f"item {self.item_id!r}: need at least {MIN_CHOICES} choices, "
                f"got {len(self.choices)}"
            )
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError(
                f"item {self.item_id!r}: answer_index {self.answer_index} "
                f"outside range 0..{len(self.choices) - 1}"
            )

    @property
    def answer(self) -> str:
        """The text of the correct choice."""
        return self.choices[self.answer_index]

    @property
    def n_choices(self) -> int:
        """How many options this item offers."""
        return len(self.choices)


@dataclass(frozen=True, slots=True)
class ItemSet:
    """An ordered, duplicate-free collection of items drawn from one benchmark.

    Attributes:
        name: Corpus name, used in reports and in the run manifest.
        items: The items, in a fixed order. Order is part of the identity of
            the set because it feeds the content hash.
    """

    name: str
    items: tuple[Item, ...]

    def __post_init__(self) -> None:
        """Reject empty or internally inconsistent sets.

        Raises:
            ValueError: If the set is empty or contains a repeated item id.
        """
        if not self.items:
            raise ValueError(f"corpus {self.name!r} is empty")
        seen: set[str] = set()
        for item in self.items:
            if item.item_id in seen:
                raise ValueError(f"corpus {self.name!r}: duplicate item_id {item.item_id!r}")
            seen.add(item.item_id)

    def __len__(self) -> int:
        """Number of items in the set."""
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        """Iterate over the items in their fixed order."""
        return iter(self.items)

    @property
    def uniform_n_choices(self) -> int | None:
        """The shared choice count, or ``None`` if items disagree.

        Several interventions (notably cyclic option permutation) are only
        well-defined on a set whose items all offer the same number of options.
        """
        counts = {item.n_choices for item in self.items}
        return counts.pop() if len(counts) == 1 else None

    @property
    def chance_accuracy(self) -> float:
        """Accuracy a uniformly random guesser would obtain on this set."""
        return sum(1.0 / item.n_choices for item in self.items) / len(self.items)


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with an interval, a p-value, and its provenance.

    Attributes:
        point: The estimated quantity, in accuracy points (0.0-1.0 scale).
        ci_low: Lower confidence bound.
        ci_high: Upper confidence bound.
        p_value: Unadjusted two-sided p-value against the null of no effect.
        n: Sample size the estimate rests on.
        method: Name of the estimator, recorded so a reader can reproduce it.
    """

    point: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    method: str

    def __post_init__(self) -> None:
        """Reject malformed estimates.

        Raises:
            ValueError: If the interval is inverted, the p-value is outside
                ``[0, 1]``, or the sample size is not positive.
        """
        if self.ci_low > self.ci_high:
            raise ValueError(f"inverted interval: [{self.ci_low}, {self.ci_high}]")
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value {self.p_value} outside [0, 1]")
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")

    @property
    def excludes_zero(self) -> bool:
        """Whether the confidence interval lies wholly above or below zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


@dataclass(frozen=True, slots=True)
class Component:
    """One named artifact's share of a reported score.

    The share is a Shapley value over the coalition game defined in
    :mod:`evalassay.stats.shapley`, so shares from a single audit sum exactly
    to the total drop under all interventions. That exactness is the reason
    Shapley is used at all: marginal effects measured one at a time overlap,
    and summing them double-counts.

    Attributes:
        name: Short machine-readable artifact name.
        description: One line a reader can understand without the source.
        estimate: The Shapley share and its bootstrap interval.
        verdict: Whether the share cleared the default-deny gate.
        adjusted_p: Family-wise adjusted p-value across the component family.
        mde: Minimum effect this audit could have detected at its sample size,
            reported instead of post-hoc power, which is not informative.
    """

    name: str
    description: str
    estimate: Estimate
    verdict: Verdict
    adjusted_p: float
    mde: float

    @property
    def attributed_points(self) -> float:
        """Points deducted from the reported score for this artifact.

        Returns:
            The Shapley share when the gate passed, otherwise ``0.0``. An
            unestablished artifact is not charged against the model.
        """
        return self.estimate.point if self.verdict is Verdict.ESTABLISHED else 0.0


@dataclass(frozen=True, slots=True)
class Finding:
    """A model-free defect measured in the benchmark itself.

    Findings describe the corpus, not the model, and so sit outside the Shapley
    decomposition. A benchmark whose answer key is guessable from the options
    alone is defective regardless of which model is pointed at it.

    Attributes:
        detector: Name of the detector that produced the finding.
        description: One line a reader can understand without the source.
        estimate: Effect size, interval and p-value for the detector statistic.
        verdict: Whether the finding cleared the default-deny gate.
        adjusted_p: Family-wise adjusted p-value across the detector family.
        mde: Minimum effect this corpus size could have detected. Without it, a
            null finding says only that nothing was found, which is compatible
            with the benchmark being clean and with the sample being too small
            to tell. The distinction is the whole point of reporting one.
        detail: Optional human-readable context, such as example item ids.
    """

    detector: str
    description: str
    estimate: Estimate
    verdict: Verdict
    adjusted_p: float
    mde: float = 0.0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything needed to reproduce a run, hashed into the report.

    Attributes:
        schema_version: Layout version of the report and manifest.
        corpus_name: Name of the audited corpus.
        corpus_hash: Content hash over the normalised items.
        n_items: Item count actually audited.
        scorer_id: Identifier of the scoring backend and model.
        scorer_deterministic: Whether re-running the scorer on identical input
            is guaranteed to give identical output. Hosted APIs are recorded as
            ``False``; a report that claims reproducibility it cannot deliver
            is worse than one that admits the gap.
        config_hash: Hash over the audit configuration.
        seed: Root seed for every pseudo-random draw in the run.
        alpha: Pre-registered family-wise significance level.
        power: Pre-registered target power, used only to report the MDE.
        bootstrap_draws: Number of bootstrap resamples.
        evalassay_version: Version of this package.
        library_versions: Sorted ``(name, version)`` pairs for libraries whose
            numerics could move a result.
    """

    schema_version: str
    corpus_name: str
    corpus_hash: str
    n_items: int
    scorer_id: str
    scorer_deterministic: bool
    config_hash: str
    seed: int
    alpha: float
    power: float
    bootstrap_draws: int
    evalassay_version: str
    library_versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AuditReport:
    """The result of one audit.

    Attributes:
        manifest: Provenance for the run.
        reported_score: Accuracy with no intervention applied, on the 0.0-1.0
            scale. This is the number a leaderboard would print.
        total_drop: Accuracy lost under all interventions applied together,
            equal by construction to the sum of every component's raw Shapley
            share, established or not.
        components: The artifact decomposition.
        findings: Model-free corpus defects.
        chance_accuracy: What a uniform guesser would score on this corpus.
        blind_accuracy: Accuracy with the question removed entirely, or ``None``
            if the diagnostic was not run.

            This sits outside the decomposition on purpose. Removing the
            question destroys the accuracy that *depended* on the question,
            which is capability, not an artifact - charging it to the model
            would invert the meaning of the report. What it does reveal is the
            floor: a model scoring well above chance with nothing to answer is,
            on those items, not answering anything.
    """

    manifest: RunManifest
    reported_score: float
    total_drop: float
    components: tuple[Component, ...]
    findings: tuple[Finding, ...]
    chance_accuracy: float = 0.0
    blind_accuracy: Estimate | None = None

    @property
    def attributed_points(self) -> float:
        """Total accuracy charged to artifacts that cleared the gate."""
        return sum(component.attributed_points for component in self.components)

    @property
    def assayed_score(self) -> float:
        """Reported score with established artifact contributions removed.

        Returns:
            The defensible capability estimate, floored at zero. Only
            established components are subtracted, so the assayed score is
            deliberately generous to the model: an artifact the audit could not
            establish is treated as absent rather than assumed.
        """
        return max(0.0, self.reported_score - self.attributed_points)

    @property
    def purity(self) -> float:
        """Fraction of the reported score that survived the audit.

        Returns:
            ``assayed_score / reported_score``, or ``nan`` when nothing was
            reported. Named for the metallurgical assay the tool is named
            after: the proportion of the ore that is actually the metal.
        """
        if self.reported_score <= 0.0:
            return math.nan
        return self.assayed_score / self.reported_score

    @property
    def established(self) -> tuple[Component, ...]:
        """Components that cleared the default-deny gate."""
        return tuple(c for c in self.components if c.verdict is Verdict.ESTABLISHED)
