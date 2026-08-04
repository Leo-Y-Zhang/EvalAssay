"""Shared machinery for the model-free pathology detectors.

Detectors describe the *benchmark*, not the model. They run before any model is
loaded and cost nothing but arithmetic, which means a corpus can be checked for
defects even by someone with no compute at all.

A detector returns a :class:`RawFinding`. It does not decide whether its own
result is significant: multiplicity correction and the default-deny gate are
applied across the whole detector family by
:func:`evalassay.pathology.runner.run_all`, because a detector that judged
itself would be marking its own homework.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

from evalassay.types import Estimate, ItemSet

FloatArray = NDArray[np.float64]

TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
"""Words, keeping internal hyphens and apostrophes, which carry meaning."""


@dataclass(frozen=True, slots=True)
class RawFinding:
    """A detector's result, before family-wise correction.

    Attributes:
        detector: Machine-readable detector name.
        description: One line a reader can understand without the source.
        estimate: Effect size, interval and unadjusted p-value.
        detail: Optional human-readable context. Never used in inference, so it
            is free to mention data-dependent things such as which option
            position was the most common.
    """

    detector: str
    description: str
    estimate: Estimate
    detail: str = ""


class Detector(Protocol):
    """The interface every pathology detector implements.

    The two attributes are declared read-only so that detectors can be frozen
    dataclasses. A protocol member written as a plain variable would demand a
    settable attribute and quietly exclude every immutable implementation.
    """

    @property
    def name(self) -> str:
        """Machine-readable detector name, stable across releases."""
        ...

    @property
    def assumes_independent_items(self) -> bool:
        """Whether this detector's inference treats items as independent draws.

        An exact repeat is not a second observation. Leaving repeats in place
        shrinks the effective sample size without shrinking the nominal one,
        which makes any test built on that assumption anti-conservative - it
        will find significance that is not there. The runner therefore hands
        detectors that declare ``True`` a deduplicated corpus, and records it.
        """
        ...

    def run(self, item_set: ItemSet, rng: np.random.Generator) -> RawFinding | None:
        """Measure this defect on a corpus.

        Args:
            item_set: The corpus to inspect.
            rng: Seeded generator, for detectors that resample.

        Returns:
            The finding, or ``None`` if the detector does not apply to this
            corpus. Returning ``None`` is not a pass: it means the question was
            never asked, and the runner records it as such.
        """
        ...


def tokenise(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Args:
        text: Raw text.

    Returns:
        The tokens, in order.
    """
    return TOKEN_PATTERN.findall(text.lower())


def wilson_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because these proportions routinely
    sit at or near zero, where the normal interval extends below zero and can
    even have zero width.

    Args:
        successes: Number of successes.
        trials: Number of trials.
        alpha: Two-sided error rate.

    Returns:
        Lower and upper bounds, clipped to ``[0, 1]``.

    Raises:
        ValueError: If the counts are impossible or alpha is out of range.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside 0..{trials}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha {alpha} outside (0, 1)")

    z = float(sps.norm.ppf(1.0 - alpha / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = (
        z
        / denominator
        * float(np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)))
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def bootstrap_mean_interval(
    values: FloatArray, rng: np.random.Generator, draws: int, alpha: float
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-item values.

    Args:
        values: One value per item.
        rng: Seeded generator.
        draws: Number of resamples.
        alpha: Two-sided error rate.

    Returns:
        Lower and upper bounds.

    Raises:
        ValueError: If there are no values.
    """
    if values.size == 0:
        raise ValueError("no values to bootstrap")
    n = values.size
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=draws)
    means = (counts @ values) / n
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def largest_uniform_subset(item_set: ItemSet) -> tuple[Sequence[int], int]:
    """Find the biggest group of items sharing a choice count.

    Several detectors are only well defined when every item offers the same
    number of options. Rather than refuse a ragged corpus outright, they run on
    its largest uniform group and say so in the finding's detail.

    Args:
        item_set: The corpus.

    Returns:
        The indices of the largest group, and the choice count they share.
    """
    groups: dict[int, list[int]] = {}
    for index, item in enumerate(item_set.items):
        groups.setdefault(item.n_choices, []).append(index)
    best = max(groups, key=lambda k: (len(groups[k]), k))
    return groups[best], best


def make_estimate(
    point: float,
    interval: tuple[float, float],
    p_value: float,
    n: int,
    method: str,
) -> Estimate:
    """Assemble an estimate, widening the interval to contain the point.

    A bootstrap or score interval can miss its own point estimate by a hair
    through discreteness. Silently returning an interval that excludes the point
    would let the gate's "excludes zero" test disagree with the reported effect,
    so the interval is widened rather than left inconsistent.

    Args:
        point: The point estimate.
        interval: Lower and upper bounds.
        p_value: Unadjusted two-sided p-value.
        n: Sample size.
        method: Estimator name.

    Returns:
        The estimate.
    """
    low, high = interval
    return Estimate(
        point=point,
        ci_low=min(low, point),
        ci_high=max(high, point),
        p_value=p_value,
        n=n,
        method=method,
    )
