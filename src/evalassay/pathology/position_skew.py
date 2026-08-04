"""Is the answer key uniform over option positions?

A benchmark whose correct answer sits at one position more often than chance
hands free accuracy to any model with a positional preference, and that accuracy
looks exactly like knowledge on a leaderboard.

The statistic is the total variation distance between the observed distribution
of answer positions and the uniform one. Total variation is used rather than
"excess share at the most common position" because the most common position is
chosen after seeing the data, and an effect size selected that way is biased
upward with no honest interval.

Total variation is *also* biased upward - it is a distance, so sampling noise
alone makes it positive - and that bias is removed by subtracting its expected
value under the uniform null, estimated by simulation. Without that subtraction
a perfectly uniform benchmark would produce a confidence interval excluding
zero, and the detector would fire on every corpus it ever saw.

**The correction is exact under the null and conservative away from it.** When
the true skew is large, the observed statistic is already close to unbiased -
noise adds little to a distance that is mostly signal - so subtracting the full
null bias understates the skew by roughly that bias. A corpus with a true total
variation of 0.375 is therefore reported at about 0.355.

That direction is deliberate. This detector's output is a criticism of somebody's
benchmark, and an estimator that can only understate the charge is the right one
to reach for. The reported skew is a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

from evalassay.pathology.base import (
    RawFinding,
    bootstrap_mean_interval,
    largest_uniform_subset,
    make_estimate,
)
from evalassay.types import ItemSet

FloatArray = NDArray[np.float64]

NULL_SIMULATIONS: Final = 2000
"""Draws used to estimate the null bias of the total variation statistic."""

BOOTSTRAP_DRAWS: Final = 2000
"""Resamples used for the interval."""


def total_variation(counts: FloatArray) -> float:
    """Total variation distance between a count vector and the uniform law.

    Args:
        counts: Counts per option position.

    Returns:
        A value in ``[0, 1]``; zero exactly when the counts are uniform.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    shares = counts / total
    uniform = 1.0 / counts.size
    return 0.5 * float(np.abs(shares - uniform).sum())


def _null_bias(n_items: int, n_choices: int, rng: np.random.Generator) -> float:
    """Expected total variation under a uniform answer key.

    Args:
        n_items: Number of items.
        n_choices: Options per item.
        rng: Seeded generator.

    Returns:
        The mean statistic across simulated uniform corpora.
    """
    simulated = rng.multinomial(n_items, np.full(n_choices, 1.0 / n_choices), size=NULL_SIMULATIONS)
    shares = simulated / n_items
    return float(0.5 * np.abs(shares - 1.0 / n_choices).sum(axis=1).mean())


@dataclass(frozen=True, slots=True)
class PositionSkew:
    """Detector for non-uniform answer-key positions."""

    name: str = "position_skew"
    assumes_independent_items: bool = True

    def run(self, item_set: ItemSet, rng: np.random.Generator) -> RawFinding | None:
        """Measure how far the answer-position distribution sits from uniform.

        Args:
            item_set: The corpus.
            rng: Seeded generator.

        Returns:
            The finding, or ``None`` if no group of items shares a choice count
            large enough to test.
        """
        indices, n_choices = largest_uniform_subset(item_set)
        n_items = len(indices)
        if n_items < n_choices * 2:
            return None

        positions = np.array([item_set.items[i].answer_index for i in indices])
        counts = np.bincount(positions, minlength=n_choices).astype(np.float64)

        observed = total_variation(counts)
        bias = _null_bias(n_items, n_choices, rng)
        point = observed - bias

        expected = np.full(n_choices, n_items / n_choices)
        p_value = float(sps.chisquare(counts, expected).pvalue)

        # Resample items, recompute the statistic, and shift by the same null
        # bias so the interval is on the same scale as the point estimate.
        per_item = np.zeros((n_items, n_choices), dtype=np.float64)
        per_item[np.arange(n_items), positions] = 1.0
        draw_counts = rng.multinomial(
            n_items, np.full(n_items, 1.0 / n_items), size=BOOTSTRAP_DRAWS
        )
        resampled = draw_counts @ per_item
        shares = resampled / n_items
        replicates = 0.5 * np.abs(shares - 1.0 / n_choices).sum(axis=1) - bias
        low, high = np.percentile(replicates, [0.5, 99.5])

        modal = int(np.argmax(counts))
        modal_share = float(counts[modal] / n_items)
        detail = (
            f"most common answer position is {modal} at {modal_share:.1%} "
            f"against {1.0 / n_choices:.1%} expected; "
            f"measured on {n_items} of {len(item_set)} items with {n_choices} options"
        )

        return RawFinding(
            detector=self.name,
            description=(
                "answer key is not uniform across option positions, which hands "
                "free accuracy to any model with a positional preference"
            ),
            estimate=make_estimate(
                point=point,
                interval=(float(low), float(high)),
                p_value=p_value,
                n=n_items,
                method=(
                    "total variation vs uniform, null-bias corrected; chi-square goodness of fit"
                ),
            ),
            detail=detail,
        )


def bootstrap_interval_for_values(
    values: FloatArray, rng: np.random.Generator, alpha: float
) -> tuple[float, float]:
    """Convenience wrapper used by sibling detectors.

    Args:
        values: One value per item.
        rng: Seeded generator.
        alpha: Two-sided error rate.

    Returns:
        Lower and upper bounds.
    """
    return bootstrap_mean_interval(values, rng, BOOTSTRAP_DRAWS, alpha)
