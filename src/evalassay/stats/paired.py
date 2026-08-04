"""Paired inference over per-item correctness vectors.

Every measurement in EvalAssay is paired: the same item is scored with and
without an intervention, so the comparison is within-item and the large
variance between easy and hard items cancels. Unpaired tests on the same data
would be valid but far less sensitive, and would make the audit's silence look
like evidence of absence when it was only lack of power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Outcome of an exact McNemar test on paired binary outcomes.

    Attributes:
        p_value: Exact two-sided p-value.
        delta: Mean change in accuracy, ``treatment - control``.
        n: Number of paired observations.
        control_only: Items the control got right and the treatment got wrong.
        treatment_only: Items the treatment got right and the control got wrong.
    """

    p_value: float
    delta: float
    n: int
    control_only: int
    treatment_only: int

    @property
    def discordant(self) -> int:
        """Number of pairs where the two conditions disagreed."""
        return self.control_only + self.treatment_only

    @property
    def discordance_rate(self) -> float:
        """Fraction of pairs where the two conditions disagreed."""
        return self.discordant / self.n


def mcnemar_exact(control: BoolArray, treatment: BoolArray) -> McNemarResult:
    """Run an exact McNemar test on paired binary outcomes.

    Concordant pairs carry no information about a difference, so the exact test
    conditions on the discordant ones: under the null they split
    ``Binomial(discordant, 0.5)``. The exact binomial form is used rather than
    the chi-square approximation because audits routinely produce few discordant
    pairs, which is exactly where the approximation misbehaves.

    Args:
        control: Per-item correctness without the intervention.
        treatment: Per-item correctness with the intervention.

    Returns:
        The test result.

    Raises:
        ValueError: If the two vectors differ in length or are empty.
    """
    if control.shape != treatment.shape:
        raise ValueError(f"shape mismatch: {control.shape} vs {treatment.shape}")
    if control.size == 0:
        raise ValueError("cannot test an empty pair of outcome vectors")

    control_only = int(np.count_nonzero(control & ~treatment))
    treatment_only = int(np.count_nonzero(~control & treatment))
    discordant = control_only + treatment_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(sps.binomtest(treatment_only, discordant, 0.5).pvalue)

    delta = float(np.mean(treatment.astype(np.float64) - control.astype(np.float64)))
    return McNemarResult(
        p_value=p_value,
        delta=delta,
        n=int(control.size),
        control_only=control_only,
        treatment_only=treatment_only,
    )


def bootstrap_counts(n: int, draws: int, rng: np.random.Generator) -> IntArray:
    """Draw multinomial resample counts for a paired item bootstrap.

    Resampling is expressed as per-item counts rather than as index lists
    because every statistic in this package is an average over items, and an
    average over a resample is a weighted average with these counts. That turns
    the whole bootstrap into one matrix product, which is what makes exact
    Shapley intervals affordable.

    Args:
        n: Number of items.
        draws: Number of bootstrap resamples.
        rng: Seeded generator; the only source of randomness in an audit.

    Returns:
        An ``(n, draws)`` integer array whose columns each sum to ``n``.

    Raises:
        ValueError: If ``n`` or ``draws`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if draws <= 0:
        raise ValueError(f"draws must be positive, got {draws}")
    probabilities = np.full(n, 1.0 / n)
    counts: IntArray = rng.multinomial(n, probabilities, size=draws).T.astype(np.int64)
    return counts


def percentile_ci(replicates: FloatArray, alpha: float) -> tuple[float, float]:
    """Take a plain percentile interval from bootstrap replicates.

    Args:
        replicates: Bootstrap replicates of the statistic.
        alpha: Two-sided error rate.

    Returns:
        Lower and upper bounds.

    Raises:
        ValueError: If there are no replicates or alpha is out of range.
    """
    if replicates.size == 0:
        raise ValueError("no bootstrap replicates")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha {alpha} outside (0, 1)")
    low, high = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def bca_ci(
    point: float,
    replicates: FloatArray,
    jackknife: FloatArray,
    alpha: float,
) -> tuple[float, float]:
    """Take a bias-corrected and accelerated bootstrap interval.

    BCa corrects for two things a percentile interval ignores: the bootstrap
    distribution sitting off-centre from the point estimate, and the variance of
    the statistic changing with its value. Both are common for the quantities
    here, which are bounded averages near the edge of their range when a model
    is strong.

    Falls back to a percentile interval when the correction is undefined, which
    happens when every replicate lands on the same value or the jackknife has no
    spread. Silently returning a degenerate BCa interval would be worse.

    Args:
        point: The statistic on the original sample.
        replicates: Bootstrap replicates of the statistic.
        jackknife: Leave-one-out values of the statistic.
        alpha: Two-sided error rate.

    Returns:
        Lower and upper bounds.

    Raises:
        ValueError: If there are no replicates or alpha is out of range.
    """
    if replicates.size == 0:
        raise ValueError("no bootstrap replicates")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha {alpha} outside (0, 1)")

    below = float(np.count_nonzero(replicates < point))
    proportion = below / replicates.size
    if proportion <= 0.0 or proportion >= 1.0:
        return percentile_ci(replicates, alpha)
    bias = float(sps.norm.ppf(proportion))

    centred = jackknife.mean() - jackknife
    sum_squares = float(np.sum(centred**2))
    if sum_squares <= 0.0:
        return percentile_ci(replicates, alpha)
    acceleration = float(np.sum(centred**3)) / (6.0 * sum_squares**1.5)

    def adjusted(quantile: float) -> float:
        """Map a nominal quantile through the bias and acceleration correction."""
        z = float(sps.norm.ppf(quantile))
        denominator = 1.0 - acceleration * (bias + z)
        if denominator <= 0.0:
            return quantile
        return float(sps.norm.cdf(bias + (bias + z) / denominator))

    low_q = adjusted(alpha / 2.0)
    high_q = adjusted(1.0 - alpha / 2.0)
    if not 0.0 < low_q < high_q < 1.0:
        return percentile_ci(replicates, alpha)

    low, high = np.percentile(replicates, [100.0 * low_q, 100.0 * high_q])
    return float(low), float(high)


def minimum_detectable_effect(discordance_rate: float, n: int, alpha: float, power: float) -> float:
    """Smallest paired accuracy difference this sample size could have detected.

    Reported instead of post-hoc power, which is a deterministic function of the
    observed p-value and therefore adds nothing. The MDE answers the question a
    reader of a null result actually has: how big would the effect have needed to
    be before this audit would have seen it?

    Uses the normal approximation to the paired difference, whose standard error
    is ``sqrt(discordance_rate / n)`` under the null.

    Args:
        discordance_rate: Observed fraction of pairs that disagreed.
        n: Number of paired observations.
        alpha: Two-sided significance level.
        power: Target power, for example 0.8.

    Returns:
        The minimum detectable effect in accuracy points. Returns ``0.0`` when
        no pair disagreed, since nothing was estimable.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if not 0.0 <= discordance_rate <= 1.0:
        raise ValueError(f"discordance_rate {discordance_rate} outside [0, 1]")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha {alpha} outside (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power {power} outside (0, 1)")
    if discordance_rate == 0.0:
        return 0.0
    return mde_from_standard_error(float(np.sqrt(discordance_rate / n)), alpha, power)


def mde_from_standard_error(standard_error: float, alpha: float, power: float) -> float:
    """Minimum detectable effect implied by an estimator's standard error.

    Used where the standard error comes from a bootstrap rather than a closed
    form, which is the case for the Shapley shares.

    Args:
        standard_error: Standard error of the estimator.
        alpha: Two-sided significance level.
        power: Target power.

    Returns:
        The smallest effect detectable at this precision.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if standard_error < 0.0:
        raise ValueError(f"standard_error must not be negative, got {standard_error}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha {alpha} outside (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power {power} outside (0, 1)")
    z_alpha = float(sps.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(sps.norm.ppf(power))
    return (z_alpha + z_power) * standard_error
