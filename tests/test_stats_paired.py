"""Paired inference: exact test, resampling, intervals and the MDE."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from evalassay.stats.paired import (
    bca_ci,
    bootstrap_counts,
    mcnemar_exact,
    minimum_detectable_effect,
    percentile_ci,
)

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def _pairs(
    both: int, control_only: int, treatment_only: int, neither: int
) -> tuple[BoolArray, BoolArray]:
    """Build paired outcome vectors with an exact contingency table."""
    control = [True] * both + [True] * control_only + [False] * treatment_only + [False] * neither
    treatment = [True] * both + [False] * control_only + [True] * treatment_only + [False] * neither
    return np.array(control, dtype=np.bool_), np.array(treatment, dtype=np.bool_)


def _f64(values: object) -> FloatArray:
    """Coerce to the float64 array type the statistics layer declares."""
    return np.asarray(values, dtype=np.float64)


def test_counts_the_discordant_cells() -> None:
    control, treatment = _pairs(both=10, control_only=7, treatment_only=2, neither=5)
    result = mcnemar_exact(control, treatment)
    assert result.control_only == 7
    assert result.treatment_only == 2
    assert result.discordant == 9
    assert result.n == 24


def test_delta_is_treatment_minus_control() -> None:
    control, treatment = _pairs(both=0, control_only=6, treatment_only=1, neither=3)
    result = mcnemar_exact(control, treatment)
    # Control right on 6 of 10, treatment right on 1 of 10.
    assert result.delta == pytest.approx(0.1 - 0.6)


def test_identical_vectors_give_p_of_one() -> None:
    outcomes = np.array([True, False, True, True, False])
    result = mcnemar_exact(outcomes, outcomes.copy())
    assert result.p_value == 1.0
    assert result.discordant == 0
    assert result.discordance_rate == 0.0


def test_balanced_discordance_is_not_significant() -> None:
    control, treatment = _pairs(both=5, control_only=8, treatment_only=8, neither=5)
    result = mcnemar_exact(control, treatment)
    assert result.p_value == pytest.approx(1.0)


def test_one_sided_discordance_is_significant() -> None:
    control, treatment = _pairs(both=5, control_only=20, treatment_only=1, neither=5)
    result = mcnemar_exact(control, treatment)
    assert result.p_value < 0.001
    assert result.delta < 0


def test_matches_the_closed_form_binomial() -> None:
    # With 3 of 10 discordant pairs favouring treatment, the exact two-sided
    # p-value is the binomial tail, which is 0.34375 for Binomial(10, 0.5).
    control, treatment = _pairs(both=0, control_only=7, treatment_only=3, neither=0)
    result = mcnemar_exact(control, treatment)
    assert result.p_value == pytest.approx(0.34375)


def test_rejects_mismatched_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mcnemar_exact(np.array([True, False]), np.array([True]))
    with pytest.raises(ValueError, match="empty"):
        mcnemar_exact(np.array([], dtype=bool), np.array([], dtype=bool))


def test_bootstrap_counts_are_a_valid_resample() -> None:
    counts = bootstrap_counts(n=25, draws=40, rng=np.random.default_rng(1))
    assert counts.shape == (25, 40)
    np.testing.assert_array_equal(counts.sum(axis=0), np.full(40, 25))
    assert counts.min() >= 0


def test_bootstrap_counts_are_reproducible_from_the_seed() -> None:
    first = bootstrap_counts(n=15, draws=20, rng=np.random.default_rng(4242))
    second = bootstrap_counts(n=15, draws=20, rng=np.random.default_rng(4242))
    np.testing.assert_array_equal(first, second)


def test_bootstrap_counts_reject_degenerate_sizes() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be positive"):
        bootstrap_counts(n=0, draws=5, rng=rng)
    with pytest.raises(ValueError, match="draws must be positive"):
        bootstrap_counts(n=5, draws=0, rng=rng)


def test_percentile_interval_brackets_the_bulk() -> None:
    replicates = _f64(np.linspace(0.0, 1.0, 1001))
    low, high = percentile_ci(replicates, alpha=0.10)
    assert low == pytest.approx(0.05, abs=1e-3)
    assert high == pytest.approx(0.95, abs=1e-3)


def test_percentile_interval_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="no bootstrap replicates"):
        percentile_ci(_f64([]), alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        percentile_ci(_f64(np.linspace(0, 1, 10)), alpha=1.5)


def test_bca_recovers_the_percentile_interval_for_a_symmetric_sample() -> None:
    rng = np.random.default_rng(2)
    replicates = _f64(rng.normal(loc=0.5, scale=0.1, size=20_000))
    jackknife = _f64(rng.normal(loc=0.5, scale=0.01, size=200))
    low, high = bca_ci(0.5, replicates, jackknife, alpha=0.05)
    # A symmetric bootstrap around the point estimate leaves little for the
    # correction to do, so BCa should land close to the plain percentile bounds.
    p_low, p_high = percentile_ci(replicates, alpha=0.05)
    assert low == pytest.approx(p_low, abs=0.02)
    assert high == pytest.approx(p_high, abs=0.02)


def test_bca_falls_back_when_replicates_are_degenerate() -> None:
    replicates = _f64(np.full(500, 0.25))
    jackknife = _f64(np.full(50, 0.25))
    low, high = bca_ci(0.25, replicates, jackknife, alpha=0.05)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_bca_falls_back_when_the_jackknife_has_no_spread() -> None:
    rng = np.random.default_rng(6)
    replicates = _f64(rng.normal(0.4, 0.05, size=2000))
    jackknife = _f64(np.full(40, 0.4))
    low, high = bca_ci(0.4, replicates, jackknife, alpha=0.05)
    expected_low, expected_high = percentile_ci(replicates, alpha=0.05)
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_high)


def test_bca_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="no bootstrap replicates"):
        bca_ci(0.0, _f64([]), _f64([1.0, 2.0]), alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        bca_ci(0.0, _f64(np.linspace(0, 1, 10)), _f64(np.linspace(0, 1, 5)), alpha=0.0)


def test_mde_shrinks_with_sample_size() -> None:
    small = minimum_detectable_effect(0.2, n=100, alpha=0.01, power=0.8)
    large = minimum_detectable_effect(0.2, n=10_000, alpha=0.01, power=0.8)
    assert large < small
    # Standard error scales as one over the square root of n.
    assert small / large == pytest.approx(10.0, rel=1e-9)


def test_mde_grows_with_discordance() -> None:
    quiet = minimum_detectable_effect(0.05, n=500, alpha=0.01, power=0.8)
    noisy = minimum_detectable_effect(0.40, n=500, alpha=0.01, power=0.8)
    assert noisy > quiet


def test_mde_is_zero_when_nothing_was_estimable() -> None:
    assert minimum_detectable_effect(0.0, n=500, alpha=0.01, power=0.8) == 0.0


def test_mde_rejects_out_of_range_arguments() -> None:
    with pytest.raises(ValueError, match="discordance_rate"):
        minimum_detectable_effect(1.5, n=10, alpha=0.05, power=0.8)
    with pytest.raises(ValueError, match="n must be positive"):
        minimum_detectable_effect(0.2, n=0, alpha=0.05, power=0.8)
    with pytest.raises(ValueError, match="alpha"):
        minimum_detectable_effect(0.2, n=10, alpha=0.0, power=0.8)
    with pytest.raises(ValueError, match="power"):
        minimum_detectable_effect(0.2, n=10, alpha=0.05, power=1.0)
