"""Holm's step-down procedure."""

from __future__ import annotations

import pytest

from evalassay.stats.multiplicity import holm_bonferroni


def test_worked_example() -> None:
    # Ranks scale by m, m-1, m-2, then the running maximum enforces monotonicity:
    # 3*0.01 = 0.03, 2*0.02 = 0.04, 1*0.03 = 0.03 -> lifted to 0.04.
    assert holm_bonferroni([0.01, 0.02, 0.03]) == pytest.approx((0.03, 0.04, 0.04))


def test_preserves_input_order() -> None:
    adjusted = holm_bonferroni([0.03, 0.01, 0.02])
    assert adjusted == pytest.approx((0.04, 0.03, 0.04))


def test_single_hypothesis_is_unchanged() -> None:
    assert holm_bonferroni([0.042]) == pytest.approx((0.042,))


def test_empty_family_is_empty() -> None:
    assert holm_bonferroni([]) == ()


def test_adjusted_values_never_exceed_one() -> None:
    adjusted = holm_bonferroni([0.4, 0.5, 0.6, 0.7])
    assert all(p <= 1.0 for p in adjusted)


def test_adjustment_is_monotone_in_the_sorted_order() -> None:
    raw = [0.001, 0.004, 0.01, 0.2, 0.9]
    adjusted = holm_bonferroni(raw)
    ordered = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
    assert ordered == sorted(ordered)


def test_never_reduces_a_p_value() -> None:
    raw = [0.02, 0.03, 0.04]
    assert all(a >= r for a, r in zip(holm_bonferroni(raw), raw, strict=True))


def test_is_no_more_conservative_than_bonferroni() -> None:
    raw = [0.001, 0.02, 0.03, 0.04]
    adjusted = holm_bonferroni(raw)
    assert all(a <= min(1.0, len(raw) * r) for a, r in zip(adjusted, raw, strict=True))


def test_rejects_values_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="outside"):
        holm_bonferroni([0.5, 1.5])
    with pytest.raises(ValueError, match="outside"):
        holm_bonferroni([-0.1])
