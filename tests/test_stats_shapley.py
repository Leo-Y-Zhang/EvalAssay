"""The Shapley layer is load-bearing, so its axioms are tested, not assumed.

Efficiency in particular is the whole argument for using Shapley at all: if the
shares did not sum to the observed drop, the decomposition would be decoration.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from evalassay.stats.shapley import (
    MAX_PLAYERS,
    coalition_accuracies,
    shapley_bootstrap,
    shapley_jackknife,
    shapley_operator,
    shapley_shares,
    total_drop,
)


def test_two_player_operator_matches_hand_computation() -> None:
    operator = shapley_operator(2)
    # Bitmask order: 00 = empty, 01 = {0}, 10 = {1}, 11 = both.
    expected_player_0 = np.array([0.5, -0.5, 0.5, -0.5])
    expected_player_1 = np.array([0.5, 0.5, -0.5, -0.5])
    np.testing.assert_allclose(operator[0], expected_player_0)
    np.testing.assert_allclose(operator[1], expected_player_1)


@pytest.mark.parametrize("n_players", [1, 2, 3, 4, 5])
def test_efficiency_shares_sum_to_total_drop(n_players: int) -> None:
    rng = np.random.default_rng(20260804 + n_players)
    operator = shapley_operator(n_players)
    for _ in range(25):
        accuracies = rng.uniform(0.0, 1.0, size=1 << n_players)
        shares = shapley_shares(operator, accuracies)
        assert shares.sum() == pytest.approx(total_drop(accuracies), abs=1e-12)


def test_dummy_player_receives_zero_share() -> None:
    # Three players; player 2 never changes accuracy, so adding it to any
    # coalition is a no-op.
    n_players = 3
    operator = shapley_operator(n_players)
    accuracies = np.empty(1 << n_players)
    for mask in range(1 << n_players):
        base = mask & 0b011
        # Deliberately non-additive in the two real players, so the test would
        # fail if the operator only handled separable games.
        interaction = 0.05 if base == 0b011 else 0.0
        accuracies[mask] = 0.9 - 0.1 * bin(base).count("1") - interaction
    shares = shapley_shares(operator, accuracies)
    assert shares[2] == pytest.approx(0.0, abs=1e-12)


def test_symmetric_players_receive_equal_shares() -> None:
    # Accuracy depends only on how many players are present, so all players are
    # interchangeable and must split the drop evenly.
    n_players = 4
    operator = shapley_operator(n_players)
    accuracies = np.array([0.8 - 0.07 * bin(mask).count("1") for mask in range(1 << n_players)])
    shares = shapley_shares(operator, accuracies)
    np.testing.assert_allclose(shares, np.full(n_players, shares[0]), atol=1e-12)


def test_linearity_in_the_accuracy_vector() -> None:
    operator = shapley_operator(3)
    rng = np.random.default_rng(11)
    a = rng.uniform(size=8)
    b = rng.uniform(size=8)
    combined = shapley_shares(operator, 0.3 * a + 0.7 * b)
    separate = 0.3 * shapley_shares(operator, a) + 0.7 * shapley_shares(operator, b)
    np.testing.assert_allclose(combined, separate, atol=1e-12)


def test_shares_reject_wrong_length_accuracy_vector() -> None:
    operator = shapley_operator(3)
    with pytest.raises(ValueError, match="expected 8 accuracies"):
        shapley_shares(operator, np.zeros(7))


@pytest.mark.parametrize("n_players", [0, -1, MAX_PLAYERS + 1])
def test_operator_rejects_out_of_range_player_counts(n_players: int) -> None:
    with pytest.raises(ValueError, match="n_players must be in"):
        shapley_operator(n_players)


FloatArray = NDArray[np.float64]


def _outcome_matrix(n_players: int, n_items: int, seed: int) -> FloatArray:
    """Build a plausible per-item outcome matrix for bootstrap tests."""
    rng = np.random.default_rng(seed)
    n_coalitions = 1 << n_players
    outcomes = np.zeros((n_coalitions, n_items), dtype=np.float64)
    base = rng.uniform(size=n_items) < 0.75
    for mask in range(n_coalitions):
        penalty = 0.08 * bin(mask).count("1")
        keep = rng.uniform(size=n_items) > penalty
        outcomes[mask] = (base & keep).astype(np.float64)
    return outcomes


def test_bootstrap_with_unit_counts_reproduces_the_point_estimate() -> None:
    n_players, n_items = 3, 60
    outcomes = _outcome_matrix(n_players, n_items, seed=3)
    operator = shapley_operator(n_players)
    point = shapley_shares(operator, coalition_accuracies(outcomes))

    unit_counts = np.ones((n_items, 1), dtype=np.int64)
    replicates = shapley_bootstrap(outcomes, operator, unit_counts)
    np.testing.assert_allclose(replicates[:, 0], point, atol=1e-12)


def test_bootstrap_chunking_does_not_change_results() -> None:
    n_players, n_items, draws = 3, 40, 100
    outcomes = _outcome_matrix(n_players, n_items, seed=5)
    operator = shapley_operator(n_players)
    rng = np.random.default_rng(99)
    counts = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=draws).T.astype(
        np.int64
    )

    whole = shapley_bootstrap(outcomes, operator, counts, chunk=draws)
    chunked = shapley_bootstrap(outcomes, operator, counts, chunk=7)
    np.testing.assert_allclose(whole, chunked, atol=1e-12)


def test_bootstrap_replicates_satisfy_efficiency_draw_by_draw() -> None:
    n_players, n_items, draws = 4, 50, 64
    outcomes = _outcome_matrix(n_players, n_items, seed=13)
    operator = shapley_operator(n_players)
    rng = np.random.default_rng(21)
    counts = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=draws).T.astype(
        np.int64
    )
    replicates = shapley_bootstrap(outcomes, operator, counts)

    resampled = (outcomes @ counts.astype(np.float64)) / n_items
    expected_totals = resampled[0] - resampled[-1]
    np.testing.assert_allclose(replicates.sum(axis=0), expected_totals, atol=1e-12)


def test_jackknife_column_matches_manual_leave_one_out() -> None:
    n_players, n_items = 2, 12
    outcomes = _outcome_matrix(n_players, n_items, seed=17)
    operator = shapley_operator(n_players)
    jackknife = shapley_jackknife(outcomes, operator)

    dropped = np.delete(outcomes, 4, axis=1)
    expected = shapley_shares(operator, coalition_accuracies(dropped))
    np.testing.assert_allclose(jackknife[:, 4], expected, atol=1e-12)


def test_jackknife_rejects_single_item() -> None:
    operator = shapley_operator(2)
    with pytest.raises(ValueError, match="at least 2 items"):
        shapley_jackknife(np.ones((4, 1)), operator)


def test_bootstrap_rejects_mismatched_shapes() -> None:
    operator = shapley_operator(3)
    with pytest.raises(ValueError, match="coalitions"):
        shapley_bootstrap(np.ones((4, 10)), operator, np.ones((10, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="items"):
        shapley_bootstrap(np.ones((8, 10)), operator, np.ones((9, 5), dtype=np.int64))


def test_bootstrap_rejects_non_positive_chunk() -> None:
    operator = shapley_operator(2)
    with pytest.raises(ValueError, match="chunk must be positive"):
        shapley_bootstrap(np.ones((4, 5)), operator, np.ones((5, 2), dtype=np.int64), chunk=0)
