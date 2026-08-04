"""Exact Shapley attribution of an accuracy drop among intervention players.

Why Shapley, and not the obvious thing
--------------------------------------
The obvious way to report artifacts in a benchmark score is to measure each one
on its own and print the list. That is wrong, and quietly so. Option-position
bias and choices-only solvability overlap: an item answered correctly by
position heuristics alone is often the same item answerable from the options
alone. Measured separately, both effects claim it. Summed, the total exceeds the
real drop, and the implied "true capability" is too low.

Shapley values are the unique attribution satisfying efficiency (the shares sum
to exactly the joint effect), symmetry (equal contributors get equal shares),
dummy (a player who never changes anything gets zero) and additivity. Efficiency
is the property that matters here: it is what makes the decomposition sum back
to the observed drop instead of merely gesturing at it.

Why it is affordable
--------------------
With a handful of players the coalition lattice is small, and the Shapley value
is a *linear* functional of the coalition values. So the whole attribution is
one fixed matrix applied to the vector of coalition accuracies, and a bootstrap
over items becomes a single matrix product rather than thousands of re-scored
evaluations. No model is called during interval estimation at all.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MAX_PLAYERS = 12
"""Guard on exact enumeration: the lattice is ``2**n_players`` wide."""

_DEFAULT_CHUNK = 2048
"""Bootstrap draws processed per block, to bound peak memory."""

MIN_JACKKNIFE_ITEMS = 2
"""Leave-one-out is undefined on a single observation."""


def coalition_size(mask: int) -> int:
    """Number of players present in a coalition bitmask.

    Args:
        mask: Coalition encoded as a bitmask.

    Returns:
        The population count.
    """
    return int(mask.bit_count())


def shapley_operator(n_players: int) -> FloatArray:
    """Build the matrix mapping coalition accuracies to Shapley shares.

    The game is ``v(S) = accuracy(empty) - accuracy(S)``: the accuracy destroyed
    by applying the interventions in ``S`` together. Expanding the Shapley
    formula and cancelling the ``accuracy(empty)`` terms leaves a linear map
    straight from accuracies to shares, which is what this returns.

    Args:
        n_players: Number of interventions being attributed.

    Returns:
        An ``(n_players, 2**n_players)`` matrix ``M`` such that ``M @ acc``
        gives the Shapley shares, where ``acc`` is indexed by coalition bitmask.

    Raises:
        ValueError: If ``n_players`` is not between 1 and :data:`MAX_PLAYERS`.
    """
    if not 1 <= n_players <= MAX_PLAYERS:
        raise ValueError(f"n_players must be in 1..{MAX_PLAYERS}, got {n_players}")

    n_coalitions = 1 << n_players
    operator = np.zeros((n_players, n_coalitions), dtype=np.float64)
    total = factorial(n_players)

    for player in range(n_players):
        bit = 1 << player
        for mask in range(n_coalitions):
            if mask & bit:
                continue
            size = coalition_size(mask)
            weight = factorial(size) * factorial(n_players - size - 1) / total
            # v(S union {i}) - v(S) == accuracy(S) - accuracy(S union {i})
            operator[player, mask] += weight
            operator[player, mask | bit] -= weight

    return operator


def shapley_shares(operator: FloatArray, accuracies: FloatArray) -> FloatArray:
    """Apply the Shapley operator to a vector of coalition accuracies.

    Args:
        operator: Matrix from :func:`shapley_operator`.
        accuracies: Accuracy per coalition, indexed by bitmask.

    Returns:
        One share per player.

    Raises:
        ValueError: If the shapes do not line up.
    """
    if accuracies.shape != (operator.shape[1],):
        raise ValueError(f"expected {operator.shape[1]} accuracies, got {accuracies.shape}")
    return np.asarray(operator @ accuracies, dtype=np.float64)


def total_drop(accuracies: FloatArray) -> float:
    """Accuracy lost when every intervention is applied together.

    By the efficiency axiom this equals the sum of the Shapley shares, which
    :func:`~evalassay.stats.shapley` tests on every run.

    Args:
        accuracies: Accuracy per coalition, indexed by bitmask.

    Returns:
        ``accuracy(empty) - accuracy(all)``.
    """
    return float(accuracies[0] - accuracies[-1])


def coalition_accuracies(outcomes: FloatArray) -> FloatArray:
    """Mean accuracy per coalition on the original sample.

    Args:
        outcomes: ``(2**n_players, n_items)`` array of per-item correctness,
            0.0 or 1.0, one row per coalition bitmask.

    Returns:
        Accuracy per coalition.
    """
    return np.asarray(outcomes.mean(axis=1), dtype=np.float64)


def shapley_bootstrap(
    outcomes: FloatArray,
    operator: FloatArray,
    counts: IntArray,
    chunk: int = _DEFAULT_CHUNK,
) -> FloatArray:
    """Bootstrap the Shapley shares by resampling items.

    Items are resampled once and the same resample is used for every coalition,
    which preserves the pairing: an item that is heavily weighted in a draw is
    heavily weighted in all conditions of that draw. Resampling each coalition
    independently would break the pairing and inflate the intervals.

    Args:
        outcomes: ``(2**n_players, n_items)`` per-item correctness.
        operator: Matrix from :func:`shapley_operator`.
        counts: ``(n_items, draws)`` multinomial resample counts.
        chunk: Draws processed per block, bounding peak memory.

    Returns:
        An ``(n_players, draws)`` array of bootstrap replicates.

    Raises:
        ValueError: If the shapes do not line up or ``chunk`` is not positive.
    """
    if outcomes.shape[0] != operator.shape[1]:
        raise ValueError(
            f"outcomes has {outcomes.shape[0]} coalitions, operator expects {operator.shape[1]}"
        )
    if outcomes.shape[1] != counts.shape[0]:
        raise ValueError(f"outcomes has {outcomes.shape[1]} items, counts has {counts.shape[0]}")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")

    n_items = outcomes.shape[1]
    draws = counts.shape[1]
    replicates = np.empty((operator.shape[0], draws), dtype=np.float64)

    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        block = counts[:, start:stop].astype(np.float64)
        accuracies = (outcomes @ block) / n_items
        replicates[:, start:stop] = operator @ accuracies

    return replicates


def shapley_jackknife(outcomes: FloatArray, operator: FloatArray) -> FloatArray:
    """Leave-one-item-out Shapley shares, for the BCa acceleration term.

    Args:
        outcomes: ``(2**n_players, n_items)`` per-item correctness.
        operator: Matrix from :func:`shapley_operator`.

    Returns:
        An ``(n_players, n_items)`` array whose column ``j`` is the attribution
        computed with item ``j`` removed.

    Raises:
        ValueError: If there are fewer than two items.
    """
    n_items = outcomes.shape[1]
    if n_items < MIN_JACKKNIFE_ITEMS:
        raise ValueError(f"jackknife needs at least {MIN_JACKKNIFE_ITEMS} items")
    totals = outcomes.sum(axis=1, keepdims=True)
    leave_one_out = (totals - outcomes) / (n_items - 1)
    return np.asarray(operator @ leave_one_out, dtype=np.float64)
