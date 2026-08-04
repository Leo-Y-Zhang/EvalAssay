"""Does "pick the longest option" beat chance?

Writers pad correct answers. A true statement often needs a qualifier that a
false one does not, so the key ends up longer than the distractors, and a model
that has learned nothing but that regularity scores above chance.

The statistic is the accuracy of the longest-option heuristic minus chance
accuracy. Ties are credited in proportion - an item whose longest length is
shared by two options contributes one half if the key is one of them - which
makes the estimator exactly unbiased under the null: if the key is uniform over
options and independent of length, the heuristic's expected score per item is
exactly one over the number of options, whatever the length pattern is.

That exact unbiasedness is why ties are handled this way rather than by
discarding tied items, which would silently change the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from evalassay.pathology.base import RawFinding, bootstrap_mean_interval, make_estimate
from evalassay.types import ItemSet

FloatArray = NDArray[np.float64]

RANDOMISATIONS: Final = 4000
"""Draws used for the randomisation p-value."""

BOOTSTRAP_DRAWS: Final = 2000
"""Resamples used for the interval."""

MIN_ITEMS: Final = 20
"""Below this the randomisation distribution is too coarse to be worth reporting."""

_RANDOMISATION_CHUNK: Final = 256
"""Randomisation draws held in memory at once."""


def _winner_mask(item_choices: tuple[str, ...]) -> FloatArray:
    """Score each option by whether it is among the longest.

    Args:
        item_choices: The options.

    Returns:
        A vector summing to one, spreading credit evenly across ties.
    """
    lengths = np.array([len(choice) for choice in item_choices], dtype=np.float64)
    winners = lengths == lengths.max()
    return np.asarray(winners / winners.sum(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LongestAnswer:
    """Detector for the longest-option heuristic beating chance."""

    name: str = "longest_answer"
    assumes_independent_items: bool = True

    def run(self, item_set: ItemSet, rng: np.random.Generator) -> RawFinding | None:
        """Measure how far the longest-option heuristic beats chance.

        Args:
            item_set: The corpus.
            rng: Seeded generator.

        Returns:
            The finding, or ``None`` if the corpus is too small.
        """
        n_items = len(item_set)
        if n_items < MIN_ITEMS:
            return None

        masks = [_winner_mask(item.choices) for item in item_set]
        scores = np.array(
            [mask[item.answer_index] for mask, item in zip(masks, item_set, strict=True)],
            dtype=np.float64,
        )
        chance = np.array([1.0 / item.n_choices for item in item_set], dtype=np.float64)
        excess = np.asarray(scores - chance, dtype=np.float64)

        point = float(excess.mean())

        # Randomisation null: re-draw the answer position uniformly for every
        # item, leaving the option lengths exactly as they are. This tests
        # precisely the hypothesis that length carries no information about the
        # key, without assuming any distribution for the statistic.
        #
        # Done as a chunked matrix operation rather than a loop. The naive
        # version is a product of two large numbers of iterations, and the fully
        # vectorised version allocates that same product as one array; chunking
        # keeps both the time and the peak memory in bounds.
        counts = np.array([item.n_choices for item in item_set], dtype=np.int64)
        padded = np.zeros((n_items, int(counts.max())), dtype=np.float64)
        for row, mask in enumerate(masks):
            padded[row, : mask.size] = mask

        rows = np.arange(n_items)
        simulated = np.empty(RANDOMISATIONS, dtype=np.float64)
        for start in range(0, RANDOMISATIONS, _RANDOMISATION_CHUNK):
            stop = min(start + _RANDOMISATION_CHUNK, RANDOMISATIONS)
            drawn = (rng.random((stop - start, n_items)) * counts).astype(np.int64)
            simulated[start:stop] = (padded[rows, drawn] - chance).mean(axis=1)

        extreme = int(np.count_nonzero(np.abs(simulated) >= abs(point)))
        # The plus-one form keeps the p-value strictly positive: a randomisation
        # test can never license a claim of exactly zero probability.
        p_value = (1 + extreme) / (RANDOMISATIONS + 1)

        low, high = bootstrap_mean_interval(excess, rng, BOOTSTRAP_DRAWS, alpha=0.01)

        tied = sum(1 for mask in masks if np.count_nonzero(mask > 0) > 1)
        detail = (
            f"heuristic accuracy {float(scores.mean()):.1%} against "
            f"{float(chance.mean()):.1%} chance; "
            f"{tied} of {n_items} items have a tie for longest option"
        )

        return RawFinding(
            detector=self.name,
            description=(
                "the longest option is the key more often than chance, so a model "
                "that has learned only that regularity scores above chance"
            ),
            estimate=make_estimate(
                point=point,
                interval=(low, high),
                p_value=p_value,
                n=n_items,
                method="longest-option heuristic excess over chance; randomisation test",
            ),
            detail=detail,
        )
