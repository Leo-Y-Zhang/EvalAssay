"""How much of the answer key is recoverable from the options alone?

The question is never shown to this probe. If a classifier that has only ever
seen answer options can pick the key above chance, then the benchmark leaks its
own answers through surface form, and any model's score on it is inflated by an
amount that has nothing to do with understanding the question.

The probe is a multinomial naive Bayes model over option tokens, trained and
evaluated under grouped cross-validation so that an item's options never appear
in both the training and the evaluation fold. It is written out in numpy rather
than pulled from a machine-learning library for two reasons: it keeps the core
of this package dependency-free, and a reviewer can read the twenty lines that
produce the number rather than trusting a call into a library.

**The reported leakage is a lower bound.** A weak probe that finds leakage
proves leakage exists; a weak probe that finds none proves only that this probe
found none. A stronger probe can raise this number and can never lower it, and
the report says so rather than implying the measurement is tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from evalassay.pathology.base import (
    RawFinding,
    bootstrap_mean_interval,
    make_estimate,
    tokenise,
)
from evalassay.types import ItemSet

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

N_FOLDS: Final = 5
"""Cross-validation folds."""

SMOOTHING: Final = 1.0
"""Laplace smoothing added to every token count."""

RANDOMISATIONS: Final = 4000
"""Draws used for the p-value under the chance null."""

BOOTSTRAP_DRAWS: Final = 2000
"""Resamples used for the interval."""

MIN_ITEMS: Final = 50
"""Below this, cross-validated folds are too small to train anything on."""

_RANDOMISATION_CHUNK: Final = 256
"""Randomisation draws held in memory at once."""


def _build_vocabulary(token_lists: list[list[str]], rows: IntArray) -> dict[str, int]:
    """Map every token seen in the training rows to a dense index.

    Args:
        token_lists: Tokens for each option, flattened across items.
        rows: Indices of the option rows in the training fold.

    Returns:
        Token to index mapping.
    """
    vocabulary: dict[str, int] = {}
    for row in rows:
        for token in token_lists[int(row)]:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def _log_odds(
    token_lists: list[list[str]],
    labels: IntArray,
    train_rows: IntArray,
    vocabulary: dict[str, int],
) -> FloatArray:
    """Fit per-token log odds of being in a correct option.

    Args:
        token_lists: Tokens for each option row.
        labels: 1 for correct options, 0 otherwise.
        train_rows: Option rows in the training fold.
        vocabulary: Token to index mapping.

    Returns:
        One log-odds weight per vocabulary token.
    """
    size = len(vocabulary)
    positive = np.full(size, SMOOTHING)
    negative = np.full(size, SMOOTHING)

    for row in train_rows:
        index = int(row)
        target = positive if labels[index] == 1 else negative
        for token in token_lists[index]:
            target[vocabulary[token]] += 1.0

    weights = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
    return np.asarray(weights, dtype=np.float64)


def _score_option(tokens: list[str], weights: FloatArray, vocabulary: dict[str, int]) -> float:
    """Score one option by summing the log odds of its tokens.

    Tokens unseen in training contribute nothing, which is the correct
    behaviour: an unseen token is evidence for neither class.

    Args:
        tokens: The option's tokens.
        weights: Per-token log odds.
        vocabulary: Token to index mapping.

    Returns:
        The option's score.
    """
    total = 0.0
    for token in tokens:
        index = vocabulary.get(token)
        if index is not None:
            total += float(weights[index])
    return total


@dataclass(frozen=True, slots=True)
class ChoicesOnly:
    """Detector for answer-key leakage through option surface form."""

    name: str = "choices_only"
    assumes_independent_items: bool = True

    def run(self, item_set: ItemSet, rng: np.random.Generator) -> RawFinding | None:
        """Measure how much of the key a probe recovers from options alone.

        Args:
            item_set: The corpus.
            rng: Seeded generator, used for fold assignment and tie-breaking.

        Returns:
            The finding, or ``None`` if the corpus is too small to cross-validate.
        """
        n_items = len(item_set)
        if n_items < MIN_ITEMS:
            return None

        token_lists: list[list[str]] = []
        labels_list: list[int] = []
        item_of_row: list[int] = []
        for index, item in enumerate(item_set):
            for choice_index, choice in enumerate(item.choices):
                token_lists.append(tokenise(choice))
                labels_list.append(1 if choice_index == item.answer_index else 0)
                item_of_row.append(index)
        labels = np.array(labels_list, dtype=np.int64)
        owner = np.array(item_of_row, dtype=np.int64)

        fold_of_item = rng.permutation(n_items) % N_FOLDS
        correct = np.zeros(n_items, dtype=np.float64)

        for fold in range(N_FOLDS):
            test_items = np.flatnonzero(fold_of_item == fold)
            if test_items.size == 0:
                continue
            test_set = {int(index) for index in test_items}
            train_rows = np.array(
                [row for row in range(len(token_lists)) if int(owner[row]) not in test_set],
                dtype=np.int64,
            )
            vocabulary = _build_vocabulary(token_lists, train_rows)
            weights = _log_odds(token_lists, labels, train_rows, vocabulary)

            for item_index in test_items:
                item = item_set.items[int(item_index)]
                base = int(np.searchsorted(owner, item_index))
                scores = np.array(
                    [
                        _score_option(token_lists[base + offset], weights, vocabulary)
                        for offset in range(item.n_choices)
                    ]
                )
                # Break ties at random. Taking the first argmax instead would
                # make a probe that learned nothing always answer position zero,
                # which would silently re-measure position skew as leakage.
                best = np.flatnonzero(scores == scores.max())
                chosen = int(best[rng.integers(best.size)])
                correct[int(item_index)] = 1.0 if chosen == item.answer_index else 0.0

        chance = np.array([1.0 / item.n_choices for item in item_set], dtype=np.float64)
        excess = np.asarray(correct - chance, dtype=np.float64)
        point = float(excess.mean())

        # Null: each item is answered correctly with probability one over its
        # own option count. Simulating this handles corpora whose items differ
        # in option count exactly, where a single binomial test would not.
        simulated = np.empty(RANDOMISATIONS, dtype=np.float64)
        for start in range(0, RANDOMISATIONS, _RANDOMISATION_CHUNK):
            stop = min(start + _RANDOMISATION_CHUNK, RANDOMISATIONS)
            hits = rng.random((stop - start, n_items)) < chance
            simulated[start:stop] = (hits - chance).mean(axis=1)

        extreme = int(np.count_nonzero(np.abs(simulated) >= abs(point)))
        p_value = (1 + extreme) / (RANDOMISATIONS + 1)

        low, high = bootstrap_mean_interval(excess, rng, BOOTSTRAP_DRAWS, alpha=0.01)

        detail = (
            f"probe accuracy {float(correct.mean()):.1%} against "
            f"{float(chance.mean()):.1%} chance, {N_FOLDS}-fold grouped cross-validation; "
            "this is a lower bound, since a stronger probe can only recover more"
        )

        return RawFinding(
            detector=self.name,
            description=(
                "the answer key is partly recoverable from the options alone, "
                "without the question, so part of any score on this benchmark "
                "does not depend on understanding the question"
            ),
            estimate=make_estimate(
                point=point,
                interval=(low, high),
                p_value=p_value,
                n=n_items,
                method="grouped cross-validated naive Bayes on options; randomisation test",
            ),
            detail=detail,
        )
