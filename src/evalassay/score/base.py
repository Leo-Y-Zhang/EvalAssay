"""The scoring interface, tie-breaking, and the score cache.

A scorer turns an item into one score per option; higher means more likely. The
audit takes the argmax and compares it to the key.

**A scorer must never read ``answer_index``.** It is present on the item only
because the audit needs it to mark the answer, and a scorer that consulted it
would report perfect accuracy in every condition. The contract is enforced by a
test that alters ``answer_index`` and requires the returned scores to be
identical, which every real backend must pass.

Ties are broken by hashing each tied option's own text, never by taking the
first maximum and never by anything that depends on option order. Taking the
first would make a scorer that cannot separate the options answer position zero
every time, which the audit would measure as a positional preference the model
does not have; keying on the ordered list would let a rotation change the winner.
Both would manufacture the very artifact this package exists to detect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from evalassay.hashing import normalise_text, prompt_digest
from evalassay.types import Item

FloatArray = NDArray[np.float64]


class Scorer(Protocol):
    """The interface every scoring backend implements."""

    @property
    def scorer_id(self) -> str:
        """Identifier of the backend and model, recorded in the run manifest."""
        ...

    @property
    def deterministic(self) -> bool:
        """Whether identical input is guaranteed to give identical output.

        Recorded in the manifest rather than assumed. A hosted API is not
        deterministic across time even at temperature zero, and a report that
        claimed reproducibility it could not deliver would be worse than one
        that stated the gap plainly.
        """
        ...

    def score(self, item: Item) -> FloatArray:
        """Score every option of an item.

        Args:
            item: The item, or a variant of one. Implementations must use only
                ``question`` and ``choices``.

        Returns:
            One score per option, higher meaning more likely.
        """
        ...


def _tie_break_key(question: str, choice: str) -> str:
    """Order-invariant sort key for one tied option.

    Depends on the question and on the option's own text, and on nothing about
    where the option sits.

    Args:
        question: The question as presented.
        choice: The option's text.

    Returns:
        A hex digest to sort by.
    """
    payload = f"{normalise_text(question)}\x00{normalise_text(choice)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def break_ties(scores: FloatArray, question: str, choices: Sequence[str]) -> int:
    """Choose an option from a score vector, breaking ties by option identity.

    Ties are resolved by hashing each tied option's *own* text together with the
    question, and taking the lowest digest. The result is deterministic, varies
    from item to item, and - the property that matters - **does not depend on
    the order the options appear in**.

    An earlier version hashed the whole ordered option list and used the result
    to index into the tied set. That was a position dependence living inside the
    mechanism meant to be free of one: rotating an item's options could change
    which tied option won, so the permutation intervention was not perfectly
    inert under continuation scoring even though scoring options independently
    means position cannot reach the model at all. The leak was small - bounded by
    the runs at exactly zero on one corpus and 0.0011 on another - but it was a
    positional artifact manufactured by the instrument, which is precisely the
    class of error this project exists to object to.

    Taking the first maximum instead would be far worse: any scorer that cannot
    separate the options would answer position zero every time, and the audit
    would measure that as a positional preference the model does not have.

    Args:
        scores: One score per option.
        question: The question as presented.
        choices: The options as presented.

    Returns:
        The index of the chosen option.

    Raises:
        ValueError: If the score vector is empty or the wrong length.
    """
    if scores.size == 0:
        raise ValueError("cannot choose from an empty score vector")
    if scores.size != len(choices):
        raise ValueError(f"got {scores.size} scores for {len(choices)} choices")

    best = np.flatnonzero(scores == scores.max())
    if best.size == 1:
        return int(best[0])

    return min(
        (int(index) for index in best),
        key=lambda index: _tie_break_key(question, choices[index]),
    )


def predict(scorer: Scorer, item: Item) -> int:
    """Score an item and return the chosen option index.

    Args:
        scorer: The backend.
        item: The item or variant.

    Returns:
        The index of the chosen option.
    """
    return break_ties(scorer.score(item), item.question, item.choices)


def is_correct(scorer: Scorer, item: Item) -> bool:
    """Whether the scorer picks the key.

    Args:
        scorer: The backend.
        item: The item or variant.

    Returns:
        ``True`` if the chosen option is the correct one.
    """
    return predict(scorer, item) == item.answer_index


class ScoreCache:
    """An in-memory cache of score vectors, keyed by prompt content.

    Coalitions overlap heavily - an item with no applicable interventions
    presents the identical prompt in many of them - so caching removes most of
    the scoring work without changing a single number.

    The key folds in the scorer identity, so scores from one model can never be
    served for another.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._entries: dict[str, FloatArray] = {}
        self._hits = 0
        self._misses = 0

    def score(self, scorer: Scorer, item: Item) -> FloatArray:
        """Return the score vector for an item, computing it only once.

        Args:
            scorer: The backend.
            item: The item or variant.

        Returns:
            One score per option.
        """
        key = prompt_digest(scorer.scorer_id, item.question, item.choices)
        cached = self._entries.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        computed = scorer.score(item)
        self._entries[key] = computed
        return computed

    def is_correct(self, scorer: Scorer, item: Item) -> bool:
        """Whether the scorer picks the key, using the cache.

        Args:
            scorer: The backend.
            item: The item or variant.

        Returns:
            ``True`` if the chosen option is the correct one.
        """
        scores = self.score(scorer, item)
        return break_ties(scores, item.question, item.choices) == item.answer_index

    @property
    def hits(self) -> int:
        """How many lookups were served from the cache."""
        return self._hits

    @property
    def misses(self) -> int:
        """How many lookups required scoring."""
        return self._misses

    @property
    def size(self) -> int:
        """How many distinct prompts are cached."""
        return len(self._entries)
