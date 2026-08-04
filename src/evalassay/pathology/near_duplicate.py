"""Does the benchmark contain the same item more than once?

Repeated items distort a score twice over. They weight whatever the repeat
happens to test more heavily than the rest of the benchmark, and if a repeated
pair carries *different* answer keys then the benchmark contradicts itself and
no model can score full marks on both.

Detection runs in two passes:

1. **Exact repeats**, by grouping items on their content digest. This pass is
   complete and cheap - it never misses one and never has to be truncated.
2. **Near repeats**, by generating candidate pairs from each item's *rarest*
   tokens and then measuring Jaccard overlap exactly.

The second pass indexes on rare tokens rather than skipping common ones. An
earlier version excluded any token appearing in more than a tenth of items,
which silently produced zero candidates on a corpus with a small vocabulary -
every token was too common to index, so nothing was ever compared and the
detector reported a clean bill of health for a corpus that was ten per cent
duplicates. Indexing from the rare end cannot fail that way, because every item
always has a rarest token.

This detector is a census, not an inference. It counts what is there rather than
estimating a population quantity, so its p-value is zero when repeats exist and
one when they do not, and the interval is a Wilson score interval on the rate.
Dressing a complete enumeration up as a hypothesis test would be theatre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from evalassay.hashing import item_digest
from evalassay.pathology.base import RawFinding, make_estimate, tokenise, wilson_interval
from evalassay.types import Item, ItemSet

JACCARD_THRESHOLD: Final = 0.9
"""Token-set overlap at which two items are treated as the same item."""

CANDIDATE_TOKENS: Final = 3
"""How many of an item's rarest tokens generate candidate pairs."""

MAX_CANDIDATES_PER_ITEM: Final = 400
"""Ceiling on candidates per item, so one ubiquitous token cannot go quadratic."""

MIN_ITEMS: Final = 10
"""Below this a duplicate rate is not worth reporting."""


def _tokens_of(item: Item) -> frozenset[str]:
    """Token set covering an item's question and all of its options.

    Args:
        item: The item.

    Returns:
        The set of distinct tokens.
    """
    tokens: set[str] = set(tokenise(item.question))
    for choice in item.choices:
        tokens.update(tokenise(choice))
    return frozenset(tokens)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity between two token sets.

    Args:
        left: First token set.
        right: Second token set.

    Returns:
        Intersection over union, or ``0.0`` if both sets are empty.
    """
    if not left and not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def exact_duplicate_groups(item_set: ItemSet) -> dict[str, list[int]]:
    """Group item indices by content digest, keeping only repeated groups.

    Args:
        item_set: The corpus.

    Returns:
        Digest to the indices sharing it, for digests occurring more than once.
    """
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(item_set.items):
        groups.setdefault(item_digest(item), []).append(index)
    return {digest: indices for digest, indices in groups.items() if len(indices) > 1}


def deduplicated(item_set: ItemSet) -> tuple[ItemSet, int]:
    """Keep the first occurrence of each distinct item.

    Detectors that assume items are independent draws need this: an exact
    repeat is not a second observation, and leaving repeats in place shrinks the
    effective sample size without shrinking the nominal one, which makes every
    test built on it anti-conservative.

    Args:
        item_set: The corpus.

    Returns:
        The deduplicated corpus and how many items were removed. The corpus is
        returned unchanged, and the count is zero, when there were no repeats.
    """
    seen: set[str] = set()
    kept: list[Item] = []
    for item in item_set.items:
        digest = item_digest(item)
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(item)

    removed = len(item_set) - len(kept)
    if removed == 0:
        return item_set, 0
    return ItemSet(name=f"{item_set.name}[deduplicated]", items=tuple(kept)), removed


def _candidate_pairs(token_sets: list[frozenset[str]]) -> tuple[set[tuple[int, int]], bool]:
    """Propose pairs worth comparing, indexing from each item's rarest tokens.

    Args:
        token_sets: Token set per item.

    Returns:
        The candidate pairs, and whether any item hit the per-item ceiling.
    """
    postings: dict[str, list[int]] = {}
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            postings.setdefault(token, []).append(index)

    pairs: set[tuple[int, int]] = set()
    truncated = False

    for index, tokens in enumerate(token_sets):
        if not tokens:
            continue
        rarest = sorted(tokens, key=lambda t: (len(postings[t]), t))[:CANDIDATE_TOKENS]
        seen: set[int] = set()
        for token in rarest:
            for other in postings[token]:
                if other == index:
                    continue
                if len(seen) >= MAX_CANDIDATES_PER_ITEM:
                    truncated = True
                    break
                seen.add(other)
            if truncated:
                break
        for other in seen:
            pairs.add((index, other) if index < other else (other, index))

    return pairs, truncated


@dataclass(frozen=True, slots=True)
class NearDuplicate:
    """Detector for repeated and near-repeated items."""

    name: str = "near_duplicate"
    # Repeats are the subject of this detector, so it must see the raw corpus.
    assumes_independent_items: bool = False

    def run(self, item_set: ItemSet, rng: np.random.Generator) -> RawFinding | None:
        """Count items that appear more than once.

        Args:
            item_set: The corpus.
            rng: Unused; the census is deterministic. Accepted so every detector
                shares one interface.

        Returns:
            The finding, or ``None`` if the corpus is too small.
        """
        del rng
        n_items = len(item_set)
        if n_items < MIN_ITEMS:
            return None

        involved = np.zeros(n_items, dtype=bool)

        exact_groups = exact_duplicate_groups(item_set)
        exact_extra = 0
        for indices in exact_groups.values():
            for index in indices:
                involved[index] = True
            exact_extra += len(indices) - 1

        token_sets = [_tokens_of(item) for item in item_set]
        pairs, truncated = _candidate_pairs(token_sets)

        near_only_pairs = 0
        contradictory_pairs = 0
        for left, right in pairs:
            if jaccard(token_sets[left], token_sets[right]) < JACCARD_THRESHOLD:
                continue
            involved[left] = True
            involved[right] = True
            if item_digest(item_set.items[left]) != item_digest(item_set.items[right]):
                near_only_pairs += 1
                if item_set.items[left].answer != item_set.items[right].answer:
                    contradictory_pairs += 1

        affected = int(np.count_nonzero(involved))
        rate = affected / n_items
        low, high = wilson_interval(affected, n_items, alpha=0.01)

        detail_parts = [
            f"{affected} of {n_items} items are repeats or near-repeats",
            f"{exact_extra} redundant copies are identical after normalisation",
            f"{near_only_pairs} further pairs exceed Jaccard {JACCARD_THRESHOLD:g} without "
            "being identical",
        ]
        if contradictory_pairs:
            detail_parts.append(
                f"{contradictory_pairs} near-duplicate pairs disagree about the correct "
                "answer, so the benchmark contradicts itself on them and no model can "
                "score full marks on both"
            )
        if truncated:
            detail_parts.append(
                f"per-item candidate ceiling of {MAX_CANDIDATES_PER_ITEM} was reached, so "
                "the near-repeat count is a lower bound; the exact-repeat count is complete"
            )

        return RawFinding(
            detector=self.name,
            description=(
                "the benchmark contains repeated items, which over-weight whatever "
                "they test and, where their keys disagree, make full marks impossible"
            ),
            estimate=make_estimate(
                point=rate,
                interval=(low, high),
                # A census: existence is observed, not inferred.
                p_value=0.0 if affected else 1.0,
                n=n_items,
                method="digest grouping plus rarest-token candidate Jaccard; census",
            ),
            detail="; ".join(detail_parts),
        )
