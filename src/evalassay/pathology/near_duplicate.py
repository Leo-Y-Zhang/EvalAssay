"""Does the benchmark contain the same item more than once?

Repeated items distort a score twice over. They weight whatever the repeat
happens to test more heavily than the rest of the benchmark, and if a repeated
pair carries *different* answer keys then the benchmark contradicts itself and
no model can score full marks on both.

Three passes, in descending order of how firmly each conclusion can be stated:

1. **Exact repeats.** Items whose question, options and key are all identical
   after normalisation. Found by grouping on a content digest, so the pass is
   complete and needs no threshold.
2. **Contradictions.** Items posing an identical question with identical
   options, but disagreeing about the key. Found by grouping on a digest that
   *excludes* the key. This also needs no threshold, which matters: it is the
   most damaging thing this detector can say, so it is the claim least willing
   to rest on a judgement call.
3. **Near repeats.** Items that are merely similar, measured by Jaccard overlap
   of token **shingles** rather than of bare tokens.

Why shingles, which is a correction
-----------------------------------
An earlier version measured similarity between token *sets*. Two MMLU items
scored a Jaccard above 0.9 under it:

    Statement 1 | The function *f* must necessarily be injective ...
    Statement 1 | The function *g* must necessarily be injective ...

Their token sets are identical, because both questions mention both ``f`` and
``g`` in their shared setup. They are nonetheless different questions with
correctly different keys, and the detector reported them as a contradiction in
the benchmark. Set overlap discards word order and multiplicity, which is
precisely the information that distinguishes those two items.

Shingles - overlapping runs of consecutive tokens - keep local order, so the two
items differ on every shingle spanning the changed word. The contradiction claim
no longer depends on this measure at all, and near repeats are reported as what
they are: an approximate signal worth a human look, not a proof.

Statistical status
------------------
This detector is a census, not an inference. It counts what is there rather than
estimating a population quantity, so its p-value is zero when repeats exist and
one when they do not, and the interval is a Wilson score interval on the rate.
Dressing a complete enumeration up as a hypothesis test would be theatre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from evalassay.hashing import item_digest, stem_digest
from evalassay.pathology.base import RawFinding, make_estimate, tokenise, wilson_interval
from evalassay.types import Item, ItemSet

SHINGLE_SIZE: Final = 3
"""Tokens per shingle. Long enough to carry local order, short enough to survive
small edits."""

JACCARD_THRESHOLD: Final = 0.9
"""Shingle overlap at which two items are treated as near repeats."""

CANDIDATE_SHINGLES: Final = 3
"""How many of an item's rarest shingles generate candidate pairs."""

MAX_CANDIDATES_PER_ITEM: Final = 400
"""Ceiling on candidates per item, so one ubiquitous shingle cannot go quadratic."""

MIN_GROUP_SIZE: Final = 2
"""A repeat needs at least two members."""

MIN_ITEMS: Final = 10
"""Below this a duplicate rate is not worth reporting."""


def shingles(item: Item) -> frozenset[tuple[str, ...]]:
    """Overlapping runs of consecutive tokens across an item's text.

    Args:
        item: The item.

    Returns:
        The distinct shingles. Items shorter than one shingle contribute their
        whole token sequence, so nothing is silently excluded from comparison.
    """
    tokens: list[str] = list(tokenise(item.question))
    for choice in item.choices:
        tokens.extend(tokenise(choice))

    if len(tokens) < SHINGLE_SIZE:
        return frozenset({tuple(tokens)}) if tokens else frozenset()

    return frozenset(
        tuple(tokens[index : index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    )


def jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    """Jaccard similarity between two shingle sets.

    Args:
        left: First shingle set.
        right: Second shingle set.

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
    """Group item indices by full content digest, keeping repeated groups.

    Args:
        item_set: The corpus.

    Returns:
        Digest to the indices sharing it, for digests occurring more than once.
    """
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(item_set.items):
        groups.setdefault(item_digest(item), []).append(index)
    return {digest: indices for digest, indices in groups.items() if len(indices) > 1}


def contradiction_groups(item_set: ItemSet) -> dict[str, list[int]]:
    """Group items posing an identical question that disagree about the key.

    Args:
        item_set: The corpus.

    Returns:
        Stem digest to the indices sharing it, for groups where the answer text
        is not the same throughout. Exact repeats are excluded, since agreeing
        about the key is the opposite of contradicting.
    """
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(item_set.items):
        groups.setdefault(stem_digest(item), []).append(index)

    contradictory: dict[str, list[int]] = {}
    for digest, indices in groups.items():
        if len(indices) < MIN_GROUP_SIZE:
            continue
        answers = {item_set.items[index].answer for index in indices}
        if len(answers) > 1:
            contradictory[digest] = indices
    return contradictory


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


def _candidate_pairs(
    shingle_sets: list[frozenset[tuple[str, ...]]],
) -> tuple[set[tuple[int, int]], bool]:
    """Propose pairs worth comparing, indexing from each item's rarest shingles.

    Indexing from the rare end rather than excluding common shingles matters: an
    earlier version excluded anything appearing in more than a tenth of items,
    which on a small-vocabulary corpus excluded everything, compared nothing,
    and reported a corpus that was a tenth duplicates as clean.

    Args:
        shingle_sets: Shingle set per item.

    Returns:
        The candidate pairs, and whether any item hit the per-item ceiling.
    """
    postings: dict[tuple[str, ...], list[int]] = {}
    for index, item_shingles in enumerate(shingle_sets):
        for shingle in item_shingles:
            postings.setdefault(shingle, []).append(index)

    pairs: set[tuple[int, int]] = set()
    truncated = False

    for index, item_shingles in enumerate(shingle_sets):
        if not item_shingles:
            continue
        rarest = sorted(item_shingles, key=lambda s: (len(postings[s]), s))[:CANDIDATE_SHINGLES]
        seen: set[int] = set()
        for shingle in rarest:
            for other in postings[shingle]:
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
    """Detector for repeated, contradictory and near-repeated items."""

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
        redundant_copies = 0
        for indices in exact_groups.values():
            for index in indices:
                involved[index] = True
            redundant_copies += len(indices) - 1

        contradictions = contradiction_groups(item_set)
        contradictory_items = 0
        for indices in contradictions.values():
            for index in indices:
                involved[index] = True
            contradictory_items += len(indices)

        shingle_sets = [shingles(item) for item in item_set]
        pairs, truncated = _candidate_pairs(shingle_sets)

        near_only_pairs = 0
        for left, right in pairs:
            if jaccard(shingle_sets[left], shingle_sets[right]) < JACCARD_THRESHOLD:
                continue
            involved[left] = True
            involved[right] = True
            if item_digest(item_set.items[left]) != item_digest(item_set.items[right]):
                near_only_pairs += 1

        affected = int(np.count_nonzero(involved))
        rate = affected / n_items
        low, high = wilson_interval(affected, n_items, alpha=0.01)

        detail_parts = [
            f"{affected} of {n_items} items are repeats, contradictions or near-repeats",
            f"{redundant_copies} redundant copies are identical after normalisation",
        ]
        if contradictory_items:
            detail_parts.append(
                f"{contradictory_items} items across {len(contradictions)} groups pose an "
                "identical question with identical options but disagree about the answer, "
                "so no model can score full marks on all of them"
            )
        detail_parts.append(
            f"{near_only_pairs} further pairs exceed shingle Jaccard {JACCARD_THRESHOLD:g} "
            "without being identical; that pass is approximate and worth a human look "
            "rather than a conclusion"
        )
        if truncated:
            detail_parts.append(
                f"per-item candidate ceiling of {MAX_CANDIDATES_PER_ITEM} was reached, so "
                "the near-repeat count is a lower bound; the exact and contradiction "
                "counts are complete"
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
                method="digest grouping for exact and contradictory repeats; "
                "rarest-shingle candidates with Jaccard for near repeats; census",
            ),
            detail="; ".join(detail_parts),
        )
