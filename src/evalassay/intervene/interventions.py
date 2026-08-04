"""The four interventions whose joint effect the audit decomposes.

Each one isolates a different way a score can be inflated without the model
understanding anything:

- **Option permutation** separates knowing the answer from preferring a
  position.
- **Hiding the question** separates answering the question from recognising the
  answer.
- **Neutral reframing** separates understanding from memorising an exact string.
- **Stronger distractors** separate knowing the right answer from knowing that
  the alternatives were weak.

Every one preserves the number of options, so chance accuracy is identical in
every condition. That is deliberate: an intervention that added an option would
drop accuracy mechanically, even for a model that guesses uniformly, and the
audit would charge that arithmetic to the model as an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from evalassay.hashing import normalise_text
from evalassay.types import Item, ItemSet

REFRAME_PREFIX: Final = "Consider the following. "
"""A semantically empty frame, prepended to change surface form and nothing else."""

MIN_OPTIONS_TO_SWAP: Final = 2
"""An item with fewer options has no wrong option to replace."""

_QUOTE_SWAPS: Final = (("'", "’"), ('"', "“"))
"""Straight to typographic quotes: different bytes, identical meaning."""


@dataclass(frozen=True, slots=True)
class PermuteOptions:
    """Place the correct option at every position in turn.

    Produces one variant per option, each a cyclic rotation of the original
    list. Rotation is used rather than an arbitrary shuffle so that the relative
    order of the options is preserved and *only* absolute position changes. A
    full shuffle would additionally disturb any ordering the item's author
    intended, which is a second effect this intervention is not trying to
    measure.

    Because the key visits every position exactly once, a model with a fixed
    positional preference gains on exactly one variant and loses on the rest,
    and the averaged accuracy is stripped of that preference.
    """

    name: str = "permute_options"
    description: str = "correct option placed at every position in turn"

    def transform(self, item: Item, rng: np.random.Generator) -> tuple[Item, ...]:
        """Produce one rotation per option position.

        Args:
            item: The item.
            rng: Unused; rotation is exhaustive, so nothing is left to chance.

        Returns:
            One variant per option.
        """
        del rng
        k = item.n_choices
        variants: list[Item] = []
        for target in range(k):
            shift = (item.answer_index - target) % k
            rotated = tuple(item.choices[(j + shift) % k] for j in range(k))
            variants.append(
                Item(
                    item_id=item.item_id,
                    question=item.question,
                    choices=rotated,
                    answer_index=target,
                    subject=item.subject,
                )
            )
        return tuple(variants)


@dataclass(frozen=True, slots=True)
class HideQuestion:
    """Remove the question entirely, leaving only the options.

    Whatever accuracy survives this is accuracy that never depended on the
    question. It is the sharpest of the four: a model scoring well above chance
    with nothing to answer is, on those items, not answering anything.
    """

    name: str = "hide_question"
    description: str = "question removed, leaving only the options"

    def transform(self, item: Item, rng: np.random.Generator) -> tuple[Item, ...]:
        """Blank the question stem.

        Args:
            item: The item.
            rng: Unused; the transformation is deterministic.

        Returns:
            A single variant with an empty question.
        """
        del rng
        return (
            Item(
                item_id=item.item_id,
                question="",
                choices=item.choices,
                answer_index=item.answer_index,
                subject=item.subject,
            ),
        )


@dataclass(frozen=True, slots=True)
class NeutralReframing:
    """Change the question's surface form without changing what it asks.

    Applies a semantically empty prefix and swaps straight quotes for
    typographic ones. Both change the token sequence a model sees while leaving
    the question identical in meaning, so any accuracy lost is accuracy that
    depended on the exact string rather than on its content.

    **This is deliberately a weak rewrite, and it is named for what it does
    rather than for what one might wish it did.** A genuine paraphrase would
    detect far more memorisation, but generating one needs a model in the loop,
    which would make the intervention non-deterministic and would put an
    unverified claim - that meaning was preserved - underneath every number in
    the report. A weak, provably meaning-preserving rewrite gives a lower bound
    on format brittleness. A strong, unverifiable one would give a number nobody
    could check.
    """

    name: str = "neutral_reframing"
    description: str = "question reframed with a semantically empty prefix"

    def transform(self, item: Item, rng: np.random.Generator) -> tuple[Item, ...]:
        """Reframe the question stem.

        Args:
            item: The item.
            rng: Unused; the transformation is deterministic.

        Returns:
            A single reframed variant. An item whose question is already empty
            is returned unchanged, so that composing this with question hiding
            is a no-op rather than an error.
        """
        del rng
        if not item.question.strip():
            return (item,)

        reframed = item.question
        for straight, typographic in _QUOTE_SWAPS:
            reframed = reframed.replace(straight, typographic)

        return (
            Item(
                item_id=item.item_id,
                question=REFRAME_PREFIX + reframed,
                choices=item.choices,
                answer_index=item.answer_index,
                subject=item.subject,
            ),
        )


@dataclass(frozen=True, slots=True)
class StrongerDistractor:
    """Replace one wrong option with a plausible statement from another item.

    Benchmarks are often easier than they look because their distractors are
    obviously wrong - misspelled, off-topic, or grammatically odd. A model can
    then reach the key by eliminating nonsense rather than by knowing anything.

    This swaps one incorrect option for the *correct answer to a different
    question*, preferring one from the same subject. Such a statement is
    well-formed, on-topic and true in its own context, and wrong here. Accuracy
    lost to it is accuracy that came from the distractors being weak.

    An option is replaced rather than added, so the option count and therefore
    chance accuracy are unchanged.

    Attributes:
        pool: ``(subject, text)`` pairs to draw replacements from, normally the
            correct answers of the other items in the corpus.
    """

    pool: tuple[tuple[str, str], ...] = field(default=())
    name: str = "stronger_distractor"
    description: str = "one weak distractor replaced by a plausible statement"

    def transform(self, item: Item, rng: np.random.Generator) -> tuple[Item, ...]:
        """Swap in a stronger distractor, if a usable one exists.

        Args:
            item: The item.
            rng: Generator derived from the seed, this intervention's name and
                the item identifier, so the same swap happens in every coalition.

        Returns:
            A single variant. The item is returned unchanged when no usable
            replacement exists - every candidate already appears among the
            options, or the pool is empty. Silently leaving the item alone is
            correct here: the intervention had no effect on this item, and the
            paired comparison will record exactly that.
        """
        if not self.pool or item.n_choices < MIN_OPTIONS_TO_SWAP:
            return (item,)

        existing = {normalise_text(choice) for choice in item.choices}
        same_subject = [text for subject, text in self.pool if subject == item.subject]
        candidates = same_subject if same_subject else [text for _, text in self.pool]
        usable = [text for text in candidates if normalise_text(text) not in existing]
        if not usable:
            return (item,)

        replacement = usable[int(rng.integers(len(usable)))]
        wrong_positions = [i for i in range(item.n_choices) if i != item.answer_index]
        target = wrong_positions[int(rng.integers(len(wrong_positions)))]

        choices = list(item.choices)
        choices[target] = replacement
        return (
            Item(
                item_id=item.item_id,
                question=item.question,
                choices=tuple(choices),
                answer_index=item.answer_index,
                subject=item.subject,
            ),
        )


def distractor_pool(
    item_set: ItemSet, exclude_item_id: str | None = None
) -> tuple[tuple[str, str], ...]:
    """Build a replacement pool from a corpus's correct answers.

    Args:
        item_set: The corpus.
        exclude_item_id: Optional item to leave out.

    Returns:
        ``(subject, answer text)`` pairs, in corpus order.
    """
    return tuple(
        (item.subject, item.answer) for item in item_set.items if item.item_id != exclude_item_id
    )
