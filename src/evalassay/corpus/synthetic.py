"""Synthetic corpora with defects planted at known magnitudes.

This module exists so the pathology detectors can be *calibrated* rather than
merely exercised. A detector that fires on a real benchmark tells you nothing
about its false-positive rate; a detector that recovers a planted eight-point
effect within its interval, and stays silent on a matched clean control, has a
known error characteristic.

Design rules that make the calibration sound:

- Option text carries no information about the option's position. If it did,
  planted position bias would leak into the choices-only probe and the two
  effects could not be told apart.
- The choices-only marker is a word *substitution*, not an addition, so it does
  not change option length and cannot be confused with the longest-answer
  defect.
- Each calibration spec should plant one defect at a time. Longest-answer
  padding and lexical leakage share a mechanism - both make the correct option
  distinguishable from its text - so planting them together measures their
  interaction rather than either one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from evalassay.types import Item, ItemSet

VOCABULARY: Final = tuple(
    f"{stem}{suffix}"
    for stem in (
        "lantern",
        "harbour",
        "meadow",
        "cinder",
        "quarry",
        "thistle",
        "beacon",
        "furrow",
        "gable",
        "kestrel",
        "marram",
        "pewter",
        "rowan",
        "sable",
        "tundra",
        "vellum",
    )
    for suffix in ("", "s", "-wise", "-like", "-fold", "-ward", "-most", "-ish")
)
"""A closed vocabulary with no natural correlation to correctness."""

MARKER_WORD: Final = "quorate"
"""Substituted into the correct option to plant lexical answer leakage."""

PAD_WORD: Final = "moreover"
"""Appended to the correct option to plant the longest-answer defect."""

SUBJECTS: Final = ("alpha", "beta", "gamma", "delta")
"""Placeholder subject labels, used to exercise stratified sampling."""

_WORDS_PER_OPTION: Final = 4
_WORDS_PER_QUESTION: Final = 9
_PAD_WORDS: Final = 6
MIN_CHOICES: Final = 2


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """A recipe for a synthetic corpus with known defects.

    Attributes:
        n_items: Number of distinct items before duplication.
        n_choices: Options per item.
        seed: Seed for every draw.
        position_bias: Fraction of items whose correct option is forced to
            ``biased_position``. The rest are placed uniformly at random.
        biased_position: Where biased items put their correct option.
        longest_answer_rate: Fraction of items whose correct option is padded so
            that it is the longest.
        choices_only_rate: Fraction of items in which one word of the correct
            option is replaced by :data:`MARKER_WORD`, making the key
            recoverable from the options alone.
        duplicate_rate: Fraction of items appended a second time under a new
            identifier, planting exact near-duplicates.
    """

    n_items: int = 400
    n_choices: int = 4
    seed: int = 0
    position_bias: float = 0.0
    biased_position: int = 0
    longest_answer_rate: float = 0.0
    choices_only_rate: float = 0.0
    duplicate_rate: float = 0.0

    def __post_init__(self) -> None:
        """Reject recipes that could not be generated.

        Raises:
            ValueError: If a count or a rate is outside its valid range.
        """
        if self.n_items < 1:
            raise ValueError(f"n_items must be positive, got {self.n_items}")
        if self.n_choices < MIN_CHOICES:
            raise ValueError(f"n_choices must be at least {MIN_CHOICES}, got {self.n_choices}")
        if not 0 <= self.biased_position < self.n_choices:
            raise ValueError(
                f"biased_position {self.biased_position} outside 0..{self.n_choices - 1}"
            )
        rates = {
            "position_bias": self.position_bias,
            "longest_answer_rate": self.longest_answer_rate,
            "choices_only_rate": self.choices_only_rate,
            "duplicate_rate": self.duplicate_rate,
        }
        for label, value in rates.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} {value} outside [0, 1]")

    @property
    def expected_position_share(self) -> float:
        """Share of items whose key sits at ``biased_position``, in expectation.

        Biased items land there by construction; unbiased items land there with
        probability ``1 / n_choices``.

        Returns:
            The expected proportion.
        """
        return self.position_bias + (1.0 - self.position_bias) / self.n_choices

    @property
    def expected_longest_answer_accuracy(self) -> float:
        """Accuracy of the longest-option heuristic, in expectation.

        Padded items are won outright. Unpadded items have options of equal word
        count, so the heuristic is reduced to guessing among ties.

        Returns:
            The expected proportion.
        """
        return self.longest_answer_rate + (1.0 - self.longest_answer_rate) / self.n_choices

    @property
    def expected_choices_only_accuracy(self) -> float:
        """Accuracy recoverable from the options alone, in expectation.

        Marked items are won outright by any probe that notices the marker.
        Unmarked items carry no signal, leaving chance.

        Returns:
            The expected proportion.
        """
        return self.choices_only_rate + (1.0 - self.choices_only_rate) / self.n_choices


def _draw_words(rng: np.random.Generator, count: int) -> list[str]:
    """Draw distinct vocabulary words.

    Args:
        rng: Seeded generator.
        count: How many words to draw.

    Returns:
        The drawn words.
    """
    indices = rng.choice(len(VOCABULARY), size=count, replace=False)
    return [VOCABULARY[int(i)] for i in indices]


def generate(spec: CorpusSpec) -> ItemSet:
    """Generate a synthetic corpus from a recipe.

    Args:
        spec: The recipe.

    Returns:
        The corpus, named to record the recipe's seed so two corpora generated
        from different seeds can never be confused in a report.
    """
    rng = np.random.default_rng(spec.seed)
    items: list[Item] = []

    biased_flags = rng.random(spec.n_items) < spec.position_bias
    padded_flags = rng.random(spec.n_items) < spec.longest_answer_rate
    marked_flags = rng.random(spec.n_items) < spec.choices_only_rate

    for index in range(spec.n_items):
        question = " ".join(_draw_words(rng, _WORDS_PER_QUESTION)) + "?"

        # Option text is drawn independently of where the option will sit, so
        # position carries no lexical signal.
        options = [" ".join(_draw_words(rng, _WORDS_PER_OPTION)) for _ in range(spec.n_choices)]

        answer_index = (
            spec.biased_position if biased_flags[index] else int(rng.integers(spec.n_choices))
        )

        if marked_flags[index]:
            words = options[answer_index].split()
            words[int(rng.integers(len(words)))] = MARKER_WORD
            options[answer_index] = " ".join(words)

        if padded_flags[index]:
            options[answer_index] = options[answer_index] + " " + " ".join([PAD_WORD] * _PAD_WORDS)

        items.append(
            Item(
                item_id=f"syn-{index:06d}",
                question=question,
                choices=tuple(options),
                answer_index=answer_index,
                subject=SUBJECTS[index % len(SUBJECTS)],
            )
        )

    if spec.duplicate_rate > 0.0:
        duplicated = rng.random(spec.n_items) < spec.duplicate_rate
        for index in range(spec.n_items):
            if duplicated[index]:
                original = items[index]
                items.append(
                    Item(
                        item_id=f"syn-dup-{index:06d}",
                        question=original.question,
                        choices=original.choices,
                        answer_index=original.answer_index,
                        subject=original.subject,
                    )
                )

    return ItemSet(name=f"synthetic[seed={spec.seed}]", items=tuple(items))
