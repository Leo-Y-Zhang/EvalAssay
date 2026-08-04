"""A simulated model whose artifacts are dialled in at known magnitudes.

This is the instrument's reference standard. Every claim EvalAssay makes about a
real model rests on the audit being able to recover an effect that is really
there and stay silent about one that is not, and neither property can be checked
against a real model, because with a real model nobody knows the right answer.

The oracle makes the right answer knowable. It is built from explicit
probabilities - how much genuine skill, how much positional preference, how much
memorisation of exact wording, how much reliance on weak distractors - and the
audit is then asked to take it apart. If it cannot recover what was deliberately
planted, nothing it says about a real model is worth reading.

**How it stays honest about what it is.** The oracle is a test double, and it
uses ``item_id`` to keep its latent state stable across conditions: whether it
"knows" a given item must not change when the options are rotated, exactly as a
real model's knowledge would not. No real backend may do that, and none does.

What the oracle does *not* do is read ``answer_index``. It locates the key by
matching the original answer's text among the presented options, so it satisfies
the same contract as every real backend and is covered by the same test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from evalassay.hashing import normalise_text
from evalassay.types import Item, ItemSet

FloatArray = NDArray[np.float64]

_SEED_MODULUS: Final = 2**32
_LATENTS: Final = 6

_SKILL, _MEMORISATION, _DISTRACTOR, _CHOICES_ONLY, _POSITION, _GUESS = range(_LATENTS)


@dataclass(frozen=True, slots=True)
class OracleSpec:
    """The artifact profile of a simulated model.

    Each probability governs one mechanism, and the mechanisms are tried in a
    fixed order so a run is reproducible and the profile is interpretable.

    Attributes:
        skill: Chance of answering correctly through genuine understanding.
            Requires the question to be present, and survives every
            intervention, because that is what genuine understanding means.
        memorisation: Chance of answering correctly by recognising the exact
            wording. Destroyed by reframing and by hiding the question.
        distractor_reliance: Chance of answering correctly only because the
            wrong options were weak. Destroyed by swapping in a stronger one.
        position_preference: Chance of falling back on a favoured position.
            Neutralised by placing the key at every position in turn.
        favoured_position: The position that preference points at.
        choices_only_skill: Chance of picking the key from the options alone.
            Survives every intervention including hiding the question, so it
            contributes nothing to the decomposition and surfaces instead in the
            blind-accuracy diagnostic.
        seed: Seed for the per-item latent draws.
    """

    skill: float = 0.0
    memorisation: float = 0.0
    distractor_reliance: float = 0.0
    position_preference: float = 0.0
    favoured_position: int = 0
    choices_only_skill: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject profiles that could not be simulated.

        Raises:
            ValueError: If a probability lies outside ``[0, 1]`` or the favoured
                position is negative.
        """
        probabilities = {
            "skill": self.skill,
            "memorisation": self.memorisation,
            "distractor_reliance": self.distractor_reliance,
            "position_preference": self.position_preference,
            "choices_only_skill": self.choices_only_skill,
        }
        for label, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} {value} outside [0, 1]")
        if self.favoured_position < 0:
            raise ValueError(
                f"favoured_position must not be negative, got {self.favoured_position}"
            )

    @property
    def label(self) -> str:
        """A short description of the profile, for reports and manifests.

        Returns:
            The non-zero mechanisms and their magnitudes.
        """
        parts = [
            f"{name}={value:g}"
            for name, value in (
                ("skill", self.skill),
                ("memorisation", self.memorisation),
                ("distractor", self.distractor_reliance),
                ("position", self.position_preference),
                ("choices_only", self.choices_only_skill),
            )
            if value
        ]
        return ",".join(parts) if parts else "inert"


class OracleScorer:
    """A deterministic simulated model built from an :class:`OracleSpec`.

    Attributes:
        spec: The artifact profile being simulated.
    """

    def __init__(self, spec: OracleSpec, corpus: ItemSet) -> None:
        """Build the oracle against the corpus it will be audited on.

        The corpus is needed to recognise what an intervention changed: whether
        the question still reads as it originally did, and whether the wrong
        options are still the original ones. A real model cannot know that and
        does not need to; the oracle needs it to simulate mechanisms that are
        supposed to be destroyed by specific interventions.

        Args:
            spec: The artifact profile.
            corpus: The untouched corpus.
        """
        self.spec = spec
        self._original: dict[str, Item] = {item.item_id: item for item in corpus.items}
        self._latents: dict[str, FloatArray] = {}

    @property
    def scorer_id(self) -> str:
        """Identifier recorded in the run manifest."""
        return f"oracle:{self.spec.label}:seed={self.spec.seed}"

    @property
    def deterministic(self) -> bool:
        """The oracle is a pure function of the item and the spec."""
        return True

    def _latents_for(self, item_id: str) -> FloatArray:
        """Per-item latent draws, stable across every condition.

        Args:
            item_id: The item identifier.

        Returns:
            Uniform draws, one per mechanism.
        """
        cached = self._latents.get(item_id)
        if cached is not None:
            return cached
        payload = f"{self.spec.seed}\x00{item_id}".encode()
        offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        drawn = np.random.default_rng(offset % _SEED_MODULUS).random(_LATENTS)
        self._latents[item_id] = drawn
        return drawn

    def _key_index(self, item: Item) -> int | None:
        """Find the key among the presented options by matching its text.

        Args:
            item: The item or variant.

        Returns:
            The index of the correct option, or ``None`` if the original item is
            unknown or its answer is not among the options.
        """
        original = self._original.get(item.item_id)
        if original is None:
            return None
        target = normalise_text(original.answer)
        for index, choice in enumerate(item.choices):
            if normalise_text(choice) == target:
                return index
        return None

    def score(self, item: Item) -> FloatArray:
        """Score every option, following the profile's mechanisms in order.

        Args:
            item: The item or variant.

        Returns:
            A one-hot score vector selecting the option this model would pick.
        """
        scores = np.zeros(item.n_choices, dtype=np.float64)
        key = self._key_index(item)
        if key is None:
            # Nothing recognisable: behave as an uninformed guesser rather than
            # crashing, so an audit on an unfamiliar corpus degrades gracefully.
            scores[int(self._latents_for(item.item_id)[_GUESS] * item.n_choices)] = 1.0
            return scores

        original = self._original[item.item_id]
        latents = self._latents_for(item.item_id)
        spec = self.spec

        question_present = bool(item.question.strip())
        wording_intact = question_present and normalise_text(item.question) == normalise_text(
            original.question
        )
        original_wrong = {
            normalise_text(choice)
            for index, choice in enumerate(original.choices)
            if index != original.answer_index
        }
        presented_wrong = {
            normalise_text(choice) for index, choice in enumerate(item.choices) if index != key
        }
        distractors_intact = presented_wrong == original_wrong

        if (
            (question_present and latents[_SKILL] < spec.skill)
            or (wording_intact and latents[_MEMORISATION] < spec.memorisation)
            or (distractors_intact and latents[_DISTRACTOR] < spec.distractor_reliance)
            or latents[_CHOICES_ONLY] < spec.choices_only_skill
        ):
            chosen = key
        elif latents[_POSITION] < spec.position_preference:
            chosen = min(spec.favoured_position, item.n_choices - 1)
        else:
            chosen = int(latents[_GUESS] * item.n_choices)

        scores[chosen] = 1.0
        return scores
