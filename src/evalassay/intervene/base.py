"""Paired interventions, and the rules for composing them.

An intervention turns one item into one or more *variants* to be scored. The
audit compares a model's accuracy across coalitions of interventions applied
together, so two properties are non-negotiable.

**Determinism per item.** An intervention that makes a random choice - which
distractor to swap in, for instance - must make the *same* choice for a given
item in every coalition it appears in. If it did not, the coalition accuracies
would differ for a reason unrelated to the interventions, and the Shapley
decomposition built on them would be attributing noise. Each intervention
therefore derives its generator from the run seed, its own name, and the item
identifier, and never from a shared stream consumed in coalition order.

**A canonical composition order.** Coalitions are sets, not sequences, so
applying the same set in two different orders must give the same result. The
order in :data:`CANONICAL_ORDER` is fixed and every coalition follows it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, Protocol

import numpy as np

from evalassay.types import Item

_SEED_MODULUS: Final = 2**32

CANONICAL_ORDER: Final = (
    "stronger_distractor",
    "neutral_reframing",
    "hide_question",
    "permute_options",
)
"""The order interventions compose in, whatever order a coalition names them.

Chosen so each step sees the output of the ones it depends on: the distractor
swap rewrites options first; reframing then rewrites the question; hiding the
question comes after reframing, so that a coalition containing both ends with no
question at all and the reframing is naturally a no-op; and option permutation
runs last so it reorders the final set of options.
"""


class Intervention(Protocol):
    """The interface every intervention implements.

    Declared with read-only properties so implementations can be frozen
    dataclasses; a plain variable member would demand a settable attribute.
    """

    @property
    def name(self) -> str:
        """Machine-readable name, and the key used for composition order."""
        ...

    @property
    def description(self) -> str:
        """One line a reader can understand without the source."""
        ...

    def transform(self, item: Item, rng: np.random.Generator) -> tuple[Item, ...]:
        """Turn one item into the variants this intervention produces.

        Args:
            item: The item, already transformed by earlier interventions in
                :data:`CANONICAL_ORDER`.
            rng: A generator derived from the run seed, this intervention's
                name, and the item identifier.

        Returns:
            One or more variants. Most interventions return exactly one; option
            permutation returns one per position so that positional preference
            is measured against every placement of the key rather than one.
        """
        ...


def intervention_rng(seed: int, name: str, item_id: str) -> np.random.Generator:
    """Derive an intervention's generator for one item.

    Folding the item identifier in is what makes a random choice stable across
    coalitions: the same item always gets the same draw, whichever other
    interventions happen to be applied alongside.

    Args:
        seed: Run seed.
        name: Intervention name.
        item_id: Identifier of the item being transformed.

    Returns:
        A seeded generator.
    """
    payload = f"{name}\x00{item_id}".encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return np.random.default_rng((seed + offset) % _SEED_MODULUS)


def order_key(intervention: Intervention) -> int:
    """Position of an intervention in the canonical composition order.

    Args:
        intervention: The intervention.

    Returns:
        Its index in :data:`CANONICAL_ORDER`.

    Raises:
        ValueError: If the intervention's name is not in the canonical order,
            which would leave composition ambiguous.
    """
    try:
        return CANONICAL_ORDER.index(intervention.name)
    except ValueError as exc:
        raise ValueError(
            f"intervention {intervention.name!r} has no place in the canonical "
            f"composition order {CANONICAL_ORDER}; add it there so coalitions "
            "compose in a defined sequence"
        ) from exc


def apply_coalition(
    interventions: Sequence[Intervention],
    item: Item,
    seed: int,
) -> tuple[Item, ...]:
    """Apply a set of interventions to one item, in canonical order.

    Args:
        interventions: The coalition. May be given in any order.
        item: The untouched item.
        seed: Run seed.

    Returns:
        The variants to score. An empty coalition returns the item unchanged,
        which is the control condition every comparison is paired against.
    """
    variants: tuple[Item, ...] = (item,)
    for intervention in sorted(interventions, key=order_key):
        rng = intervention_rng(seed, intervention.name, item.item_id)
        variants = tuple(
            produced for variant in variants for produced in intervention.transform(variant, rng)
        )
    return variants
