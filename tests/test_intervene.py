"""Interventions: what they change, what they must not, and how they compose."""

from __future__ import annotations

import numpy as np
import pytest

from evalassay.intervene.base import (
    CANONICAL_ORDER,
    Intervention,
    apply_coalition,
    intervention_rng,
    order_key,
)
from evalassay.intervene.interventions import (
    REFRAME_PREFIX,
    HideQuestion,
    NeutralReframing,
    PermuteOptions,
    StrongerDistractor,
    distractor_pool,
)
from evalassay.types import Item, ItemSet


def _item(answer_index: int = 1, n_choices: int = 4) -> Item:
    return Item(
        item_id="x1",
        question="Which is a metal?",
        choices=tuple(f"option-{i}" for i in range(n_choices)),
        answer_index=answer_index,
        subject="science",
    )


# --------------------------------------------------------------------------
# Option permutation
# --------------------------------------------------------------------------


def test_permutation_places_the_key_at_every_position() -> None:
    variants = PermuteOptions().transform(_item(), np.random.default_rng(0))
    assert len(variants) == 4
    assert [v.answer_index for v in variants] == [0, 1, 2, 3]


def test_permutation_keeps_the_key_pointing_at_the_same_text() -> None:
    item = _item(answer_index=2)
    for variant in PermuteOptions().transform(item, np.random.default_rng(0)):
        assert variant.answer_index == variant.choices.index(item.answer)
        assert variant.answer == item.answer


def test_permutation_preserves_the_option_multiset() -> None:
    item = _item()
    for variant in PermuteOptions().transform(item, np.random.default_rng(0)):
        assert sorted(variant.choices) == sorted(item.choices)


def test_permutation_preserves_relative_order_cyclically() -> None:
    # Rotation, not shuffle: the author's intended ordering survives, so only
    # absolute position changes.
    item = _item()
    for variant in PermuteOptions().transform(item, np.random.default_rng(0)):
        start = variant.choices.index(item.choices[0])
        rotated = tuple(variant.choices[(start + j) % 4] for j in range(4))
        assert rotated == item.choices


def test_permutation_preserves_the_option_count() -> None:
    # Chance accuracy must be identical in every condition.
    for variant in PermuteOptions().transform(_item(n_choices=5), np.random.default_rng(0)):
        assert variant.n_choices == 5


def test_permutation_ignores_the_generator() -> None:
    first = PermuteOptions().transform(_item(), np.random.default_rng(1))
    second = PermuteOptions().transform(_item(), np.random.default_rng(999))
    assert first == second


# --------------------------------------------------------------------------
# Hiding the question
# --------------------------------------------------------------------------


def test_hiding_empties_the_question_and_leaves_the_options() -> None:
    item = _item()
    (variant,) = HideQuestion().transform(item, np.random.default_rng(0))
    assert variant.question == ""
    assert variant.choices == item.choices
    assert variant.answer_index == item.answer_index


# --------------------------------------------------------------------------
# Neutral reframing
# --------------------------------------------------------------------------


def test_reframing_changes_the_stem_but_not_the_options_or_key() -> None:
    item = _item()
    (variant,) = NeutralReframing().transform(item, np.random.default_rng(0))
    assert variant.question != item.question
    assert variant.question.startswith(REFRAME_PREFIX)
    assert item.question in variant.question
    assert variant.choices == item.choices
    assert variant.answer_index == item.answer_index


def test_reframing_swaps_quote_style() -> None:
    item = Item(item_id="q", question="What is a 'metal'?", choices=("a", "b"), answer_index=0)
    (variant,) = NeutralReframing().transform(item, np.random.default_rng(0))
    assert "'" not in variant.question


def test_reframing_leaves_an_already_hidden_question_alone() -> None:
    # So that composing reframing with hiding is a no-op rather than producing
    # a prompt consisting only of the frame.
    hidden = Item(item_id="q", question="", choices=("a", "b"), answer_index=0)
    (variant,) = NeutralReframing().transform(hidden, np.random.default_rng(0))
    assert variant.question == ""


# --------------------------------------------------------------------------
# Stronger distractors
# --------------------------------------------------------------------------


def _pool_corpus() -> ItemSet:
    return ItemSet(
        name="pool",
        items=tuple(
            Item(
                item_id=f"p{i}",
                question=f"question {i}",
                choices=(f"true-statement-{i}", f"false-{i}a", f"false-{i}b", f"false-{i}c"),
                answer_index=0,
                subject="science",
            )
            for i in range(10)
        ),
    )


def test_stronger_distractor_preserves_the_option_count() -> None:
    # Adding an option would drop accuracy mechanically even for a uniform
    # guesser, and the audit would charge that arithmetic to the model.
    intervention = StrongerDistractor(pool=distractor_pool(_pool_corpus()))
    (variant,) = intervention.transform(_item(), np.random.default_rng(3))
    assert variant.n_choices == 4


def test_stronger_distractor_never_touches_the_key() -> None:
    item = _item(answer_index=2)
    intervention = StrongerDistractor(pool=distractor_pool(_pool_corpus()))
    (variant,) = intervention.transform(item, np.random.default_rng(3))
    assert variant.answer_index == 2
    assert variant.answer == item.answer


def test_stronger_distractor_changes_exactly_one_wrong_option() -> None:
    item = _item()
    intervention = StrongerDistractor(pool=distractor_pool(_pool_corpus()))
    (variant,) = intervention.transform(item, np.random.default_rng(3))
    differences = [i for i in range(4) if variant.choices[i] != item.choices[i]]
    assert len(differences) == 1
    assert differences[0] != item.answer_index


def test_stronger_distractor_draws_from_the_pool() -> None:
    corpus = _pool_corpus()
    intervention = StrongerDistractor(pool=distractor_pool(corpus))
    (variant,) = intervention.transform(_item(), np.random.default_rng(3))
    answers = {item.answer for item in corpus}
    assert any(choice in answers for choice in variant.choices)


def test_stronger_distractor_is_inert_without_a_pool() -> None:
    item = _item()
    (variant,) = StrongerDistractor().transform(item, np.random.default_rng(3))
    assert variant == item


def test_stronger_distractor_is_inert_when_every_candidate_is_already_present() -> None:
    item = Item(item_id="x", question="q", choices=("alpha", "beta"), answer_index=0)
    pool = (("", "alpha"), ("", "beta"))
    (variant,) = StrongerDistractor(pool=pool).transform(item, np.random.default_rng(3))
    assert variant == item


def test_stronger_distractor_prefers_the_same_subject() -> None:
    pool = (("other", "off-topic statement"), ("science", "on-topic statement"))
    intervention = StrongerDistractor(pool=pool)
    (variant,) = intervention.transform(_item(), np.random.default_rng(3))
    assert "on-topic statement" in variant.choices
    assert "off-topic statement" not in variant.choices


def test_distractor_pool_can_exclude_an_item() -> None:
    corpus = _pool_corpus()
    pool = distractor_pool(corpus, exclude_item_id="p3")
    assert len(pool) == len(corpus) - 1
    assert all(text != "true-statement-3" for _, text in pool)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_an_empty_coalition_is_the_identity() -> None:
    item = _item()
    assert apply_coalition([], item, seed=1) == (item,)


def test_composition_does_not_depend_on_the_order_given() -> None:
    # Coalitions are sets. Applying the same set in two orders must agree, or
    # the coalition accuracies feeding the decomposition would be ambiguous.
    item = _item()
    pool = distractor_pool(_pool_corpus())
    members: list[Intervention] = [
        StrongerDistractor(pool=pool),
        NeutralReframing(),
        PermuteOptions(),
    ]
    forward = apply_coalition(members, item, seed=5)
    backward = apply_coalition(list(reversed(members)), item, seed=5)
    assert forward == backward


def test_hiding_the_question_absorbs_reframing() -> None:
    item = _item()
    both = apply_coalition([NeutralReframing(), HideQuestion()], item, seed=5)
    hidden_only = apply_coalition([HideQuestion()], item, seed=5)
    assert both == hidden_only
    assert both[0].question == ""


def test_permutation_multiplies_the_variant_count() -> None:
    item = _item()
    pool = distractor_pool(_pool_corpus())
    members: list[Intervention] = [StrongerDistractor(pool=pool), PermuteOptions()]
    variants = apply_coalition(members, item, seed=5)
    assert len(variants) == 4
    assert sorted(v.answer_index for v in variants) == [0, 1, 2, 3]


def test_a_random_choice_is_stable_across_coalitions() -> None:
    # The same item must get the same distractor whichever other interventions
    # are applied alongside, or the pairing that the whole decomposition rests
    # on would be comparing different items.
    item = _item()
    pool = distractor_pool(_pool_corpus())
    distractor = StrongerDistractor(pool=pool)

    alone = apply_coalition([distractor], item, seed=5)[0]
    with_reframing = apply_coalition([distractor, NeutralReframing()], item, seed=5)[0]
    with_permutation = apply_coalition([distractor, PermuteOptions()], item, seed=5)

    assert sorted(alone.choices) == sorted(with_reframing.choices)
    assert sorted(alone.choices) == sorted(with_permutation[0].choices)


def test_composition_is_reproducible_from_the_seed() -> None:
    item = _item()
    pool = distractor_pool(_pool_corpus())
    members: list[Intervention] = [StrongerDistractor(pool=pool), NeutralReframing()]
    assert apply_coalition(members, item, seed=8) == apply_coalition(members, item, seed=8)


def test_composition_changes_with_the_seed() -> None:
    item = _item()
    pool = distractor_pool(_pool_corpus())
    outcomes = {
        apply_coalition([StrongerDistractor(pool=pool)], item, seed=seed)[0].choices
        for seed in range(12)
    }
    assert len(outcomes) > 1


def test_intervention_generators_are_stable_and_distinct() -> None:
    first = intervention_rng(1, "a", "item").random()
    assert intervention_rng(1, "a", "item").random() == first
    assert intervention_rng(1, "b", "item").random() != first
    assert intervention_rng(1, "a", "other").random() != first
    assert intervention_rng(2, "a", "item").random() != first


def test_every_intervention_has_a_place_in_the_canonical_order() -> None:
    for intervention in (
        StrongerDistractor(),
        NeutralReframing(),
        HideQuestion(),
        PermuteOptions(),
    ):
        assert 0 <= order_key(intervention) < len(CANONICAL_ORDER)


def test_an_unplaced_intervention_is_refused() -> None:
    class Rogue:
        name = "rogue"
        description = "not in the canonical order"

        def transform(self, item: Item, rng: np.random.Generator) -> tuple[Item, ...]:
            return (item,)

    with pytest.raises(ValueError, match="no place in the canonical"):
        order_key(Rogue())
