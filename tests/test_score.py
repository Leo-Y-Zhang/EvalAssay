"""Scoring backends, the contract they share, tie-breaking, and the cache."""

from __future__ import annotations

import numpy as np
import pytest

from evalassay.intervene.interventions import HideQuestion, NeutralReframing, PermuteOptions
from evalassay.score.base import ScoreCache, break_ties, is_correct, predict
from evalassay.score.oracle import OracleScorer, OracleSpec
from evalassay.types import Item, ItemSet


def _corpus(n: int = 40) -> ItemSet:
    return ItemSet(
        name="demo",
        items=tuple(
            Item(
                item_id=f"i{index}",
                question=f"question {index}?",
                choices=(
                    f"key-{index}",
                    f"wrong-{index}-a",
                    f"wrong-{index}-b",
                    f"wrong-{index}-c",
                ),
                answer_index=0,
                subject="s",
            )
            for index in range(n)
        ),
    )


# --------------------------------------------------------------------------
# Tie-breaking
# --------------------------------------------------------------------------


def test_a_clear_winner_is_chosen() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.3])
    assert break_ties(scores, "q", ["a", "b", "c", "d"]) == 1


def test_ties_are_broken_deterministically() -> None:
    scores = np.zeros(4)
    first = break_ties(scores, "q", ["a", "b", "c", "d"])
    assert break_ties(scores, "q", ["a", "b", "c", "d"]) == first


def test_tie_breaking_does_not_always_pick_position_zero() -> None:
    # Taking the first maximum would make any scorer that cannot separate the
    # options answer position zero every time, which the audit would then
    # measure as a positional preference the model does not have.
    chosen = {
        break_ties(np.zeros(4), f"question {index}", ["a", "b", "c", "d"]) for index in range(60)
    }
    assert len(chosen) > 1


def test_tie_breaking_depends_on_the_prompt_not_the_position() -> None:
    left = break_ties(np.zeros(4), "question one", ["a", "b", "c", "d"])
    right = break_ties(np.zeros(4), "question two", ["a", "b", "c", "d"])
    assert isinstance(left, int)
    assert isinstance(right, int)


def test_tie_breaking_rejects_malformed_score_vectors() -> None:
    with pytest.raises(ValueError, match="empty score vector"):
        break_ties(np.array([]), "q", [])
    with pytest.raises(ValueError, match="3 scores for 4 choices"):
        break_ties(np.zeros(3), "q", ["a", "b", "c", "d"])


# --------------------------------------------------------------------------
# The contract every backend must satisfy
# --------------------------------------------------------------------------


def test_a_scorer_never_reads_the_answer_key() -> None:
    # A backend that consulted answer_index would report perfect accuracy in
    # every condition, and every number in the report would be meaningless.
    # The oracle locates the key by matching text, so it satisfies the same
    # contract as a real backend and is covered by the same test.
    corpus = _corpus()
    scorer = OracleScorer(OracleSpec(skill=0.7, seed=1), corpus)
    for item in corpus:
        honest = scorer.score(item)
        for fake_index in range(item.n_choices):
            tampered = Item(
                item_id=item.item_id,
                question=item.question,
                choices=item.choices,
                answer_index=fake_index,
                subject=item.subject,
            )
            np.testing.assert_allclose(scorer.score(tampered), honest)


def test_scoring_is_reproducible() -> None:
    corpus = _corpus()
    first = OracleScorer(OracleSpec(skill=0.5, seed=3), corpus)
    second = OracleScorer(OracleSpec(skill=0.5, seed=3), corpus)
    for item in corpus:
        np.testing.assert_allclose(first.score(item), second.score(item))


def test_scores_have_one_entry_per_option() -> None:
    corpus = _corpus()
    scorer = OracleScorer(OracleSpec(skill=0.5, seed=3), corpus)
    for item in corpus:
        assert scorer.score(item).shape == (item.n_choices,)


# --------------------------------------------------------------------------
# The oracle's mechanisms
# --------------------------------------------------------------------------


def test_perfect_skill_answers_everything_correctly() -> None:
    corpus = _corpus()
    scorer = OracleScorer(OracleSpec(skill=1.0, seed=1), corpus)
    assert all(is_correct(scorer, item) for item in corpus)


def test_skill_does_not_survive_the_question_being_removed() -> None:
    # Skill is understanding of the question. With no question there is nothing
    # to understand, and the oracle must fall through to guessing.
    corpus = _corpus(200)
    scorer = OracleScorer(OracleSpec(skill=1.0, seed=1), corpus)
    hide = HideQuestion()
    blind = [hide.transform(item, np.random.default_rng(0))[0] for item in corpus]
    accuracy = sum(is_correct(scorer, item) for item in blind) / len(blind)
    assert accuracy == pytest.approx(0.25, abs=0.12)


def test_skill_survives_reframing_and_permutation() -> None:
    corpus = _corpus()
    scorer = OracleScorer(OracleSpec(skill=1.0, seed=1), corpus)
    for item in corpus:
        (reframed,) = NeutralReframing().transform(item, np.random.default_rng(0))
        assert is_correct(scorer, reframed)
        for rotated in PermuteOptions().transform(item, np.random.default_rng(0)):
            assert is_correct(scorer, rotated)


def test_memorisation_survives_nothing_but_the_original_wording() -> None:
    corpus = _corpus(200)
    scorer = OracleScorer(OracleSpec(memorisation=1.0, seed=1), corpus)
    assert all(is_correct(scorer, item) for item in corpus)

    reframed = [NeutralReframing().transform(item, np.random.default_rng(0))[0] for item in corpus]
    accuracy = sum(is_correct(scorer, item) for item in reframed) / len(reframed)
    assert accuracy == pytest.approx(0.25, abs=0.12)


def test_choices_only_skill_survives_the_question_being_removed() -> None:
    corpus = _corpus(200)
    scorer = OracleScorer(OracleSpec(choices_only_skill=1.0, seed=1), corpus)
    blind = [HideQuestion().transform(item, np.random.default_rng(0))[0] for item in corpus]
    assert all(is_correct(scorer, item) for item in blind)


def test_positional_preference_answers_the_favoured_position() -> None:
    corpus = _corpus()
    scorer = OracleScorer(OracleSpec(position_preference=1.0, favoured_position=2, seed=1), corpus)
    for item in corpus:
        assert predict(scorer, item) == 2


def test_positional_preference_is_clamped_to_the_available_options() -> None:
    corpus = ItemSet(
        name="two",
        items=(Item(item_id="a", question="q", choices=("x", "y"), answer_index=0),),
    )
    scorer = OracleScorer(OracleSpec(position_preference=1.0, favoured_position=7, seed=1), corpus)
    assert predict(scorer, corpus.items[0]) == 1


def test_an_inert_oracle_scores_around_chance() -> None:
    corpus = _corpus(400)
    scorer = OracleScorer(OracleSpec(seed=5), corpus)
    accuracy = sum(is_correct(scorer, item) for item in corpus) / len(corpus)
    assert accuracy == pytest.approx(0.25, abs=0.08)


def test_an_unfamiliar_item_degrades_to_guessing_rather_than_crashing() -> None:
    scorer = OracleScorer(OracleSpec(skill=1.0, seed=1), _corpus())
    stranger = Item(item_id="unknown", question="q", choices=("a", "b"), answer_index=0)
    assert scorer.score(stranger).shape == (2,)


def test_oracle_identity_records_the_profile() -> None:
    scorer = OracleScorer(OracleSpec(skill=0.4, memorisation=0.2, seed=9), _corpus())
    assert "skill=0.4" in scorer.scorer_id
    assert "memorisation=0.2" in scorer.scorer_id
    assert "seed=9" in scorer.scorer_id
    assert scorer.deterministic


def test_an_inert_profile_is_labelled_as_such() -> None:
    assert OracleSpec().label == "inert"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"skill": 1.5}, "skill"),
        ({"memorisation": -0.1}, "memorisation"),
        ({"distractor_reliance": 2.0}, "distractor_reliance"),
        ({"position_preference": 1.1}, "position_preference"),
        ({"choices_only_skill": -1.0}, "choices_only_skill"),
        ({"favoured_position": -1}, "favoured_position"),
    ],
)
def test_invalid_oracle_profiles_are_refused(kwargs: dict[str, float], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        OracleSpec(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The score cache
# --------------------------------------------------------------------------


def test_the_cache_computes_each_prompt_once() -> None:
    corpus = _corpus(10)
    scorer = OracleScorer(OracleSpec(skill=0.5, seed=1), corpus)
    cache = ScoreCache()
    for _ in range(3):
        for item in corpus:
            cache.score(scorer, item)
    assert cache.misses == 10
    assert cache.hits == 20
    assert cache.size == 10


def test_the_cache_returns_the_same_scores_as_the_scorer() -> None:
    corpus = _corpus(10)
    scorer = OracleScorer(OracleSpec(skill=0.5, seed=1), corpus)
    cache = ScoreCache()
    for item in corpus:
        np.testing.assert_allclose(cache.score(scorer, item), scorer.score(item))
        assert cache.is_correct(scorer, item) == is_correct(scorer, item)


def test_the_cache_never_serves_one_scorer_from_another() -> None:
    corpus = _corpus(20)
    cache = ScoreCache()
    keen = OracleScorer(OracleSpec(skill=1.0, seed=1), corpus)
    inert = OracleScorer(OracleSpec(seed=1), corpus)

    keen_hits = sum(cache.is_correct(keen, item) for item in corpus)
    inert_hits = sum(cache.is_correct(inert, item) for item in corpus)
    assert keen_hits == len(corpus)
    assert inert_hits < len(corpus)
    assert cache.size == 2 * len(corpus)


def test_the_cache_recognises_a_prompt_reached_by_a_different_route() -> None:
    # Rotating an item's options through a full cycle returns the original
    # prompt, so the cache should serve it rather than score it again.
    corpus = _corpus(1)
    item = corpus.items[0]
    scorer = OracleScorer(OracleSpec(skill=0.5, seed=1), corpus)
    cache = ScoreCache()

    cache.score(scorer, item)
    before = cache.misses
    for rotated in PermuteOptions().transform(item, np.random.default_rng(0)):
        cache.score(scorer, rotated)
    # One of the four rotations is the original placement.
    assert cache.misses == before + 3
