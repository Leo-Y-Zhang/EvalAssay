"""The synthetic generator must plant defects at the magnitudes it promises.

These tests are the foundation of the calibration harness: if the generator did
not really plant what its spec claims, every later statement about the
instrument's error characteristic would be measuring the wrong thing.
"""

from __future__ import annotations

import pytest

from evalassay.corpus.synthetic import (
    MARKER_WORD,
    PAD_WORDS,
    VOCABULARY,
    WORD_LENGTH,
    CorpusSpec,
    generate,
)
from evalassay.hashing import item_digest

LARGE = 4000
"""Enough items that a planted rate is estimated to within a few points."""

TOLERANCE = 0.03
"""Roughly four binomial standard errors at the sample size used here."""


def test_a_clean_corpus_has_uniform_answer_positions() -> None:
    corpus = generate(CorpusSpec(n_items=LARGE, n_choices=4, seed=1))
    shares = [0.0] * 4
    for item in corpus:
        shares[item.answer_index] += 1.0 / len(corpus)
    for share in shares:
        assert share == pytest.approx(0.25, abs=TOLERANCE)


def test_planted_position_bias_appears_at_the_stated_rate() -> None:
    spec = CorpusSpec(n_items=LARGE, n_choices=4, seed=2, position_bias=0.5, biased_position=2)
    corpus = generate(spec)
    at_biased = sum(1 for item in corpus if item.answer_index == 2) / len(corpus)
    assert at_biased == pytest.approx(spec.expected_position_share, abs=TOLERANCE)
    assert spec.expected_position_share == pytest.approx(0.5 + 0.5 * 0.25)


def test_planted_longest_answer_defect_appears_at_the_stated_rate() -> None:
    spec = CorpusSpec(n_items=LARGE, n_choices=4, seed=3, longest_answer_rate=0.4)
    corpus = generate(spec)
    wins = 0.0
    for item in corpus:
        lengths = [len(choice) for choice in item.choices]
        longest = max(lengths)
        winners = [i for i, length in enumerate(lengths) if length == longest]
        # Credit a tie only in proportion, matching how the heuristic would fare.
        if item.answer_index in winners:
            wins += 1 / len(winners)
    assert wins / len(corpus) == pytest.approx(spec.expected_longest_answer_accuracy, abs=TOLERANCE)


def test_planted_lexical_marker_appears_only_in_correct_options() -> None:
    spec = CorpusSpec(n_items=LARGE, n_choices=4, seed=4, choices_only_rate=0.3)
    corpus = generate(spec)
    marked = 0
    for item in corpus:
        for index, choice in enumerate(item.choices):
            if MARKER_WORD in choice:
                assert index == item.answer_index
                marked += 1
    assert marked / len(corpus) == pytest.approx(0.3, abs=TOLERANCE)


def test_the_marker_does_not_change_option_length() -> None:
    # Substitution, not addition: otherwise lexical leakage would be
    # indistinguishable from the longest-answer defect.
    spec = CorpusSpec(n_items=500, n_choices=4, seed=5, choices_only_rate=1.0)
    for item in generate(spec):
        word_counts = {len(choice.split()) for choice in item.choices}
        assert len(word_counts) == 1


def test_padding_makes_the_correct_option_strictly_longest() -> None:
    spec = CorpusSpec(n_items=500, n_choices=4, seed=6, longest_answer_rate=1.0)
    for item in generate(spec):
        lengths = [len(choice) for choice in item.choices]
        assert lengths[item.answer_index] == max(lengths)
        assert lengths.count(max(lengths)) == 1


def test_padding_adds_no_distinctive_token() -> None:
    # A repeated filler word would be trivially learnable by the choices-only
    # probe, so the longest-answer fixture would plant lexical leakage too and
    # the two defects could not be calibrated apart.
    spec = CorpusSpec(n_items=400, n_choices=4, seed=21, longest_answer_rate=1.0)
    corpus = generate(spec)
    in_correct: dict[str, int] = {}
    in_wrong: dict[str, int] = {}
    for item in corpus:
        for index, choice in enumerate(item.choices):
            target = in_correct if index == item.answer_index else in_wrong
            for word in set(choice.split()):
                target[word] = target.get(word, 0) + 1
    # No token should appear in correct options far more often than in wrong
    # ones, other than by ordinary sampling noise.
    for word, correct_count in in_correct.items():
        if correct_count >= 20:
            assert in_wrong.get(word, 0) > 0, f"{word!r} appears only in correct options"


def test_padding_adds_the_stated_number_of_words() -> None:
    padded = generate(CorpusSpec(n_items=200, n_choices=4, seed=22, longest_answer_rate=1.0))
    for item in padded:
        counts = [len(choice.split()) for choice in item.choices]
        others = [c for i, c in enumerate(counts) if i != item.answer_index]
        assert counts[item.answer_index] == others[0] + PAD_WORDS


def test_every_vocabulary_word_has_the_same_length() -> None:
    # Load-bearing: it is what makes the lexical marker length-neutral.
    assert all(len(word) == WORD_LENGTH for word in VOCABULARY)
    assert len(MARKER_WORD) == WORD_LENGTH
    assert MARKER_WORD not in VOCABULARY


def test_a_clean_corpus_gives_every_option_the_same_length() -> None:
    # Consequence of the equal-length vocabulary: the longest-option heuristic
    # is reduced to a full tie, so its expected accuracy is exactly chance.
    for item in generate(CorpusSpec(n_items=200, n_choices=4, seed=23)):
        assert len({len(choice) for choice in item.choices}) == 1


def test_option_text_carries_no_signal_about_position() -> None:
    # If it did, planted position bias would leak into the choices-only probe
    # and the two effects could not be separated.
    spec = CorpusSpec(n_items=2000, n_choices=4, seed=7, position_bias=0.8, biased_position=1)
    corpus = generate(spec)
    vocabulary_at_position: list[set[str]] = [set() for _ in range(4)]
    for item in corpus:
        for index, choice in enumerate(item.choices):
            vocabulary_at_position[index].update(choice.split())
    # Every position draws from the same closed vocabulary.
    for words in vocabulary_at_position:
        assert words <= set(VOCABULARY)
        assert len(words) > len(VOCABULARY) // 2


def test_duplicates_are_planted_at_the_stated_rate() -> None:
    spec = CorpusSpec(n_items=2000, n_choices=4, seed=8, duplicate_rate=0.1)
    corpus = generate(spec)
    extra = len(corpus) - spec.n_items
    assert extra / spec.n_items == pytest.approx(0.1, abs=TOLERANCE)


def test_planted_duplicates_are_content_identical_but_separately_identified() -> None:
    spec = CorpusSpec(n_items=200, n_choices=4, seed=9, duplicate_rate=1.0)
    corpus = generate(spec)
    originals = {item.item_id: item for item in corpus if not item.item_id.startswith("syn-dup-")}
    duplicates = [item for item in corpus if item.item_id.startswith("syn-dup-")]
    assert duplicates
    for duplicate in duplicates:
        source = originals["syn-" + duplicate.item_id.removeprefix("syn-dup-")]
        assert item_digest(duplicate) == item_digest(source)
        assert duplicate.item_id != source.item_id


def test_a_clean_corpus_plants_no_duplicates() -> None:
    corpus = generate(CorpusSpec(n_items=500, n_choices=4, seed=10))
    digests = [item_digest(item) for item in corpus]
    assert len(set(digests)) == len(digests)


def test_generation_is_deterministic_in_the_seed() -> None:
    spec = CorpusSpec(n_items=200, n_choices=4, seed=11, position_bias=0.3)
    assert generate(spec) == generate(spec)


def test_generation_changes_with_the_seed() -> None:
    a = generate(CorpusSpec(n_items=200, n_choices=4, seed=12))
    b = generate(CorpusSpec(n_items=200, n_choices=4, seed=13))
    assert a != b


def test_corpus_name_records_the_seed() -> None:
    assert "seed=14" in generate(CorpusSpec(n_items=10, seed=14)).name


def test_subjects_are_assigned_so_stratification_can_be_exercised() -> None:
    corpus = generate(CorpusSpec(n_items=400, seed=15))
    assert len({item.subject for item in corpus}) == 4


def test_expected_values_match_their_closed_forms() -> None:
    spec = CorpusSpec(
        n_choices=5, position_bias=0.2, longest_answer_rate=0.4, choices_only_rate=0.6
    )
    assert spec.expected_position_share == pytest.approx(0.2 + 0.8 / 5)
    assert spec.expected_longest_answer_accuracy == pytest.approx(0.4 + 0.6 / 5)
    assert spec.expected_choices_only_accuracy == pytest.approx(0.6 + 0.4 / 5)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"n_items": 0}, "n_items must be positive"),
        ({"n_choices": 1}, "n_choices must be at least 2"),
        ({"biased_position": 9}, "biased_position 9 outside"),
        ({"position_bias": 1.5}, "position_bias 1.5 outside"),
        ({"longest_answer_rate": -0.1}, "longest_answer_rate -0.1 outside"),
        ({"choices_only_rate": 2.0}, "choices_only_rate 2.0 outside"),
        ({"duplicate_rate": -1.0}, "duplicate_rate -1.0 outside"),
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict[str, object], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        CorpusSpec(**kwargs)  # type: ignore[arg-type]
