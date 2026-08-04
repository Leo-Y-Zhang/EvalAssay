"""Content addressing: what must change a hash, and what must not.

Non-ASCII fixtures are written as Python escape sequences, never as pasted
characters. An editor or transfer step that silently normalises a source file
would otherwise turn the Unicode tests into tautologies that pass while testing
nothing, and the escapes are plain ASCII bytes on disk so nothing can rewrite
them.
"""

from __future__ import annotations

import pytest

from evalassay.hashing import (
    HASH_PREFIX,
    config_hash,
    corpus_hash,
    is_digest,
    item_digest,
    normalise_text,
    prompt_digest,
    stable_json,
)
from evalassay.types import Item, ItemSet

E_ACUTE_COMPOSED = "éclair"
"""Single code point: LATIN SMALL LETTER E WITH ACUTE."""

E_ACUTE_DECOMPOSED = "éclair"
"""Two code points: bare 'e' followed by COMBINING ACUTE ACCENT."""

NON_BREAKING = "non breaking"
"""Words joined by U+00A0, which renders like a space but is a distinct byte."""


def _item(item_id: str = "a", question: str = "Which one?", answer_index: int = 0) -> Item:
    return Item(
        item_id=item_id,
        question=question,
        choices=("alpha", "beta", "gamma", "delta"),
        answer_index=answer_index,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  spaced  out  ", "spaced out"),
        ("line\nbreak", "line break"),
        ("tab\tsep", "tab sep"),
        ("carriage\r\nreturn", "carriage return"),
        (NON_BREAKING, "non breaking"),
        ("already clean", "already clean"),
    ],
)
def test_normalisation_collapses_incidental_whitespace(raw: str, expected: str) -> None:
    assert normalise_text(raw) == expected


def test_the_non_breaking_fixture_is_not_a_plain_space() -> None:
    # Guards the parametrised case above: if the fixture were rewritten to an
    # ordinary space, that case would pass without exercising anything.
    assert NON_BREAKING != "non breaking"
    assert " " in NON_BREAKING
    assert len(NON_BREAKING) == len("non breaking")


def test_normalisation_unifies_unicode_composition() -> None:
    assert E_ACUTE_DECOMPOSED != E_ACUTE_COMPOSED
    assert len(E_ACUTE_DECOMPOSED) == len(E_ACUTE_COMPOSED) + 1
    assert normalise_text(E_ACUTE_DECOMPOSED) == normalise_text(E_ACUTE_COMPOSED)


def test_composition_difference_survives_into_the_digest() -> None:
    # The two spellings must hash alike, which is the point of normalising.
    left = Item(item_id="x", question=E_ACUTE_DECOMPOSED, choices=("a", "b"), answer_index=0)
    right = Item(item_id="x", question=E_ACUTE_COMPOSED, choices=("a", "b"), answer_index=0)
    assert item_digest(left) == item_digest(right)


def test_normalisation_preserves_case_and_punctuation() -> None:
    # Both change what a model sees, so neither may be normalised away.
    assert normalise_text("The Cat?") == "The Cat?"
    assert normalise_text("the cat") != normalise_text("The Cat")
    assert normalise_text("cat") != normalise_text("cat?")


def test_stable_json_is_insensitive_to_key_order() -> None:
    assert stable_json({"b": 1, "a": 2}) == stable_json({"a": 2, "b": 1})


def test_stable_json_escapes_non_ascii() -> None:
    assert "\\u00e9" in stable_json({"k": E_ACUTE_COMPOSED})


def test_stable_json_has_no_incidental_whitespace() -> None:
    assert stable_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_digests_are_recognisable_and_well_formed() -> None:
    digest = item_digest(_item())
    assert digest.startswith(HASH_PREFIX)
    assert is_digest(digest)


@pytest.mark.parametrize(
    "value",
    ["", "sha256:", "deadbeef", "sha256:" + "z" * 64, "sha256:" + "0" * 63, "md5:" + "0" * 64],
)
def test_malformed_digests_are_rejected(value: str) -> None:
    assert not is_digest(value)


def test_item_digest_ignores_the_identifier_and_subject() -> None:
    # Two publishers labelling the same question differently must collide, or
    # near-duplicate detection cannot work across sources.
    left = Item(item_id="mmlu-1", question="q", choices=("a", "b"), answer_index=0, subject="law")
    right = Item(item_id="arc-9", question="q", choices=("a", "b"), answer_index=0, subject="bio")
    assert item_digest(left) == item_digest(right)


def test_item_digest_ignores_incidental_whitespace() -> None:
    assert item_digest(_item(question="Which  one?")) == item_digest(_item(question="Which one?"))


def test_item_digest_changes_with_the_answer_key() -> None:
    assert item_digest(_item(answer_index=0)) != item_digest(_item(answer_index=1))


def test_item_digest_changes_with_choice_order() -> None:
    base = Item(item_id="x", question="q", choices=("a", "b", "c"), answer_index=0)
    reordered = Item(item_id="x", question="q", choices=("b", "a", "c"), answer_index=1)
    # Same options, same correct text, different presentation: a different item
    # as far as a model is concerned.
    assert item_digest(base) != item_digest(reordered)


def test_corpus_hash_depends_on_name_order_and_content() -> None:
    first, second = _item("a"), _item("b", question="Another?")
    base = corpus_hash(ItemSet(name="demo", items=(first, second)))

    assert corpus_hash(ItemSet(name="demo", items=(first, second))) == base
    assert corpus_hash(ItemSet(name="other", items=(first, second))) != base
    assert corpus_hash(ItemSet(name="demo", items=(second, first))) != base
    assert corpus_hash(ItemSet(name="demo", items=(first,))) != base


def test_corpus_hash_notices_a_renamed_item() -> None:
    # Identity is excluded from an item digest but included in the corpus hash,
    # because a run consumed a specific labelled list.
    first = _item("a")
    relabelled = _item("a-renamed")
    assert item_digest(first) == item_digest(relabelled)
    assert corpus_hash(ItemSet(name="d", items=(first,))) != corpus_hash(
        ItemSet(name="d", items=(relabelled,))
    )


def test_config_hash_is_order_insensitive_but_value_sensitive() -> None:
    assert config_hash({"alpha": 0.01, "seed": 7}) == config_hash({"seed": 7, "alpha": 0.01})
    assert config_hash({"alpha": 0.01, "seed": 7}) != config_hash({"alpha": 0.05, "seed": 7})


def test_prompt_digest_separates_scorers() -> None:
    # A cache that served one model's scores to another would silently corrupt
    # every downstream number.
    left = prompt_digest("local:tiny", "q", ["a", "b"])
    right = prompt_digest("local:large", "q", ["a", "b"])
    assert left != right


def test_prompt_digest_distinguishes_a_hidden_question() -> None:
    shown = prompt_digest("local:tiny", "Which one?", ["a", "b"])
    hidden = prompt_digest("local:tiny", "", ["a", "b"])
    assert shown != hidden


def test_prompt_digest_distinguishes_option_order() -> None:
    left = prompt_digest("local:tiny", "q", ["a", "b"])
    right = prompt_digest("local:tiny", "q", ["b", "a"])
    assert left != right


def test_prompt_digest_is_stable_across_equivalent_whitespace() -> None:
    left = prompt_digest("local:tiny", "Which  one?", ["a ", " b"])
    right = prompt_digest("local:tiny", "Which one?", ["a", "b"])
    assert left == right
