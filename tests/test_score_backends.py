"""The two backends that talk to something outside this process.

The hosted-API backend is exercised through an injected fake client, so its
prompt construction, reply parsing and abstention handling are covered without a
network call or a bill.

The local backend needs real weights. Its tests are marked slow and skip
themselves when no model is available, so a checkout with no model still reports
honestly rather than passing vacuously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from evalassay.score.api import LABELS, SYSTEM_PROMPT, ApiScorer
from evalassay.score.base import break_ties, predict
from evalassay.types import Item

MODEL_FOR_LOCAL_TESTS = "Qwen/Qwen2.5-0.5B-Instruct"
"""Only used if it is already present; these tests never trigger a download."""


# --------------------------------------------------------------------------
# The hosted-API backend
# --------------------------------------------------------------------------


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _FakeMessages:
    """Stands in for the client's messages resource."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else ""
        return _Response(content=[_Block(text=reply)])


class _FakeClient:
    """Stands in for the hosted-model client."""

    def __init__(self, replies: list[str]) -> None:
        self.messages = _FakeMessages(replies)


def _item(n_choices: int = 4, question: str = "Which is a metal?") -> Item:
    return Item(
        item_id="x",
        question=question,
        choices=tuple(f"option {index}" for index in range(n_choices)),
        answer_index=1,
    )


def _scorer(replies: list[str]) -> tuple[ApiScorer, _FakeClient]:
    client = _FakeClient(replies)
    return ApiScorer("test-model", client=client), client


def test_a_letter_reply_selects_that_option() -> None:
    scorer, _ = _scorer(["C"])
    np.testing.assert_allclose(scorer.score(_item()), [0.0, 0.0, 1.0, 0.0])


def test_a_verbose_reply_still_yields_a_choice() -> None:
    scorer, _ = _scorer(["The answer is B."])
    assert predict(scorer, _item()) == 1


def test_a_lowercase_reply_is_accepted() -> None:
    scorer, _ = _scorer(["d"])
    assert predict(scorer, _item()) == 3


def test_a_label_beyond_the_option_count_is_not_selected() -> None:
    # Four options offered, the model names the fifth.
    scorer, _ = _scorer(["E"])
    scores = scorer.score(_item())
    assert scores.sum() == 0.0
    assert scorer.abstentions == 1


def test_an_unparseable_reply_is_recorded_as_an_abstention() -> None:
    # Retrying until the model says something parseable would quietly select
    # for the items it finds easy.
    scorer, client = _scorer(["I would rather not say."])
    scores = scorer.score(_item())
    assert scores.sum() == 0.0
    assert scorer.abstentions == 1
    assert len(client.messages.calls) == 1


def test_an_abstention_still_produces_a_usable_choice() -> None:
    scorer, _ = _scorer(["???"])
    chosen = predict(scorer, _item())
    assert 0 <= chosen < 4


def test_abstentions_are_counted_across_calls() -> None:
    scorer, _ = _scorer(["A", "nope", "also nope"])
    for _ in range(3):
        scorer.score(_item())
    assert scorer.abstentions == 2
    assert scorer.describe()["abstentions"] == 2


def test_the_prompt_labels_every_option() -> None:
    scorer, client = _scorer(["A"])
    scorer.score(_item())
    prompt = client.messages.calls[0]["messages"][0]["content"]
    for index in range(4):
        assert f"{LABELS[index]}. option {index}" in prompt
    assert "Which is a metal?" in prompt


def test_a_hidden_question_is_simply_absent_from_the_prompt() -> None:
    scorer, client = _scorer(["A"])
    scorer.score(_item(question=""))
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert prompt.startswith("A. option 0")


def test_the_system_prompt_constrains_the_reply() -> None:
    scorer, client = _scorer(["A"])
    scorer.score(_item())
    assert client.messages.calls[0]["system"] == SYSTEM_PROMPT
    assert client.messages.calls[0]["max_tokens"] == 4


def test_more_options_than_labels_is_refused() -> None:
    scorer, _ = _scorer(["A"])
    with pytest.raises(ValueError, match="cannot label"):
        scorer.score(_item(n_choices=len(LABELS) + 1))


def test_the_backend_declares_that_it_is_not_reproducible() -> None:
    # A hosted model is a moving target even at temperature zero, and the report
    # carries this through rather than implying a promise it cannot keep.
    scorer, _ = _scorer([])
    assert scorer.deterministic is False
    assert scorer.scorer_id == "api:test-model"
    assert scorer.describe()["deterministic"] is False


def test_the_api_backend_never_reads_the_answer_key() -> None:
    scorer, client = _scorer(["A", "A"])
    honest = _item()
    tampered = Item(
        item_id=honest.item_id,
        question=honest.question,
        choices=honest.choices,
        answer_index=3,
    )
    scorer.score(honest)
    scorer.score(tampered)
    assert client.messages.calls[0] == client.messages.calls[1]


# --------------------------------------------------------------------------
# The local backend
# --------------------------------------------------------------------------


def _local_scorer(style: str = "cloze") -> Any:
    """Build a local scorer, or skip if no model is available offline.

    Args:
        style: Scoring style to build.

    Returns:
        The scorer.
    """
    pytest.importorskip("torch", reason="local scoring needs the optional extras")
    pytest.importorskip("transformers", reason="local scoring needs the optional extras")

    from evalassay.score.local import LocalScorer

    try:
        return LocalScorer(MODEL_FOR_LOCAL_TESTS, style=style)
    except Exception as exc:
        pytest.skip(f"no local model available: {type(exc).__name__}")


@pytest.fixture(scope="module")
def local_scorer() -> Any:
    """A locally loaded model scoring in the cloze style."""
    return _local_scorer("cloze")


@pytest.fixture(scope="module")
def labelled_scorer() -> Any:
    """A locally loaded model scoring in the labelled style."""
    return _local_scorer("labelled")


@pytest.mark.slow
def test_local_scoring_returns_one_score_per_option(local_scorer: Any) -> None:
    scores = local_scorer.score(_item())
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


@pytest.mark.slow
def test_local_scoring_is_deterministic(local_scorer: Any) -> None:
    item = _item()
    np.testing.assert_allclose(local_scorer.score(item), local_scorer.score(item))
    assert local_scorer.deterministic is True


@pytest.mark.slow
def test_the_local_backend_never_reads_the_answer_key(local_scorer: Any) -> None:
    honest = _item()
    scores = local_scorer.score(honest)
    for fake_index in range(honest.n_choices):
        tampered = Item(
            item_id=honest.item_id,
            question=honest.question,
            choices=honest.choices,
            answer_index=fake_index,
        )
        np.testing.assert_allclose(local_scorer.score(tampered), scores)


@pytest.mark.slow
def test_permuting_options_permutes_the_scores(local_scorer: Any) -> None:
    # Scores follow the option text, so rotating the options must rotate the
    # scores. If it did not, the permutation intervention would be measuring
    # something about the scorer rather than about the model.
    item = _item()
    rotated = Item(
        item_id=item.item_id,
        question=item.question,
        choices=item.choices[1:] + item.choices[:1],
        answer_index=0,
    )
    original = local_scorer.score(item)
    moved = local_scorer.score(rotated)
    np.testing.assert_allclose(moved, np.roll(original, -1), rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_a_hidden_question_changes_the_scores(local_scorer: Any) -> None:
    shown = local_scorer.score(_item())
    hidden = local_scorer.score(_item(question=""))
    assert not np.allclose(shown, hidden)


@pytest.mark.slow
def test_length_normalisation_is_recorded_in_the_identity(local_scorer: Any) -> None:
    assert local_scorer.scorer_id.endswith(":norm")
    assert local_scorer.describe()["length_normalise"] is True


@pytest.mark.slow
def test_an_empty_option_does_not_break_scoring(local_scorer: Any) -> None:
    # A degenerate option must not fail a whole run.
    item = Item(item_id="e", question="q?", choices=("", "something"), answer_index=1)
    scores = local_scorer.score(item)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert 0 <= break_ties(scores, item.question, item.choices) < 2


# --------------------------------------------------------------------------
# The two local scoring styles
# --------------------------------------------------------------------------


def test_an_unknown_style_is_refused() -> None:
    pytest.importorskip("torch", reason="local scoring needs the optional extras")
    from evalassay.score.local import LocalScorer

    with pytest.raises(ValueError, match="style must be one of"):
        LocalScorer(MODEL_FOR_LOCAL_TESTS, style="freeform")


@pytest.mark.slow
def test_the_style_is_recorded_in_the_scorer_identity(labelled_scorer: Any) -> None:
    # The manifest has to distinguish the two, because they can give different
    # answers on the same items.
    assert ":labelled:" in labelled_scorer.scorer_id
    assert labelled_scorer.describe()["style"] == "labelled"


@pytest.mark.slow
def test_the_labelled_prompt_shows_the_option_list(labelled_scorer: Any) -> None:
    prompt = labelled_scorer._prompt_for(_item())
    for index in range(4):
        assert f"{'ABCD'[index]}. option {index}" in prompt
    assert prompt.rstrip().endswith("Answer:")


@pytest.mark.slow
def test_the_cloze_prompt_hides_the_option_list(local_scorer: Any) -> None:
    # This is why option permutation cannot bite under cloze scoring: nothing
    # about an option's position reaches the model.
    prompt = local_scorer._prompt_for(_item())
    for index in range(4):
        assert f"option {index}" not in prompt


@pytest.mark.slow
def test_cloze_scoring_is_invariant_to_option_order(local_scorer: Any) -> None:
    # The structural property that makes the audit call permutation inert here.
    item = _item()
    base = local_scorer.score(item)
    rotated = Item(
        item_id=item.item_id,
        question=item.question,
        choices=item.choices[1:] + item.choices[:1],
        answer_index=0,
    )
    np.testing.assert_allclose(local_scorer.score(rotated), np.roll(base, -1), rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_labelled_scoring_is_not_invariant_to_option_order(labelled_scorer: Any) -> None:
    # Under labelled scoring the model sees the list, so moving an option can
    # change its score. Without this the permutation intervention would have
    # nothing to measure against any backend.
    item = _item()
    base = labelled_scorer.score(item)
    rotated = Item(
        item_id=item.item_id,
        question=item.question,
        choices=item.choices[1:] + item.choices[:1],
        answer_index=0,
    )
    moved = labelled_scorer.score(rotated)
    assert not np.allclose(moved, np.roll(base, -1), rtol=1e-3, atol=1e-4)
