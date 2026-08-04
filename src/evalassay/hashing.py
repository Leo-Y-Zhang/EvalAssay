"""Content-addressed hashing for corpora, configuration and prompts.

Two runs of EvalAssay are comparable only if they agree about what they were
run on. Every identity in the system is therefore a hash over normalised
content rather than a filename, a download URL, or a version string that a
publisher can change without telling anyone.

Normalisation here is deliberately conservative. It removes differences that no
loader should be allowed to introduce (encoding form, line endings, incidental
whitespace) and preserves everything else, including case and punctuation,
because those change what a model sees.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final

from evalassay.types import Item, ItemSet

HASH_PREFIX: Final = "sha256:"
"""Prefix carried by every digest, so a bare hex string is never mistaken for one."""

_DIGEST_CHARS: Final = 64


def normalise_text(text: str) -> str:
    """Reduce text to the form used for hashing and comparison.

    Applies Unicode NFC composition, converts every run of whitespace
    (including newlines and non-breaking spaces) to a single space, and strips
    the ends.

    Args:
        text: Raw text from a corpus loader.

    Returns:
        The normalised form.
    """
    composed = unicodedata.normalize("NFC", text)
    return " ".join(composed.split())


def stable_json(value: Any) -> str:
    """Serialise a value to JSON that depends only on its content.

    Keys are sorted, separators are fixed, and non-ASCII characters are
    escaped, so the output does not depend on dictionary insertion order or on
    the platform's default encoding.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        A canonical JSON string.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: str) -> str:
    """Hash a canonical payload string.

    Args:
        payload: Already-canonical text.

    Returns:
        A prefixed hex digest.
    """
    return HASH_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_digest(value: str) -> bool:
    """Whether a string has the shape of a digest produced here.

    Args:
        value: Candidate string.

    Returns:
        ``True`` if the value is correctly prefixed and hex of the right length.
    """
    if not value.startswith(HASH_PREFIX):
        return False
    body = value[len(HASH_PREFIX) :]
    return len(body) == _DIGEST_CHARS and all(c in "0123456789abcdef" for c in body)


def item_digest(item: Item) -> str:
    """Hash one item's scoreable content.

    The identifier and subject are excluded on purpose: two items with the same
    question, the same options in the same order, and the same key are the same
    item, whatever a publisher labelled them. Near-duplicate detection depends
    on this.

    Args:
        item: The item to hash.

    Returns:
        A prefixed hex digest.
    """
    payload = stable_json(
        {
            "question": normalise_text(item.question),
            "choices": [normalise_text(choice) for choice in item.choices],
            "answer_index": item.answer_index,
        }
    )
    return _digest(payload)


def corpus_hash(item_set: ItemSet) -> str:
    """Hash a whole corpus, including its name and item order.

    Order is included because it is part of what a run consumed: a subsampled
    corpus in a different order is a different corpus, and silently treating
    the two as equal would let a reader believe a result had been reproduced
    when it had not.

    Args:
        item_set: The corpus.

    Returns:
        A prefixed hex digest.
    """
    payload = stable_json(
        {
            "name": item_set.name,
            "items": [{"id": item.item_id, "digest": item_digest(item)} for item in item_set.items],
        }
    )
    return _digest(payload)


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash an audit configuration.

    Args:
        config: Configuration mapping. Values must be JSON-serialisable.

    Returns:
        A prefixed hex digest.
    """
    return _digest(stable_json(dict(config)))


def prompt_digest(scorer_id: str, question: str, choices: Sequence[str]) -> str:
    """Hash a scoring request, for use as a cache key.

    The scorer identity is folded in so that cached scores from one model can
    never be served to another.

    Args:
        scorer_id: Identifier of the scoring backend and model.
        question: The question stem as presented, possibly empty.
        choices: The options as presented, in presentation order.

    Returns:
        A prefixed hex digest.
    """
    payload = stable_json(
        {
            "scorer": scorer_id,
            "question": normalise_text(question),
            "choices": [normalise_text(choice) for choice in choices],
        }
    )
    return _digest(payload)
