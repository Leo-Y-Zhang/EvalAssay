"""The canonical on-disk corpus format, and deterministic subsampling.

Every loader converts into this one format, so the rest of the pipeline never
learns anything about where a benchmark came from. The format is JSON Lines
because it streams, diffs readably, and can be inspected without this package
installed - a corpus a reader cannot open is a corpus a reader cannot check.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from evalassay.types import Item, ItemSet

REQUIRED_FIELDS: Final = ("id", "question", "choices", "answer_index")
"""Fields every record must carry. ``subject`` is optional."""


def item_to_record(item: Item) -> dict[str, Any]:
    """Convert an item to its canonical record.

    Args:
        item: The item.

    Returns:
        A JSON-serialisable mapping.
    """
    record: dict[str, Any] = {
        "id": item.item_id,
        "question": item.question,
        "choices": list(item.choices),
        "answer_index": item.answer_index,
    }
    if item.subject:
        record["subject"] = item.subject
    return record


def record_to_item(record: dict[str, Any], *, where: str = "record") -> Item:
    """Convert a canonical record to an item, validating as it goes.

    Args:
        record: A parsed JSON object.
        where: Human-readable location, used in error messages.

    Returns:
        The item.

    Raises:
        ValueError: If a field is missing or has the wrong type. Errors name the
            offending field and location, because a corpus that fails to load
            with an opaque message is a corpus nobody will fix.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{where}: missing field(s) {', '.join(missing)}")

    item_id = record["id"]
    if not isinstance(item_id, str):
        raise ValueError(f"{where}: 'id' must be a string, got {type(item_id).__name__}")

    question = record["question"]
    if not isinstance(question, str):
        raise ValueError(f"{where}: 'question' must be a string, got {type(question).__name__}")

    choices = record["choices"]
    if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
        raise ValueError(f"{where}: 'choices' must be a list of strings")

    answer_index = record["answer_index"]
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise ValueError(
            f"{where}: 'answer_index' must be an integer, got {type(answer_index).__name__}"
        )

    subject = record.get("subject", "")
    if not isinstance(subject, str):
        raise ValueError(f"{where}: 'subject' must be a string, got {type(subject).__name__}")

    try:
        return Item(
            item_id=item_id,
            question=question,
            choices=tuple(choices),
            answer_index=answer_index,
            subject=subject,
        )
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def write_jsonl(item_set: ItemSet, path: Path) -> None:
    """Write a corpus to canonical JSON Lines.

    Args:
        item_set: The corpus.
        path: Destination file. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item_to_record(item), sort_keys=True, ensure_ascii=False)
        for item in item_set.items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path, name: str | None = None) -> ItemSet:
    """Read a corpus from canonical JSON Lines.

    Blank lines are skipped so a hand-edited file still loads.

    Args:
        path: Source file.
        name: Corpus name. Defaults to the file stem.

    Returns:
        The corpus.

    Raises:
        ValueError: If any line is not a JSON object or fails validation. The
            message carries the line number.
    """
    items: list[Item] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            where = f"{path.name}:{line_number}"
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{where}: expected a JSON object, got {type(parsed).__name__}")
            items.append(record_to_item(parsed, where=where))

    return ItemSet(name=name if name is not None else path.stem, items=tuple(items))


def from_items(name: str, items: Iterable[Item]) -> ItemSet:
    """Build a corpus from an iterable of items.

    Args:
        name: Corpus name.
        items: The items, in the order they should be audited.

    Returns:
        The corpus.
    """
    return ItemSet(name=name, items=tuple(items))


def _selected_indices(
    item_set: ItemSet, size: int, rng: np.random.Generator, *, stratify_by_subject: bool
) -> list[int]:
    """Choose which item indices to keep.

    Args:
        item_set: The corpus.
        size: How many items to keep.
        rng: Seeded generator.
        stratify_by_subject: Whether to allocate the quota across subjects.

    Returns:
        Sorted indices into the original corpus.
    """
    total = len(item_set)
    if not stratify_by_subject:
        return sorted(int(i) for i in rng.choice(total, size=size, replace=False))

    by_subject: dict[str, list[int]] = {}
    for index, item in enumerate(item_set.items):
        by_subject.setdefault(item.subject, []).append(index)

    # Largest-remainder allocation, so the quota is filled exactly and the
    # rounding is not silently biased toward whichever subject sorts first.
    subjects = sorted(by_subject)
    exact = {s: size * len(by_subject[s]) / total for s in subjects}
    quota = {s: min(len(by_subject[s]), int(exact[s])) for s in subjects}

    remaining = size - sum(quota.values())
    order = sorted(subjects, key=lambda s: (-(exact[s] - int(exact[s])), s))
    cursor = 0
    while remaining > 0 and cursor < len(order) * 2:
        subject = order[cursor % len(order)]
        if quota[subject] < len(by_subject[subject]):
            quota[subject] += 1
            remaining -= 1
        cursor += 1

    chosen: list[int] = []
    for subject in subjects:
        pool = by_subject[subject]
        take = quota[subject]
        if take:
            chosen.extend(int(i) for i in rng.choice(pool, size=take, replace=False))
    return sorted(chosen)


def subsample(
    item_set: ItemSet,
    size: int,
    seed: int,
    *,
    stratify_by_subject: bool = False,
) -> ItemSet:
    """Take a deterministic subsample, preserving the original item order.

    Order is preserved rather than shuffled so that two subsamples of the same
    corpus at different sizes remain visually comparable, and so the corpus hash
    depends only on which items were chosen.

    Args:
        item_set: The corpus to sample from.
        size: How many items to keep. Must not exceed the corpus size.
        seed: Seed for the selection.
        stratify_by_subject: Allocate the quota proportionally across subjects.
            Worth using on benchmarks that bundle many topics, where a plain
            random sample can under-represent small subjects badly enough to
            change what the audit is measuring.

    Returns:
        The subsampled corpus, named with a suffix recording the size and seed
        so a report can never confuse it with the full set.

    Raises:
        ValueError: If ``size`` is not between 1 and the corpus size.
    """
    total = len(item_set)
    if not 1 <= size <= total:
        raise ValueError(f"size must be in 1..{total}, got {size}")
    if size == total:
        return item_set

    rng = np.random.default_rng(seed)
    indices = _selected_indices(item_set, size, rng, stratify_by_subject=stratify_by_subject)
    kept = tuple(item_set.items[i] for i in indices)
    return ItemSet(name=f"{item_set.name}[n={size},seed={seed}]", items=kept)


def subjects_of(item_set: ItemSet) -> Sequence[str]:
    """List the distinct subjects present, in sorted order.

    Args:
        item_set: The corpus.

    Returns:
        Sorted distinct subject labels, including the empty label if present.
    """
    return sorted({item.subject for item in item_set.items})
