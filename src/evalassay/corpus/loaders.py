"""Loaders for the public multiple-choice benchmark formats.

No benchmark data ships with this package. These read files you obtained
yourself, in the layout their publishers distribute, and convert them into the
canonical form the rest of the pipeline uses.

Each loader validates hard and fails loudly. A loader that silently drops
malformed rows would change the denominator of every accuracy in the report
without saying so, which is exactly the class of quiet error this project exists
to catch.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from evalassay.types import Item, ItemSet

LETTERS: Final = "ABCDEFGH"
"""Answer labels, in the order publishers use them."""

_MMLU_MIN_COLUMNS: Final = 4


def _answer_index(label: str, n_choices: int, where: str) -> int:
    """Resolve a publisher's answer label to a zero-based index.

    Handles both letter labels ("A") and one-based numeric labels ("1"), which
    different releases of the same benchmark have used.

    Args:
        label: The published label.
        n_choices: How many options the item has.
        where: Location, for error messages.

    Returns:
        The zero-based index.

    Raises:
        ValueError: If the label is unrecognised or out of range.
    """
    cleaned = label.strip()
    if not cleaned:
        raise ValueError(f"{where}: empty answer label")

    if cleaned.upper() in LETTERS:
        index = LETTERS.index(cleaned.upper())
    elif cleaned.isdigit():
        index = int(cleaned) - 1
    else:
        raise ValueError(f"{where}: unrecognised answer label {label!r}")

    if not 0 <= index < n_choices:
        raise ValueError(f"{where}: answer label {label!r} outside 0..{n_choices - 1}")
    return index


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield line numbers and parsed objects from a JSON Lines file.

    Args:
        path: Source file.

    Yields:
        ``(line_number, object)`` pairs, skipping blank lines.

    Raises:
        ValueError: If a line is not a JSON object.
    """
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"{path.name}:{line_number}: expected an object, got {type(parsed).__name__}"
                )
            yield line_number, parsed


def load_mmlu_csv(path: Path, subject: str | None = None) -> ItemSet:
    """Load one MMLU subject file.

    MMLU ships headerless CSV: question, then one column per option, then the
    answer letter in the final column.

    Args:
        path: The subject CSV.
        subject: Subject label. Defaults to the file stem with underscores
            turned into spaces, matching how the benchmark names its subjects.

    Returns:
        The corpus.

    Raises:
        ValueError: If a row is too short or its answer label is unusable.
    """
    label = subject if subject is not None else path.stem.replace("_", " ")
    items: list[Item] = []

    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            where = f"{path.name}:{row_number}"
            if len(row) < _MMLU_MIN_COLUMNS:
                raise ValueError(f"{where}: expected question, options and answer, got {len(row)}")

            question, *rest = row
            answer_label = rest[-1]
            choices = tuple(rest[:-1])
            items.append(
                Item(
                    item_id=f"{path.stem}-{row_number:05d}",
                    question=question,
                    choices=choices,
                    answer_index=_answer_index(answer_label, len(choices), where),
                    subject=label,
                )
            )

    return ItemSet(name=f"mmlu:{label}", items=tuple(items))


def load_mmlu_directory(directory: Path) -> ItemSet:
    """Load every MMLU subject CSV in a directory into one corpus.

    Subjects are loaded in sorted filename order so the corpus hash does not
    depend on the filesystem's directory ordering.

    Args:
        directory: Directory of subject CSVs.

    Returns:
        The combined corpus.

    Raises:
        ValueError: If the directory contains no CSV files.
    """
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise ValueError(f"no CSV files in {directory}")
    items: list[Item] = []
    for path in paths:
        items.extend(load_mmlu_csv(path).items)
    return ItemSet(name="mmlu", items=tuple(items))


def load_arc_jsonl(path: Path) -> ItemSet:
    """Load an ARC (AI2 Reasoning Challenge) JSON Lines file.

    Args:
        path: Source file.

    Returns:
        The corpus.

    Raises:
        ValueError: If a record lacks the expected structure.
    """
    items: list[Item] = []
    for line_number, record in _iter_jsonl(path):
        where = f"{path.name}:{line_number}"

        question_block = record.get("question")
        if not isinstance(question_block, dict):
            raise ValueError(f"{where}: 'question' must be an object")
        stem = question_block.get("stem")
        if not isinstance(stem, str):
            raise ValueError(f"{where}: 'question.stem' must be a string")

        raw_choices = question_block.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"{where}: 'question.choices' must be a non-empty list")

        texts: list[str] = []
        labels: list[str] = []
        for choice in raw_choices:
            if not isinstance(choice, dict) or "text" not in choice or "label" not in choice:
                raise ValueError(f"{where}: each choice needs 'text' and 'label'")
            texts.append(str(choice["text"]))
            labels.append(str(choice["label"]))

        answer_key = record.get("answerKey")
        if not isinstance(answer_key, str):
            raise ValueError(f"{where}: 'answerKey' must be a string")

        # Prefer the item's own labels, since ARC mixes letter and numeric keys
        # within a single release.
        if answer_key in labels:
            answer_index = labels.index(answer_key)
        else:
            answer_index = _answer_index(answer_key, len(texts), where)

        items.append(
            Item(
                item_id=str(record.get("id", f"{path.stem}-{line_number:05d}")),
                question=stem,
                choices=tuple(texts),
                answer_index=answer_index,
            )
        )

    return ItemSet(name=f"arc:{path.stem}", items=tuple(items))


def load_hellaswag_jsonl(path: Path) -> ItemSet:
    """Load a HellaSwag JSON Lines file.

    HellaSwag frames the task as sentence completion: a context and several
    endings, with a zero-based label. It is included because completion-style
    items behave differently under the hide-question intervention than
    question-style items do, and an audit that only ever saw one framing would
    generalise badly.

    Args:
        path: Source file.

    Returns:
        The corpus.

    Raises:
        ValueError: If a record lacks the expected structure, or is unlabelled
            (the public test split ships without labels and cannot be scored).
    """
    items: list[Item] = []
    for line_number, record in _iter_jsonl(path):
        where = f"{path.name}:{line_number}"

        context = record.get("ctx", record.get("ctx_a"))
        if not isinstance(context, str):
            raise ValueError(f"{where}: 'ctx' must be a string")

        endings = record.get("endings")
        if not isinstance(endings, list) or not endings:
            raise ValueError(f"{where}: 'endings' must be a non-empty list")

        raw_label = record.get("label")
        if raw_label is None or raw_label == "":
            raise ValueError(f"{where}: record has no label; the unlabelled split cannot be scored")
        try:
            answer_index = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: 'label' must be an integer, got {raw_label!r}") from exc

        items.append(
            Item(
                item_id=str(record.get("ind", f"{path.stem}-{line_number:05d}")),
                question=context,
                choices=tuple(str(ending) for ending in endings),
                answer_index=answer_index,
                subject=str(record.get("activity_label", "")),
            )
        )

    return ItemSet(name=f"hellaswag:{path.stem}", items=tuple(items))


def load_truthfulqa_mc_jsonl(path: Path) -> ItemSet:
    """Load TruthfulQA multiple-choice records.

    Uses the ``mc1_targets`` block, which has exactly one correct option.
    Option order is taken as published rather than sorted, because reordering
    would destroy the very position information the audit measures.

    Args:
        path: Source file.

    Returns:
        The corpus.

    Raises:
        ValueError: If a record lacks ``mc1_targets`` or does not have exactly
            one correct option.
    """
    items: list[Item] = []
    for line_number, record in _iter_jsonl(path):
        where = f"{path.name}:{line_number}"

        question = record.get("question")
        if not isinstance(question, str):
            raise ValueError(f"{where}: 'question' must be a string")

        targets = record.get("mc1_targets")
        if not isinstance(targets, dict) or not targets:
            raise ValueError(f"{where}: 'mc1_targets' must be a non-empty object")

        texts = [str(text) for text in targets]
        flags = [int(value) for value in targets.values()]
        correct = [index for index, flag in enumerate(flags) if flag == 1]
        if len(correct) != 1:
            raise ValueError(f"{where}: expected exactly one correct option, found {len(correct)}")

        items.append(
            Item(
                item_id=f"{path.stem}-{line_number:05d}",
                question=question,
                choices=tuple(texts),
                answer_index=correct[0],
                subject=str(record.get("category", "")),
            )
        )

    return ItemSet(name=f"truthfulqa:{path.stem}", items=tuple(items))


def load_arc_parquet(path: Path) -> ItemSet:
    """Load ARC from the parquet form the dataset hub distributes.

    The parquet layout nests the options as a struct of parallel ``text`` and
    ``label`` arrays rather than as a list of objects, so it needs its own
    reader even though the content is identical to the JSON Lines form.

    Args:
        path: The parquet file.

    Returns:
        The corpus.

    Raises:
        ImportError: If pandas is unavailable.
        ValueError: If a row's answer key is not among its own labels. Rows are
            never skipped: dropping them would change the denominator of every
            accuracy computed downstream without saying so.
    """
    try:
        import pandas as pd  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading parquet needs pandas; install it or use the JSON Lines loader"
        ) from exc

    frame = pd.read_parquet(path)
    items: list[Item] = []
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        where = f"{path.name}:row {position}"
        choices = row.choices
        texts = [str(text) for text in choices["text"]]
        labels = [str(label) for label in choices["label"]]
        key = str(row.answerKey)

        answer_index = labels.index(key) if key in labels else _answer_index(key, len(texts), where)

        items.append(
            Item(
                item_id=str(row.id),
                question=str(row.question),
                choices=tuple(texts),
                answer_index=answer_index,
            )
        )

    return ItemSet(name=f"arc:{path.parent.name}", items=tuple(items))
