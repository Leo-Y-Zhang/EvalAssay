"""Benchmark loaders, tested against fixtures in each publisher's layout.

No benchmark data ships with this project, so these fixtures reproduce the
documented shape of each format rather than sampling the real files. That is
enough to pin the parsing contract and to prove the loaders fail loudly rather
than silently dropping rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalassay.corpus.loaders import (
    load_arc_jsonl,
    load_hellaswag_jsonl,
    load_mmlu_csv,
    load_mmlu_directory,
    load_truthfulqa_mc_jsonl,
)

MMLU_ROWS = 'What is 2 + 2?,3,4,5,6,B\n"Which, if any, is a noun?",run,quickly,harbour,very,C\n'


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    return _write(path, "\n".join(json.dumps(record) for record in records) + "\n")


# --------------------------------------------------------------------------
# MMLU
# --------------------------------------------------------------------------


def test_mmlu_reads_headerless_rows(tmp_path: Path) -> None:
    corpus = load_mmlu_csv(_write(tmp_path / "world_history.csv", MMLU_ROWS))
    assert len(corpus) == 2
    assert corpus.items[0].question == "What is 2 + 2?"
    assert corpus.items[0].choices == ("3", "4", "5", "6")
    assert corpus.items[0].answer == "4"


def test_mmlu_handles_quoted_fields_containing_commas(tmp_path: Path) -> None:
    corpus = load_mmlu_csv(_write(tmp_path / "grammar.csv", MMLU_ROWS))
    assert corpus.items[1].question == "Which, if any, is a noun?"
    assert corpus.items[1].answer == "harbour"


def test_mmlu_derives_the_subject_from_the_filename(tmp_path: Path) -> None:
    corpus = load_mmlu_csv(_write(tmp_path / "world_history.csv", MMLU_ROWS))
    assert corpus.items[0].subject == "world history"
    assert corpus.name == "mmlu:world history"


def test_mmlu_subject_can_be_overridden(tmp_path: Path) -> None:
    corpus = load_mmlu_csv(_write(tmp_path / "wh.csv", MMLU_ROWS), subject="custom")
    assert corpus.items[0].subject == "custom"


def test_mmlu_skips_blank_rows(tmp_path: Path) -> None:
    corpus = load_mmlu_csv(_write(tmp_path / "s.csv", MMLU_ROWS + "\n,,,,\n"))
    assert len(corpus) == 2


def test_mmlu_rejects_a_short_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected question, options and answer"):
        load_mmlu_csv(_write(tmp_path / "s.csv", "only,two\n"))


def test_mmlu_rejects_an_unusable_answer_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognised answer label"):
        load_mmlu_csv(_write(tmp_path / "s.csv", "q,a,b,c,d,Z!\n"))


def test_mmlu_rejects_an_answer_label_past_the_last_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"outside 0\.\.3"):
        load_mmlu_csv(_write(tmp_path / "s.csv", "q,a,b,c,d,F\n"))


def test_mmlu_directory_loads_subjects_in_sorted_order(tmp_path: Path) -> None:
    _write(tmp_path / "zoology.csv", MMLU_ROWS)
    _write(tmp_path / "anatomy.csv", MMLU_ROWS)
    corpus = load_mmlu_directory(tmp_path)
    assert len(corpus) == 4
    # Sorted filename order, not filesystem order, so the corpus hash is stable.
    assert corpus.items[0].subject == "anatomy"
    assert corpus.items[2].subject == "zoology"


def test_mmlu_directory_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no CSV files"):
        load_mmlu_directory(tmp_path)


# --------------------------------------------------------------------------
# ARC
# --------------------------------------------------------------------------


def _arc_record(
    answer_key: str = "B", labels: tuple[str, ...] = ("A", "B", "C", "D")
) -> dict[str, object]:
    return {
        "id": "ARC-1",
        "question": {
            "stem": "Which is a metal?",
            "choices": [
                {"text": text, "label": label}
                for text, label in zip(("wood", "iron", "glass", "cloth"), labels, strict=True)
            ],
        },
        "answerKey": answer_key,
    }


def test_arc_reads_the_nested_question_block(tmp_path: Path) -> None:
    corpus = load_arc_jsonl(_jsonl(tmp_path / "arc.jsonl", [_arc_record()]))
    assert corpus.items[0].item_id == "ARC-1"
    assert corpus.items[0].question == "Which is a metal?"
    assert corpus.items[0].answer == "iron"


def test_arc_accepts_numeric_labels(tmp_path: Path) -> None:
    # Some releases label options 1-4 instead of A-D within the same benchmark.
    record = _arc_record(answer_key="2", labels=("1", "2", "3", "4"))
    corpus = load_arc_jsonl(_jsonl(tmp_path / "arc.jsonl", [record]))
    assert corpus.items[0].answer == "iron"


def test_arc_prefers_the_items_own_labels(tmp_path: Path) -> None:
    # Labels that are neither letters nor digits still resolve, because the
    # item's own label list is consulted first.
    record = _arc_record(answer_key="ii", labels=("i", "ii", "iii", "iv"))
    corpus = load_arc_jsonl(_jsonl(tmp_path / "arc.jsonl", [record]))
    assert corpus.items[0].answer == "iron"


def test_arc_rejects_a_missing_question_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'question' must be an object"):
        load_arc_jsonl(_jsonl(tmp_path / "arc.jsonl", [{"id": "x", "answerKey": "A"}]))


def test_arc_rejects_a_malformed_choice(tmp_path: Path) -> None:
    record = _arc_record()
    choices = record["question"]["choices"]  # type: ignore[index]
    choices[0] = {"text": "wood"}
    with pytest.raises(ValueError, match="each choice needs 'text' and 'label'"):
        load_arc_jsonl(_jsonl(tmp_path / "arc.jsonl", [record]))


def test_arc_rejects_an_unresolvable_answer_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognised answer label"):
        load_arc_jsonl(_jsonl(tmp_path / "arc.jsonl", [_arc_record(answer_key="???")]))


# --------------------------------------------------------------------------
# HellaSwag
# --------------------------------------------------------------------------


def _hellaswag_record(label: object = 2) -> dict[str, object]:
    return {
        "ind": 41,
        "ctx": "A man is sharpening a knife. He",
        "endings": ["sings", "sleeps", "tests the edge", "flies"],
        "label": label,
        "activity_label": "Kitchen",
    }


def test_hellaswag_reads_context_and_endings(tmp_path: Path) -> None:
    corpus = load_hellaswag_jsonl(_jsonl(tmp_path / "hs.jsonl", [_hellaswag_record()]))
    item = corpus.items[0]
    assert item.item_id == "41"
    assert item.question == "A man is sharpening a knife. He"
    assert item.answer == "tests the edge"
    assert item.subject == "Kitchen"


def test_hellaswag_accepts_a_string_label(tmp_path: Path) -> None:
    corpus = load_hellaswag_jsonl(_jsonl(tmp_path / "hs.jsonl", [_hellaswag_record(label="2")]))
    assert corpus.items[0].answer == "tests the edge"


def test_hellaswag_refuses_the_unlabelled_split(tmp_path: Path) -> None:
    # The public test split ships without labels. Loading it silently would
    # produce an accuracy computed against nothing.
    with pytest.raises(ValueError, match="cannot be scored"):
        load_hellaswag_jsonl(_jsonl(tmp_path / "hs.jsonl", [_hellaswag_record(label="")]))


def test_hellaswag_rejects_a_non_integer_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'label' must be an integer"):
        load_hellaswag_jsonl(_jsonl(tmp_path / "hs.jsonl", [_hellaswag_record(label="third")]))


def test_hellaswag_rejects_an_empty_endings_list(tmp_path: Path) -> None:
    record = _hellaswag_record()
    record["endings"] = []
    with pytest.raises(ValueError, match="'endings' must be a non-empty list"):
        load_hellaswag_jsonl(_jsonl(tmp_path / "hs.jsonl", [record]))


# --------------------------------------------------------------------------
# TruthfulQA
# --------------------------------------------------------------------------


def _truthfulqa_record(targets: dict[str, int] | None = None) -> dict[str, object]:
    return {
        "question": "What happens if you crack your knuckles?",
        "mc1_targets": targets
        if targets is not None
        else {"Nothing in particular happens": 1, "You get arthritis": 0},
        "category": "Misconceptions",
    }


def test_truthfulqa_reads_the_mc1_block(tmp_path: Path) -> None:
    corpus = load_truthfulqa_mc_jsonl(_jsonl(tmp_path / "tqa.jsonl", [_truthfulqa_record()]))
    item = corpus.items[0]
    assert item.answer == "Nothing in particular happens"
    assert item.subject == "Misconceptions"


def test_truthfulqa_preserves_published_option_order(tmp_path: Path) -> None:
    # Sorting the options would destroy the position information the audit
    # exists to measure.
    targets = {"zebra answer": 0, "apple answer": 1}
    corpus = load_truthfulqa_mc_jsonl(_jsonl(tmp_path / "tqa.jsonl", [_truthfulqa_record(targets)]))
    assert corpus.items[0].choices == ("zebra answer", "apple answer")
    assert corpus.items[0].answer_index == 1


@pytest.mark.parametrize(
    "targets",
    [
        {"a": 1, "b": 1},
        {"a": 0, "b": 0},
    ],
)
def test_truthfulqa_rejects_a_key_that_is_not_exactly_one(
    tmp_path: Path, targets: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="exactly one correct option"):
        load_truthfulqa_mc_jsonl(_jsonl(tmp_path / "tqa.jsonl", [_truthfulqa_record(targets)]))


def test_truthfulqa_rejects_a_missing_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'mc1_targets' must be a non-empty object"):
        load_truthfulqa_mc_jsonl(_jsonl(tmp_path / "tqa.jsonl", [{"question": "q"}]))


# --------------------------------------------------------------------------
# Shared behaviour
# --------------------------------------------------------------------------


def test_jsonl_loaders_report_the_offending_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "arc.jsonl", json.dumps(_arc_record()) + "\nbroken\n")
    with pytest.raises(ValueError, match=r"arc\.jsonl:2: invalid JSON"):
        load_arc_jsonl(path)


def test_jsonl_loaders_skip_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "arc.jsonl", "\n" + json.dumps(_arc_record()) + "\n\n")
    assert len(load_arc_jsonl(path)) == 1
