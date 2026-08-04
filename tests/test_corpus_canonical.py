"""Canonical corpus format: round-tripping, validation, and sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalassay.corpus.canonical import (
    from_items,
    item_to_record,
    read_jsonl,
    record_to_item,
    subjects_of,
    subsample,
    write_jsonl,
)
from evalassay.hashing import corpus_hash
from evalassay.types import Item, ItemSet


def _item(index: int, subject: str = "alpha") -> Item:
    return Item(
        item_id=f"i-{index}",
        question=f"question {index}?",
        choices=("a", "b", "c", "d"),
        answer_index=index % 4,
        subject=subject,
    )


def _corpus(count: int = 10, subject: str = "alpha") -> ItemSet:
    return ItemSet(name="demo", items=tuple(_item(i, subject) for i in range(count)))


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    original = _corpus(5)
    path = tmp_path / "corpus.jsonl"
    write_jsonl(original, path)
    restored = read_jsonl(path, name="demo")
    assert restored == original
    assert corpus_hash(restored) == corpus_hash(original)


def test_round_trip_survives_non_ascii_content(tmp_path: Path) -> None:
    item = Item(item_id="x", question="quel été?", choices=("oui", "non"), answer_index=0)
    path = tmp_path / "unicode.jsonl"
    write_jsonl(ItemSet(name="u", items=(item,)), path)
    assert read_jsonl(path, name="u").items[0] == item


def test_write_creates_missing_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "corpus.jsonl"
    write_jsonl(_corpus(2), path)
    assert path.exists()


def test_name_defaults_to_the_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "my_benchmark.jsonl"
    write_jsonl(_corpus(2), path)
    assert read_jsonl(path).name == "my_benchmark"


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "gappy.jsonl"
    record = json.dumps(item_to_record(_item(0)))
    path.write_text(f"\n{record}\n\n", encoding="utf-8")
    assert len(read_jsonl(path)) == 1


def test_subject_is_omitted_when_absent() -> None:
    plain = Item(item_id="x", question="q", choices=("a", "b"), answer_index=0)
    assert "subject" not in item_to_record(plain)
    assert record_to_item(item_to_record(plain)).subject == ""


@pytest.mark.parametrize("field", ["id", "question", "choices", "answer_index"])
def test_missing_required_fields_are_named(field: str) -> None:
    record = item_to_record(_item(0))
    del record[field]
    with pytest.raises(ValueError, match=field):
        record_to_item(record)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("id", 7, "'id' must be a string"),
        ("question", None, "'question' must be a string"),
        ("choices", "abc", "'choices' must be a list of strings"),
        ("choices", [1, 2], "'choices' must be a list of strings"),
        ("answer_index", "0", "'answer_index' must be an integer"),
        ("answer_index", True, "'answer_index' must be an integer"),
        ("subject", 3, "'subject' must be a string"),
    ],
)
def test_wrong_types_are_rejected_by_field(field: str, value: object, expected: str) -> None:
    record = item_to_record(_item(0))
    record[field] = value
    with pytest.raises(ValueError, match=expected):
        record_to_item(record)


def test_item_level_validation_is_reported_with_its_location() -> None:
    record = item_to_record(_item(0))
    record["answer_index"] = 99
    with pytest.raises(ValueError, match=r"line 4: .*outside range"):
        record_to_item(record, where="line 4")


def test_invalid_json_reports_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    good = json.dumps(item_to_record(_item(0)))
    path.write_text(f"{good}\nnot json at all\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:2: invalid JSON"):
        read_jsonl(path)


def test_non_object_lines_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "listy.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_jsonl(path)


def test_from_items_builds_a_corpus() -> None:
    built = from_items("built", (_item(0), _item(1)))
    assert built.name == "built"
    assert len(built) == 2


def test_subsample_is_deterministic_in_the_seed() -> None:
    corpus = _corpus(50)
    first = subsample(corpus, 10, seed=99)
    second = subsample(corpus, 10, seed=99)
    assert [i.item_id for i in first] == [i.item_id for i in second]


def test_subsample_changes_with_the_seed() -> None:
    corpus = _corpus(50)
    a = [i.item_id for i in subsample(corpus, 10, seed=1)]
    b = [i.item_id for i in subsample(corpus, 10, seed=2)]
    assert a != b


def test_subsample_preserves_original_order() -> None:
    corpus = _corpus(50)
    sampled = [i.item_id for i in subsample(corpus, 12, seed=5)]
    positions = [int(item_id.split("-")[1]) for item_id in sampled]
    assert positions == sorted(positions)


def test_subsample_records_size_and_seed_in_the_name() -> None:
    sampled = subsample(_corpus(50), 7, seed=3)
    assert "n=7" in sampled.name
    assert "seed=3" in sampled.name


def test_subsample_of_the_whole_corpus_is_the_corpus() -> None:
    corpus = _corpus(10)
    assert subsample(corpus, 10, seed=1) is corpus


def test_subsample_rejects_impossible_sizes() -> None:
    corpus = _corpus(10)
    with pytest.raises(ValueError, match=r"size must be in 1\.\.10"):
        subsample(corpus, 11, seed=1)
    with pytest.raises(ValueError, match=r"size must be in 1\.\.10"):
        subsample(corpus, 0, seed=1)


def test_subsample_draws_without_replacement() -> None:
    sampled = subsample(_corpus(40), 20, seed=8)
    ids = [i.item_id for i in sampled]
    assert len(set(ids)) == len(ids)


def _mixed_corpus() -> ItemSet:
    # 60 alpha, 30 beta, 10 gamma.
    items = (
        tuple(_item(i, "alpha") for i in range(60))
        + tuple(_item(100 + i, "beta") for i in range(30))
        + tuple(_item(200 + i, "gamma") for i in range(10))
    )
    return ItemSet(name="mixed", items=items)


def test_stratified_subsample_fills_the_quota_exactly() -> None:
    sampled = subsample(_mixed_corpus(), 30, seed=4, stratify_by_subject=True)
    assert len(sampled) == 30


def test_stratified_subsample_keeps_subject_proportions() -> None:
    sampled = subsample(_mixed_corpus(), 30, seed=4, stratify_by_subject=True)
    counts = dict.fromkeys(("alpha", "beta", "gamma"), 0)
    for item in sampled:
        counts[item.subject] += 1
    assert counts == {"alpha": 18, "beta": 9, "gamma": 3}


def test_stratified_subsample_does_not_lose_a_tiny_subject() -> None:
    # A plain random sample can miss a rare subject entirely; stratifying is
    # worth the complexity precisely because that changes what is measured.
    sampled = subsample(_mixed_corpus(), 10, seed=7, stratify_by_subject=True)
    assert "gamma" in {item.subject for item in sampled}


def test_stratified_subsample_is_deterministic() -> None:
    corpus = _mixed_corpus()
    a = [i.item_id for i in subsample(corpus, 25, seed=11, stratify_by_subject=True)]
    b = [i.item_id for i in subsample(corpus, 25, seed=11, stratify_by_subject=True)]
    assert a == b


def test_stratified_quota_never_exceeds_a_subject_pool() -> None:
    sampled = subsample(_mixed_corpus(), 95, seed=2, stratify_by_subject=True)
    counts: dict[str, int] = {}
    for item in sampled:
        counts[item.subject] = counts.get(item.subject, 0) + 1
    assert counts["gamma"] <= 10
    assert len(sampled) == 95


def test_subjects_are_listed_in_sorted_order() -> None:
    assert list(subjects_of(_mixed_corpus())) == ["alpha", "beta", "gamma"]
