"""Corpus loading, generation and sampling.

Every benchmark enters the pipeline through this package and leaves it as an
:class:`~evalassay.types.ItemSet`. Nothing downstream knows which publisher a
corpus came from, which is what lets the same audit run against a real benchmark
and against a synthetic one with known planted defects.
"""

from __future__ import annotations

from evalassay.corpus.canonical import (
    REQUIRED_FIELDS,
    from_items,
    item_to_record,
    read_jsonl,
    record_to_item,
    subjects_of,
    subsample,
    write_jsonl,
)
from evalassay.corpus.loaders import (
    load_arc_jsonl,
    load_hellaswag_jsonl,
    load_mmlu_csv,
    load_mmlu_directory,
    load_truthfulqa_mc_jsonl,
)
from evalassay.corpus.synthetic import (
    MARKER_WORD,
    PAD_WORDS,
    VOCABULARY,
    WORD_LENGTH,
    CorpusSpec,
    generate,
)

__all__ = [
    "MARKER_WORD",
    "PAD_WORDS",
    "REQUIRED_FIELDS",
    "VOCABULARY",
    "WORD_LENGTH",
    "CorpusSpec",
    "from_items",
    "generate",
    "item_to_record",
    "load_arc_jsonl",
    "load_hellaswag_jsonl",
    "load_mmlu_csv",
    "load_mmlu_directory",
    "load_truthfulqa_mc_jsonl",
    "read_jsonl",
    "record_to_item",
    "subjects_of",
    "subsample",
    "write_jsonl",
]
