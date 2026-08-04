r"""The committed example report must stay a true example.

`docs/example-report.txt` and `docs/example-report.json` exist so a reader can
see what this tool produces without installing or running anything. A committed
artifact that drifts out of date is worse than no artifact, because it looks
authoritative while describing a version that no longer exists.

So the example is regenerated here and compared against what is on disk. The
version line is excluded from the comparison, since library versions differ
legitimately between machines; everything else must match exactly, which is what
catches a change in layout, wording or arithmetic.

Regenerate with:

    python -m evalassay.cli demo --seed 11 --audit-seed 7 --bootstrap 2000 \\
        --skill 0.4 --memorisation 0.25 --distractor 0.2 \\
        --json docs/example-report.json > docs/example-report.txt

then delete the line naming the JSON destination, which would otherwise record a
local path in a committed file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalassay.audit import AuditConfig, run_audit
from evalassay.corpus.synthetic import CorpusSpec, generate
from evalassay.report.render import render
from evalassay.report.serialise import to_json
from evalassay.score.oracle import OracleScorer, OracleSpec
from evalassay.stats.decision import GateConfig

DOCS = Path(__file__).parent.parent / "docs"
TEXT = DOCS / "example-report.txt"
JSON = DOCS / "example-report.json"


def _rebuild() -> tuple[str, str]:
    """Reproduce the example exactly as the documented command would.

    Returns:
        The rendered text and the serialised JSON.
    """
    corpus = generate(CorpusSpec(n_items=500, n_choices=4, seed=11))
    oracle = OracleSpec(skill=0.4, memorisation=0.25, distractor_reliance=0.2, seed=11)
    report = run_audit(
        corpus,
        OracleScorer(oracle, corpus),
        AuditConfig(seed=7, gate=GateConfig(bootstrap_draws=2000)),
    )
    return render(report), to_json(report)


def _without_versions(text: str) -> list[str]:
    """Drop the line that names library versions.

    Args:
        text: Report text.

    Returns:
        The remaining lines.
    """
    return [line for line in text.splitlines() if not line.startswith("version ")]


def test_the_example_files_exist() -> None:
    assert TEXT.exists(), "the committed example report is missing"
    assert JSON.exists(), "the committed example JSON is missing"


def test_the_example_text_still_matches_what_the_tool_produces() -> None:
    rendered, _ = _rebuild()
    rendered_lines = _without_versions(rendered)
    committed_lines = _without_versions(TEXT.read_text(encoding="utf-8"))

    # The committed file wraps the report in the demo's preamble and closing
    # remark, so the report itself is compared where it starts.
    start = committed_lines.index("EvalAssay report")
    assert committed_lines[start : start + len(rendered_lines)] == rendered_lines


def test_the_example_json_still_matches_what_the_tool_produces() -> None:
    _, serialised = _rebuild()
    fresh = json.loads(serialised)
    committed = json.loads(JSON.read_text(encoding="utf-8"))

    fresh["manifest"].pop("library_versions")
    committed["manifest"].pop("library_versions")
    assert fresh == committed


def test_the_example_records_no_local_path() -> None:
    # A committed artifact must say nothing about the machine that produced it.
    text = TEXT.read_text(encoding="utf-8")
    assert "machine-readable report written" not in text
    for marker in ("C:\\", "/home/", "/Users/"):
        assert marker not in text


def test_the_example_demonstrates_both_outcomes() -> None:
    # It is only worth committing if it shows the instrument charging what was
    # planted and refusing what was not.
    text = TEXT.read_text(encoding="utf-8")
    assert "charged" in text
    assert "not established" in text


def test_the_example_shares_sum_to_the_observed_drop() -> None:
    committed = json.loads(JSON.read_text(encoding="utf-8"))
    total = sum(component["estimate"]["point"] for component in committed["components"])
    assert total == pytest.approx(committed["total_drop"], abs=1e-9)
