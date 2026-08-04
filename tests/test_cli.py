"""The command line interface.

The ``demo`` subcommand gets the most attention here, because it is the first
thing a reviewer runs and the only one that needs no model, no dataset and no
network. If it ever stopped demonstrating what it claims to demonstrate, the
project's central argument would be unsupported at the point of first contact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalassay.cli import LOADERS, build_parser, main


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    assert main(list(argv)) == 0
    return capsys.readouterr().out


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


def test_demo_runs_without_a_model_or_a_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    output = _run(capsys, "demo", "--bootstrap", "2000")
    assert "EvalAssay report" in output
    assert "Assayed capability" in output


def test_demo_states_what_it_planted_before_reporting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _run(capsys, "demo", "--bootstrap", "2000")
    planted_at = output.index("planted in it")
    report_at = output.index("EvalAssay report")
    assert planted_at < report_at


def test_demo_charges_the_artifacts_it_planted(capsys: pytest.CaptureFixture[str]) -> None:
    output = _run(
        capsys,
        "demo",
        "--skill",
        "0.4",
        "--memorisation",
        "0.3",
        "--distractor",
        "0.0",
        "--bootstrap",
        "2000",
    )
    reframing = next(line for line in output.splitlines() if "neutral_reframing" in line)
    assert "charged" in reframing


def test_demo_refuses_an_artifact_it_did_not_plant(capsys: pytest.CaptureFixture[str]) -> None:
    output = _run(
        capsys,
        "demo",
        "--skill",
        "0.5",
        "--memorisation",
        "0.0",
        "--distractor",
        "0.0",
        "--bootstrap",
        "2000",
    )
    for name in ("neutral_reframing", "stronger_distractor", "permute_options"):
        line = next(line for line in output.splitlines() if name in line)
        assert "not established" in line, f"{name} was charged without being planted"


def test_demo_plants_the_skew_that_makes_positional_preference_matter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A positional preference buys nothing on a uniform key, so asking the demo
    # for one has to skew the benchmark as well or it would demonstrate nothing.
    output = _run(
        capsys,
        "demo",
        "--skill",
        "0.3",
        "--memorisation",
        "0.0",
        "--distractor",
        "0.0",
        "--position",
        "0.4",
        "--bootstrap",
        "2000",
    )
    permutation = next(line for line in output.splitlines() if "permute_options" in line)
    assert "charged" in permutation


def test_demo_can_write_machine_readable_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    destination = tmp_path / "nested" / "report.json"
    output = _run(capsys, "demo", "--bootstrap", "2000", "--json", str(destination))
    assert destination.exists()
    assert "machine-readable report written" in output
    assert json.loads(destination.read_text(encoding="utf-8"))["manifest"]["n_items"] == 500


def test_demo_is_reproducible(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    _run(capsys, "demo", "--bootstrap", "2000", "--json", str(first))
    _run(capsys, "demo", "--bootstrap", "2000", "--json", str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# pathology and convert
# --------------------------------------------------------------------------


def _write_canonical(path: Path, count: int = 60) -> Path:
    lines = [
        json.dumps(
            {
                "id": f"i{index}",
                "question": f"question {index}?",
                "choices": ["alpha", "beta", "gamma", "delta"],
                "answer_index": index % 4,
            }
        )
        for index in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_pathology_inspects_a_corpus_without_a_model(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    corpus = _write_canonical(tmp_path / "corpus.jsonl")
    output = _run(capsys, "pathology", str(corpus))
    assert "Benchmark defects" in output
    for detector in ("position_skew", "longest_answer", "choices_only", "near_duplicate"):
        assert detector in output


def test_pathology_says_when_a_detector_could_not_be_measured(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A question that was never asked must not read as one asked and answered.
    corpus = _write_canonical(tmp_path / "tiny.jsonl", count=12)
    output = _run(capsys, "pathology", str(corpus))
    assert "not measured on a corpus this small" in output


def test_convert_round_trips_through_the_canonical_format(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = _write_canonical(tmp_path / "in.jsonl")
    destination = tmp_path / "out.jsonl"
    output = _run(capsys, "convert", str(source), str(destination), "--format", "canonical")
    assert "wrote 60 items" in output

    # Content survives exactly; byte-for-byte equality is not the claim, because
    # conversion canonicalises key order on the way through. That canonical form
    # is the point: two files with the same items hash alike whatever order
    # their fields were written in.
    original = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    converted = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert converted == original

    second_pass = tmp_path / "again.jsonl"
    _run(capsys, "convert", str(destination), str(second_pass), "--format", "canonical")
    assert second_pass.read_text(encoding="utf-8") == destination.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Parsing and failure handling
# --------------------------------------------------------------------------


def test_every_advertised_format_has_a_loader() -> None:
    parser = build_parser()
    assert parser is not None
    assert set(LOADERS) >= {"arc", "canonical", "mmlu-csv", "mmlu-dir", "truthfulqa"}


def test_the_gate_thresholds_are_configurable() -> None:
    args = build_parser().parse_args(
        ["demo", "--alpha", "0.05", "--power", "0.9", "--min-effect", "0.02"]
    )
    assert args.alpha == 0.05
    assert args.power == 0.9
    assert args.min_effect == 0.02


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_a_missing_corpus_exits_cleanly_rather_than_traceback(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["pathology", str(tmp_path / "absent.jsonl")])
    assert caught.value.code == 2


def test_an_unusable_threshold_exits_cleanly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    corpus = _write_canonical(tmp_path / "corpus.jsonl")
    with pytest.raises(SystemExit) as caught:
        main(["pathology", str(corpus), "--alpha", "0"])
    assert caught.value.code == 2
    assert "alpha" in capsys.readouterr().err


def test_the_version_flag_reports_the_package_version() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
