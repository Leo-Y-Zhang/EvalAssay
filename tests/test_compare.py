"""Comparing two audits, and refusing when they are not comparable.

Two reported scores are worth subtracting only if everything except the factor
under study was held fixed. The manifests exist so that the check is mechanical
rather than a matter of the reader's memory, and the tests below are mostly
about the refusals: a comparison that quietly succeeds on incomparable runs
produces a plausible number that means nothing, which is worse than an error.
"""

from __future__ import annotations

import json

import pytest

from evalassay.audit import AuditConfig, run_audit
from evalassay.corpus.synthetic import CorpusSpec, generate
from evalassay.report.compare import NotComparableError, compare, render_comparison
from evalassay.report.serialise import from_json, report_from_dict, report_to_dict, to_json
from evalassay.score.oracle import OracleScorer, OracleSpec
from evalassay.stats.decision import GateConfig
from evalassay.types import AuditReport


def _report(
    *,
    corpus_seed: int = 11,
    oracle: OracleSpec | None = None,
    alpha: float = 0.01,
    items: int = 300,
) -> AuditReport:
    corpus = generate(CorpusSpec(n_items=items, n_choices=4, seed=corpus_seed))
    spec = oracle if oracle is not None else OracleSpec(skill=0.5, seed=1)
    return run_audit(
        corpus,
        OracleScorer(spec, corpus),
        AuditConfig(
            seed=7,
            gate=GateConfig(alpha=alpha, bootstrap_draws=2000),
            run_pathology_layer=False,
            measure_blind=False,
        ),
    )


# --------------------------------------------------------------------------
# Round-tripping, which is what entitles the format to be called machine-readable
# --------------------------------------------------------------------------


def test_a_report_survives_a_round_trip_through_json() -> None:
    original = _report()
    restored = from_json(to_json(original))
    assert restored == original


def test_the_round_trip_is_byte_stable() -> None:
    original = _report()
    assert to_json(from_json(to_json(original))) == to_json(original)


def test_the_dict_round_trip_agrees_with_the_json_one() -> None:
    original = _report()
    assert report_from_dict(report_to_dict(original)) == original


def test_a_report_with_every_optional_part_round_trips() -> None:
    corpus = generate(CorpusSpec(n_items=300, n_choices=4, seed=11))
    full = run_audit(
        corpus,
        OracleScorer(OracleSpec(skill=0.4, memorisation=0.3, seed=2), corpus),
        AuditConfig(seed=7, gate=GateConfig(bootstrap_draws=2000)),
    )
    assert full.blind_accuracy is not None
    assert full.findings
    assert from_json(to_json(full)) == full


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------


def test_two_scorers_on_one_corpus_are_differenced() -> None:
    weak = _report(oracle=OracleSpec(skill=0.3, seed=1))
    strong = _report(oracle=OracleSpec(skill=0.8, seed=1))
    result = compare(weak, strong)

    assert result.reported_delta == pytest.approx(strong.reported_score - weak.reported_score)
    assert result.reported_delta > 0
    assert result.n_items == 300


def test_the_comparison_names_artifacts_established_in_only_one_run() -> None:
    clean = _report(oracle=OracleSpec(skill=0.5, seed=1))
    memorising = _report(oracle=OracleSpec(skill=0.3, memorisation=0.4, seed=1))
    result = compare(clean, memorising)
    assert "neutral_reframing" in result.charged_only_in_variant
    assert result.charged_only_in_baseline == ()


def test_a_different_corpus_is_refused() -> None:
    # The difference would be a statement about the items, not the model.
    with pytest.raises(NotComparableError, match="different corpora"):
        compare(_report(corpus_seed=11), _report(corpus_seed=12))


def test_a_different_item_count_is_refused() -> None:
    with pytest.raises(NotComparableError, match="different corpora"):
        compare(_report(items=300), _report(items=200))


def test_different_thresholds_are_refused() -> None:
    with pytest.raises(NotComparableError, match="different thresholds"):
        compare(_report(alpha=0.01), _report(alpha=0.05))


def test_comparing_a_run_with_itself_is_refused() -> None:
    report = _report()
    with pytest.raises(NotComparableError, match="nothing to compare"):
        compare(report, report)


def test_the_rendered_comparison_shows_both_sides_and_the_difference() -> None:
    weak = _report(oracle=OracleSpec(skill=0.3, seed=1))
    strong = _report(oracle=OracleSpec(skill=0.8, seed=1))
    text = render_comparison(compare(weak, strong))

    assert "EvalAssay comparison" in text
    assert "baseline" in text
    assert "variant" in text
    assert f"{strong.reported_score:.4f}" in text
    assert "attributable to whatever distinguishes the two scorers" in text


def test_the_rendered_comparison_names_a_baseline_only_artifact() -> None:
    # The asymmetric case: something established in the baseline and not in the
    # variant is the direction a reader is most likely to care about, since it
    # says the change removed an artifact.
    memorising = _report(oracle=OracleSpec(skill=0.3, memorisation=0.4, seed=1))
    clean = _report(oracle=OracleSpec(skill=0.5, seed=1))
    text = render_comparison(compare(memorising, clean))
    assert "charged only in baseline: neutral_reframing" in text


def test_the_rendered_comparison_names_a_variant_only_artifact() -> None:
    clean = _report(oracle=OracleSpec(skill=0.5, seed=1))
    memorising = _report(oracle=OracleSpec(skill=0.3, memorisation=0.4, seed=1))
    text = render_comparison(compare(clean, memorising))
    assert "charged only in variant:" in text
    assert "neutral_reframing" in text


def test_the_rendered_comparison_says_when_nothing_differed() -> None:
    left = _report(oracle=OracleSpec(skill=0.50, seed=1))
    right = _report(oracle=OracleSpec(skill=0.52, seed=1))
    text = render_comparison(compare(left, right))
    assert "the same artifacts were established in both runs" in text


# --------------------------------------------------------------------------
# Through the command line
# --------------------------------------------------------------------------


def test_the_cli_compares_two_saved_reports(
    capsys: pytest.CaptureFixture[str], tmp_path: object
) -> None:
    from pathlib import Path

    from evalassay.cli import main

    directory = Path(str(tmp_path))
    weak = directory / "weak.json"
    strong = directory / "strong.json"
    weak.write_text(to_json(_report(oracle=OracleSpec(skill=0.3, seed=1))), encoding="utf-8")
    strong.write_text(to_json(_report(oracle=OracleSpec(skill=0.8, seed=1))), encoding="utf-8")

    assert main(["compare", str(weak), str(strong)]) == 0
    output = capsys.readouterr().out
    assert "EvalAssay comparison" in output
    assert "reported score" in output


def test_the_cli_refuses_incomparable_reports(
    capsys: pytest.CaptureFixture[str], tmp_path: object
) -> None:
    from pathlib import Path

    from evalassay.cli import main

    directory = Path(str(tmp_path))
    left = directory / "left.json"
    right = directory / "right.json"
    left.write_text(to_json(_report(corpus_seed=11)), encoding="utf-8")
    right.write_text(to_json(_report(corpus_seed=12)), encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        main(["compare", str(left), str(right)])
    assert caught.value.code == 2
    assert "different corpora" in capsys.readouterr().err


def test_a_saved_report_is_valid_json_with_sorted_keys() -> None:
    text = to_json(_report())
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)
