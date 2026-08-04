"""Rendering and serialisation.

The property that matters most here is negative: a quantity the gate refused
must never appear as a number. Printing an unestablished estimate in the same
column as an established one invites exactly the misreading the whole tool
exists to prevent.
"""

from __future__ import annotations

import json
import math

import pytest

from evalassay.report.render import render, render_blind, render_findings
from evalassay.report.serialise import report_to_dict, to_json
from evalassay.types import (
    AuditReport,
    Component,
    Estimate,
    Finding,
    RunManifest,
    Verdict,
)


def _estimate(point: float = 0.05, low: float = 0.03, high: float = 0.07) -> Estimate:
    return Estimate(point=point, ci_low=low, ci_high=high, p_value=0.0004, n=250, method="test")


def _component(name: str, verdict: Verdict, point: float = 0.05) -> Component:
    description = "an artifact"
    if verdict is Verdict.NOT_ESTABLISHED:
        description = "an artifact - not established: adjusted p 0.4000 exceeds alpha 0.0100"
    return Component(
        name=name,
        description=description,
        estimate=_estimate(point),
        verdict=verdict,
        adjusted_p=0.0004 if verdict is Verdict.ESTABLISHED else 0.4,
        mde=0.0195,
    )


def _manifest(deterministic: bool = True) -> RunManifest:
    return RunManifest(
        schema_version="1",
        corpus_name="demo-corpus",
        corpus_hash="sha256:" + "ab" * 32,
        n_items=250,
        scorer_id="oracle:test",
        scorer_deterministic=deterministic,
        config_hash="sha256:" + "cd" * 32,
        seed=7,
        alpha=0.01,
        power=0.8,
        bootstrap_draws=10_000,
        evalassay_version="0.1.0",
        library_versions=(("numpy", "2.2.1"), ("scipy", "1.17.1")),
    )


def _report(
    *,
    deterministic: bool = True,
    blind: Estimate | None = None,
    findings: tuple[Finding, ...] = (),
    reported: float = 0.72,
) -> AuditReport:
    components = (
        _component("charged_one", Verdict.ESTABLISHED, 0.06),
        _component("refused_one", Verdict.NOT_ESTABLISHED, 0.09),
    )
    return AuditReport(
        manifest=_manifest(deterministic),
        reported_score=reported,
        total_drop=0.15,
        components=components,
        findings=findings,
        chance_accuracy=0.25,
        blind_accuracy=blind,
    )


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------


def test_an_unestablished_component_is_never_printed_as_a_number() -> None:
    text = render(_report())
    refused_line = next(line for line in text.splitlines() if "refused_one" in line)
    assert "0.09" not in refused_line
    assert "not established" in refused_line


def test_an_unestablished_component_reports_what_it_could_have_detected() -> None:
    assert "MDE 0.0195" in render(_report())


def test_an_unestablished_component_gives_the_refusal_reason() -> None:
    assert "exceeds alpha" in render(_report())


def test_an_established_component_is_printed_as_a_deduction() -> None:
    text = render(_report())
    charged_line = next(line for line in text.splitlines() if "charged_one" in line)
    assert "-0.0600" in charged_line
    assert "charged" in charged_line


def test_provenance_precedes_the_result() -> None:
    lines = render(_report()).splitlines()
    corpus_at = next(i for i, line in enumerate(lines) if "demo-corpus" in line)
    score_at = next(i for i, line in enumerate(lines) if "Reported score" in line)
    assert corpus_at < score_at


def test_the_header_carries_the_hashes_and_thresholds() -> None:
    text = render(_report())
    assert "sha256:abababababab" in text
    assert "sha256:cdcdcdcdcdcd" in text
    assert "alpha=0.01" in text
    assert "seed=7" in text
    assert "numpy 2.2.1" in text


def test_purity_and_assayed_capability_are_shown() -> None:
    text = render(_report())
    assert "Assayed capability" in text
    assert "0.6600" in text
    assert "91.7%" in text


def test_a_non_deterministic_backend_is_called_out() -> None:
    # A report that claimed reproducibility it cannot deliver would be worse
    # than one that admits the gap.
    text = render(_report(deterministic=False))
    assert "NOT deterministic" in text
    assert "does not certify reproducibility" in text


def test_a_deterministic_backend_carries_no_warning() -> None:
    assert "does not certify reproducibility" not in render(_report())


def test_purity_renders_when_undefined() -> None:
    report = AuditReport(
        manifest=_manifest(),
        reported_score=0.0,
        total_drop=0.0,
        components=(),
        findings=(),
        chance_accuracy=0.25,
    )
    assert math.isnan(report.purity)
    assert "n/a" in render(report)


# --------------------------------------------------------------------------
# The blind diagnostic and the findings block
# --------------------------------------------------------------------------


def test_blind_accuracy_is_shown_with_its_excess_over_chance() -> None:
    blind = Estimate(point=0.37, ci_low=0.31, ci_high=0.43, p_value=0.001, n=250, method="m")
    text = "\n".join(render_blind(_report(blind=blind)))
    assert "0.3700" in text
    assert "0.1200" in text
    assert "not answering the question" in text


def test_a_blind_score_at_chance_gets_no_accusation() -> None:
    blind = Estimate(point=0.255, ci_low=0.21, ci_high=0.30, p_value=0.9, n=250, method="m")
    text = "\n".join(render_blind(_report(blind=blind)))
    assert "not answering the question" not in text


def test_the_blind_block_is_omitted_when_it_was_not_measured() -> None:
    assert render_blind(_report(blind=None)) == []


def test_findings_are_marked_as_describing_the_benchmark() -> None:
    finding = Finding(
        detector="position_skew",
        description="key is not uniform",
        estimate=_estimate(0.2, 0.15, 0.25),
        verdict=Verdict.ESTABLISHED,
        adjusted_p=0.0001,
        detail="most common answer position is 1",
    )
    text = "\n".join(render_findings(_report(findings=(finding,))))
    assert "position_skew" in text
    assert "most common answer position is 1" in text
    assert "do not move the" in text


def test_the_findings_block_is_omitted_when_the_layer_did_not_run() -> None:
    assert render_findings(_report(findings=())) == []


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_the_json_parses_and_carries_the_manifest() -> None:
    parsed = json.loads(to_json(_report()))
    assert parsed["manifest"]["corpus_hash"].startswith("sha256:")
    assert parsed["manifest"]["n_items"] == 250
    assert parsed["manifest"]["library_versions"]["numpy"] == "2.2.1"


def test_verdicts_are_written_as_words_not_booleans() -> None:
    # "not established" must not be readable as "refuted", and a boolean field
    # would collapse that distinction.
    parsed = json.loads(to_json(_report()))
    verdicts = {c["name"]: c["verdict"] for c in parsed["components"]}
    assert verdicts == {"charged_one": "established", "refused_one": "not_established"}


def test_the_json_carries_more_than_the_text() -> None:
    # Unadjusted p-values and refusal reasons let a reader re-apply their own
    # thresholds without re-running the audit.
    parsed = json.loads(to_json(_report()))
    refused = next(c for c in parsed["components"] if c["name"] == "refused_one")
    assert refused["estimate"]["p_value"] == pytest.approx(0.0004)
    assert refused["adjusted_p"] == pytest.approx(0.4)
    assert "not established" in refused["description"]
    assert refused["attributed_points"] == 0.0


def test_the_json_is_byte_identical_across_runs() -> None:
    # Which is what makes the reproducibility claim checkable with diff.
    assert to_json(_report()) == to_json(_report())


def test_undefined_numbers_become_null_rather_than_nan() -> None:
    # JSON has no NaN; emitting the non-standard literal would break strict
    # parsers without warning.
    report = AuditReport(
        manifest=_manifest(),
        reported_score=0.0,
        total_drop=0.0,
        components=(),
        findings=(),
        chance_accuracy=0.25,
    )
    text = to_json(report)
    assert "NaN" not in text
    assert json.loads(text)["purity"] is None


def test_a_missing_blind_measurement_is_null() -> None:
    assert json.loads(to_json(_report(blind=None)))["blind_accuracy"] is None


def test_the_dict_form_matches_the_json_form() -> None:
    assert report_to_dict(_report()) == json.loads(to_json(_report()))


def test_the_compact_form_has_no_newlines() -> None:
    assert "\n" not in to_json(_report(), indent=0)
