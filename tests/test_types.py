"""Core value types, their validation, and the arithmetic of a report."""

from __future__ import annotations

import math

import pytest

from evalassay.types import (
    AuditReport,
    Component,
    Estimate,
    Finding,
    Item,
    ItemSet,
    RunManifest,
    Verdict,
)


def _item(item_id: str = "a", answer_index: int = 0, n_choices: int = 4) -> Item:
    return Item(
        item_id=item_id,
        question="Which one?",
        choices=tuple(f"option {i}" for i in range(n_choices)),
        answer_index=answer_index,
    )


def test_item_exposes_its_answer_text() -> None:
    item = _item(answer_index=2)
    assert item.answer == "option 2"
    assert item.n_choices == 4


def test_item_rejects_a_blank_identifier() -> None:
    with pytest.raises(ValueError, match="item_id"):
        Item(item_id="", question="q", choices=("a", "b"), answer_index=0)


def test_item_rejects_too_few_choices() -> None:
    with pytest.raises(ValueError, match="at least 2 choices"):
        Item(item_id="x", question="q", choices=("only",), answer_index=0)


@pytest.mark.parametrize("answer_index", [-1, 4, 99])
def test_item_rejects_an_answer_index_off_the_end(answer_index: int) -> None:
    with pytest.raises(ValueError, match="outside range"):
        Item(item_id="x", question="q", choices=("a", "b", "c", "d"), answer_index=answer_index)


def test_item_permits_an_empty_question() -> None:
    # The hide-question intervention depends on this being legal.
    item = Item(item_id="x", question="", choices=("a", "b"), answer_index=1)
    assert item.question == ""


def test_item_is_immutable() -> None:
    item = _item()
    with pytest.raises(AttributeError):
        item.question = "changed"  # type: ignore[misc]


def test_item_set_iterates_and_sizes() -> None:
    item_set = ItemSet(name="demo", items=(_item("a"), _item("b")))
    assert len(item_set) == 2
    assert [i.item_id for i in item_set] == ["a", "b"]


def test_item_set_rejects_emptiness_and_duplicates() -> None:
    with pytest.raises(ValueError, match="is empty"):
        ItemSet(name="demo", items=())
    with pytest.raises(ValueError, match="duplicate item_id"):
        ItemSet(name="demo", items=(_item("a"), _item("a")))


def test_uniform_choice_count_is_none_when_items_disagree() -> None:
    even = ItemSet(name="d", items=(_item("a"), _item("b")))
    assert even.uniform_n_choices == 4
    ragged = ItemSet(name="d", items=(_item("a"), _item("b", n_choices=3)))
    assert ragged.uniform_n_choices is None


def test_chance_accuracy_averages_over_ragged_items() -> None:
    ragged = ItemSet(name="d", items=(_item("a", n_choices=4), _item("b", n_choices=2)))
    assert ragged.chance_accuracy == pytest.approx((0.25 + 0.5) / 2)


def test_estimate_rejects_an_inverted_interval() -> None:
    with pytest.raises(ValueError, match="inverted interval"):
        Estimate(point=0.1, ci_low=0.3, ci_high=0.2, p_value=0.01, n=10, method="m")


def test_estimate_rejects_impossible_p_values_and_sizes() -> None:
    with pytest.raises(ValueError, match="p_value"):
        Estimate(point=0.1, ci_low=0.0, ci_high=0.2, p_value=1.5, n=10, method="m")
    with pytest.raises(ValueError, match="n must be positive"):
        Estimate(point=0.1, ci_low=0.0, ci_high=0.2, p_value=0.5, n=0, method="m")


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.01, 0.09, True), (-0.09, -0.01, True), (-0.01, 0.09, False), (0.0, 0.09, False)],
)
def test_excludes_zero_is_strict(low: float, high: float, expected: bool) -> None:
    estimate = Estimate(point=0.05, ci_low=low, ci_high=high, p_value=0.01, n=10, method="m")
    assert estimate.excludes_zero is expected


def _component(name: str, point: float, verdict: Verdict) -> Component:
    return Component(
        name=name,
        description=f"{name} artifact",
        estimate=Estimate(
            point=point, ci_low=point - 0.01, ci_high=point + 0.01, p_value=0.001, n=500, method="s"
        ),
        verdict=verdict,
        adjusted_p=0.001,
        mde=0.01,
    )


def test_unestablished_components_are_never_charged() -> None:
    established = _component("a", 0.05, Verdict.ESTABLISHED)
    refused = _component("b", 0.09, Verdict.NOT_ESTABLISHED)
    assert established.attributed_points == pytest.approx(0.05)
    assert refused.attributed_points == 0.0


def _manifest() -> RunManifest:
    return RunManifest(
        schema_version="1",
        corpus_name="demo",
        corpus_hash="sha256:" + "0" * 64,
        n_items=500,
        scorer_id="oracle:test",
        scorer_deterministic=True,
        config_hash="sha256:" + "1" * 64,
        seed=1,
        alpha=0.01,
        power=0.8,
        bootstrap_draws=10_000,
        evalassay_version="0.1.0",
        library_versions=(("numpy", "2.2.1"),),
    )


def _report(components: tuple[Component, ...], reported: float = 0.80) -> AuditReport:
    return AuditReport(
        manifest=_manifest(),
        reported_score=reported,
        total_drop=sum(c.estimate.point for c in components),
        components=components,
        findings=(),
    )


def test_assayed_score_subtracts_only_established_components() -> None:
    report = _report(
        (
            _component("a", 0.05, Verdict.ESTABLISHED),
            _component("b", 0.09, Verdict.NOT_ESTABLISHED),
        )
    )
    assert report.attributed_points == pytest.approx(0.05)
    assert report.assayed_score == pytest.approx(0.75)
    assert report.purity == pytest.approx(0.75 / 0.80)


def test_established_filters_the_component_list() -> None:
    report = _report(
        (
            _component("a", 0.05, Verdict.ESTABLISHED),
            _component("b", 0.09, Verdict.NOT_ESTABLISHED),
        )
    )
    assert [c.name for c in report.established] == ["a"]


def test_assayed_score_is_floored_at_zero() -> None:
    report = _report((_component("a", 0.95, Verdict.ESTABLISHED),), reported=0.30)
    assert report.assayed_score == 0.0
    assert report.purity == 0.0


def test_purity_is_undefined_when_nothing_was_reported() -> None:
    report = _report((_component("a", 0.0, Verdict.NOT_ESTABLISHED),), reported=0.0)
    assert math.isnan(report.purity)


def test_a_report_with_no_established_artifacts_is_fully_pure() -> None:
    report = _report((_component("a", 0.09, Verdict.NOT_ESTABLISHED),))
    assert report.purity == pytest.approx(1.0)


def test_findings_sit_outside_the_decomposition() -> None:
    finding = Finding(
        detector="position_skew",
        description="answer key is not uniform over positions",
        estimate=Estimate(point=0.2, ci_low=0.1, ci_high=0.3, p_value=0.0001, n=500, method="chi2"),
        verdict=Verdict.ESTABLISHED,
        adjusted_p=0.0001,
    )
    report = AuditReport(
        manifest=_manifest(),
        reported_score=0.8,
        total_drop=0.0,
        components=(),
        findings=(finding,),
    )
    # A corpus defect describes the benchmark, not the model, so it must not
    # move the assayed score.
    assert report.assayed_score == pytest.approx(0.8)
    assert report.purity == pytest.approx(1.0)
