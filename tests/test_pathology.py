"""Model-free detectors: do they find planted defects, and stay quiet otherwise?

Two properties matter and both are tested against corpora whose defects are
known exactly:

*Sensitivity* - each detector recovers its planted effect, near the magnitude
that was planted rather than merely in the right direction.

*Specificity* - on a clean corpus, every detector stays silent. This is the
property a tool like this lives or dies by. A detector that fires on a real
benchmark tells you nothing unless you know it would not have fired on a clean
one.
"""

from __future__ import annotations

import numpy as np
import pytest

from evalassay.corpus.synthetic import CorpusSpec, generate
from evalassay.pathology import run_all
from evalassay.pathology.base import tokenise, wilson_interval
from evalassay.pathology.choices_only import ChoicesOnly
from evalassay.pathology.longest_answer import LongestAnswer
from evalassay.pathology.near_duplicate import (
    JACCARD_THRESHOLD,
    NearDuplicate,
    contradiction_groups,
    deduplicated,
    exact_duplicate_groups,
    jaccard,
    shingles,
)
from evalassay.pathology.position_skew import PositionSkew, total_variation
from evalassay.pathology.runner import default_detectors
from evalassay.types import Item, ItemSet, Verdict

N_ITEMS = 600
"""Large enough to resolve the planted effects, small enough to keep tests quick."""

RECOVERY_TOLERANCE = 0.05
"""How close a recovered effect must be to the planted one."""


def _findings(corpus_spec: CorpusSpec, seed: int = 7) -> dict[str, object]:
    report = run_all(generate(corpus_spec), seed=seed)
    return {finding.detector: finding for finding in report.findings}


def _clean() -> CorpusSpec:
    return CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=1)


# --------------------------------------------------------------------------
# Specificity: silence on a clean corpus
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_no_detector_fires_on_a_clean_corpus(seed: int) -> None:
    report = run_all(generate(CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=seed)), seed=seed + 100)
    fired = [f.detector for f in report.findings if f.verdict is Verdict.ESTABLISHED]
    assert fired == [], f"false positives on a clean corpus: {fired}"


def test_every_detector_actually_ran_on_a_clean_corpus() -> None:
    # Silence must mean "asked and found nothing", never "never asked".
    report = run_all(generate(_clean()), seed=3)
    assert report.skipped == ()
    assert {f.detector for f in report.findings} == {d.name for d in default_detectors()}


def test_refusal_reasons_are_recorded_for_silent_detectors() -> None:
    report = run_all(generate(_clean()), seed=3)
    for finding in report.findings:
        if finding.verdict is Verdict.NOT_ESTABLISHED:
            assert "not established:" in finding.detail


# --------------------------------------------------------------------------
# Sensitivity: recovering planted effects
# --------------------------------------------------------------------------


def test_position_skew_recovers_planted_bias() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=2, position_bias=0.5, biased_position=2)
    finding = _findings(spec)["position_skew"]
    assert finding.verdict is Verdict.ESTABLISHED  # type: ignore[attr-defined]

    # True total variation: the biased position holds 0.625 of the key, the
    # other three hold 0.125 each.
    planted = 0.5 * (abs(0.625 - 0.25) + 3 * abs(0.125 - 0.25))
    assert planted == pytest.approx(0.375)
    # Reported as a lower bound: the null-bias subtraction is exact under the
    # null and conservative away from it.
    point = finding.estimate.point  # type: ignore[attr-defined]
    assert point < planted
    assert point == pytest.approx(planted, abs=RECOVERY_TOLERANCE)


def test_position_skew_names_the_modal_position() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=2, position_bias=0.6, biased_position=3)
    finding = _findings(spec)["position_skew"]
    assert "answer position is 3" in finding.detail  # type: ignore[attr-defined]
    assert "67" in finding.detail  # type: ignore[attr-defined]


def test_longest_answer_recovers_planted_padding() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=3, longest_answer_rate=0.4)
    finding = _findings(spec)["longest_answer"]
    assert finding.verdict is Verdict.ESTABLISHED  # type: ignore[attr-defined]
    planted = spec.expected_longest_answer_accuracy - 0.25
    assert planted == pytest.approx(0.30)
    assert finding.estimate.point == pytest.approx(planted, abs=RECOVERY_TOLERANCE)  # type: ignore[attr-defined]


def test_choices_only_recovers_planted_leakage() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=4, choices_only_rate=0.3)
    finding = _findings(spec)["choices_only"]
    assert finding.verdict is Verdict.ESTABLISHED  # type: ignore[attr-defined]
    planted = spec.expected_choices_only_accuracy - 0.25
    assert planted == pytest.approx(0.225)
    assert finding.estimate.point == pytest.approx(planted, abs=RECOVERY_TOLERANCE)  # type: ignore[attr-defined]


def test_near_duplicate_recovers_planted_repeats() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=5, duplicate_rate=0.1)
    corpus = generate(spec)
    finding = _findings(spec)["near_duplicate"]
    assert finding.verdict is Verdict.ESTABLISHED  # type: ignore[attr-defined]

    # Every planted repeat makes two items members of a duplicate group.
    extra = len(corpus) - spec.n_items
    expected_rate = 2 * extra / len(corpus)
    assert finding.estimate.point == pytest.approx(expected_rate, abs=0.01)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Orthogonality: a planted defect must not fire the wrong detector
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=2, position_bias=0.5, biased_position=2),
            "position_skew",
        ),
        (
            CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=3, longest_answer_rate=0.4),
            "longest_answer",
        ),
        (CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=4, choices_only_rate=0.3), "choices_only"),
        (CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=5, duplicate_rate=0.1), "near_duplicate"),
    ],
)
def test_a_planted_defect_fires_only_its_own_detector(spec: CorpusSpec, expected: str) -> None:
    report = run_all(generate(spec), seed=7)
    fired = sorted(f.detector for f in report.findings if f.verdict is Verdict.ESTABLISHED)
    assert fired == [expected]


# --------------------------------------------------------------------------
# The independence correction
# --------------------------------------------------------------------------


def test_duplicates_are_withheld_from_detectors_assuming_independence() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=5, duplicate_rate=0.1)
    report = run_all(generate(spec), seed=7)
    assert report.duplicates_removed > 0

    by_name = {f.detector: f for f in report.findings}
    # The duplicate detector must have seen the raw corpus.
    assert by_name["near_duplicate"].estimate.n == len(generate(spec))
    # The others must have seen the deduplicated one.
    assert by_name["choices_only"].estimate.n == len(generate(spec)) - report.duplicates_removed


def test_no_deduplication_happens_on_a_clean_corpus() -> None:
    assert run_all(generate(_clean()), seed=3).duplicates_removed == 0


def test_detectors_are_independent_of_which_others_run() -> None:
    # A detector's draws are derived from the seed and its own name, so adding
    # or removing a sibling cannot move its result.
    corpus = generate(CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=6, position_bias=0.4))
    full = run_all(corpus, seed=9)
    alone = run_all(corpus, seed=9, detectors=[PositionSkew()])

    full_point = next(f for f in full.findings if f.detector == "position_skew").estimate.point
    assert alone.findings[0].estimate.point == pytest.approx(full_point)


def test_running_the_family_is_reproducible() -> None:
    corpus = generate(CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=6, choices_only_rate=0.2))
    first = run_all(corpus, seed=11)
    second = run_all(corpus, seed=11)
    assert [f.estimate.point for f in first.findings] == [f.estimate.point for f in second.findings]


# --------------------------------------------------------------------------
# Runner behaviour
# --------------------------------------------------------------------------


def test_detectors_decline_on_a_corpus_too_small_to_measure() -> None:
    tiny = ItemSet(
        name="tiny",
        items=tuple(
            Item(item_id=f"t{i}", question="q", choices=("a", "b", "c", "d"), answer_index=i % 4)
            for i in range(6)
        ),
    )
    report = run_all(tiny, seed=1)
    assert set(report.skipped) == {d.name for d in default_detectors()}
    assert report.findings == ()


def test_multiplicity_correction_is_applied_across_the_family() -> None:
    report = run_all(generate(_clean()), seed=3)
    for finding in report.findings:
        assert finding.adjusted_p >= finding.estimate.p_value


def test_established_filters_the_finding_list() -> None:
    spec = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=2, position_bias=0.6)
    report = run_all(generate(spec), seed=7)
    assert [f.detector for f in report.established] == ["position_skew"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_total_variation_is_zero_only_for_uniform_counts() -> None:
    uniform = np.array([25.0, 25.0, 25.0, 25.0], dtype=np.float64)
    degenerate = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert total_variation(uniform) == pytest.approx(0.0)
    assert total_variation(degenerate) == pytest.approx(0.75)
    assert total_variation(np.zeros(4, dtype=np.float64)) == 0.0


def test_total_variation_is_scale_invariant() -> None:
    counts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    scaled = np.asarray(counts * 7, dtype=np.float64)
    assert total_variation(counts) == pytest.approx(total_variation(scaled))


def _sh(*runs: str) -> frozenset[tuple[str, ...]]:
    """Build a shingle set from whitespace-separated runs."""
    return frozenset(tuple(run.split()) for run in runs)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (_sh("a b c"), _sh("a b c"), 1.0),
        (_sh("a b c"), _sh("d e f"), 0.0),
        (_sh("a b c", "b c d"), _sh("a b c", "x y z"), 1 / 3),
        (frozenset(), frozenset(), 0.0),
        (_sh("a b c"), frozenset(), 0.0),
    ],
)
def test_jaccard(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]], expected: float
) -> None:
    assert jaccard(left, right) == pytest.approx(expected)


def test_shingles_keep_word_order() -> None:
    # The correction that motivated shingles: two items whose token sets are
    # identical but whose word order differs must not look identical.
    left = Item(item_id="a", question="the function f must be", choices=("f", "g"), answer_index=0)
    right = Item(item_id="b", question="the function g must be", choices=("f", "g"), answer_index=0)
    assert jaccard(shingles(left), shingles(right)) < JACCARD_THRESHOLD


def test_a_short_item_still_produces_a_shingle() -> None:
    tiny = Item(item_id="t", question="hi", choices=("a", "b"), answer_index=0)
    assert shingles(tiny)


def test_contradictions_need_identical_questions_not_similar_ones() -> None:
    # Fuzzy similarity produced false contradictions on real data, so this claim
    # rests on exact identity of question and options.
    left = Item(item_id="a", question="which letter?", choices=("x", "y"), answer_index=0)
    right = Item(item_id="b", question="which letter?", choices=("x", "y"), answer_index=1)
    nearly = Item(item_id="c", question="which letter now?", choices=("x", "y"), answer_index=1)

    found = contradiction_groups(ItemSet(name="c", items=(left, right, nearly)))
    assert len(found) == 1
    assert sorted(next(iter(found.values()))) == [0, 1]


def test_agreeing_repeats_are_not_contradictions() -> None:
    left = Item(item_id="a", question="q", choices=("x", "y"), answer_index=0)
    right = Item(item_id="b", question="q", choices=("x", "y"), answer_index=0)
    assert contradiction_groups(ItemSet(name="c", items=(left, right))) == {}


def test_tokenise_lowercases_and_keeps_internal_punctuation() -> None:
    assert tokenise("The Cat's HAT, well-worn!") == ["the", "cat's", "hat", "well-worn"]


def test_tokenise_drops_standalone_punctuation() -> None:
    assert tokenise("--- ??? !!!") == []


def test_wilson_interval_brackets_the_proportion() -> None:
    low, high = wilson_interval(50, 100, alpha=0.05)
    assert low < 0.5 < high


def test_wilson_interval_stays_inside_the_unit_range_at_the_edges() -> None:
    low, high = wilson_interval(0, 100, alpha=0.01)
    assert low == 0.0
    assert 0.0 < high < 1.0
    low, high = wilson_interval(100, 100, alpha=0.01)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_wilson_interval_rejects_impossible_arguments() -> None:
    with pytest.raises(ValueError, match="trials must be positive"):
        wilson_interval(0, 0, alpha=0.05)
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(5, 3, alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        wilson_interval(1, 3, alpha=1.0)


def _dup_corpus() -> ItemSet:
    base = Item(item_id="a", question="q one", choices=("x", "y"), answer_index=0)
    copy = Item(item_id="b", question="q one", choices=("x", "y"), answer_index=0)
    other = Item(item_id="c", question="q two", choices=("x", "y"), answer_index=1)
    return ItemSet(name="d", items=(base, copy, other))


def test_exact_duplicate_groups_reports_only_repeated_digests() -> None:
    groups = exact_duplicate_groups(_dup_corpus())
    assert len(groups) == 1
    assert next(iter(groups.values())) == [0, 1]


def test_deduplicated_keeps_the_first_occurrence() -> None:
    unique, removed = deduplicated(_dup_corpus())
    assert removed == 1
    assert [item.item_id for item in unique] == ["a", "c"]
    assert "deduplicated" in unique.name


def test_deduplicated_returns_the_corpus_untouched_when_clean() -> None:
    corpus = generate(_clean())
    unique, removed = deduplicated(corpus)
    assert removed == 0
    assert unique is corpus


def test_near_duplicate_detects_repeats_despite_a_tiny_vocabulary() -> None:
    # Regression: an earlier candidate-generation scheme skipped every token
    # that appeared in more than a tenth of items, which on a small-vocabulary
    # corpus meant every token, so no pair was ever compared and a corpus that
    # was a tenth duplicates was reported clean.
    spec = CorpusSpec(n_items=200, n_choices=4, seed=31, duplicate_rate=0.2)
    corpus = generate(spec)
    finding = NearDuplicate().run(corpus, np.random.default_rng(0))
    assert finding is not None
    assert finding.estimate.point > 0.2


def test_near_duplicate_flags_contradictory_keys() -> None:
    left = Item(item_id="a", question="same question", choices=("x", "y"), answer_index=0)
    right = Item(item_id="b", question="same question", choices=("x", "y"), answer_index=1)
    filler = tuple(
        Item(item_id=f"f{i}", question=f"filler {i}", choices=("p", "q"), answer_index=0)
        for i in range(20)
    )
    corpus = ItemSet(name="c", items=(left, right, *filler))
    finding = NearDuplicate().run(corpus, np.random.default_rng(0))
    assert finding is not None
    assert "disagree about the answer" in finding.detail


def test_detectors_expose_whether_they_assume_independent_items() -> None:
    assert PositionSkew().assumes_independent_items
    assert LongestAnswer().assumes_independent_items
    assert ChoicesOnly().assumes_independent_items
    assert not NearDuplicate().assumes_independent_items
