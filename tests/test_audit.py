"""The audit end to end, and the calibration that certifies it.

The calibration tests are the ones that matter. Everything EvalAssay says about
a real model rests on two properties that cannot be checked against a real
model, because with a real model nobody knows the right answer:

*It recovers an artifact that is really there.* An oracle built with a known
amount of memorisation must have that amount attributed to the reframing
intervention, within the interval the audit itself reports.

*It stays silent about one that is not.* An oracle with no artifacts at all must
have nothing charged against it, and an inert guesser must not be accused of
anything either.

Marked slow and selected by name, so continuous integration runs them on every
commit. A calibration that only runs when someone remembers to ask is not a
calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from evalassay.audit import AuditConfig, inert_players, run_audit
from evalassay.corpus.synthetic import CorpusSpec, generate
from evalassay.intervene.interventions import NeutralReframing
from evalassay.score.oracle import OracleScorer, OracleSpec
from evalassay.types import AuditReport, Verdict

N_ITEMS = 500
"""Enough to resolve a ten-point artifact comfortably."""


def _audit(corpus_spec: CorpusSpec, oracle: OracleSpec, seed: int = 7) -> AuditReport:
    corpus = generate(corpus_spec)
    return run_audit(corpus, OracleScorer(oracle, corpus), AuditConfig(seed=seed))


def _uniform() -> CorpusSpec:
    return CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=11)


def _charged(report: AuditReport) -> dict[str, float]:
    return {c.name: c.estimate.point for c in report.components if c.verdict is Verdict.ESTABLISHED}


# --------------------------------------------------------------------------
# Structural guarantees
# --------------------------------------------------------------------------


def test_shares_sum_to_the_total_drop() -> None:
    # The efficiency axiom, checked on a real run rather than only in the
    # abstract. This is the entire argument for using Shapley over separately
    # measured marginal effects, so it is worth checking where it is used.
    report = _audit(_uniform(), OracleSpec(skill=0.4, memorisation=0.2, seed=3))
    total = sum(c.estimate.point for c in report.components)
    assert total == pytest.approx(report.total_drop, abs=1e-12)


def test_the_audit_is_reproducible() -> None:
    spec, oracle = _uniform(), OracleSpec(skill=0.4, memorisation=0.2, seed=3)
    first, second = _audit(spec, oracle), _audit(spec, oracle)
    assert [c.estimate.point for c in first.components] == [
        c.estimate.point for c in second.components
    ]
    assert first.manifest.corpus_hash == second.manifest.corpus_hash
    assert first.manifest.config_hash == second.manifest.config_hash


def test_the_manifest_records_what_the_run_consumed() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.5, seed=3))
    manifest = report.manifest
    assert manifest.n_items == N_ITEMS
    assert manifest.corpus_hash.startswith("sha256:")
    assert manifest.config_hash.startswith("sha256:")
    assert manifest.scorer_id.startswith("oracle:")
    assert manifest.scorer_deterministic is True
    assert dict(manifest.library_versions).keys() >= {"numpy", "scipy", "python"}


def test_purity_is_the_assayed_share_of_the_reported_score() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.4, memorisation=0.3, seed=3))
    assert report.assayed_score == pytest.approx(report.reported_score - report.attributed_points)
    assert report.purity == pytest.approx(report.assayed_score / report.reported_score)


def test_unestablished_components_are_not_deducted() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.6, seed=1))
    assert all(
        c.attributed_points == 0.0
        for c in report.components
        if c.verdict is not Verdict.ESTABLISHED
    )


def test_every_component_reports_a_minimum_detectable_effect() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.5, seed=3))
    assert all(c.mde >= 0.0 for c in report.components)


def test_refusal_reasons_are_carried_into_the_description() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.6, seed=1))
    for component in report.components:
        if component.verdict is Verdict.NOT_ESTABLISHED:
            assert "not established:" in component.description


# --------------------------------------------------------------------------
# Calibration: silence when there is nothing to find
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_calibration_pure_skill_is_charged_nothing(seed: int) -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.6, seed=seed))
    assert _charged(report) == {}
    assert report.purity == pytest.approx(1.0)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [4, 5, 6])
def test_calibration_an_inert_guesser_is_charged_nothing(seed: int) -> None:
    report = _audit(_uniform(), OracleSpec(seed=seed))
    assert _charged(report) == {}


@pytest.mark.slow
def test_calibration_positional_preference_costs_nothing_on_a_uniform_key() -> None:
    # A model that always answers position one scores exactly chance whether or
    # not the options are rotated, when the benchmark's own key is uniform.
    # Positional preference is only an artifact in combination with a skewed
    # benchmark, and the audit must not charge for it otherwise.
    report = _audit(
        _uniform(),
        OracleSpec(skill=0.4, position_preference=0.3, favoured_position=1, seed=2),
    )
    assert "permute_options" not in _charged(report)


# --------------------------------------------------------------------------
# Calibration: recovering artifacts that are really there
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_calibration_memorisation_is_attributed_to_reframing() -> None:
    skill, memorisation = 0.4, 0.3
    report = _audit(_uniform(), OracleSpec(skill=skill, memorisation=memorisation, seed=3))

    charged = _charged(report)
    assert set(charged) == {"neutral_reframing"}

    # Reachable in closed form from the oracle's cascade: memorisation fires
    # only when skill did not, and once reframed those items fall back to
    # guessing among the options.
    planted = (1 - skill) * memorisation * (1 - 0.25)
    component = next(c for c in report.components if c.name == "neutral_reframing")
    assert component.estimate.ci_low <= planted <= component.estimate.ci_high


@pytest.mark.slow
def test_calibration_distractor_reliance_is_attributed_to_the_distractor_swap() -> None:
    skill, reliance = 0.4, 0.3
    report = _audit(_uniform(), OracleSpec(skill=skill, distractor_reliance=reliance, seed=4))

    charged = _charged(report)
    assert set(charged) == {"stronger_distractor"}

    planted = (1 - skill) * reliance * (1 - 0.25)
    component = next(c for c in report.components if c.name == "stronger_distractor")
    assert component.estimate.ci_low <= planted <= component.estimate.ci_high


@pytest.mark.slow
def test_calibration_positional_preference_is_attributed_to_permutation() -> None:
    # Skewed key plus positional preference: now the preference really does buy
    # accuracy, and rotating the options really does take it away.
    skewed = CorpusSpec(n_items=N_ITEMS, n_choices=4, seed=11, position_bias=0.5, biased_position=1)
    report = _audit(
        skewed, OracleSpec(skill=0.4, position_preference=0.3, favoured_position=1, seed=2)
    )

    charged = _charged(report)
    assert set(charged) == {"permute_options"}

    # The preference is right whenever the key sits at the favoured position,
    # which the planted skew makes far more likely than chance; rotation
    # reduces that to chance.
    key_at_favoured = 0.5 + 0.5 * 0.25
    planted = (1 - 0.4) * 0.3 * (key_at_favoured - 0.25)
    component = next(c for c in report.components if c.name == "permute_options")
    assert component.estimate.ci_low <= planted <= component.estimate.ci_high


@pytest.mark.slow
def test_calibration_two_artifacts_are_separated() -> None:
    report = _audit(
        _uniform(),
        OracleSpec(skill=0.3, memorisation=0.25, distractor_reliance=0.25, seed=6),
    )
    charged = _charged(report)
    assert set(charged) == {"neutral_reframing", "stronger_distractor"}
    # Shapley shares still sum to the joint drop, which is what separating
    # overlapping artifacts by hand would fail to guarantee.
    total = sum(c.estimate.point for c in report.components)
    assert total == pytest.approx(report.total_drop, abs=1e-12)


# --------------------------------------------------------------------------
# Interventions that cannot bite against a given backend
# --------------------------------------------------------------------------


def test_an_intervention_that_changes_nothing_is_identified_as_inert() -> None:
    outcomes = np.zeros((8, 5))
    outcomes[0] = [1, 1, 0, 0, 1]
    for mask in range(8):
        # Player 0 (bit 1) never changes anything; player 1 (bit 2) does.
        outcomes[mask] = outcomes[0] - (0.5 if mask & 0b010 else 0.0)
    assert inert_players(outcomes, 3) == {0, 2}


def test_a_player_that_moves_one_item_is_not_inert() -> None:
    outcomes = np.ones((8, 4))
    outcomes[0b001, 2] = 0.0
    assert 0 not in inert_players(outcomes, 3)


def test_an_inert_intervention_says_so_rather_than_not_established() -> None:
    # Reporting "not established" would read as measured-and-too-small, when the
    # truth is that nothing could be measured at all.
    corpus = generate(_uniform())
    report = run_audit(
        corpus,
        OracleScorer(OracleSpec(skill=0.5, seed=1), corpus),
        AuditConfig(seed=7, run_pathology_layer=False, measure_blind=False),
        players=(NeutralReframing(),),
    )
    component = report.components[0]
    assert component.verdict is Verdict.NOT_ESTABLISHED
    assert "inert against this backend" in component.description


# --------------------------------------------------------------------------
# Blind accuracy
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_blind_accuracy_finds_a_model_that_does_not_need_the_question() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.3, choices_only_skill=0.5, seed=8))
    assert report.blind_accuracy is not None
    planted = 0.5 + 0.5 * 0.25
    assert report.blind_accuracy.ci_low <= planted <= report.blind_accuracy.ci_high


@pytest.mark.slow
def test_blind_accuracy_sits_at_chance_for_a_model_that_does() -> None:
    report = _audit(_uniform(), OracleSpec(skill=0.8, seed=9))
    assert report.blind_accuracy is not None
    assert report.blind_accuracy.point == pytest.approx(0.25, abs=0.06)


@pytest.mark.slow
def test_choices_only_skill_is_not_charged_as_an_artifact() -> None:
    # It survives every intervention in the decomposition, so it contributes
    # nothing to any share, and surfaces in the blind diagnostic instead.
    report = _audit(_uniform(), OracleSpec(skill=0.3, choices_only_skill=0.4, seed=8))
    assert _charged(report) == {}
    assert report.blind_accuracy is not None
    assert report.blind_accuracy.point > 0.4


def test_the_blind_diagnostic_can_be_switched_off() -> None:
    corpus = generate(_uniform())
    report = run_audit(
        corpus,
        OracleScorer(OracleSpec(skill=0.5, seed=1), corpus),
        AuditConfig(seed=7, measure_blind=False, run_pathology_layer=False),
    )
    assert report.blind_accuracy is None
    assert report.findings == ()


def test_the_pathology_layer_runs_alongside_the_decomposition() -> None:
    corpus = generate(_uniform())
    report = run_audit(corpus, OracleScorer(OracleSpec(skill=0.5, seed=1), corpus), AuditConfig())
    assert {f.detector for f in report.findings} == {
        "position_skew",
        "longest_answer",
        "choices_only",
        "near_duplicate",
    }
