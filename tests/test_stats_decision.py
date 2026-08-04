"""The default-deny gate, including every reason it can refuse."""

from __future__ import annotations

import pytest

from evalassay.stats.decision import GateConfig, decide
from evalassay.types import Estimate, Verdict


def _estimate(point: float, low: float, high: float, p: float = 0.0001) -> Estimate:
    return Estimate(point=point, ci_low=low, ci_high=high, p_value=p, n=500, method="test")


def test_a_clear_effect_is_established() -> None:
    decision = decide(_estimate(0.06, 0.04, 0.08), adjusted_p=0.0002, config=GateConfig())
    assert decision.verdict is Verdict.ESTABLISHED
    assert decision.reason == ""
    assert decision.established


def test_refuses_when_the_adjusted_p_value_misses_alpha() -> None:
    decision = decide(_estimate(0.06, 0.04, 0.08), adjusted_p=0.03, config=GateConfig())
    assert decision.verdict is Verdict.NOT_ESTABLISHED
    assert "exceeds alpha" in decision.reason


def test_refuses_when_the_interval_straddles_zero() -> None:
    decision = decide(_estimate(0.06, -0.01, 0.13), adjusted_p=0.001, config=GateConfig())
    assert decision.verdict is Verdict.NOT_ESTABLISHED
    assert "includes zero" in decision.reason


def test_refuses_an_effect_too_small_to_matter() -> None:
    decision = decide(_estimate(0.002, 0.001, 0.003), adjusted_p=0.0001, config=GateConfig())
    assert decision.verdict is Verdict.NOT_ESTABLISHED
    assert "below minimum reportable" in decision.reason


def test_refuses_to_charge_an_intervention_that_helped() -> None:
    decision = decide(_estimate(-0.05, -0.08, -0.02), adjusted_p=0.0001, config=GateConfig())
    assert decision.verdict is Verdict.NOT_ESTABLISHED
    assert "does not reduce accuracy" in decision.reason


def test_can_be_configured_to_report_effects_in_either_direction() -> None:
    config = GateConfig(require_positive=False)
    decision = decide(_estimate(-0.05, -0.08, -0.02), adjusted_p=0.0001, config=config)
    assert decision.verdict is Verdict.ESTABLISHED


def test_reasons_are_reported_in_a_fixed_order() -> None:
    # Fails significance, interval and size at once; significance is checked first.
    decision = decide(_estimate(0.001, -0.01, 0.01), adjusted_p=0.9, config=GateConfig())
    assert "exceeds alpha" in decision.reason


def test_rejects_an_out_of_range_adjusted_p_value() -> None:
    with pytest.raises(ValueError, match="adjusted_p"):
        decide(_estimate(0.05, 0.03, 0.07), adjusted_p=1.4, config=GateConfig())


def test_config_rejects_thresholds_that_cannot_produce_an_audit() -> None:
    with pytest.raises(ValueError, match="alpha"):
        GateConfig(alpha=0.0)
    with pytest.raises(ValueError, match="power"):
        GateConfig(power=1.0)
    with pytest.raises(ValueError, match="bootstrap_draws"):
        GateConfig(bootstrap_draws=10)
    with pytest.raises(ValueError, match="min_effect"):
        GateConfig(min_effect=-0.1)


def test_config_serialises_for_hashing() -> None:
    config = GateConfig(alpha=0.02, power=0.9, bootstrap_draws=2000, min_effect=0.01)
    assert config.as_dict() == {
        "alpha": 0.02,
        "power": 0.9,
        "bootstrap_draws": 2000,
        "min_effect": 0.01,
        "require_positive": True,
    }


def test_default_alpha_is_stricter_than_convention() -> None:
    # Documented as deliberate: a false artifact claim is an accusation.
    assert GateConfig().alpha <= 0.01
