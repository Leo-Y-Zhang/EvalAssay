"""The default-deny gate.

A decomposition that always prints a number for every artifact is
indistinguishable from one that prints numbers at random, because it has no way
to be wrong. This gate is what makes the audit falsifiable: a quantity is
reported as a deduction only if it clears every pre-registered condition, and
otherwise it is reported as not established, with the reason it failed.

The conditions are deliberately conservative and deliberately asymmetric. An
artifact must be statistically significant after family-wise correction, its
interval must exclude zero, it must be practically large enough to matter, and
it must point in the direction of harming the model. Anything else is treated as
absent. The audit is therefore generous to the model it is auditing, which is
the only defensible bias for a tool whose output is a criticism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from evalassay.types import Estimate, Verdict

MIN_BOOTSTRAP_DRAWS: Final = 1000
"""Below this the percentile tails are too coarse for a stable interval."""


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Pre-registered thresholds for the default-deny gate.

    These are hashed into the run manifest, so a reader can tell whether the
    thresholds were chosen before or after seeing the result.

    Attributes:
        alpha: Family-wise significance level. Defaults to 0.01 rather than the
            conventional 0.05 because the cost of a false artifact claim here is
            an unfair accusation against a model or a benchmark.
        power: Target power, used only to report the minimum detectable effect
            alongside a null result.
        bootstrap_draws: Bootstrap resamples per audit.
        min_effect: Smallest effect worth reporting, in accuracy points. An
            effect below this is treated as absent even if significant, because
            a statistically real half-point artifact is not a finding anyone
            should act on.
        require_positive: Whether a component must reduce accuracy to be
            charged. Interventions that help the model are reported but never
            deducted.
    """

    alpha: float = 0.01
    power: float = 0.80
    bootstrap_draws: int = 10_000
    min_effect: float = 0.005
    require_positive: bool = True

    def __post_init__(self) -> None:
        """Reject configurations that could not produce a meaningful audit.

        Raises:
            ValueError: If any threshold is outside its valid range.
        """
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha {self.alpha} outside (0, 1)")
        if not 0.0 < self.power < 1.0:
            raise ValueError(f"power {self.power} outside (0, 1)")
        if self.bootstrap_draws < MIN_BOOTSTRAP_DRAWS:
            raise ValueError(
                f"bootstrap_draws {self.bootstrap_draws} too small for a stable "
                f"interval; use at least {MIN_BOOTSTRAP_DRAWS}"
            )
        if self.min_effect < 0.0:
            raise ValueError(f"min_effect {self.min_effect} must not be negative")

    def as_dict(self) -> dict[str, float | int | bool]:
        """Configuration as a plain mapping, for hashing into the manifest.

        Returns:
            The thresholds keyed by name.
        """
        return {
            "alpha": self.alpha,
            "power": self.power,
            "bootstrap_draws": self.bootstrap_draws,
            "min_effect": self.min_effect,
            "require_positive": self.require_positive,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """Result of applying the gate to one quantity.

    Attributes:
        verdict: Whether the quantity was established.
        reason: Empty when established; otherwise the first condition that
            failed, phrased so a reader never has to guess why a number is
            missing from the report.
    """

    verdict: Verdict
    reason: str

    @property
    def established(self) -> bool:
        """Whether the quantity cleared the gate."""
        return self.verdict is Verdict.ESTABLISHED


def decide(estimate: Estimate, adjusted_p: float, config: GateConfig) -> Decision:
    """Apply the default-deny gate to one estimate.

    Conditions are checked in a fixed order and the first failure is reported,
    so the reason is deterministic rather than depending on evaluation order.

    Args:
        estimate: The point estimate with its interval and raw p-value.
        adjusted_p: Family-wise adjusted p-value for this quantity.
        config: Pre-registered thresholds.

    Returns:
        The decision and, on failure, the reason.

    Raises:
        ValueError: If ``adjusted_p`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= adjusted_p <= 1.0:
        raise ValueError(f"adjusted_p {adjusted_p} outside [0, 1]")

    if adjusted_p > config.alpha:
        return Decision(
            Verdict.NOT_ESTABLISHED,
            f"adjusted p {adjusted_p:.4f} exceeds alpha {config.alpha:.4f}",
        )

    if not estimate.excludes_zero:
        return Decision(
            Verdict.NOT_ESTABLISHED,
            f"interval [{estimate.ci_low:+.4f}, {estimate.ci_high:+.4f}] includes zero",
        )

    if abs(estimate.point) < config.min_effect:
        return Decision(
            Verdict.NOT_ESTABLISHED,
            f"effect {estimate.point:+.4f} below minimum reportable {config.min_effect:.4f}",
        )

    if config.require_positive and estimate.point <= 0.0:
        return Decision(
            Verdict.NOT_ESTABLISHED,
            f"effect {estimate.point:+.4f} does not reduce accuracy; not charged",
        )

    return Decision(Verdict.ESTABLISHED, "")
