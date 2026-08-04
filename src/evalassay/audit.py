"""The audit: score every coalition, attribute the drop, gate the attribution.

The shape of a run
------------------
Three interventions are treated as players in a cooperative game. For each of
the eight subsets, every item is transformed by that subset and scored, giving a
per-item correctness matrix of eight rows. The Shapley operator turns those
eight coalition accuracies into three shares that sum, exactly, to the accuracy
lost when all three are applied together.

Intervals come from resampling *items*, not from re-scoring. Because the Shapley
value is linear in the coalition accuracies, a bootstrap replicate is a matrix
product over the correctness matrix that is already in memory, so ten thousand
replicates cost no model calls at all.

Why permutation is averaged rather than sampled
-----------------------------------------------
The permutation intervention places the key at every position in turn and the
item's correctness is the mean over those placements. Sampling one placement
would leave positional noise in the estimate that the audit would then have to
separate from positional preference - the very thing it is trying to measure.

Blind accuracy is measured separately
-------------------------------------
Hiding the question is not one of the three players, because the accuracy it
destroys is accuracy that *needed* the question, which is capability rather than
an artifact. It is measured as its own paired comparison and reported as a
floor.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import scipy
from numpy.typing import NDArray

from evalassay import __version__
from evalassay.hashing import config_hash, corpus_hash
from evalassay.intervene.base import Intervention, apply_coalition
from evalassay.intervene.interventions import (
    HideQuestion,
    NeutralReframing,
    PermuteOptions,
    StrongerDistractor,
    distractor_pool,
)
from evalassay.pathology.runner import run_all as run_pathology
from evalassay.score.base import ScoreCache, Scorer
from evalassay.stats.decision import GateConfig, decide
from evalassay.stats.multiplicity import holm_bonferroni
from evalassay.stats.paired import (
    bca_ci,
    bootstrap_counts,
    mcnemar_exact,
    mde_from_standard_error,
)
from evalassay.stats.shapley import (
    coalition_accuracies,
    shapley_bootstrap,
    shapley_jackknife,
    shapley_operator,
    shapley_shares,
    total_drop,
)
from evalassay.types import (
    SCHEMA_VERSION,
    AuditReport,
    Component,
    Estimate,
    Finding,
    ItemSet,
    RunManifest,
)
from evalassay.types import (
    Estimate as EstimateType,
)

FloatArray = NDArray[np.float64]

SHAPLEY_METHOD: Final = "exact Shapley over intervention coalitions; BCa item bootstrap"
BLIND_METHOD: Final = "paired exact McNemar; BCa item bootstrap"


def default_players(item_set: ItemSet) -> tuple[Intervention, ...]:
    """The three artifact-removing interventions the decomposition attributes.

    Args:
        item_set: The corpus, used to build the distractor replacement pool.

    Returns:
        The players, in canonical order.
    """
    players: tuple[Intervention, ...] = (
        StrongerDistractor(pool=distractor_pool(item_set)),
        NeutralReframing(),
        PermuteOptions(),
    )
    return players


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Pre-registered settings for a run.

    Attributes:
        seed: Root seed for every pseudo-random draw.
        gate: Default-deny thresholds.
        measure_blind: Whether to run the hidden-question diagnostic.
        run_pathology_layer: Whether to run the model-free detectors too.
    """

    seed: int = 0
    gate: GateConfig = field(default_factory=GateConfig)
    measure_blind: bool = True
    run_pathology_layer: bool = True

    def as_dict(self) -> dict[str, object]:
        """Configuration as a plain mapping, for hashing into the manifest.

        Returns:
            The settings keyed by name.
        """
        return {
            "seed": self.seed,
            "measure_blind": self.measure_blind,
            "run_pathology_layer": self.run_pathology_layer,
            **{f"gate.{key}": value for key, value in self.gate.as_dict().items()},
        }


def _coalition_outcomes(
    item_set: ItemSet,
    players: tuple[Intervention, ...],
    scorer: Scorer,
    cache: ScoreCache,
    seed: int,
) -> FloatArray:
    """Score every coalition on every item.

    Args:
        item_set: The corpus.
        players: The interventions being attributed.
        scorer: The scoring backend.
        cache: Score cache, shared across coalitions.
        seed: Run seed.

    Returns:
        A ``(2**n_players, n_items)`` matrix of per-item correctness. Entries lie
        in ``[0, 1]`` rather than being binary, because option permutation
        contributes the mean over every placement of the key.
    """
    n_players = len(players)
    outcomes = np.zeros((1 << n_players, len(item_set)), dtype=np.float64)

    for mask in range(1 << n_players):
        coalition = [players[i] for i in range(n_players) if mask & (1 << i)]
        for column, item in enumerate(item_set):
            variants = apply_coalition(coalition, item, seed)
            hits = sum(1.0 for variant in variants if cache.is_correct(scorer, variant))
            outcomes[mask, column] = hits / len(variants)

    return outcomes


def _blind_outcomes(
    item_set: ItemSet, scorer: Scorer, cache: ScoreCache, seed: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Score every item with and without its question.

    Args:
        item_set: The corpus.
        scorer: The scoring backend.
        cache: Score cache.
        seed: Run seed.

    Returns:
        Paired correctness with the question shown and with it hidden.
    """
    hide = HideQuestion()
    shown = np.zeros(len(item_set), dtype=bool)
    hidden = np.zeros(len(item_set), dtype=bool)
    for index, item in enumerate(item_set):
        shown[index] = cache.is_correct(scorer, item)
        blind_variant = apply_coalition([hide], item, seed)[0]
        hidden[index] = cache.is_correct(scorer, blind_variant)
    return shown, hidden


NUMERICAL_TOLERANCE: Final = 1e-12
"""Below this, a share is arithmetic noise rather than a small effect."""


def _bootstrap_p_value(replicates: FloatArray, point: float) -> float:
    """Two-sided bootstrap p-value against a null of no effect.

    The tail counted is the one *opposite* the point estimate: for a positive
    effect, the evidence against the null is how rarely a resample lands at or
    below zero.

    The tolerance is not decoration. An intervention that changes nothing on any
    item produces shares that are algebraically zero but land on values like
    ``1e-18`` in floating point, all of one sign. Counting the opposite tail
    then finds nothing there and returns a p-value near zero - a confident claim
    of an effect that is exactly, provably absent. Treating anything within
    numerical noise of zero as null removes that failure mode at its source.

    Args:
        replicates: Bootstrap replicates of the statistic.
        point: The statistic on the original sample.

    Returns:
        A p-value, never exactly zero: a finite resample cannot license a claim
        of zero probability, so the count is corrected by one.
    """
    if replicates.size == 0 or abs(point) <= NUMERICAL_TOLERANCE:
        return 1.0
    if float(np.max(np.abs(replicates))) <= NUMERICAL_TOLERANCE:
        return 1.0

    if point > 0.0:
        tail = int(np.count_nonzero(replicates <= 0.0))
    else:
        tail = int(np.count_nonzero(replicates >= 0.0))
    return min(1.0, 2.0 * (1 + tail) / (replicates.size + 1))


def _library_versions() -> tuple[tuple[str, str], ...]:
    """Versions of libraries whose numerics could move a result.

    Returns:
        Sorted ``(name, version)`` pairs.
    """
    return tuple(
        sorted(
            (
                ("numpy", np.__version__),
                ("python", platform.python_version()),
                ("scipy", scipy.__version__),
            )
        )
    )


def run_audit(
    item_set: ItemSet,
    scorer: Scorer,
    config: AuditConfig | None = None,
    players: tuple[Intervention, ...] | None = None,
) -> AuditReport:
    """Audit a model on a corpus and decompose its reported score.

    Args:
        item_set: The corpus.
        scorer: The scoring backend.
        config: Pre-registered settings. Defaults to :class:`AuditConfig`.
        players: The interventions to attribute. Defaults to
            :func:`default_players`.

    Returns:
        The report: the reported score, the artifact decomposition, the
        model-free corpus findings, and the run manifest.
    """
    settings = config if config is not None else AuditConfig()
    family = players if players is not None else default_players(item_set)
    cache = ScoreCache()

    outcomes = _coalition_outcomes(item_set, family, scorer, cache, settings.seed)
    accuracies = coalition_accuracies(outcomes)
    operator = shapley_operator(len(family))
    shares = shapley_shares(operator, accuracies)

    rng = np.random.default_rng(settings.seed)
    counts = bootstrap_counts(len(item_set), settings.gate.bootstrap_draws, rng)
    replicates = shapley_bootstrap(outcomes, operator, counts)
    jackknife = shapley_jackknife(outcomes, operator)

    raw_p = [_bootstrap_p_value(replicates[i], float(shares[i])) for i in range(len(family))]
    adjusted = holm_bonferroni(raw_p)

    components: list[Component] = []
    for index, player in enumerate(family):
        low, high = bca_ci(
            float(shares[index]),
            replicates[index],
            jackknife[index],
            alpha=settings.gate.alpha,
        )
        estimate = Estimate(
            point=float(shares[index]),
            ci_low=min(low, float(shares[index])),
            ci_high=max(high, float(shares[index])),
            p_value=raw_p[index],
            n=len(item_set),
            method=SHAPLEY_METHOD,
        )
        decision = decide(estimate, adjusted[index], settings.gate)
        components.append(
            Component(
                name=player.name,
                description=player.description
                if decision.established
                else f"{player.description} - not established: {decision.reason}",
                estimate=estimate,
                verdict=decision.verdict,
                adjusted_p=adjusted[index],
                mde=mde_from_standard_error(
                    float(replicates[index].std(ddof=1)),
                    settings.gate.alpha,
                    settings.gate.power,
                ),
            )
        )

    blind = None
    if settings.measure_blind:
        blind = _measure_blind(item_set, scorer, cache, settings, rng)

    findings: tuple[Finding, ...] = ()
    if settings.run_pathology_layer:
        findings = run_pathology(item_set, settings.seed, settings.gate).findings

    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        corpus_name=item_set.name,
        corpus_hash=corpus_hash(item_set),
        n_items=len(item_set),
        scorer_id=scorer.scorer_id,
        scorer_deterministic=scorer.deterministic,
        config_hash=config_hash(settings.as_dict()),
        seed=settings.seed,
        alpha=settings.gate.alpha,
        power=settings.gate.power,
        bootstrap_draws=settings.gate.bootstrap_draws,
        evalassay_version=__version__,
        library_versions=_library_versions(),
    )

    return AuditReport(
        manifest=manifest,
        reported_score=float(accuracies[0]),
        total_drop=total_drop(accuracies),
        components=tuple(components),
        findings=findings,
        chance_accuracy=item_set.chance_accuracy,
        blind_accuracy=blind,
    )


def _measure_blind(
    item_set: ItemSet,
    scorer: Scorer,
    cache: ScoreCache,
    settings: AuditConfig,
    rng: np.random.Generator,
) -> EstimateType:
    """Measure accuracy with the question removed entirely.

    Args:
        item_set: The corpus.
        scorer: The scoring backend.
        cache: Score cache.
        settings: Pre-registered settings.
        rng: Seeded generator.

    Returns:
        The blind accuracy, with an interval and the paired p-value against the
        question-shown condition. The point estimate is an accuracy, not a drop,
        because a floor is what this diagnostic is for.
    """
    shown, hidden = _blind_outcomes(item_set, scorer, cache, settings.seed)
    test = mcnemar_exact(shown, hidden)

    values = hidden.astype(np.float64)
    counts = bootstrap_counts(values.size, settings.gate.bootstrap_draws, rng)
    replicates = np.asarray((values @ counts.astype(np.float64)) / values.size, dtype=np.float64)
    leave_one_out = np.asarray((values.sum() - values) / (values.size - 1), dtype=np.float64)
    low, high = bca_ci(float(values.mean()), replicates, leave_one_out, alpha=settings.gate.alpha)

    point = float(values.mean())
    return Estimate(
        point=point,
        ci_low=min(low, point),
        ci_high=max(high, point),
        p_value=test.p_value,
        n=values.size,
        method=BLIND_METHOD,
    )
