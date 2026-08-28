from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from math import isfinite, log
from typing import cast

import numpy as np
import pytest

from corum.baselines import DecisionCosts
from corum.calibration import (
    OBSERVATION_ORDER,
    PairKey,
    fit_panel_calibrations,
    fit_reviewer_pair_calibration,
)
from corum.decision import DecisionPolicy
from corum.dependence import DependenceModel, fit_dependence
from corum.fusion import (
    BatchFusedPosterior,
    FusionContext,
    build_fusion_context,
    fuse_review_matrix,
)
from corum.metrics import policy_candidates, select_decision_policy
from corum.models import (
    CalibrationExample,
    ExecutionState,
    FusedPosterior,
    Observation,
    Reviewer,
    Truth,
)
from corum.simulation import (
    ReviewerSpec,
    Scenario,
    ScenarioPhase,
    SimulatedPanel,
    simulate_experiment,
)

_SCENARIO_NAMES = (
    "heterogeneous_pair",
    "low_prevalence_pair",
    "same_lineage_independent",
    "missing_pair_member",
)
_BASE_SEEDS = tuple(10_007 + 137 * index for index in range(8))
_FIT_CASES = (160, 640)
_POLICY_CASES = 400
_TEST_CASES = 2_500
_POSTERIOR_DRAWS = 256
_CREDIBLE_MASS = 0.95
_MATRIX_CHUNK_SIZE = 4_096
_MARGINAL_PRIOR_STRENGTH = 1.5
_PAIR_PRODUCT_PRIOR_STRENGTH = 9.0
_MIN_PAIRED_PER_TRUTH = 30
_DEPENDENCE_SHRINKAGE = 0.25
_MINIMUM_OVERLAP = 10
_LINEAGE_CAP = 1.0
_MINIMUM_COVERAGE = 0.50
_NLL_EPSILON = 1e-15
_BOOTSTRAP_DRAWS = 2_000
_BOOTSTRAP_SEED = 20_260_917
_COSTS = DecisionCosts(false_pass=1.0, false_fail=0.2, defer=0.1)

_PASS_ACTION = np.int8(0)
_FAIL_ACTION = np.int8(1)
_DEFER_ACTION = np.int8(2)

_POLICY_GRID = tuple(
    (pass_threshold, fail_threshold, 2, 2, minimum_ess)
    for pass_threshold in (0.80, 0.90, 0.95)
    for fail_threshold in (0.05, 0.10, 0.20)
    for minimum_ess in (1.0, 1.5)
)

_PAIR_KEYS: Mapping[str, PairKey] = {
    "heterogeneous_pair": ("pair-a", "pair-b"),
    "low_prevalence_pair": ("pair-x", "pair-y"),
    "same_lineage_independent": ("pair-a", "pair-b"),
    "missing_pair_member": ("pair-a", "pair-b"),
}

_GATE_AUDIT_STATE: dict[str, object] = {}


def _spec(
    reviewer_id: str,
    lineage: str,
    pass_row: tuple[float, float, float],
    fail_row: tuple[float, float, float],
    *,
    timeout_rate: float = 0.0,
) -> ReviewerSpec:
    return ReviewerSpec(
        reviewer=Reviewer(
            reviewer_id=reviewer_id,
            vendor="synthetic",
            family=reviewer_id,
            lineage=lineage,
            cost=1.0,
        ),
        likelihoods=np.array((pass_row, fail_row), dtype=float),
        timeout_rate=timeout_rate,
        invalid_rate=0.0,
    )


def _phase(
    reviewers: tuple[ReviewerSpec, ...],
    *,
    prior_pass: float,
    pair_correlation: float | None,
) -> ScenarioPhase:
    return ScenarioPhase(
        reviewers=reviewers,
        prior_pass=prior_pass,
        lineage_error_correlation=(
            {} if pair_correlation is None else {"pair": pair_correlation}
        ),
        difficulty_rate=0.0,
        informative_missingness=0.0,
        adversarial_reviewer_id=None,
    )


def _locked_scenarios() -> Mapping[str, Scenario]:
    heterogeneous = (
        _spec("solo-high", "solo-high", (0.86, 0.10, 0.04), (0.13, 0.83, 0.04)),
        _spec("pair-b", "pair", (0.71, 0.22, 0.07), (0.28, 0.66, 0.06)),
        _spec("solo-mid", "solo-mid", (0.75, 0.18, 0.07), (0.24, 0.71, 0.05)),
        _spec("pair-a", "pair", (0.78, 0.17, 0.05), (0.22, 0.73, 0.05)),
    )
    low_prevalence = (
        _spec("pair-x", "pair", (0.68, 0.26, 0.06), (0.30, 0.64, 0.06)),
        _spec("solo-high", "solo-high", (0.88, 0.08, 0.04), (0.12, 0.84, 0.04)),
        _spec("pair-y", "pair", (0.74, 0.20, 0.06), (0.26, 0.69, 0.05)),
        _spec("solo-mid", "solo-mid", (0.77, 0.17, 0.06), (0.22, 0.73, 0.05)),
    )
    independent = (
        _spec("solo-high", "solo-high", (0.84, 0.12, 0.04), (0.12, 0.84, 0.04)),
        _spec("pair-b", "pair", (0.70, 0.23, 0.07), (0.23, 0.70, 0.07)),
        _spec("solo-mid", "solo-mid", (0.75, 0.19, 0.06), (0.19, 0.75, 0.06)),
        _spec("pair-a", "pair", (0.77, 0.18, 0.05), (0.18, 0.77, 0.05)),
    )
    missing_calibration = (
        _spec("solo-high", "solo-high", (0.85, 0.11, 0.04), (0.11, 0.85, 0.04)),
        _spec("solo-mid", "solo-mid", (0.72, 0.20, 0.08), (0.20, 0.72, 0.08)),
        _spec("pair-a", "pair", (0.73, 0.22, 0.05), (0.22, 0.73, 0.05)),
        _spec("pair-b", "pair", (0.75, 0.20, 0.05), (0.20, 0.75, 0.05)),
    )
    missing_test = (
        missing_calibration[0],
        missing_calibration[1],
        missing_calibration[2],
        _spec(
            "pair-b",
            "pair",
            (0.75, 0.20, 0.05),
            (0.20, 0.75, 0.05),
            timeout_rate=0.50,
        ),
    )
    phases = {
        "heterogeneous_pair": (
            _phase(heterogeneous, prior_pass=0.62, pair_correlation=0.58),
            _phase(heterogeneous, prior_pass=0.62, pair_correlation=0.58),
        ),
        "low_prevalence_pair": (
            _phase(low_prevalence, prior_pass=0.45, pair_correlation=0.72),
            _phase(low_prevalence, prior_pass=0.45, pair_correlation=0.72),
        ),
        "same_lineage_independent": (
            _phase(independent, prior_pass=0.55, pair_correlation=None),
            _phase(independent, prior_pass=0.55, pair_correlation=None),
        ),
        "missing_pair_member": (
            _phase(
                missing_calibration,
                prior_pass=0.70,
                pair_correlation=0.64,
            ),
            _phase(missing_test, prior_pass=0.70, pair_correlation=0.64),
        ),
    }
    return {
        name: Scenario(name=name, calibration=phases[name][0], test=phases[name][1])
        for name in _SCENARIO_NAMES
    }


def _reviewer_snapshot(
    reviewer_id: str,
    lineage: str,
    pass_row: tuple[float, float, float],
    fail_row: tuple[float, float, float],
    timeout_rate: float = 0.0,
) -> tuple[object, ...]:
    return (
        reviewer_id,
        "synthetic",
        reviewer_id,
        lineage,
        1.0,
        (pass_row, fail_row),
        timeout_rate,
        0.0,
    )


def _phase_literal(
    reviewers: tuple[tuple[object, ...], ...],
    prior_pass: float,
    correlation: tuple[tuple[str, float], ...],
) -> tuple[object, ...]:
    return (reviewers, prior_pass, correlation, 0.0, 0.0, None)


_HETEROGENEOUS_SNAPSHOT = _phase_literal(
    (
        _reviewer_snapshot(
            "solo-high", "solo-high", (0.86, 0.10, 0.04), (0.13, 0.83, 0.04)
        ),
        _reviewer_snapshot("pair-b", "pair", (0.71, 0.22, 0.07), (0.28, 0.66, 0.06)),
        _reviewer_snapshot(
            "solo-mid", "solo-mid", (0.75, 0.18, 0.07), (0.24, 0.71, 0.05)
        ),
        _reviewer_snapshot("pair-a", "pair", (0.78, 0.17, 0.05), (0.22, 0.73, 0.05)),
    ),
    0.62,
    (("pair", 0.58),),
)
_LOW_PREVALENCE_SNAPSHOT = _phase_literal(
    (
        _reviewer_snapshot("pair-x", "pair", (0.68, 0.26, 0.06), (0.30, 0.64, 0.06)),
        _reviewer_snapshot(
            "solo-high", "solo-high", (0.88, 0.08, 0.04), (0.12, 0.84, 0.04)
        ),
        _reviewer_snapshot("pair-y", "pair", (0.74, 0.20, 0.06), (0.26, 0.69, 0.05)),
        _reviewer_snapshot(
            "solo-mid", "solo-mid", (0.77, 0.17, 0.06), (0.22, 0.73, 0.05)
        ),
    ),
    0.45,
    (("pair", 0.72),),
)
_INDEPENDENT_SNAPSHOT = _phase_literal(
    (
        _reviewer_snapshot(
            "solo-high", "solo-high", (0.84, 0.12, 0.04), (0.12, 0.84, 0.04)
        ),
        _reviewer_snapshot("pair-b", "pair", (0.70, 0.23, 0.07), (0.23, 0.70, 0.07)),
        _reviewer_snapshot(
            "solo-mid", "solo-mid", (0.75, 0.19, 0.06), (0.19, 0.75, 0.06)
        ),
        _reviewer_snapshot("pair-a", "pair", (0.77, 0.18, 0.05), (0.18, 0.77, 0.05)),
    ),
    0.55,
    (),
)
_MISSING_CALIBRATION_SNAPSHOT = _phase_literal(
    (
        _reviewer_snapshot(
            "solo-high", "solo-high", (0.85, 0.11, 0.04), (0.11, 0.85, 0.04)
        ),
        _reviewer_snapshot(
            "solo-mid", "solo-mid", (0.72, 0.20, 0.08), (0.20, 0.72, 0.08)
        ),
        _reviewer_snapshot("pair-a", "pair", (0.73, 0.22, 0.05), (0.22, 0.73, 0.05)),
        _reviewer_snapshot("pair-b", "pair", (0.75, 0.20, 0.05), (0.20, 0.75, 0.05)),
    ),
    0.70,
    (("pair", 0.64),),
)
_MISSING_TEST_SNAPSHOT = _phase_literal(
    (
        _reviewer_snapshot(
            "solo-high", "solo-high", (0.85, 0.11, 0.04), (0.11, 0.85, 0.04)
        ),
        _reviewer_snapshot(
            "solo-mid", "solo-mid", (0.72, 0.20, 0.08), (0.20, 0.72, 0.08)
        ),
        _reviewer_snapshot("pair-a", "pair", (0.73, 0.22, 0.05), (0.22, 0.73, 0.05)),
        _reviewer_snapshot(
            "pair-b",
            "pair",
            (0.75, 0.20, 0.05),
            (0.20, 0.75, 0.05),
            0.50,
        ),
    ),
    0.70,
    (("pair", 0.64),),
)
_SCENARIO_SNAPSHOTS: Mapping[
    str,
    tuple[tuple[object, ...], tuple[object, ...]],
] = {
    "heterogeneous_pair": (_HETEROGENEOUS_SNAPSHOT, _HETEROGENEOUS_SNAPSHOT),
    "low_prevalence_pair": (
        _LOW_PREVALENCE_SNAPSHOT,
        _LOW_PREVALENCE_SNAPSHOT,
    ),
    "same_lineage_independent": (_INDEPENDENT_SNAPSHOT, _INDEPENDENT_SNAPSHOT),
    "missing_pair_member": (
        _MISSING_CALIBRATION_SNAPSHOT,
        _MISSING_TEST_SNAPSHOT,
    ),
}


@dataclass(frozen=True, slots=True)
class _Score:
    cases: int
    fail_cases: int
    decided: int
    false_passes: int
    loss_units: int
    brier_sum: float
    nll_sum: float

    @property
    def loss(self) -> float:
        return self.loss_units / (10.0 * self.cases)

    @property
    def coverage(self) -> float:
        return self.decided / self.cases

    @property
    def false_pass_rate(self) -> float:
        return self.false_passes / self.fail_cases

    @property
    def brier(self) -> float:
        return self.brier_sum / self.cases

    @property
    def nll(self) -> float:
        return self.nll_sum / self.cases


@dataclass(frozen=True, slots=True)
class _ProbabilityTotals:
    cases: int
    brier_sum: float
    nll_sum: float


@dataclass(frozen=True, slots=True)
class _ReferenceSelection:
    policy: tuple[float, float, int, int, float]
    constraint_satisfied: bool
    score: _Score


@dataclass(frozen=True, slots=True)
class _MissingAudit:
    exactly_one_cases: int
    total_cases: int
    max_probability_delta: float
    max_lower_delta: float
    max_upper_delta: float
    both_pair: _ProbabilityTotals
    both_naive: _ProbabilityTotals
    both_power: _ProbabilityTotals


@dataclass(frozen=True, slots=True)
class _RunRecord:
    scenario: str
    fit_cases: int
    base_seed: int
    simulation_seed: int
    pair: _Score
    naive: _Score
    power: _Score
    majority: _Score
    policy_constraints: tuple[bool, bool, bool]
    joint_row_totals: tuple[int, int]
    missing: _MissingAudit | None
    gate_violations: int


def _phase_snapshot(phase: ScenarioPhase) -> tuple[object, ...]:
    reviewers = tuple(
        (
            spec.reviewer.reviewer_id,
            spec.reviewer.vendor,
            spec.reviewer.family,
            spec.reviewer.lineage,
            spec.reviewer.cost,
            tuple(tuple(float(value) for value in row) for row in spec.likelihoods),
            spec.timeout_rate,
            spec.invalid_rate,
        )
        for spec in phase.reviewers
    )
    return (
        reviewers,
        phase.prior_pass,
        tuple(sorted(phase.lineage_error_correlation.items())),
        phase.difficulty_rate,
        phase.informative_missingness,
        phase.adversarial_reviewer_id,
    )


def _canonical_policy(
    policy: DecisionPolicy,
) -> tuple[float, float, int, int, float]:
    return (
        policy.pass_threshold,
        policy.fail_threshold,
        policy.min_valid_reviewers,
        policy.min_lineages,
        policy.min_effective_sample_size,
    )


def _split_ids(
    calibration: SimulatedPanel,
    fit_cases: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(calibration.truths))
    assert len(ordered) == fit_cases + _POLICY_CASES
    fit_ids = ordered[:fit_cases]
    policy_ids = ordered[fit_cases:]
    assert len(fit_ids) == fit_cases
    assert len(policy_ids) == _POLICY_CASES
    assert set(fit_ids).isdisjoint(policy_ids)
    return fit_ids, policy_ids


def _examples_for(
    panel: SimulatedPanel,
    case_ids: Sequence[str],
) -> tuple[CalibrationExample, ...]:
    selected = set(case_ids)
    return tuple(
        CalibrationExample(panel.truths[review.case_id], review)
        for review in panel.reviews
        if review.case_id in selected
    )


def _review_matrix(
    panel: SimulatedPanel,
    case_ids: Sequence[str],
    reviewer_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    row_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
    column_by_reviewer = {
        reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
    }
    observations = np.full(
        (len(case_ids), len(reviewer_ids)),
        -1,
        dtype=np.int64,
    )
    valid_mask = np.zeros_like(observations, dtype=bool)
    seen: set[tuple[str, str]] = set()
    for review in panel.reviews:
        row = row_by_case.get(review.case_id)
        if row is None:
            continue
        key = (review.case_id, review.reviewer_id)
        assert review.reviewer_id in column_by_reviewer
        assert key not in seen
        seen.add(key)
        if review.state is not ExecutionState.VALID:
            continue
        observation = review.observation
        if observation is None:
            raise AssertionError("VALID locked review lost its observation")
        column = column_by_reviewer[review.reviewer_id]
        observations[row, column] = OBSERVATION_ORDER.index(observation)
        valid_mask[row, column] = True
    assert seen == {
        (case_id, reviewer_id) for case_id in case_ids for reviewer_id in reviewer_ids
    }
    return observations, valid_mask


def _batch_slice(
    batch: BatchFusedPosterior,
    start: int,
    stop: int,
) -> BatchFusedPosterior:
    return BatchFusedPosterior(
        pass_probability=batch.pass_probability[start:stop],
        lower=batch.lower[start:stop],
        upper=batch.upper[start:stop],
        valid_reviewers=batch.valid_reviewers[start:stop],
        lineage_count=batch.lineage_count[start:stop],
        effective_sample_size=batch.effective_sample_size[start:stop],
    )


def _posterior_mapping(
    batch: BatchFusedPosterior,
    case_ids: Sequence[str],
) -> dict[str, FusedPosterior | None]:
    result: dict[str, FusedPosterior | None] = {}
    for index, case_id in enumerate(case_ids):
        if int(batch.valid_reviewers[index]) == 0:
            result[case_id] = None
            continue
        result[case_id] = FusedPosterior(
            pass_probability=float(batch.pass_probability[index]),
            lower=float(batch.lower[index]),
            upper=float(batch.upper[index]),
            valid_reviewers=int(batch.valid_reviewers[index]),
            lineage_count=int(batch.lineage_count[index]),
            effective_sample_size=float(batch.effective_sample_size[index]),
            samples=(),
        )
    return result


def _reference_actions(
    batch: BatchFusedPosterior,
    policy: tuple[float, float, int, int, float],
) -> np.ndarray:
    pass_threshold, fail_threshold, min_reviewers, min_lineages, min_ess = policy
    quorum = (
        (batch.valid_reviewers >= min_reviewers)
        & (batch.lineage_count >= min_lineages)
        & (batch.effective_sample_size >= min_ess)
    )
    actions = np.full(batch.pass_probability.size, _DEFER_ACTION, dtype=np.int8)
    actions[quorum & (batch.lower >= pass_threshold)] = _PASS_ACTION
    actions[quorum & (batch.upper <= fail_threshold)] = _FAIL_ACTION
    return actions


def _reference_majority(
    observations: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    pass_votes = np.sum(valid_mask & (observations == 0), axis=1)
    fail_votes = np.sum(valid_mask & (observations == 1), axis=1)
    actions = np.full(observations.shape[0], _DEFER_ACTION, dtype=np.int8)
    actions[pass_votes > fail_votes] = _PASS_ACTION
    actions[fail_votes > pass_votes] = _FAIL_ACTION
    return actions


def _reference_score(
    truth_is_pass: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> _Score:
    if (
        truth_is_pass.ndim != 1
        or actions.shape != truth_is_pass.shape
        or truth_is_pass.size == 0
    ):
        raise AssertionError("reference score requires aligned non-empty vectors")
    fail = ~truth_is_pass
    false_pass = (actions == _PASS_ACTION) & fail
    false_fail = (actions == _FAIL_ACTION) & truth_is_pass
    defer = actions == _DEFER_ACTION
    fail_cases = int(np.count_nonzero(fail))
    if fail_cases == 0:
        raise AssertionError("locked partition lost the FAIL class")
    loss_units = int(
        np.count_nonzero(defer)
        + 10 * np.count_nonzero(false_pass)
        + 2 * np.count_nonzero(false_fail)
    )
    if probabilities is None:
        brier_sum = float("nan")
        nll_sum = float("nan")
    else:
        if probabilities.shape != truth_is_pass.shape:
            raise AssertionError("reference probabilities must align with truth")
        outcomes = truth_is_pass.astype(float)
        brier_sum = float(np.square(probabilities - outcomes).sum())
        observed = np.where(truth_is_pass, probabilities, 1.0 - probabilities)
        clipped = np.clip(observed, _NLL_EPSILON, 1.0 - _NLL_EPSILON)
        nll_sum = float(-np.log(clipped).sum())
    return _Score(
        cases=int(truth_is_pass.size),
        fail_cases=fail_cases,
        decided=int(np.count_nonzero(actions != _DEFER_ACTION)),
        false_passes=int(np.count_nonzero(false_pass)),
        loss_units=loss_units,
        brier_sum=brier_sum,
        nll_sum=nll_sum,
    )


def _probability_totals(
    truth_is_pass: np.ndarray,
    probabilities: np.ndarray,
    selector: np.ndarray,
) -> _ProbabilityTotals:
    selected_truth = truth_is_pass[selector]
    selected_probability = probabilities[selector]
    if selected_truth.size == 0:
        raise AssertionError("locked probability slice is empty")
    outcomes = selected_truth.astype(float)
    observed = np.where(
        selected_truth,
        selected_probability,
        1.0 - selected_probability,
    )
    return _ProbabilityTotals(
        cases=int(selected_truth.size),
        brier_sum=float(np.square(selected_probability - outcomes).sum()),
        nll_sum=float(
            -np.log(np.clip(observed, _NLL_EPSILON, 1.0 - _NLL_EPSILON)).sum()
        ),
    )


def _reference_select_policy(
    truth_is_pass: np.ndarray,
    batch: BatchFusedPosterior,
) -> _ReferenceSelection:
    candidates = tuple(
        (policy, _reference_score(truth_is_pass, _reference_actions(batch, policy)))
        for policy in _POLICY_GRID
    )
    feasible = tuple(
        candidate
        for candidate in candidates
        if candidate[1].coverage >= _MINIMUM_COVERAGE
    )
    if feasible:
        winner = min(
            feasible,
            key=lambda candidate: (
                candidate[1].loss_units,
                candidate[1].false_pass_rate,
                -candidate[1].decided,
                candidate[0],
            ),
        )
        satisfied = True
    else:
        winner = min(
            candidates,
            key=lambda candidate: (
                -candidate[1].decided,
                candidate[1].loss_units,
                candidate[1].false_pass_rate,
                candidate[0],
            ),
        )
        satisfied = False
    return _ReferenceSelection(
        policy=winner[0],
        constraint_satisfied=satisfied,
        score=winner[1],
    )


def _assert_production_selection(
    truth_is_pass: np.ndarray,
    case_ids: Sequence[str],
    batch: BatchFusedPosterior,
    reference: _ReferenceSelection,
) -> None:
    truth_mapping = {
        case_id: Truth.PASS if bool(truth) else Truth.FAIL
        for case_id, truth in zip(case_ids, truth_is_pass, strict=True)
    }
    selection = select_decision_policy(
        truth_mapping,
        _posterior_mapping(batch, case_ids),
        {case_id: () for case_id in case_ids},
        costs=_COSTS,
        min_coverage=_MINIMUM_COVERAGE,
    )
    assert _canonical_policy(selection.policy) == reference.policy
    assert selection.constraint_satisfied is reference.constraint_satisfied
    assert selection.decision_loss == pytest.approx(reference.score.loss, abs=1e-12)
    assert selection.coverage == pytest.approx(reference.score.coverage, abs=1e-12)


def _identity_dependence(aware: DependenceModel) -> DependenceModel:
    return DependenceModel(
        reviewer_ids=aware.reviewer_ids,
        correlation=np.eye(len(aware.reviewer_ids), dtype=float),
        lineage_by_reviewer=aware.lineage_by_reviewer,
    )


def _reference_weights(
    dependence: DependenceModel,
    reviewer_ids: Sequence[str],
) -> dict[str, float]:
    subset = tuple(reviewer_ids)
    index_by_id = {
        reviewer_id: index for index, reviewer_id in enumerate(dependence.reviewer_ids)
    }
    weights: dict[str, float] = {}
    for reviewer_id in subset:
        correlation_sum = 0.0
        for other_id in subset:
            if other_id == reviewer_id:
                continue
            pair = tuple(sorted((reviewer_id, other_id)))
            correlation = dependence._weight_overrides.get(
                cast(PairKey, pair),
                float(
                    dependence.correlation[
                        index_by_id[reviewer_id],
                        index_by_id[other_id],
                    ]
                ),
            )
            correlation_sum += min(max(float(correlation), 0.0), 1.0)
        weights[reviewer_id] = 1.0 / (1.0 + correlation_sum)
    return weights


def _reference_posterior(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: Sequence[str],
    context: FusionContext,
    *,
    mode: str,
) -> tuple[np.ndarray, float, float] | None:
    valid_ids = tuple(
        reviewer_id
        for reviewer_id, valid in zip(reviewer_ids, valid_mask, strict=True)
        if valid
    )
    if not valid_ids:
        return None
    log_pass = np.full(context.draws, log(context.prior_pass), dtype=float)
    log_fail = np.full(context.draws, log(1.0 - context.prior_pass), dtype=float)
    tiny = np.finfo(np.float64).tiny
    used: set[str] = set()
    if mode == "pair":
        column_by_id = {
            reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
        }
        for pair, draws in context.pair_likelihood_draws.items():
            first, second = pair
            first_column = column_by_id[first]
            second_column = column_by_id[second]
            if valid_mask[first_column] and valid_mask[second_column]:
                first_code = int(observations[first_column])
                second_code = int(observations[second_column])
                log_pass += np.log(
                    np.maximum(draws[:, 0, first_code, second_code], tiny)
                )
                log_fail += np.log(
                    np.maximum(draws[:, 1, first_code, second_code], tiny)
                )
                used.update(pair)
        weights = {reviewer_id: 1.0 for reviewer_id in valid_ids}
    elif mode == "power":
        weights = _reference_weights(context.dependence, valid_ids)
    elif mode == "naive":
        weights = {reviewer_id: 1.0 for reviewer_id in valid_ids}
    else:
        raise AssertionError(f"unknown reference fusion mode: {mode}")

    column_by_id = {
        reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
    }
    for reviewer_id in valid_ids:
        if reviewer_id in used:
            continue
        code = int(observations[column_by_id[reviewer_id]])
        draws = context.likelihood_draws[reviewer_id]
        weight = weights[reviewer_id]
        log_pass += weight * np.log(np.maximum(draws[:, 0, code], tiny))
        log_fail += weight * np.log(np.maximum(draws[:, 1, code], tiny))
    maximum = np.maximum(log_pass, log_fail)
    pass_mass = np.exp(log_pass - maximum)
    fail_mass = np.exp(log_fail - maximum)
    samples = pass_mass / (pass_mass + fail_mass)
    lower_quantile = (1.0 - context.credible_mass) * 0.5
    lower, upper = np.quantile(
        samples,
        (lower_quantile, 1.0 - lower_quantile),
    )
    valid_codes = observations[valid_mask]
    if np.all(valid_codes == OBSERVATION_ORDER.index(Observation.ABSTAIN)):
        lower, upper = 0.0, 1.0
    return samples, float(lower), float(upper)


def _assert_batch_reference(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: tuple[str, ...],
    batches: Mapping[str, BatchFusedPosterior],
    contexts: Mapping[str, FusionContext],
) -> None:
    rows_by_evidence: dict[
        tuple[tuple[bool, ...], tuple[int, ...]],
        list[int],
    ] = {}
    for row_index, (row, pattern) in enumerate(
        zip(observations, valid_mask, strict=True)
    ):
        key = (
            tuple(bool(value) for value in pattern),
            tuple(int(value) for value in row[pattern]),
        )
        rows_by_evidence.setdefault(key, []).append(row_index)

    for row_indices in rows_by_evidence.values():
        row_index = row_indices[0]
        pattern = valid_mask[row_index]
        subset = tuple(
            reviewer_id
            for reviewer_id, valid in zip(reviewer_ids, pattern, strict=True)
            if valid
        )
        for mode in ("naive", "power", "pair"):
            batch = batches[mode]
            context = contexts[mode]
            reference = _reference_posterior(
                observations[row_index],
                pattern,
                reviewer_ids,
                context,
                mode=mode,
            )
            if reference is None:
                assert np.all(np.isnan(batch.pass_probability[row_indices]))
                assert np.all(batch.valid_reviewers[row_indices] == 0)
                assert np.all(batch.lineage_count[row_indices] == 0)
                assert np.all(batch.effective_sample_size[row_indices] == 0.0)
                continue
            samples, lower, upper = reference
            np.testing.assert_allclose(
                batch.pass_probability[row_indices],
                float(samples.mean()),
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                batch.lower[row_indices],
                lower,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                batch.upper[row_indices],
                upper,
                rtol=0.0,
                atol=1e-12,
            )
            assert np.all(batch.valid_reviewers[row_indices] == len(subset))
            expected_lineages = len(
                {context.lineage_by_reviewer[reviewer_id] for reviewer_id in subset}
            )
            assert np.all(batch.lineage_count[row_indices] == expected_lineages)
            if mode == "naive":
                expected_ess = float(len(subset))
            else:
                reference_weights = _reference_weights(context.dependence, subset)
                production_weights = context.dependence.weights_for(subset)
                assert dict(production_weights) == pytest.approx(reference_weights)
                expected_ess = min(
                    max(sum(reference_weights.values()), 1.0),
                    float(len(subset)),
                )
            np.testing.assert_allclose(
                batch.effective_sample_size[row_indices],
                expected_ess,
                rtol=0.0,
                atol=1e-12,
            )


def _truth_vector(
    panel: SimulatedPanel,
    case_ids: Sequence[str],
) -> np.ndarray:
    return np.array(
        [panel.truths[case_id] is Truth.PASS for case_id in case_ids],
        dtype=bool,
    )


def _reference_pair_counts(
    truth_is_pass: np.ndarray,
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: Sequence[str],
    pair: PairKey,
) -> np.ndarray:
    column_by_id = {
        reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
    }
    first_column = column_by_id[pair[0]]
    second_column = column_by_id[pair[1]]
    both = valid_mask[:, first_column] & valid_mask[:, second_column]
    truth_codes = np.where(truth_is_pass, 0, 1).astype(np.int64)
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    np.add.at(
        counts,
        (
            truth_codes[both],
            observations[both, first_column],
            observations[both, second_column],
        ),
        1,
    )
    return counts


def _build_run(
    scenario_name: str,
    scenario_index: int,
    fit_cases: int,
    fit_index: int,
    base_seed: int,
) -> _RunRecord:
    scenario = _locked_scenarios()[scenario_name]
    simulation_seed = base_seed + 100_000 * scenario_index + 10_000 * fit_index
    fusion_seed = 2_000_000 + simulation_seed
    calibration_panel, test_panel = simulate_experiment(
        scenario,
        n_calibration=fit_cases + _POLICY_CASES,
        n_test=_TEST_CASES,
        seed=simulation_seed,
    )
    fit_ids, policy_ids = _split_ids(calibration_panel, fit_cases)
    test_ids = tuple(sorted(test_panel.truths))
    assert len(test_ids) == _TEST_CASES
    assert set(fit_ids).isdisjoint(policy_ids)
    assert set(fit_ids).isdisjoint(test_ids)
    assert set(policy_ids).isdisjoint(test_ids)

    reviewers = tuple(spec.reviewer for spec in scenario.calibration.reviewers)
    fit_examples = _examples_for(calibration_panel, fit_ids)
    assert len(fit_examples) == fit_cases * len(reviewers)
    calibrations = fit_panel_calibrations(
        reviewers,
        fit_examples,
        prior_strength=_MARGINAL_PRIOR_STRENGTH,
    )
    aware_dependence = fit_dependence(
        reviewers,
        fit_examples,
        shrinkage=_DEPENDENCE_SHRINKAGE,
        min_overlap=_MINIMUM_OVERLAP,
        lineage_cap=_LINEAGE_CAP,
    )
    reviewer_ids = aware_dependence.reviewer_ids
    pair_key = _PAIR_KEYS[scenario_name]
    pair_calibration = fit_reviewer_pair_calibration(
        pair_key,
        fit_examples,
        reviewer_calibrations=calibrations,
        prior_strength=_PAIR_PRODUCT_PRIOR_STRENGTH,
        min_paired_per_truth=_MIN_PAIRED_PER_TRUTH,
    )

    fit_observations, fit_mask = _review_matrix(
        calibration_panel,
        fit_ids,
        reviewer_ids,
    )
    fit_truth = _truth_vector(calibration_panel, fit_ids)
    reference_counts = _reference_pair_counts(
        fit_truth,
        fit_observations,
        fit_mask,
        reviewer_ids,
        pair_key,
    )
    np.testing.assert_array_equal(
        pair_calibration.observed_counts,
        reference_counts,
    )
    joint_row_totals_array = reference_counts.sum(axis=(1, 2))
    assert np.all(joint_row_totals_array >= _MIN_PAIRED_PER_TRUTH)

    pair_context = build_fusion_context(
        calibrations,
        aware_dependence,
        prior_pass=scenario.calibration.prior_pass,
        draws=_POSTERIOR_DRAWS,
        credible_mass=_CREDIBLE_MASS,
        seed=fusion_seed,
        pair_calibrations={pair_key: pair_calibration},
    )
    power_context = FusionContext(
        likelihood_draws=pair_context.likelihood_draws,
        dependence=aware_dependence,
        lineage_by_reviewer=aware_dependence.lineage_by_reviewer,
        prior_pass=pair_context.prior_pass,
        credible_mass=pair_context.credible_mass,
        pair_likelihood_draws={},
    )
    naive_dependence = _identity_dependence(aware_dependence)
    naive_context = FusionContext(
        likelihood_draws=pair_context.likelihood_draws,
        dependence=naive_dependence,
        lineage_by_reviewer=naive_dependence.lineage_by_reviewer,
        prior_pass=pair_context.prior_pass,
        credible_mass=pair_context.credible_mass,
        pair_likelihood_draws={},
    )
    contexts = {
        "naive": naive_context,
        "power": power_context,
        "pair": pair_context,
    }
    assert tuple(pair_context.pair_likelihood_draws) == (pair_key,)
    assert not power_context.pair_likelihood_draws
    assert not naive_context.pair_likelihood_draws
    assert pair_context.prior_pass == scenario.calibration.prior_pass
    for reviewer_id in reviewer_ids:
        pair_bytes = pair_context.likelihood_draws[reviewer_id].tobytes()
        assert power_context.likelihood_draws[reviewer_id].tobytes() == pair_bytes
        assert naive_context.likelihood_draws[reviewer_id].tobytes() == pair_bytes
    np.testing.assert_array_equal(
        naive_dependence.correlation,
        np.eye(len(reviewer_ids), dtype=float),
    )

    policy_observations, policy_mask = _review_matrix(
        calibration_panel,
        policy_ids,
        reviewer_ids,
    )
    test_observations, test_mask = _review_matrix(
        test_panel,
        test_ids,
        reviewer_ids,
    )
    combined_observations = np.vstack((policy_observations, test_observations))
    combined_mask = np.vstack((policy_mask, test_mask))
    original_bytes = combined_observations.tobytes() + combined_mask.tobytes()
    combined_batches = {
        mode: fuse_review_matrix(
            combined_observations,
            combined_mask,
            reviewer_ids,
            context,
            chunk_size=_MATRIX_CHUNK_SIZE,
        )
        for mode, context in contexts.items()
    }
    assert combined_observations.tobytes() + combined_mask.tobytes() == original_bytes
    _assert_batch_reference(
        combined_observations,
        combined_mask,
        reviewer_ids,
        combined_batches,
        contexts,
    )

    policy_batches = {
        mode: _batch_slice(batch, 0, _POLICY_CASES)
        for mode, batch in combined_batches.items()
    }
    test_batches = {
        mode: _batch_slice(
            batch,
            _POLICY_CASES,
            _POLICY_CASES + _TEST_CASES,
        )
        for mode, batch in combined_batches.items()
    }
    policy_truth = _truth_vector(calibration_panel, policy_ids)
    test_truth = _truth_vector(test_panel, test_ids)
    assert {bool(value) for value in policy_truth} == {False, True}
    assert {bool(value) for value in test_truth} == {False, True}

    selections = {
        mode: _reference_select_policy(policy_truth, batch)
        for mode, batch in policy_batches.items()
    }
    for mode in ("naive", "power", "pair"):
        _assert_production_selection(
            policy_truth,
            policy_ids,
            policy_batches[mode],
            selections[mode],
        )
    scores = {
        mode: _reference_score(
            test_truth,
            _reference_actions(test_batches[mode], selections[mode].policy),
            test_batches[mode].pass_probability,
        )
        for mode in ("naive", "power", "pair")
    }
    majority_score = _reference_score(
        test_truth,
        _reference_majority(test_observations, test_mask),
    )

    missing: _MissingAudit | None = None
    if scenario_name == "missing_pair_member":
        column_by_id = {
            reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
        }
        first_column = column_by_id[pair_key[0]]
        second_column = column_by_id[pair_key[1]]
        exactly_one = test_mask[:, first_column] ^ test_mask[:, second_column]
        both = test_mask[:, first_column] & test_mask[:, second_column]
        assert np.any(exactly_one)
        assert np.any(both)
        pair_batch = test_batches["pair"]
        naive_batch = test_batches["naive"]
        power_batch = test_batches["power"]
        np.testing.assert_array_equal(
            pair_batch.valid_reviewers[exactly_one],
            naive_batch.valid_reviewers[exactly_one],
        )
        np.testing.assert_array_equal(
            pair_batch.lineage_count[exactly_one],
            naive_batch.lineage_count[exactly_one],
        )
        np.testing.assert_array_equal(
            pair_batch.effective_sample_size,
            power_batch.effective_sample_size,
        )
        missing = _MissingAudit(
            exactly_one_cases=int(np.count_nonzero(exactly_one)),
            total_cases=_TEST_CASES,
            max_probability_delta=float(
                np.max(
                    np.abs(
                        pair_batch.pass_probability[exactly_one]
                        - naive_batch.pass_probability[exactly_one]
                    )
                )
            ),
            max_lower_delta=float(
                np.max(
                    np.abs(
                        pair_batch.lower[exactly_one] - naive_batch.lower[exactly_one]
                    )
                )
            ),
            max_upper_delta=float(
                np.max(
                    np.abs(
                        pair_batch.upper[exactly_one] - naive_batch.upper[exactly_one]
                    )
                )
            ),
            both_pair=_probability_totals(
                test_truth,
                pair_batch.pass_probability,
                both,
            ),
            both_naive=_probability_totals(
                test_truth,
                naive_batch.pass_probability,
                both,
            ),
            both_power=_probability_totals(
                test_truth,
                power_batch.pass_probability,
                both,
            ),
        )

    gate_violations = sum(
        bool(gates)
        for gates in (
            *calibration_panel.gates.values(),
            *test_panel.gates.values(),
        )
    )
    return _RunRecord(
        scenario=scenario_name,
        fit_cases=fit_cases,
        base_seed=base_seed,
        simulation_seed=simulation_seed,
        pair=scores["pair"],
        naive=scores["naive"],
        power=scores["power"],
        majority=majority_score,
        policy_constraints=tuple(
            selections[mode].constraint_satisfied for mode in ("naive", "power", "pair")
        ),
        joint_row_totals=(
            int(joint_row_totals_array[0]),
            int(joint_row_totals_array[1]),
        ),
        missing=missing,
        gate_violations=gate_violations,
    )


def _sum_score_field(
    records: Sequence[_RunRecord],
    method: str,
    field: str,
) -> float:
    return sum(float(getattr(getattr(record, method), field)) for record in records)


def _micro_score(
    records: Sequence[_RunRecord],
    method: str,
    numerator: str,
) -> float:
    denominator = _sum_score_field(records, method, "cases")
    return _sum_score_field(records, method, numerator) / denominator


def _micro_loss(records: Sequence[_RunRecord], method: str) -> float:
    loss_units = _sum_score_field(records, method, "loss_units")
    cases = _sum_score_field(records, method, "cases")
    return loss_units / (10.0 * cases)


def _micro_false_pass(records: Sequence[_RunRecord], method: str) -> float:
    false_passes = _sum_score_field(records, method, "false_passes")
    fail_cases = _sum_score_field(records, method, "fail_cases")
    return false_passes / fail_cases


def _missing_totals(
    record: _RunRecord,
    method: str,
) -> _ProbabilityTotals:
    if record.missing is None:
        raise AssertionError("missing totals requested for a non-missing scenario")
    return cast(_ProbabilityTotals, getattr(record.missing, f"both_{method}"))


def _micro_missing_nll(
    records: Sequence[_RunRecord],
    method: str,
) -> float:
    totals = tuple(_missing_totals(record, method) for record in records)
    return sum(total.nll_sum for total in totals) / sum(total.cases for total in totals)


_Totals = Callable[[_RunRecord], tuple[float, float]]
_Contrast = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _bootstrap_interval(
    records: Sequence[_RunRecord],
    pair_totals: _Totals,
    baseline_totals: _Totals,
    contrast: _Contrast,
) -> tuple[float, float, float]:
    grouped: dict[tuple[str, int], list[_RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.scenario, record.fit_cases), []).append(record)
    ordered_groups = tuple(
        tuple(sorted(group, key=lambda record: record.base_seed))
        for _, group in sorted(grouped.items())
    )
    assert ordered_groups
    assert all(len(group) == len(_BASE_SEEDS) for group in ordered_groups)
    assert all(
        tuple(record.base_seed for record in group) == _BASE_SEEDS
        for group in ordered_groups
    )
    pair_sums = np.array(
        [[pair_totals(record)[0] for record in group] for group in ordered_groups],
        dtype=float,
    )
    pair_counts = np.array(
        [[pair_totals(record)[1] for record in group] for group in ordered_groups],
        dtype=float,
    )
    baseline_sums = np.array(
        [[baseline_totals(record)[0] for record in group] for group in ordered_groups],
        dtype=float,
    )
    baseline_counts = np.array(
        [[baseline_totals(record)[1] for record in group] for group in ordered_groups],
        dtype=float,
    )
    assert np.all(pair_counts > 0.0)
    assert np.all(baseline_counts > 0.0)
    point_pair = pair_sums.sum() / pair_counts.sum()
    point_baseline = baseline_sums.sum() / baseline_counts.sum()
    point = float(contrast(np.array(point_pair), np.array(point_baseline)))

    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        len(_BASE_SEEDS),
        size=(_BOOTSTRAP_DRAWS, len(ordered_groups), len(_BASE_SEEDS)),
    )
    target_shape = indices.shape
    sampled_pair_sums = np.take_along_axis(
        np.broadcast_to(pair_sums, target_shape),
        indices,
        axis=2,
    )
    sampled_pair_counts = np.take_along_axis(
        np.broadcast_to(pair_counts, target_shape),
        indices,
        axis=2,
    )
    sampled_baseline_sums = np.take_along_axis(
        np.broadcast_to(baseline_sums, target_shape),
        indices,
        axis=2,
    )
    sampled_baseline_counts = np.take_along_axis(
        np.broadcast_to(baseline_counts, target_shape),
        indices,
        axis=2,
    )
    pair_estimates = sampled_pair_sums.sum(axis=(1, 2)) / sampled_pair_counts.sum(
        axis=(1, 2)
    )
    baseline_estimates = sampled_baseline_sums.sum(
        axis=(1, 2)
    ) / sampled_baseline_counts.sum(axis=(1, 2))
    estimates = contrast(pair_estimates, baseline_estimates)
    lower, upper = np.quantile(estimates, (0.025, 0.975), method="linear")
    assert isfinite(point) and isfinite(float(lower)) and isfinite(float(upper))
    return point, float(lower), float(upper)


def _nll_totals(method: str) -> _Totals:
    return lambda record: (
        float(getattr(record, method).nll_sum),
        float(getattr(record, method).cases),
    )


def _loss_totals(method: str) -> _Totals:
    return lambda record: (
        float(getattr(record, method).loss_units),
        float(10 * getattr(record, method).cases),
    )


def _missing_nll_totals(method: str) -> _Totals:
    return lambda record: (
        float(_missing_totals(record, method).nll_sum),
        float(_missing_totals(record, method).cases),
    )


def _benefit(pair: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return baseline - pair


def _relative_degradation(
    pair: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    return (pair - baseline) / baseline


def _assert_locked_contract() -> None:
    scenarios = _locked_scenarios()
    assert _SCENARIO_NAMES == (
        "heterogeneous_pair",
        "low_prevalence_pair",
        "same_lineage_independent",
        "missing_pair_member",
    )
    assert _BASE_SEEDS == tuple(10_007 + 137 * index for index in range(8))
    assert (_FIT_CASES, _POLICY_CASES, _TEST_CASES) == ((160, 640), 400, 2_500)
    assert (
        _POSTERIOR_DRAWS,
        _CREDIBLE_MASS,
        _MATRIX_CHUNK_SIZE,
    ) == (256, 0.95, 4_096)
    assert (
        _MARGINAL_PRIOR_STRENGTH,
        _PAIR_PRODUCT_PRIOR_STRENGTH,
        _MIN_PAIRED_PER_TRUTH,
    ) == (1.5, 9.0, 30)
    assert (
        _DEPENDENCE_SHRINKAGE,
        _MINIMUM_OVERLAP,
        _LINEAGE_CAP,
    ) == (0.25, 10, 1.0)
    assert (_MINIMUM_COVERAGE, _NLL_EPSILON) == (0.50, 1e-15)
    assert (_BOOTSTRAP_DRAWS, _BOOTSTRAP_SEED) == (2_000, 20_260_917)
    assert (_COSTS.false_pass, _COSTS.false_fail, _COSTS.defer) == (1.0, 0.2, 0.1)
    assert OBSERVATION_ORDER == (
        Observation.PASS,
        Observation.FAIL,
        Observation.ABSTAIN,
    )
    assert tuple(_canonical_policy(policy) for policy in policy_candidates()) == (
        _POLICY_GRID
    )
    assert _PAIR_KEYS == {
        "heterogeneous_pair": ("pair-a", "pair-b"),
        "low_prevalence_pair": ("pair-x", "pair-y"),
        "same_lineage_independent": ("pair-a", "pair-b"),
        "missing_pair_member": ("pair-a", "pair-b"),
    }
    for scenario_name in _SCENARIO_NAMES:
        scenario = scenarios[scenario_name]
        expected_calibration, expected_test = _SCENARIO_SNAPSHOTS[scenario_name]
        assert scenario.name == scenario_name
        assert _phase_snapshot(scenario.calibration) == expected_calibration
        assert _phase_snapshot(scenario.test) == expected_test
    derived_seeds = {
        base_seed + 100_000 * scenario_index + 10_000 * fit_index
        for scenario_index, _ in enumerate(_SCENARIO_NAMES)
        for fit_index, _ in enumerate(_FIT_CASES)
        for base_seed in _BASE_SEEDS
    }
    assert len(derived_seeds) == 64
    assert derived_seeds.isdisjoint(range(20))


def test_locked_pair_scenario_snapshots_and_constants() -> None:
    _assert_locked_contract()


def _gate_payload_json(result: Mapping[str, object]) -> str:
    return json.dumps(dict(result), sort_keys=True, allow_nan=False)


def _write_gate_payload(
    pytestconfig: pytest.Config,
    result: Mapping[str, object],
) -> str:
    payload = _gate_payload_json(result)
    terminal = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line("PAIR_VALUE_GATE_RESULT " + payload)
    _GATE_AUDIT_STATE["payload_written"] = True
    return payload


def _freeze_unexpected_gate_failure(
    test: Callable[[pytest.Config], None],
) -> Callable[[pytest.Config], None]:
    @wraps(test)
    def wrapped(pytestconfig: pytest.Config) -> None:
        expected_runs = len(_SCENARIO_NAMES) * len(_FIT_CASES) * len(_BASE_SEEDS)
        _GATE_AUDIT_STATE.clear()
        _GATE_AUDIT_STATE.update(
            completed_runs=0,
            expected_runs=expected_runs,
            failed_run=None,
            payload_written=False,
        )
        try:
            test(pytestconfig)
        except Exception as error:
            if bool(_GATE_AUDIT_STATE.get("payload_written")):
                raise
            failure_result: dict[str, object] = {
                "schema_version": 1,
                "verdict": "FAIL",
                "completed_runs": int(_GATE_AUDIT_STATE["completed_runs"]),
                "expected_runs": int(_GATE_AUDIT_STATE["expected_runs"]),
                "failed_run": _GATE_AUDIT_STATE.get("failed_run"),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "gate_a": {
                    "verdict": "FAIL",
                    "failures": ["registered_judge_execution"],
                    "predicates": {"registered_judge_execution": False},
                },
                "gate_b": {
                    "verdict": "NOT_EVALUATED",
                    "failures": [],
                    "predicates": {},
                },
                "diagnostics": {},
            }
            payload = _write_gate_payload(pytestconfig, failure_result)
            raise AssertionError("PAIR_BLOCK_ADMISSION_FAILED: " + payload) from error

    return wrapped


@_freeze_unexpected_gate_failure
def test_locked_pair_value_gate(pytestconfig: pytest.Config) -> None:
    _assert_locked_contract()
    expected_runs = len(_SCENARIO_NAMES) * len(_FIT_CASES) * len(_BASE_SEEDS)
    records_buffer: list[_RunRecord] = []
    for scenario_index, scenario_name in enumerate(_SCENARIO_NAMES):
        for fit_index, fit_cases in enumerate(_FIT_CASES):
            for base_seed in _BASE_SEEDS:
                _GATE_AUDIT_STATE["failed_run"] = {
                    "scenario": scenario_name,
                    "fit_cases": fit_cases,
                    "base_seed": base_seed,
                    "simulation_seed": (
                        base_seed + 100_000 * scenario_index + 10_000 * fit_index
                    ),
                }
                records_buffer.append(
                    _build_run(
                        scenario_name,
                        scenario_index,
                        fit_cases,
                        fit_index,
                        base_seed,
                    )
                )
                _GATE_AUDIT_STATE["completed_runs"] = len(records_buffer)
    _GATE_AUDIT_STATE["failed_run"] = None

    records = tuple(records_buffer)

    assert len(records) == expected_runs == 64
    assert (
        len(
            {
                (record.scenario, record.fit_cases, record.base_seed)
                for record in records
            }
        )
        == 64
    )
    assert len({record.simulation_seed for record in records}) == 64

    diagnostics: dict[str, float | int | bool] = {}
    gate_a: dict[str, bool] = {}
    gate_b: dict[str, bool] = {}

    finite_invariants = True
    for record in records:
        finite_invariants &= record.joint_row_totals[0] >= _MIN_PAIRED_PER_TRUTH
        finite_invariants &= record.joint_row_totals[1] >= _MIN_PAIRED_PER_TRUTH
        finite_invariants &= record.gate_violations == 0
        finite_invariants &= (record.missing is not None) is (
            record.scenario == "missing_pair_member"
        )
        for method in ("pair", "naive", "power"):
            score = cast(_Score, getattr(record, method))
            finite_invariants &= score.cases == _TEST_CASES
            finite_invariants &= 0 < score.fail_cases < score.cases
            finite_invariants &= 0 <= score.decided <= score.cases
            finite_invariants &= 0 <= score.false_passes <= score.fail_cases
            finite_invariants &= 0 <= score.loss_units <= 10 * score.cases
            finite_invariants &= all(
                isfinite(value) for value in (score.loss, score.brier, score.nll)
            )
            finite_invariants &= score.brier >= 0.0 and score.nll >= 0.0
        finite_invariants &= isfinite(record.majority.loss)
    gate_a["finite_and_active_invariants"] = finite_invariants
    diagnostics["joint_row_total_min"] = min(
        min(record.joint_row_totals) for record in records
    )
    diagnostics["joint_row_total_max"] = max(
        max(record.joint_row_totals) for record in records
    )
    diagnostics["total_gate_violations"] = sum(
        record.gate_violations for record in records
    )

    correlated = tuple(
        record
        for record in records
        if record.scenario
        in {
            "heterogeneous_pair",
            "low_prevalence_pair",
            "missing_pair_member",
        }
    )
    correlated_pair_nll = _micro_score(correlated, "pair", "nll_sum")
    correlated_pair_brier = _micro_score(correlated, "pair", "brier_sum")
    diagnostics["correlated_pair_nll"] = correlated_pair_nll
    diagnostics["correlated_pair_brier"] = correlated_pair_brier
    for baseline in ("naive", "power"):
        baseline_nll = _micro_score(correlated, baseline, "nll_sum")
        baseline_brier = _micro_score(correlated, baseline, "brier_sum")
        improvement = (baseline_nll - correlated_pair_nll) / baseline_nll
        interval = _bootstrap_interval(
            correlated,
            _nll_totals("pair"),
            _nll_totals(baseline),
            _benefit,
        )
        diagnostics[f"correlated_{baseline}_nll"] = baseline_nll
        diagnostics[f"correlated_{baseline}_brier"] = baseline_brier
        diagnostics[f"correlated_vs_{baseline}_relative_improvement"] = improvement
        diagnostics[f"correlated_vs_{baseline}_bootstrap_point"] = interval[0]
        diagnostics[f"correlated_vs_{baseline}_bootstrap_lower"] = interval[1]
        diagnostics[f"correlated_vs_{baseline}_bootstrap_upper"] = interval[2]
        gate_a[f"correlated_nll_improves_3pct_vs_{baseline}"] = improvement >= 0.03
        gate_a[f"correlated_nll_interval_positive_vs_{baseline}"] = interval[1] > 0.0
        gate_a[f"correlated_brier_not_worse_vs_{baseline}"] = (
            correlated_pair_brier <= baseline_brier
        )

    low_fit = tuple(record for record in correlated if record.fit_cases == 160)
    low_pair_nll = _micro_score(low_fit, "pair", "nll_sum")
    diagnostics["low_fit_pair_nll"] = low_pair_nll
    for baseline in ("naive", "power"):
        baseline_nll = _micro_score(low_fit, baseline, "nll_sum")
        improvement = (baseline_nll - low_pair_nll) / baseline_nll
        interval = _bootstrap_interval(
            low_fit,
            _nll_totals("pair"),
            _nll_totals(baseline),
            _benefit,
        )
        diagnostics[f"low_fit_vs_{baseline}_relative_improvement"] = improvement
        diagnostics[f"low_fit_{baseline}_nll"] = baseline_nll
        diagnostics[f"low_fit_vs_{baseline}_bootstrap_point"] = interval[0]
        diagnostics[f"low_fit_vs_{baseline}_bootstrap_lower"] = interval[1]
        diagnostics[f"low_fit_vs_{baseline}_bootstrap_upper"] = interval[2]
        gate_a[f"low_fit_nll_improves_1pct_vs_{baseline}"] = improvement >= 0.01
        gate_a[f"low_fit_nll_interval_positive_vs_{baseline}"] = interval[1] > 0.0

    for scenario_name in _SCENARIO_NAMES:
        for fit_cases in _FIT_CASES:
            sliced = tuple(
                record
                for record in records
                if record.scenario == scenario_name and record.fit_cases == fit_cases
            )
            pair_nll = _micro_score(sliced, "pair", "nll_sum")
            pair_brier = _micro_score(sliced, "pair", "brier_sum")
            for baseline in ("naive", "power"):
                baseline_nll = _micro_score(sliced, baseline, "nll_sum")
                baseline_brier = _micro_score(sliced, baseline, "brier_sum")
                prefix = f"{scenario_name}_{fit_cases}_vs_{baseline}"
                diagnostics[f"{prefix}_pair_nll"] = pair_nll
                diagnostics[f"{prefix}_baseline_nll"] = baseline_nll
                diagnostics[f"{prefix}_pair_brier"] = pair_brier
                diagnostics[f"{prefix}_baseline_brier"] = baseline_brier
                gate_a[f"{prefix}_nll_regression"] = pair_nll <= 1.01 * baseline_nll
                gate_a[f"{prefix}_brier_regression"] = (
                    pair_brier <= 1.01 * baseline_brier
                )

    independent = tuple(
        record for record in records if record.scenario == "same_lineage_independent"
    )
    independent_pair_nll = _micro_score(independent, "pair", "nll_sum")
    independent_pair_brier = _micro_score(independent, "pair", "brier_sum")
    diagnostics["independent_pair_nll"] = independent_pair_nll
    diagnostics["independent_pair_brier"] = independent_pair_brier
    for baseline in ("naive", "power"):
        baseline_nll = _micro_score(independent, baseline, "nll_sum")
        baseline_brier = _micro_score(independent, baseline, "brier_sum")
        nll_degradation = (independent_pair_nll - baseline_nll) / baseline_nll
        brier_degradation = (independent_pair_brier - baseline_brier) / baseline_brier
        interval = _bootstrap_interval(
            independent,
            _nll_totals("pair"),
            _nll_totals(baseline),
            _relative_degradation,
        )
        diagnostics[f"independent_vs_{baseline}_nll_degradation"] = nll_degradation
        diagnostics[f"independent_vs_{baseline}_brier_degradation"] = brier_degradation
        diagnostics[f"independent_{baseline}_nll"] = baseline_nll
        diagnostics[f"independent_{baseline}_brier"] = baseline_brier
        diagnostics[f"independent_vs_{baseline}_bootstrap_point"] = interval[0]
        diagnostics[f"independent_vs_{baseline}_bootstrap_lower"] = interval[1]
        diagnostics[f"independent_vs_{baseline}_bootstrap_upper"] = interval[2]
        gate_a[f"independent_nll_degradation_vs_{baseline}"] = nll_degradation <= 0.01
        gate_a[f"independent_brier_degradation_vs_{baseline}"] = (
            brier_degradation <= 0.01
        )
        gate_a[f"independent_interval_upper_vs_{baseline}"] = interval[2] <= 0.01

    missing_records = tuple(
        record for record in records if record.scenario == "missing_pair_member"
    )
    gate_a["missing_fraction_per_run"] = all(
        record.missing is not None
        and 0.44
        <= record.missing.exactly_one_cases / record.missing.total_cases
        <= 0.56
        for record in missing_records
    )
    gate_a["missing_probability_fallback"] = all(
        record.missing is not None and record.missing.max_probability_delta <= 1e-12
        for record in missing_records
    )
    gate_a["missing_lower_fallback"] = all(
        record.missing is not None and record.missing.max_lower_delta <= 1e-12
        for record in missing_records
    )
    gate_a["missing_upper_fallback"] = all(
        record.missing is not None and record.missing.max_upper_delta <= 1e-12
        for record in missing_records
    )
    missing_fractions = tuple(
        cast(_MissingAudit, record.missing).exactly_one_cases
        / cast(_MissingAudit, record.missing).total_cases
        for record in missing_records
    )
    diagnostics["missing_exactly_one_fraction_min"] = min(missing_fractions)
    diagnostics["missing_exactly_one_fraction_max"] = max(missing_fractions)
    diagnostics["missing_probability_delta_max"] = max(
        cast(_MissingAudit, record.missing).max_probability_delta
        for record in missing_records
    )
    diagnostics["missing_lower_delta_max"] = max(
        cast(_MissingAudit, record.missing).max_lower_delta
        for record in missing_records
    )
    diagnostics["missing_upper_delta_max"] = max(
        cast(_MissingAudit, record.missing).max_upper_delta
        for record in missing_records
    )
    pair_both_nll = _micro_missing_nll(missing_records, "pair")
    diagnostics["missing_both_pair_nll"] = pair_both_nll
    for baseline in ("naive", "power"):
        baseline_nll = _micro_missing_nll(missing_records, baseline)
        improvement = (baseline_nll - pair_both_nll) / baseline_nll
        interval = _bootstrap_interval(
            missing_records,
            _missing_nll_totals("pair"),
            _missing_nll_totals(baseline),
            _benefit,
        )
        diagnostics[f"missing_both_vs_{baseline}_relative_improvement"] = improvement
        diagnostics[f"missing_both_{baseline}_nll"] = baseline_nll
        diagnostics[f"missing_both_vs_{baseline}_bootstrap_point"] = interval[0]
        diagnostics[f"missing_both_vs_{baseline}_bootstrap_lower"] = interval[1]
        diagnostics[f"missing_both_vs_{baseline}_bootstrap_upper"] = interval[2]
        gate_a[f"missing_both_improves_2pct_vs_{baseline}"] = improvement >= 0.02
        gate_a[f"missing_both_interval_positive_vs_{baseline}"] = interval[1] > 0.0

    pair_loss = _micro_loss(records, "pair")
    majority_loss = _micro_loss(records, "majority")
    naive_loss = _micro_loss(records, "naive")
    power_loss = _micro_loss(records, "power")
    pair_coverage = _micro_score(records, "pair", "decided")
    pair_false_pass = _micro_false_pass(records, "pair")
    majority_false_pass = _micro_false_pass(records, "majority")
    loss_interval = _bootstrap_interval(
        records,
        _loss_totals("pair"),
        _loss_totals("majority"),
        _benefit,
    )
    diagnostics.update(
        {
            "pooled_pair_loss": pair_loss,
            "pooled_majority_loss": majority_loss,
            "pooled_naive_loss": naive_loss,
            "pooled_power_loss": power_loss,
            "pooled_pair_coverage": pair_coverage,
            "pooled_pair_false_pass": pair_false_pass,
            "pooled_majority_false_pass": majority_false_pass,
            "decision_benefit_bootstrap_point": loss_interval[0],
            "decision_benefit_bootstrap_lower": loss_interval[1],
            "decision_benefit_bootstrap_upper": loss_interval[2],
        }
    )
    gate_b["pair_loss_beats_majority_10pct"] = pair_loss <= 0.90 * majority_loss
    gate_b["pair_majority_interval_positive"] = loss_interval[1] > 0.0
    gate_b["pair_loss_not_worse_than_naive"] = pair_loss <= naive_loss
    gate_b["pair_loss_not_worse_than_power"] = pair_loss <= power_loss
    for scenario_name in _SCENARIO_NAMES:
        for fit_cases in _FIT_CASES:
            sliced = tuple(
                record
                for record in records
                if record.scenario == scenario_name and record.fit_cases == fit_cases
            )
            sliced_pair_loss = _micro_loss(sliced, "pair")
            for baseline in ("majority", "power", "naive"):
                baseline_loss = _micro_loss(sliced, baseline)
                slice_prefix = f"{scenario_name}_{fit_cases}_loss_vs_{baseline}"
                diagnostics[f"{slice_prefix}_pair"] = sliced_pair_loss
                diagnostics[f"{slice_prefix}_baseline"] = baseline_loss
                gate_b[f"{scenario_name}_{fit_cases}_loss_vs_{baseline}"] = (
                    sliced_pair_loss <= baseline_loss + 0.01
                )
    gate_b["pair_coverage"] = pair_coverage >= _MINIMUM_COVERAGE
    gate_b["pair_false_pass_boundary"] = pair_false_pass <= majority_false_pass + 0.02
    gate_b["all_policy_constraints"] = all(
        all(record.policy_constraints) for record in records
    )
    diagnostics["satisfied_policy_constraints"] = sum(
        sum(record.policy_constraints) for record in records
    )
    diagnostics["expected_policy_constraints"] = len(records) * 3
    gate_b["zero_gate_violations"] = (
        sum(record.gate_violations for record in records) == 0
    )
    gate_b["finite_vertical_metrics"] = all(
        isfinite(value)
        for value in (
            pair_loss,
            majority_loss,
            naive_loss,
            power_loss,
            pair_coverage,
            pair_false_pass,
            majority_false_pass,
            loss_interval[0],
            loss_interval[1],
            loss_interval[2],
        )
    )

    gate_a_failures = tuple(name for name, passed in gate_a.items() if not passed)
    gate_b_failures = tuple(name for name, passed in gate_b.items() if not passed)
    verdict = "PASS" if not gate_a_failures and not gate_b_failures else "FAIL"
    result: dict[str, object] = {
        "schema_version": 1,
        "verdict": verdict,
        "completed_runs": len(records),
        "expected_runs": expected_runs,
        "failed_run": None,
        "error": None,
        "gate_a": {
            "verdict": "PASS" if not gate_a_failures else "FAIL",
            "failures": gate_a_failures,
            "predicates": gate_a,
        },
        "gate_b": {
            "verdict": "PASS" if not gate_b_failures else "FAIL",
            "failures": gate_b_failures,
            "predicates": gate_b,
        },
        "diagnostics": diagnostics,
    }
    payload = _write_gate_payload(pytestconfig, result)

    failure_labels = []
    if gate_a_failures:
        failure_labels.append("PAIR_BLOCK_ADMISSION_FAILED")
    if gate_b_failures:
        failure_labels.append("PAIR_CORE_CLOSURE_FAILED")
    assert not failure_labels, " + ".join(failure_labels) + ": " + payload
