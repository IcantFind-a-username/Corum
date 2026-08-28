from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import cast

import numpy as np
import pytest

from corum.baselines import DecisionCosts, majority_decision
from corum.calibration import (
    OBSERVATION_ORDER,
    fit_panel_calibrations,
)
from corum.decision import DecisionPolicy, decide
from corum.dependence import DependenceModel, fit_dependence
from corum.fusion import (
    BatchFusedPosterior,
    FusionContext,
    build_fusion_context,
    fuse_review_matrix,
)
from corum.metrics import (
    evaluate_decisions,
    policy_candidates,
    select_decision_policy,
    stratified_paired_bootstrap,
)
from corum.models import (
    Action,
    CalibrationExample,
    ExecutionState,
    FusedPosterior,
    Observation,
    Review,
    Truth,
)
from corum.simulation import (
    ScenarioPhase,
    SimulatedPanel,
    builtin_scenarios,
    simulate_experiment,
)

_SCENARIO_NAMES = ("independent", "clone_pair", "majority_trap")
_SEEDS = tuple(range(20))
_CALIBRATION_CASES = 2_000
_FIT_CASES = 1_600
_POLICY_CASES = 400
_TEST_CASES = 5_000
_PRIOR_STRENGTH = 1.5
_DEPENDENCE_SHRINKAGE = 0.25
_MINIMUM_OVERLAP = 10
_LINEAGE_CAP = 1.0
_POSTERIOR_DRAWS = 512
_CREDIBLE_MASS = 0.95
_MATRIX_CHUNK_SIZE = 4_096
_PRIOR_PASS = 0.80
_MINIMUM_COVERAGE = 0.50
_BOOTSTRAP_DRAWS = 2_000
_BOOTSTRAP_SEED = 20_260_828
_NLL_EPSILON = 1e-15
_COSTS = DecisionCosts()

_POLICY_GRID = tuple(
    (pass_threshold, fail_threshold, 2, 2, minimum_ess)
    for pass_threshold in (0.80, 0.90, 0.95)
    for fail_threshold in (0.05, 0.10, 0.20)
    for minimum_ess in (1.0, 1.5)
)

_INDEPENDENT_PHASE_SNAPSHOT = (
    (
        (
            "r1",
            "simulated",
            "general",
            "independent-1",
            1.0,
            (
                (0.82, 0.13000000000000006, 0.05),
                (0.13000000000000006, 0.82, 0.05),
            ),
            0.0,
            0.0,
        ),
        (
            "r2",
            "simulated",
            "general",
            "independent-2",
            1.0,
            ((0.76, 0.19, 0.05), (0.19, 0.76, 0.05)),
            0.0,
            0.0,
        ),
        (
            "r3",
            "simulated",
            "general",
            "independent-3",
            1.0,
            (
                (0.7, 0.25000000000000006, 0.05),
                (0.25000000000000006, 0.7, 0.05),
            ),
            0.0,
            0.0,
        ),
    ),
    0.8,
    (),
    0.0,
    0.0,
    None,
)

_CLONE_PHASE_SNAPSHOT = (
    (
        (
            "clone-a",
            "simulated",
            "general",
            "clone",
            1.0,
            ((0.76, 0.19, 0.05), (0.19, 0.76, 0.05)),
            0.0,
            0.0,
        ),
        (
            "clone-b",
            "simulated",
            "general",
            "clone",
            1.0,
            ((0.76, 0.19, 0.05), (0.19, 0.76, 0.05)),
            0.0,
            0.0,
        ),
        (
            "strong",
            "simulated",
            "general",
            "independent",
            1.0,
            (
                (0.84, 0.11000000000000003, 0.05),
                (0.11000000000000003, 0.84, 0.05),
            ),
            0.0,
            0.0,
        ),
    ),
    0.8,
    (("clone", 0.85),),
    0.0,
    0.0,
    None,
)

_MAJORITY_TRAP_PHASE_SNAPSHOT = (
    (
        (
            "weak-a",
            "simulated",
            "general",
            "weak-clones",
            1.0,
            ((0.62, 0.33, 0.05), (0.33, 0.62, 0.05)),
            0.0,
            0.0,
        ),
        (
            "weak-b",
            "simulated",
            "general",
            "weak-clones",
            1.0,
            ((0.63, 0.32, 0.05), (0.32, 0.63, 0.05)),
            0.0,
            0.0,
        ),
        (
            "strong",
            "simulated",
            "general",
            "strong",
            1.0,
            (
                (0.86, 0.09000000000000001, 0.05),
                (0.09000000000000001, 0.86, 0.05),
            ),
            0.0,
            0.0,
        ),
    ),
    0.8,
    (("weak-clones", 0.82),),
    0.0,
    0.0,
    None,
)

_SCENARIO_SNAPSHOTS = {
    "independent": (
        _INDEPENDENT_PHASE_SNAPSHOT,
        _INDEPENDENT_PHASE_SNAPSHOT,
    ),
    "clone_pair": (_CLONE_PHASE_SNAPSHOT, _CLONE_PHASE_SNAPSHOT),
    "majority_trap": (
        _MAJORITY_TRAP_PHASE_SNAPSHOT,
        _MAJORITY_TRAP_PHASE_SNAPSHOT,
    ),
}


@dataclass(frozen=True, slots=True)
class _ReferenceScore:
    cases: int
    fail_cases: int
    decided: int
    false_passes: int
    loss_sum: float
    brier_sum: float
    nll_sum: float

    @property
    def loss(self) -> float:
        return self.loss_sum / self.cases

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
class _ReferenceSelection:
    policy: tuple[float, float, int, int, float]
    constraint_satisfied: bool
    score: _ReferenceScore


@dataclass(frozen=True, slots=True)
class _RunRecord:
    scenario: str
    seed: int
    corum: _ReferenceScore
    majority: _ReferenceScore
    naive: _ReferenceScore
    policy_constraint_satisfied: bool
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


def _reference_majority(reviews: Sequence[Review]) -> Action:
    pass_votes = sum(
        review.state is ExecutionState.VALID
        and review.observation is Observation.PASS
        for review in reviews
    )
    fail_votes = sum(
        review.state is ExecutionState.VALID
        and review.observation is Observation.FAIL
        for review in reviews
    )
    if pass_votes > fail_votes:
        return Action.PASS
    if fail_votes > pass_votes:
        return Action.FAIL
    return Action.DEFER


def _reference_score(
    truths: Sequence[Truth],
    actions: Sequence[Action],
    probabilities: Sequence[float] | None = None,
) -> _ReferenceScore:
    if not truths or len(actions) != len(truths):
        raise AssertionError("reference score requires aligned non-empty rows")
    if probabilities is not None and len(probabilities) != len(truths):
        raise AssertionError("reference probabilities must align with truths")

    fail_cases = 0
    decided = 0
    false_passes = 0
    loss_sum = 0.0
    brier_sum = 0.0
    nll_sum = 0.0
    for index, (truth, action) in enumerate(zip(truths, actions, strict=True)):
        fail_cases += int(truth is Truth.FAIL)
        decided += int(action is not Action.DEFER)
        if action is Action.DEFER:
            loss_sum += 0.1
        elif action is Action.PASS and truth is Truth.FAIL:
            false_passes += 1
            loss_sum += 1.0
        elif action is Action.FAIL and truth is Truth.PASS:
            loss_sum += 0.2

        if probabilities is not None:
            probability = probabilities[index]
            outcome = 1.0 if truth is Truth.PASS else 0.0
            brier_sum += (probability - outcome) ** 2
            observed_probability = (
                probability if truth is Truth.PASS else 1.0 - probability
            )
            clipped = min(
                max(observed_probability, _NLL_EPSILON),
                1.0 - _NLL_EPSILON,
            )
            nll_sum -= log(clipped)

    if fail_cases == 0:
        raise AssertionError("locked reference partition lost the FAIL class")
    if probabilities is None:
        brier_sum = float("nan")
        nll_sum = float("nan")
    return _ReferenceScore(
        cases=len(truths),
        fail_cases=fail_cases,
        decided=decided,
        false_passes=false_passes,
        loss_sum=loss_sum,
        brier_sum=brier_sum,
        nll_sum=nll_sum,
    )


def _reference_actions(
    batch: BatchFusedPosterior,
    policy: tuple[float, float, int, int, float],
) -> tuple[Action, ...]:
    pass_threshold, fail_threshold, min_reviewers, min_lineages, min_ess = policy
    actions: list[Action] = []
    for index in range(batch.pass_probability.size):
        if (
            int(batch.valid_reviewers[index]) < min_reviewers
            or int(batch.lineage_count[index]) < min_lineages
            or float(batch.effective_sample_size[index]) < min_ess
        ):
            actions.append(Action.DEFER)
        elif float(batch.lower[index]) >= pass_threshold:
            actions.append(Action.PASS)
        elif float(batch.upper[index]) <= fail_threshold:
            actions.append(Action.FAIL)
        else:
            actions.append(Action.DEFER)
    return tuple(actions)


def _reference_select_policy(
    truths: Sequence[Truth],
    batch: BatchFusedPosterior,
) -> _ReferenceSelection:
    candidates: list[
        tuple[
            tuple[float, float, int, int, float],
            _ReferenceScore,
        ]
    ] = []
    for policy in _POLICY_GRID:
        actions = _reference_actions(batch, policy)
        candidates.append((policy, _reference_score(truths, actions)))

    feasible = [
        candidate
        for candidate in candidates
        if candidate[1].coverage >= _MINIMUM_COVERAGE
    ]
    if feasible:
        winner = min(
            feasible,
            key=lambda candidate: (
                candidate[1].loss,
                candidate[1].false_pass_rate,
                -candidate[1].coverage,
                candidate[0],
            ),
        )
        satisfied = True
    else:
        winner = min(
            candidates,
            key=lambda candidate: (
                -candidate[1].coverage,
                candidate[1].loss,
                candidate[1].false_pass_rate,
                candidate[0],
            ),
        )
        satisfied = False
    return _ReferenceSelection(winner[0], satisfied, winner[1])


def _split_calibration_cases(
    panel: SimulatedPanel,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(panel.truths))
    assert len(ordered) == _CALIBRATION_CASES
    fit_ids = ordered[:_FIT_CASES]
    policy_ids = ordered[_FIT_CASES:]
    assert len(fit_ids) == _FIT_CASES
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
        (case_id, reviewer_id)
        for case_id in case_ids
        for reviewer_id in reviewer_ids
    }
    return observations, valid_mask


def _reviews_by_case(
    panel: SimulatedPanel,
    case_ids: Sequence[str],
) -> dict[str, tuple[Review, ...]]:
    selected = set(case_ids)
    grouped: dict[str, list[Review]] = {case_id: [] for case_id in case_ids}
    for review in panel.reviews:
        if review.case_id in selected:
            grouped[review.case_id].append(review)
    return {case_id: tuple(grouped[case_id]) for case_id in case_ids}


def _posterior_at(
    batch: BatchFusedPosterior,
    index: int,
) -> FusedPosterior | None:
    if int(batch.valid_reviewers[index]) == 0:
        return None
    return FusedPosterior(
        pass_probability=float(batch.pass_probability[index]),
        lower=float(batch.lower[index]),
        upper=float(batch.upper[index]),
        valid_reviewers=int(batch.valid_reviewers[index]),
        lineage_count=int(batch.lineage_count[index]),
        effective_sample_size=float(batch.effective_sample_size[index]),
        samples=(),
    )


def _posterior_mapping(
    batch: BatchFusedPosterior,
    case_ids: Sequence[str],
) -> dict[str, FusedPosterior | None]:
    return {
        case_id: _posterior_at(batch, index)
        for index, case_id in enumerate(case_ids)
    }


def _assert_production_score(
    production: Mapping[str, float],
    reference: _ReferenceScore,
    *,
    probabilities: bool,
) -> None:
    assert production["decision_loss"] == pytest.approx(reference.loss, abs=1e-12)
    assert production["coverage"] == pytest.approx(reference.coverage, abs=1e-12)
    assert production["false_pass_rate"] == pytest.approx(
        reference.false_pass_rate,
        abs=1e-12,
    )
    if probabilities:
        assert production["brier"] == pytest.approx(reference.brier, abs=1e-12)
        assert production["log_loss"] == pytest.approx(reference.nll, abs=1e-12)


def _reference_bootstrap(
    rows: Sequence[Mapping[str, object]],
) -> tuple[float, float, float]:
    point = sum(
        float(cast(float, row["benefit"])) for row in rows
    ) / len(rows)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    estimates = np.empty(_BOOTSTRAP_DRAWS, dtype=float)
    for draw_index in range(_BOOTSTRAP_DRAWS):
        sampled: list[Mapping[str, object]] = []
        for group in grouped.values():
            indices = generator.integers(0, len(group), size=len(group))
            sampled.extend(group[int(index)] for index in indices)
        estimates[draw_index] = sum(
            float(cast(float, row["benefit"])) for row in sampled
        ) / len(sampled)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return point, float(lower), float(upper)


def _bootstrap_metric(rows: Sequence[Mapping[str, object]]) -> float:
    return sum(
        float(cast(float, row["benefit"])) for row in rows
    ) / len(rows)


def _sum_score_field(
    records: Sequence[_RunRecord],
    score_name: str,
    field: str,
) -> float:
    return sum(
        float(getattr(getattr(record, score_name), field))
        for record in records
    )


def _micro_mean(
    records: Sequence[_RunRecord],
    score_name: str,
    numerator: str,
) -> float:
    total_cases = _sum_score_field(records, score_name, "cases")
    return _sum_score_field(records, score_name, numerator) / total_cases


def _micro_false_pass_rate(
    records: Sequence[_RunRecord],
    score_name: str,
) -> float:
    numerator = _sum_score_field(records, score_name, "false_passes")
    denominator = _sum_score_field(records, score_name, "fail_cases")
    return numerator / denominator


def _identity_dependence(aware: DependenceModel) -> DependenceModel:
    return DependenceModel(
        reviewer_ids=aware.reviewer_ids,
        correlation=np.eye(len(aware.reviewer_ids), dtype=float),
        lineage_by_reviewer=aware.lineage_by_reviewer,
    )


def _build_run(
    scenario_name: str,
    scenario_index: int,
    seed: int,
) -> _RunRecord:
    scenario = builtin_scenarios()[scenario_name]
    calibration_panel, test_panel = simulate_experiment(
        scenario,
        n_calibration=_CALIBRATION_CASES,
        n_test=_TEST_CASES,
        seed=seed,
    )
    fit_ids, policy_ids = _split_calibration_cases(calibration_panel)
    test_ids = tuple(sorted(test_panel.truths))
    assert len(test_ids) == _TEST_CASES
    assert set(fit_ids).isdisjoint(policy_ids)
    assert set(fit_ids).isdisjoint(test_ids)
    assert set(policy_ids).isdisjoint(test_ids)

    reviewers = tuple(
        spec.reviewer for spec in scenario.calibration.reviewers
    )
    fit_examples = _examples_for(calibration_panel, fit_ids)
    assert len(fit_examples) == len(fit_ids) * len(reviewers)
    assert {
        (example.review.case_id, example.review.reviewer_id)
        for example in fit_examples
    } == {
        (case_id, reviewer.reviewer_id)
        for case_id in fit_ids
        for reviewer in reviewers
    }
    calibrations = fit_panel_calibrations(
        reviewers,
        fit_examples,
        prior_strength=_PRIOR_STRENGTH,
    )
    aware_dependence = fit_dependence(
        reviewers,
        fit_examples,
        shrinkage=_DEPENDENCE_SHRINKAGE,
        min_overlap=_MINIMUM_OVERLAP,
        lineage_cap=_LINEAGE_CAP,
    )
    fusion_seed = 10_000 + 100 * scenario_index + seed
    aware_context = build_fusion_context(
        calibrations,
        aware_dependence,
        prior_pass=_PRIOR_PASS,
        draws=_POSTERIOR_DRAWS,
        credible_mass=_CREDIBLE_MASS,
        seed=fusion_seed,
    )
    naive_dependence = _identity_dependence(aware_dependence)
    naive_context = FusionContext(
        likelihood_draws=aware_context.likelihood_draws,
        dependence=naive_dependence,
        lineage_by_reviewer=aware_context.lineage_by_reviewer,
        prior_pass=aware_context.prior_pass,
        credible_mass=aware_context.credible_mass,
    )
    assert aware_context.prior_pass == naive_context.prior_pass == _PRIOR_PASS
    assert dict(aware_context.lineage_by_reviewer) == dict(
        naive_context.lineage_by_reviewer
    )
    np.testing.assert_array_equal(
        naive_dependence.correlation,
        np.eye(len(naive_dependence.reviewer_ids)),
    )
    assert dict(naive_dependence._weight_overrides) == {}
    for reviewer_id in aware_dependence.reviewer_ids:
        aware_draws = aware_context.likelihood_draws[reviewer_id]
        naive_draws = naive_context.likelihood_draws[reviewer_id]
        assert aware_draws.shape == naive_draws.shape
        assert aware_draws.dtype == naive_draws.dtype
        assert aware_draws.tobytes() == naive_draws.tobytes()

    reviewer_ids = aware_dependence.reviewer_ids
    policy_observations, policy_mask = _review_matrix(
        calibration_panel,
        policy_ids,
        reviewer_ids,
    )
    for pattern in np.unique(policy_mask, axis=0):
        subset = tuple(
            reviewer_id
            for reviewer_id, included in zip(
                reviewer_ids,
                pattern,
                strict=True,
            )
            if included
        )
        weights = naive_dependence.weights_for(subset)
        assert all(weight == 1.0 for weight in weights.values())

    policy_batch = fuse_review_matrix(
        policy_observations,
        policy_mask,
        reviewer_ids,
        aware_context,
        chunk_size=_MATRIX_CHUNK_SIZE,
    )
    policy_truths = tuple(calibration_panel.truths[case_id] for case_id in policy_ids)
    assert set(policy_truths) == {Truth.PASS, Truth.FAIL}
    reference_selection = _reference_select_policy(policy_truths, policy_batch)
    production_selection = select_decision_policy(
        {case_id: truth for case_id, truth in zip(policy_ids, policy_truths, strict=True)},
        _posterior_mapping(policy_batch, policy_ids),
        {case_id: calibration_panel.gates[case_id] for case_id in policy_ids},
        costs=_COSTS,
        min_coverage=_MINIMUM_COVERAGE,
    )
    assert _canonical_policy(production_selection.policy) == reference_selection.policy
    assert (
        production_selection.constraint_satisfied
        is reference_selection.constraint_satisfied
    )
    assert production_selection.decision_loss == pytest.approx(
        reference_selection.score.loss,
        abs=1e-12,
    )
    assert production_selection.coverage == pytest.approx(
        reference_selection.score.coverage,
        abs=1e-12,
    )

    test_observations, test_mask = _review_matrix(
        test_panel,
        test_ids,
        reviewer_ids,
    )
    for pattern in np.unique(test_mask, axis=0):
        subset = tuple(
            reviewer_id
            for reviewer_id, included in zip(
                reviewer_ids,
                pattern,
                strict=True,
            )
            if included
        )
        weights = naive_dependence.weights_for(subset)
        assert all(weight == 1.0 for weight in weights.values())

    original_review_bytes = test_observations.tobytes() + test_mask.tobytes()
    aware_batch = fuse_review_matrix(
        test_observations,
        test_mask,
        reviewer_ids,
        aware_context,
        chunk_size=_MATRIX_CHUNK_SIZE,
    )
    naive_batch = fuse_review_matrix(
        test_observations,
        test_mask,
        reviewer_ids,
        naive_context,
        chunk_size=_MATRIX_CHUNK_SIZE,
    )
    assert test_observations.tobytes() + test_mask.tobytes() == original_review_bytes
    assert np.all(aware_batch.valid_reviewers > 0)
    assert np.all(naive_batch.valid_reviewers > 0)

    test_truths = tuple(test_panel.truths[case_id] for case_id in test_ids)
    reference_corum_actions = _reference_actions(
        aware_batch,
        reference_selection.policy,
    )
    posterior_by_id = _posterior_mapping(aware_batch, test_ids)
    production_corum_actions = tuple(
        decide(
            posterior_by_id[case_id],
            test_panel.gates[case_id],
            production_selection.policy,
        ).action
        for case_id in test_ids
    )
    assert production_corum_actions == reference_corum_actions

    grouped_reviews = _reviews_by_case(test_panel, test_ids)
    assert all(
        len(reviews) == len(reviewer_ids)
        and {review.reviewer_id for review in reviews} == set(reviewer_ids)
        for reviews in grouped_reviews.values()
    )
    reference_majority_actions = tuple(
        _reference_majority(grouped_reviews[case_id])
        for case_id in test_ids
    )
    production_majority_actions = tuple(
        majority_decision(grouped_reviews[case_id])
        for case_id in test_ids
    )
    assert production_majority_actions == reference_majority_actions

    aware_probabilities = tuple(float(value) for value in aware_batch.pass_probability)
    naive_probabilities = tuple(float(value) for value in naive_batch.pass_probability)
    corum_score = _reference_score(
        test_truths,
        reference_corum_actions,
        aware_probabilities,
    )
    majority_score = _reference_score(test_truths, reference_majority_actions)
    naive_score = _reference_score(
        test_truths,
        (Action.DEFER,) * len(test_truths),
        naive_probabilities,
    )

    truth_mapping = {
        case_id: truth for case_id, truth in zip(test_ids, test_truths, strict=True)
    }
    corum_metrics = evaluate_decisions(
        truth_mapping,
        {
            case_id: action
            for case_id, action in zip(
                test_ids,
                production_corum_actions,
                strict=True,
            )
        },
        probabilities={
            case_id: probability
            for case_id, probability in zip(
                test_ids,
                aware_probabilities,
                strict=True,
            )
        },
        costs=_COSTS,
    )
    majority_metrics = evaluate_decisions(
        truth_mapping,
        {
            case_id: action
            for case_id, action in zip(
                test_ids,
                production_majority_actions,
                strict=True,
            )
        },
        costs=_COSTS,
    )
    naive_metrics = evaluate_decisions(
        truth_mapping,
        {case_id: Action.DEFER for case_id in test_ids},
        probabilities={
            case_id: probability
            for case_id, probability in zip(
                test_ids,
                naive_probabilities,
                strict=True,
            )
        },
        costs=_COSTS,
    )
    _assert_production_score(corum_metrics, corum_score, probabilities=True)
    _assert_production_score(majority_metrics, majority_score, probabilities=False)
    assert naive_metrics["brier"] == pytest.approx(naive_score.brier, abs=1e-12)
    assert naive_metrics["log_loss"] == pytest.approx(naive_score.nll, abs=1e-12)

    gate_violations = sum(
        bool(gates)
        for gates in (
            *calibration_panel.gates.values(),
            *test_panel.gates.values(),
        )
    )
    return _RunRecord(
        scenario=scenario_name,
        seed=seed,
        corum=corum_score,
        majority=majority_score,
        naive=naive_score,
        policy_constraint_satisfied=reference_selection.constraint_satisfied,
        gate_violations=gate_violations,
    )


def _assert_locked_contract() -> None:
    scenarios = builtin_scenarios()

    assert _SCENARIO_NAMES == ("independent", "clone_pair", "majority_trap")
    assert _SEEDS == tuple(range(20))
    assert _CALIBRATION_CASES == 2_000
    assert (_FIT_CASES, _POLICY_CASES, _TEST_CASES) == (1_600, 400, 5_000)
    assert (
        _PRIOR_STRENGTH,
        _DEPENDENCE_SHRINKAGE,
        _MINIMUM_OVERLAP,
        _LINEAGE_CAP,
    ) == (1.5, 0.25, 10, 1.0)
    assert (
        _POSTERIOR_DRAWS,
        _CREDIBLE_MASS,
        _MATRIX_CHUNK_SIZE,
        _PRIOR_PASS,
    ) == (512, 0.95, 4_096, 0.80)
    assert (_BOOTSTRAP_DRAWS, _BOOTSTRAP_SEED) == (2_000, 20_260_828)
    assert _MINIMUM_COVERAGE == 0.50
    assert _NLL_EPSILON == 1e-15
    assert (
        _COSTS.false_pass,
        _COSTS.false_fail,
        _COSTS.defer,
    ) == (1.0, 0.2, 0.1)
    assert OBSERVATION_ORDER == (
        Observation.PASS,
        Observation.FAIL,
        Observation.ABSTAIN,
    )
    assert tuple(_canonical_policy(policy) for policy in policy_candidates()) == (
        _POLICY_GRID
    )
    for scenario_name in _SCENARIO_NAMES:
        scenario = scenarios[scenario_name]
        expected_calibration, expected_test = _SCENARIO_SNAPSHOTS[scenario_name]
        assert scenario.name == scenario_name
        assert _phase_snapshot(scenario.calibration) == expected_calibration
        assert _phase_snapshot(scenario.test) == expected_test


def test_locked_scenario_snapshots_and_constants() -> None:
    _assert_locked_contract()


def test_locked_core_value_gate() -> None:
    _assert_locked_contract()
    records = tuple(
        _build_run(scenario_name, scenario_index, seed)
        for scenario_index, scenario_name in enumerate(_SCENARIO_NAMES)
        for seed in _SEEDS
    )
    assert len(records) == 60
    assert len({(record.scenario, record.seed) for record in records}) == 60

    bootstrap_rows = tuple(
        {
            "scenario": record.scenario,
            "seed": record.seed,
            "benefit": record.majority.loss - record.corum.loss,
        }
        for record in records
    )
    reference_interval = _reference_bootstrap(bootstrap_rows)
    production_interval = stratified_paired_bootstrap(
        bootstrap_rows,
        _bootstrap_metric,
        strata=("scenario",),
        draws=_BOOTSTRAP_DRAWS,
        seed=_BOOTSTRAP_SEED,
    )
    assert production_interval == pytest.approx(reference_interval, abs=1e-15)

    pooled_corum_loss = _micro_mean(records, "corum", "loss_sum")
    pooled_majority_loss = _micro_mean(records, "majority", "loss_sum")
    pooled_coverage = _micro_mean(records, "corum", "decided")
    pooled_false_pass = _micro_false_pass_rate(records, "corum")
    pooled_majority_false_pass = _micro_false_pass_rate(records, "majority")
    assert reference_interval[0] == pytest.approx(
        pooled_majority_loss - pooled_corum_loss,
        abs=1e-15,
    )

    diagnostics: dict[str, float | int | bool] = {
        "pooled_corum_loss": pooled_corum_loss,
        "pooled_majority_loss": pooled_majority_loss,
        "bootstrap_lower": reference_interval[1],
        "pooled_coverage": pooled_coverage,
        "pooled_false_pass_rate": pooled_false_pass,
        "pooled_majority_false_pass_rate": pooled_majority_false_pass,
        "policy_constraints_satisfied": sum(
            record.policy_constraint_satisfied for record in records
        ),
        "gate_violations": sum(record.gate_violations for record in records),
    }
    predicates: dict[str, bool] = {
        "pooled_loss_improvement": (
            pooled_corum_loss <= 0.90 * pooled_majority_loss
        ),
        "paired_interval_positive": reference_interval[1] > 0.0,
        "all_policy_constraints": all(
            record.policy_constraint_satisfied for record in records
        ),
        "pooled_coverage": pooled_coverage >= 0.50,
        "pooled_false_pass_finite": isfinite(pooled_false_pass),
        "pooled_false_pass_boundary": (
            pooled_false_pass <= pooled_majority_false_pass + 0.02
        ),
        "empty_gate_violations": (
            sum(record.gate_violations for record in records) == 0
        ),
    }

    for scenario_name in _SCENARIO_NAMES:
        scenario_records = tuple(
            record for record in records if record.scenario == scenario_name
        )
        corum_loss = _micro_mean(scenario_records, "corum", "loss_sum")
        majority_loss = _micro_mean(
            scenario_records,
            "majority",
            "loss_sum",
        )
        aware_nll = _micro_mean(scenario_records, "corum", "nll_sum")
        naive_nll = _micro_mean(scenario_records, "naive", "nll_sum")
        aware_brier = _micro_mean(scenario_records, "corum", "brier_sum")
        naive_brier = _micro_mean(scenario_records, "naive", "brier_sum")
        diagnostics[f"{scenario_name}_corum_loss"] = corum_loss
        diagnostics[f"{scenario_name}_majority_loss"] = majority_loss
        diagnostics[f"{scenario_name}_aware_nll"] = aware_nll
        diagnostics[f"{scenario_name}_naive_nll"] = naive_nll
        diagnostics[f"{scenario_name}_aware_brier"] = aware_brier
        diagnostics[f"{scenario_name}_naive_brier"] = naive_brier
        predicates[f"{scenario_name}_loss_boundary"] = (
            corum_loss <= majority_loss + 0.01
        )
        valid_naive_denominators = (
            isfinite(naive_nll)
            and naive_nll > 0.0
            and isfinite(naive_brier)
            and naive_brier > 0.0
        )
        valid_aware_scores = (
            isfinite(aware_nll) and isfinite(aware_brier)
        )
        predicates[f"{scenario_name}_naive_denominators"] = (
            valid_naive_denominators
        )
        predicates[f"{scenario_name}_finite_aware_scores"] = valid_aware_scores
        valid_probability_comparison = (
            valid_naive_denominators and valid_aware_scores
        )
        if scenario_name == "independent" and valid_probability_comparison:
            predicates["independent_nll_degradation"] = (
                (aware_nll - naive_nll) / naive_nll <= 0.01
            )
            predicates["independent_brier_degradation"] = (
                (aware_brier - naive_brier) / naive_brier <= 0.01
            )
        elif scenario_name == "independent":
            predicates["independent_nll_degradation"] = False
            predicates["independent_brier_degradation"] = False
        elif valid_probability_comparison:
            best_relative_improvement = max(
                (naive_nll - aware_nll) / naive_nll,
                (naive_brier - aware_brier) / naive_brier,
            )
            diagnostics[
                f"{scenario_name}_best_relative_improvement"
            ] = best_relative_improvement
            predicates[f"{scenario_name}_dependence_improvement"] = (
                best_relative_improvement >= 0.05
            )
        else:
            diagnostics[
                f"{scenario_name}_best_relative_improvement"
            ] = float("nan")
            predicates[f"{scenario_name}_dependence_improvement"] = False

    failures = {
        name: diagnostics
        for name, passed in predicates.items()
        if not passed
    }
    assert not failures, f"CORE_VALUE_GATE_FAILED: {failures}"
