from dataclasses import FrozenInstanceError
from decimal import localcontext
from math import inf, nan
from typing import Any

import numpy as np
import pytest

from corum.baselines import (
    DecisionCosts,
    best_single_reviewer,
    linear_pool_probability,
    majority_decision,
)
from corum.calibration import ReviewerCalibration
from corum.models import (
    Action,
    CalibrationExample,
    ExecutionState,
    Observation,
    Review,
    Truth,
)


def _review(
    reviewer_id: str,
    observation: Observation | None,
    *,
    case_id: str = "case-1",
    state: ExecutionState = ExecutionState.VALID,
) -> Review:
    return Review(
        case_id=case_id,
        reviewer_id=reviewer_id,
        observation=observation,
        state=state,
    )


def _example(
    reviewer_id: str,
    truth: Truth,
    observation: Observation | None,
    *,
    case_id: str,
    state: ExecutionState = ExecutionState.VALID,
) -> CalibrationExample:
    return CalibrationExample(
        truth=truth,
        review=_review(
            reviewer_id,
            observation,
            case_id=case_id,
            state=state,
        ),
    )


def _calibration(
    reviewer_id: str,
    likelihoods: list[list[float]] | np.ndarray,
) -> ReviewerCalibration:
    probabilities = np.asarray(likelihoods, dtype=float)
    strength = 10.0
    return ReviewerCalibration(
        reviewer_id=reviewer_id,
        alpha=probabilities * strength,
        observed_counts=np.zeros((2, 3), dtype=np.int64),
        prior_strength=strength,
    )


def _symmetric_calibration(
    reviewer_id: str,
    accuracy: float,
    *,
    abstain: float = 0.05,
) -> ReviewerCalibration:
    wrong = 1.0 - accuracy - abstain
    return _calibration(
        reviewer_id,
        [
            [accuracy, wrong, abstain],
            [wrong, accuracy, abstain],
        ],
    )


def test_decision_costs_are_frozen_normalized_and_use_published_defaults() -> None:
    costs = DecisionCosts()

    assert costs == DecisionCosts(false_pass=1, false_fail=0.2, defer=0.1)
    assert isinstance(costs.false_pass, float)
    with pytest.raises(FrozenInstanceError):
        costs.false_pass = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("false_pass", True, TypeError),
        ("false_pass", -0.1, ValueError),
        ("false_fail", nan, ValueError),
        ("defer", inf, ValueError),
        ("defer", 10**400, ValueError),
    ],
)
def test_decision_costs_reject_invalid_values(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field):
        DecisionCosts(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reviews", "expected"),
    [
        (
            (
                _review("a", Observation.PASS),
                _review("b", Observation.PASS),
                _review("c", Observation.FAIL),
            ),
            Action.PASS,
        ),
        (
            (
                _review("a", Observation.FAIL),
                _review("b", Observation.FAIL),
                _review("c", Observation.PASS),
            ),
            Action.FAIL,
        ),
        (
            (
                _review("a", Observation.PASS),
                _review("b", Observation.FAIL),
            ),
            Action.DEFER,
        ),
        ((), Action.DEFER),
        (
            (
                _review("a", Observation.ABSTAIN),
                _review("b", Observation.ABSTAIN),
            ),
            Action.DEFER,
        ),
    ],
)
def test_majority_counts_only_directional_valid_votes(
    reviews: tuple[Review, ...],
    expected: Action,
) -> None:
    assert majority_decision(reviews) is expected


@pytest.mark.parametrize(
    "state",
    [
        ExecutionState.TIMEOUT,
        ExecutionState.INVALID,
        ExecutionState.REFUSAL,
        ExecutionState.NOT_CALLED,
    ],
)
def test_majority_excludes_every_non_valid_execution(state: ExecutionState) -> None:
    reviews = (
        _review("valid", Observation.PASS),
        _review("missing", None, state=state),
    )

    assert majority_decision(reviews) is Action.PASS


def test_majority_is_invariant_to_review_order() -> None:
    reviews = (
        _review("a", Observation.PASS),
        _review("b", Observation.FAIL),
        _review("c", Observation.PASS),
    )

    assert majority_decision(reviews) is majority_decision(tuple(reversed(reviews)))


def test_majority_rejects_duplicates_mixed_cases_and_bad_rows() -> None:
    with pytest.raises(ValueError, match="duplicate reviewer"):
        majority_decision(
            (
                _review("a", Observation.PASS),
                _review("a", Observation.FAIL),
            )
        )
    with pytest.raises(ValueError, match="one case"):
        majority_decision(
            (
                _review("a", Observation.PASS, case_id="case-1"),
                _review("b", Observation.PASS, case_id="case-2"),
            )
        )
    with pytest.raises(TypeError, match=r"reviews\[0\].*Review"):
        majority_decision(("not-a-review",))  # type: ignore[arg-type]


def test_linear_pool_matches_hand_calculated_prior_aware_posteriors() -> None:
    calibrations = {
        "r1": _calibration(
            "r1",
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
        ),
        "r2": _calibration(
            "r2",
            [[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]],
        ),
    }
    reviews = (
        _review("r1", Observation.PASS),
        _review("r2", Observation.FAIL),
    )

    probability = linear_pool_probability(
        reviews,
        calibrations,
        prior_pass=0.25,
    )

    assert probability == pytest.approx(83 / 266)
    assert probability != pytest.approx(0.25)


def test_linear_pool_includes_valid_abstain_after_reviewer_specific_bayes() -> None:
    calibrations = {
        "directional": _calibration(
            "directional",
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]],
        ),
        "neutral": _calibration(
            "neutral",
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]],
        ),
    }
    reviews = (
        _review("directional", Observation.ABSTAIN),
        _review("neutral", Observation.ABSTAIN),
    )

    probability = linear_pool_probability(
        reviews,
        calibrations,
        prior_pass=0.5,
    )

    assert probability == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_linear_pool_bayes_update_is_stable_for_subnormal_prior() -> None:
    epsilon = 5e-324
    calibration = _calibration(
        "extreme",
        [
            [0.5, 0.5, epsilon],
            [epsilon, 0.5, 0.5],
        ],
    )

    probability = linear_pool_probability(
        (_review("extreme", Observation.PASS),),
        {"extreme": calibration},
        prior_pass=epsilon,
    )

    assert probability == pytest.approx(1.0 / 3.0)


def test_linear_pool_excludes_non_valid_and_returns_none_without_valid_rows() -> None:
    calibrations = {
        "valid": _symmetric_calibration("valid", 0.8),
        "missing": _symmetric_calibration("missing", 0.8),
    }
    valid_only = linear_pool_probability(
        (_review("valid", Observation.PASS),),
        calibrations,
        prior_pass=0.5,
    )
    with_timeout = linear_pool_probability(
        (
            _review("valid", Observation.PASS),
            _review("missing", None, state=ExecutionState.TIMEOUT),
        ),
        calibrations,
        prior_pass=0.5,
    )

    assert with_timeout == valid_only
    assert (
        linear_pool_probability(
            (_review("missing", None, state=ExecutionState.INVALID),),
            calibrations,
            prior_pass=0.5,
        )
        is None
    )


def test_linear_pool_rejects_invalid_panel_and_calibration_schema() -> None:
    r1 = _symmetric_calibration("r1", 0.8)
    r2 = _symmetric_calibration("r2", 0.8)
    with pytest.raises(ValueError, match="duplicate reviewer"):
        linear_pool_probability(
            (
                _review("r1", Observation.PASS),
                _review("r1", Observation.FAIL),
            ),
            {"r1": r1},
            prior_pass=0.5,
        )
    with pytest.raises(ValueError, match="one case"):
        linear_pool_probability(
            (
                _review("r1", Observation.PASS, case_id="case-1"),
                _review("r2", Observation.PASS, case_id="case-2"),
            ),
            {"r1": r1, "r2": r2},
            prior_pass=0.5,
        )
    with pytest.raises(ValueError, match="missing calibration.*unknown"):
        linear_pool_probability(
            (_review("unknown", None, state=ExecutionState.TIMEOUT),),
            {"r1": r1},
            prior_pass=0.5,
        )
    with pytest.raises(ValueError, match="mapping key.*reviewer_id"):
        linear_pool_probability(
            (_review("r1", Observation.PASS),),
            {"r1": r2},
            prior_pass=0.5,
        )


@pytest.mark.parametrize("prior", [0.0, 1.0, nan, inf, True])
def test_linear_pool_requires_open_finite_prior(prior: object) -> None:
    with pytest.raises((TypeError, ValueError), match="prior_pass"):
        linear_pool_probability(
            (_review("r1", Observation.PASS),),
            {"r1": _symmetric_calibration("r1", 0.8)},
            prior_pass=prior,  # type: ignore[arg-type]
        )


def test_best_single_selects_lowest_policy_loss_from_complete_grid() -> None:
    calibrations = {
        "good": _symmetric_calibration("good", 0.9),
        "weak": _symmetric_calibration("weak", 0.6),
    }
    rows = tuple(
        _example(
            reviewer_id,
            truth,
            Observation.PASS if truth is Truth.PASS else Observation.FAIL,
            case_id=case_id,
        )
        for reviewer_id in calibrations
        for case_id, truth in (
            ("pass-1", Truth.PASS),
            ("pass-2", Truth.PASS),
            ("fail-1", Truth.FAIL),
            ("fail-2", Truth.FAIL),
        )
    )

    selected = best_single_reviewer(
        rows,
        calibrations,
        prior_pass=0.5,
        pass_threshold=0.8,
        fail_threshold=0.2,
        costs=DecisionCosts(),
    )

    assert selected == "good"


def test_best_single_exact_loss_tie_breaks_by_reviewer_id() -> None:
    calibrations = {
        "z-reviewer": _symmetric_calibration("z-reviewer", 0.8),
        "a-reviewer": _symmetric_calibration("a-reviewer", 0.8),
    }
    rows = tuple(
        _example(
            reviewer_id,
            Truth.PASS,
            Observation.PASS,
            case_id="policy-case",
        )
        for reviewer_id in calibrations
    )

    selected = best_single_reviewer(
        rows,
        calibrations,
        prior_pass=0.5,
        pass_threshold=0.8,
        fail_threshold=0.2,
        costs=DecisionCosts(),
    )

    assert selected == "a-reviewer"


def test_best_single_ranks_finite_extreme_costs_without_overflow() -> None:
    calibrations = {
        "a-worse": _symmetric_calibration("a-worse", 0.9),
        "b-better": _symmetric_calibration("b-better", 0.9),
    }
    rows = tuple(
        _example(
            reviewer_id,
            Truth.FAIL,
            (
                Observation.FAIL
                if reviewer_id == "b-better" and case_index == 3
                else Observation.PASS
            ),
            case_id=f"case-{case_index}",
        )
        for reviewer_id in calibrations
        for case_index in range(4)
    )

    selected = best_single_reviewer(
        rows,
        calibrations,
        prior_pass=0.5,
        pass_threshold=0.8,
        fail_threshold=0.2,
        costs=DecisionCosts(false_pass=1e308),
    )

    assert selected == "b-better"


def test_best_single_preserves_small_active_cost_when_larger_cost_is_inactive() -> None:
    epsilon = 5e-324
    calibrations = {
        "a-worse": _symmetric_calibration("a-worse", 0.9),
        "b-better": _symmetric_calibration("b-better", 0.9),
    }
    rows = tuple(
        _example(
            reviewer_id,
            Truth.PASS,
            (
                Observation.PASS
                if reviewer_id == "b-better" and case_index == 1
                else Observation.FAIL
            ),
            case_id=f"case-{case_index}",
        )
        for reviewer_id in calibrations
        for case_index in range(2)
    )

    selected = best_single_reviewer(
        rows,
        calibrations,
        prior_pass=0.5,
        pass_threshold=0.8,
        fail_threshold=0.2,
        costs=DecisionCosts(
            false_pass=1e308,
            false_fail=epsilon,
            defer=0.0,
        ),
    )

    assert selected == "b-better"


def test_best_single_is_independent_of_callers_decimal_context() -> None:
    calibrations = {
        "a-worse": _symmetric_calibration("a-worse", 0.9),
        "b-better": _symmetric_calibration("b-better", 0.9),
    }
    rows = (
        _example("a-worse", Truth.FAIL, Observation.PASS, case_id="fail"),
        _example("a-worse", Truth.PASS, Observation.PASS, case_id="pass"),
        _example("b-better", Truth.FAIL, Observation.FAIL, case_id="fail"),
        _example("b-better", Truth.PASS, Observation.FAIL, case_id="pass"),
    )

    with localcontext() as context:
        context.prec = 1
        selected = best_single_reviewer(
            rows,
            calibrations,
            prior_pass=0.5,
            pass_threshold=0.8,
            fail_threshold=0.2,
            costs=DecisionCosts(false_pass=1.1, false_fail=1.0, defer=0.0),
        )

    assert selected == "b-better"


def test_best_single_abstain_never_authorizes_an_action() -> None:
    calibrations = {
        "a-abstain": _calibration(
            "a-abstain",
            [[0.05, 0.05, 0.90], [0.49, 0.50, 0.01]],
        ),
        "b-directional": _symmetric_calibration("b-directional", 0.9),
    }
    rows = (
        _example(
            "a-abstain",
            Truth.PASS,
            Observation.ABSTAIN,
            case_id="policy-case",
        ),
        _example(
            "b-directional",
            Truth.PASS,
            Observation.PASS,
            case_id="policy-case",
        ),
    )

    selected = best_single_reviewer(
        rows,
        calibrations,
        prior_pass=0.5,
        pass_threshold=0.8,
        fail_threshold=0.2,
        costs=DecisionCosts(),
    )

    assert selected == "b-directional"


def test_best_single_rejects_incomplete_or_malformed_policy_ledger() -> None:
    calibrations = {
        "a": _symmetric_calibration("a", 0.8),
        "b": _symmetric_calibration("b", 0.8),
    }
    complete = (
        _example("a", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("a", Truth.FAIL, Observation.FAIL, case_id="case-2"),
        _example("b", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("b", Truth.FAIL, Observation.FAIL, case_id="case-2"),
    )
    kwargs: dict[str, Any] = {
        "prior_pass": 0.5,
        "pass_threshold": 0.8,
        "fail_threshold": 0.2,
        "costs": DecisionCosts(),
    }

    with pytest.raises(ValueError, match="policy case grid"):
        best_single_reviewer(complete[:-1], calibrations, **kwargs)
    with pytest.raises(ValueError, match="duplicate reviewer-case"):
        best_single_reviewer(complete + (complete[0],), calibrations, **kwargs)
    conflicting = complete[:2] + (
        _example("b", Truth.FAIL, Observation.FAIL, case_id="case-1"),
    ) + complete[3:]
    with pytest.raises(ValueError, match="conflicting truth"):
        best_single_reviewer(conflicting, calibrations, **kwargs)
    unknown = complete + (
        _example("unknown", Truth.PASS, Observation.PASS, case_id="case-1"),
    )
    with pytest.raises(ValueError, match="unknown reviewer"):
        best_single_reviewer(unknown, calibrations, **kwargs)


def test_best_single_rejects_empty_candidates_rows_and_bad_thresholds() -> None:
    calibration = _symmetric_calibration("a", 0.8)
    row = _example("a", Truth.PASS, Observation.PASS, case_id="case-1")

    with pytest.raises(ValueError, match="calibrations.*at least one"):
        best_single_reviewer(
            (row,),
            {},
            prior_pass=0.5,
            pass_threshold=0.8,
            fail_threshold=0.2,
            costs=DecisionCosts(),
        )
    with pytest.raises(ValueError, match="policy_rows.*at least one"):
        best_single_reviewer(
            (),
            {"a": calibration},
            prior_pass=0.5,
            pass_threshold=0.8,
            fail_threshold=0.2,
            costs=DecisionCosts(),
        )
    with pytest.raises(ValueError, match="fail_threshold.*lower"):
        best_single_reviewer(
            (row,),
            {"a": calibration},
            prior_pass=0.5,
            pass_threshold=0.2,
            fail_threshold=0.8,
            costs=DecisionCosts(),
        )


def test_best_single_rejects_calibration_key_identity_mismatch() -> None:
    row = _example("a", Truth.PASS, Observation.PASS, case_id="case-1")

    with pytest.raises(ValueError, match="mapping key.*reviewer_id"):
        best_single_reviewer(
            (row,),
            {"a": _symmetric_calibration("different", 0.8)},
            prior_pass=0.5,
            pass_threshold=0.8,
            fail_threshold=0.2,
            costs=DecisionCosts(),
        )
