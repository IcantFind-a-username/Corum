from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import exp, isfinite, log, log1p
from numbers import Real

from corum.calibration import OBSERVATION_ORDER, ReviewerCalibration
from corum.models import (
    Action,
    CalibrationExample,
    ExecutionState,
    Observation,
    Review,
    Truth,
)


def _non_negative_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite and non-negative") from error
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return numeric


def _open_probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite and within (0, 1)") from error
    if not isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(f"{field} must be finite and within (0, 1)")
    return numeric


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite and within [0, 1]") from error
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return numeric


@dataclass(frozen=True, slots=True)
class DecisionCosts:
    false_pass: float = 1.0
    false_fail: float = 0.2
    defer: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "false_pass",
            _non_negative_real(self.false_pass, "false_pass"),
        )
        object.__setattr__(
            self,
            "false_fail",
            _non_negative_real(self.false_fail, "false_fail"),
        )
        object.__setattr__(
            self,
            "defer",
            _non_negative_real(self.defer, "defer"),
        )


def _validated_reviews(reviews: Sequence[Review]) -> tuple[Review, ...]:
    rows = tuple(reviews)
    reviewer_ids: set[str] = set()
    case_ids: set[str] = set()
    for index, review in enumerate(rows):
        if not isinstance(review, Review):
            raise TypeError(f"reviews[{index}] must be a Review")
        if review.reviewer_id in reviewer_ids:
            raise ValueError(f"duplicate reviewer ID: {review.reviewer_id}")
        reviewer_ids.add(review.reviewer_id)
        case_ids.add(review.case_id)
    if len(case_ids) > 1:
        raise ValueError("reviews must contain exactly one case")
    return rows


def majority_decision(reviews: Sequence[Review]) -> Action:
    rows = _validated_reviews(reviews)
    pass_votes = sum(
        review.state is ExecutionState.VALID
        and review.observation is Observation.PASS
        for review in rows
    )
    fail_votes = sum(
        review.state is ExecutionState.VALID
        and review.observation is Observation.FAIL
        for review in rows
    )
    if pass_votes > fail_votes:
        return Action.PASS
    if fail_votes > pass_votes:
        return Action.FAIL
    return Action.DEFER


def _validated_calibrations(
    calibrations: Mapping[str, ReviewerCalibration],
    *,
    require_non_empty: bool,
) -> dict[str, ReviewerCalibration]:
    if not isinstance(calibrations, Mapping):
        raise TypeError("calibrations must be a mapping")
    copied = dict(calibrations)
    if require_non_empty and not copied:
        raise ValueError("calibrations must contain at least one candidate")
    for reviewer_id, calibration in copied.items():
        if not isinstance(reviewer_id, str):
            raise TypeError("calibration mapping keys must be strings")
        if not isinstance(calibration, ReviewerCalibration):
            raise TypeError(
                f"calibrations[{reviewer_id!r}] must be a ReviewerCalibration"
            )
        if calibration.reviewer_id != reviewer_id:
            raise ValueError(
                "calibration mapping key must match calibration.reviewer_id; "
                f"key={reviewer_id!r}, reviewer_id={calibration.reviewer_id!r}"
            )
    return copied


def _reviewer_posterior(
    observation: Observation,
    calibration: ReviewerCalibration,
    prior_pass: float,
) -> float:
    likelihoods = calibration.mean_likelihoods()
    observation_index = OBSERVATION_ORDER.index(observation)
    pass_likelihood = float(likelihoods[0, observation_index])
    fail_likelihood = float(likelihoods[1, observation_index])
    if pass_likelihood <= 0.0 and fail_likelihood <= 0.0:
        raise ValueError("calibration produced zero posterior mass")
    if pass_likelihood <= 0.0:
        return 0.0
    if fail_likelihood <= 0.0:
        return 1.0
    log_pass = log(prior_pass) + log(pass_likelihood)
    log_fail = log1p(-prior_pass) + log(fail_likelihood)
    maximum = max(log_pass, log_fail)
    pass_mass = exp(log_pass - maximum)
    fail_mass = exp(log_fail - maximum)
    denominator = pass_mass + fail_mass
    return pass_mass / denominator


def linear_pool_probability(
    reviews: Sequence[Review],
    calibrations: Mapping[str, ReviewerCalibration],
    *,
    prior_pass: float,
) -> float | None:
    rows = _validated_reviews(reviews)
    prior = _open_probability(prior_pass, "prior_pass")
    calibration_by_id = _validated_calibrations(
        calibrations,
        require_non_empty=False,
    )
    unknown = sorted(
        {review.reviewer_id for review in rows} - set(calibration_by_id)
    )
    if unknown:
        raise ValueError("missing calibration for reviewer IDs: " + ", ".join(unknown))

    posteriors = [
        _reviewer_posterior(
            review.observation,
            calibration_by_id[review.reviewer_id],
            prior,
        )
        for review in rows
        if review.state is ExecutionState.VALID and review.observation is not None
    ]
    if not posteriors:
        return None
    return sum(posteriors) / len(posteriors)


def _thresholds(
    pass_threshold: object,
    fail_threshold: object,
) -> tuple[float, float]:
    pass_value = _probability(pass_threshold, "pass_threshold")
    fail_value = _probability(fail_threshold, "fail_threshold")
    if fail_value >= pass_value:
        raise ValueError("fail_threshold must be lower than pass_threshold")
    return pass_value, fail_value


def _action_loss(action: Action, truth: Truth, costs: DecisionCosts) -> float:
    if action is Action.DEFER:
        return costs.defer
    if action is Action.PASS and truth is Truth.FAIL:
        return costs.false_pass
    if action is Action.FAIL and truth is Truth.PASS:
        return costs.false_fail
    return 0.0


def best_single_reviewer(
    policy_rows: Sequence[CalibrationExample],
    calibrations: Mapping[str, ReviewerCalibration],
    *,
    prior_pass: float,
    pass_threshold: float,
    fail_threshold: float,
    costs: DecisionCosts,
) -> str:
    calibration_by_id = _validated_calibrations(
        calibrations,
        require_non_empty=True,
    )
    if not isinstance(costs, DecisionCosts):
        raise TypeError("costs must be a DecisionCosts")
    prior = _open_probability(prior_pass, "prior_pass")
    pass_value, fail_value = _thresholds(pass_threshold, fail_threshold)
    rows = tuple(policy_rows)
    if not rows:
        raise ValueError("policy_rows must contain at least one row")

    rows_by_reviewer: dict[str, dict[str, CalibrationExample]] = {
        reviewer_id: {} for reviewer_id in calibration_by_id
    }
    truth_by_case: dict[str, Truth] = {}
    unknown_reviewers: set[str] = set()
    for index, example in enumerate(rows):
        if not isinstance(example, CalibrationExample):
            raise TypeError(f"policy_rows[{index}] must be a CalibrationExample")
        reviewer_id = example.review.reviewer_id
        case_id = example.review.case_id
        if reviewer_id not in rows_by_reviewer:
            unknown_reviewers.add(reviewer_id)
            continue
        if case_id in rows_by_reviewer[reviewer_id]:
            raise ValueError(
                "duplicate reviewer-case row: "
                f"reviewer_id={reviewer_id!r}, case_id={case_id!r}"
            )
        if case_id in truth_by_case and truth_by_case[case_id] is not example.truth:
            raise ValueError(f"conflicting truth for policy case {case_id!r}")
        truth_by_case[case_id] = example.truth
        rows_by_reviewer[reviewer_id][case_id] = example
    if unknown_reviewers:
        raise ValueError(
            "policy_rows contain unknown reviewer IDs: "
            + ", ".join(sorted(unknown_reviewers))
        )

    expected_cases = set(truth_by_case)
    for reviewer_id, reviewer_rows in rows_by_reviewer.items():
        if set(reviewer_rows) != expected_cases:
            raise ValueError(
                "every reviewer must cover the same policy case grid; "
                f"reviewer_id={reviewer_id!r}"
            )

    losses: list[tuple[Fraction, str]] = []
    for reviewer_id in sorted(calibration_by_id):
        total_loss = Fraction()
        for case_id in sorted(expected_cases):
            example = rows_by_reviewer[reviewer_id][case_id]
            review = example.review
            action = Action.DEFER
            if (
                review.state is ExecutionState.VALID
                and review.observation is not None
                and review.observation is not Observation.ABSTAIN
            ):
                posterior = _reviewer_posterior(
                    review.observation,
                    calibration_by_id[reviewer_id],
                    prior,
                )
                if posterior >= pass_value:
                    action = Action.PASS
                elif posterior <= fail_value:
                    action = Action.FAIL
            total_loss += Fraction.from_float(
                _action_loss(action, example.truth, costs)
            )
        losses.append((total_loss, reviewer_id))
    return min(losses)[1]
