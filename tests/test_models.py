from collections.abc import Callable
from dataclasses import FrozenInstanceError
from fractions import Fraction
from importlib.metadata import version
from math import inf, nan

import pytest

import corum
from corum.models import (
    Action,
    CalibrationExample,
    Decision,
    ExecutionState,
    FusedPosterior,
    GateState,
    HardGate,
    Observation,
    Review,
    Reviewer,
    Truth,
)


def _posterior(
    *,
    pass_probability: float = 0.6,
    lower: float = 0.4,
    upper: float = 0.8,
    valid_reviewers: int = 2,
    lineage_count: int = 2,
    effective_sample_size: float = 1.5,
    samples: tuple[float, ...] = (0.4, 0.8),
) -> FusedPosterior:
    return FusedPosterior(
        pass_probability=pass_probability,
        lower=lower,
        upper=upper,
        valid_reviewers=valid_reviewers,
        lineage_count=lineage_count,
        effective_sample_size=effective_sample_size,
        samples=samples,
    )


def test_enum_values_form_the_public_wire_contract() -> None:
    assert tuple(Truth) == (Truth.PASS, Truth.FAIL)
    assert tuple(Observation) == (
        Observation.PASS,
        Observation.FAIL,
        Observation.ABSTAIN,
    )
    assert tuple(ExecutionState) == (
        ExecutionState.VALID,
        ExecutionState.TIMEOUT,
        ExecutionState.INVALID,
        ExecutionState.REFUSAL,
        ExecutionState.NOT_CALLED,
    )
    assert tuple(Action) == (Action.PASS, Action.FAIL, Action.DEFER)
    assert tuple(GateState) == (
        GateState.PASS,
        GateState.FAIL,
        GateState.UNRESOLVED,
    )
    assert all(member.value == member.name for enum_type in (
        Truth,
        Observation,
        ExecutionState,
        Action,
        GateState,
    ) for member in enum_type)


def test_valid_review_requires_an_observation() -> None:
    with pytest.raises(
        ValueError,
        match="observation is required when state is VALID",
    ):
        Review(
            case_id="case-1",
            reviewer_id="reviewer-1",
            observation=None,
            state=ExecutionState.VALID,
        )


def test_valid_review_accepts_an_observation() -> None:
    review = Review(
        case_id="case-1",
        reviewer_id="reviewer-1",
        observation=Observation.ABSTAIN,
        state=ExecutionState.VALID,
    )

    assert review.observation is Observation.ABSTAIN


@pytest.mark.parametrize(
    ("record_factory", "message"),
    [
        (
            lambda: Review(
                case_id="case-1",
                reviewer_id="reviewer-1",
                observation=None,
                state="VALID",  # type: ignore[arg-type]
            ),
            "Review.state must be an ExecutionState",
        ),
        (
            lambda: Review(
                case_id="case-1",
                reviewer_id="reviewer-1",
                observation="PASS",  # type: ignore[arg-type]
                state=ExecutionState.VALID,
            ),
            "Review.observation must be an Observation or None",
        ),
        (
            lambda: CalibrationExample(
                truth="PASS",  # type: ignore[arg-type]
                review=Review(
                    "case-1",
                    "reviewer-1",
                    Observation.PASS,
                    ExecutionState.VALID,
                ),
            ),
            "CalibrationExample.truth must be a Truth",
        ),
        (
            lambda: HardGate(
                gate_id="gate-1",
                state="PASS",  # type: ignore[arg-type]
            ),
            "HardGate.state must be a GateState",
        ),
        (
            lambda: Decision(
                action="PASS",  # type: ignore[arg-type]
                reasons=("decisive",),
                posterior=None,
            ),
            "Decision.action must be an Action",
        ),
    ],
)
def test_enum_fields_reject_raw_strings(
    record_factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        record_factory()


@pytest.mark.parametrize(
    ("record_factory", "message"),
    [
        (
            lambda: CalibrationExample(
                truth=Truth.PASS,
                review="not-a-review",  # type: ignore[arg-type]
            ),
            "CalibrationExample.review must be a Review",
        ),
        (
            lambda: Decision(
                action=Action.PASS,
                reasons=("decisive",),
                posterior="not-a-posterior",  # type: ignore[arg-type]
            ),
            "Decision.posterior must be a FusedPosterior or None",
        ),
    ],
)
def test_record_fields_reject_wrong_domain_types(
    record_factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        record_factory()


@pytest.mark.parametrize(
    "state",
    [
        ExecutionState.TIMEOUT,
        ExecutionState.INVALID,
        ExecutionState.REFUSAL,
        ExecutionState.NOT_CALLED,
    ],
)
def test_non_valid_review_requires_no_observation(state: ExecutionState) -> None:
    with pytest.raises(
        ValueError,
        match="observation must be None when state is not VALID",
    ):
        Review(
            case_id="case-1",
            reviewer_id="reviewer-1",
            observation=Observation.PASS,
            state=state,
        )


@pytest.mark.parametrize(
    "state",
    [
        ExecutionState.TIMEOUT,
        ExecutionState.INVALID,
        ExecutionState.REFUSAL,
        ExecutionState.NOT_CALLED,
    ],
)
def test_non_valid_review_accepts_no_observation(state: ExecutionState) -> None:
    review = Review(
        case_id="case-1",
        reviewer_id="reviewer-1",
        observation=None,
        state=state,
    )

    assert review.state is state


def test_negative_reviewer_cost_has_an_actionable_error() -> None:
    with pytest.raises(ValueError, match="Reviewer.cost must be non-negative"):
        Reviewer(
            reviewer_id="reviewer-1",
            vendor="vendor",
            family="family",
            lineage="lineage",
            cost=-0.01,
        )


@pytest.mark.parametrize("invalid_cost", [True, "1.0"])
def test_reviewer_cost_rejects_non_real_values(invalid_cost: object) -> None:
    with pytest.raises(TypeError, match="Reviewer.cost must be a real number"):
        Reviewer(
            reviewer_id="reviewer-1",
            vendor="vendor",
            family="family",
            lineage="lineage",
            cost=invalid_cost,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_cost", [nan, inf])
def test_reviewer_cost_rejects_non_finite_values(invalid_cost: float) -> None:
    with pytest.raises(
        ValueError,
        match="Reviewer.cost must be finite and non-negative",
    ):
        Reviewer(
            reviewer_id="reviewer-1",
            vendor="vendor",
            family="family",
            lineage="lineage",
            cost=invalid_cost,
        )


def test_reviewer_cost_accepts_finite_real_that_overflows_float_conversion() -> None:
    cost = Fraction(10**400, 1)

    reviewer = Reviewer(
        reviewer_id="reviewer-1",
        vendor="vendor",
        family="family",
        lineage="lineage",
        cost=cost,  # type: ignore[arg-type]
    )

    assert reviewer.cost == cost


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_negative_token_count_has_an_actionable_error(field: str) -> None:
    token_counts = {field: -1}

    with pytest.raises(
        ValueError,
        match=rf"Review\.{field} must be non-negative",
    ):
        Review(
            case_id="case-1",
            reviewer_id="reviewer-1",
            observation=Observation.PASS,
            state=ExecutionState.VALID,
            **token_counts,
        )


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("invalid_value", [True, 1.5, "1"])
def test_token_count_rejects_non_integer_values(
    field: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=rf"Review\.{field} must be an integer"):
        Review(
            case_id="case-1",
            reviewer_id="reviewer-1",
            observation=Observation.PASS,
            state=ExecutionState.VALID,
            **{field: invalid_value},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("record_factory", "field"),
    [
        (
            lambda value: Reviewer(value, "vendor", "family", "lineage"),
            "reviewer_id",
        ),
        (
            lambda value: Review(
                value,
                "reviewer-1",
                Observation.PASS,
                ExecutionState.VALID,
            ),
            "case_id",
        ),
        (
            lambda value: Review(
                "case-1",
                value,
                Observation.PASS,
                ExecutionState.VALID,
            ),
            "reviewer_id",
        ),
        (lambda value: HardGate(value, GateState.PASS), "gate_id"),
    ],
)
def test_blank_stable_identifier_has_an_actionable_error(
    record_factory: Callable[[str], object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must not be blank"):
        record_factory(" \t")


def test_all_records_are_frozen() -> None:
    review = Review(
        "case-1",
        "reviewer-1",
        Observation.PASS,
        ExecutionState.VALID,
    )
    posterior = _posterior()
    records_and_fields = [
        (Reviewer("reviewer-1", "vendor", "family", "lineage"), "cost"),
        (review, "state"),
        (CalibrationExample(Truth.PASS, review), "truth"),
        (HardGate("gate-1", GateState.PASS), "state"),
        (posterior, "lower"),
        (Decision(Action.PASS, ("decisive",), posterior), "action"),
    ]

    for record, field in records_and_fields:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)


@pytest.mark.parametrize(
    ("record_factory", "message"),
    [
        (
            lambda: _posterior(
                samples=[0.4, 0.8],  # type: ignore[arg-type]
            ),
            "FusedPosterior.samples must be a tuple",
        ),
        (
            lambda: Decision(
                action=Action.PASS,
                reasons=["decisive"],  # type: ignore[arg-type]
                posterior=None,
            ),
            "Decision.reasons must be a tuple",
        ),
    ],
)
def test_frozen_records_reject_mutable_sequence_inputs(
    record_factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        record_factory()


def test_decision_reasons_reject_mutable_nested_values() -> None:
    with pytest.raises(TypeError, match=r"Decision\.reasons\[0\] must be a str"):
        Decision(
            action=Action.PASS,
            reasons=(["mutable"],),  # type: ignore[arg-type]
            posterior=None,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        (field, invalid_value)
        for field in ("pass_probability", "lower", "upper")
        for invalid_value in (nan, inf, -0.01, 1.01)
    ],
)
def test_posterior_probability_fields_reject_non_finite_or_out_of_range_values(
    field: str,
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"FusedPosterior\.{field} must be finite and within \[0, 1\]",
    ):
        _posterior(**{field: invalid_value})


@pytest.mark.parametrize("invalid_sample", [nan, inf, -0.01, 1.01])
def test_each_posterior_sample_must_be_a_finite_probability(
    invalid_sample: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"FusedPosterior\.samples\[1\] must be finite and within \[0, 1\]",
    ):
        _posterior(samples=(0.5, invalid_sample))


def test_posterior_samples_reject_non_numeric_values() -> None:
    with pytest.raises(
        TypeError,
        match=r"FusedPosterior\.samples\[1\] must be a real number",
    ):
        _posterior(samples=(0.5, "0.6"))  # type: ignore[arg-type]


def test_posterior_rejects_an_unordered_interval() -> None:
    with pytest.raises(
        ValueError,
        match="FusedPosterior.lower must not exceed FusedPosterior.upper",
    ):
        _posterior(lower=0.8, upper=0.2)


@pytest.mark.parametrize("field", ["valid_reviewers", "lineage_count"])
@pytest.mark.parametrize("invalid_value", [True, 1.5, "1"])
def test_posterior_counts_reject_non_integer_values(
    field: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"FusedPosterior\.{field} must be an integer",
    ):
        _posterior(**{field: invalid_value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["valid_reviewers", "lineage_count"])
def test_posterior_counts_reject_negative_values(field: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"FusedPosterior\.{field} must be non-negative",
    ):
        _posterior(**{field: -1})


@pytest.mark.parametrize("invalid_value", [True, "1.5"])
def test_posterior_ess_rejects_non_real_values(invalid_value: object) -> None:
    with pytest.raises(
        TypeError,
        match="FusedPosterior.effective_sample_size must be a real number",
    ):
        _posterior(effective_sample_size=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [
        (nan, "FusedPosterior.effective_sample_size must be finite and non-negative"),
        (inf, "FusedPosterior.effective_sample_size must be finite and non-negative"),
        (-0.1, "FusedPosterior.effective_sample_size must be non-negative"),
    ],
)
def test_posterior_ess_rejects_non_finite_or_negative_values(
    invalid_value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _posterior(effective_sample_size=invalid_value)


def test_posterior_accepts_exactly_zero_empty_panel_metadata() -> None:
    posterior = _posterior(
        valid_reviewers=0,
        lineage_count=0,
        effective_sample_size=0.0,
    )

    assert posterior.valid_reviewers == 0
    assert posterior.lineage_count == 0
    assert posterior.effective_sample_size == 0.0


@pytest.mark.parametrize(
    ("valid_reviewers", "lineage_count", "effective_sample_size"),
    [
        (0, 0, 0.1),
        (0, 1, 0.0),
        (1, 0, 1.0),
        (1, 2, 1.0),
        (2, 1, 0.5),
        (2, 1, 2.1),
    ],
)
def test_posterior_rejects_inconsistent_panel_metadata(
    valid_reviewers: int,
    lineage_count: int,
    effective_sample_size: float,
) -> None:
    with pytest.raises(ValueError, match="FusedPosterior.*metadata"):
        _posterior(
            valid_reviewers=valid_reviewers,
            lineage_count=lineage_count,
            effective_sample_size=effective_sample_size,
        )


def test_package_exposes_its_distribution_version() -> None:
    assert corum.__version__ == version("corum")
