from dataclasses import FrozenInstanceError
from math import inf, nan

import numpy as np
import pytest

import corum
from corum.calibration import (
    OBSERVATION_ORDER,
    ReviewerCalibration,
    fit_panel_calibrations,
    fit_reviewer_calibration,
)
from corum.models import (
    CalibrationExample,
    ExecutionState,
    Observation,
    Review,
    Reviewer,
    Truth,
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
        review=Review(
            case_id=case_id,
            reviewer_id=reviewer_id,
            observation=observation,
            state=state,
        ),
    )


def _reviewer(reviewer_id: str) -> Reviewer:
    return Reviewer(
        reviewer_id=reviewer_id,
        vendor="vendor",
        family="family",
        lineage="lineage",
    )


def test_calibration_api_is_exported_from_the_package() -> None:
    assert corum.ReviewerCalibration is ReviewerCalibration
    assert corum.fit_reviewer_calibration is fit_reviewer_calibration
    assert corum.fit_panel_calibrations is fit_panel_calibrations


def test_valid_observations_are_counted_in_declared_row_and_column_order() -> None:
    examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-2"),
        _example("reviewer-1", Truth.PASS, Observation.FAIL, case_id="case-3"),
        _example("reviewer-1", Truth.PASS, Observation.ABSTAIN, case_id="case-4"),
        _example("reviewer-1", Truth.FAIL, Observation.FAIL, case_id="case-5"),
        _example("reviewer-1", Truth.FAIL, Observation.FAIL, case_id="case-6"),
        _example("reviewer-1", Truth.FAIL, Observation.ABSTAIN, case_id="case-7"),
        _example("reviewer-2", Truth.FAIL, Observation.PASS, case_id="case-8"),
    ]

    calibration = fit_reviewer_calibration("reviewer-1", examples)

    assert OBSERVATION_ORDER == (
        Observation.PASS,
        Observation.FAIL,
        Observation.ABSTAIN,
    )
    np.testing.assert_array_equal(
        calibration.observed_counts,
        np.array([[2, 1, 1], [0, 2, 1]]),
    )


def test_parent_prior_is_normalized_and_scaled_to_declared_strength() -> None:
    examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("reviewer-1", Truth.FAIL, Observation.ABSTAIN, case_id="case-2"),
    ]
    parent_prior = np.array([[6.0, 3.0, 1.0], [2.0, 2.0, 6.0]])

    calibration = fit_reviewer_calibration(
        "reviewer-1",
        examples,
        parent_prior=parent_prior,
        prior_strength=2.0,
    )

    np.testing.assert_allclose(
        calibration.alpha,
        np.array([[2.2, 0.6, 0.2], [0.4, 0.4, 2.2]]),
    )
    np.testing.assert_allclose(
        calibration.alpha.sum(axis=1)
        - calibration.observed_counts.sum(axis=1),
        np.array([2.0, 2.0]),
    )


def test_mean_likelihoods_are_normalized_row_wise() -> None:
    calibration = fit_reviewer_calibration(
        "reviewer-1",
        [
            _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
            _example("reviewer-1", Truth.FAIL, Observation.FAIL, case_id="case-2"),
        ],
    )

    likelihoods = calibration.mean_likelihoods()

    assert likelihoods.shape == (2, 3)
    np.testing.assert_allclose(likelihoods.sum(axis=1), np.ones(2))


@pytest.mark.parametrize(
    "state",
    [
        ExecutionState.TIMEOUT,
        ExecutionState.INVALID,
        ExecutionState.REFUSAL,
        ExecutionState.NOT_CALLED,
    ],
)
def test_non_valid_executions_are_not_semantic_observations(
    state: ExecutionState,
) -> None:
    calibration = fit_reviewer_calibration(
        "reviewer-1",
        [
            _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
            _example(
                "reviewer-1",
                Truth.PASS,
                None,
                case_id="case-2",
                state=state,
            ),
        ],
    )

    np.testing.assert_array_equal(
        calibration.observed_counts,
        np.array([[1, 0, 0], [0, 0, 0]]),
    )


def test_panel_rejects_examples_from_unknown_reviewers() -> None:
    with pytest.raises(ValueError, match="unknown reviewer.*reviewer-2"):
        fit_panel_calibrations(
            [_reviewer("reviewer-1")],
            [
                _example(
                    "reviewer-2",
                    Truth.PASS,
                    Observation.PASS,
                    case_id="case-1",
                )
            ],
        )


def test_direct_fit_rejects_duplicate_reviewer_case_examples() -> None:
    examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("reviewer-1", Truth.PASS, Observation.FAIL, case_id="case-1"),
    ]

    with pytest.raises(
        ValueError,
        match=r"duplicate reviewer-case key.*reviewer-1.*case-1",
    ):
        fit_reviewer_calibration("reviewer-1", examples)


def test_panel_fit_rejects_duplicate_reviewer_case_examples() -> None:
    examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("reviewer-1", Truth.PASS, Observation.FAIL, case_id="case-1"),
    ]

    with pytest.raises(
        ValueError,
        match=r"duplicate reviewer-case key.*reviewer-1.*case-1",
    ):
        fit_panel_calibrations([_reviewer("reviewer-1")], examples)


def test_panel_fit_rejects_duplicate_reviewer_ids() -> None:
    with pytest.raises(ValueError, match=r"duplicate reviewer_id.*reviewer-1"):
        fit_panel_calibrations(
            [_reviewer("reviewer-1"), _reviewer("reviewer-1")],
            [],
        )


def test_panel_fit_rejects_conflicting_truth_for_the_same_case() -> None:
    examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("reviewer-2", Truth.FAIL, Observation.FAIL, case_id="case-1"),
    ]

    with pytest.raises(ValueError, match=r"conflicting truth.*case-1"):
        fit_panel_calibrations(
            [_reviewer("reviewer-1"), _reviewer("reviewer-2")],
            examples,
        )


def test_reviewer_without_data_shrinks_to_smoothed_pooled_parent() -> None:
    examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-2"),
        _example("reviewer-1", Truth.PASS, Observation.ABSTAIN, case_id="case-3"),
        _example("reviewer-1", Truth.FAIL, Observation.FAIL, case_id="case-4"),
    ]

    calibrations = fit_panel_calibrations(
        [_reviewer("reviewer-1"), _reviewer("reviewer-2")],
        examples,
        prior_strength=3.0,
    )

    np.testing.assert_allclose(
        calibrations["reviewer-2"].mean_likelihoods(),
        np.array(
            [
                [0.5, 1.0 / 6.0, 1.0 / 3.0],
                [0.25, 0.5, 0.25],
            ]
        ),
    )
    np.testing.assert_allclose(
        calibrations["reviewer-2"].alpha.sum(axis=1),
        np.array([3.0, 3.0]),
    )


def test_seeded_likelihood_samples_are_deterministic_and_keep_draw_axis() -> None:
    calibration = fit_reviewer_calibration(
        "reviewer-1",
        [
            _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="case-1"),
            _example("reviewer-1", Truth.FAIL, Observation.FAIL, case_id="case-2"),
        ],
    )

    first = calibration.sample_likelihoods(7, np.random.default_rng(42))
    second = calibration.sample_likelihoods(7, np.random.default_rng(42))

    assert first.shape == (7, 2, 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=2), np.ones((7, 2)))


@pytest.mark.parametrize(
    "parent_prior",
    [
        np.ones((3, 2)),
        np.array([[nan, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        np.array([[inf, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        np.array([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    ],
)
def test_malformed_parent_prior_is_rejected(parent_prior: np.ndarray) -> None:
    with pytest.raises(ValueError, match="parent_prior"):
        fit_reviewer_calibration(
            "reviewer-1",
            [],
            parent_prior=parent_prior,
        )


@pytest.mark.parametrize("prior_strength", [0.0, -1.0, nan, inf])
def test_prior_strength_must_be_positive_and_finite(
    prior_strength: float,
) -> None:
    with pytest.raises(ValueError, match="prior_strength"):
        fit_reviewer_calibration(
            "reviewer-1",
            [],
            prior_strength=prior_strength,
        )


def test_reviewer_id_must_not_be_blank() -> None:
    with pytest.raises(ValueError, match="reviewer_id"):
        fit_reviewer_calibration(" \t", [])


def test_sparse_data_still_produces_strictly_positive_alphas() -> None:
    calibration = fit_reviewer_calibration(
        "reviewer-1",
        [
            _example(
                "reviewer-1",
                Truth.PASS,
                Observation.PASS,
                case_id="case-1",
            )
        ],
    )

    assert np.all(calibration.alpha > 0.0)


def test_calibration_record_rejects_wrong_array_shapes_and_nonpositive_alpha() -> None:
    with pytest.raises(ValueError, match="alpha.*shape"):
        ReviewerCalibration(
            reviewer_id="reviewer-1",
            alpha=np.ones((3, 2)),
            observed_counts=np.zeros((2, 3)),
            prior_strength=1.5,
        )

    with pytest.raises(ValueError, match="observed_counts.*shape"):
        ReviewerCalibration(
            reviewer_id="reviewer-1",
            alpha=np.ones((2, 3)),
            observed_counts=np.zeros((3, 2)),
            prior_strength=1.5,
        )

    with pytest.raises(ValueError, match="alpha.*positive"):
        ReviewerCalibration(
            reviewer_id="reviewer-1",
            alpha=np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
            observed_counts=np.zeros((2, 3)),
            prior_strength=1.5,
        )


def test_calibration_record_rejects_fractional_observed_counts() -> None:
    observed_counts = np.zeros((2, 3))
    observed_counts[0, 0] = 0.5

    with pytest.raises(ValueError, match="observed_counts.*integer"):
        ReviewerCalibration(
            reviewer_id="reviewer-1",
            alpha=observed_counts + 0.5,
            observed_counts=observed_counts,
            prior_strength=1.5,
        )


def test_calibration_record_rejects_nonpositive_inferred_prior_alpha() -> None:
    with pytest.raises(ValueError, match="inferred prior.*positive"):
        ReviewerCalibration(
            reviewer_id="reviewer-1",
            alpha=np.array([[1.0, 0.75, 0.75], [0.5, 0.5, 0.5]]),
            observed_counts=np.array([[1, 0, 0], [0, 0, 0]]),
            prior_strength=1.5,
        )


def test_calibration_record_rejects_prior_mass_that_disagrees_with_strength() -> None:
    with pytest.raises(ValueError, match="inferred prior.*prior_strength"):
        ReviewerCalibration(
            reviewer_id="reviewer-1",
            alpha=np.ones((2, 3)),
            observed_counts=np.zeros((2, 3)),
            prior_strength=1.5,
        )


def test_frozen_calibration_defensively_owns_read_only_arrays() -> None:
    source_alpha = np.full((2, 3), 0.5)
    source_counts = np.zeros((2, 3))
    calibration = ReviewerCalibration(
        reviewer_id="reviewer-1",
        alpha=source_alpha,
        observed_counts=source_counts,
        prior_strength=1.5,
    )

    source_alpha[0, 0] = 99.0
    source_counts[0, 0] = 99.0

    assert calibration.alpha[0, 0] == 0.5
    assert calibration.observed_counts[0, 0] == 0.0
    assert not calibration.alpha.flags.writeable
    assert not calibration.observed_counts.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        calibration.alpha[0, 0] = 2.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        calibration.alpha.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        calibration.observed_counts.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        calibration.alpha = np.ones((2, 3))


def test_likelihood_methods_return_read_only_arrays() -> None:
    calibration = fit_reviewer_calibration("reviewer-1", [])

    means = calibration.mean_likelihoods()
    samples = calibration.sample_likelihoods(
        2,
        np.random.default_rng(7),
    )

    assert not means.flags.writeable
    assert not samples.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        means.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        samples.setflags(write=True)


def test_cold_start_interval_is_wider_than_same_rate_large_sample_interval() -> None:
    cold_examples = [
        _example("reviewer-1", Truth.PASS, Observation.PASS, case_id="cold-1"),
        _example("reviewer-1", Truth.PASS, Observation.FAIL, case_id="cold-2"),
        _example("reviewer-1", Truth.PASS, Observation.ABSTAIN, case_id="cold-3"),
    ]
    large_examples = [
        _example(
            "reviewer-1",
            Truth.PASS,
            observation,
            case_id=f"large-{repetition}-{observation.value}",
        )
        for repetition in range(100)
        for observation in (
            Observation.PASS,
            Observation.FAIL,
            Observation.ABSTAIN,
        )
    ]
    cold = fit_reviewer_calibration("reviewer-1", cold_examples)
    large = fit_reviewer_calibration("reviewer-1", large_examples)

    cold_pass = cold.sample_likelihoods(
        20_000,
        np.random.default_rng(123),
    )[:, 0, 0]
    large_pass = large.sample_likelihoods(
        20_000,
        np.random.default_rng(123),
    )[:, 0, 0]
    cold_width = np.quantile(cold_pass, 0.975) - np.quantile(cold_pass, 0.025)
    large_width = np.quantile(large_pass, 0.975) - np.quantile(
        large_pass,
        0.025,
    )

    assert cold_width > large_width
