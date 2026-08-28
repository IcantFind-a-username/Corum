from dataclasses import FrozenInstanceError
from math import inf, nan

import numpy as np
import pytest

import corum
from corum.calibration import (
    OBSERVATION_ORDER,
    PairKey,
    ReviewerCalibration,
    ReviewerPairCalibration,
    fit_panel_calibrations,
    fit_reviewer_calibration,
    fit_reviewer_pair_calibration,
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


def _singleton_record(
    reviewer_id: str,
    parent: np.ndarray,
    *,
    observed_counts: np.ndarray | None = None,
    prior_strength: float = 2.0,
) -> ReviewerCalibration:
    counts = (
        np.zeros((2, 3), dtype=np.int64)
        if observed_counts is None
        else np.asarray(observed_counts)
    )
    return ReviewerCalibration(
        reviewer_id=reviewer_id,
        alpha=np.asarray(parent, dtype=float) * prior_strength + counts,
        observed_counts=counts,
        prior_strength=prior_strength,
    )


def _pair_record(
    reviewer_ids: PairKey = ("pair-a", "pair-b"),
    *,
    minimum: int = 1,
) -> ReviewerPairCalibration:
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    counts[0, 0, 0] = minimum
    counts[1, 1, 1] = minimum
    return ReviewerPairCalibration(
        reviewer_ids=reviewer_ids,
        alpha=np.ones((2, 3, 3), dtype=float) + counts,
        observed_counts=counts,
        prior_strength=9.0,
        min_paired_per_truth=minimum,
    )


def test_calibration_api_is_exported_from_the_package() -> None:
    assert corum.ReviewerCalibration is ReviewerCalibration
    assert corum.fit_reviewer_calibration is fit_reviewer_calibration
    assert corum.fit_panel_calibrations is fit_panel_calibrations


def test_pair_calibration_api_is_exported_from_the_package() -> None:
    assert corum.PairKey is PairKey
    assert corum.ReviewerPairCalibration is ReviewerPairCalibration
    assert corum.fit_reviewer_pair_calibration is fit_reviewer_pair_calibration


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
        calibration.alpha.sum(axis=1) - calibration.observed_counts.sum(axis=1),
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


def test_pair_fit_counts_ordered_joint_cells_and_uses_product_parent_prior() -> None:
    parent_a = np.array(
        [[0.60, 0.30, 0.10], [0.20, 0.70, 0.10]],
    )
    parent_b = np.array(
        [[0.20, 0.70, 0.10], [0.40, 0.50, 0.10]],
    )
    calibration_a = _singleton_record(
        "pair-a",
        parent_a,
        observed_counts=np.array([[10, 2, 1], [3, 9, 1]]),
        prior_strength=2.0,
    )
    calibration_b = _singleton_record(
        "pair-b",
        parent_b,
        observed_counts=np.array([[4, 12, 2], [6, 8, 2]]),
        prior_strength=3.0,
    )
    extra = _singleton_record(
        "unrelated",
        np.full((2, 3), 1.0 / 3.0),
    )
    examples = [
        _example("pair-b", Truth.FAIL, Observation.FAIL, case_id="f-1"),
        _example("pair-a", Truth.PASS, Observation.ABSTAIN, case_id="p-2"),
        _example("pair-a", Truth.FAIL, Observation.FAIL, case_id="f-1"),
        _example("pair-b", Truth.PASS, Observation.PASS, case_id="p-2"),
        _example("pair-a", Truth.PASS, Observation.PASS, case_id="p-1"),
        _example("pair-b", Truth.PASS, Observation.FAIL, case_id="p-1"),
        _example("pair-b", Truth.FAIL, Observation.ABSTAIN, case_id="f-2"),
        _example("pair-a", Truth.FAIL, Observation.PASS, case_id="f-2"),
        _example("pair-b", Truth.PASS, Observation.ABSTAIN, case_id="p-3"),
        _example("pair-a", Truth.PASS, Observation.FAIL, case_id="p-3"),
        _example(
            "pair-a",
            Truth.FAIL,
            None,
            case_id="f-invalid",
            state=ExecutionState.TIMEOUT,
        ),
        _example(
            "pair-b",
            Truth.FAIL,
            Observation.PASS,
            case_id="f-invalid",
        ),
        _example(
            "unrelated",
            Truth.PASS,
            Observation.PASS,
            case_id="extra",
        ),
    ]

    calibration = fit_reviewer_pair_calibration(
        ("pair-a", "pair-b"),
        examples,
        reviewer_calibrations={
            "unrelated": extra,
            "pair-b": calibration_b,
            "pair-a": calibration_a,
        },
        prior_strength=9.0,
        min_paired_per_truth=2,
    )

    expected_counts = np.zeros((2, 3, 3), dtype=np.int64)
    expected_counts[0, 0, 1] = 1
    expected_counts[0, 2, 0] = 1
    expected_counts[0, 1, 2] = 1
    expected_counts[1, 1, 1] = 1
    expected_counts[1, 0, 2] = 1
    expected_prior = (
        np.stack(
            [np.outer(parent_a[index], parent_b[index]) for index in range(2)],
        )
        * 9.0
    )
    np.testing.assert_array_equal(calibration.observed_counts, expected_counts)
    np.testing.assert_allclose(
        calibration.alpha - calibration.observed_counts,
        expected_prior,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        (calibration.alpha - calibration.observed_counts).sum(axis=(1, 2)),
        np.array([9.0, 9.0]),
    )
    assert calibration.reviewer_ids == ("pair-a", "pair-b")
    assert calibration.min_paired_per_truth == 2


def test_pair_product_parent_does_not_directly_reuse_singleton_counts() -> None:
    parent_a = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    parent_b = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    examples = [
        _example("pair-a", Truth.PASS, Observation.PASS, case_id="p"),
        _example("pair-b", Truth.PASS, Observation.PASS, case_id="p"),
        _example("pair-a", Truth.FAIL, Observation.FAIL, case_id="f"),
        _example("pair-b", Truth.FAIL, Observation.FAIL, case_id="f"),
    ]
    empty_counts = np.zeros((2, 3), dtype=np.int64)
    large_counts = np.array([[100, 20, 10], [10, 120, 20]], dtype=np.int64)
    first = fit_reviewer_pair_calibration(
        ("pair-a", "pair-b"),
        examples,
        reviewer_calibrations={
            "pair-a": _singleton_record(
                "pair-a", parent_a, observed_counts=empty_counts
            ),
            "pair-b": _singleton_record(
                "pair-b", parent_b, observed_counts=empty_counts
            ),
        },
        min_paired_per_truth=1,
    )
    second = fit_reviewer_pair_calibration(
        ("pair-a", "pair-b"),
        examples,
        reviewer_calibrations={
            "pair-a": _singleton_record(
                "pair-a", parent_a, observed_counts=large_counts
            ),
            "pair-b": _singleton_record(
                "pair-b", parent_b, observed_counts=large_counts
            ),
        },
        min_paired_per_truth=1,
    )

    np.testing.assert_allclose(
        first.alpha - first.observed_counts,
        second.alpha - second.observed_counts,
        rtol=0.0,
        atol=1e-12,
    )


def test_pair_fit_rejects_sparse_truth_rows_at_the_registered_boundary() -> None:
    examples = [
        row
        for truth, prefix, observation in (
            (Truth.PASS, "p", Observation.PASS),
            (Truth.FAIL, "f", Observation.FAIL),
        )
        for index in range(29)
        for row in (
            _example("pair-a", truth, observation, case_id=f"{prefix}-{index}"),
            _example("pair-b", truth, observation, case_id=f"{prefix}-{index}"),
        )
    ]
    calibrations = {
        "pair-a": _singleton_record("pair-a", np.full((2, 3), 1.0 / 3.0)),
        "pair-b": _singleton_record("pair-b", np.full((2, 3), 1.0 / 3.0)),
    }

    with pytest.raises(ValueError, match=r"paired-valid.*30.*PASS.*FAIL"):
        fit_reviewer_pair_calibration(
            ("pair-a", "pair-b"),
            examples,
            reviewer_calibrations=calibrations,
        )

    for truth, prefix, observation in (
        (Truth.PASS, "p", Observation.PASS),
        (Truth.FAIL, "f", Observation.FAIL),
    ):
        examples.extend(
            (
                _example("pair-a", truth, observation, case_id=f"{prefix}-29"),
                _example("pair-b", truth, observation, case_id=f"{prefix}-29"),
            )
        )
    calibration = fit_reviewer_pair_calibration(
        ("pair-a", "pair-b"),
        examples,
        reviewer_calibrations=calibrations,
    )
    np.testing.assert_array_equal(
        calibration.observed_counts.sum(axis=(1, 2)),
        np.array([30, 30]),
    )


@pytest.mark.parametrize(
    "reviewer_ids",
    [
        ("", "pair-b"),
        (" \t", "pair-b"),
        ("pair-a", "pair-a"),
        ("pair-b", "pair-a"),
        (1, "pair-b"),
    ],
)
def test_pair_keys_must_be_nonblank_distinct_strings_in_canonical_order(
    reviewer_ids: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="pair|reviewer"):
        fit_reviewer_pair_calibration(
            reviewer_ids,  # type: ignore[arg-type]
            [],
            reviewer_calibrations={},
            min_paired_per_truth=1,
        )


@pytest.mark.parametrize(
    "reviewer_ids",
    [
        ("", "pair-b"),
        (" \t", "pair-b"),
        ("pair-a", "pair-a"),
        ("pair-b", "pair-a"),
        (1, "pair-b"),
    ],
)
def test_pair_record_cannot_bypass_key_validation(reviewer_ids: object) -> None:
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    counts[0, 0, 0] = 1
    counts[1, 1, 1] = 1

    with pytest.raises((TypeError, ValueError), match="pair|reviewer"):
        ReviewerPairCalibration(
            reviewer_ids=reviewer_ids,  # type: ignore[arg-type]
            alpha=np.ones((2, 3, 3)) + counts,
            observed_counts=counts,
            prior_strength=9.0,
            min_paired_per_truth=1,
        )


def test_pair_fit_requires_matching_singleton_calibration_entries() -> None:
    valid = _singleton_record("pair-a", np.full((2, 3), 1.0 / 3.0))
    mismatched = _singleton_record("other", np.full((2, 3), 1.0 / 3.0))

    with pytest.raises(ValueError, match=r"missing.*pair-b"):
        fit_reviewer_pair_calibration(
            ("pair-a", "pair-b"),
            [],
            reviewer_calibrations={"pair-a": valid},
            min_paired_per_truth=1,
        )
    with pytest.raises(ValueError, match=r"key.*reviewer_id.*pair-b.*other"):
        fit_reviewer_pair_calibration(
            ("pair-a", "pair-b"),
            [],
            reviewer_calibrations={"pair-a": valid, "pair-b": mismatched},
            min_paired_per_truth=1,
        )


@pytest.mark.parametrize("minimum", [True, 0, -1, 1.5])
def test_pair_minimum_must_be_a_positive_integer(minimum: object) -> None:
    with pytest.raises((TypeError, ValueError), match="min_paired_per_truth"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.ones((2, 3, 3)),
            observed_counts=np.zeros((2, 3, 3)),
            prior_strength=9.0,
            min_paired_per_truth=minimum,  # type: ignore[arg-type]
        )


def test_pair_record_rejects_malformed_arrays_prior_mass_and_sparse_counts() -> None:
    with pytest.raises(ValueError, match=r"alpha.*shape"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.ones((2, 9)),
            observed_counts=np.ones((2, 3, 3)),
            prior_strength=9.0,
            min_paired_per_truth=1,
        )
    with pytest.raises(ValueError, match=r"observed_counts.*shape"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.ones((2, 3, 3)),
            observed_counts=np.ones((2, 9)),
            prior_strength=9.0,
            min_paired_per_truth=1,
        )
    with pytest.raises(ValueError, match=r"observed_counts.*integer"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.full((2, 3, 3), 1.5),
            observed_counts=np.full((2, 3, 3), 0.5),
            prior_strength=9.0,
            min_paired_per_truth=1,
        )
    with pytest.raises(ValueError, match=r"inferred prior.*prior_strength"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.full((2, 3, 3), 2.0),
            observed_counts=np.ones((2, 3, 3)),
            prior_strength=8.0,
            min_paired_per_truth=1,
        )
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    counts[0, 0, 0] = 1
    counts[1, 1, 1] = 1
    alpha = np.ones((2, 3, 3)) + counts
    alpha[0, 0, 0] = counts[0, 0, 0]
    with pytest.raises(ValueError, match=r"inferred prior.*strictly positive"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=alpha,
            observed_counts=counts,
            prior_strength=9.0,
            min_paired_per_truth=1,
        )
    sparse_counts = np.zeros((2, 3, 3), dtype=np.int64)
    sparse_counts[0, 0, 0] = 30
    sparse_counts[1, 1, 1] = 29
    with pytest.raises(ValueError, match=r"paired-valid.*30.*FAIL"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.ones((2, 3, 3)) + sparse_counts,
            observed_counts=sparse_counts,
            prior_strength=9.0,
            min_paired_per_truth=30,
        )


@pytest.mark.parametrize("prior_strength", [0.0, -1.0, nan, inf])
def test_pair_record_requires_positive_finite_prior_strength(
    prior_strength: float,
) -> None:
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    counts[0, 0, 0] = 1
    counts[1, 1, 1] = 1
    with pytest.raises(ValueError, match="prior_strength"):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=np.ones((2, 3, 3)) + counts,
            observed_counts=counts,
            prior_strength=prior_strength,
            min_paired_per_truth=1,
        )


@pytest.mark.parametrize(
    ("alpha_value", "counts_value", "message"),
    [
        (0.0, 0.0, "alpha.*strictly positive"),
        (nan, 0.0, "alpha.*finite"),
        (1.0, -1.0, "observed_counts.*non-negative"),
        (1.0, nan, "observed_counts.*finite"),
    ],
)
def test_pair_record_rejects_impossible_numeric_entries(
    alpha_value: float,
    counts_value: float,
    message: str,
) -> None:
    alpha = np.ones((2, 3, 3), dtype=float)
    counts = np.zeros((2, 3, 3), dtype=float)
    counts[0, 0, 0] = 1
    counts[1, 1, 1] = 1
    alpha += counts
    alpha[0, 0, 1] = alpha_value
    counts[0, 0, 1] = counts_value

    with pytest.raises(ValueError, match=message):
        ReviewerPairCalibration(
            reviewer_ids=("pair-a", "pair-b"),
            alpha=alpha,
            observed_counts=counts,
            prior_strength=9.0,
            min_paired_per_truth=1,
        )


def test_pair_likelihood_methods_are_seeded_normalized_and_irreversibly_read_only() -> (
    None
):
    calibration = _pair_record()

    means = calibration.mean_likelihoods()
    first = calibration.sample_likelihoods(7, np.random.default_rng(42))
    second = calibration.sample_likelihoods(7, np.random.default_rng(42))
    reference_rng = np.random.default_rng(42)
    reference = np.stack(
        [
            reference_rng.dirichlet(
                calibration.alpha[truth_index].reshape(-1),
                size=7,
            ).reshape(7, 3, 3)
            for truth_index in range(2)
        ],
        axis=1,
    )

    assert means.shape == (2, 3, 3)
    assert first.shape == (7, 2, 3, 3)
    np.testing.assert_allclose(
        means,
        calibration.alpha / calibration.alpha.sum(axis=(1, 2), keepdims=True),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(means.sum(axis=(1, 2)), np.ones(2))
    np.testing.assert_allclose(first.sum(axis=(2, 3)), np.ones((7, 2)))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, reference)
    for array in (
        calibration.alpha,
        calibration.observed_counts,
        means,
        first,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)

    empirical = calibration.sample_likelihoods(20_000, np.random.default_rng(91))
    np.testing.assert_allclose(empirical.mean(axis=0), means, rtol=0.0, atol=0.01)


@pytest.mark.parametrize("draws", [True, 0, -1, 1.5])
def test_pair_likelihood_sampling_rejects_invalid_draw_counts(draws: object) -> None:
    with pytest.raises((TypeError, ValueError), match="draws"):
        _pair_record().sample_likelihoods(
            draws,  # type: ignore[arg-type]
            np.random.default_rng(3),
        )


def test_pair_record_defensively_owns_inputs_and_is_frozen() -> None:
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    counts[0, 0, 0] = 1
    counts[1, 1, 1] = 1
    alpha = np.ones((2, 3, 3)) + counts
    calibration = ReviewerPairCalibration(
        reviewer_ids=("pair-a", "pair-b"),
        alpha=alpha,
        observed_counts=counts,
        prior_strength=9.0,
        min_paired_per_truth=1,
    )

    alpha[0, 0, 0] = 99.0
    counts[0, 0, 0] = 99

    assert calibration.alpha[0, 0, 0] == 2.0
    assert calibration.observed_counts[0, 0, 0] == 1
    assert not hasattr(calibration, "__dict__")
    with pytest.raises(FrozenInstanceError):
        calibration.reviewer_ids = ("x", "y")


def test_pair_fit_rejects_duplicate_rows_and_conflicting_case_truth() -> None:
    calibrations = {
        reviewer_id: _singleton_record(
            reviewer_id,
            np.full((2, 3), 1.0 / 3.0),
        )
        for reviewer_id in ("pair-a", "pair-b")
    }
    duplicate = [
        _example("pair-a", Truth.PASS, Observation.PASS, case_id="case"),
        _example("pair-a", Truth.PASS, Observation.FAIL, case_id="case"),
    ]
    conflict = [
        _example("pair-a", Truth.PASS, Observation.PASS, case_id="case"),
        _example("pair-b", Truth.FAIL, Observation.FAIL, case_id="case"),
    ]

    with pytest.raises(ValueError, match="duplicate reviewer-case"):
        fit_reviewer_pair_calibration(
            ("pair-a", "pair-b"),
            duplicate,
            reviewer_calibrations=calibrations,
            min_paired_per_truth=1,
        )
    with pytest.raises(ValueError, match="conflicting truth"):
        fit_reviewer_pair_calibration(
            ("pair-a", "pair-b"),
            conflict,
            reviewer_calibrations=calibrations,
            min_paired_per_truth=1,
        )
