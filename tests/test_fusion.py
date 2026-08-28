from collections.abc import Mapping

import numpy as np
import pytest

from corum.calibration import (
    OBSERVATION_ORDER,
    PairKey,
    ReviewerCalibration,
    ReviewerPairCalibration,
)
from corum.dependence import DependenceModel
from corum.fusion import (
    BatchFusedPosterior,
    FusionContext,
    build_fusion_context,
    fuse_known_likelihoods,
    fuse_known_pair_likelihoods,
    fuse_review_matrix,
    fuse_reviews,
)
from corum.models import ExecutionState, Observation, Review


def _dependence(
    reviewer_ids: tuple[str, ...],
    *,
    correlation: np.ndarray | None = None,
    lineages: Mapping[str, str] | None = None,
) -> DependenceModel:
    reviewer_count = len(reviewer_ids)
    return DependenceModel(
        reviewer_ids=reviewer_ids,
        correlation=(
            np.eye(reviewer_count, dtype=float) if correlation is None else correlation
        ),
        lineage_by_reviewer=(
            {reviewer_id: f"lineage-{reviewer_id}" for reviewer_id in reviewer_ids}
            if lineages is None
            else lineages
        ),
    )


def _calibration(
    reviewer_id: str,
    likelihoods: np.ndarray,
    *,
    strength: float = 40.0,
) -> ReviewerCalibration:
    return ReviewerCalibration(
        reviewer_id=reviewer_id,
        alpha=np.asarray(likelihoods, dtype=float) * strength,
        observed_counts=np.zeros((2, 3), dtype=np.int64),
        prior_strength=strength,
    )


def _joint_likelihoods() -> np.ndarray:
    return np.array(
        [
            [
                [0.40, 0.24, 0.06],
                [0.08, 0.10, 0.02],
                [0.04, 0.04, 0.02],
            ],
            [
                [0.06, 0.08, 0.02],
                [0.12, 0.54, 0.08],
                [0.02, 0.07, 0.01],
            ],
        ],
        dtype=float,
    )


def _pair_calibration(
    reviewer_ids: PairKey,
    likelihoods: np.ndarray | None = None,
    *,
    strength: float = 90.0,
) -> ReviewerPairCalibration:
    joint = _joint_likelihoods() if likelihoods is None else likelihoods
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    counts[:, 0, 0] = 1
    return ReviewerPairCalibration(
        reviewer_ids=reviewer_ids,
        alpha=np.asarray(joint, dtype=float) * strength + counts,
        observed_counts=counts,
        prior_strength=strength,
        min_paired_per_truth=1,
    )


def _fixed_context(
    *,
    reviewer_ids: tuple[str, ...] = ("r1", "r2", "r3"),
    correlation: np.ndarray | None = None,
    lineages: Mapping[str, str] | None = None,
    draws: int = 32,
    prior_pass: float = 0.5,
    pair_likelihoods: Mapping[PairKey, np.ndarray] | None = None,
) -> FusionContext:
    base_likelihoods = {
        "r1": np.array([[0.80, 0.15, 0.05], [0.15, 0.80, 0.05]]),
        "r2": np.array([[0.70, 0.20, 0.10], [0.20, 0.70, 0.10]]),
        "r3": np.array([[0.75, 0.15, 0.10], [0.10, 0.80, 0.10]]),
        "r4": np.array([[0.65, 0.25, 0.10], [0.25, 0.65, 0.10]]),
    }
    likelihood_draws = {
        reviewer_id: np.repeat(
            base_likelihoods[reviewer_id][None, :, :],
            draws,
            axis=0,
        )
        for reviewer_id in reviewer_ids
    }
    dependence = _dependence(
        reviewer_ids,
        correlation=correlation,
        lineages=lineages,
    )
    return FusionContext(
        likelihood_draws=likelihood_draws,
        dependence=dependence,
        lineage_by_reviewer=dependence.lineage_by_reviewer,
        prior_pass=prior_pass,
        credible_mass=0.95,
        pair_likelihood_draws=(
            {}
            if pair_likelihoods is None
            else {
                pair: np.repeat(joint[None, :, :, :], draws, axis=0)
                for pair, joint in pair_likelihoods.items()
            }
        ),
    )


def _review(
    case_id: str,
    reviewer_id: str,
    observation: Observation | None,
    state: ExecutionState,
) -> Review:
    return Review(
        case_id=case_id,
        reviewer_id=reviewer_id,
        observation=observation,
        state=state,
    )


def test_known_likelihood_oracle_matches_hand_calculated_bayes_result() -> None:
    observations = {"r1": Observation.PASS, "r2": Observation.FAIL}
    likelihoods = {
        "r1": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
        "r2": np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]),
    }

    posterior = fuse_known_likelihoods(
        observations,
        likelihoods,
        {"r1": 1.0, "r2": 1.0},
        prior_pass=0.6,
    )

    expected = (0.6 * 0.8 * 0.3) / ((0.6 * 0.8 * 0.3) + (0.4 * 0.2 * 0.7))
    assert posterior == pytest.approx(expected, abs=1e-12)


def test_monte_carlo_context_matches_oracle_within_sampling_error() -> None:
    likelihoods = np.array(
        [[0.80, 0.15, 0.05], [0.20, 0.75, 0.05]],
        dtype=float,
    )
    context = build_fusion_context(
        {"r1": _calibration("r1", likelihoods, strength=100_000.0)},
        _dependence(("r1",)),
        prior_pass=0.5,
        draws=8_192,
        seed=43,
    )
    result = fuse_reviews(
        (
            _review(
                "case-1",
                "r1",
                Observation.PASS,
                ExecutionState.VALID,
            ),
        ),
        context,
    )

    assert result is not None
    oracle = fuse_known_likelihoods(
        {"r1": Observation.PASS},
        {"r1": likelihoods},
        {"r1": 1.0},
        prior_pass=0.5,
    )
    sample_standard_error = float(np.std(result.samples, ddof=1)) / np.sqrt(
        len(result.samples)
    )
    assert abs(result.pass_probability - oracle) <= 6.0 * sample_standard_error


def test_known_likelihood_oracle_is_stable_for_extreme_evidence() -> None:
    reviewer_ids = tuple(f"r{index}" for index in range(100))
    observations = {reviewer_id: Observation.PASS for reviewer_id in reviewer_ids}
    likelihoods = {
        reviewer_id: np.array([[1.0 - 1e-12, 5e-13, 5e-13], [1e-12, 0.9, 0.1 - 1e-12]])
        for reviewer_id in reviewer_ids
    }

    posterior = fuse_known_likelihoods(
        observations,
        likelihoods,
        {reviewer_id: 1.0 for reviewer_id in reviewer_ids},
        prior_pass=0.5,
    )

    assert np.isfinite(posterior)
    assert posterior > 1.0 - 1e-12


def test_known_likelihood_oracle_is_permutation_invariant() -> None:
    likelihoods = {
        "r1": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
        "r2": np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]),
    }
    first = fuse_known_likelihoods(
        {"r1": Observation.PASS, "r2": Observation.FAIL},
        likelihoods,
        {"r1": 0.75, "r2": 1.0},
        prior_pass=0.4,
    )
    second = fuse_known_likelihoods(
        {"r2": Observation.FAIL, "r1": Observation.PASS},
        {"r2": likelihoods["r2"], "r1": likelihoods["r1"]},
        {"r2": 1.0, "r1": 0.75},
        prior_pass=0.4,
    )

    assert first == second


def test_build_context_samples_once_reproducibly_and_ignores_mapping_order() -> None:
    reviewer_ids = ("r1", "r2")
    likelihoods = {
        "r1": np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        "r2": np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]),
    }
    calibrations = {
        reviewer_id: _calibration(reviewer_id, likelihoods[reviewer_id])
        for reviewer_id in reviewer_ids
    }
    dependence = _dependence(reviewer_ids)

    first = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.4,
        draws=128,
        credible_mass=0.9,
        seed=17,
    )
    second = build_fusion_context(
        {"r2": calibrations["r2"], "r1": calibrations["r1"]},
        dependence,
        prior_pass=0.4,
        draws=128,
        credible_mass=0.9,
        seed=17,
    )
    different_seed = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.4,
        draws=128,
        credible_mass=0.9,
        seed=18,
    )

    for reviewer_id in reviewer_ids:
        assert (
            first.likelihood_draws[reviewer_id].tobytes()
            == second.likelihood_draws[reviewer_id].tobytes()
        )
        assert (
            first.likelihood_draws[reviewer_id].tobytes()
            != different_seed.likelihood_draws[reviewer_id].tobytes()
        )


def test_context_defensively_owns_read_only_likelihood_draws_and_lineages() -> None:
    draws = np.ones((4, 2, 3), dtype=float) / 3.0
    source_mapping = {"r1": draws}
    source_lineages = {"r1": "lineage-1"}
    context = FusionContext(
        likelihood_draws=source_mapping,
        dependence=_dependence(("r1",), lineages=source_lineages),
        lineage_by_reviewer=source_lineages,
        prior_pass=0.5,
        credible_mass=0.95,
    )

    draws[0, 0, 0] = 0.9
    source_mapping["r2"] = np.ones((4, 2, 3)) / 3.0
    source_lineages["r1"] = "changed"

    assert context.likelihood_draws["r1"][0, 0, 0] == pytest.approx(1.0 / 3.0)
    assert tuple(context.likelihood_draws) == ("r1",)
    assert context.lineage_by_reviewer["r1"] == "lineage-1"
    with pytest.raises(ValueError):
        context.likelihood_draws["r1"][0, 0, 0] = 0.5
    with pytest.raises(TypeError):
        context.lineage_by_reviewer["r1"] = "mutated"  # type: ignore[index]


def test_context_rejects_an_empty_reviewer_registry() -> None:
    dependence = _dependence(())

    with pytest.raises(ValueError, match="at least one reviewer"):
        FusionContext(
            likelihood_draws={},
            dependence=dependence,
            lineage_by_reviewer={},
            prior_pass=0.5,
            credible_mass=0.95,
        )


def test_context_rejects_unrepresentable_probability_metadata() -> None:
    with pytest.raises(
        ValueError,
        match="prior_pass must be finite and representable as a float",
    ):
        FusionContext(
            likelihood_draws={"r1": np.ones((4, 2, 3)) / 3.0},
            dependence=_dependence(("r1",)),
            lineage_by_reviewer={"r1": "lineage-r1"},
            prior_pass=10**400,
            credible_mass=0.95,
        )


def test_known_fusion_rejects_unrepresentable_numeric_inputs() -> None:
    valid_likelihood = np.array(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
    )
    with pytest.raises(
        ValueError,
        match=r"weights\['r1'\] must be finite and representable as a float",
    ):
        fuse_known_likelihoods(
            {"r1": Observation.PASS},
            {"r1": valid_likelihood},
            {"r1": 10**400},
            prior_pass=0.5,
        )
    with pytest.raises(ValueError, match="must be a numeric array"):
        fuse_known_likelihoods(
            {"r1": Observation.PASS},
            {
                "r1": np.array(
                    [[10**400, 0, 0], [0, 1, 0]],
                    dtype=object,
                )
            },
            {"r1": 1.0},
            prior_pass=0.5,
        )


def test_build_context_requires_exactly_the_dependence_reviewers() -> None:
    calibration = _calibration(
        "r1",
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
    )

    with pytest.raises(
        ValueError,
        match="calibration reviewer IDs must match dependence reviewer IDs",
    ):
        build_fusion_context(
            {"r1": calibration},
            _dependence(("r1", "r2")),
            prior_pass=0.5,
            draws=16,
            seed=1,
        )


def test_scalar_fusion_reuses_common_parameter_draws_across_cases() -> None:
    reviewer_ids = ("r1", "r2")
    calibrations = {
        "r1": _calibration("r1", np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])),
        "r2": _calibration("r2", np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])),
    }
    context = build_fusion_context(
        calibrations,
        _dependence(reviewer_ids),
        prior_pass=0.5,
        draws=64,
        seed=11,
    )
    first = fuse_reviews(
        [
            _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-1", "r2", Observation.FAIL, ExecutionState.VALID),
        ],
        context,
    )
    second = fuse_reviews(
        [
            _review("case-2", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-2", "r2", Observation.FAIL, ExecutionState.VALID),
        ],
        context,
    )

    assert first is not None
    assert second is not None
    assert first.samples == second.samples


def test_correlated_clones_are_less_confident_than_independent_duplicates() -> None:
    reviewer_ids = ("r1", "r2")
    independent = _fixed_context(reviewer_ids=reviewer_ids)
    correlated = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=np.ones((2, 2), dtype=float),
        lineages={"r1": "clone", "r2": "clone"},
    )
    reviews = [
        _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
        _review("case-1", "r2", Observation.PASS, ExecutionState.VALID),
    ]

    independent_posterior = fuse_reviews(reviews, independent)
    correlated_posterior = fuse_reviews(reviews, correlated)

    assert independent_posterior is not None
    assert correlated_posterior is not None
    assert (
        independent_posterior.pass_probability > correlated_posterior.pass_probability
    )
    assert correlated_posterior.effective_sample_size == pytest.approx(1.0)
    assert correlated_posterior.lineage_count == 1


def test_single_queried_reviewer_keeps_full_weight_despite_unqueried_clone() -> None:
    context = _fixed_context(
        reviewer_ids=("r1", "r2"),
        correlation=np.ones((2, 2), dtype=float),
        lineages={"r1": "clone", "r2": "clone"},
    )
    posterior = fuse_reviews(
        [_review("case-1", "r1", Observation.PASS, ExecutionState.VALID)],
        context,
    )
    expected = fuse_known_likelihoods(
        {"r1": Observation.PASS},
        {"r1": context.likelihood_draws["r1"][0]},
        {"r1": 1.0},
        prior_pass=0.5,
    )

    assert posterior is not None
    assert posterior.pass_probability == expected
    assert posterior.effective_sample_size == pytest.approx(1.0)


def test_missing_executions_contribute_nothing_to_scalar_fusion() -> None:
    context = _fixed_context(reviewer_ids=("r1", "r2"))
    with_timeout = fuse_reviews(
        [
            _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-1", "r2", None, ExecutionState.TIMEOUT),
        ],
        context,
    )
    valid_only = fuse_reviews(
        [_review("case-1", "r1", Observation.PASS, ExecutionState.VALID)],
        context,
    )

    assert with_timeout == valid_only


def test_scalar_fusion_rejects_duplicate_reviewers_and_mixed_case_ids() -> None:
    context = _fixed_context(reviewer_ids=("r1", "r2"))
    with pytest.raises(ValueError, match="duplicate reviewer IDs"):
        fuse_reviews(
            [
                _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
                _review("case-1", "r1", Observation.FAIL, ExecutionState.VALID),
            ],
            context,
        )
    with pytest.raises(ValueError, match="one case_id"):
        fuse_reviews(
            [
                _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
                _review("case-2", "r2", Observation.FAIL, ExecutionState.VALID),
            ],
            context,
        )


def test_scalar_fusion_rejects_reviewer_without_calibration() -> None:
    context = _fixed_context(reviewer_ids=("r1",))

    with pytest.raises(ValueError, match="unknown reviewer IDs: r2"):
        fuse_reviews(
            [_review("case-1", "r2", Observation.PASS, ExecutionState.VALID)],
            context,
        )


def test_all_invalid_scalar_panel_returns_no_posterior() -> None:
    context = _fixed_context(reviewer_ids=("r1", "r2"))

    posterior = fuse_reviews(
        [
            _review("case-1", "r1", None, ExecutionState.INVALID),
            _review("case-1", "r2", None, ExecutionState.NOT_CALLED),
        ],
        context,
    )

    assert posterior is None


def test_all_abstain_panel_preserves_evidence_but_widens_decision_bounds() -> None:
    reviewer_ids = ("r1", "r2")
    abstain_predicts_pass = np.array(
        [[0.05, 0.05, 0.90], [0.49, 0.50, 0.01]],
    )
    likelihood_draws = {
        reviewer_id: np.repeat(
            abstain_predicts_pass[None, :, :],
            32,
            axis=0,
        )
        for reviewer_id in reviewer_ids
    }
    dependence = _dependence(reviewer_ids)
    context = FusionContext(
        likelihood_draws=likelihood_draws,
        dependence=dependence,
        lineage_by_reviewer=dependence.lineage_by_reviewer,
        prior_pass=0.5,
        credible_mass=0.95,
    )

    scalar = fuse_reviews(
        [
            _review("case-1", reviewer_id, Observation.ABSTAIN, ExecutionState.VALID)
            for reviewer_id in reviewer_ids
        ],
        context,
    )
    batch = fuse_review_matrix(
        np.full((1, 2), 2, dtype=np.int64),
        np.ones((1, 2), dtype=bool),
        reviewer_ids,
        context,
    )

    assert scalar is not None
    assert scalar.pass_probability > 0.99
    assert scalar.lower == 0.0
    assert scalar.upper == 1.0
    assert scalar.valid_reviewers == 2
    assert scalar.lineage_count == 2
    assert scalar.effective_sample_size == 2.0
    assert batch.pass_probability[0] == scalar.pass_probability
    assert batch.lower[0] == 0.0
    assert batch.upper[0] == 1.0
    assert batch.valid_reviewers[0] == 2
    assert batch.lineage_count[0] == 2
    assert batch.effective_sample_size[0] == 2.0


def test_scalar_and_matrix_paths_are_byte_identical_for_mixed_states() -> None:
    context = _fixed_context()
    reviewer_ids = ("r1", "r2", "r3")
    observations = np.array(
        [
            [0, 99, -1],
            [99, 1, 99],
            [99, 99, 99],
            [2, 0, -1],
        ],
        dtype=np.int64,
    )
    valid_mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    scalar_rows = [
        [
            _review("case-0", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-0", "r2", None, ExecutionState.TIMEOUT),
            _review("case-0", "r3", None, ExecutionState.INVALID),
        ],
        [
            _review("case-1", "r1", None, ExecutionState.REFUSAL),
            _review("case-1", "r2", Observation.FAIL, ExecutionState.VALID),
            _review("case-1", "r3", None, ExecutionState.NOT_CALLED),
        ],
        [
            _review("case-2", "r1", None, ExecutionState.TIMEOUT),
            _review("case-2", "r2", None, ExecutionState.REFUSAL),
            _review("case-2", "r3", None, ExecutionState.NOT_CALLED),
        ],
        [
            _review("case-3", "r1", Observation.ABSTAIN, ExecutionState.VALID),
            _review("case-3", "r2", Observation.PASS, ExecutionState.VALID),
            _review("case-3", "r3", None, ExecutionState.TIMEOUT),
        ],
    ]

    batch = fuse_review_matrix(
        observations,
        valid_mask,
        reviewer_ids,
        context,
        chunk_size=2,
    )

    for row_index, reviews in enumerate(scalar_rows):
        scalar = fuse_reviews(reviews, context)
        if scalar is None:
            assert np.isnan(batch.pass_probability[row_index])
            assert np.isnan(batch.lower[row_index])
            assert np.isnan(batch.upper[row_index])
            assert batch.valid_reviewers[row_index] == 0
            assert batch.lineage_count[row_index] == 0
            assert batch.effective_sample_size[row_index] == 0.0
            continue
        assert (
            np.asarray(scalar.pass_probability).tobytes()
            == batch.pass_probability[row_index].tobytes()
        )
        assert np.asarray(scalar.lower).tobytes() == batch.lower[row_index].tobytes()
        assert np.asarray(scalar.upper).tobytes() == batch.upper[row_index].tobytes()
        assert scalar.valid_reviewers == batch.valid_reviewers[row_index]
        assert scalar.lineage_count == batch.lineage_count[row_index]
        assert scalar.effective_sample_size == batch.effective_sample_size[row_index]


def test_matrix_column_permutation_is_byte_identical_to_scalar_path() -> None:
    reviewer_ids = tuple(f"r{index}" for index in range(10))
    rng = np.random.default_rng(107)
    likelihood_draws = {
        reviewer_id: np.stack(
            (
                rng.dirichlet((8.0, 2.0, 1.0), size=512),
                rng.dirichlet((2.0, 8.0, 1.0), size=512),
            ),
            axis=1,
        )
        for reviewer_id in reviewer_ids
    }
    dependence = _dependence(reviewer_ids)
    context = FusionContext(
        likelihood_draws=likelihood_draws,
        dependence=dependence,
        lineage_by_reviewer=dependence.lineage_by_reviewer,
        prior_pass=0.37,
        credible_mass=0.95,
    )
    observations = tuple(
        Observation.PASS if index % 2 == 0 else Observation.FAIL
        for index in range(len(reviewer_ids))
    )
    scalar = fuse_reviews(
        [
            _review("case-1", reviewer_id, observation, ExecutionState.VALID)
            for reviewer_id, observation in zip(
                reviewer_ids,
                observations,
                strict=True,
            )
        ],
        context,
    )
    permuted_ids = tuple(reversed(reviewer_ids))
    code_by_observation = {
        Observation.PASS: 0,
        Observation.FAIL: 1,
        Observation.ABSTAIN: 2,
    }
    code_by_reviewer = {
        reviewer_id: code_by_observation[observation]
        for reviewer_id, observation in zip(
            reviewer_ids,
            observations,
            strict=True,
        )
    }
    batch = fuse_review_matrix(
        np.array(
            [[code_by_reviewer[reviewer_id] for reviewer_id in permuted_ids]],
            dtype=np.int64,
        ),
        np.ones((1, len(permuted_ids)), dtype=bool),
        permuted_ids,
        context,
    )

    assert scalar is not None
    assert (
        np.asarray(scalar.pass_probability).tobytes()
        == batch.pass_probability[0].tobytes()
    )
    assert np.asarray(scalar.lower).tobytes() == batch.lower[0].tobytes()
    assert np.asarray(scalar.upper).tobytes() == batch.upper[0].tobytes()


def test_matrix_valid_mask_is_the_only_authority_for_contribution() -> None:
    context = _fixed_context(reviewer_ids=("r1", "r2"))
    ignored_codes = np.array([[0, -999], [1, 999]], dtype=np.int64)
    canonical_codes = np.array([[0, -1], [1, -1]], dtype=np.int64)
    mask = np.array([[True, False], [True, False]], dtype=bool)

    ignored = fuse_review_matrix(ignored_codes, mask, ("r1", "r2"), context)
    canonical = fuse_review_matrix(canonical_codes, mask, ("r1", "r2"), context)

    assert ignored.pass_probability.tobytes() == canonical.pass_probability.tobytes()
    assert ignored.lower.tobytes() == canonical.lower.tobytes()
    assert ignored.upper.tobytes() == canonical.upper.tobytes()


def test_matrix_rejects_invalid_code_under_true_mask() -> None:
    context = _fixed_context(reviewer_ids=("r1",))

    with pytest.raises(
        ValueError, match="valid observations must use codes 0, 1, or 2"
    ):
        fuse_review_matrix(
            np.array([[-1]], dtype=np.int64),
            np.array([[True]], dtype=bool),
            ("r1",),
            context,
        )


@pytest.mark.parametrize(
    ("observations", "valid_mask", "message"),
    [
        (
            np.zeros((2, 1), dtype=float),
            np.zeros((2, 1), dtype=bool),
            "observations must contain integer codes",
        ),
        (
            np.zeros((2, 1), dtype=np.int64),
            np.zeros((2, 1), dtype=np.int64),
            "valid_mask must contain booleans",
        ),
        (
            np.zeros((2, 1), dtype=np.int64),
            np.zeros((1, 1), dtype=bool),
            "observations and valid_mask must have the same shape",
        ),
    ],
)
def test_matrix_rejects_malformed_inputs(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    message: str,
) -> None:
    context = _fixed_context(reviewer_ids=("r1",))

    with pytest.raises((TypeError, ValueError), match=message):
        fuse_review_matrix(observations, valid_mask, ("r1",), context)


def test_batch_result_defensively_owns_read_only_arrays() -> None:
    source = np.array([0.5, 0.6])
    batch = BatchFusedPosterior(
        pass_probability=source,
        lower=np.array([0.4, 0.5]),
        upper=np.array([0.6, 0.7]),
        valid_reviewers=np.array([1, 1]),
        lineage_count=np.array([1, 1]),
        effective_sample_size=np.array([1.0, 1.0]),
    )

    source[0] = 0.9

    assert batch.pass_probability[0] == pytest.approx(0.5)
    for array in (
        batch.pass_probability,
        batch.lower,
        batch.upper,
        batch.valid_reviewers,
        batch.lineage_count,
        batch.effective_sample_size,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0


def test_batch_result_rejects_unrepresentable_numeric_arrays() -> None:
    with pytest.raises(
        ValueError, match="pass_probability must contain numeric values"
    ):
        BatchFusedPosterior(
            pass_probability=np.array([10**400], dtype=object),
            lower=np.array([0.4]),
            upper=np.array([0.6]),
            valid_reviewers=np.array([1]),
            lineage_count=np.array([1]),
            effective_sample_size=np.array([1.0]),
        )


def test_fusion_reports_multi_lineage_quorum_metadata() -> None:
    context = _fixed_context(
        reviewer_ids=("r1", "r2", "r3"),
        lineages={"r1": "shared", "r2": "shared", "r3": "independent"},
    )

    posterior = fuse_reviews(
        [
            _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-1", "r2", Observation.FAIL, ExecutionState.VALID),
            _review("case-1", "r3", Observation.PASS, ExecutionState.VALID),
        ],
        context,
    )

    assert posterior is not None
    assert posterior.valid_reviewers == 3
    assert posterior.lineage_count == 2
    assert posterior.effective_sample_size == pytest.approx(3.0)


def test_known_pair_oracle_uses_one_oriented_joint_factor_without_double_counting() -> (
    None
):
    singleton_likelihoods = {
        "r1": np.array([[0.20, 0.70, 0.10], [0.80, 0.10, 0.10]]),
        "r2": np.array([[0.70, 0.20, 0.10], [0.10, 0.80, 0.10]]),
    }

    posterior = fuse_known_pair_likelihoods(
        {"r2": Observation.PASS, "r1": Observation.FAIL},
        singleton_likelihoods,
        {("r1", "r2"): _joint_likelihoods()},
        prior_pass=0.4,
    )

    expected = (0.4 * 0.08) / ((0.4 * 0.08) + (0.6 * 0.12))
    assert expected == pytest.approx(4.0 / 13.0, abs=1e-15)
    assert posterior == pytest.approx(expected, abs=1e-12)


def test_known_pair_oracle_maps_all_nine_ordered_joint_cells() -> None:
    joint = np.array(
        [
            [
                [0.18, 0.17, 0.16],
                [0.15, 0.10, 0.08],
                [0.07, 0.05, 0.04],
            ],
            [
                [0.04, 0.05, 0.06],
                [0.07, 0.08, 0.10],
                [0.14, 0.20, 0.26],
            ],
        ],
        dtype=float,
    )
    likelihoods = {
        "r1": np.full((2, 3), 1.0 / 3.0),
        "r2": np.full((2, 3), 1.0 / 3.0),
    }
    prior_pass = 0.37

    for first_index, first in enumerate(OBSERVATION_ORDER):
        for second_index, second in enumerate(OBSERVATION_ORDER):
            actual = fuse_known_pair_likelihoods(
                {"r2": second, "r1": first},
                likelihoods,
                {("r1", "r2"): joint},
                prior_pass=prior_pass,
            )
            pass_likelihood = joint[0, first_index, second_index]
            fail_likelihood = joint[1, first_index, second_index]
            expected = (prior_pass * pass_likelihood) / (
                (prior_pass * pass_likelihood) + ((1.0 - prior_pass) * fail_likelihood)
            )
            assert actual == pytest.approx(expected, abs=1e-12)


def test_pair_fusion_bounds_a_legal_zero_joint_cell_away_from_log_zero() -> None:
    joint = _joint_likelihoods()
    joint[0, 1, 1] += joint[0, 1, 0]
    joint[0, 1, 0] = 0.0
    likelihoods = {
        "r1": np.full((2, 3), 1.0 / 3.0),
        "r2": np.full((2, 3), 1.0 / 3.0),
    }
    observations = {"r1": Observation.FAIL, "r2": Observation.PASS}

    known = fuse_known_pair_likelihoods(
        observations,
        likelihoods,
        {("r1", "r2"): joint},
        prior_pass=0.4,
    )
    context = _fixed_context(
        reviewer_ids=("r1", "r2"),
        prior_pass=0.4,
        pair_likelihoods={("r1", "r2"): joint},
    )
    posterior = fuse_reviews(
        [
            _review("case", "r1", Observation.FAIL, ExecutionState.VALID),
            _review("case", "r2", Observation.PASS, ExecutionState.VALID),
        ],
        context,
    )

    assert 0.0 < known < 1e-300
    assert posterior is not None
    assert 0.0 < posterior.pass_probability < 1e-300


def test_known_pair_oracle_reduces_to_naive_for_product_joint_and_empty_map() -> None:
    likelihoods = {
        "r1": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
        "r2": np.array([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]),
    }
    observations = {"r1": Observation.PASS, "r2": Observation.FAIL}
    product_joint = np.stack(
        [
            np.outer(likelihoods["r1"][truth], likelihoods["r2"][truth])
            for truth in range(2)
        ],
    )
    naive = fuse_known_likelihoods(
        observations,
        likelihoods,
        {"r1": 1.0, "r2": 1.0},
        prior_pass=0.6,
    )

    product = fuse_known_pair_likelihoods(
        observations,
        likelihoods,
        {("r1", "r2"): product_joint},
        prior_pass=0.6,
    )
    empty = fuse_known_pair_likelihoods(
        observations,
        likelihoods,
        {},
        prior_pass=0.6,
    )

    assert naive == pytest.approx(0.144 / (0.144 + 0.064), abs=1e-12)
    assert product == pytest.approx(naive, abs=1e-12)
    assert empty == pytest.approx(naive, abs=1e-12)


@pytest.mark.parametrize(
    ("pair_likelihoods", "message"),
    [
        ({("r2", "r1"): _joint_likelihoods()}, "canonical"),
        ({("r1", "r1"): _joint_likelihoods()}, "distinct"),
        ({("", "r1"): _joint_likelihoods()}, "blank"),
        (
            {
                ("r1", "r2"): _joint_likelihoods(),
                ("r2", "r3"): _joint_likelihoods(),
            },
            "overlap",
        ),
        ({("r1", "z-missing"): _joint_likelihoods()}, "unknown"),
        ({("r1", "r2"): np.ones((2, 9)) / 9.0}, "shape"),
        (
            {("r1", "r2"): np.full((2, 3, 3), 0.2)},
            "sum",
        ),
        (
            {
                ("r1", "r2"): np.array(
                    [
                        [
                            [-0.01, 0.65, 0.06],
                            [0.08, 0.10, 0.02],
                            [0.04, 0.04, 0.02],
                        ],
                        _joint_likelihoods()[1],
                    ]
                )
            },
            "finite probabilities",
        ),
        (
            {
                ("r1", "r2"): np.where(
                    np.indices((2, 3, 3)).sum(axis=0) == 0,
                    np.nan,
                    _joint_likelihoods(),
                )
            },
            "finite probabilities",
        ),
        (
            {("r1", "r2"): np.full((2, 3, 3), "bad", dtype=object)},
            "numeric",
        ),
    ],
)
def test_known_pair_oracle_rejects_invalid_pair_registry(
    pair_likelihoods: Mapping[PairKey, np.ndarray],
    message: str,
) -> None:
    likelihoods = {
        reviewer_id: np.full((2, 3), 1.0 / 3.0) for reviewer_id in ("r1", "r2", "r3")
    }
    with pytest.raises(ValueError, match=message):
        fuse_known_pair_likelihoods(
            {"r1": Observation.PASS, "r2": Observation.FAIL},
            likelihoods,
            pair_likelihoods,
            prior_pass=0.5,
        )


@pytest.mark.parametrize(
    "observations",
    [
        {"r1": Observation.PASS, "r3": Observation.FAIL},
        {"r2": Observation.FAIL, "r3": Observation.PASS},
        {"r1": Observation.ABSTAIN, "r3": Observation.FAIL},
    ],
)
def test_known_pair_oracle_uses_singleton_fallback_for_one_observed_member(
    observations: Mapping[str, Observation],
) -> None:
    likelihoods = {
        "r1": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
        "r2": np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        "r3": np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]),
    }
    expected = fuse_known_likelihoods(
        observations,
        likelihoods,
        {reviewer_id: 1.0 for reviewer_id in observations},
        prior_pass=0.35,
    )
    extreme_joint = np.full((2, 3, 3), 1e-6)
    extreme_joint[:, 2, 2] = 1.0 - 8e-6

    actual = fuse_known_pair_likelihoods(
        observations,
        likelihoods,
        {("r1", "r2"): extreme_joint},
        prior_pass=0.35,
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_build_pair_context_preserves_singleton_bytes_and_sorts_pair_sampling() -> None:
    reviewer_ids = ("r1", "r2", "r3", "r4")
    base = np.array([[0.75, 0.20, 0.05], [0.15, 0.80, 0.05]])
    calibrations = {
        reviewer_id: _calibration(reviewer_id, base, strength=60.0)
        for reviewer_id in reviewer_ids
    }
    dependence = _dependence(reviewer_ids)
    first_joint = _joint_likelihoods()
    second_joint = np.flip(_joint_likelihoods(), axis=2).copy()
    first_pair = _pair_calibration(("r1", "r2"), first_joint)
    second_pair = _pair_calibration(("r3", "r4"), second_joint)
    reverse_mapping = {
        ("r3", "r4"): second_pair,
        ("r1", "r2"): first_pair,
    }

    first = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.45,
        draws=64,
        seed=117,
        pair_calibrations=reverse_mapping,
    )
    second = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.45,
        draws=64,
        seed=117,
        pair_calibrations={
            ("r1", "r2"): first_pair,
            ("r3", "r4"): second_pair,
        },
    )
    legacy = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.45,
        draws=64,
        seed=117,
    )
    explicit_empty = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.45,
        draws=64,
        seed=117,
        pair_calibrations={},
    )
    explicit_none = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.45,
        draws=64,
        seed=117,
        pair_calibrations=None,
    )
    different = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=0.45,
        draws=64,
        seed=118,
        pair_calibrations=reverse_mapping,
    )

    assert tuple(first.pair_likelihood_draws) == (
        ("r1", "r2"),
        ("r3", "r4"),
    )
    assert dict(legacy.pair_likelihood_draws) == {}
    assert dict(explicit_empty.pair_likelihood_draws) == {}
    assert dict(explicit_none.pair_likelihood_draws) == {}
    manual_rng = np.random.default_rng(117)
    expected_singletons = {
        reviewer_id: calibrations[reviewer_id].sample_likelihoods(64, manual_rng)
        for reviewer_id in dependence.reviewer_ids
    }
    expected_pairs = {
        pair: reverse_mapping[pair].sample_likelihoods(64, manual_rng)
        for pair in sorted(reverse_mapping)
    }
    for reviewer_id in reviewer_ids:
        assert (
            first.likelihood_draws[reviewer_id].tobytes()
            == second.likelihood_draws[reviewer_id].tobytes()
            == legacy.likelihood_draws[reviewer_id].tobytes()
            == explicit_empty.likelihood_draws[reviewer_id].tobytes()
            == explicit_none.likelihood_draws[reviewer_id].tobytes()
            == expected_singletons[reviewer_id].tobytes()
        )
    for pair in first.pair_likelihood_draws:
        assert (
            first.pair_likelihood_draws[pair].tobytes()
            == second.pair_likelihood_draws[pair].tobytes()
            == expected_pairs[pair].tobytes()
        )
        assert (
            first.pair_likelihood_draws[pair].tobytes()
            != different.pair_likelihood_draws[pair].tobytes()
        )


def test_pair_context_defensively_owns_read_only_joint_draws() -> None:
    joint_draws = np.repeat(_joint_likelihoods()[None, :, :, :], 8, axis=0)
    source_mapping = {("r1", "r2"): joint_draws}
    dependence = _dependence(("r1", "r2"))
    context = FusionContext(
        likelihood_draws={
            reviewer_id: np.full((8, 2, 3), 1.0 / 3.0)
            for reviewer_id in dependence.reviewer_ids
        },
        dependence=dependence,
        lineage_by_reviewer=dependence.lineage_by_reviewer,
        prior_pass=0.5,
        credible_mass=0.95,
        pair_likelihood_draws=source_mapping,
    )

    joint_draws[0, 0, 0, 0] = 0.99
    source_mapping.clear()

    stored = context.pair_likelihood_draws[("r1", "r2")]
    assert stored[0, 0, 0, 0] == pytest.approx(0.40)
    assert not stored.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        stored.setflags(write=True)
    with pytest.raises(TypeError):
        context.pair_likelihood_draws[("r1", "r2")] = stored  # type: ignore[index]


@pytest.mark.parametrize(
    ("pair_draws", "message"),
    [
        (
            {("r1", "r2"): np.repeat(_joint_likelihoods()[None], 7, axis=0)},
            "draw count",
        ),
        (
            {("r1", "r2"): np.ones((8, 2, 9)) / 9.0},
            "shape",
        ),
        (
            {("r1", "r2"): np.full((8, 2, 3, 3), 0.2)},
            "sum",
        ),
        (
            {
                ("r1", "r2"): np.repeat(
                    np.array(
                        [
                            [
                                [-0.01, 0.65, 0.06],
                                [0.08, 0.10, 0.02],
                                [0.04, 0.04, 0.02],
                            ],
                            _joint_likelihoods()[1],
                        ]
                    )[None],
                    8,
                    axis=0,
                )
            },
            "finite probabilities",
        ),
        (
            {
                ("r1", "r2"): np.full(
                    (8, 2, 3, 3),
                    "bad",
                    dtype=object,
                )
            },
            "numeric",
        ),
        (
            {
                ("r1", "r2"): np.repeat(
                    np.where(
                        np.indices((2, 3, 3)).sum(axis=0) == 0,
                        np.nan,
                        _joint_likelihoods(),
                    )[None],
                    8,
                    axis=0,
                )
            },
            "finite probabilities",
        ),
        (
            {("r2", "r1"): np.repeat(_joint_likelihoods()[None], 8, axis=0)},
            "canonical",
        ),
        (
            {
                ("r1", "r2"): np.repeat(_joint_likelihoods()[None], 8, axis=0),
                ("r2", "r3"): np.repeat(_joint_likelihoods()[None], 8, axis=0),
            },
            "overlap",
        ),
        (
            {("r1", "z-missing"): np.repeat(_joint_likelihoods()[None], 8, axis=0)},
            "unknown",
        ),
    ],
)
def test_pair_context_rejects_invalid_joint_draw_registry(
    pair_draws: Mapping[PairKey, np.ndarray],
    message: str,
) -> None:
    reviewer_ids = ("r1", "r2", "r3")
    dependence = _dependence(reviewer_ids)
    with pytest.raises(ValueError, match=message):
        FusionContext(
            likelihood_draws={
                reviewer_id: np.full((8, 2, 3), 1.0 / 3.0)
                for reviewer_id in reviewer_ids
            },
            dependence=dependence,
            lineage_by_reviewer=dependence.lineage_by_reviewer,
            prior_pass=0.5,
            credible_mass=0.95,
            pair_likelihood_draws=pair_draws,
        )


@pytest.mark.parametrize(
    ("pair_calibrations", "message"),
    [
        ({("r2", "r1"): _pair_calibration(("r1", "r2"))}, "canonical"),
        (
            {
                ("r1", "r2"): _pair_calibration(("r1", "r2")),
                ("r2", "r3"): _pair_calibration(("r2", "r3")),
            },
            "overlap",
        ),
        (
            {("r1", "z-missing"): _pair_calibration(("r1", "z-missing"))},
            "unknown",
        ),
        ({("r1", "r2"): _pair_calibration(("r1", "r3"))}, "mapping key"),
    ],
)
def test_build_context_rejects_invalid_pair_calibration_registry(
    pair_calibrations: Mapping[PairKey, ReviewerPairCalibration],
    message: str,
) -> None:
    reviewer_ids = ("r1", "r2", "r3")
    likelihood = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    with pytest.raises(ValueError, match=message):
        build_fusion_context(
            {
                reviewer_id: _calibration(reviewer_id, likelihood)
                for reviewer_id in reviewer_ids
            },
            _dependence(reviewer_ids),
            prior_pass=0.5,
            draws=8,
            seed=1,
            pair_calibrations=pair_calibrations,
        )


def test_complete_pair_and_singleton_use_joint_once_without_power_tempering() -> None:
    context = _fixed_context(
        reviewer_ids=("r1", "r2", "r3"),
        correlation=np.ones((3, 3), dtype=float),
        lineages={"r1": "pair", "r2": "pair", "r3": "solo"},
        prior_pass=0.6,
        pair_likelihoods={("r1", "r2"): _joint_likelihoods()},
    )
    reviews = [
        _review("case", "r1", Observation.FAIL, ExecutionState.VALID),
        _review("case", "r2", Observation.PASS, ExecutionState.VALID),
        _review("case", "r3", Observation.PASS, ExecutionState.VALID),
    ]

    posterior = fuse_reviews(reviews, context)

    expected = (0.6 * 0.08 * 0.75) / ((0.6 * 0.08 * 0.75) + (0.4 * 0.12 * 0.10))
    assert expected == pytest.approx(15.0 / 17.0, abs=1e-15)
    assert posterior is not None
    assert posterior.pass_probability == pytest.approx(expected, abs=1e-12)
    assert posterior.lower == pytest.approx(expected, abs=1e-12)
    assert posterior.upper == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(posterior.samples, expected, rtol=0.0, atol=1e-12)
    assert posterior.valid_reviewers == 3
    assert posterior.lineage_count == 2
    assert posterior.effective_sample_size == pytest.approx(1.0)


def test_two_nonadjacent_pair_blocks_each_contribute_exactly_once() -> None:
    reviewer_ids = ("r1", "r3", "r2", "r4")
    context = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=np.ones((4, 4), dtype=float),
        lineages={"r1": "pair-1", "r4": "pair-1", "r2": "pair-2", "r3": "pair-2"},
        prior_pass=0.55,
        pair_likelihoods={
            ("r2", "r3"): _joint_likelihoods(),
            ("r1", "r4"): _joint_likelihoods(),
        },
    )
    reviews = [
        _review("case", "r1", Observation.FAIL, ExecutionState.VALID),
        _review("case", "r3", Observation.FAIL, ExecutionState.VALID),
        _review("case", "r2", Observation.PASS, ExecutionState.VALID),
        _review("case", "r4", Observation.PASS, ExecutionState.VALID),
    ]

    posterior = fuse_reviews(reviews, context)
    batch = fuse_review_matrix(
        np.array([[1, 1, 0, 0]], dtype=np.int64),
        np.ones((1, 4), dtype=bool),
        reviewer_ids,
        context,
    )

    expected = (0.55 * 0.08 * 0.24) / ((0.55 * 0.08 * 0.24) + (0.45 * 0.12 * 0.08))
    assert expected == pytest.approx(22.0 / 31.0, abs=1e-15)
    assert posterior is not None
    assert posterior.pass_probability == pytest.approx(expected, abs=1e-12)
    assert posterior.lower == pytest.approx(expected, abs=1e-12)
    assert posterior.upper == pytest.approx(expected, abs=1e-12)
    assert posterior.valid_reviewers == 4
    assert posterior.lineage_count == 2
    assert posterior.effective_sample_size == pytest.approx(1.0)
    assert batch.pass_probability[0] == posterior.pass_probability
    assert batch.lower[0] == posterior.lower
    assert batch.upper[0] == posterior.upper


def test_unobserved_pair_leaves_other_singleton_blocks_untempered() -> None:
    reviewer_ids = ("r1", "r2", "r3", "r4")
    context = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=np.ones((4, 4), dtype=float),
        lineages={"r1": "pair", "r2": "pair", "r3": "solo-3", "r4": "solo-4"},
        prior_pass=0.4,
        pair_likelihoods={("r1", "r2"): _joint_likelihoods()},
    )
    reviews = [
        _review("case", "r1", None, ExecutionState.TIMEOUT),
        _review("case", "r2", None, ExecutionState.INVALID),
        _review("case", "r3", Observation.PASS, ExecutionState.VALID),
        _review("case", "r4", Observation.FAIL, ExecutionState.VALID),
    ]

    posterior = fuse_reviews(reviews, context)
    batch = fuse_review_matrix(
        np.array([[999, -999, 0, 1]], dtype=np.int64),
        np.array([[False, False, True, True]], dtype=bool),
        reviewer_ids,
        context,
    )

    expected = (0.4 * 0.75 * 0.25) / ((0.4 * 0.75 * 0.25) + (0.6 * 0.10 * 0.65))
    assert expected == pytest.approx(25.0 / 38.0, abs=1e-15)
    assert posterior is not None
    assert posterior.pass_probability == pytest.approx(expected, abs=1e-12)
    assert posterior.valid_reviewers == 2
    assert posterior.lineage_count == 2
    assert posterior.effective_sample_size == pytest.approx(1.0)
    assert batch.pass_probability[0] == posterior.pass_probability
    assert batch.effective_sample_size[0] == posterior.effective_sample_size


def test_one_valid_pair_member_matches_naive_but_keeps_fitted_ess() -> None:
    reviewer_ids = ("r1", "r2", "r3")
    pair_context = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=np.ones((3, 3), dtype=float),
        lineages={"r1": "pair", "r2": "pair", "r3": "solo"},
        prior_pass=0.6,
        pair_likelihoods={("r1", "r2"): _joint_likelihoods()},
    )
    identity = _dependence(
        reviewer_ids,
        lineages=pair_context.lineage_by_reviewer,
    )
    naive_context = FusionContext(
        likelihood_draws=pair_context.likelihood_draws,
        dependence=identity,
        lineage_by_reviewer=identity.lineage_by_reviewer,
        prior_pass=pair_context.prior_pass,
        credible_mass=pair_context.credible_mass,
    )
    reviews = [
        _review("case", "r1", Observation.PASS, ExecutionState.VALID),
        _review("case", "r2", None, ExecutionState.TIMEOUT),
        _review("case", "r3", Observation.PASS, ExecutionState.VALID),
    ]

    pair = fuse_reviews(reviews, pair_context)
    naive = fuse_reviews(reviews, naive_context)
    matrix = fuse_review_matrix(
        np.array([[0, 999, 0]], dtype=np.int64),
        np.array([[True, False, True]], dtype=bool),
        reviewer_ids,
        pair_context,
    )

    expected = (0.6 * 0.80 * 0.75) / ((0.6 * 0.80 * 0.75) + (0.4 * 0.15 * 0.10))
    assert expected == pytest.approx(60.0 / 61.0, abs=1e-15)
    assert pair is not None
    assert naive is not None
    assert pair.samples == naive.samples
    assert pair.pass_probability == pytest.approx(expected, abs=1e-12)
    assert pair.pass_probability == naive.pass_probability
    assert pair.lower == naive.lower
    assert pair.upper == naive.upper
    assert pair.valid_reviewers == naive.valid_reviewers == 2
    assert pair.lineage_count == naive.lineage_count == 2
    assert pair.effective_sample_size == pytest.approx(1.0)
    assert naive.effective_sample_size == pytest.approx(2.0)
    assert matrix.pass_probability[0] == pair.pass_probability
    assert matrix.lower[0] == pair.lower
    assert matrix.upper[0] == pair.upper
    assert matrix.effective_sample_size[0] == pair.effective_sample_size


def test_pair_all_abstain_widens_bounds_and_all_invalid_is_empty() -> None:
    reviewer_ids = ("r1", "r2", "r3")
    context = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=np.ones((3, 3), dtype=float),
        lineages={"r1": "pair", "r2": "pair", "r3": "solo"},
        prior_pass=0.6,
        pair_likelihoods={("r1", "r2"): _joint_likelihoods()},
    )
    abstain = fuse_reviews(
        [
            _review("case", "r1", Observation.ABSTAIN, ExecutionState.VALID),
            _review("case", "r2", Observation.ABSTAIN, ExecutionState.VALID),
            _review("case", "r3", None, ExecutionState.INVALID),
        ],
        context,
    )
    invalid_reviews = [
        _review("invalid", reviewer_id, None, ExecutionState.INVALID)
        for reviewer_id in reviewer_ids
    ]
    empty = fuse_reviews(invalid_reviews, context)
    batch = fuse_review_matrix(
        np.array([[2, 2, -1], [999, 999, 999]], dtype=np.int64),
        np.array([[True, True, False], [False, False, False]], dtype=bool),
        reviewer_ids,
        context,
    )

    expected = (0.6 * 0.02) / ((0.6 * 0.02) + (0.4 * 0.01))
    assert abstain is not None
    assert abstain.pass_probability == pytest.approx(expected, abs=1e-12)
    assert abstain.lower == 0.0
    assert abstain.upper == 1.0
    assert abstain.valid_reviewers == 2
    assert abstain.lineage_count == 1
    assert abstain.effective_sample_size == pytest.approx(1.0)
    assert empty is None
    assert batch.pass_probability[0] == abstain.pass_probability
    assert batch.lower[0] == 0.0
    assert batch.upper[0] == 1.0
    assert np.isnan(batch.pass_probability[1])
    assert batch.valid_reviewers[1] == 0
    assert batch.lineage_count[1] == 0
    assert batch.effective_sample_size[1] == 0.0


def test_pair_scalar_matrix_permutation_chunk_and_mask_paths_are_identical() -> None:
    reviewer_ids = ("r1", "r2", "r3")
    context = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=np.array(
            [[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]],
        ),
        lineages={"r1": "pair", "r2": "pair", "r3": "solo"},
        prior_pass=0.55,
        pair_likelihoods={("r1", "r2"): _joint_likelihoods()},
    )
    observations = np.array(
        [
            [0, 1, 0],
            [0, 999, 1],
            [999, 0, 0],
            [999, 999, 0],
            [999, 999, 999],
            [2, 2, -999],
        ],
        dtype=np.int64,
    )
    valid_mask = np.array(
        [
            [True, True, True],
            [True, False, True],
            [False, True, True],
            [False, False, True],
            [False, False, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    state_by_mask = {
        True: ExecutionState.VALID,
        False: ExecutionState.TIMEOUT,
    }
    observation_by_code = {
        0: Observation.PASS,
        1: Observation.FAIL,
        2: Observation.ABSTAIN,
    }
    scalar_rows = []
    for row_index in range(observations.shape[0]):
        scalar_rows.append(
            [
                _review(
                    f"case-{row_index}",
                    reviewer_id,
                    (
                        observation_by_code[int(observations[row_index, column])]
                        if valid_mask[row_index, column]
                        else None
                    ),
                    state_by_mask[bool(valid_mask[row_index, column])],
                )
                for column, reviewer_id in enumerate(reviewer_ids)
            ]
        )

    first = fuse_review_matrix(
        observations,
        valid_mask,
        reviewer_ids,
        context,
        chunk_size=1,
    )
    reversed_ids = tuple(reversed(reviewer_ids))
    second = fuse_review_matrix(
        observations[:, ::-1],
        valid_mask[:, ::-1],
        reversed_ids,
        context,
        chunk_size=3,
    )

    for first_array, second_array in (
        (first.pass_probability, second.pass_probability),
        (first.lower, second.lower),
        (first.upper, second.upper),
        (first.valid_reviewers, second.valid_reviewers),
        (first.lineage_count, second.lineage_count),
        (first.effective_sample_size, second.effective_sample_size),
    ):
        assert first_array.tobytes() == second_array.tobytes()
    for row_index, reviews in enumerate(scalar_rows):
        scalar = fuse_reviews(reviews, context)
        if scalar is None:
            assert np.isnan(first.pass_probability[row_index])
            assert first.valid_reviewers[row_index] == 0
            continue
        assert (
            np.asarray(scalar.pass_probability).tobytes()
            == first.pass_probability[row_index].tobytes()
        )
        assert np.asarray(scalar.lower).tobytes() == first.lower[row_index].tobytes()
        assert np.asarray(scalar.upper).tobytes() == first.upper[row_index].tobytes()
        assert scalar.valid_reviewers == first.valid_reviewers[row_index]
        assert scalar.lineage_count == first.lineage_count[row_index]
        assert scalar.effective_sample_size == first.effective_sample_size[row_index]


def test_pair_context_reuses_joint_parameter_draws_across_case_ids() -> None:
    reviewer_ids = ("r1", "r2")
    likelihood = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    calibrations = {
        reviewer_id: _calibration(reviewer_id, likelihood)
        for reviewer_id in reviewer_ids
    }
    context = build_fusion_context(
        calibrations,
        _dependence(reviewer_ids),
        prior_pass=0.5,
        draws=64,
        seed=11,
        pair_calibrations={("r1", "r2"): _pair_calibration(("r1", "r2"))},
    )
    first = fuse_reviews(
        [
            _review("case-1", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-1", "r2", Observation.FAIL, ExecutionState.VALID),
        ],
        context,
    )
    second = fuse_reviews(
        [
            _review("case-2", "r1", Observation.PASS, ExecutionState.VALID),
            _review("case-2", "r2", Observation.FAIL, ExecutionState.VALID),
        ],
        context,
    )

    assert first is not None
    assert second is not None
    assert first.samples == second.samples


def test_explicit_empty_pair_context_preserves_legacy_power_fusion() -> None:
    reviewer_ids = ("r1", "r2")
    correlation = np.ones((2, 2), dtype=float)
    lineages = {"r1": "pair", "r2": "pair"}
    explicit = _fixed_context(
        reviewer_ids=reviewer_ids,
        correlation=correlation,
        lineages=lineages,
        prior_pass=0.4,
        pair_likelihoods={},
    )
    omitted = FusionContext(
        likelihood_draws=explicit.likelihood_draws,
        dependence=explicit.dependence,
        lineage_by_reviewer=explicit.lineage_by_reviewer,
        prior_pass=explicit.prior_pass,
        credible_mass=explicit.credible_mass,
    )
    reviews = [
        _review("case", "r1", Observation.PASS, ExecutionState.VALID),
        _review("case", "r2", Observation.PASS, ExecutionState.VALID),
    ]
    explicit_result = fuse_reviews(reviews, explicit)
    omitted_result = fuse_reviews(reviews, omitted)
    weights = explicit.dependence.weights_for(reviewer_ids)
    expected = fuse_known_likelihoods(
        {"r1": Observation.PASS, "r2": Observation.PASS},
        {
            reviewer_id: explicit.likelihood_draws[reviewer_id][0]
            for reviewer_id in reviewer_ids
        },
        weights,
        prior_pass=0.4,
    )

    assert explicit_result is not None
    assert omitted_result is not None
    assert explicit_result == omitted_result
    assert explicit_result.pass_probability == pytest.approx(expected, abs=1e-15)
