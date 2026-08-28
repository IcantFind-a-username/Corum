from dataclasses import FrozenInstanceError
from itertools import combinations
from math import inf, nan

import numpy as np
import pytest

from corum.dependence import DependenceModel, fit_dependence
from corum.models import (
    CalibrationExample,
    ExecutionState,
    Observation,
    Review,
    Reviewer,
    Truth,
)


def _reviewer(
    reviewer_id: str,
    *,
    lineage: str | None = None,
    cost: float = 1.0,
) -> Reviewer:
    return Reviewer(
        reviewer_id=reviewer_id,
        vendor="vendor",
        family="family",
        lineage=lineage or f"lineage-{reviewer_id}",
        cost=cost,
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


def _error_pattern_examples(
    patterns: dict[str, tuple[int, ...]],
    *,
    case_prefix: str = "case",
) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    for reviewer_id, errors in patterns.items():
        for index, error in enumerate(errors):
            examples.append(
                _example(
                    reviewer_id,
                    Truth.PASS,
                    Observation.FAIL if error else Observation.PASS,
                    case_id=f"{case_prefix}-{index}",
                )
            )
    return examples


def test_independent_errors_receive_unit_weights_without_accuracy_scaling() -> None:
    reviewers = [_reviewer("a"), _reviewer("b")]
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 1, 1),
            "b": (0, 1, 0, 1),
        }
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    np.testing.assert_allclose(model.correlation, np.eye(2), atol=1e-12)
    assert dict(model.weights_for(("a", "b"))) == {"a": 1.0, "b": 1.0}
    assert model.effective_sample_size(("a", "b")) == pytest.approx(2.0)


def test_exact_clones_contribute_one_effective_review_for_the_queried_subset() -> None:
    reviewers = [
        _reviewer("a", lineage="clone-lineage"),
        _reviewer("b", lineage="clone-lineage"),
    ]
    examples = _error_pattern_examples(
        {
            "a": (0, 1, 0, 1),
            "b": (0, 1, 0, 1),
        }
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    weights = model.weights_for(("a", "b"))
    assert weights["a"] == pytest.approx(0.5)
    assert weights["b"] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert model.effective_sample_size(("a", "b")) == pytest.approx(1.0)
    assert dict(model.weights_for(("a",))) == {"a": 1.0}


def test_four_exact_clones_have_total_weight_near_one() -> None:
    reviewer_ids = ("a", "b", "c", "d")
    reviewers = [
        _reviewer(reviewer_id, lineage="clone-lineage") for reviewer_id in reviewer_ids
    ]
    examples = _error_pattern_examples(
        {reviewer_id: (0, 0, 1, 1) for reviewer_id in reviewer_ids}
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    weights = model.weights_for(reviewer_ids)
    assert all(weight == pytest.approx(0.25) for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert model.effective_sample_size(reviewer_ids) == pytest.approx(1.0)


def test_weights_use_only_the_exact_queried_subset() -> None:
    model = DependenceModel(
        reviewer_ids=("a", "b", "c"),
        correlation=np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        lineage_by_reviewer={"a": "clone", "b": "clone", "c": "other"},
    )

    assert dict(model.weights_for(("a",))) == {"a": 1.0}
    assert dict(model.weights_for(("a", "c"))) == {"a": 1.0, "c": 1.0}
    assert dict(model.weights_for(("a", "b", "c"))) == {
        "a": 0.5,
        "b": 0.5,
        "c": 1.0,
    }


def test_negative_error_correlation_is_preserved_but_not_extra_information() -> None:
    reviewers = [_reviewer("a"), _reviewer("b")]
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 1, 1),
            "b": (1, 1, 0, 0),
        }
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    assert model.correlation[0, 1] == pytest.approx(-1.0)
    assert dict(model.weights_for(("a", "b"))) == {"a": 1.0, "b": 1.0}
    assert model.effective_sample_size(("a", "b")) == pytest.approx(2.0)


def test_empirical_correlation_is_shrunk_toward_zero() -> None:
    examples = _error_pattern_examples(
        {
            "a": (0, 1, 0, 1),
            "b": (0, 1, 0, 1),
        }
    )

    model = fit_dependence(
        [_reviewer("a"), _reviewer("b")],
        examples,
        shrinkage=0.25,
        min_overlap=4,
    )

    assert model.correlation[0, 1] == pytest.approx(0.75)
    assert model.weights_for(("a", "b"))["a"] == pytest.approx(1.0 / 1.75)


def test_sparse_overlap_uses_lineage_fallback_and_unrelated_default() -> None:
    reviewers = [
        _reviewer("a", lineage="shared"),
        _reviewer("b", lineage="shared"),
        _reviewer("c", lineage="unrelated"),
    ]
    examples = _error_pattern_examples(
        {
            "a": (0, 1),
            "b": (0, 1),
            "c": (0, 1),
        }
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.9,
        min_overlap=3,
        lineage_cap=0.8,
    )

    index = {reviewer_id: i for i, reviewer_id in enumerate(model.reviewer_ids)}
    assert model.correlation[index["a"], index["b"]] == pytest.approx(0.8)
    assert model.correlation[index["a"], index["c"]] == pytest.approx(0.0)
    assert model.weights_for(("a", "b"))["a"] == pytest.approx(1.0 / 1.8)


def test_projected_diagnostic_does_not_weaken_sparse_lineage_fallback() -> None:
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 1, 1),
            "c": (0, 0, 1, 1),
        },
        case_prefix="ac",
    )
    examples.extend(
        _error_pattern_examples(
            {
                "b": (0, 0, 1, 1),
                "c": (1, 1, 0, 0),
            },
            case_prefix="bc",
        )
    )
    reviewers = [
        _reviewer("a", lineage="shared"),
        _reviewer("b", lineage="shared"),
        _reviewer("c", lineage="other"),
    ]

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
        lineage_cap=1.0,
    )
    permuted = fit_dependence(
        list(reversed(reviewers)),
        list(reversed(examples)),
        shrinkage=0.0,
        min_overlap=4,
        lineage_cap=1.0,
    )

    index = {reviewer_id: i for i, reviewer_id in enumerate(model.reviewer_ids)}
    assert model.correlation[index["a"], index["b"]] == pytest.approx(0.5)
    assert dict(model.weights_for(("a", "b"))) == {"a": 0.5, "b": 0.5}
    assert model.effective_sample_size(("a", "b")) == pytest.approx(1.0)
    assert dict(permuted.weights_for(("b", "a"))) == {"b": 0.5, "a": 0.5}


def test_projected_diagnostic_does_not_invent_unrelated_fallback_weight() -> None:
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 1, 1),
            "c": (0, 0, 1, 1),
        },
        case_prefix="ac",
    )
    examples.extend(
        _error_pattern_examples(
            {
                "b": (0, 0, 1, 1),
                "c": (0, 0, 1, 1),
            },
            case_prefix="bc",
        )
    )
    reviewers = [_reviewer("a"), _reviewer("b"), _reviewer("c")]

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    index = {reviewer_id: i for i, reviewer_id in enumerate(model.reviewer_ids)}
    assert model.correlation[index["a"], index["b"]] == pytest.approx(
        0.09383632135605444
    )
    assert dict(model.weights_for(("a", "b"))) == {"a": 1.0, "b": 1.0}
    assert model.effective_sample_size(("a", "b")) == pytest.approx(2.0)


def test_estimable_same_lineage_pair_uses_empirical_correlation() -> None:
    reviewers = [
        _reviewer("a", lineage="shared"),
        _reviewer("b", lineage="shared"),
    ]
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 1, 1),
            "b": (0, 1, 0, 1),
        }
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
        lineage_cap=1.0,
    )

    np.testing.assert_allclose(model.correlation, np.eye(2), atol=1e-12)
    assert dict(model.weights_for(("a", "b"))) == {"a": 1.0, "b": 1.0}


def test_only_paired_valid_reviews_count_as_overlap() -> None:
    examples = [
        _example("a", Truth.PASS, Observation.PASS, case_id="paired-1"),
        _example("b", Truth.PASS, Observation.FAIL, case_id="paired-1"),
        _example("a", Truth.PASS, Observation.FAIL, case_id="paired-2"),
        _example("b", Truth.PASS, Observation.PASS, case_id="paired-2"),
    ]
    for index in range(8):
        examples.extend(
            [
                _example(
                    "a",
                    Truth.PASS,
                    Observation.PASS,
                    case_id=f"a-only-{index}",
                ),
                _example(
                    "b",
                    Truth.PASS,
                    None,
                    case_id=f"a-only-{index}",
                    state=ExecutionState.TIMEOUT,
                ),
                _example(
                    "a",
                    Truth.PASS,
                    None,
                    case_id=f"b-only-{index}",
                    state=ExecutionState.INVALID,
                ),
                _example(
                    "b",
                    Truth.PASS,
                    Observation.PASS,
                    case_id=f"b-only-{index}",
                ),
            ]
        )

    model = fit_dependence(
        [_reviewer("a"), _reviewer("b")],
        examples,
        shrinkage=0.0,
        min_overlap=2,
    )

    assert model.correlation[0, 1] == pytest.approx(-1.0)


def test_semantic_truth_matches_are_not_cross_enum_errors() -> None:
    examples: list[CalibrationExample] = []
    truths = (Truth.PASS, Truth.FAIL, Truth.PASS, Truth.FAIL)
    for case_index, truth in enumerate(truths):
        for reviewer_id in ("a", "b"):
            examples.append(
                _example(
                    reviewer_id,
                    truth,
                    Observation.PASS,
                    case_id=f"case-{case_index}",
                )
            )

    model = fit_dependence(
        [_reviewer("a"), _reviewer("b")],
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    assert model.correlation[0, 1] == pytest.approx(1.0)


def test_valid_abstain_is_encoded_as_an_error() -> None:
    examples: list[CalibrationExample] = []
    for case_index in range(4):
        abstains = case_index % 2 == 0
        examples.extend(
            [
                _example(
                    "a",
                    Truth.PASS,
                    Observation.ABSTAIN if abstains else Observation.PASS,
                    case_id=f"case-{case_index}",
                ),
                _example(
                    "b",
                    Truth.PASS,
                    Observation.FAIL if abstains else Observation.PASS,
                    case_id=f"case-{case_index}",
                ),
            ]
        )

    model = fit_dependence(
        [_reviewer("a"), _reviewer("b")],
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    assert model.correlation[0, 1] == pytest.approx(1.0)


def test_constant_error_pairs_have_deterministic_lineage_fallbacks() -> None:
    reviewers = [
        _reviewer("a", lineage="shared"),
        _reviewer("b", lineage="shared"),
        _reviewer("c", lineage="other"),
    ]
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 0, 0),
            "b": (0, 0, 0, 0),
            "c": (0, 0, 0, 0),
        }
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
        lineage_cap=0.7,
    )

    index = {reviewer_id: i for i, reviewer_id in enumerate(model.reviewer_ids)}
    assert model.correlation[index["a"], index["b"]] == pytest.approx(0.7)
    assert model.correlation[index["a"], index["c"]] == pytest.approx(0.0)
    assert np.all(np.isfinite(model.correlation))


def test_fit_is_invariant_to_reviewer_and_example_permutation() -> None:
    reviewers = [
        _reviewer("c", lineage="third"),
        _reviewer("a", lineage="first"),
        _reviewer("b", lineage="second"),
    ]
    examples = _error_pattern_examples(
        {
            "a": (0, 0, 1, 1),
            "b": (0, 1, 0, 1),
            "c": (0, 1, 1, 0),
        }
    )

    first = fit_dependence(reviewers, examples, shrinkage=0.0, min_overlap=4)
    second = fit_dependence(
        list(reversed(reviewers)),
        list(reversed(examples)),
        shrinkage=0.0,
        min_overlap=4,
    )

    assert first.reviewer_ids == second.reviewer_ids == ("a", "b", "c")
    np.testing.assert_array_equal(first.correlation, second.correlation)
    assert (
        dict(first.lineage_by_reviewer)
        == dict(second.lineage_by_reviewer)
        == {
            "a": "first",
            "b": "second",
            "c": "third",
        }
    )


def test_pairwise_estimates_are_projected_to_a_correlation_matrix() -> None:
    examples: list[CalibrationExample] = []
    pair_patterns = (
        ("ab", "a", "b", (0, 0, 1, 1), (0, 0, 1, 1)),
        ("ac", "a", "c", (0, 0, 1, 1), (0, 0, 1, 1)),
        ("bc", "b", "c", (0, 0, 1, 1), (1, 1, 0, 0)),
    )
    for prefix, first_id, second_id, first_errors, second_errors in pair_patterns:
        examples.extend(
            _error_pattern_examples(
                {
                    first_id: first_errors,
                    second_id: second_errors,
                },
                case_prefix=prefix,
            )
        )

    model = fit_dependence(
        [_reviewer("a"), _reviewer("b"), _reviewer("c")],
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    assert np.all(np.isfinite(model.correlation))
    np.testing.assert_allclose(model.correlation, model.correlation.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(model.correlation), np.ones(3))
    assert np.linalg.eigvalsh(model.correlation).min() >= -1e-12


def test_effective_sample_size_is_sum_of_weights_and_within_bounds() -> None:
    reviewer_ids = ("a", "b", "c", "d")
    model = DependenceModel(
        reviewer_ids=reviewer_ids,
        correlation=np.array(
            [
                [1.0, 0.8, 0.0, -0.2],
                [0.8, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
                [-0.2, -0.2, 0.3, 1.0],
            ]
        ),
        lineage_by_reviewer={reviewer_id: reviewer_id for reviewer_id in reviewer_ids},
    )

    for subset_size in range(1, len(reviewer_ids) + 1):
        for subset in combinations(reviewer_ids, subset_size):
            weights = model.weights_for(subset)
            ess = model.effective_sample_size(subset)
            assert np.isfinite(ess)
            assert ess == pytest.approx(sum(weights.values()))
            assert 1.0 <= ess <= float(subset_size)


def test_empty_subset_has_no_weights_and_zero_effective_sample_size() -> None:
    model = DependenceModel(
        reviewer_ids=("a",),
        correlation=np.eye(1),
        lineage_by_reviewer={"a": "lineage"},
    )

    assert dict(model.weights_for(())) == {}
    assert model.effective_sample_size(()) == 0.0


def test_model_defensively_owns_immutable_correlation_and_mappings() -> None:
    correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
    lineages = {"a": "first", "b": "second"}
    model = DependenceModel(
        reviewer_ids=("a", "b"),
        correlation=correlation,
        lineage_by_reviewer=lineages,
    )
    correlation[0, 1] = 0.0
    lineages["a"] = "mutated"

    assert model.correlation[0, 1] == pytest.approx(0.3)
    assert dict(model.lineage_by_reviewer) == {"a": "first", "b": "second"}
    with pytest.raises(ValueError):
        model.correlation[0, 0] = 0.0
    with pytest.raises(ValueError):
        model.correlation.setflags(write=True)
    with pytest.raises(TypeError):
        model.lineage_by_reviewer["a"] = "mutated"  # type: ignore[index]
    weights = model.weights_for(("a", "b"))
    with pytest.raises(TypeError):
        weights["a"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        model.reviewer_ids = ("b", "a")


def test_weight_overrides_are_canonical_defensive_and_immutable() -> None:
    overrides = {("b", "a"): 0.75}
    model = DependenceModel(
        reviewer_ids=("a", "b"),
        correlation=np.eye(2),
        lineage_by_reviewer={"a": "first", "b": "second"},
        _weight_overrides=overrides,
    )
    overrides[("b", "a")] = 0.0

    assert dict(model._weight_overrides) == {("a", "b"): 0.75}
    assert model.weights_for(("a", "b"))["a"] == pytest.approx(1.0 / 1.75)
    with pytest.raises(TypeError):
        model._weight_overrides[("a", "b")] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({("a",): 0.5}, ValueError),
        ({("a", "missing"): 0.5}, ValueError),
        ({("a", "a"): 0.5}, ValueError),
        ({("a", "b"): 0.5, ("b", "a"): 0.5}, ValueError),
        ({("a", "b"): nan}, ValueError),
        ({("a", "b"): 1.1}, ValueError),
        ({("a", "b"): True}, TypeError),
    ],
)
def test_direct_model_rejects_malformed_weight_overrides(
    overrides: dict[tuple[str, ...], object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="weight override"):
        DependenceModel(
            reviewer_ids=("a", "b"),
            correlation=np.eye(2),
            lineage_by_reviewer={"a": "first", "b": "second"},
            _weight_overrides=overrides,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "lineages",
    [
        {"a": "first"},
        {"a": "first", "b": "second", "c": "extra"},
    ],
)
def test_direct_model_requires_exact_complete_lineage_keys(
    lineages: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="lineage_by_reviewer.*exactly"):
        DependenceModel(
            reviewer_ids=("a", "b"),
            correlation=np.eye(2),
            lineage_by_reviewer=lineages,
        )


@pytest.mark.parametrize(
    "correlation",
    [
        np.ones((2, 3)),
        np.array([[1.0, nan], [nan, 1.0]]),
        np.array([[1.0, 0.2], [0.1, 1.0]]),
        np.array([[0.9, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ],
)
def test_direct_model_rejects_malformed_correlation(
    correlation: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="correlation"):
        DependenceModel(
            reviewer_ids=("a", "b"),
            correlation=correlation,
            lineage_by_reviewer={"a": "first", "b": "second"},
        )


def test_direct_model_canonicalizes_tolerated_roundoff_above_one() -> None:
    model = DependenceModel(
        reviewer_ids=("a", "b"),
        correlation=np.array(
            [
                [1.0, 1.0 + 5e-11],
                [1.0 + 5e-11, 1.0],
            ]
        ),
        lineage_by_reviewer={"a": "shared", "b": "shared"},
    )

    assert np.max(model.correlation) <= 1.0
    assert np.min(model.correlation) >= -1.0
    assert np.linalg.eigvalsh(model.correlation).min() >= -1e-12
    assert model.effective_sample_size(("a", "b")) >= 1.0
    assert model.effective_sample_size(("a", "b")) == pytest.approx(1.0)


def test_many_fitted_clones_keep_effective_sample_size_at_least_one() -> None:
    reviewer_ids = tuple(f"r{index:03}" for index in range(18))
    reviewers = [
        _reviewer(reviewer_id, lineage="clone-lineage") for reviewer_id in reviewer_ids
    ]
    examples = _error_pattern_examples(
        {reviewer_id: (0, 0, 1, 1) for reviewer_id in reviewer_ids}
    )

    model = fit_dependence(
        reviewers,
        examples,
        shrinkage=0.0,
        min_overlap=4,
    )

    assert np.max(model.correlation) <= 1.0
    assert np.min(model.correlation) >= -1.0
    assert np.linalg.eigvalsh(model.correlation).min() >= -1e-12
    assert model.effective_sample_size(reviewer_ids) >= 1.0
    assert model.effective_sample_size(reviewer_ids) == pytest.approx(1.0)


def test_many_sparse_clone_fallbacks_have_strictly_bounded_ess() -> None:
    reviewer_ids = tuple(f"r{index:03}" for index in range(49))
    reviewers = [
        _reviewer(reviewer_id, lineage="clone-lineage") for reviewer_id in reviewer_ids
    ]

    model = fit_dependence(reviewers, [])

    assert model.effective_sample_size(reviewer_ids) == 1.0


def test_direct_model_rejects_duplicate_reviewer_ids() -> None:
    with pytest.raises(ValueError, match=r"duplicate reviewer_id.*a"):
        DependenceModel(
            reviewer_ids=("a", "a"),
            correlation=np.eye(2),
            lineage_by_reviewer={"a": "lineage"},
        )


def test_weights_reject_unknown_and_duplicate_queried_ids() -> None:
    model = DependenceModel(
        reviewer_ids=("a", "b"),
        correlation=np.eye(2),
        lineage_by_reviewer={"a": "first", "b": "second"},
    )

    with pytest.raises(ValueError, match=r"unknown reviewer IDs.*missing"):
        model.weights_for(("a", "missing"))
    with pytest.raises(ValueError, match=r"duplicate reviewer IDs.*a"):
        model.weights_for(("a", "a"))


def test_fit_rejects_duplicate_reviewer_ids() -> None:
    with pytest.raises(ValueError, match=r"duplicate reviewer_id.*a"):
        fit_dependence([_reviewer("a"), _reviewer("a")], [])


def test_fit_rejects_examples_from_unknown_reviewers() -> None:
    examples = [
        _example(
            "missing",
            Truth.PASS,
            Observation.PASS,
            case_id="case-1",
        )
    ]

    with pytest.raises(ValueError, match=r"unknown reviewer IDs.*missing"):
        fit_dependence([_reviewer("a")], examples)


def test_fit_rejects_duplicate_reviewer_case_keys() -> None:
    examples = [
        _example("a", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("a", Truth.PASS, Observation.FAIL, case_id="case-1"),
    ]

    with pytest.raises(
        ValueError,
        match=r"duplicate reviewer-case key.*a.*case-1",
    ):
        fit_dependence([_reviewer("a")], examples)


def test_fit_rejects_conflicting_truth_for_a_case() -> None:
    examples = [
        _example("a", Truth.PASS, Observation.PASS, case_id="case-1"),
        _example("b", Truth.FAIL, Observation.FAIL, case_id="case-1"),
    ]

    with pytest.raises(ValueError, match=r"conflicting truth.*case-1"):
        fit_dependence([_reviewer("a"), _reviewer("b")], examples)


@pytest.mark.parametrize("cost", [nan, inf])
def test_fit_rejects_non_finite_reviewer_cost(cost: float) -> None:
    with pytest.raises(ValueError, match=r"cost.*finite"):
        fit_dependence([_reviewer("a", cost=cost)], [])


def test_fit_rejects_non_string_lineage_with_an_actionable_error() -> None:
    reviewer = Reviewer(
        reviewer_id="a",
        vendor="vendor",
        family="family",
        lineage=42,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match=r"lineage.*string"):
        fit_dependence([reviewer], [])


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"shrinkage": -0.1}, ValueError, "shrinkage"),
        ({"shrinkage": 1.1}, ValueError, "shrinkage"),
        ({"shrinkage": nan}, ValueError, "shrinkage"),
        ({"shrinkage": inf}, ValueError, "shrinkage"),
        ({"shrinkage": True}, TypeError, "shrinkage"),
        ({"min_overlap": 0}, ValueError, "min_overlap"),
        ({"min_overlap": 1.5}, TypeError, "min_overlap"),
        ({"min_overlap": True}, TypeError, "min_overlap"),
        ({"lineage_cap": -0.1}, ValueError, "lineage_cap"),
        ({"lineage_cap": 1.1}, ValueError, "lineage_cap"),
        ({"lineage_cap": nan}, ValueError, "lineage_cap"),
        ({"lineage_cap": inf}, ValueError, "lineage_cap"),
        ({"lineage_cap": False}, TypeError, "lineage_cap"),
    ],
)
def test_fit_rejects_malformed_parameters(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        fit_dependence([_reviewer("a")], [], **kwargs)  # type: ignore[arg-type]
