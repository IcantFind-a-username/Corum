from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from corum.models import (
    CalibrationExample,
    ExecutionState,
    Observation,
    Reviewer,
    Truth,
)

_MATRIX_TOLERANCE = 1e-10


def _canonical_pair(first_id: str, second_id: str) -> tuple[str, str]:
    return (first_id, second_id) if first_id < second_id else (second_id, first_id)


def _validated_reviewer_ids(reviewer_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(reviewer_ids)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for reviewer_id in ids:
        if not isinstance(reviewer_id, str):
            raise TypeError("reviewer_ids must contain only strings")
        if not reviewer_id.strip():
            raise ValueError("reviewer_ids must not contain blank values")
        if reviewer_id in seen:
            duplicates.add(reviewer_id)
        seen.add(reviewer_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate reviewer_id values: {duplicate_list}")
    return ids


def _immutable_lineages(
    reviewer_ids: tuple[str, ...],
    lineage_by_reviewer: Mapping[str, str],
) -> Mapping[str, str]:
    if not isinstance(lineage_by_reviewer, Mapping):
        raise TypeError("lineage_by_reviewer must be a mapping")
    copied = dict(lineage_by_reviewer)
    expected_keys = set(reviewer_ids)
    actual_keys = set(copied)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(repr(key) for key in actual_keys - expected_keys)
        raise ValueError(
            "lineage_by_reviewer keys must match reviewer_ids exactly; "
            f"missing={missing!r}, extra={extra!r}"
        )

    ordered: dict[str, str] = {}
    for reviewer_id in reviewer_ids:
        lineage = copied[reviewer_id]
        if not isinstance(lineage, str):
            raise TypeError(
                "lineage_by_reviewer values must be strings; "
                f"reviewer_id={reviewer_id!r}"
            )
        if not lineage.strip():
            raise ValueError(
                "lineage_by_reviewer values must not be blank; "
                f"reviewer_id={reviewer_id!r}"
            )
        ordered[reviewer_id] = lineage
    return MappingProxyType(ordered)


def _immutable_weight_overrides(
    reviewer_ids: tuple[str, ...],
    weight_overrides: Mapping[tuple[str, str], float],
) -> Mapping[tuple[str, str], float]:
    if not isinstance(weight_overrides, Mapping):
        raise TypeError("weight overrides must be a mapping")
    known_ids = set(reviewer_ids)
    canonical_overrides: dict[tuple[str, str], float] = {}
    for pair, value in weight_overrides.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("weight override keys must be reviewer ID pairs")
        first_id, second_id = pair
        if not isinstance(first_id, str) or not isinstance(second_id, str):
            raise TypeError("weight override pair IDs must be strings")
        if first_id == second_id:
            raise ValueError("weight override pairs must contain distinct reviewer IDs")
        unknown_ids = sorted({first_id, second_id} - known_ids)
        if unknown_ids:
            unknown_list = ", ".join(unknown_ids)
            raise ValueError(
                f"weight override pair contains unknown reviewer IDs: {unknown_list}"
            )
        canonical_pair = _canonical_pair(first_id, second_id)
        if canonical_pair in canonical_overrides:
            raise ValueError(f"duplicate weight override pair: {canonical_pair!r}")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("weight override values must be real numbers")
        numeric_value = float(value)
        if not isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise ValueError("weight override values must be finite and within [0, 1]")
        canonical_overrides[canonical_pair] = numeric_value
    return MappingProxyType(dict(sorted(canonical_overrides.items())))


def _immutable_correlation(
    correlation: np.ndarray,
    reviewer_count: int,
) -> np.ndarray:
    try:
        matrix = np.array(correlation, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("correlation must be a numeric matrix") from error
    expected_shape = (reviewer_count, reviewer_count)
    if matrix.shape != expected_shape:
        raise ValueError(f"correlation must have shape {expected_shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must contain only finite values")
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=_MATRIX_TOLERANCE,
        atol=_MATRIX_TOLERANCE,
    ):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(
        np.diag(matrix),
        np.ones(reviewer_count),
        rtol=_MATRIX_TOLERANCE,
        atol=_MATRIX_TOLERANCE,
    ):
        raise ValueError("correlation diagonal must contain only ones")

    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 1.0)
    if reviewer_count:
        minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
        if minimum_eigenvalue < -_MATRIX_TOLERANCE:
            raise ValueError("correlation must be positive semidefinite")
        if minimum_eigenvalue < 0.0 or np.any(np.abs(matrix) > 1.0):
            matrix = _project_to_correlation(matrix)

    return np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)


@dataclass(frozen=True, slots=True)
class DependenceModel:
    reviewer_ids: tuple[str, ...]
    correlation: np.ndarray
    lineage_by_reviewer: Mapping[str, str]
    _weight_overrides: Mapping[tuple[str, str], float] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        reviewer_ids = _validated_reviewer_ids(self.reviewer_ids)
        correlation = _immutable_correlation(
            self.correlation,
            len(reviewer_ids),
        )
        lineages = _immutable_lineages(
            reviewer_ids,
            self.lineage_by_reviewer,
        )
        weight_overrides = _immutable_weight_overrides(
            reviewer_ids,
            self._weight_overrides,
        )
        object.__setattr__(self, "reviewer_ids", reviewer_ids)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "lineage_by_reviewer", lineages)
        object.__setattr__(self, "_weight_overrides", weight_overrides)

    def _validated_subset(self, reviewer_ids: Sequence[str]) -> tuple[str, ...]:
        queried_ids = tuple(reviewer_ids)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for reviewer_id in queried_ids:
            if not isinstance(reviewer_id, str):
                raise TypeError("queried reviewer IDs must be strings")
            if reviewer_id in seen:
                duplicates.add(reviewer_id)
            seen.add(reviewer_id)
        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate reviewer IDs: {duplicate_list}")

        known_ids = set(self.reviewer_ids)
        unknown = sorted(seen - known_ids)
        if unknown:
            unknown_list = ", ".join(unknown)
            raise ValueError(f"unknown reviewer IDs: {unknown_list}")
        return queried_ids

    def weights_for(self, reviewer_ids: Sequence[str]) -> Mapping[str, float]:
        queried_ids = self._validated_subset(reviewer_ids)
        index_by_id = {
            reviewer_id: index for index, reviewer_id in enumerate(self.reviewer_ids)
        }
        queried_indices = [index_by_id[reviewer_id] for reviewer_id in queried_ids]
        weights: dict[str, float] = {}
        for reviewer_id, reviewer_index in zip(
            queried_ids,
            queried_indices,
            strict=True,
        ):
            positive_correlation_sum = 0.0
            for other_id, other_index in zip(
                queried_ids,
                queried_indices,
                strict=True,
            ):
                if other_index == reviewer_index:
                    continue
                pair = _canonical_pair(reviewer_id, other_id)
                correlation = self._weight_overrides.get(
                    pair,
                    float(self.correlation[reviewer_index, other_index]),
                )
                positive_correlation_sum += min(max(correlation, 0.0), 1.0)
            weights[reviewer_id] = 1.0 / (1.0 + positive_correlation_sum)
        return MappingProxyType(weights)

    def effective_sample_size(self, reviewer_ids: Sequence[str]) -> float:
        weights = self.weights_for(reviewer_ids)
        reviewer_count = len(weights)
        if reviewer_count == 0:
            return 0.0
        effective_sample_size = float(sum(weights.values()))
        return min(max(effective_sample_size, 1.0), float(reviewer_count))


def _validate_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return numeric_value


def _validate_min_overlap(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("min_overlap must be an int")
    numeric_value = int(value)
    if numeric_value <= 0:
        raise ValueError("min_overlap must be positive")
    return numeric_value


def _validated_reviewers(reviewers: Sequence[Reviewer]) -> tuple[Reviewer, ...]:
    reviewer_by_id: dict[str, Reviewer] = {}
    duplicate_ids: set[str] = set()
    for index, reviewer in enumerate(reviewers):
        if not isinstance(reviewer, Reviewer):
            raise TypeError(f"reviewers[{index}] must be a Reviewer")
        if reviewer.reviewer_id in reviewer_by_id:
            duplicate_ids.add(reviewer.reviewer_id)
        reviewer_by_id[reviewer.reviewer_id] = reviewer
        if not isfinite(float(reviewer.cost)):
            raise ValueError(
                f"Reviewer.cost must be finite; reviewer_id={reviewer.reviewer_id!r}"
            )
        if not isinstance(reviewer.lineage, str):
            raise TypeError(
                "Reviewer.lineage must be a string; "
                f"reviewer_id={reviewer.reviewer_id!r}"
            )
        if not reviewer.lineage.strip():
            raise ValueError(
                "Reviewer.lineage must not be blank; "
                f"reviewer_id={reviewer.reviewer_id!r}"
            )
    if duplicate_ids:
        duplicate_list = ", ".join(sorted(duplicate_ids))
        raise ValueError(f"duplicate reviewer_id values: {duplicate_list}")
    return tuple(reviewer_by_id[key] for key in sorted(reviewer_by_id))


def _validate_examples(
    examples: Sequence[CalibrationExample],
    reviewer_ids: set[str],
) -> None:
    seen_reviewer_cases: set[tuple[str, str]] = set()
    truth_by_case: dict[str, Truth] = {}
    unknown_reviewer_ids: set[str] = set()
    for index, example in enumerate(examples):
        if not isinstance(example, CalibrationExample):
            raise TypeError(f"examples[{index}] must be a CalibrationExample")
        reviewer_id = example.review.reviewer_id
        case_id = example.review.case_id
        reviewer_case = (reviewer_id, case_id)
        if reviewer_case in seen_reviewer_cases:
            raise ValueError(
                "duplicate reviewer-case key: "
                f"reviewer_id={reviewer_id!r}, case_id={case_id!r}"
            )
        seen_reviewer_cases.add(reviewer_case)

        if case_id in truth_by_case and truth_by_case[case_id] is not example.truth:
            raise ValueError(
                f"conflicting truth for case_id={case_id!r}: "
                f"{truth_by_case[case_id].value} and {example.truth.value}"
            )
        truth_by_case[case_id] = example.truth
        if reviewer_id not in reviewer_ids:
            unknown_reviewer_ids.add(reviewer_id)

    if unknown_reviewer_ids:
        unknown_list = ", ".join(sorted(unknown_reviewer_ids))
        raise ValueError(f"examples contain unknown reviewer IDs: {unknown_list}")


def _semantic_error(observation: Observation, truth: Truth) -> int:
    if observation is Observation.ABSTAIN:
        return 1
    if observation is Observation.PASS:
        return int(truth is Truth.FAIL)
    if observation is Observation.FAIL:
        return int(truth is Truth.PASS)
    raise TypeError("valid observations must be an Observation")


def _empirical_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    if first.size == 0 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = sqrt(
        float(np.dot(first_centered, first_centered))
        * float(np.dot(second_centered, second_centered))
    )
    if denominator == 0.0 or not isfinite(denominator):
        return None
    correlation = float(np.dot(first_centered, second_centered)) / denominator
    if not isfinite(correlation):
        return None
    return min(max(correlation, -1.0), 1.0)


def _project_to_correlation(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.copy()
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped_eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * clipped_eigenvalues) @ eigenvectors.T
    diagonal = np.diag(projected)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("dependence projection produced an invalid diagonal")
    scales = np.sqrt(diagonal)
    projected = projected / np.outer(scales, scales)
    projected = (projected + projected.T) * 0.5
    np.fill_diagonal(projected, 1.0)
    projected = np.clip(projected, -1.0, 1.0)
    if not np.all(np.isfinite(projected)):
        raise ValueError("dependence projection produced non-finite values")
    if float(np.linalg.eigvalsh(projected).min()) < -_MATRIX_TOLERANCE:
        raise ValueError("dependence projection produced a non-PSD matrix")
    return projected


def fit_dependence(
    reviewers: Sequence[Reviewer],
    examples: Sequence[CalibrationExample],
    *,
    shrinkage: float = 0.25,
    min_overlap: int = 10,
    lineage_cap: float = 1.0,
) -> DependenceModel:
    shrinkage_value = _validate_unit_interval(shrinkage, "shrinkage")
    overlap_threshold = _validate_min_overlap(min_overlap)
    lineage_fallback = _validate_unit_interval(lineage_cap, "lineage_cap")
    ordered_reviewers = _validated_reviewers(reviewers)
    reviewer_ids = tuple(reviewer.reviewer_id for reviewer in ordered_reviewers)
    reviewer_id_set = set(reviewer_ids)
    _validate_examples(examples, reviewer_id_set)

    errors_by_reviewer: dict[str, dict[str, int]] = {
        reviewer_id: {} for reviewer_id in reviewer_ids
    }
    for example in examples:
        if example.review.state is not ExecutionState.VALID:
            continue
        observation = example.review.observation
        if observation is None:
            raise ValueError(
                "VALID calibration reviews require an observation; "
                f"reviewer_id={example.review.reviewer_id!r}, "
                f"case_id={example.review.case_id!r}"
            )
        errors_by_reviewer[example.review.reviewer_id][example.review.case_id] = (
            _semantic_error(observation, example.truth)
        )

    raw_correlation = np.eye(len(reviewer_ids), dtype=float)
    lineage_by_reviewer = {
        reviewer.reviewer_id: reviewer.lineage for reviewer in ordered_reviewers
    }
    weight_overrides: dict[tuple[str, str], float] = {}
    for first_index, first_id in enumerate(reviewer_ids):
        first_errors = errors_by_reviewer[first_id]
        for second_index in range(first_index + 1, len(reviewer_ids)):
            second_id = reviewer_ids[second_index]
            second_errors = errors_by_reviewer[second_id]
            overlap = sorted(first_errors.keys() & second_errors.keys())
            correlation: float | None = None
            if len(overlap) >= overlap_threshold:
                first_values = np.array(
                    [first_errors[case_id] for case_id in overlap],
                    dtype=float,
                )
                second_values = np.array(
                    [second_errors[case_id] for case_id in overlap],
                    dtype=float,
                )
                correlation = _empirical_correlation(first_values, second_values)

            if correlation is None:
                correlation = (
                    lineage_fallback
                    if lineage_by_reviewer[first_id] == lineage_by_reviewer[second_id]
                    else 0.0
                )
                weight_overrides[(first_id, second_id)] = correlation
            else:
                correlation *= 1.0 - shrinkage_value
            raw_correlation[first_index, second_index] = correlation
            raw_correlation[second_index, first_index] = correlation

    return DependenceModel(
        reviewer_ids=reviewer_ids,
        correlation=_project_to_correlation(raw_correlation),
        lineage_by_reviewer=lineage_by_reviewer,
        _weight_overrides=weight_overrides,
    )
