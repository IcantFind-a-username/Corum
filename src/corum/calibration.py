from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from corum.models import (
    CalibrationExample,
    ExecutionState,
    Observation,
    Reviewer,
    Truth,
)

OBSERVATION_ORDER: tuple[Observation, ...] = (
    Observation.PASS,
    Observation.FAIL,
    Observation.ABSTAIN,
)

_CALIBRATION_SHAPE = (2, 3)


def _validate_prior_strength(prior_strength: float) -> float:
    strength = float(prior_strength)
    if not isfinite(strength) or strength <= 0.0:
        raise ValueError("prior_strength must be positive and finite")
    return strength


def _normalized_parent_prior(parent_prior: np.ndarray | None) -> np.ndarray:
    if parent_prior is None:
        prior = np.ones(_CALIBRATION_SHAPE, dtype=float)
    else:
        prior = np.array(parent_prior, dtype=float, copy=True)
    if prior.shape != _CALIBRATION_SHAPE:
        raise ValueError("parent_prior must have shape (2, 3)")
    if not np.all(np.isfinite(prior)):
        raise ValueError("parent_prior must contain only finite values")
    if np.any(prior <= 0.0):
        raise ValueError("parent_prior must contain only positive values")
    row_totals = prior.sum(axis=1, keepdims=True)
    if np.any(row_totals <= 0.0):
        raise ValueError("parent_prior rows must have positive total mass")
    return prior / row_totals


def _validate_examples(examples: Sequence[CalibrationExample]) -> None:
    seen_reviewer_cases: set[tuple[str, str]] = set()
    truth_by_case: dict[str, Truth] = {}
    for example in examples:
        reviewer_case = (
            example.review.reviewer_id,
            example.review.case_id,
        )
        if reviewer_case in seen_reviewer_cases:
            reviewer_id, case_id = reviewer_case
            raise ValueError(
                "duplicate reviewer-case key: "
                f"reviewer_id={reviewer_id!r}, case_id={case_id!r}"
            )
        seen_reviewer_cases.add(reviewer_case)

        case_id = example.review.case_id
        if case_id in truth_by_case and truth_by_case[case_id] is not example.truth:
            raise ValueError(
                f"conflicting truth for case_id={case_id!r}: "
                f"{truth_by_case[case_id].value} and {example.truth.value}"
            )
        truth_by_case[case_id] = example.truth


@dataclass(frozen=True, slots=True)
class ReviewerCalibration:
    reviewer_id: str
    alpha: np.ndarray
    observed_counts: np.ndarray
    prior_strength: float

    def __post_init__(self) -> None:
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id must not be blank")
        strength = _validate_prior_strength(self.prior_strength)
        alpha = np.array(self.alpha, dtype=float, copy=True)
        observed_counts = np.array(self.observed_counts, copy=True)
        if alpha.shape != _CALIBRATION_SHAPE:
            raise ValueError("alpha must have shape (2, 3)")
        if observed_counts.shape != _CALIBRATION_SHAPE:
            raise ValueError("observed_counts must have shape (2, 3)")
        if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
            raise ValueError("alpha entries must be finite and strictly positive")
        if not np.all(np.isfinite(observed_counts)) or np.any(
            observed_counts < 0.0
        ):
            raise ValueError(
                "observed_counts entries must be finite and non-negative"
            )
        if not np.all(observed_counts == np.floor(observed_counts)):
            raise ValueError("observed_counts entries must be integer-valued")
        inferred_prior = alpha - observed_counts
        if np.any(inferred_prior <= 0.0):
            raise ValueError(
                "inferred prior alpha entries must be strictly positive"
            )
        if not np.allclose(
            inferred_prior.sum(axis=1),
            strength,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "inferred prior row sums must equal prior_strength"
            )
        alpha = np.frombuffer(alpha.tobytes(), dtype=alpha.dtype).reshape(alpha.shape)
        observed_counts = np.frombuffer(
            observed_counts.tobytes(),
            dtype=observed_counts.dtype,
        ).reshape(observed_counts.shape)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "observed_counts", observed_counts)
        object.__setattr__(self, "prior_strength", strength)

    def mean_likelihoods(self) -> np.ndarray:
        likelihoods = self.alpha / self.alpha.sum(axis=1, keepdims=True)
        return np.frombuffer(
            likelihoods.tobytes(),
            dtype=likelihoods.dtype,
        ).reshape(likelihoods.shape)

    def sample_likelihoods(
        self,
        draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if isinstance(draws, bool) or not isinstance(draws, int):
            raise TypeError("draws must be an int")
        if draws <= 0:
            raise ValueError("draws must be positive")
        samples = np.stack(
            [
                rng.dirichlet(self.alpha[truth_index], size=draws)
                for truth_index in range(_CALIBRATION_SHAPE[0])
            ],
            axis=1,
        )
        return np.frombuffer(
            samples.tobytes(),
            dtype=samples.dtype,
        ).reshape(samples.shape)


def fit_reviewer_calibration(
    reviewer_id: str,
    examples: Sequence[CalibrationExample],
    *,
    parent_prior: np.ndarray | None = None,
    prior_strength: float = 1.5,
) -> ReviewerCalibration:
    if not reviewer_id.strip():
        raise ValueError("reviewer_id must not be blank")
    strength = _validate_prior_strength(prior_strength)
    _validate_examples(examples)
    prior = _normalized_parent_prior(parent_prior) * strength
    observed_counts = np.zeros(_CALIBRATION_SHAPE, dtype=np.int64)

    for example in examples:
        if (
            example.review.reviewer_id != reviewer_id
            or example.review.state is not ExecutionState.VALID
        ):
            continue
        observation = example.review.observation
        if observation is None:
            continue
        truth_index = 0 if example.truth is Truth.PASS else 1
        observation_index = OBSERVATION_ORDER.index(observation)
        observed_counts[truth_index, observation_index] += 1

    return ReviewerCalibration(
        reviewer_id=reviewer_id,
        alpha=prior + observed_counts,
        observed_counts=observed_counts,
        prior_strength=strength,
    )


def fit_panel_calibrations(
    reviewers: Sequence[Reviewer],
    examples: Sequence[CalibrationExample],
    *,
    prior_strength: float = 1.5,
) -> dict[str, ReviewerCalibration]:
    strength = _validate_prior_strength(prior_strength)
    reviewer_ids: set[str] = set()
    duplicate_reviewer_ids: set[str] = set()
    for reviewer in reviewers:
        if reviewer.reviewer_id in reviewer_ids:
            duplicate_reviewer_ids.add(reviewer.reviewer_id)
        reviewer_ids.add(reviewer.reviewer_id)
    if duplicate_reviewer_ids:
        duplicates = ", ".join(sorted(duplicate_reviewer_ids))
        raise ValueError(f"duplicate reviewer_id values: {duplicates}")

    _validate_examples(examples)
    unknown_reviewer_ids = sorted(
        {
            example.review.reviewer_id
            for example in examples
            if example.review.reviewer_id not in reviewer_ids
        }
    )
    if unknown_reviewer_ids:
        unknown = ", ".join(unknown_reviewer_ids)
        raise ValueError(f"examples contain unknown reviewer IDs: {unknown}")

    pooled_counts = np.zeros(_CALIBRATION_SHAPE, dtype=np.int64)
    for example in examples:
        if example.review.state is not ExecutionState.VALID:
            continue
        observation = example.review.observation
        if observation is None:
            continue
        truth_index = 0 if example.truth is Truth.PASS else 1
        observation_index = OBSERVATION_ORDER.index(observation)
        pooled_counts[truth_index, observation_index] += 1

    smoothed_parent_prior = pooled_counts.astype(float) + 1.0
    return {
        reviewer.reviewer_id: fit_reviewer_calibration(
            reviewer.reviewer_id,
            examples,
            parent_prior=smoothed_parent_prior,
            prior_strength=strength,
        )
        for reviewer in reviewers
    }
