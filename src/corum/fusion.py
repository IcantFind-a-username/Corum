from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from corum.calibration import (
    OBSERVATION_ORDER,
    PairKey,
    ReviewerCalibration,
    ReviewerPairCalibration,
)
from corum.dependence import DependenceModel
from corum.models import ExecutionState, FusedPosterior, Observation, Review

_LIKELIHOOD_SHAPE = (2, 3)
_JOINT_LIKELIHOOD_SHAPE = (2, 3, 3)
_MIN_PROBABILITY = np.finfo(np.float64).tiny


def _validate_open_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{name} must be finite and representable as a float"
        ) from error
    if not isfinite(numeric_value) or not 0.0 < numeric_value < 1.0:
        raise ValueError(f"{name} must be finite and strictly within (0, 1)")
    return numeric_value


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an int")
    numeric_value = int(value)
    if numeric_value <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric_value


def _read_only_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _validated_pair_key(value: object) -> PairKey:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("pair key must be a two-item tuple")
    first, second = value
    if not isinstance(first, str) or not isinstance(second, str):
        raise TypeError("pair key reviewer IDs must be strings")
    if not first.strip() or not second.strip():
        raise ValueError("pair key reviewer IDs must not be blank")
    if first == second:
        raise ValueError("pair key reviewer IDs must be distinct")
    if first > second:
        raise ValueError("pair key must use canonical sorted reviewer order")
    return first, second


def _validated_pair_partition(
    raw_pairs: Sequence[object],
    reviewer_ids: tuple[str, ...],
) -> tuple[PairKey, ...]:
    known_ids = set(reviewer_ids)
    used_ids: set[str] = set()
    pairs: list[PairKey] = []
    for raw_pair in raw_pairs:
        pair = _validated_pair_key(raw_pair)
        unknown = sorted(set(pair) - known_ids)
        if unknown:
            raise ValueError("unknown reviewer IDs in pair key: " + ", ".join(unknown))
        overlap = sorted(set(pair) & used_ids)
        if overlap:
            raise ValueError(
                "pair keys must not overlap; repeated reviewer IDs: "
                + ", ".join(overlap)
            )
        used_ids.update(pair)
        pairs.append(pair)
    return tuple(sorted(pairs))


def _validated_joint_likelihood(
    value: object,
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.shape != _JOINT_LIKELIHOOD_SHAPE:
        raise ValueError(f"{name} must have shape (2, 3, 3)")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must be finite probabilities")
    if not np.allclose(array.sum(axis=(1, 2)), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} truth rows must sum to one")
    return array


def _validated_joint_likelihood_draws(
    value: object,
    *,
    name: str,
    expected_draws: int,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != 4 or array.shape[1:] != _JOINT_LIKELIHOOD_SHAPE:
        raise ValueError(f"{name} must have shape (draws, 2, 3, 3)")
    if array.shape[0] != expected_draws:
        raise ValueError("all pair likelihood arrays must use the same draw count")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must be finite probabilities")
    if not np.allclose(array.sum(axis=(2, 3)), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} truth rows must sum to one")
    return _read_only_copy(array)


def _validated_likelihood_array(
    value: object,
    *,
    name: str,
    expected_draws: int | None = None,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != 3 or array.shape[1:] != _LIKELIHOOD_SHAPE:
        raise ValueError(f"{name} must have shape (draws, 2, 3)")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one draw")
    if expected_draws is not None and array.shape[0] != expected_draws:
        raise ValueError("all likelihood draw arrays must use the same draw count")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must be finite probabilities")
    if not np.allclose(array.sum(axis=2), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} truth rows must sum to one")
    return _read_only_copy(array)


def _immutable_likelihood_draws(
    likelihood_draws: Mapping[str, np.ndarray],
    reviewer_ids: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    if not isinstance(likelihood_draws, Mapping):
        raise TypeError("likelihood_draws must be a mapping")
    expected_keys = set(reviewer_ids)
    actual_keys = set(likelihood_draws)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(repr(key) for key in actual_keys - expected_keys)
        raise ValueError(
            "likelihood draw reviewer IDs must match dependence reviewer IDs exactly; "
            f"missing={missing!r}, extra={extra!r}"
        )
    copied: dict[str, np.ndarray] = {}
    draw_count: int | None = None
    for reviewer_id in reviewer_ids:
        array = _validated_likelihood_array(
            likelihood_draws[reviewer_id],
            name=f"likelihood_draws[{reviewer_id!r}]",
            expected_draws=draw_count,
        )
        if draw_count is None:
            draw_count = array.shape[0]
        copied[reviewer_id] = array
    return MappingProxyType(copied)


def _immutable_pair_likelihood_draws(
    pair_likelihood_draws: Mapping[PairKey, np.ndarray],
    reviewer_ids: tuple[str, ...],
    *,
    expected_draws: int,
) -> Mapping[PairKey, np.ndarray]:
    if not isinstance(pair_likelihood_draws, Mapping):
        raise TypeError("pair_likelihood_draws must be a mapping")
    pairs = _validated_pair_partition(
        tuple(pair_likelihood_draws),
        reviewer_ids,
    )
    copied = {
        pair: _validated_joint_likelihood_draws(
            pair_likelihood_draws[pair],
            name=f"pair_likelihood_draws[{pair!r}]",
            expected_draws=expected_draws,
        )
        for pair in pairs
    }
    return MappingProxyType(copied)


def _validated_pair_calibration_registry(
    pair_calibrations: Mapping[PairKey, ReviewerPairCalibration] | None,
    reviewer_ids: tuple[str, ...],
) -> Mapping[PairKey, ReviewerPairCalibration]:
    if pair_calibrations is None:
        return MappingProxyType({})
    if not isinstance(pair_calibrations, Mapping):
        raise TypeError("pair_calibrations must be a mapping or None")
    pairs = _validated_pair_partition(tuple(pair_calibrations), reviewer_ids)
    copied: dict[PairKey, ReviewerPairCalibration] = {}
    for pair in pairs:
        calibration = pair_calibrations[pair]
        if not isinstance(calibration, ReviewerPairCalibration):
            raise TypeError(
                f"pair_calibrations[{pair!r}] must be a ReviewerPairCalibration"
            )
        if calibration.reviewer_ids != pair:
            raise ValueError(
                "pair calibration mapping key must match record reviewer_ids; "
                f"key={pair!r}, reviewer_ids={calibration.reviewer_ids!r}"
            )
        copied[pair] = calibration
    return MappingProxyType(copied)


def _immutable_lineages(
    lineage_by_reviewer: Mapping[str, str],
    dependence: DependenceModel,
) -> Mapping[str, str]:
    if not isinstance(lineage_by_reviewer, Mapping):
        raise TypeError("lineage_by_reviewer must be a mapping")
    copied = dict(lineage_by_reviewer)
    expected = dict(dependence.lineage_by_reviewer)
    if copied != expected:
        raise ValueError("lineage_by_reviewer must match dependence lineage metadata")
    return MappingProxyType(
        {reviewer_id: copied[reviewer_id] for reviewer_id in dependence.reviewer_ids}
    )


@dataclass(frozen=True, slots=True)
class FusionContext:
    likelihood_draws: Mapping[str, np.ndarray]
    dependence: DependenceModel
    lineage_by_reviewer: Mapping[str, str]
    prior_pass: float
    credible_mass: float
    pair_likelihood_draws: Mapping[PairKey, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dependence, DependenceModel):
            raise TypeError("dependence must be a DependenceModel")
        if not self.dependence.reviewer_ids:
            raise ValueError("dependence must contain at least one reviewer")
        prior_pass = _validate_open_probability(self.prior_pass, "prior_pass")
        credible_mass = _validate_open_probability(
            self.credible_mass,
            "credible_mass",
        )
        likelihood_draws = _immutable_likelihood_draws(
            self.likelihood_draws,
            self.dependence.reviewer_ids,
        )
        lineages = _immutable_lineages(
            self.lineage_by_reviewer,
            self.dependence,
        )
        first_id = self.dependence.reviewer_ids[0]
        pair_likelihood_draws = _immutable_pair_likelihood_draws(
            self.pair_likelihood_draws,
            self.dependence.reviewer_ids,
            expected_draws=int(likelihood_draws[first_id].shape[0]),
        )
        object.__setattr__(self, "likelihood_draws", likelihood_draws)
        object.__setattr__(
            self,
            "pair_likelihood_draws",
            pair_likelihood_draws,
        )
        object.__setattr__(self, "lineage_by_reviewer", lineages)
        object.__setattr__(self, "prior_pass", prior_pass)
        object.__setattr__(self, "credible_mass", credible_mass)

    @property
    def draws(self) -> int:
        first_id = self.dependence.reviewer_ids[0]
        return int(self.likelihood_draws[first_id].shape[0])


def build_fusion_context(
    calibrations: Mapping[str, ReviewerCalibration],
    dependence: DependenceModel,
    *,
    prior_pass: float,
    draws: int = 512,
    credible_mass: float = 0.95,
    seed: int,
    pair_calibrations: Mapping[PairKey, ReviewerPairCalibration] | None = None,
) -> FusionContext:
    if not isinstance(calibrations, Mapping):
        raise TypeError("calibrations must be a mapping")
    if not isinstance(dependence, DependenceModel):
        raise TypeError("dependence must be a DependenceModel")
    draw_count = _validate_positive_integer(draws, "draws")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an int")
    expected_ids = set(dependence.reviewer_ids)
    actual_ids = set(calibrations)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(repr(key) for key in actual_ids - expected_ids)
        raise ValueError(
            "calibration reviewer IDs must match dependence reviewer IDs exactly; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for reviewer_id in dependence.reviewer_ids:
        calibration = calibrations[reviewer_id]
        if not isinstance(calibration, ReviewerCalibration):
            raise TypeError(
                f"calibrations[{reviewer_id!r}] must be a ReviewerCalibration"
            )
        if calibration.reviewer_id != reviewer_id:
            raise ValueError(
                "calibration mapping key must match calibration.reviewer_id; "
                f"key={reviewer_id!r}, reviewer_id={calibration.reviewer_id!r}"
            )
    checked_pair_calibrations = _validated_pair_calibration_registry(
        pair_calibrations,
        dependence.reviewer_ids,
    )
    rng = np.random.default_rng(int(seed))
    likelihood_draws = {
        reviewer_id: calibrations[reviewer_id].sample_likelihoods(draw_count, rng)
        for reviewer_id in dependence.reviewer_ids
    }
    pair_likelihood_draws = {
        pair: checked_pair_calibrations[pair].sample_likelihoods(draw_count, rng)
        for pair in checked_pair_calibrations
    }
    return FusionContext(
        likelihood_draws=likelihood_draws,
        dependence=dependence,
        lineage_by_reviewer=dependence.lineage_by_reviewer,
        prior_pass=prior_pass,
        credible_mass=credible_mass,
        pair_likelihood_draws=pair_likelihood_draws,
    )


def _validated_known_likelihood(
    likelihood: object,
    reviewer_id: str,
) -> np.ndarray:
    try:
        array = np.array(likelihood, dtype=float, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"likelihoods[{reviewer_id!r}] must be a numeric array"
        ) from error
    if array.shape != _LIKELIHOOD_SHAPE:
        raise ValueError(f"likelihoods[{reviewer_id!r}] must have shape (2, 3)")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(
            f"likelihoods[{reviewer_id!r}] entries must be finite probabilities"
        )
    if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"likelihoods[{reviewer_id!r}] truth rows must sum to one")
    return array


def fuse_known_likelihoods(
    observations: Mapping[str, Observation],
    likelihoods: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    prior_pass: float,
) -> float:
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping")
    if not isinstance(likelihoods, Mapping):
        raise TypeError("likelihoods must be a mapping")
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    prior = _validate_open_probability(prior_pass, "prior_pass")
    observation_ids = set(observations)
    if set(weights) != observation_ids:
        raise ValueError("weight reviewer IDs must match observation reviewer IDs")
    missing_likelihoods = sorted(observation_ids - set(likelihoods))
    if missing_likelihoods:
        raise ValueError(
            "missing likelihoods for reviewer IDs: " + ", ".join(missing_likelihoods)
        )

    log_pass = log(prior)
    log_fail = log1p_negative(prior)
    for reviewer_id in sorted(observation_ids):
        observation = observations[reviewer_id]
        if not isinstance(observation, Observation):
            raise TypeError(f"observations[{reviewer_id!r}] must be an Observation")
        weight = weights[reviewer_id]
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError(f"weights[{reviewer_id!r}] must be a real number")
        try:
            numeric_weight = float(weight)
        except OverflowError as error:
            raise ValueError(
                f"weights[{reviewer_id!r}] must be finite and representable as a float"
            ) from error
        if not isfinite(numeric_weight) or not 0.0 <= numeric_weight <= 1.0:
            raise ValueError(
                f"weights[{reviewer_id!r}] must be finite and within [0, 1]"
            )
        likelihood = _validated_known_likelihood(
            likelihoods[reviewer_id],
            reviewer_id,
        )
        observation_index = OBSERVATION_ORDER.index(observation)
        pass_likelihood = max(
            float(likelihood[0, observation_index]),
            _MIN_PROBABILITY,
        )
        fail_likelihood = max(
            float(likelihood[1, observation_index]),
            _MIN_PROBABILITY,
        )
        log_pass += numeric_weight * log(pass_likelihood)
        log_fail += numeric_weight * log(fail_likelihood)

    maximum = max(log_pass, log_fail)
    pass_mass = exp(log_pass - maximum)
    fail_mass = exp(log_fail - maximum)
    return pass_mass / (pass_mass + fail_mass)


def fuse_known_pair_likelihoods(
    observations: Mapping[str, Observation],
    likelihoods: Mapping[str, np.ndarray],
    pair_likelihoods: Mapping[PairKey, np.ndarray],
    *,
    prior_pass: float,
) -> float:
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping")
    if not isinstance(likelihoods, Mapping):
        raise TypeError("likelihoods must be a mapping")
    if not isinstance(pair_likelihoods, Mapping):
        raise TypeError("pair_likelihoods must be a mapping")
    prior = _validate_open_probability(prior_pass, "prior_pass")
    observation_ids = set(observations)
    missing_likelihoods = sorted(observation_ids - set(likelihoods))
    if missing_likelihoods:
        raise ValueError(
            "missing likelihoods for reviewer IDs: " + ", ".join(missing_likelihoods)
        )
    observation_codes: dict[str, int] = {}
    validated_singletons: dict[str, np.ndarray] = {}
    for reviewer_id in sorted(observation_ids):
        observation = observations[reviewer_id]
        if not isinstance(observation, Observation):
            raise TypeError(f"observations[{reviewer_id!r}] must be an Observation")
        observation_codes[reviewer_id] = OBSERVATION_ORDER.index(observation)
        validated_singletons[reviewer_id] = _validated_known_likelihood(
            likelihoods[reviewer_id],
            reviewer_id,
        )

    pairs = _validated_pair_partition(
        tuple(pair_likelihoods),
        tuple(likelihoods),
    )
    validated_pairs = {
        pair: _validated_joint_likelihood(
            pair_likelihoods[pair],
            name=f"pair_likelihoods[{pair!r}]",
        )
        for pair in pairs
    }

    log_pass = log(prior)
    log_fail = log1p_negative(prior)
    used_ids: set[str] = set()
    for pair in pairs:
        first, second = pair
        if first not in observation_codes or second not in observation_codes:
            continue
        joint = validated_pairs[pair]
        first_code = observation_codes[first]
        second_code = observation_codes[second]
        log_pass += log(
            max(
                float(joint[0, first_code, second_code]),
                _MIN_PROBABILITY,
            )
        )
        log_fail += log(
            max(
                float(joint[1, first_code, second_code]),
                _MIN_PROBABILITY,
            )
        )
        used_ids.update(pair)

    for reviewer_id in sorted(observation_ids - used_ids):
        likelihood = validated_singletons[reviewer_id]
        observation_index = observation_codes[reviewer_id]
        log_pass += log(max(float(likelihood[0, observation_index]), _MIN_PROBABILITY))
        log_fail += log(max(float(likelihood[1, observation_index]), _MIN_PROBABILITY))

    maximum = max(log_pass, log_fail)
    pass_mass = exp(log_pass - maximum)
    fail_mass = exp(log_fail - maximum)
    return pass_mass / (pass_mass + fail_mass)


def log1p_negative(probability: float) -> float:
    return log(1.0 - probability)


def _immutable_vector(
    value: object,
    *,
    name: str,
    integer: bool = False,
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an array") from error
    if source.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if integer:
        if np.issubdtype(source.dtype, np.bool_) or not np.issubdtype(
            source.dtype,
            np.integer,
        ):
            raise TypeError(f"{name} must contain integers")
        array = np.array(source, dtype=np.int64, copy=True)
    else:
        try:
            array = np.array(source, dtype=float, copy=True)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric values") from error
    return _read_only_copy(array)


@dataclass(frozen=True, slots=True)
class BatchFusedPosterior:
    pass_probability: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    valid_reviewers: np.ndarray
    lineage_count: np.ndarray
    effective_sample_size: np.ndarray

    def __post_init__(self) -> None:
        pass_probability = _immutable_vector(
            self.pass_probability,
            name="pass_probability",
        )
        lower = _immutable_vector(self.lower, name="lower")
        upper = _immutable_vector(self.upper, name="upper")
        valid_reviewers = _immutable_vector(
            self.valid_reviewers,
            name="valid_reviewers",
            integer=True,
        )
        lineage_count = _immutable_vector(
            self.lineage_count,
            name="lineage_count",
            integer=True,
        )
        effective_sample_size = _immutable_vector(
            self.effective_sample_size,
            name="effective_sample_size",
        )
        arrays = (
            pass_probability,
            lower,
            upper,
            valid_reviewers,
            lineage_count,
            effective_sample_size,
        )
        lengths = {array.shape[0] for array in arrays}
        if len(lengths) != 1:
            raise ValueError("batch posterior arrays must have equal length")
        if np.any(valid_reviewers < 0) or np.any(lineage_count < 0):
            raise ValueError("batch quorum counts must be non-negative")
        empty = valid_reviewers == 0
        if np.any(lineage_count[empty] != 0) or np.any(
            effective_sample_size[empty] != 0.0
        ):
            raise ValueError("empty batch rows must have zero quorum metadata")
        non_empty = ~empty
        if np.any(lineage_count[non_empty] < 1) or np.any(
            lineage_count[non_empty] > valid_reviewers[non_empty]
        ):
            raise ValueError("non-empty batch rows have inconsistent lineage counts")
        if np.any(~np.isfinite(effective_sample_size)) or np.any(
            effective_sample_size < 0.0
        ):
            raise ValueError("effective_sample_size must be finite and non-negative")
        if np.any(effective_sample_size[non_empty] < 1.0) or np.any(
            effective_sample_size[non_empty] > valid_reviewers[non_empty]
        ):
            raise ValueError("non-empty batch rows have inconsistent ESS")
        for name, array in (
            ("pass_probability", pass_probability),
            ("lower", lower),
            ("upper", upper),
        ):
            if np.any(~np.isnan(array[empty])):
                raise ValueError(f"empty batch rows must have NaN {name}")
            values = array[non_empty]
            if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(
                    f"non-empty {name} values must be finite probabilities"
                )
        if np.any(lower[non_empty] > upper[non_empty]):
            raise ValueError("batch lower values must not exceed upper values")
        object.__setattr__(self, "pass_probability", pass_probability)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "valid_reviewers", valid_reviewers)
        object.__setattr__(self, "lineage_count", lineage_count)
        object.__setattr__(self, "effective_sample_size", effective_sample_size)


def _validated_reviewer_ids(
    reviewer_ids: Sequence[str],
    context: FusionContext,
) -> tuple[str, ...]:
    ids = tuple(reviewer_ids)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for reviewer_id in ids:
        if not isinstance(reviewer_id, str):
            raise TypeError("reviewer_ids must contain only strings")
        if reviewer_id in seen:
            duplicates.add(reviewer_id)
        seen.add(reviewer_id)
    if duplicates:
        raise ValueError("duplicate reviewer IDs: " + ", ".join(sorted(duplicates)))
    unknown = sorted(seen - set(context.dependence.reviewer_ids))
    if unknown:
        raise ValueError("unknown reviewer IDs: " + ", ".join(unknown))
    return ids


def _validated_matrix_inputs(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    observation_array = np.asarray(observations)
    mask_array = np.asarray(valid_mask)
    if observation_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("observations and valid_mask must be two-dimensional")
    if observation_array.shape != mask_array.shape:
        raise ValueError("observations and valid_mask must have the same shape")
    if observation_array.shape[1] != len(reviewer_ids):
        raise ValueError("matrix column count must match reviewer_ids")
    if np.issubdtype(observation_array.dtype, np.bool_) or not np.issubdtype(
        observation_array.dtype,
        np.integer,
    ):
        raise TypeError("observations must contain integer codes")
    if not np.issubdtype(mask_array.dtype, np.bool_):
        raise TypeError("valid_mask must contain booleans")
    valid_codes = observation_array[mask_array]
    if np.any((valid_codes < 0) | (valid_codes >= len(OBSERVATION_ORDER))):
        raise ValueError("valid observations must use codes 0, 1, or 2")
    return (
        np.ascontiguousarray(observation_array, dtype=np.int64),
        np.ascontiguousarray(mask_array, dtype=bool),
    )


def _fuse_matrix_kernel(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: tuple[str, ...],
    context: FusionContext,
    *,
    chunk_size: int,
    include_samples: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    case_count = observations.shape[0]
    draw_count = context.draws
    pass_probability = np.full(case_count, np.nan, dtype=np.float64)
    lower = np.full(case_count, np.nan, dtype=np.float64)
    upper = np.full(case_count, np.nan, dtype=np.float64)
    valid_reviewers = np.zeros(case_count, dtype=np.int64)
    lineage_count = np.zeros(case_count, dtype=np.int64)
    effective_sample_size = np.zeros(case_count, dtype=np.float64)
    samples = (
        np.full((case_count, draw_count), np.nan, dtype=np.float64)
        if include_samples
        else None
    )
    lower_quantile = (1.0 - context.credible_mass) * 0.5
    upper_quantile = 1.0 - lower_quantile

    for chunk_start in range(0, case_count, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, case_count)
        chunk_observations = observations[chunk_start:chunk_stop]
        chunk_mask = valid_mask[chunk_start:chunk_stop]
        patterns, inverse = np.unique(chunk_mask, axis=0, return_inverse=True)
        for pattern_index, pattern in enumerate(patterns):
            local_rows = np.flatnonzero(inverse == pattern_index)
            subset_ids = tuple(
                reviewer_id
                for reviewer_id, contributes in zip(
                    reviewer_ids,
                    pattern,
                    strict=True,
                )
                if contributes
            )
            reviewer_count = len(subset_ids)
            if reviewer_count == 0:
                continue
            contributing_columns = np.flatnonzero(pattern)
            contributing_codes = chunk_observations[
                np.ix_(local_rows, contributing_columns)
            ]
            all_abstain = np.all(
                contributing_codes == OBSERVATION_ORDER.index(Observation.ABSTAIN),
                axis=1,
            )
            global_rows = chunk_start + local_rows
            log_pass = np.full(
                (local_rows.size, draw_count),
                log(context.prior_pass),
                dtype=np.float64,
            )
            log_fail = np.full(
                (local_rows.size, draw_count),
                log1p_negative(context.prior_pass),
                dtype=np.float64,
            )
            if not context.pair_likelihood_draws:
                weights = context.dependence.weights_for(subset_ids)
                for reviewer_index, reviewer_id in enumerate(reviewer_ids):
                    if not pattern[reviewer_index]:
                        continue
                    codes = chunk_observations[local_rows, reviewer_index]
                    reviewer_draws = context.likelihood_draws[reviewer_id]
                    pass_values = np.take(reviewer_draws[:, 0, :], codes, axis=1).T
                    fail_values = np.take(reviewer_draws[:, 1, :], codes, axis=1).T
                    weight = weights[reviewer_id]
                    log_pass += weight * np.log(
                        np.maximum(pass_values, _MIN_PROBABILITY)
                    )
                    log_fail += weight * np.log(
                        np.maximum(fail_values, _MIN_PROBABILITY)
                    )
            else:
                column_by_reviewer = {
                    reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
                }
                used_pair_members: set[str] = set()
                for pair, joint_draws in context.pair_likelihood_draws.items():
                    first_index = column_by_reviewer.get(pair[0])
                    second_index = column_by_reviewer.get(pair[1])
                    first_valid = first_index is not None and bool(pattern[first_index])
                    second_valid = second_index is not None and bool(
                        pattern[second_index]
                    )
                    if not first_valid or not second_valid:
                        continue
                    if first_index is None or second_index is None:
                        raise RuntimeError("valid pair member lost its matrix column")
                    first_codes = chunk_observations[local_rows, first_index]
                    second_codes = chunk_observations[local_rows, second_index]
                    pass_values = joint_draws[:, 0, first_codes, second_codes].T
                    fail_values = joint_draws[:, 1, first_codes, second_codes].T
                    log_pass += np.log(np.maximum(pass_values, _MIN_PROBABILITY))
                    log_fail += np.log(np.maximum(fail_values, _MIN_PROBABILITY))
                    used_pair_members.update(pair)

                for reviewer_index, reviewer_id in enumerate(reviewer_ids):
                    if not pattern[reviewer_index] or reviewer_id in used_pair_members:
                        continue
                    codes = chunk_observations[local_rows, reviewer_index]
                    reviewer_draws = context.likelihood_draws[reviewer_id]
                    pass_values = np.take(reviewer_draws[:, 0, :], codes, axis=1).T
                    fail_values = np.take(reviewer_draws[:, 1, :], codes, axis=1).T
                    log_pass += np.log(np.maximum(pass_values, _MIN_PROBABILITY))
                    log_fail += np.log(np.maximum(fail_values, _MIN_PROBABILITY))
            maximum = np.maximum(log_pass, log_fail)
            pass_mass = np.exp(log_pass - maximum)
            fail_mass = np.exp(log_fail - maximum)
            posterior_samples = pass_mass / (pass_mass + fail_mass)
            quantiles = np.quantile(
                posterior_samples,
                [lower_quantile, upper_quantile],
                axis=1,
            )
            # Calibrated abstention remains diagnostic evidence, but a panel
            # with no affirmative PASS/FAIL vote must never authorize an
            # action. Widening only its decision bounds preserves the learned
            # posterior samples and truthful quorum metadata while ensuring
            # every valid DecisionPolicy defers.
            quantiles[0, all_abstain] = 0.0
            quantiles[1, all_abstain] = 1.0
            pass_probability[global_rows] = posterior_samples.mean(axis=1)
            lower[global_rows] = quantiles[0]
            upper[global_rows] = quantiles[1]
            valid_reviewers[global_rows] = reviewer_count
            lineage_count[global_rows] = len(
                {context.lineage_by_reviewer[reviewer_id] for reviewer_id in subset_ids}
            )
            effective_sample_size[global_rows] = (
                context.dependence.effective_sample_size(subset_ids)
            )
            if samples is not None:
                samples[global_rows] = posterior_samples

    return (
        pass_probability,
        lower,
        upper,
        valid_reviewers,
        lineage_count,
        effective_sample_size,
        samples,
    )


def fuse_review_matrix(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: Sequence[str],
    context: FusionContext,
    *,
    chunk_size: int = 4_096,
) -> BatchFusedPosterior:
    if not isinstance(context, FusionContext):
        raise TypeError("context must be a FusionContext")
    ids = _validated_reviewer_ids(reviewer_ids, context)
    observation_array, mask_array = _validated_matrix_inputs(
        observations,
        valid_mask,
        ids,
    )
    canonical_ids = tuple(
        reviewer_id
        for reviewer_id in context.dependence.reviewer_ids
        if reviewer_id in set(ids)
    )
    if ids != canonical_ids:
        input_index = {reviewer_id: index for index, reviewer_id in enumerate(ids)}
        canonical_columns = [input_index[reviewer_id] for reviewer_id in canonical_ids]
        observation_array = np.ascontiguousarray(
            observation_array[:, canonical_columns]
        )
        mask_array = np.ascontiguousarray(mask_array[:, canonical_columns])
    chunk = _validate_positive_integer(chunk_size, "chunk_size")
    summaries = _fuse_matrix_kernel(
        observation_array,
        mask_array,
        canonical_ids,
        context,
        chunk_size=chunk,
        include_samples=False,
    )
    return BatchFusedPosterior(
        pass_probability=summaries[0],
        lower=summaries[1],
        upper=summaries[2],
        valid_reviewers=summaries[3],
        lineage_count=summaries[4],
        effective_sample_size=summaries[5],
    )


def fuse_reviews(
    reviews: Sequence[Review],
    context: FusionContext,
) -> FusedPosterior | None:
    if not isinstance(context, FusionContext):
        raise TypeError("context must be a FusionContext")
    review_tuple = tuple(reviews)
    seen_ids: set[str] = set()
    duplicates: set[str] = set()
    case_ids: set[str] = set()
    for index, review in enumerate(review_tuple):
        if not isinstance(review, Review):
            raise TypeError(f"reviews[{index}] must be a Review")
        if review.reviewer_id in seen_ids:
            duplicates.add(review.reviewer_id)
        seen_ids.add(review.reviewer_id)
        case_ids.add(review.case_id)
    if duplicates:
        raise ValueError("duplicate reviewer IDs: " + ", ".join(sorted(duplicates)))
    if len(case_ids) > 1:
        raise ValueError("fuse_reviews accepts reviews for exactly one case_id")
    unknown = sorted(seen_ids - set(context.dependence.reviewer_ids))
    if unknown:
        raise ValueError("unknown reviewer IDs: " + ", ".join(unknown))

    reviewer_ids = context.dependence.reviewer_ids
    index_by_id = {reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)}
    observation_codes = np.full((1, len(reviewer_ids)), -1, dtype=np.int64)
    valid_mask = np.zeros((1, len(reviewer_ids)), dtype=bool)
    for review in review_tuple:
        if review.state is not ExecutionState.VALID:
            continue
        observation = review.observation
        if observation is None:
            raise ValueError("VALID reviews require an observation")
        reviewer_index = index_by_id[review.reviewer_id]
        observation_codes[0, reviewer_index] = OBSERVATION_ORDER.index(observation)
        valid_mask[0, reviewer_index] = True
    if not np.any(valid_mask):
        return None

    summaries = _fuse_matrix_kernel(
        observation_codes,
        valid_mask,
        reviewer_ids,
        context,
        chunk_size=1,
        include_samples=True,
    )
    samples = summaries[6]
    if samples is None:
        raise RuntimeError("scalar fusion kernel did not return posterior samples")
    return FusedPosterior(
        pass_probability=float(summaries[0][0]),
        lower=float(summaries[1][0]),
        upper=float(summaries[2][0]),
        valid_reviewers=int(summaries[3][0]),
        lineage_count=int(summaries[4][0]),
        effective_sample_size=float(summaries[5][0]),
        samples=tuple(float(value) for value in samples[0]),
    )
