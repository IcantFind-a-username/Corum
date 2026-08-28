from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, sqrt
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from scipy.special import ndtr, ndtri  # type: ignore[import-untyped]

from corum.models import (
    ExecutionState,
    HardGate,
    Observation,
    Review,
    Reviewer,
    Truth,
)

_LIKELIHOOD_SHAPE = (2, 3)
_CORRELATION_TOLERANCE = 1e-8
_MAX_LATENT_CORRELATION = 1.0
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(96)
_UNIT_INTERVAL_NODES = 0.5 * (_LEGENDRE_NODES + 1.0)


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field} must be finite and representable as a float"
        ) from error
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return numeric


def _correlation(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field} must be finite and representable as a float"
        ) from error
    if not isfinite(numeric) or not -1.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be finite and within [-1, 1]")
    return numeric


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an int")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{field} must be non-negative")
    return numeric


def _read_only_array(value: object, field: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a numeric array") from error
    if array.shape != _LIKELIHOOD_SHAPE:
        raise ValueError(f"{field} must have shape {_LIKELIHOOD_SHAPE}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{field} must contain probabilities within [0, 1]")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{field} truth rows must sum to one")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True, slots=True)
class ReviewerSpec:
    reviewer: Reviewer
    likelihoods: np.ndarray
    timeout_rate: float = 0.0
    invalid_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer, Reviewer):
            raise TypeError("reviewer must be a Reviewer")
        likelihoods = _read_only_array(self.likelihoods, "likelihoods")
        timeout_rate = _probability(self.timeout_rate, "timeout_rate")
        invalid_rate = _probability(self.invalid_rate, "invalid_rate")
        if timeout_rate + invalid_rate > 1.0 + 1e-12:
            raise ValueError("timeout_rate and invalid_rate must sum to at most one")
        object.__setattr__(self, "likelihoods", likelihoods)
        object.__setattr__(self, "timeout_rate", timeout_rate)
        object.__setattr__(self, "invalid_rate", invalid_rate)


@dataclass(frozen=True, slots=True)
class ScenarioPhase:
    reviewers: tuple[ReviewerSpec, ...]
    prior_pass: float
    lineage_error_correlation: Mapping[str, float]
    difficulty_rate: float = 0.0
    informative_missingness: float = 0.0
    adversarial_reviewer_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reviewers, tuple):
            raise TypeError("reviewers must be a tuple")
        if not self.reviewers:
            raise ValueError("reviewers must contain at least one ReviewerSpec")
        reviewer_ids: list[str] = []
        lineage_counts: dict[str, int] = {}
        for index, spec in enumerate(self.reviewers):
            if not isinstance(spec, ReviewerSpec):
                raise TypeError(f"reviewers[{index}] must be a ReviewerSpec")
            reviewer_ids.append(spec.reviewer.reviewer_id)
            lineage = spec.reviewer.lineage
            lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
        duplicates = sorted(
            reviewer_id
            for reviewer_id in set(reviewer_ids)
            if reviewer_ids.count(reviewer_id) > 1
        )
        if duplicates:
            raise ValueError("duplicate reviewer IDs: " + ", ".join(duplicates))

        if not isinstance(self.lineage_error_correlation, Mapping):
            raise TypeError("lineage_error_correlation must be a mapping")
        correlations: dict[str, float] = {}
        for lineage, target in self.lineage_error_correlation.items():
            if not isinstance(lineage, str) or not lineage.strip():
                raise ValueError("correlation lineage keys must be non-blank strings")
            if lineage not in lineage_counts:
                raise ValueError(f"unknown lineage in correlation mapping: {lineage}")
            if lineage_counts[lineage] < 2:
                raise ValueError(
                    f"correlated lineage must contain at least two reviewers: {lineage}"
                )
            numeric_target = _probability(
                target,
                f"lineage_error_correlation[{lineage!r}]",
            )
            if numeric_target >= 1.0:
                raise ValueError("lineage error correlation targets must be below one")
            correlations[lineage] = numeric_target

        prior_pass = _probability(self.prior_pass, "prior_pass")
        difficulty_rate = _probability(self.difficulty_rate, "difficulty_rate")
        informative_missingness = _probability(
            self.informative_missingness,
            "informative_missingness",
        )
        adversarial = self.adversarial_reviewer_id
        if adversarial is not None:
            if not isinstance(adversarial, str):
                raise TypeError("adversarial reviewer ID must be a string or None")
            if adversarial not in reviewer_ids:
                raise ValueError(f"unknown adversarial reviewer ID: {adversarial}")

        object.__setattr__(self, "prior_pass", prior_pass)
        object.__setattr__(self, "difficulty_rate", difficulty_rate)
        object.__setattr__(
            self,
            "informative_missingness",
            informative_missingness,
        )
        object.__setattr__(
            self,
            "lineage_error_correlation",
            MappingProxyType(correlations),
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    calibration: ScenarioPhase
    test: ScenarioPhase

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if not isinstance(self.calibration, ScenarioPhase):
            raise TypeError("calibration must be a ScenarioPhase")
        if not isinstance(self.test, ScenarioPhase):
            raise TypeError("test must be a ScenarioPhase")
        calibration_ids = tuple(
            spec.reviewer.reviewer_id for spec in self.calibration.reviewers
        )
        test_ids = tuple(spec.reviewer.reviewer_id for spec in self.test.reviewers)
        if calibration_ids != test_ids:
            raise ValueError("scenario phases must use the same reviewer IDs and order")
        calibration_lineages = tuple(
            spec.reviewer.lineage for spec in self.calibration.reviewers
        )
        test_lineages = tuple(spec.reviewer.lineage for spec in self.test.reviewers)
        if calibration_lineages != test_lineages:
            raise ValueError("scenario phases must preserve reviewer lineages")


@dataclass(frozen=True, slots=True)
class LineageCorrelationDiagnostic:
    """Observed pair-mean correlation with minimum pairwise VALID overlap."""

    reviewer_ids: tuple[str, ...]
    target_error_correlation: float
    solved_latent_correlation: float
    minimum_eigenvalue: float
    realized_error_correlation: float
    overlap_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer_ids, tuple) or len(self.reviewer_ids) < 2:
            raise ValueError("reviewer_ids must contain at least two reviewers")
        if len(set(self.reviewer_ids)) != len(self.reviewer_ids):
            raise ValueError("reviewer_ids must not contain duplicates")
        target = _probability(
            self.target_error_correlation,
            "target_error_correlation",
        )
        solved = _probability(
            self.solved_latent_correlation,
            "solved_latent_correlation",
        )
        realized = _correlation(
            self.realized_error_correlation,
            "realized_error_correlation",
        )
        minimum = self.minimum_eigenvalue
        if isinstance(minimum, bool) or not isinstance(minimum, Real):
            raise TypeError("minimum_eigenvalue must be a real number")
        if not isfinite(float(minimum)) or minimum < -1e-10:
            raise ValueError("minimum_eigenvalue must be finite and non-negative")
        overlap_count = _non_negative_int(self.overlap_count, "overlap_count")
        object.__setattr__(self, "target_error_correlation", target)
        object.__setattr__(self, "solved_latent_correlation", solved)
        object.__setattr__(self, "minimum_eigenvalue", float(minimum))
        object.__setattr__(self, "realized_error_correlation", realized)
        object.__setattr__(self, "overlap_count", overlap_count)


@dataclass(frozen=True, slots=True)
class SimulatedPanel:
    seed: int
    truths: Mapping[str, Truth]
    difficulty_by_case: Mapping[str, bool]
    reviews: tuple[Review, ...]
    gates: Mapping[str, tuple[HardGate, ...]]
    lineage_diagnostics: Mapping[str, LineageCorrelationDiagnostic]

    def __post_init__(self) -> None:
        seed = _non_negative_int(self.seed, "seed")
        if not isinstance(self.truths, Mapping):
            raise TypeError("truths must be a mapping")
        if not isinstance(self.difficulty_by_case, Mapping):
            raise TypeError("difficulty_by_case must be a mapping")
        if not isinstance(self.gates, Mapping):
            raise TypeError("gates must be a mapping")
        if not isinstance(self.lineage_diagnostics, Mapping):
            raise TypeError("lineage_diagnostics must be a mapping")
        if not isinstance(self.reviews, tuple):
            raise TypeError("reviews must be a tuple")

        truths = dict(self.truths)
        difficulty = dict(self.difficulty_by_case)
        gates = dict(self.gates)
        diagnostics = dict(self.lineage_diagnostics)
        case_ids = set(truths)
        if set(difficulty) != case_ids or set(gates) != case_ids:
            raise ValueError("truth, difficulty, and gate case IDs must match")
        for case_id, truth in truths.items():
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError("case IDs must be non-blank strings")
            if not isinstance(truth, Truth):
                raise TypeError(f"truths[{case_id!r}] must be a Truth")
            if not isinstance(difficulty[case_id], bool):
                raise TypeError(f"difficulty_by_case[{case_id!r}] must be a bool")
            if not isinstance(gates[case_id], tuple) or any(
                not isinstance(gate, HardGate) for gate in gates[case_id]
            ):
                raise TypeError(f"gates[{case_id!r}] must be a tuple of HardGate")
        review_keys: set[tuple[str, str]] = set()
        for index, review in enumerate(self.reviews):
            if not isinstance(review, Review):
                raise TypeError(f"reviews[{index}] must be a Review")
            if review.case_id not in case_ids:
                raise ValueError("reviews contain an unknown case ID")
            key = (review.case_id, review.reviewer_id)
            if key in review_keys:
                raise ValueError(
                    "duplicate review for case/reviewer: "
                    f"{review.case_id!r}/{review.reviewer_id!r}"
                )
            review_keys.add(key)
        for lineage, diagnostic in diagnostics.items():
            if not isinstance(lineage, str) or not lineage.strip():
                raise ValueError("diagnostic lineage keys must be non-blank strings")
            if not isinstance(diagnostic, LineageCorrelationDiagnostic):
                raise TypeError("lineage diagnostics must contain diagnostic records")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "truths", MappingProxyType(truths))
        object.__setattr__(
            self,
            "difficulty_by_case",
            MappingProxyType(difficulty),
        )
        object.__setattr__(self, "gates", MappingProxyType(gates))
        object.__setattr__(
            self,
            "lineage_diagnostics",
            MappingProxyType(diagnostics),
        )


@lru_cache(maxsize=8_192)
def _conditional_joint_error(first_q: float, second_q: float, rho: float) -> float:
    if first_q <= 0.0 or second_q <= 0.0:
        return 0.0
    if first_q >= 1.0:
        return second_q
    if second_q >= 1.0:
        return first_q
    if rho <= _CORRELATION_TOLERANCE:
        return first_q * second_q
    if rho >= 1.0:
        return min(first_q, second_q)
    integration_q, conditional_q = sorted((first_q, second_q))
    uniforms = integration_q * (1.0 - _UNIT_INTERVAL_NODES**2)
    first_values = ndtri(uniforms)
    conditional_threshold = float(ndtri(conditional_q))
    residual = sqrt(1.0 - rho * rho)
    conditional = ndtr(
        (conditional_threshold - rho * first_values) / residual,
    )
    return float(
        integration_q
        * np.sum(
            _LEGENDRE_WEIGHTS * _UNIT_INTERVAL_NODES * conditional,
        )
    )


def _error_probabilities(spec: ReviewerSpec) -> tuple[float, float]:
    return (
        1.0 - float(spec.likelihoods[0, 0]),
        1.0 - float(spec.likelihoods[1, 1]),
    )


def _pair_expected_error_correlation(
    first: ReviewerSpec,
    second: ReviewerSpec,
    *,
    prior_pass: float,
    rho: float,
) -> float:
    first_q = _error_probabilities(first)
    second_q = _error_probabilities(second)
    first_mean = prior_pass * first_q[0] + (1.0 - prior_pass) * first_q[1]
    second_mean = prior_pass * second_q[0] + (1.0 - prior_pass) * second_q[1]
    denominator = sqrt(
        first_mean
        * (1.0 - first_mean)
        * second_mean
        * (1.0 - second_mean)
    )
    if denominator <= 0.0 or not isfinite(denominator):
        raise ValueError("registered correlation has a zero-variance error process")
    joint = prior_pass * _conditional_joint_error(
        first_q[0],
        second_q[0],
        rho,
    ) + (1.0 - prior_pass) * _conditional_joint_error(
        first_q[1],
        second_q[1],
        rho,
    )
    return (joint - first_mean * second_mean) / denominator


def _lineage_expected_error_correlation(
    specs: tuple[ReviewerSpec, ...],
    *,
    prior_pass: float,
    rho: float,
) -> float:
    correlations = [
        _pair_expected_error_correlation(
            specs[first_index],
            specs[second_index],
            prior_pass=prior_pass,
            rho=rho,
        )
        for first_index in range(len(specs))
        for second_index in range(first_index + 1, len(specs))
    ]
    return float(np.mean(correlations))


def _solve_latent_correlation(
    specs: tuple[ReviewerSpec, ...],
    *,
    prior_pass: float,
    target: float,
) -> float:
    lower_value = _lineage_expected_error_correlation(
        specs,
        prior_pass=prior_pass,
        rho=0.0,
    )
    upper_value = _lineage_expected_error_correlation(
        specs,
        prior_pass=prior_pass,
        rho=_MAX_LATENT_CORRELATION,
    )
    if target < lower_value - _CORRELATION_TOLERANCE or target > upper_value + 1e-6:
        raise ValueError(
            "lineage error correlation target is unreachable; "
            f"target={target:.6f}, reachable=[{lower_value:.6f}, {upper_value:.6f}]"
        )
    if abs(target - lower_value) <= _CORRELATION_TOLERANCE:
        return 0.0
    low = 0.0
    high = _MAX_LATENT_CORRELATION
    for _ in range(64):
        midpoint = (low + high) * 0.5
        value = _lineage_expected_error_correlation(
            specs,
            prior_pass=prior_pass,
            rho=midpoint,
        )
        if value < target:
            low = midpoint
        else:
            high = midpoint
    solved = (low + high) * 0.5
    residual = abs(
        _lineage_expected_error_correlation(
            specs,
            prior_pass=prior_pass,
            rho=solved,
        )
        - target
    )
    if residual > 1e-5:
        raise ValueError("lineage error correlation target could not be solved")
    return solved


def _semantic_codes(
    phase: ScenarioPhase,
    truth_indices: np.ndarray,
    rng: np.random.Generator,
    solved_by_lineage: Mapping[str, float],
) -> np.ndarray:
    case_count = truth_indices.size
    reviewer_count = len(phase.reviewers)
    codes = np.empty((case_count, reviewer_count), dtype=np.int8)
    indices_by_lineage: dict[str, list[int]] = {}
    for index, spec in enumerate(phase.reviewers):
        indices_by_lineage.setdefault(spec.reviewer.lineage, []).append(index)

    for lineage, reviewer_indices in indices_by_lineage.items():
        rho = solved_by_lineage.get(lineage, 0.0)
        independent = rng.standard_normal((case_count, len(reviewer_indices)))
        if rho > 0.0:
            common = rng.standard_normal((case_count, 1))
            normals = sqrt(rho) * common + sqrt(1.0 - rho) * independent
        else:
            normals = independent
        error_uniforms = ndtr(normals)
        category_uniforms = rng.random((case_count, len(reviewer_indices)))

        for local_index, reviewer_index in enumerate(reviewer_indices):
            spec = phase.reviewers[reviewer_index]
            likelihoods = spec.likelihoods
            error_q = np.where(
                truth_indices == 0,
                1.0 - likelihoods[0, 0],
                1.0 - likelihoods[1, 1],
            )
            is_error = error_uniforms[:, local_index] < error_q
            is_pass_truth = truth_indices == 0
            reviewer_codes = np.where(is_pass_truth, 0, 1).astype(np.int8)

            pass_error_total = 1.0 - likelihoods[0, 0]
            fail_error_total = 1.0 - likelihoods[1, 1]
            pass_wrong_share = (
                float(likelihoods[0, 1] / pass_error_total)
                if pass_error_total > 0.0
                else 0.0
            )
            fail_wrong_share = (
                float(likelihoods[1, 0] / fail_error_total)
                if fail_error_total > 0.0
                else 0.0
            )
            category_uniform = category_uniforms[:, local_index]
            pass_error_codes = np.where(category_uniform < pass_wrong_share, 1, 2)
            fail_error_codes = np.where(category_uniform < fail_wrong_share, 0, 2)
            error_codes = np.where(is_pass_truth, pass_error_codes, fail_error_codes)
            reviewer_codes[is_error] = error_codes[is_error]
            codes[:, reviewer_index] = reviewer_codes
    return codes


def _execution_states(
    phase: ScenarioPhase,
    truth_indices: np.ndarray,
    difficulty: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    case_count = truth_indices.size
    states = np.zeros((case_count, len(phase.reviewers)), dtype=np.int8)
    signal = 0.5 * difficulty.astype(float) + 0.5 * (truth_indices == 1)
    for reviewer_index, spec in enumerate(phase.reviewers):
        base_total = spec.timeout_rate + spec.invalid_rate
        extra = phase.informative_missingness * signal * (1.0 - base_total)
        if base_total > 0.0:
            timeout_share = spec.timeout_rate / base_total
        else:
            timeout_share = 0.5
        timeout_probability = spec.timeout_rate + extra * timeout_share
        missing_probability = np.minimum(base_total + extra, 1.0)
        draws = rng.random(case_count)
        states[:, reviewer_index] = np.where(
            draws < timeout_probability,
            1,
            np.where(draws < missing_probability, 2, 0),
        )
    return states


def _realized_diagnostic(
    *,
    lineage: str,
    reviewer_indices: tuple[int, ...],
    phase: ScenarioPhase,
    truth_indices: np.ndarray,
    semantic_codes: np.ndarray,
    execution_states: np.ndarray,
    target: float,
    solved: float,
) -> LineageCorrelationDiagnostic:
    correlations: list[float] = []
    overlap_counts: list[int] = []
    correct_codes = np.where(truth_indices == 0, 0, 1)
    for first_position, first_index in enumerate(reviewer_indices):
        for second_index in reviewer_indices[first_position + 1 :]:
            overlap = (execution_states[:, first_index] == 0) & (
                execution_states[:, second_index] == 0
            )
            overlap_count = int(np.count_nonzero(overlap))
            if overlap_count < 2:
                raise ValueError(
                    f"registered lineage {lineage!r} has insufficient VALID overlap"
                )
            first_errors = semantic_codes[overlap, first_index] != correct_codes[overlap]
            second_errors = (
                semantic_codes[overlap, second_index] != correct_codes[overlap]
            )
            if np.all(first_errors == first_errors[0]) or np.all(
                second_errors == second_errors[0]
            ):
                raise ValueError(
                    f"registered lineage {lineage!r} has zero realized error variance"
                )
            correlation = float(np.corrcoef(first_errors, second_errors)[0, 1])
            if not isfinite(correlation):
                raise ValueError(
                    f"registered lineage {lineage!r} produced non-finite correlation"
                )
            correlations.append(correlation)
            overlap_counts.append(overlap_count)
    reviewer_ids = tuple(
        phase.reviewers[index].reviewer.reviewer_id for index in reviewer_indices
    )
    minimum_eigenvalue = min(
        1.0 - solved,
        1.0 + (len(reviewer_indices) - 1) * solved,
    )
    return LineageCorrelationDiagnostic(
        reviewer_ids=reviewer_ids,
        target_error_correlation=target,
        solved_latent_correlation=solved,
        minimum_eigenvalue=minimum_eigenvalue,
        realized_error_correlation=float(np.mean(correlations)),
        overlap_count=min(overlap_counts),
    )


def _simulate_panel(
    phase: ScenarioPhase,
    n_cases: int,
    *,
    seed: int,
    case_prefix: str,
) -> SimulatedPanel:
    if not isinstance(phase, ScenarioPhase):
        raise TypeError("phase must be a ScenarioPhase")
    case_count = _non_negative_int(n_cases, "n_cases")
    numeric_seed = _non_negative_int(seed, "seed")
    rng = np.random.default_rng(numeric_seed)
    case_ids = tuple(f"{case_prefix}-{index:06d}" for index in range(case_count))
    truth_indices = (rng.random(case_count) >= phase.prior_pass).astype(np.int8)
    difficulty = rng.random(case_count) < phase.difficulty_rate

    specs_by_lineage: dict[str, tuple[ReviewerSpec, ...]] = {}
    indices_by_lineage: dict[str, tuple[int, ...]] = {}
    for lineage in phase.lineage_error_correlation:
        indices = tuple(
            index
            for index, spec in enumerate(phase.reviewers)
            if spec.reviewer.lineage == lineage
        )
        indices_by_lineage[lineage] = indices
        specs_by_lineage[lineage] = tuple(phase.reviewers[index] for index in indices)
    solved_by_lineage = {
        lineage: _solve_latent_correlation(
            specs_by_lineage[lineage],
            prior_pass=phase.prior_pass,
            target=target,
        )
        for lineage, target in phase.lineage_error_correlation.items()
    }

    semantic_codes = _semantic_codes(
        phase,
        truth_indices,
        rng,
        solved_by_lineage,
    )
    execution_states = _execution_states(
        phase,
        truth_indices,
        difficulty,
        rng,
    )
    truths = {
        case_id: Truth.PASS if truth_index == 0 else Truth.FAIL
        for case_id, truth_index in zip(case_ids, truth_indices, strict=True)
    }
    difficulty_by_case = {
        case_id: bool(is_difficult)
        for case_id, is_difficult in zip(case_ids, difficulty, strict=True)
    }
    observation_order = (
        Observation.PASS,
        Observation.FAIL,
        Observation.ABSTAIN,
    )
    state_order = (
        ExecutionState.VALID,
        ExecutionState.TIMEOUT,
        ExecutionState.INVALID,
    )
    reviews = tuple(
        Review(
            case_id=case_id,
            reviewer_id=spec.reviewer.reviewer_id,
            observation=(
                observation_order[int(semantic_codes[case_index, reviewer_index])]
                if execution_states[case_index, reviewer_index] == 0
                else None
            ),
            state=state_order[int(execution_states[case_index, reviewer_index])],
        )
        for case_index, case_id in enumerate(case_ids)
        for reviewer_index, spec in enumerate(phase.reviewers)
    )
    gates = {case_id: () for case_id in case_ids}
    diagnostics = {
        lineage: _realized_diagnostic(
            lineage=lineage,
            reviewer_indices=indices_by_lineage[lineage],
            phase=phase,
            truth_indices=truth_indices,
            semantic_codes=semantic_codes,
            execution_states=execution_states,
            target=target,
            solved=solved_by_lineage[lineage],
        )
        for lineage, target in phase.lineage_error_correlation.items()
    }
    return SimulatedPanel(
        seed=numeric_seed,
        truths=truths,
        difficulty_by_case=difficulty_by_case,
        reviews=reviews,
        gates=gates,
        lineage_diagnostics=diagnostics,
    )


def simulate_panel(
    phase: ScenarioPhase,
    n_cases: int,
    *,
    seed: int,
) -> SimulatedPanel:
    return _simulate_panel(
        phase,
        n_cases,
        seed=seed,
        case_prefix="case",
    )


def simulate_experiment(
    scenario: Scenario,
    *,
    n_calibration: int,
    n_test: int,
    seed: int,
) -> tuple[SimulatedPanel, SimulatedPanel]:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    calibration_count = _non_negative_int(n_calibration, "n_calibration")
    test_count = _non_negative_int(n_test, "n_test")
    numeric_seed = _non_negative_int(seed, "seed")
    child_sequences = np.random.SeedSequence(numeric_seed).spawn(2)
    child_seeds = tuple(
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    )
    calibration = _simulate_panel(
        scenario.calibration,
        calibration_count,
        seed=child_seeds[0],
        case_prefix="calibration",
    )
    test = _simulate_panel(
        scenario.test,
        test_count,
        seed=child_seeds[1],
        case_prefix="test",
    )
    return calibration, test


def _scenario_spec(
    reviewer_id: str,
    *,
    lineage: str,
    accuracy: float,
    abstain: float = 0.05,
    timeout_rate: float = 0.0,
    invalid_rate: float = 0.0,
    cost: float = 1.0,
    family: str = "general",
) -> ReviewerSpec:
    return ReviewerSpec(
        reviewer=Reviewer(
            reviewer_id,
            "simulated",
            family,
            lineage,
            cost,
        ),
        likelihoods=np.array(
            [
                [accuracy, 1.0 - accuracy - abstain, abstain],
                [1.0 - accuracy - abstain, accuracy, abstain],
            ]
        ),
        timeout_rate=timeout_rate,
        invalid_rate=invalid_rate,
    )


def _builtins() -> dict[str, Scenario]:
    independent_reviewers = (
        _scenario_spec("r1", lineage="independent-1", accuracy=0.82),
        _scenario_spec("r2", lineage="independent-2", accuracy=0.76),
        _scenario_spec("r3", lineage="independent-3", accuracy=0.70),
    )
    independent = Scenario(
        "independent",
        ScenarioPhase(independent_reviewers, 0.8, {}),
        ScenarioPhase(independent_reviewers, 0.8, {}),
    )

    clone_reviewers = (
        _scenario_spec("clone-a", lineage="clone", accuracy=0.76),
        _scenario_spec("clone-b", lineage="clone", accuracy=0.76),
        _scenario_spec("strong", lineage="independent", accuracy=0.84),
    )
    clone_pair = Scenario(
        "clone_pair",
        ScenarioPhase(clone_reviewers, 0.8, {"clone": 0.85}),
        ScenarioPhase(clone_reviewers, 0.8, {"clone": 0.85}),
    )

    trap_reviewers = (
        _scenario_spec("weak-a", lineage="weak-clones", accuracy=0.62),
        _scenario_spec("weak-b", lineage="weak-clones", accuracy=0.63),
        _scenario_spec("strong", lineage="strong", accuracy=0.86),
    )
    majority_trap = Scenario(
        "majority_trap",
        ScenarioPhase(trap_reviewers, 0.8, {"weak-clones": 0.82}),
        ScenarioPhase(trap_reviewers, 0.8, {"weak-clones": 0.82}),
    )

    missing_reviewers = (
        _scenario_spec(
            "fast",
            lineage="fast",
            accuracy=0.75,
            timeout_rate=0.03,
            invalid_rate=0.02,
        ),
        _scenario_spec(
            "careful",
            lineage="careful",
            accuracy=0.84,
            timeout_rate=0.04,
            invalid_rate=0.03,
        ),
        _scenario_spec(
            "abstaining",
            lineage="abstaining",
            accuracy=0.70,
            abstain=0.20,
            timeout_rate=0.02,
            invalid_rate=0.04,
        ),
    )
    informative_missingness = Scenario(
        "informative_missingness",
        ScenarioPhase(missing_reviewers, 0.8, {}, difficulty_rate=0.35),
        ScenarioPhase(
            missing_reviewers,
            0.8,
            {},
            difficulty_rate=0.45,
            informative_missingness=0.55,
        ),
    )

    drift_calibration_reviewers = (
        _scenario_spec("stable", lineage="shared", accuracy=0.82),
        _scenario_spec("shifted", lineage="shared", accuracy=0.75),
        _scenario_spec("independent", lineage="independent", accuracy=0.80),
    )
    drift_test_reviewers = (
        _scenario_spec(
            "stable",
            lineage="shared",
            accuracy=0.65,
            timeout_rate=0.03,
        ),
        _scenario_spec(
            "shifted",
            lineage="shared",
            accuracy=0.25,
            timeout_rate=0.04,
        ),
        _scenario_spec(
            "independent",
            lineage="independent",
            accuracy=0.72,
            invalid_rate=0.03,
        ),
    )
    drift = Scenario(
        "drift",
        ScenarioPhase(
            drift_calibration_reviewers,
            0.8,
            {"shared": 0.10},
            difficulty_rate=0.2,
        ),
        ScenarioPhase(
            drift_test_reviewers,
            0.6,
            {"shared": 0.30},
            difficulty_rate=0.5,
            informative_missingness=0.10,
            adversarial_reviewer_id="shifted",
        ),
    )

    cascade_reviewers = (
        _scenario_spec(
            "cheap",
            lineage="cheap",
            accuracy=0.72,
            cost=0.1,
        ),
        _scenario_spec(
            "balanced",
            lineage="balanced",
            accuracy=0.80,
            cost=0.6,
        ),
        _scenario_spec(
            "expert",
            lineage="expert",
            accuracy=0.90,
            cost=2.0,
        ),
    )
    cascade_cost = Scenario(
        "cascade_cost",
        ScenarioPhase(cascade_reviewers, 0.8, {}, difficulty_rate=0.4),
        ScenarioPhase(cascade_reviewers, 0.8, {}, difficulty_rate=0.5),
    )

    return {
        scenario.name: scenario
        for scenario in (
            independent,
            clone_pair,
            majority_trap,
            informative_missingness,
            drift,
            cascade_cost,
        )
    }


def builtin_scenarios() -> Mapping[str, Scenario]:
    return MappingProxyType(_builtins())
