from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real


class Truth(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class Observation(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"


class ExecutionState(str, Enum):
    VALID = "VALID"
    TIMEOUT = "TIMEOUT"
    INVALID = "INVALID"
    REFUSAL = "REFUSAL"
    NOT_CALLED = "NOT_CALLED"


class Action(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    DEFER = "DEFER"


class GateState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"


def _require_non_blank(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _require_non_negative(value: float, field: str) -> None:
    if value < 0:
        raise ValueError(f"{field} must be non-negative")


def _require_instance(value: object, expected_type: type[object], field: str) -> None:
    if not isinstance(value, expected_type):
        article = "an" if expected_type.__name__[0] in "AEIOU" else "a"
        raise TypeError(f"{field} must be {article} {expected_type.__name__}")


def _require_probability(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    numeric_value = float(value)
    if not isfinite(numeric_value) or not 0 <= numeric_value <= 1:
        raise ValueError(f"{field} must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class Reviewer:
    reviewer_id: str
    vendor: str
    family: str
    lineage: str
    cost: float = 1.0

    def __post_init__(self) -> None:
        _require_non_blank(self.reviewer_id, "Reviewer.reviewer_id")
        _require_non_negative(self.cost, "Reviewer.cost")


@dataclass(frozen=True, slots=True)
class Review:
    case_id: str
    reviewer_id: str
    observation: Observation | None
    state: ExecutionState
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        _require_instance(self.state, ExecutionState, "Review.state")
        if self.observation is not None and not isinstance(
            self.observation, Observation
        ):
            raise TypeError("Review.observation must be an Observation or None")
        _require_non_blank(self.case_id, "Review.case_id")
        _require_non_blank(self.reviewer_id, "Review.reviewer_id")
        _require_non_negative(self.input_tokens, "Review.input_tokens")
        _require_non_negative(self.output_tokens, "Review.output_tokens")
        if self.state is ExecutionState.VALID and self.observation is None:
            raise ValueError("Review.observation is required when state is VALID")
        if self.state is not ExecutionState.VALID and self.observation is not None:
            raise ValueError(
                "Review.observation must be None when state is not VALID"
            )


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    truth: Truth
    review: Review

    def __post_init__(self) -> None:
        _require_instance(self.truth, Truth, "CalibrationExample.truth")
        _require_instance(self.review, Review, "CalibrationExample.review")


@dataclass(frozen=True, slots=True)
class HardGate:
    gate_id: str
    state: GateState
    trusted_deterministic: bool = True

    def __post_init__(self) -> None:
        _require_instance(self.state, GateState, "HardGate.state")
        _require_non_blank(self.gate_id, "HardGate.gate_id")


@dataclass(frozen=True, slots=True)
class FusedPosterior:
    pass_probability: float
    lower: float
    upper: float
    valid_reviewers: int
    lineage_count: int
    effective_sample_size: float
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_instance(self.samples, tuple, "FusedPosterior.samples")
        probability_fields = (
            ("pass_probability", self.pass_probability),
            ("lower", self.lower),
            ("upper", self.upper),
        )
        for field, value in probability_fields:
            _require_probability(value, f"FusedPosterior.{field}")
        for index, sample in enumerate(self.samples):
            _require_probability(sample, f"FusedPosterior.samples[{index}]")
        if self.lower > self.upper:
            raise ValueError(
                "FusedPosterior.lower must not exceed FusedPosterior.upper"
            )


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reasons: tuple[str, ...]
    posterior: FusedPosterior | None

    def __post_init__(self) -> None:
        _require_instance(self.action, Action, "Decision.action")
        _require_instance(self.reasons, tuple, "Decision.reasons")
        for index, reason in enumerate(self.reasons):
            if not isinstance(reason, str):
                raise TypeError(f"Decision.reasons[{index}] must be a str")
        if self.posterior is not None and not isinstance(
            self.posterior, FusedPosterior
        ):
            raise TypeError("Decision.posterior must be a FusedPosterior or None")
