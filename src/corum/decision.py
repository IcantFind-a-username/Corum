from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

from corum.models import (
    Action,
    Decision,
    FusedPosterior,
    GateState,
    HardGate,
)


def _require_probability(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field} must be finite and representable as a float"
        ) from error
    if not isfinite(numeric_value) or not 0 <= numeric_value <= 1:
        raise ValueError(f"{field} must be finite and within [0, 1]")


def _require_non_negative_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an int")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")


def _require_non_negative_real(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field} must be finite and representable as a float"
        ) from error
    if not isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{field} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Threshold and quorum policy for converting evidence into an action."""

    pass_threshold: float = 0.90
    fail_threshold: float = 0.10
    min_valid_reviewers: int = 2
    min_lineages: int = 2
    min_effective_sample_size: float = 1.5

    def __post_init__(self) -> None:
        _require_probability(self.pass_threshold, "pass_threshold")
        _require_probability(self.fail_threshold, "fail_threshold")
        if self.fail_threshold >= self.pass_threshold:
            raise ValueError(
                "fail_threshold must be lower than pass_threshold"
            )
        _require_non_negative_int(
            self.min_valid_reviewers,
            "min_valid_reviewers",
        )
        _require_non_negative_int(self.min_lineages, "min_lineages")
        _require_non_negative_real(
            self.min_effective_sample_size,
            "min_effective_sample_size",
        )


def decide(
    posterior: FusedPosterior | None,
    gates: Sequence[HardGate],
    policy: DecisionPolicy,
) -> Decision:
    """Apply deterministic gates, quorum checks, then risk thresholds."""

    if posterior is not None and not isinstance(posterior, FusedPosterior):
        raise TypeError("posterior must be a FusedPosterior or None")
    if not isinstance(policy, DecisionPolicy):
        raise TypeError("policy must be a DecisionPolicy")

    checked_gates = tuple(gates)
    for index, gate in enumerate(checked_gates):
        if not isinstance(gate, HardGate):
            raise TypeError(f"gates[{index}] must be a HardGate")

    trusted_states = {
        gate.state for gate in checked_gates if gate.trusted_deterministic
    }
    if GateState.FAIL in trusted_states:
        return Decision(Action.FAIL, ("hard_gate_failed",), posterior)
    if GateState.UNRESOLVED in trusted_states:
        return Decision(Action.DEFER, ("hard_gate_unresolved",), posterior)

    if posterior is None:
        return Decision(Action.DEFER, ("missing_posterior",), None)

    quorum_reasons: list[str] = []
    if posterior.valid_reviewers < policy.min_valid_reviewers:
        quorum_reasons.append("insufficient_valid_reviewers")
    if posterior.lineage_count < policy.min_lineages:
        quorum_reasons.append("insufficient_lineages")
    if posterior.effective_sample_size < policy.min_effective_sample_size:
        quorum_reasons.append("insufficient_effective_sample_size")
    if quorum_reasons:
        return Decision(Action.DEFER, tuple(quorum_reasons), posterior)

    if posterior.lower >= policy.pass_threshold:
        return Decision(Action.PASS, ("pass_threshold_met",), posterior)
    if posterior.upper <= policy.fail_threshold:
        return Decision(Action.FAIL, ("fail_threshold_met",), posterior)
    return Decision(Action.DEFER, ("posterior_uncertain",), posterior)
