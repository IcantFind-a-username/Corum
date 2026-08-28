from collections.abc import Callable

import pytest

from corum.decision import DecisionPolicy, decide
from corum.models import (
    Action,
    FusedPosterior,
    GateState,
    HardGate,
)


def _posterior(
    *,
    pass_probability: float = 0.5,
    lower: float = 0.4,
    upper: float = 0.6,
    valid_reviewers: int = 2,
    lineage_count: int = 2,
    effective_sample_size: float = 2.0,
) -> FusedPosterior:
    return FusedPosterior(
        pass_probability=pass_probability,
        lower=lower,
        upper=upper,
        valid_reviewers=valid_reviewers,
        lineage_count=lineage_count,
        effective_sample_size=effective_sample_size,
        samples=(pass_probability,),
    )


def test_trusted_fail_gate_has_absolute_precedence() -> None:
    posterior = _posterior(pass_probability=0.99, lower=0.95, upper=1.0)
    gates = (
        HardGate("unresolved", GateState.UNRESOLVED),
        HardGate("failed", GateState.FAIL),
    )

    decision = decide(posterior, gates, DecisionPolicy())

    assert decision.action is Action.FAIL
    assert decision.reasons == ("hard_gate_failed",)
    assert decision.posterior is posterior


def test_trusted_unresolved_gate_blocks_otherwise_confident_pass() -> None:
    posterior = _posterior(pass_probability=0.95, lower=0.90, upper=0.99)

    decision = decide(
        posterior,
        (HardGate("unresolved", GateState.UNRESOLVED),),
        DecisionPolicy(),
    )

    assert decision.action is Action.DEFER
    assert decision.reasons == ("hard_gate_unresolved",)


def test_untrusted_gates_do_not_override_statistical_decision() -> None:
    posterior = _posterior(pass_probability=0.95, lower=0.90, upper=0.99)
    gates = (
        HardGate("failed", GateState.FAIL, trusted_deterministic=False),
        HardGate(
            "unresolved",
            GateState.UNRESOLVED,
            trusted_deterministic=False,
        ),
    )

    decision = decide(posterior, gates, DecisionPolicy())

    assert decision.action is Action.PASS
    assert decision.reasons == ("pass_threshold_met",)


def test_missing_posterior_defers_instead_of_guessing() -> None:
    decision = decide(None, (), DecisionPolicy())

    assert decision.action is Action.DEFER
    assert decision.reasons == ("missing_posterior",)
    assert decision.posterior is None


def test_all_quorum_failures_are_reported_in_stable_order() -> None:
    posterior = _posterior(
        valid_reviewers=1,
        lineage_count=1,
        effective_sample_size=1.0,
    )
    policy = DecisionPolicy(
        min_valid_reviewers=3,
        min_lineages=2,
        min_effective_sample_size=1.5,
    )

    decision = decide(posterior, (), policy)

    assert decision.action is Action.DEFER
    assert decision.reasons == (
        "insufficient_valid_reviewers",
        "insufficient_lineages",
        "insufficient_effective_sample_size",
    )


def test_pass_threshold_boundary_is_inclusive() -> None:
    posterior = _posterior(pass_probability=0.94, lower=0.90, upper=0.98)

    decision = decide(posterior, (), DecisionPolicy(pass_threshold=0.90))

    assert decision.action is Action.PASS
    assert decision.reasons == ("pass_threshold_met",)


def test_fail_threshold_boundary_is_inclusive() -> None:
    posterior = _posterior(pass_probability=0.06, lower=0.01, upper=0.10)

    decision = decide(posterior, (), DecisionPolicy(fail_threshold=0.10))

    assert decision.action is Action.FAIL
    assert decision.reasons == ("fail_threshold_met",)


def test_interval_between_thresholds_defers_as_uncertain() -> None:
    posterior = _posterior(pass_probability=0.5, lower=0.2, upper=0.8)

    decision = decide(posterior, (), DecisionPolicy())

    assert decision.action is Action.DEFER
    assert decision.reasons == ("posterior_uncertain",)


def test_all_invalid_and_all_abstain_panels_defer() -> None:
    all_invalid = decide(None, (), DecisionPolicy())
    all_abstain = decide(
        _posterior(pass_probability=0.999, lower=0.0, upper=1.0),
        (),
        DecisionPolicy(),
    )

    assert all_invalid.action is Action.DEFER
    assert all_abstain.action is Action.DEFER


@pytest.mark.parametrize(
    ("policy_factory", "error_type", "message"),
    [
        (
            lambda: DecisionPolicy(pass_threshold=True),
            TypeError,
            "pass_threshold must be a real number",
        ),
        (
            lambda: DecisionPolicy(fail_threshold=float("nan")),
            ValueError,
            "fail_threshold must be finite and within",
        ),
        (
            lambda: DecisionPolicy(pass_threshold=0.1, fail_threshold=0.2),
            ValueError,
            "fail_threshold must be lower than pass_threshold",
        ),
        (
            lambda: DecisionPolicy(min_valid_reviewers=1.5),
            TypeError,
            "min_valid_reviewers must be an int",
        ),
        (
            lambda: DecisionPolicy(min_lineages=-1),
            ValueError,
            "min_lineages must be non-negative",
        ),
        (
            lambda: DecisionPolicy(min_effective_sample_size=float("inf")),
            ValueError,
            "min_effective_sample_size must be finite and non-negative",
        ),
    ],
)
def test_policy_rejects_invalid_parameters(
    policy_factory: Callable[[], DecisionPolicy],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        policy_factory()


def test_decide_rejects_non_gate_values() -> None:
    with pytest.raises(TypeError, match=r"gates\[0\] must be a HardGate"):
        decide(
            _posterior(),
            ("not-a-gate",),  # type: ignore[arg-type]
            DecisionPolicy(),
        )


def test_decide_rejects_wrong_policy_type() -> None:
    with pytest.raises(TypeError, match="policy must be a DecisionPolicy"):
        decide(
            _posterior(),
            (),
            "not-a-policy",  # type: ignore[arg-type]
        )
