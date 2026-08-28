from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isfinite, log
from numbers import Integral, Real

import numpy as np

from corum.baselines import DecisionCosts
from corum.decision import DecisionPolicy, decide
from corum.models import Action, Decision, FusedPosterior, HardGate, Truth

_LOG_LOSS_EPSILON = 1e-15


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _probability(value: object, field: str, *, open_interval: bool = False) -> float:
    numeric = _finite_real(value, field)
    valid = 0.0 < numeric < 1.0 if open_interval else 0.0 <= numeric <= 1.0
    interval = "(0, 1)" if open_interval else "[0, 1]"
    if not valid:
        raise ValueError(f"{field} must be within {interval}")
    return numeric


def _mapping_copy(value: object, field: str) -> dict[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return dict(value)


def _validated_truths(truths: Mapping[str, Truth]) -> dict[str, Truth]:
    copied = _mapping_copy(truths, "truths")
    if not copied:
        raise ValueError("truths must not be empty")
    validated: dict[str, Truth] = {}
    for case_id, truth in copied.items():
        if not isinstance(case_id, str):
            raise TypeError("truth case IDs must be strings")
        if not case_id.strip():
            raise ValueError("truth case IDs must not be blank")
        if not isinstance(truth, Truth):
            raise TypeError(f"truths[{case_id!r}] must be a Truth")
        validated[case_id] = truth
    return validated


def _require_exact_ids(
    expected: set[str],
    actual: set[object],
    field: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{field} case IDs must exactly match truths case IDs")


def _validated_actions(
    decisions: Mapping[str, Decision | Action],
    expected_ids: set[str],
) -> dict[str, Action]:
    copied = _mapping_copy(decisions, "decisions")
    _require_exact_ids(expected_ids, set(copied), "decisions")
    actions: dict[str, Action] = {}
    for case_id in expected_ids:
        value = copied[case_id]
        if isinstance(value, Decision):
            actions[case_id] = value.action
        elif isinstance(value, Action):
            actions[case_id] = value
        else:
            raise TypeError(
                f"decisions[{case_id!r}] must be a Decision or Action"
            )
    return actions


def _validated_probabilities(
    probabilities: Mapping[str, float] | None,
    expected_ids: set[str],
) -> dict[str, float] | None:
    if probabilities is None:
        return None
    copied = _mapping_copy(probabilities, "probabilities")
    _require_exact_ids(expected_ids, set(copied), "probabilities")
    return {
        case_id: _probability(copied[case_id], f"probabilities[{case_id!r}]")
        for case_id in expected_ids
    }


def _validated_weights(
    sample_weights: Mapping[str, float] | None,
    expected_ids: set[str],
) -> dict[str, float]:
    if sample_weights is None:
        return {case_id: 1.0 for case_id in expected_ids}
    copied = _mapping_copy(sample_weights, "sample_weights")
    _require_exact_ids(expected_ids, set(copied), "sample_weights")
    weights: dict[str, float] = {}
    for case_id in expected_ids:
        weight = _finite_real(
            copied[case_id],
            f"sample_weights[{case_id!r}]",
        )
        if weight < 0.0:
            raise ValueError(
                f"sample_weights[{case_id!r}] must be non-negative"
            )
        weights[case_id] = weight
    maximum = max(weights.values())
    if maximum <= 0.0:
        raise ValueError("sample_weights must have positive total mass")
    return weights


def _weighted_fraction(
    weights: Mapping[str, Fraction],
    numerator_ids: Sequence[str],
    denominator_ids: Sequence[str],
) -> float:
    denominator_mass = sum(
        (weights[case_id] for case_id in denominator_ids),
        start=Fraction(),
    )
    if denominator_mass <= 0:
        return float("nan")
    numerator_mass = sum(
        (weights[case_id] for case_id in numerator_ids),
        start=Fraction(),
    )
    return float(numerator_mass / denominator_mass)


def _decision_loss(
    action: Action,
    truth: Truth,
    costs: DecisionCosts,
) -> float:
    if action is Action.DEFER:
        return costs.defer
    if action is Action.PASS and truth is Truth.FAIL:
        return costs.false_pass
    if action is Action.FAIL and truth is Truth.PASS:
        return costs.false_fail
    return 0.0


def _probability_metrics(
    truths: Mapping[str, Truth],
    probabilities: Mapping[str, float] | None,
    weights: Mapping[str, Fraction],
) -> tuple[float, float, float]:
    if probabilities is None:
        missing = float("nan")
        return missing, missing, missing

    total_mass = sum(weights.values(), start=Fraction())
    brier_numerator = Fraction()
    log_loss_numerator = Fraction()
    bin_ids: list[list[str]] = [[] for _ in range(10)]
    for case_id, truth in truths.items():
        probability = probabilities[case_id]
        weight = weights[case_id]
        outcome = 1.0 if truth is Truth.PASS else 0.0
        brier_numerator += weight * Fraction.from_float(
            (probability - outcome) ** 2
        )
        observed_probability = (
            probability if truth is Truth.PASS else 1.0 - probability
        )
        clipped = min(
            max(observed_probability, _LOG_LOSS_EPSILON),
            1.0 - _LOG_LOSS_EPSILON,
        )
        log_loss_numerator += weight * Fraction.from_float(-log(clipped))

        bin_index = min(int(probability * 10.0), 9)
        bin_ids[bin_index].append(case_id)

    ece = Fraction()
    for case_ids in bin_ids:
        positive_ids = [case_id for case_id in case_ids if weights[case_id] > 0]
        if not positive_ids:
            continue
        bin_mass = sum(
            (weights[case_id] for case_id in positive_ids),
            start=Fraction(),
        )
        probability_mass = sum(
            (
                weights[case_id]
                * Fraction.from_float(probabilities[case_id])
                for case_id in positive_ids
            ),
            start=Fraction(),
        )
        pass_mass = sum(
            (
                weights[case_id] * int(truths[case_id] is Truth.PASS)
                for case_id in positive_ids
            ),
            start=Fraction(),
        )
        mean_probability = probability_mass / bin_mass
        pass_frequency = pass_mass / bin_mass
        ece += bin_mass / total_mass * abs(
            mean_probability - pass_frequency
        )
    return (
        float(brier_numerator / total_mass),
        float(log_loss_numerator / total_mass),
        float(ece),
    )


def evaluate_decisions(
    truths: Mapping[str, Truth],
    decisions: Mapping[str, Decision | Action],
    *,
    probabilities: Mapping[str, float] | None = None,
    costs: DecisionCosts = DecisionCosts(),  # noqa: B008
    sample_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate one exact case partition under one declared weighting."""

    truth_by_id = _validated_truths(truths)
    case_ids = set(truth_by_id)
    action_by_id = _validated_actions(decisions, case_ids)
    probability_by_id = _validated_probabilities(probabilities, case_ids)
    weights = _validated_weights(sample_weights, case_ids)
    if not isinstance(costs, DecisionCosts):
        raise TypeError("costs must be a DecisionCosts")

    ordered_ids = tuple(truth_by_id)
    decided_ids = tuple(
        case_id
        for case_id in ordered_ids
        if action_by_id[case_id] is not Action.DEFER
    )
    fail_truth_ids = tuple(
        case_id
        for case_id in ordered_ids
        if truth_by_id[case_id] is Truth.FAIL
    )
    pass_truth_ids = tuple(
        case_id
        for case_id in ordered_ids
        if truth_by_id[case_id] is Truth.PASS
    )
    false_pass_ids = tuple(
        case_id
        for case_id in fail_truth_ids
        if action_by_id[case_id] is Action.PASS
    )
    false_fail_ids = tuple(
        case_id
        for case_id in pass_truth_ids
        if action_by_id[case_id] is Action.FAIL
    )
    pass_action_ids = tuple(
        case_id
        for case_id in ordered_ids
        if action_by_id[case_id] is Action.PASS
    )
    losses = {
        case_id: _decision_loss(action_by_id[case_id], truth, costs)
        for case_id, truth in truth_by_id.items()
    }
    exact_weights = {
        case_id: Fraction.from_float(weights[case_id])
        for case_id in ordered_ids
    }
    exact_total = sum(exact_weights.values(), start=Fraction())
    weighted_loss = sum(
        (
            exact_weights[case_id]
            * Fraction.from_float(losses[case_id])
            for case_id in ordered_ids
        ),
        start=Fraction(),
    )

    coverage = _weighted_fraction(exact_weights, decided_ids, ordered_ids)
    brier, log_loss_value, ece = _probability_metrics(
        truth_by_id,
        probability_by_id,
        exact_weights,
    )
    return {
        "coverage": coverage,
        "defer_rate": 1.0 - coverage,
        "false_pass_rate": _weighted_fraction(
            exact_weights,
            false_pass_ids,
            fail_truth_ids,
        ),
        "false_fail_rate": _weighted_fraction(
            exact_weights,
            false_fail_ids,
            pass_truth_ids,
        ),
        "false_safe_risk": _weighted_fraction(
            exact_weights,
            false_pass_ids,
            pass_action_ids,
        ),
        "selective_risk": _weighted_fraction(
            exact_weights,
            false_pass_ids + false_fail_ids,
            decided_ids,
        ),
        "decision_loss": float(weighted_loss / exact_total),
        "brier": brier,
        "log_loss": log_loss_value,
        "ece": ece,
    }


def target_prevalence_weights(
    truths: Mapping[str, Truth],
    *,
    target_fail_prevalence: float,
) -> dict[str, float]:
    truth_by_id = _validated_truths(truths)
    target = _probability(
        target_fail_prevalence,
        "target_fail_prevalence",
        open_interval=True,
    )
    count = len(truth_by_id)
    fail_count = sum(truth is Truth.FAIL for truth in truth_by_id.values())
    if fail_count == 0 or fail_count == count:
        raise ValueError("truths must contain both PASS and FAIL classes")

    empirical_fail = fail_count / count
    class_weights = {
        Truth.FAIL: target / empirical_fail,
        Truth.PASS: (1.0 - target) / (1.0 - empirical_fail),
    }
    raw = {
        case_id: class_weights[truth]
        for case_id, truth in truth_by_id.items()
    }
    mean_weight = fsum(raw.values()) / count
    return {case_id: weight / mean_weight for case_id, weight in raw.items()}


@dataclass(frozen=True, slots=True)
class PolicySelection:
    policy: DecisionPolicy
    constraint_satisfied: bool
    decision_loss: float
    coverage: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy, DecisionPolicy):
            raise TypeError("policy must be a DecisionPolicy")
        if not isinstance(self.constraint_satisfied, bool):
            raise TypeError("constraint_satisfied must be a bool")
        loss = _finite_real(self.decision_loss, "decision_loss")
        if loss < 0.0:
            raise ValueError("decision_loss must be non-negative")
        object.__setattr__(self, "decision_loss", loss)
        object.__setattr__(
            self,
            "coverage",
            _probability(self.coverage, "coverage"),
        )


def policy_candidates() -> tuple[DecisionPolicy, ...]:
    return tuple(
        DecisionPolicy(
            pass_threshold=pass_threshold,
            fail_threshold=fail_threshold,
            min_valid_reviewers=2,
            min_lineages=2,
            min_effective_sample_size=minimum_ess,
        )
        for pass_threshold in (0.8, 0.9, 0.95)
        for fail_threshold in (0.05, 0.1, 0.2)
        for minimum_ess in (1.0, 1.5)
    )


def _canonical_policy(policy: DecisionPolicy) -> tuple[float, float, int, int, float]:
    return (
        policy.pass_threshold,
        policy.fail_threshold,
        policy.min_valid_reviewers,
        policy.min_lineages,
        policy.min_effective_sample_size,
    )


def select_decision_policy(
    truths: Mapping[str, Truth],
    posteriors: Mapping[str, FusedPosterior | None],
    gates: Mapping[str, Sequence[HardGate]],
    *,
    costs: DecisionCosts,
    min_coverage: float = 0.50,
) -> PolicySelection:
    truth_by_id = _validated_truths(truths)
    if set(truth_by_id.values()) != {Truth.PASS, Truth.FAIL}:
        raise ValueError("policy partition must contain both PASS and FAIL classes")
    case_ids = set(truth_by_id)
    posterior_by_id = _mapping_copy(posteriors, "posteriors")
    gate_by_id = _mapping_copy(gates, "gates")
    _require_exact_ids(case_ids, set(posterior_by_id), "posteriors")
    _require_exact_ids(case_ids, set(gate_by_id), "gates")
    if not isinstance(costs, DecisionCosts):
        raise TypeError("costs must be a DecisionCosts")
    minimum_coverage = _probability(min_coverage, "min_coverage")

    checked_posteriors: dict[str, FusedPosterior | None] = {}
    checked_gates: dict[str, tuple[HardGate, ...]] = {}
    for case_id in case_ids:
        posterior = posterior_by_id[case_id]
        if posterior is not None and not isinstance(posterior, FusedPosterior):
            raise TypeError(
                f"posteriors[{case_id!r}] must be a FusedPosterior or None"
            )
        checked_posteriors[case_id] = posterior
        raw_gates = gate_by_id[case_id]
        if isinstance(raw_gates, (str, bytes)) or not isinstance(
            raw_gates,
            Sequence,
        ):
            raise TypeError(f"gates[{case_id!r}] must be a sequence")
        checked_gates[case_id] = tuple(raw_gates)

    candidates: list[
        tuple[DecisionPolicy, float, float, float]
    ] = []
    for policy in policy_candidates():
        decisions = {
            case_id: decide(
                checked_posteriors[case_id],
                checked_gates[case_id],
                policy,
            )
            for case_id in truth_by_id
        }
        metrics = evaluate_decisions(
            truth_by_id,
            decisions,
            costs=costs,
        )
        candidates.append(
            (
                policy,
                metrics["decision_loss"],
                metrics["false_pass_rate"],
                metrics["coverage"],
            )
        )

    feasible = [
        candidate
        for candidate in candidates
        if candidate[3] >= minimum_coverage
    ]
    if feasible:
        winner = min(
            feasible,
            key=lambda candidate: (
                candidate[1],
                candidate[2],
                -candidate[3],
                _canonical_policy(candidate[0]),
            ),
        )
        constraint_satisfied = True
    else:
        winner = min(
            candidates,
            key=lambda candidate: (
                -candidate[3],
                candidate[1],
                candidate[2],
                _canonical_policy(candidate[0]),
            ),
        )
        constraint_satisfied = False
    return PolicySelection(
        policy=winner[0],
        constraint_satisfied=constraint_satisfied,
        decision_loss=winner[1],
        coverage=winner[3],
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{field} must be positive")
    return numeric


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{field} must be non-negative")
    return numeric


def _metric_value(
    metric: Callable[[Sequence[Mapping[str, object]]], float],
    rows: Sequence[Mapping[str, object]],
) -> float:
    result = _finite_real(metric(rows), "metric result")
    return result


def stratified_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    metric: Callable[[Sequence[Mapping[str, object]]], float],
    *,
    strata: Sequence[str],
    draws: int = 2_000,
    seed: int,
) -> tuple[float, float, float]:
    if not callable(metric):
        raise TypeError("metric must be callable")
    checked_draws = _positive_int(draws, "draws")
    checked_seed = _non_negative_int(seed, "seed")
    checked_rows = tuple(rows)
    if not checked_rows:
        raise ValueError("rows must not be empty")

    stratum_names = tuple(strata)
    if not stratum_names:
        raise ValueError("strata must not be empty")
    for index, name in enumerate(stratum_names):
        if not isinstance(name, str):
            raise TypeError(f"strata[{index}] must be a string")
        if not name.strip():
            raise ValueError(f"strata[{index}] must not be blank")
    if len(set(stratum_names)) != len(stratum_names):
        raise ValueError("strata names must be unique")

    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for index, row in enumerate(checked_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        missing = [name for name in stratum_names if name not in row]
        if missing:
            raise ValueError(
                f"rows[{index}] is missing stratum fields: {', '.join(missing)}"
            )
        key = tuple(row[name] for name in stratum_names)
        try:
            hash(key)
        except TypeError as error:
            raise TypeError(
                f"rows[{index}] stratum values must be hashable"
            ) from error
        grouped.setdefault(key, []).append(row)

    point = _metric_value(metric, checked_rows)
    generator = np.random.default_rng(checked_seed)
    estimates = np.empty(checked_draws, dtype=float)
    for draw_index in range(checked_draws):
        sampled: list[Mapping[str, object]] = []
        for group in grouped.values():
            indices = generator.integers(0, len(group), size=len(group))
            sampled.extend(group[int(index)] for index in indices)
        estimates[draw_index] = _metric_value(metric, sampled)

    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return point, float(lower), float(upper)
