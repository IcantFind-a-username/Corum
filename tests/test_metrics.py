from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import FrozenInstanceError
from decimal import localcontext
from math import isfinite, isnan, log
from typing import Any, cast

import pytest

from corum.baselines import DecisionCosts
from corum.decision import DecisionPolicy
from corum.metrics import (
    PolicySelection,
    evaluate_decisions,
    policy_candidates,
    select_decision_policy,
    stratified_paired_bootstrap,
    target_prevalence_weights,
)
from corum.models import Action, Decision, FusedPosterior, Truth


def _posterior(
    *,
    probability: float = 0.5,
    lower: float = 0.4,
    upper: float = 0.6,
) -> FusedPosterior:
    return FusedPosterior(
        pass_probability=probability,
        lower=lower,
        upper=upper,
        valid_reviewers=2,
        lineage_count=2,
        effective_sample_size=2.0,
        samples=(),
    )


def _policy_tuple(policy: DecisionPolicy) -> tuple[float, float, int, int, float]:
    return (
        policy.pass_threshold,
        policy.fail_threshold,
        policy.min_valid_reviewers,
        policy.min_lineages,
        policy.min_effective_sample_size,
    )


def test_evaluate_decisions_matches_four_case_hand_calculation() -> None:
    truths = {
        "a": Truth.FAIL,
        "b": Truth.PASS,
        "c": Truth.PASS,
        "d": Truth.FAIL,
    }
    decisions = {
        "a": Action.PASS,
        "b": Action.FAIL,
        "c": Action.PASS,
        "d": Action.DEFER,
    }
    probabilities = {"a": 0.9, "b": 0.2, "c": 0.8, "d": 0.1}

    metrics = evaluate_decisions(
        truths,
        decisions,
        probabilities=probabilities,
    )

    assert set(metrics) == {
        "coverage",
        "defer_rate",
        "false_pass_rate",
        "false_fail_rate",
        "false_safe_risk",
        "selective_risk",
        "decision_loss",
        "brier",
        "log_loss",
        "ece",
    }
    assert metrics == pytest.approx(
        {
            "coverage": 0.75,
            "defer_rate": 0.25,
            "false_pass_rate": 0.5,
            "false_fail_rate": 0.5,
            "false_safe_risk": 0.5,
            "selective_risk": 2.0 / 3.0,
            "decision_loss": 0.325,
            "brier": 0.375,
            "log_loss": 1.06013176810005,
            "ece": 0.5,
        }
    )


def test_evaluate_decisions_applies_one_declared_weighting() -> None:
    truths = {
        "a": Truth.FAIL,
        "b": Truth.PASS,
        "c": Truth.PASS,
        "d": Truth.FAIL,
    }
    decisions = {
        "a": Action.PASS,
        "b": Action.FAIL,
        "c": Action.PASS,
        "d": Action.DEFER,
    }
    probabilities = {"a": 0.9, "b": 0.2, "c": 0.8, "d": 0.1}
    weights = target_prevalence_weights(
        truths,
        target_fail_prevalence=0.2,
    )

    metrics = evaluate_decisions(
        truths,
        decisions,
        probabilities=probabilities,
        sample_weights=weights,
    )

    assert weights == pytest.approx({"a": 0.4, "b": 1.6, "c": 1.6, "d": 0.4})
    assert metrics == pytest.approx(
        {
            "coverage": 0.9,
            "defer_rate": 0.1,
            "false_pass_rate": 0.5,
            "false_fail_rate": 0.5,
            "false_safe_risk": 0.2,
            "selective_risk": 5.0 / 9.0,
            "decision_loss": 0.19,
            "brier": 0.354,
            "log_loss": 0.973827146364511,
            "ece": 0.5,
        }
    )


def test_metrics_are_independent_of_callers_decimal_context() -> None:
    truths = {"fail": Truth.FAIL, "pass": Truth.PASS}
    decisions = {"fail": Action.PASS, "pass": Action.FAIL}
    probabilities = {"fail": 0.8, "pass": 0.3}
    expected = evaluate_decisions(
        truths,
        decisions,
        probabilities=probabilities,
        sample_weights={"fail": 1.0, "pass": 3.0},
    )

    with localcontext() as context:
        context.prec = 1
        actual = evaluate_decisions(
            truths,
            decisions,
            probabilities=probabilities,
            sample_weights={"fail": 1.0, "pass": 3.0},
        )

    assert actual == pytest.approx(expected)


def test_decision_wrappers_and_weight_scaling_do_not_change_metrics() -> None:
    truths = {"fail": Truth.FAIL, "pass": Truth.PASS}
    actions = {"fail": Action.FAIL, "pass": Action.PASS}
    wrapped = {
        case_id: Decision(action, ("test",), None)
        for case_id, action in actions.items()
    }
    probabilities = {"fail": 0.1, "pass": 0.9}
    weights = {"fail": 0.4, "pass": 1.6}

    direct = evaluate_decisions(
        truths,
        actions,
        probabilities=probabilities,
        sample_weights=weights,
    )
    scaled = evaluate_decisions(
        truths,
        wrapped,
        probabilities=probabilities,
        sample_weights={case_id: 7.0 * weight for case_id, weight in weights.items()},
    )

    assert direct == pytest.approx(scaled)


@pytest.mark.parametrize("scale", [1e308, 5e-324])
def test_weight_scaling_is_stable_at_float_extremes(scale: float) -> None:
    truths = {"fail": Truth.FAIL, "pass": Truth.PASS}
    decisions = {"fail": Action.PASS, "pass": Action.FAIL}
    probabilities = {"fail": 0.8, "pass": 0.3}
    reference = evaluate_decisions(
        truths,
        decisions,
        probabilities=probabilities,
        sample_weights={"fail": 1.0, "pass": 1.0},
    )

    extreme = evaluate_decisions(
        truths,
        decisions,
        probabilities=probabilities,
        sample_weights={"fail": scale, "pass": scale},
    )

    assert extreme == pytest.approx(reference)


def test_conditional_rates_preserve_tiny_positive_class_mass() -> None:
    metrics = evaluate_decisions(
        {"rare-fail": Truth.FAIL, "common-pass": Truth.PASS},
        {"rare-fail": Action.PASS, "common-pass": Action.PASS},
        sample_weights={"rare-fail": 5e-324, "common-pass": 1e308},
    )

    assert metrics["false_pass_rate"] == 1.0


def test_weighted_sums_preserve_representable_aggregate_subnormal_mass() -> None:
    epsilon = 5e-324
    truths = {
        "tiny-a": Truth.FAIL,
        "tiny-b": Truth.FAIL,
        "large": Truth.FAIL,
    }
    weights = {"tiny-a": epsilon, "tiny-b": epsilon, "large": 2.0}

    metrics = evaluate_decisions(
        truths,
        {
            "tiny-a": Action.PASS,
            "tiny-b": Action.PASS,
            "large": Action.FAIL,
        },
        probabilities={"tiny-a": 1.0, "tiny-b": 1.0, "large": 0.0},
        sample_weights=weights,
    )

    assert metrics["false_pass_rate"] == epsilon
    assert metrics["brier"] == epsilon


def test_decision_loss_averages_finite_extreme_costs_without_overflow() -> None:
    metrics = evaluate_decisions(
        {"a": Truth.FAIL, "b": Truth.FAIL},
        {"a": Action.PASS, "b": Action.PASS},
        costs=DecisionCosts(false_pass=1e308),
    )

    assert isfinite(metrics["decision_loss"])
    assert metrics["decision_loss"] == pytest.approx(1e308)


def test_decision_loss_preserves_small_actual_cost_when_larger_cost_is_inactive() -> None:
    epsilon = 5e-324

    metrics = evaluate_decisions(
        {"case": Truth.PASS},
        {"case": Action.DEFER},
        costs=DecisionCosts(false_pass=1e308, defer=epsilon),
    )

    assert metrics["decision_loss"] == epsilon


def test_undefined_conditionals_are_nan_and_not_silent_zero() -> None:
    truths = {"fail": Truth.FAIL, "pass": Truth.PASS}
    decisions = {case_id: Action.DEFER for case_id in truths}

    metrics = evaluate_decisions(truths, decisions)

    assert metrics["coverage"] == 0.0
    assert metrics["defer_rate"] == 1.0
    assert metrics["decision_loss"] == pytest.approx(0.1)
    assert metrics["false_pass_rate"] == 0.0
    assert metrics["false_fail_rate"] == 0.0
    assert isnan(metrics["false_safe_risk"])
    assert isnan(metrics["selective_risk"])
    assert isnan(metrics["brier"])
    assert isnan(metrics["log_loss"])
    assert isnan(metrics["ece"])

    no_fail = evaluate_decisions(
        {"pass": Truth.PASS},
        {"pass": Action.PASS},
    )
    no_pass = evaluate_decisions(
        {"fail": Truth.FAIL},
        {"fail": Action.FAIL},
    )
    assert isnan(no_fail["false_pass_rate"])
    assert isnan(no_pass["false_fail_rate"])


def test_probability_scores_include_deferred_cases_and_clip_only_for_log_loss() -> None:
    truths = {"wrong-pass": Truth.FAIL, "wrong-fail": Truth.PASS}
    probabilities = {"wrong-pass": 1.0, "wrong-fail": 0.0}
    decided = {"wrong-pass": Action.PASS, "wrong-fail": Action.FAIL}
    deferred = {case_id: Action.DEFER for case_id in truths}

    decided_metrics = evaluate_decisions(
        truths,
        decided,
        probabilities=probabilities,
    )
    deferred_metrics = evaluate_decisions(
        truths,
        deferred,
        probabilities=probabilities,
    )

    assert decided_metrics["brier"] == 1.0
    assert decided_metrics["log_loss"] == pytest.approx(-log(1e-15))
    assert isfinite(decided_metrics["log_loss"])
    for name in ("brier", "log_loss", "ece"):
        assert deferred_metrics[name] == pytest.approx(decided_metrics[name])


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"fail": 0.11, "pass": 0.19}, 0.35),
        ({"fail": 0.05, "pass": 0.10}, 0.475),
    ],
)
def test_ece_uses_registered_left_closed_bins(
    probabilities: dict[str, float],
    expected: float,
) -> None:
    metrics = evaluate_decisions(
        {"fail": Truth.FAIL, "pass": Truth.PASS},
        {"fail": Action.DEFER, "pass": Action.DEFER},
        probabilities=probabilities,
    )

    assert metrics["ece"] == pytest.approx(expected)


def test_ece_weights_bin_mass_confidence_and_frequency() -> None:
    metrics = evaluate_decisions(
        {"fail": Truth.FAIL, "pass": Truth.PASS},
        {"fail": Action.DEFER, "pass": Action.DEFER},
        probabilities={"fail": 0.11, "pass": 0.19},
        sample_weights={"fail": 1.0, "pass": 3.0},
    )

    assert metrics["ece"] == pytest.approx(0.58)


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("decisions", {"x": Action.PASS}, "case IDs"),
        ("probabilities", {"x": 0.5}, "case IDs"),
        ("sample_weights", {"x": 1.0}, "case IDs"),
    ],
)
def test_evaluate_decisions_rejects_case_id_mismatches(
    keyword: str,
    value: object,
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "truths": {"case": Truth.PASS},
        "decisions": {"case": Action.PASS},
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=match):
        evaluate_decisions(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("truths", "decisions", "probabilities", "weights", "error"),
    [
        ({}, {}, None, None, ValueError),
        ({"a": "PASS"}, {"a": Action.PASS}, None, None, TypeError),
        ({"a": Truth.PASS}, {"a": "PASS"}, None, None, TypeError),
        ({"a": Truth.PASS}, {"a": Action.PASS}, {"a": -0.1}, None, ValueError),
        ({"a": Truth.PASS}, {"a": Action.PASS}, {"a": True}, None, TypeError),
        ({"a": Truth.PASS}, {"a": Action.PASS}, None, {"a": -1.0}, ValueError),
        ({"a": Truth.PASS}, {"a": Action.PASS}, None, {"a": float("nan")}, ValueError),
        ({"a": Truth.PASS}, {"a": Action.PASS}, None, {"a": True}, TypeError),
        ({"a": Truth.PASS}, {"a": Action.PASS}, None, {"a": 0.0}, ValueError),
    ],
)
def test_evaluate_decisions_rejects_malformed_inputs(
    truths: dict[str, Any],
    decisions: dict[str, Any],
    probabilities: dict[str, Any] | None,
    weights: dict[str, Any] | None,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        evaluate_decisions(
            truths,
            decisions,
            probabilities=probabilities,
            sample_weights=weights,
        )


def test_target_prevalence_weights_are_mean_one_and_hit_the_target() -> None:
    balanced = {
        "f1": Truth.FAIL,
        "f2": Truth.FAIL,
        "p1": Truth.PASS,
        "p2": Truth.PASS,
    }
    unbalanced = {
        "f": Truth.FAIL,
        "p1": Truth.PASS,
        "p2": Truth.PASS,
        "p3": Truth.PASS,
    }

    balanced_weights = target_prevalence_weights(
        balanced,
        target_fail_prevalence=0.2,
    )
    unbalanced_weights = target_prevalence_weights(
        unbalanced,
        target_fail_prevalence=0.5,
    )

    assert balanced_weights == pytest.approx(
        {"f1": 0.4, "f2": 0.4, "p1": 1.6, "p2": 1.6}
    )
    assert sum(balanced_weights.values()) == pytest.approx(4.0)
    assert (
        balanced_weights["f1"] + balanced_weights["f2"]
    ) / sum(balanced_weights.values()) == pytest.approx(0.2)
    assert unbalanced_weights == pytest.approx(
        {"f": 2.0, "p1": 2.0 / 3.0, "p2": 2.0 / 3.0, "p3": 2.0 / 3.0}
    )


@pytest.mark.parametrize(
    ("truths", "target", "error"),
    [
        ({}, 0.2, ValueError),
        ({"p": Truth.PASS}, 0.2, ValueError),
        ({"f": Truth.FAIL}, 0.2, ValueError),
        ({"f": Truth.FAIL, "p": Truth.PASS}, 0.0, ValueError),
        ({"f": Truth.FAIL, "p": Truth.PASS}, 1.0, ValueError),
        ({"f": Truth.FAIL, "p": Truth.PASS}, float("nan"), ValueError),
        ({"f": Truth.FAIL, "p": Truth.PASS}, True, TypeError),
    ],
)
def test_target_prevalence_weights_rejects_invalid_inputs(
    truths: dict[str, Truth],
    target: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        target_prevalence_weights(
            truths,
            target_fail_prevalence=target,  # type: ignore[arg-type]
        )


def test_policy_candidates_are_the_exact_stable_registered_grid() -> None:
    expected = tuple(
        (pass_threshold, fail_threshold, 2, 2, ess)
        for pass_threshold in (0.8, 0.9, 0.95)
        for fail_threshold in (0.05, 0.1, 0.2)
        for ess in (1.0, 1.5)
    )

    assert tuple(map(_policy_tuple, policy_candidates())) == expected


@pytest.mark.parametrize(
    ("fail_count", "pass_count", "costs", "expected"),
    [
        (1, 10, DecisionCosts(), (0.8, 0.05, 2, 2, 1.0)),
        (
            1,
            7,
            DecisionCosts(defer=0.125),
            (0.9, 0.05, 2, 2, 1.0),
        ),
    ],
)
def test_policy_selection_orders_by_loss_then_false_pass_rate(
    fail_count: int,
    pass_count: int,
    costs: DecisionCosts,
    expected: tuple[float, float, int, int, float],
) -> None:
    truths = {
        **{f"fail-{index}": Truth.FAIL for index in range(fail_count)},
        **{f"pass-{index}": Truth.PASS for index in range(pass_count)},
    }
    posteriors = {
        case_id: _posterior(probability=0.9, lower=0.85, upper=0.95)
        for case_id in truths
    }
    gates = {case_id: () for case_id in truths}

    selection = select_decision_policy(
        truths,
        posteriors,
        gates,
        costs=costs,
        min_coverage=0.0,
    )

    assert _policy_tuple(selection.policy) == expected
    assert selection.constraint_satisfied is True


def test_policy_selection_uses_coverage_then_canonical_tuple() -> None:
    truths = {"fail": Truth.FAIL, "pass": Truth.PASS}
    posteriors = {
        case_id: _posterior(probability=0.1, lower=0.05, upper=0.15)
        for case_id in truths
    }
    gates = {case_id: () for case_id in truths}

    selection = select_decision_policy(
        truths,
        posteriors,
        gates,
        costs=DecisionCosts(),
        min_coverage=0.0,
    )

    assert _policy_tuple(selection.policy) == (0.8, 0.2, 2, 2, 1.0)

    decisive_posteriors = {
        "fail": _posterior(probability=0.02, lower=0.01, upper=0.04),
        "pass": _posterior(probability=0.98, lower=0.96, upper=0.99),
    }
    canonical = select_decision_policy(
        truths,
        decisive_posteriors,
        gates,
        costs=DecisionCosts(),
    )
    assert _policy_tuple(canonical.policy) == (0.8, 0.05, 2, 2, 1.0)
    assert canonical.constraint_satisfied is True
    assert canonical.decision_loss == 0.0
    assert canonical.coverage == 1.0


def test_policy_selection_falls_back_to_coverage_when_none_is_feasible() -> None:
    truths = {"danger": Truth.FAIL, "uncertain": Truth.PASS}
    posteriors = {
        "danger": _posterior(probability=0.9, lower=0.85, upper=0.95),
        "uncertain": _posterior(),
    }
    gates = {case_id: () for case_id in truths}

    selection = select_decision_policy(
        truths,
        posteriors,
        gates,
        costs=DecisionCosts(),
        min_coverage=0.75,
    )

    assert _policy_tuple(selection.policy) == (0.8, 0.05, 2, 2, 1.0)
    assert selection.coverage == 0.5
    assert selection.decision_loss == pytest.approx(0.55)
    assert selection.constraint_satisfied is False


class _GuardedMapping(Mapping[str, Any]):
    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)
        self.reads: list[str] = []

    def __getitem__(self, key: str) -> Any:
        if key not in self._values:
            raise AssertionError(f"read outside policy partition: {key}")
        self.reads.append(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def test_policy_selection_reads_only_exact_policy_partition_ids() -> None:
    truths = _GuardedMapping({"f": Truth.FAIL, "p": Truth.PASS})
    posteriors = _GuardedMapping(
        {
            "f": _posterior(probability=0.02, lower=0.01, upper=0.04),
            "p": _posterior(probability=0.98, lower=0.96, upper=0.99),
        }
    )
    gates = _GuardedMapping({"f": (), "p": ()})

    select_decision_policy(
        truths,
        posteriors,
        gates,
        costs=DecisionCosts(),
    )

    for guarded in (truths, posteriors, gates):
        assert set(guarded.reads) == {"f", "p"}


@pytest.mark.parametrize(
    ("truths", "posteriors", "gates", "coverage", "error"),
    [
        ({}, {}, {}, 0.5, ValueError),
        ({"p": Truth.PASS}, {"p": None}, {"p": ()}, 0.5, ValueError),
        (
            {"f": Truth.FAIL, "p": Truth.PASS},
            {"f": None},
            {"f": (), "p": ()},
            0.5,
            ValueError,
        ),
        (
            {"f": Truth.FAIL, "p": Truth.PASS},
            {"f": None, "p": None},
            {"f": (), "p": ()},
            True,
            TypeError,
        ),
        (
            {"f": Truth.FAIL, "p": Truth.PASS},
            {"f": None, "p": None},
            {"f": (), "p": ()},
            1.1,
            ValueError,
        ),
    ],
)
def test_policy_selection_rejects_invalid_partitions(
    truths: dict[str, Truth],
    posteriors: dict[str, FusedPosterior | None],
    gates: dict[str, tuple[()]],
    coverage: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        select_decision_policy(
            truths,
            posteriors,
            gates,
            costs=DecisionCosts(),
            min_coverage=coverage,  # type: ignore[arg-type]
        )


def test_policy_selection_is_an_immutable_result() -> None:
    selection = PolicySelection(DecisionPolicy(), True, 0.1, 0.5)

    with pytest.raises(FrozenInstanceError):
        selection.coverage = 0.9  # type: ignore[misc]


def _paired_benefit(rows: Sequence[Mapping[str, object]]) -> float:
    return sum(
        float(cast(float, row["baseline"]))
        - float(cast(float, row["candidate"]))
        for row in rows
    ) / len(rows)


def test_bootstrap_resamples_indivisible_paired_rows_within_strata() -> None:
    rows = [
        {"scenario": "a", "seed": 0, "baseline": 2.0, "candidate": 1.0},
        {"scenario": "a", "seed": 1, "baseline": 4.0, "candidate": 3.0},
        {"scenario": "b", "seed": 0, "baseline": 7.0, "candidate": 6.0},
    ]
    expected_counts = Counter(row["scenario"] for row in rows)
    calls = 0

    def checked_metric(sample: Sequence[Mapping[str, object]]) -> float:
        nonlocal calls
        calls += 1
        assert Counter(row["scenario"] for row in sample) == expected_counts
        return _paired_benefit(sample)

    result = stratified_paired_bootstrap(
        rows,
        checked_metric,
        strata=("scenario",),
        draws=25,
        seed=7,
    )

    assert result == pytest.approx((1.0, 1.0, 1.0))
    assert calls == 26


def test_bootstrap_is_seed_deterministic_and_keeps_original_point_estimate() -> None:
    rows = [
        {"scenario": "a", "seed": index, "value": float(index**2)}
        for index in range(8)
    ]

    def mean_value(sample: Sequence[Mapping[str, object]]) -> float:
        return sum(
            float(cast(float, row["value"])) for row in sample
        ) / len(sample)

    first = stratified_paired_bootstrap(
        rows,
        mean_value,
        strata=("scenario",),
        draws=101,
        seed=11,
    )
    repeated = stratified_paired_bootstrap(
        rows,
        mean_value,
        strata=("scenario",),
        draws=101,
        seed=11,
    )
    assert first == repeated
    assert first[0] == pytest.approx(sum(index**2 for index in range(8)) / 8)


@pytest.mark.parametrize(
    ("rows", "strata", "draws", "seed", "metric", "error"),
    [
        ([], ("scenario",), 10, 1, _paired_benefit, ValueError),
        ([{"scenario": "a"}], (), 10, 1, lambda rows: 0.0, ValueError),
        ([{"scenario": "a"}], ("scenario", "scenario"), 10, 1, lambda rows: 0.0, ValueError),
        ([{"scenario": "a"}], ("",), 10, 1, lambda rows: 0.0, ValueError),
        ([{"other": "a"}], ("scenario",), 10, 1, lambda rows: 0.0, ValueError),
        ([{"scenario": []}], ("scenario",), 10, 1, lambda rows: 0.0, TypeError),
        ([{"scenario": "a"}], ("scenario",), 0, 1, lambda rows: 0.0, ValueError),
        ([{"scenario": "a"}], ("scenario",), True, 1, lambda rows: 0.0, TypeError),
        ([{"scenario": "a"}], ("scenario",), 10, -1, lambda rows: 0.0, ValueError),
        ([{"scenario": "a"}], ("scenario",), 10, True, lambda rows: 0.0, TypeError),
        ([{"scenario": "a"}], ("scenario",), 10, 1, lambda rows: float("nan"), ValueError),
        ([{"scenario": "a"}], ("scenario",), 10, 1, lambda rows: float("inf"), ValueError),
        (["not-a-row"], ("scenario",), 10, 1, lambda rows: 0.0, TypeError),
        ([{"scenario": "a"}], ("scenario",), 10, 1, None, TypeError),
    ],
)
def test_bootstrap_rejects_malformed_inputs(
    rows: list[Mapping[str, object]],
    strata: tuple[str, ...],
    draws: object,
    seed: object,
    metric: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        stratified_paired_bootstrap(
            rows,
            metric,
            strata=strata,
            draws=draws,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )
