from collections.abc import Mapping

import numpy as np
import pytest

from corum.models import ExecutionState, Observation, Reviewer, Truth
from corum.simulation import (
    LineageCorrelationDiagnostic,
    ReviewerSpec,
    Scenario,
    ScenarioPhase,
    SimulatedPanel,
    builtin_scenarios,
    simulate_experiment,
    simulate_panel,
)


def _reviewer_spec(
    reviewer_id: str,
    *,
    lineage: str | None = None,
    family: str = "family",
    pass_accuracy: float = 0.80,
    fail_accuracy: float = 0.80,
    abstain: float = 0.05,
    timeout_rate: float = 0.0,
    invalid_rate: float = 0.0,
    cost: float = 1.0,
) -> ReviewerSpec:
    return ReviewerSpec(
        reviewer=Reviewer(
            reviewer_id=reviewer_id,
            vendor="test-vendor",
            family=family,
            lineage=lineage or f"lineage-{reviewer_id}",
            cost=cost,
        ),
        likelihoods=np.array(
            [
                [pass_accuracy, 1.0 - pass_accuracy - abstain, abstain],
                [1.0 - fail_accuracy - abstain, fail_accuracy, abstain],
            ],
            dtype=float,
        ),
        timeout_rate=timeout_rate,
        invalid_rate=invalid_rate,
    )


def _phase(
    reviewers: tuple[ReviewerSpec, ...] | None = None,
    *,
    prior_pass: float = 0.6,
    correlations: Mapping[str, float] | None = None,
    difficulty_rate: float = 0.0,
    informative_missingness: float = 0.0,
    adversarial_reviewer_id: str | None = None,
) -> ScenarioPhase:
    specs = reviewers or (
        _reviewer_spec("r1"),
        _reviewer_spec("r2"),
        _reviewer_spec("r3"),
    )
    return ScenarioPhase(
        reviewers=specs,
        prior_pass=prior_pass,
        lineage_error_correlation={} if correlations is None else correlations,
        difficulty_rate=difficulty_rate,
        informative_missingness=informative_missingness,
        adversarial_reviewer_id=adversarial_reviewer_id,
    )


def _review_signature(panel: SimulatedPanel) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            review.case_id,
            review.reviewer_id,
            review.observation,
            review.state,
            review.input_tokens,
            review.output_tokens,
        )
        for review in panel.reviews
    )


def _phase_signature(phase: ScenarioPhase) -> tuple[object, ...]:
    return (
        tuple(
            (
                spec.reviewer.reviewer_id,
                spec.reviewer.vendor,
                spec.reviewer.family,
                spec.reviewer.lineage,
                spec.reviewer.cost,
                spec.likelihoods.tobytes(),
                spec.timeout_rate,
                spec.invalid_rate,
            )
            for spec in phase.reviewers
        ),
        phase.prior_pass,
        tuple(sorted(phase.lineage_error_correlation.items())),
        phase.difficulty_rate,
        phase.informative_missingness,
        phase.adversarial_reviewer_id,
    )


def _error_by_case(
    panel: SimulatedPanel,
    reviewer_id: str,
) -> dict[str, int]:
    errors: dict[str, int] = {}
    for review in panel.reviews:
        if reviewer_id != review.reviewer_id or review.state is not ExecutionState.VALID:
            continue
        truth = panel.truths[review.case_id]
        correct = Observation.PASS if truth is Truth.PASS else Observation.FAIL
        errors[review.case_id] = int(review.observation is not correct)
    return errors


def test_reviewer_spec_defensively_owns_likelihoods() -> None:
    source = np.array([[0.8, 0.15, 0.05], [0.15, 0.8, 0.05]])
    spec = ReviewerSpec(
        reviewer=Reviewer("r1", "vendor", "family", "lineage"),
        likelihoods=source,
    )

    source[0, 0] = 0.1

    assert spec.likelihoods[0, 0] == pytest.approx(0.8)
    assert not spec.likelihoods.flags.writeable
    with pytest.raises(ValueError):
        spec.likelihoods[0, 0] = 0.2


@pytest.mark.parametrize(
    ("likelihoods", "message"),
    [
        (np.ones((2, 2)), "shape"),
        (np.array([[0.8, 0.3, -0.1], [0.2, 0.7, 0.1]]), "probabilities"),
        (np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.1]]), "sum to one"),
        (np.array([[np.nan, 0.5, 0.5], [0.2, 0.7, 0.1]]), "finite"),
    ],
)
def test_reviewer_spec_rejects_invalid_likelihoods(
    likelihoods: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ReviewerSpec(
            reviewer=Reviewer("r1", "vendor", "family", "lineage"),
            likelihoods=likelihoods,
        )


def test_reviewer_spec_rejects_invalid_execution_rates() -> None:
    with pytest.raises(ValueError, match="sum to at most one"):
        _reviewer_spec("r1", timeout_rate=0.7, invalid_rate=0.4)
    with pytest.raises(TypeError, match="timeout_rate must be a real number"):
        _reviewer_spec("r1", timeout_rate=True)  # type: ignore[arg-type]


def test_scenario_phase_validates_registry_and_correlation_scope() -> None:
    duplicate = (_reviewer_spec("r1"), _reviewer_spec("r1"))
    with pytest.raises(ValueError, match="duplicate reviewer IDs"):
        _phase(duplicate)
    with pytest.raises(ValueError, match="unknown lineage"):
        _phase(correlations={"unknown": 0.5})
    with pytest.raises(ValueError, match="at least two reviewers"):
        _phase(correlations={"lineage-r1": 0.5})
    with pytest.raises(ValueError, match="adversarial reviewer"):
        _phase(adversarial_reviewer_id="unknown")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prior_pass", float("nan")),
        ("prior_pass", 1.1),
        ("difficulty_rate", -0.1),
        ("informative_missingness", float("inf")),
    ],
)
def test_scenario_phase_rejects_invalid_probabilities(
    field: str,
    value: float,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        _phase(**kwargs)  # type: ignore[arg-type]


def test_simulation_is_seeded_reproducible_and_structurally_complete() -> None:
    phase = _phase(difficulty_rate=0.3)

    first = simulate_panel(phase, 500, seed=17)
    replay = simulate_panel(phase, 500, seed=17)
    different = simulate_panel(phase, 500, seed=18)

    assert first.seed == 17
    assert dict(first.truths) == dict(replay.truths)
    assert dict(first.difficulty_by_case) == dict(replay.difficulty_by_case)
    assert _review_signature(first) == _review_signature(replay)
    assert dict(first.truths) != dict(different.truths) or _review_signature(
        first
    ) != _review_signature(different)
    assert len(first.truths) == 500
    assert len(first.reviews) == 1_500
    assert set(first.truths) == set(first.difficulty_by_case) == set(first.gates)
    assert all(gates == () for gates in first.gates.values())
    keys = {(review.case_id, review.reviewer_id) for review in first.reviews}
    assert len(keys) == len(first.reviews)
    assert all(
        (review.state is ExecutionState.VALID) == (review.observation is not None)
        for review in first.reviews
    )


def test_simulated_panel_rejects_duplicate_case_reviewer_rows() -> None:
    panel = simulate_panel(_phase(), 1, seed=17)

    with pytest.raises(ValueError, match="duplicate review"):
        SimulatedPanel(
            seed=panel.seed,
            truths=panel.truths,
            difficulty_by_case=panel.difficulty_by_case,
            reviews=panel.reviews + (panel.reviews[0],),
            gates=panel.gates,
            lineage_diagnostics=panel.lineage_diagnostics,
        )


def test_simulation_matches_truth_likelihood_and_execution_marginals() -> None:
    spec = _reviewer_spec(
        "r1",
        pass_accuracy=0.75,
        fail_accuracy=0.65,
        abstain=0.10,
        timeout_rate=0.08,
        invalid_rate=0.04,
    )
    phase = _phase((spec,), prior_pass=0.7)

    panel = simulate_panel(phase, 30_000, seed=91)

    pass_rate = sum(truth is Truth.PASS for truth in panel.truths.values()) / 30_000
    timeout_rate = sum(
        review.state is ExecutionState.TIMEOUT for review in panel.reviews
    ) / 30_000
    invalid_rate = sum(
        review.state is ExecutionState.INVALID for review in panel.reviews
    ) / 30_000
    assert pass_rate == pytest.approx(0.7, abs=0.015)
    assert timeout_rate == pytest.approx(0.08, abs=0.01)
    assert invalid_rate == pytest.approx(0.04, abs=0.01)

    for truth_index, truth in enumerate((Truth.PASS, Truth.FAIL)):
        valid = [
            review
            for review in panel.reviews
            if review.state is ExecutionState.VALID
            and panel.truths[review.case_id] is truth
        ]
        frequencies = np.array(
            [
                sum(review.observation is observation for review in valid) / len(valid)
                for observation in (
                    Observation.PASS,
                    Observation.FAIL,
                    Observation.ABSTAIN,
                )
            ]
        )
        np.testing.assert_allclose(
            frequencies,
            spec.likelihoods[truth_index],
            atol=0.02,
            rtol=0.0,
        )


def test_registered_lineage_hits_observed_error_correlation_target() -> None:
    reviewers = (
        _reviewer_spec("clone-a", lineage="clone", family="a"),
        _reviewer_spec("clone-b", lineage="clone", family="b"),
    )
    phase = _phase(
        reviewers,
        prior_pass=0.5,
        correlations={"clone": 0.8},
    )

    panel = simulate_panel(phase, 100_000, seed=1234)
    diagnostic = panel.lineage_diagnostics["clone"]

    assert diagnostic.reviewer_ids == ("clone-a", "clone-b")
    assert diagnostic.target_error_correlation == 0.8
    assert 0.0 <= diagnostic.solved_latent_correlation < 1.0
    assert diagnostic.minimum_eigenvalue >= -1e-12
    assert diagnostic.overlap_count == 100_000
    assert diagnostic.realized_error_correlation == pytest.approx(0.8, abs=0.03)

    first_errors = _error_by_case(panel, "clone-a")
    second_errors = _error_by_case(panel, "clone-b")
    overlap = sorted(first_errors.keys() & second_errors.keys())
    empirical = float(
        np.corrcoef(
            [first_errors[case_id] for case_id in overlap],
            [second_errors[case_id] for case_id in overlap],
        )[0, 1]
    )
    assert diagnostic.realized_error_correlation == pytest.approx(empirical, abs=1e-12)


def test_near_perfect_observed_correlation_target_is_solved_continuously() -> None:
    reviewers = (
        _reviewer_spec("near-clone-a", lineage="near-clone"),
        _reviewer_spec("near-clone-b", lineage="near-clone"),
    )
    phase = _phase(
        reviewers,
        prior_pass=0.5,
        correlations={"near-clone": 0.99995},
    )

    panel = simulate_panel(phase, 10_000, seed=1_234)
    diagnostic = panel.lineage_diagnostics["near-clone"]

    assert 0.0 < diagnostic.solved_latent_correlation < 1.0
    assert diagnostic.realized_error_correlation > 0.99


def test_realized_correlation_diagnostic_accepts_finite_negative_sampling_noise() -> None:
    diagnostic = LineageCorrelationDiagnostic(
        reviewer_ids=("r1", "r2"),
        target_error_correlation=0.0,
        solved_latent_correlation=0.0,
        minimum_eigenvalue=1.0,
        realized_error_correlation=-0.01,
        overlap_count=100,
    )

    assert diagnostic.realized_error_correlation == -0.01


def test_heterogeneous_lineage_target_is_the_unweighted_pair_mean() -> None:
    reviewers = (
        _reviewer_spec(
            "heterogeneous-a",
            lineage="heterogeneous",
            pass_accuracy=0.90,
            fail_accuracy=0.70,
            abstain=0.05,
        ),
        _reviewer_spec(
            "heterogeneous-b",
            lineage="heterogeneous",
            pass_accuracy=0.75,
            fail_accuracy=0.80,
            abstain=0.10,
        ),
        _reviewer_spec(
            "heterogeneous-c",
            lineage="heterogeneous",
            pass_accuracy=0.65,
            fail_accuracy=0.60,
            abstain=0.15,
        ),
    )
    phase = _phase(
        reviewers,
        prior_pass=0.4,
        correlations={"heterogeneous": 0.45},
    )

    panel = simulate_panel(phase, 100_000, seed=4321)
    reviewer_index = {
        spec.reviewer.reviewer_id: index for index, spec in enumerate(reviewers)
    }
    error_matrix = np.empty((100_000, len(reviewers)), dtype=np.int8)
    case_index = {case_id: index for index, case_id in enumerate(panel.truths)}
    for review in panel.reviews:
        truth = panel.truths[review.case_id]
        correct = Observation.PASS if truth is Truth.PASS else Observation.FAIL
        error_matrix[
            case_index[review.case_id],
            reviewer_index[review.reviewer_id],
        ] = int(review.observation is not correct)
    correlation_matrix = np.corrcoef(error_matrix, rowvar=False)
    pair_correlations = correlation_matrix[np.triu_indices(len(reviewers), k=1)]
    expected_mean = float(np.mean(pair_correlations))
    diagnostic = panel.lineage_diagnostics["heterogeneous"]

    assert diagnostic.realized_error_correlation == pytest.approx(
        expected_mean,
        abs=1e-12,
    )
    assert diagnostic.realized_error_correlation == pytest.approx(0.45, abs=0.03)
    assert diagnostic.minimum_eigenvalue >= -1e-12


def test_informative_missingness_depends_on_difficulty_and_truth() -> None:
    spec = _reviewer_spec(
        "r1",
        timeout_rate=0.02,
        invalid_rate=0.02,
    )
    phase = _phase(
        (spec,),
        prior_pass=0.5,
        difficulty_rate=0.5,
        informative_missingness=0.6,
    )

    panel = simulate_panel(phase, 20_000, seed=222)

    missing_by_case = {
        review.case_id: review.state is not ExecutionState.VALID
        for review in panel.reviews
    }
    hard = [
        missing_by_case[case_id]
        for case_id, is_hard in panel.difficulty_by_case.items()
        if is_hard
    ]
    easy = [
        missing_by_case[case_id]
        for case_id, is_hard in panel.difficulty_by_case.items()
        if not is_hard
    ]
    failed_truth = [
        missing_by_case[case_id]
        for case_id, truth in panel.truths.items()
        if truth is Truth.FAIL
    ]
    passed_truth = [
        missing_by_case[case_id]
        for case_id, truth in panel.truths.items()
        if truth is Truth.PASS
    ]
    assert np.mean(hard) > np.mean(easy) + 0.15
    assert np.mean(failed_truth) > np.mean(passed_truth) + 0.15


def test_simulate_experiment_uses_disjoint_derived_seeds_and_case_ids() -> None:
    scenario = Scenario("example", _phase(), _phase(prior_pass=0.4))

    calibration, test = simulate_experiment(
        scenario,
        n_calibration=300,
        n_test=400,
        seed=99,
    )

    assert calibration.seed != test.seed
    assert set(calibration.truths).isdisjoint(test.truths)
    assert len(calibration.truths) == 300
    assert len(test.truths) == 400


def test_empty_independent_panel_is_explicit() -> None:
    panel = simulate_panel(_phase(), 0, seed=5)

    assert panel.seed == 5
    assert dict(panel.truths) == {}
    assert dict(panel.difficulty_by_case) == {}
    assert panel.reviews == ()
    assert dict(panel.gates) == {}
    assert dict(panel.lineage_diagnostics) == {}


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_simulate_panel_rejects_invalid_case_count(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="n_cases"):
        simulate_panel(_phase(), value, seed=1)  # type: ignore[arg-type]


def test_simulate_panel_rejects_invalid_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        simulate_panel(_phase(), 10, seed=True)  # type: ignore[arg-type]


def test_unreachable_registered_correlation_is_rejected() -> None:
    reviewers = (
        _reviewer_spec(
            "r1",
            lineage="mixed",
            pass_accuracy=0.95,
            fail_accuracy=0.95,
            abstain=0.01,
        ),
        _reviewer_spec(
            "r2",
            lineage="mixed",
            pass_accuracy=0.55,
            fail_accuracy=0.55,
            abstain=0.20,
        ),
    )
    phase = _phase(reviewers, correlations={"mixed": 0.99})

    with pytest.raises(ValueError, match="unreachable"):
        simulate_panel(phase, 1_000, seed=7)


def test_builtin_scenarios_are_exact_immutable_and_meaningfully_distinct() -> None:
    scenarios = builtin_scenarios()

    assert set(scenarios) == {
        "independent",
        "clone_pair",
        "majority_trap",
        "informative_missingness",
        "drift",
        "cascade_cost",
    }
    assert all(key == scenario.name for key, scenario in scenarios.items())
    with pytest.raises(TypeError):
        scenarios["new"] = scenarios["independent"]  # type: ignore[index]

    clone = scenarios["clone_pair"]
    assert any(
        0.8 <= target <= 0.95
        for target in clone.test.lineage_error_correlation.values()
    )

    majority = scenarios["majority_trap"].test
    lineage_counts = {
        lineage: sum(
            spec.reviewer.lineage == lineage for spec in majority.reviewers
        )
        for lineage in {spec.reviewer.lineage for spec in majority.reviewers}
    }
    assert max(lineage_counts.values()) >= 2
    accuracies = [
        float((spec.likelihoods[0, 0] + spec.likelihoods[1, 1]) / 2.0)
        for spec in majority.reviewers
    ]
    assert max(accuracies) > min(accuracies)

    informative = scenarios["informative_missingness"].test
    assert informative.difficulty_rate > 0
    assert informative.informative_missingness > 0
    assert any(
        spec.timeout_rate > 0 and spec.invalid_rate > 0
        for spec in informative.reviewers
    )

    drift = scenarios["drift"]
    assert drift.calibration.prior_pass != drift.test.prior_pass
    assert any(
        not np.array_equal(calibration.likelihoods, test.likelihoods)
        for calibration, test in zip(
            drift.calibration.reviewers,
            drift.test.reviewers,
            strict=True,
        )
    )
    assert (
        dict(drift.calibration.lineage_error_correlation)
        != dict(drift.test.lineage_error_correlation)
    )
    assert drift.calibration.informative_missingness != drift.test.informative_missingness
    adversarial_id = drift.test.adversarial_reviewer_id
    assert adversarial_id is not None
    calibration_by_id = {
        spec.reviewer.reviewer_id: spec for spec in drift.calibration.reviewers
    }
    test_by_id = {spec.reviewer.reviewer_id: spec for spec in drift.test.reviewers}
    calibration_adversary = calibration_by_id[adversarial_id]
    test_adversary = test_by_id[adversarial_id]
    for truth_index in range(2):
        correct_index = truth_index
        wrong_index = 1 - truth_index
        assert (
            calibration_adversary.likelihoods[truth_index, correct_index]
            > calibration_adversary.likelihoods[truth_index, wrong_index]
        )
        assert (
            test_adversary.likelihoods[truth_index, wrong_index]
            > test_adversary.likelihoods[truth_index, correct_index]
        )

    cascade = scenarios["cascade_cost"].test
    assert cascade.difficulty_rate > 0
    assert len({spec.reviewer.cost for spec in cascade.reviewers}) > 1


def test_every_builtin_phase_runs_through_the_public_simulator() -> None:
    seed = 7_000
    for scenario in builtin_scenarios().values():
        for phase in (scenario.calibration, scenario.test):
            panel = simulate_panel(phase, 1_000, seed=seed)
            seed += 1
            assert len(panel.truths) == 1_000
            assert len(panel.reviews) == 1_000 * len(phase.reviewers)


def test_every_builtin_registered_lineage_meets_the_statistical_gate() -> None:
    checked: set[tuple[str, str, str]] = set()
    seed = 8_000
    previous_signature: tuple[object, ...] | None = None
    previous_panel: SimulatedPanel | None = None
    for scenario_name, scenario in builtin_scenarios().items():
        for phase_name, phase in (
            ("calibration", scenario.calibration),
            ("test", scenario.test),
        ):
            if not phase.lineage_error_correlation:
                continue
            statistical_phase = ScenarioPhase(
                reviewers=phase.reviewers,
                prior_pass=phase.prior_pass,
                lineage_error_correlation=phase.lineage_error_correlation,
                difficulty_rate=phase.difficulty_rate,
                informative_missingness=0.0,
                adversarial_reviewer_id=phase.adversarial_reviewer_id,
            )
            signature = _phase_signature(statistical_phase)
            if signature == previous_signature:
                assert previous_panel is not None
                panel = previous_panel
            else:
                panel = simulate_panel(statistical_phase, 100_000, seed=seed)
                seed += 1
                previous_signature = signature
                previous_panel = panel
            for lineage, target in phase.lineage_error_correlation.items():
                diagnostic = panel.lineage_diagnostics[lineage]
                assert diagnostic.realized_error_correlation == pytest.approx(
                    target,
                    abs=0.03,
                )
                assert diagnostic.minimum_eigenvalue >= -1e-12
                checked.add((scenario_name, phase_name, lineage))

    assert checked == {
        ("clone_pair", "calibration", "clone"),
        ("clone_pair", "test", "clone"),
        ("majority_trap", "calibration", "weak-clones"),
        ("majority_trap", "test", "weak-clones"),
        ("drift", "calibration", "shared"),
        ("drift", "test", "shared"),
    }
