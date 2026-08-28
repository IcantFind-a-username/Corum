# SDD: Task 6 baselines, metrics, and locked core value gate

- Status: approved
- Accepted base: `9366053`
- Exact documentation commit: `docs: lock task 6 evaluation contract`
- Exact implementation commit: `feat: add consensus baselines and metrics`
- Allowed documentation file: `docs/sdd/0006-baselines-metrics-core-value.md`
- Allowed implementation files: `src/corum/baselines.py`, `src/corum/metrics.py`,
  `tests/test_baselines.py`, `tests/test_metrics.py`, and
  `tests/test_core_value.py`

## Outcome

Provide ordinary-vote and calibrated baselines plus deterministic decision, probability,
policy-selection, prevalence-weighting, and paired-bootstrap metrics. Commit an
independent locked judge before its first execution, then determine whether the existing
full-static Corum core has enough synthetic value over ordinary unweighted majority to
justify Task 7. A passing result authorizes external validation only; it is not a
real-world usefulness claim.

## Non-goals

No cascade, UI, project reader, LLM adapter, provider SDK, report renderer, network call,
paid inference, threshold search outside the registered grid, or test-set tuning belongs
in Task 6. The current `evaluate_decisions` interface has no review ledger, so monetary
cost and token metrics remain for the registered cascade/experiment tasks; this task must
not fabricate them as zero. Empty gates in the three early scenarios do not satisfy the
later non-empty 1,000-canary hard-gate requirement.

## Baseline contract

Implement the exact public interfaces in Task 6 of `docs/plans/corum-mvp.md`.

- `DecisionCosts` is immutable and accepts finite non-negative real costs only.
- `majority_decision` counts only `VALID` `PASS` and `FAIL` observations. `ABSTAIN` and
  non-valid executions cast no directional vote; ties, empty panels, and panels without a
  directional vote return `DEFER`.
- `linear_pool_probability` converts every valid observation, including `ABSTAIN`, into
  its reviewer-specific Bayes posterior under the declared prior, then takes the
  unweighted arithmetic mean. It returns `None` when there is no valid observation.
- Both panel baselines reject duplicate reviewers, mixed case IDs, and malformed rows.
  `linear_pool_probability` additionally rejects missing calibrations and calibration
  keys that disagree with their records; majority has no calibration input.
- `best_single_reviewer` requires at least one calibration candidate and a non-empty
  policy case grid, and uses only the supplied policy rows. Every candidate
  must have exactly one row for the same policy-case grid. Non-valid and `ABSTAIN` rows
  produce `DEFER`; other valid observations use the reviewer posterior mean and inclusive
  pass/fail thresholds. Lowest mean registered decision loss wins, with reviewer ID as
  the exact-tie breaker.

## Metric contract

`evaluate_decisions` validates exact case-ID agreement. It computes the requested
weighting in one call; callers invoke it separately and namespace unweighted and target-
weighted reports. Undefined conditional metrics return `math.nan`.

- `coverage = mass(action != DEFER) / total mass` and `defer_rate = 1 - coverage`.
- `false_pass_rate = P(action=PASS | truth=FAIL)` and
  `false_fail_rate = P(action=FAIL | truth=PASS)`.
- `false_safe_risk = P(truth=FAIL | action=PASS)`. It is distinct from
  `false_pass_rate` and is reported as an additional user-facing diagnostic. The Core
  Value Gate's registered two-percentage-point boundary uses `false_pass_rate`. No PASS
  action makes `false_safe_risk` undefined, never zero.
- `selective_risk` conditions on non-deferred actions. Decision loss uses false PASS
  `1.0`, false FAIL `0.2`, DEFER `0.1`, and correct `0` by default.
- Brier, log loss, and ECE score every supplied case probability independently of action.
  Log loss clips only for evaluation at `1e-15`. ECE uses the exact intervals
  `[0.0, 0.1)`, `[0.1, 0.2)`, ..., `[0.8, 0.9)`, `[0.9, 1.0]`, and applies sample weights
  to bin mass, mean confidence, and empirical PASS frequency.
- Sample weights must be finite, non-negative, exact-keyed, and have positive total mass;
  multiplying all weights by a positive constant leaves every metric unchanged.
- `target_prevalence_weights` requires both truth classes and a target strictly inside
  `(0, 1)`. It returns target prevalence divided by empirical prevalence, normalized to
  mean one. A balanced sample at target `P(FAIL)=0.20` therefore assigns `0.4` to FAIL
  and `1.6` to PASS.

`policy_candidates()` returns exactly 18 policies in stable Cartesian order: pass
thresholds `(0.80, 0.90, 0.95)`, fail thresholds `(0.05, 0.10, 0.20)`, fixed reviewer and
lineage quorum two, and ESS `(1.0, 1.5)`. The canonical policy tuple is field order:
`(pass_threshold, fail_threshold, min_valid_reviewers, min_lineages,
min_effective_sample_size)`.

`select_decision_policy` reads only the exact supplied policy IDs and selects on their
empirical distribution, as required by its fixed public signature; target-prevalence
weights are reporting-only here. It requires a non-empty policy partition containing
both truth classes so `false_pass_rate` has a stable denominator. Among candidates
meeting minimum coverage, order by
lower loss, lower `false_pass_rate`, higher coverage, then canonical policy tuple. If none
meet coverage, order by higher coverage, lower loss, lower `false_pass_rate`, then the
same tuple and return `constraint_satisfied=False`.

`stratified_paired_bootstrap` treats a complete row as the indivisible paired sampling
unit, uses `np.random.default_rng(seed)`, resamples the original count independently
within each named stratum, and returns the point estimate plus default-linear 2.5% and
97.5% quantiles. `strata` must contain unique non-blank names, and every row must provide
hashable values for all of them. Empty input or strata, non-finite metric results,
invalid draws, bool/negative seed, or other malformed input fails explicitly.

## Locked Core Value Gate

`tests/test_core_value.py` is included in the registered implementation commit despite
its omission from the roadmap's illustrative `git add` snippet. It must be committed and
independently reviewed before its first execution. This is an owner-approved workflow
exception: pre-commit full verification explicitly ignores only this locked judge, then
the first post-commit command executes it. After that first result, the simulator,
baselines, metrics, judge, scenario snapshots, constants, reference calculations, and
thresholds are frozen. A bounded repair may change only `calibration.py`, `dependence.py`,
`fusion.py`, `decision.py`, and their regression tests. A subsequently proven simulator
defect invalidates the result and requires owner-approved prospective re-registration;
it is never an ordinary repair cycle.

Lock the following constants:

- scenarios: `("independent", "clone_pair", "majority_trap")` with literal snapshots of
  every phase field, reviewer identity, lineage, cost, likelihood, execution rate, and
  correlation target;
- seeds: `tuple(range(20))`;
- per seed: 2,000 calibration cases ordered by case ID, first 1,600 for likelihood and
  dependence fitting, last 400 for policy selection, and 5,000 disjoint test cases;
- prior strength `1.5`, dependence shrinkage `0.25`, minimum overlap `10`, lineage cap
  `1.0`, posterior draws `512`, credible mass `0.95`, matrix chunk size `4_096`, and fusion
  prior PASS `0.80` taken from the calibration phase rather than estimated from test
  truth, with fusion seed `10_000 + 100 * scenario_index + seed`;
- policy selection is performed independently for each scenario/seed on its 400 policy
  cases using the fixed 18-policy grid and minimum coverage `0.50`;
- bootstrap draws `2_000`, seed `20_260_828`, and 60 indivisible rows containing the mean
  paired case-loss benefit for each scenario/seed, resampled within scenario;
- NLL evaluation clipping `1e-15` and the published `DecisionCosts()`.

Corum, majority, and naive fusion receive byte-identical reviews. Gate loss, coverage,
error rates, NLL, and Brier use the empirical unweighted locked test cases; the simulator
has pre-registered `P(FAIL)=0.20`, and no realized test prevalence is estimated and fed
back into scoring or policy. Pooled and per-scenario values are micro metrics formed by
combining their case-level numerators and denominators. Each bootstrap row's
`benefit = majority_loss - corum_loss` is the paired unweighted mean within one seed.

Dependence-aware and
naive contexts reuse the same Dirichlet likelihood draws and prior; the naive ablation
changes only the dependence matrix to identity while retaining lineage metadata and
quorum. Policy and test fusion reuse the same context. The judge independently implements
reference majority, policy ranking, decision loss, coverage, `false_pass_rate`, Brier,
NLL, paired bootstrap, aggregation, and final gate predicates. Production baseline,
metric, policy, and bootstrap results are cross-checked against those references and
never grade themselves.

All of these must hold:

- pooled Corum loss is at most `0.90 * majority loss`, and the stratified paired 95%
  benefit interval has lower bound strictly above zero;
- every one of the 60 policy selections has `constraint_satisfied=True`; Corum loss is no
  more than `0.01` worse in any scenario, pooled coverage is at least `0.50`, pooled
  `false_pass_rate` is finite and no more than `0.02` above majority, and the empty
  registered gate sets produce zero violations without being called canaries;
- every naive NLL/Brier denominator is finite and strictly positive; in `clone_pair` and
  `majority_trap`, the better of `(naive - aware) / naive` for NLL and Brier is at least
  `0.05`; in `independent`, `(aware - naive) / naive <= 0.01` holds separately for NLL
  and Brier.

Failure blocks Task 7. Permit at most three bounded repair cycles with the unchanged
judge, then record `CORE_VALUE_GATE_FAILED` and return pivot/stop judgment to the owner.

## TDD evidence

- Baseline RED/GREEN: `uv run pytest tests/test_baselines.py -q`
- Metric RED/GREEN: `uv run pytest tests/test_metrics.py -q`
- Static: Ruff and mypy on both production modules and their unit tests
- Pre-commit full: repository pytest and coverage with exactly
  `--ignore=tests/test_core_value.py`, branch coverage at least 80%, Ruff, mypy, and
  `git diff --check`
- Locked judge, first run only after commit: `uv run pytest tests/test_core_value.py -q`
- Post-result full: if the locked judge passes, run the complete repository suite with no
  ignore; if it fails, report the failing gate unchanged before any bounded repair.

## Review and completion

Require an independent read-only review of baseline fairness, metric denominators,
weighting, ECE, bootstrap pairing, split isolation, shared likelihood draws, scenario
snapshots, and every locked gate constant before the first judge execution. Fix every
Critical and Important finding prospectively. Report the first gate result unchanged;
synthetic PASS, FAIL, and any blocked repair outcome remain honest evidence rather than a
product claim.
