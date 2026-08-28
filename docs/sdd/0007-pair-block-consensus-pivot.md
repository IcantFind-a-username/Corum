# SDD: Task 6B pair-block consensus pivot

- Status: approved
- Accepted base: `f84fbb4`
- Exact documentation commit: `docs: register pair-block consensus pivot`
- Exact test-contract commit: `test: lock pair-block value gate`
- Exact implementation commit: `feat: fuse calibrated reviewer pairs`
- Allowed documentation files: `AGENTS.md`, `docs/specs/corum-mvp-design.md`,
  `docs/plans/corum-mvp.md`, and this SDD
- Allowed test-contract files: `tests/test_calibration.py`, `tests/test_fusion.py`, and
  `tests/test_pair_value.py`
- Allowed implementation files: `src/corum/__init__.py`, `src/corum/calibration.py`,
  `src/corum/fusion.py`, and `scripts/benchmark_fusion.py`

## Outcome

Replace the rejected power-likelihood dependence mechanism as Corum's candidate core with
an optional, fixed pair-block model that learns a nine-cell joint observation likelihood
for one or more disjoint reviewer pairs. Prove the implementation against analytic,
missingness, negative-control, scalar/batch, and immutability tests. Then run a fresh
independent synthetic judge against ordinary majority, naive independent Bayes, and the
frozen `f84fbb4` power heuristic. Task 7 remains blocked unless both registered gates
pass. Synthetic success authorizes external validation only; it does not prove usefulness
on real developer projects.

## Decision record and non-goals

Task 6A remains `CORE_VALUE_GATE_FAILED`: repaired pooled decision loss was `0.03962`
versus majority `0.06664`, while the registered `majority_trap` probability improvement
was `3.5293%`, below its `5%` threshold. Those numbers, its seeds, and its judge are not
changed or rerun to admit this pivot. The owner approved this prospective SDD on
2026-08-28.

This task does not implement an adaptive cascade, learned or test-selected pairing,
overlapping/composite pair products, higher-order cliques, a UI, project ingestion, an LLM
adapter, a provider SDK, reporting, network access, or paid inference. It does not change
`dependence.py`, `decision.py`, `simulation.py`, `baselines.py`, `metrics.py`, or
`tests/test_core_value.py`. It does not weaken reviewer, lineage, or ESS quorum and does
not call a pair-block interval a full-panel credible interval or risk guarantee.

## Statistical contract

Let a registered pair be the canonical tuple `(a, b)` with two distinct non-blank
reviewer IDs and `a < b`. For truth `y`, its ordered joint observation has nine cells:

`theta[(a,b), y] ~ Dirichlet(alpha_product_prior[y] + paired_counts[y])`.

Only reviews for the same case and truth where both pair members are `VALID` contribute a
paired count, so the fitted table is conditional on both members being valid. `ABSTAIN` is
a valid third category. A non-valid review or a missing semantic observation contributes
no paired cell. Duplicate reviewer/case rows, conflicting case truth, unknown calibration
keys, and malformed calibration records fail explicitly. Within each truth/both-valid row,
the update assumes calibration cases are conditionally exchangeable and share a stable
joint cell-probability vector.

`reviewer_calibrations` may be the complete panel mapping. It must contain both pair IDs,
each key must match its `ReviewerCalibration.reviewer_id`, and unrelated valid panel
entries are ignored; a missing/mismatched pair entry fails.

The product prior for each truth row is the outer product of the two singleton *inferred
parent priors*: each singleton parent is `(alpha - observed_counts)` normalized by its row
sum. It must not use singleton posterior means, which would directly add each reviewer's
observed counts again. The pooled parent itself is estimated from the fit split and can
indirectly include the same cases, so this is a plug-in empirical-Bayes prior rather than
an independent prior; its center and strength uncertainty are not propagated. The product
prior has total row strength exactly `9.0`. This is one joint-row pseudo-count per cell
under a uniform parent, chosen prospectively to stabilize the 160-case condition; the
fresh judge decides whether that choice earns admission.
Every truth row must contain at least `30` paired-valid fit cases or fitting raises an
actionable error and the caller must omit that pair. If no active pair remains, this
preserves legacy fusion; otherwise those reviewers become ordinary singleton blocks.

The reviewer set is partitioned prospectively into globally disjoint pair and singleton
blocks. Given the explicit cross-block conditional-independence assumption:

`P(observations | truth) = product_B P(observations_B | truth)`.

When both pair members are valid, fusion adds exactly one joint factor and never either
singleton factor. When exactly one is valid, it uses an explicit baseline-compatible
approximation: that reviewer's unchanged separately fitted singleton factor, not the
marginal of the joint draw; the missing member adds nothing. Equality with naive fusion in
this slice is an implementation/compatibility property, not a consequence of the exact
block factorization. The approximation assumes the validity/missingness process is
ignorable for the singleton fallback. Other singleton blocks use exponent one. Pair joint
factors are never multiplied by legacy dependence weights. `DependenceModel` still
computes unchanged all-valid-reviewer ESS and lineage diagnostics for quorum. With no pair
calibration/draw mapping, every output and seeded draw remains byte-for-byte identical to
the legacy power path.

This is a block model, not a complete dependence model. Pair blocks are declared without
policy or test access and may not overlap. Cross-block dependence, higher-order effects,
non-ignorable validity, clustered or repeated cases, unmodeled case-difficulty mixtures,
within-fit nonstationarity, distribution shift, adversarial collusion, and truth-label
error remain limitations tested later on external data. The posterior interval does not
propagate these uncertainties or empirical-Bayes hyperparameter uncertainty.

Primary-source basis:

- Cox and Reid, 2004, low-dimensional pairwise/composite likelihood construction; this
  informs the pair factor but does not make Corum's block product exact:
  <https://doi.org/10.1093/biomet/91.3.729>
- Dawid and Studeny, 1999, conditional-product factorization:
  <https://proceedings.mlr.press/r2/dawid99a.html>
- Mosimann, 1962, compound multinomial and multivariate-beta/Dirichlet construction:
  <https://doi.org/10.1093/biomet/49.1-2.65>

These references motivate the construction but do not justify Corum's outer-product
empirical-Bayes center, strength `9.0`, minimum count `30`, or pair partition; nor do they
prove that Corum beats voting or that the block-independence assumption holds. A single
future output uses a categorical posterior predictive; Dirichlet-multinomial terminology
applies to a batch of counts after integrating over its shared probability vector.

## Public interface

```python
PairKey = tuple[str, str]

@dataclass(frozen=True, slots=True)
class ReviewerPairCalibration:
    reviewer_ids: PairKey
    alpha: np.ndarray             # shape (2, 3, 3)
    observed_counts: np.ndarray   # shape (2, 3, 3)
    prior_strength: float
    min_paired_per_truth: int = 30

    def mean_likelihoods(self) -> np.ndarray: ...
    def sample_likelihoods(
        self,
        draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...          # shape (draws, 2, 3, 3)

def fit_reviewer_pair_calibration(
    reviewer_ids: PairKey,
    examples: Sequence[CalibrationExample],
    *,
    reviewer_calibrations: Mapping[str, ReviewerCalibration],
    prior_strength: float = 9.0,
    min_paired_per_truth: int = 30,
) -> ReviewerPairCalibration: ...

def fuse_known_pair_likelihoods(
    observations: Mapping[str, Observation],
    likelihoods: Mapping[str, np.ndarray],       # singleton shape (2, 3)
    pair_likelihoods: Mapping[PairKey, np.ndarray],  # joint shape (2, 3, 3)
    *,
    prior_pass: float,
) -> float: ...
```

`FusionContext` appends an empty-by-default immutable
`pair_likelihood_draws: Mapping[PairKey, np.ndarray]` field. `build_fusion_context`
appends the optional keyword
`pair_calibrations: Mapping[PairKey, ReviewerPairCalibration] | None = None`.
Singleton draws are sampled first in the existing reviewer order; joint draws are sampled
afterward in sorted pair-key order. Therefore an absent/empty pair mapping consumes no
additional random values and preserves the legacy seeded result.

The two empty-mapping semantics are intentionally different and tested:
`fuse_known_pair_likelihoods(..., pair_likelihoods={})` is an exponent-one naive singleton
oracle, while an empty `FusionContext.pair_likelihood_draws` selects the byte-compatible
legacy power path using its `DependenceModel` weights.

`ReviewerPairCalibration` validates that both observed-count row totals meet its stored
`min_paired_per_truth`, so direct construction cannot bypass the activation invariant.
All stored arrays are defensive read-only copies. Singleton draws have shape
`(draws, 2, 3)`; pair draws have shape `(draws, 2, 3, 3)`; every truth row is finite,
within `[0, 1]`, and sums to one. All draw counts match. Pair mappings reject noncanonical,
self, blank, overlapping, unknown, mismatched-record, or duplicate logical pairs. Scalar
and matrix fusion use the same kernel, `valid_mask` remains the only authority in matrix
fusion, and reviewer column permutations preserve results exactly after canonicalization.

## Unit and integration acceptance

Tests are written and committed before production changes. They cover:

- a hand-counted `2 x 3 x 3` fit including `ABSTAIN`, invalid states, shuffled input, and
  both truth rows;
- exact product-parent prior construction, proof that reviewer-specific observed counts
  are not directly added to that empirical-Bayes center, row strength `9.0`, sparse-row
  rejection at 29/30, and key errors;
- constructor validation, finite normalized posterior means, deterministic draws,
  defensive immutability, malformed arrays, and draw-count errors;
- a hand-derived known joint Bayes result to `1e-12`, outer-product reduction to naive
  Bayes, complete-pair single counting, and canonical pair validation;
- exactly-one-member fallback equality with naive fusion to `1e-12`, all-invalid behavior,
  calibrated abstention, and unchanged quorum metadata;
- byte-for-byte legacy outputs with no pair mapping, fixed-seed reproducibility with pair
  mappings, common parameter draws across cases, scalar/matrix equality, mixed execution
  states, reviewer permutations, and matrix chunk equality;
- 10,000 cases, four reviewers, and 256 draws with pair fusion under five seconds and
  working arrays below 512 MiB on the development CPU.

Repository branch coverage remains at least 80%. The historical failed
`tests/test_core_value.py` is retained and excluded from ordinary green verification; it
is not deleted, weakened, or treated as a current required pass.

## Fresh locked data design

The independent judge is `tests/test_pair_value.py`. It is reviewed and then committed
without execution before production implementation. It locks:

```python
SCENARIO_NAMES = (
    "heterogeneous_pair",
    "low_prevalence_pair",
    "same_lineage_independent",
    "missing_pair_member",
)
SEEDS = tuple(10_007 + 137 * i for i in range(8))  # base seeds
FIT_CASES = (160, 640)
POLICY_CASES = 400
TEST_CASES = 2_500
POSTERIOR_DRAWS = 256
CREDIBLE_MASS = 0.95
MATRIX_CHUNK_SIZE = 4_096
MARGINAL_PRIOR_STRENGTH = 1.5
PAIR_PRODUCT_PRIOR_STRENGTH = 9.0
MIN_PAIRED_PER_TRUTH = 30
DEPENDENCE_SHRINKAGE = 0.25
MINIMUM_OVERLAP = 10
LINEAGE_CAP = 1.0
MINIMUM_COVERAGE = 0.50
NLL_EPSILON = 1e-15
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20_260_917
```

For zero-based `scenario_index`, `fit_index`, and each `base_seed` above, the exact RNG
derivation is:

```python
simulation_seed = base_seed + 100_000 * scenario_index + 10_000 * fit_index
fusion_seed = 2_000_000 + simulation_seed
```

Every scenario/fit-size cell therefore has an independent simulation/test panel instead
of reusing one test panel as if it were an independent bootstrap observation. The three
probability methods in one cell use the scenario literal `prior_pass` as the inference
prior. The pair context samples singleton arrays and then joint arrays from
`default_rng(fusion_seed)`; the judge constructs naive and power contexts from those exact
singleton arrays and asserts byte equality rather than merely recreating them from a seed.

The published loss is `DecisionCosts(false_pass=1.0, false_fail=0.2, defer=0.1)`.
There are `4 scenarios x 2 fit sizes x 8 seeds = 64` runs and about 210,000 generated
cases. Each run gives its first fit-size cases only to likelihood/dependence fitting, its
next 400 cases only to policy selection, and 2,500 disjoint cases only to testing. The
scenario name, correlation target, and test truth never enter model fitting or fusion.
Naive, power, and pair methods receive the same observations, class prior, marginal
calibrations, singleton Dirichlet draws, and reviewer lineage metadata. Policy and test
reuse one context per method. The probability and quorum contracts are deliberately
separate: naive uses an identity `DependenceModel`, so its factor exponents are one and its
ESS is the number of valid reviewers; power uses the fit-only `DependenceModel` for both
its factor exponents and ESS; pair uses exponent-one block factors but the same fitted
model as power for reviewer-level ESS diagnostics. All three use the actual distinct
lineage count. Thus Gate A isolates probability quality, while Gate B compares each fully
specified method including its declared quorum behavior.

The four scenario literals below use row order `(truth PASS, truth FAIL)` and column order
`(observation PASS, FAIL, ABSTAIN)`.

In every scenario, reviewer `vendor` is `"synthetic"`, `family` equals `reviewer_id`, cost
is `1.0`, `difficulty_rate=0.0`, `informative_missingness=0.0`, and
`adversarial_reviewer_id=None`. Calibration and test phases have the same reviewer order,
prior, likelihoods, and correlation map unless the missing scenario explicitly changes a
test timeout. All timeout and invalid rates are zero unless listed. Thus the registered
missing fallback is missing-completely-at-random in this judge; non-ignorable validity
remains an external limitation.

### `heterogeneous_pair`

- prior PASS `0.62`; pair error correlation `0.58`; pair members are columns 1 and 3
- active pair key `("pair-a", "pair-b")`
- `solo-high`, lineage `solo-high`: `(.86,.10,.04)`, `(.13,.83,.04)`
- `pair-b`, lineage `pair`: `(.71,.22,.07)`, `(.28,.66,.06)`
- `solo-mid`, lineage `solo-mid`: `(.75,.18,.07)`, `(.24,.71,.05)`
- `pair-a`, lineage `pair`: `(.78,.17,.05)`, `(.22,.73,.05)`

### `low_prevalence_pair`

- prior PASS `0.45`; pair error correlation `0.72`; pair members are columns 0 and 2
- active pair key `("pair-x", "pair-y")`
- `pair-x`, lineage `pair`: `(.68,.26,.06)`, `(.30,.64,.06)`
- `solo-high`, lineage `solo-high`: `(.88,.08,.04)`, `(.12,.84,.04)`
- `pair-y`, lineage `pair`: `(.74,.20,.06)`, `(.26,.69,.05)`
- `solo-mid`, lineage `solo-mid`: `(.77,.17,.06)`, `(.22,.73,.05)`

### `same_lineage_independent`

- prior PASS `0.55`; pair members share lineage `pair`, but correlation mapping is empty
- active pair key `("pair-a", "pair-b")`
- `solo-high`, lineage `solo-high`: `(.84,.12,.04)`, `(.12,.84,.04)`
- `pair-b`, lineage `pair`: `(.70,.23,.07)`, `(.23,.70,.07)`
- `solo-mid`, lineage `solo-mid`: `(.75,.19,.06)`, `(.19,.75,.06)`
- `pair-a`, lineage `pair`: `(.77,.18,.05)`, `(.18,.77,.05)`

### `missing_pair_member`

- prior PASS `0.70`; pair error correlation `0.64`; pair members are `pair-a`, `pair-b`
- active pair key `("pair-a", "pair-b")`
- calibration timeout rates are zero; test `pair-b` timeout rate is `0.50`; all others zero
- `solo-high`, lineage `solo-high`: `(.85,.11,.04)`, `(.11,.85,.04)`
- `solo-mid`, lineage `solo-mid`: `(.72,.20,.08)`, `(.20,.72,.08)`
- `pair-a`, lineage `pair`: `(.73,.22,.05)`, `(.22,.73,.05)`
- `pair-b`, lineage `pair`: `(.75,.20,.05)`, `(.20,.75,.05)`

The judge compares four methods:

1. frozen ordinary majority: only valid PASS/FAIL votes count; abstentions, missing votes,
   ties, and empty votes defer;
2. naive Bayes: singleton likelihood factors all have exponent one and identity-model
   ESS equals the number of valid reviewers;
3. frozen power heuristic from `f84fbb4`: `w_i = 1 / (1 + sum_j max(rho_ij, 0))`
   fitted with the constants above, used for both powers and ESS, and cross-checked against
   a small reference formula;
4. pair block: a both-valid pair contributes its joint factor once, exactly-one-valid
   contributes the shared singleton factor, other singletons have exponent one, and the
   fitted model from method 3 supplies ESS without tempering likelihoods.

## Gate A: pair-block admission

NLL is the sole primary probability metric; the judge may not choose between NLL and
Brier after seeing results. Pooling is at the case level, while paired uncertainty
resamples seed-run rows within scenario and fit-size strata. Define:

`relative_improvement(base) = (pooled_NLL_base - pooled_NLL_pair) / pooled_NLL_base`.

Each indivisible bootstrap row stores the method-specific metric sum and denominator for
one independently derived seed/scenario/fit-size slice. A bootstrap draw samples eight
complete rows with replacement inside every included scenario/fit-size stratum, then
recomputes micro-pooled metrics from sums and denominators before taking the paired
difference or relative degradation. It uses
`np.random.default_rng(BOOTSTRAP_SEED)`, 2,000 draws, and NumPy's default linear 2.5% and
97.5% quantiles. Missing-member sub-slices retain their own observed denominators; no run
or case receives a post-hoc weight.

All of the following must hold:

1. Across `heterogeneous_pair`, `low_prevalence_pair`, and `missing_pair_member`, pair NLL
   improves by at least 3% over naive and separately at least 3% over power. Both paired
   95% bootstrap lower bounds for `NLL_base - NLL_pair` are strictly positive.
2. In the same correlated pool restricted to `FIT_CASES=160`, both improvements are at
   least 1%, with both bootstrap lower bounds strictly positive.
3. Correlated-pool pair Brier is no greater than naive and separately no greater than
   power Brier. In every individual scenario/fit-size slice, pair NLL and Brier are each
   no more than 1% worse than naive and separately no more than 1% worse than power.
4. In `same_lineage_independent`, pair degradation versus naive and versus power is at
   most 1% separately for NLL and Brier. Each paired bootstrap 95% upper bound on relative
   NLL degradation is at most 1%. For metric `M`, relative degradation is fixed as
   `(M_pair - M_base) / M_base`. Brier has only these point guardrails; the judge does not
   add or select a Brier interval.
5. In every `missing_pair_member` run, the exactly-one-valid-pair fraction is within
   `[0.44, 0.56]`; on that slice pair and naive `pass_probability`, `lower`, and `upper`
   each agree within `1e-12`, and valid-reviewer/lineage counts agree exactly. Pair ESS is
   separately checked against the fitted model and is not required to equal identity-naive
   ESS. On the pooled both-valid slice, pair NLL improves by at least 2% over naive and
   separately at least 2% over power, with both paired bootstrap lower bounds strictly
   positive; bootstrap strata are fit size.
6. All 64 runs activate exactly their one registered pair, with both joint-count truth-row
   totals at least 30; sparse calibration is a gate failure and no run is omitted or
   silently changed to power fusion. Every probability, likelihood row, score, and interval
   is finite, within range, and normalized as applicable. Joint counts are finite,
   non-negative integers whose row totals match paired-valid fit cases. Pair choice is
   fit-declared, paired cases are counted once, and no test-dependent selection or unknown
   observation cell occurs.

Every failed attempt records `PAIR_BLOCK_ADMISSION_FAILED` and leaves the component
unadmitted. The bounded repair rule below applies; failure after the repair budget is
exhausted is the final rejection.

## Gate B: fresh vertical core closure

Naive, power, and pair each independently select from the same frozen 18-policy grid on
only their policy split. Majority has no tunable policy. The judge independently recomputes
policy ranking/tie-breaks, decisions, majority, decision loss, coverage, false-PASS rate,
NLL, Brier, and paired bootstrap instead of letting production metrics grade themselves.

All of the following must hold:

- pooled pair decision loss is at most `0.90 * pooled majority loss`, and the paired 95%
  bootstrap lower bound for `majority_loss - pair_loss` is strictly positive;
- pooled pair loss is no greater than pooled power loss and separately no greater than
  pooled naive loss;
- in every scenario/fit-size slice, pair loss is no more than `0.01` worse than each of
  majority, power, and naive separately;
- pair pooled coverage is at least `0.50`, and pair pooled false-PASS rate is no more than
  `0.02` above majority;
- all `64 x 3` probability-method policy selections satisfy the coverage constraint;
- all gate-violation counts are zero and all denominators/results are finite.

Coverage is `P(action != DEFER)` and false-PASS rate is
`P(action = PASS | truth = FAIL)` on the locked test cases. Both use micro-pooled case
counts rather than an unweighted mean of run percentages.
The Gate B benefit interval uses the same bootstrap algorithm defined for Gate A: one
indivisible seed/scenario/fit-size row containing pair and majority loss sums/counts,
eight-row resampling inside every scenario/fit-size stratum, micro-pooling after sampling,
and the locked draw count, seed, and linear quantiles.

Gate B cannot override Gate A. Only a pass on both gates permits a prospective roadmap
update that unlocks Task 7.

## Freeze and stop rule

The documentation commit lands first. The test contract, including the literal scenarios,
reference calculations, constants, thresholds, and judge, is then written RED,
independently reviewed, and committed without running `tests/test_pair_value.py`.
Production code is implemented only after that commit and committed separately. The first
judge execution freezes its complete output as the Task 6B result.

After first execution, scenarios, seeds, splits, baselines, metrics, thresholds, judge, and
all pre-registered test assertions do not change. At most two bounded repair cycles may
change only pair calibration/fusion production code and add or strengthen regression
tests. A repair may never delete, skip, or relax an existing test. Attempts 0, 1, and 2 and
their full outputs remain permanent. During repairs a failing component remains
unadmitted. Gate A pass with Gate B failure leaves it experimental but keeps Task 7
blocked. If either gate remains failed after two repairs, record final rejection/failure
and return pivot/stop judgment to the owner. A proven judge defect invalidates the entire
run and requires a new gate version with an entirely new seed set; it is never repaired in
place and reused as a passing result.

## TDD evidence

- Documentation check: `git diff --check`
- RED calibration/fusion: `uv run pytest tests/test_calibration.py tests/test_fusion.py -q`
- Pre-implementation judge collection remains intentionally unexecuted
- GREEN focused: `uv run pytest tests/test_calibration.py tests/test_fusion.py -q`
- Pre-judge repository suite: `uv run pytest -q --ignore=tests/test_core_value.py
  --ignore=tests/test_pair_value.py`
- Coverage: `uv run pytest --cov=corum --cov-branch --cov-report=term-missing
  --cov-fail-under=80 --ignore=tests/test_core_value.py --ignore=tests/test_pair_value.py`
- Static: `uv run ruff format --check .`, `uv run ruff check .`, and
  `uv run mypy src/corum scripts`
- Performance: `uv run python scripts/benchmark_fusion.py --cases 10000 --reviewers 4
  --draws 256 --pair-block --max-seconds 5`
- First frozen judge run, only after reviewed implementation commit:
  `uv run pytest tests/test_pair_value.py -q`

## Review and completion

The registered performance command activates exactly two pair keys,
`("reviewer-0", "reviewer-1")` and `("reviewer-2", "reviewer-3")`, and its memory report
includes both joint draw tensors and pair-kernel working arrays.

Before the test-contract commit, require independent read-only review of statistical
identifiability, pair prior construction, missing fallback, gate fairness, split isolation,
negative controls, paired uncertainty, reference independence, and literal scenario
snapshots. Before the implementation commit, require review of every new test and confirm
the judge has not executed, plus review of the complete production diff. Before the first
judge run, require independent read-only code review with no open Critical or Important
finding, plus all unit, coverage, static, and performance checks green. Record exact
commits, commands, counts, runtime, both gate verdicts, and remaining external validation.
Never describe a synthetic result as proof of real-project effectiveness.
