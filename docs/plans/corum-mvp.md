# Corum MVP Implementation Plan

> Follow `AGENTS.md` and `DEVELOPMENT.md`. Every production behavior starts with a
> failing test, and every task receives an independent review before the next task.

**Goal:** Build and evaluate a reproducible, low-cost MVP that demonstrates whether
dependence-aware Bayesian consensus can improve decision loss and reviewer-call cost
without hiding uncertainty behind majority voting.

**Architecture:** A small typed Python library owns the statistical core. Review records
remain immutable, calibration learns class-conditional three-way observation
likelihoods, and a registered fixed pair-block path learns joint likelihoods without
double counting. The legacy dependence-weight path remains a frozen baseline, posterior
sampling propagates cold-start uncertainty, and a risk policy emits `PASS`, `FAIL`, or
`DEFER`. A simulator, baselines, metrics, experiment runner, and HaluEval adapter sit
around the core without introducing provider or orchestration abstractions.

**Tech stack:** Python 3.11+, NumPy, SciPy, standard-library `argparse`, pytest, Ruff,
mypy, Hatchling, GitHub Actions, Markdown, JSON/JSONL, and a Kaggle-compatible Jupyter
notebook.

**Project constraints:** The implementation is clean-room and must never contain legacy
project or employer-specific names, code, schemas, prompts, data, or results. No paid API
call is authorized. HaluEval data remains downloaded, checksummed, and uncommitted.

---

## Task 1: Package scaffold and immutable domain contract

**Files:**

- Create: `pyproject.toml`
- Create: `src/corum/__init__.py`
- Create: `src/corum/models.py`
- Create: `src/corum/py.typed`
- Create: `tests/test_models.py`

**Required public interface:**

```python
class Truth(str, Enum): PASS = "PASS"; FAIL = "FAIL"
class Observation(str, Enum): PASS = "PASS"; FAIL = "FAIL"; ABSTAIN = "ABSTAIN"
class ExecutionState(str, Enum):
    VALID = "VALID"; TIMEOUT = "TIMEOUT"; INVALID = "INVALID"
    REFUSAL = "REFUSAL"; NOT_CALLED = "NOT_CALLED"
class Action(str, Enum): PASS = "PASS"; FAIL = "FAIL"; DEFER = "DEFER"
class GateState(str, Enum): PASS = "PASS"; FAIL = "FAIL"; UNRESOLVED = "UNRESOLVED"

@dataclass(frozen=True, slots=True)
class Reviewer:
    reviewer_id: str
    vendor: str
    family: str
    lineage: str
    cost: float = 1.0

@dataclass(frozen=True, slots=True)
class Review:
    case_id: str
    reviewer_id: str
    observation: Observation | None
    state: ExecutionState
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass(frozen=True, slots=True)
class CalibrationExample:
    truth: Truth
    review: Review

@dataclass(frozen=True, slots=True)
class HardGate:
    gate_id: str
    state: GateState
    trusted_deterministic: bool = True

@dataclass(frozen=True, slots=True)
class FusedPosterior:
    pass_probability: float
    lower: float
    upper: float
    valid_reviewers: int
    lineage_count: int
    effective_sample_size: float
    samples: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reasons: tuple[str, ...]
    posterior: FusedPosterior | None
```

**Step 1: Write failing contract tests**

Test that:

- a `VALID` review requires an observation;
- a non-`VALID` review requires `observation is None`;
- negative cost or token counts fail with actionable `ValueError` messages;
- blank stable identifiers fail;
- frozen records reject mutation;
- `FusedPosterior` rejects non-finite/out-of-range or unordered interval values;
- package version is exposed as `corum.__version__`.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_models.py -q`

Expected: collection/import failure because the package contract does not exist.

**Step 3: Implement the smallest validated dataclasses and project metadata**

Use only standard-library dataclasses and enums. Keep validation in `__post_init__` and
do not add serialization frameworks. Configure Hatchling for a `src/` layout and define
dependency groups for tests and quality tools.

**Step 4: Verify GREEN and quality**

Run:

```bash
uv run pytest tests/test_models.py -q
uv run ruff check src/corum/models.py tests/test_models.py
uv run mypy src/corum/models.py
```

Expected: all commands exit zero.

**Step 5: Commit**

```bash
git add pyproject.toml src/corum tests/test_models.py
git commit -m "feat: define consensus domain contract"
```

---

## Task 2: Dirichlet reviewer calibration with uncertainty propagation

**Files:**

- Modify: `pyproject.toml`
- Create: `src/corum/calibration.py`
- Create: `tests/test_calibration.py`
- Modify: `src/corum/__init__.py`

**Required public interface:**

```python
OBSERVATION_ORDER: tuple[Observation, ...]

@dataclass(frozen=True, slots=True)
class ReviewerCalibration:
    reviewer_id: str
    alpha: np.ndarray  # shape (2 truth classes, 3 observations)
    observed_counts: np.ndarray
    prior_strength: float

    def mean_likelihoods(self) -> np.ndarray: ...
    def sample_likelihoods(self, draws: int, rng: np.random.Generator) -> np.ndarray: ...

def fit_reviewer_calibration(
    reviewer_id: str,
    examples: Sequence[CalibrationExample],
    *,
    parent_prior: np.ndarray | None = None,
    prior_strength: float = 1.5,
) -> ReviewerCalibration: ...

def fit_panel_calibrations(
    reviewers: Sequence[Reviewer],
    examples: Sequence[CalibrationExample],
    *,
    prior_strength: float = 1.5,
) -> dict[str, ReviewerCalibration]: ...
```

Truth row order is `(PASS, FAIL)` and observation column order is
`(PASS, FAIL, ABSTAIN)`. Non-valid executions are audited but never counted as semantic
observations. The pooled parent prior is estimated only from the supplied calibration
split, smoothed symmetrically, normalized row-wise, and scaled to exactly the declared
pseudo-count strength.

**Step 1: Write failing behavioral tests**

Cover hand-derived counts, row normalization, `ABSTAIN` as an explicit third outcome,
non-valid exclusion, unknown reviewer rejection, no-data shrinkage to the parent prior,
deterministic seeded samples, shape validation, positive alphas, and a cold-start interval
that is wider than a large-sample interval for the same empirical rate.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_calibration.py -q`

Expected: import failure for `corum.calibration`.

**Step 3: Implement count fitting and posterior sampling**

Use `numpy.random.Generator.dirichlet` once per truth row and preserve the full draw axis.
Return defensive read-only arrays from frozen records. Reject empty reviewer IDs,
non-positive strengths, malformed priors, NaNs, and rows with zero total mass.
Add `numpy>=1.26` as the first production dependency because this task is its first
runtime use.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_calibration.py -q
uv run ruff check src/corum/calibration.py tests/test_calibration.py
uv run mypy src/corum/calibration.py
```

**Step 5: Commit**

```bash
git add pyproject.toml src/corum/calibration.py src/corum/__init__.py tests/test_calibration.py
git commit -m "feat: calibrate reviewer likelihoods"
```

---

## Task 3: Dependence estimation and duplicate-evidence control

**Files:**

- Modify: `pyproject.toml`
- Create: `src/corum/dependence.py`
- Create: `tests/test_dependence.py`

**Required public interface:**

```python
@dataclass(frozen=True, slots=True)
class DependenceModel:
    reviewer_ids: tuple[str, ...]
    correlation: np.ndarray
    lineage_by_reviewer: Mapping[str, str]

    def weights_for(self, reviewer_ids: Sequence[str]) -> Mapping[str, float]: ...
    def effective_sample_size(self, reviewer_ids: Sequence[str]) -> float: ...

def fit_dependence(
    reviewers: Sequence[Reviewer],
    examples: Sequence[CalibrationExample],
    *,
    shrinkage: float = 0.25,
    min_overlap: int = 10,
    lineage_cap: float = 1.0,
) -> DependenceModel: ...
```

For each paired valid review, encode semantic error as `observation != truth`; an
`ABSTAIN` is an error for this correlation diagnostic because it did not recover truth.
Estimate Pearson/phi error correlation on overlapping case IDs, shrink off-diagonal
values toward zero, clip negative correlations to zero for weighting, project the matrix
to a finite symmetric positive-semidefinite correlation matrix, and apply
`w_i(S) = 1 / (1 + sum_{j in S, j != i} max(rho_ij, 0))` for the exact reviewer subset
being fused. When overlap is insufficient, reviewers sharing
a lineage use a conservative cap; unrelated reviewers default to zero correlation.

**Step 1: Write failing tests**

Test independent reviewers (`w=1`, `ESS=n`), two exact clones (total weight and ESS near
one), a four-clone lineage, negative correlation not increasing information weight,
sparse-overlap lineage fallback, permutation invariance, PSD/symmetry/diagonal-one
invariants, unknown ID errors, finite ESS bounds `[1, n]` for non-empty subsets, and a
single queried reviewer receiving weight one even when unqueried clones exist. Assert
lineage metadata is immutable, complete, and preserved under reviewer permutation.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_dependence.py -q`

**Step 3: Implement with explicit numerical safeguards**

Use eigenvalue clipping followed by diagonal renormalization for the PSD projection. Do
not multiply weights by accuracy: calibration likelihoods already contain reliability.
Add `scipy>=1.12` as a production dependency only if the implementation actually uses a
SciPy numerical primitive; otherwise keep the NumPy implementation and omit SciPy until
the simulator first requires it.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_dependence.py -q
uv run ruff check src/corum/dependence.py tests/test_dependence.py
uv run mypy src/corum/dependence.py
```

**Step 5: Commit**

```bash
git add pyproject.toml src/corum/dependence.py tests/test_dependence.py
git commit -m "feat: discount correlated reviewer evidence"
```

---

## Task 4: Posterior fusion, hard gates, and risk-aware action policy

**Files:**

- Create: `src/corum/fusion.py`
- Create: `src/corum/decision.py`
- Create: `scripts/benchmark_fusion.py`
- Create: `tests/test_fusion.py`
- Create: `tests/test_decision.py`

**Required public interface:**

```python
@dataclass(frozen=True, slots=True)
class FusionContext:
    likelihood_draws: Mapping[str, np.ndarray]  # each shape (draws, 2, 3)
    dependence: DependenceModel
    lineage_by_reviewer: Mapping[str, str]
    prior_pass: float
    credible_mass: float

def build_fusion_context(
    calibrations: Mapping[str, ReviewerCalibration],
    dependence: DependenceModel,
    *,
    prior_pass: float,
    draws: int = 512,
    credible_mass: float = 0.95,
    seed: int,
) -> FusionContext: ...

def fuse_known_likelihoods(
    observations: Mapping[str, Observation],
    likelihoods: Mapping[str, np.ndarray],  # each shape (2, 3)
    weights: Mapping[str, float],
    *,
    prior_pass: float,
) -> float: ...

def fuse_reviews(
    reviews: Sequence[Review],
    context: FusionContext,
) -> FusedPosterior | None: ...

@dataclass(frozen=True, slots=True)
class BatchFusedPosterior:
    pass_probability: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    valid_reviewers: np.ndarray
    lineage_count: np.ndarray
    effective_sample_size: np.ndarray

def fuse_review_matrix(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    reviewer_ids: Sequence[str],
    context: FusionContext,
    *,
    chunk_size: int = 4_096,
) -> BatchFusedPosterior: ...

@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    pass_threshold: float = 0.90
    fail_threshold: float = 0.10
    min_valid_reviewers: int = 2
    min_lineages: int = 2
    min_effective_sample_size: float = 1.5

def decide(
    posterior: FusedPosterior | None,
    gates: Sequence[HardGate],
    policy: DecisionPolicy,
) -> Decision: ...
```

Fusion validates one case per call, rejects duplicate reviewer IDs, and accumulates
weighted class log-likelihoods from parameter draws sampled once in `FusionContext` and
reused across cases. It recomputes dependence weights for the valid reviewer subset.
Missing executions never contribute likelihood. An empty valid panel returns `None`.
The exact known-likelihood function is deterministic and exists both as an oracle and as
an analytic implementation check; it is not allowed to consume test truth.
The returned interval propagates Dirichlet likelihood uncertainty conditional on a
point-estimated dependence matrix. Names, documentation, and reports must not call it a
full correlated-output credible interval or formal risk guarantee. Nominal coverage is
tested only in the independent correctly specified scenario against the oracle conditional
probability; correlated scenarios use empirical held-out risk and calibration diagnostics.
The matrix path returns mean/lower/upper arrays, shares the same tested kernel as scalar
fusion, and chunks cases so memory is `O(draws * reviewers * chunk_size)` rather than
`O(draws * all_cases)`. It is the required path for the locked benchmark.
`observations` has shape `(cases, reviewers)` with integer codes in the declared
observation order; `valid_mask` has the same shape and is the only authority on whether a
cell contributes. Codes under a false mask are ignored and normally use `-1`; a true mask
with `-1` is rejected. `BatchFusedPosterior` contains per-case mean, lower, upper, valid
reviewer count, lineage count, and ESS arrays. All-invalid rows carry NaN probability
fields and zero quorum diagnostics so the decision layer emits `DEFER`.

Decision precedence is fixed:

1. any trusted deterministic gate `FAIL` returns `FAIL` with `hard_gate_failed`;
2. any trusted gate `UNRESOLVED` prevents `PASS` and ordinarily returns `DEFER`;
3. missing posterior or failed reviewer/lineage/ESS quorum returns `DEFER`;
4. lower conditional posterior bound at or above pass threshold returns `PASS`;
5. upper conditional posterior bound at or below fail threshold returns `FAIL`;
6. all other cases return `DEFER`.

**Step 1: Write failing fusion tests**

Use `fuse_known_likelihoods` to compare against a hand-calculated Bayes result to
`1e-12`, then test the Monte Carlo context against an error bound derived from its sample
standard error rather than `1e-6`. Cover log-space stability, permutation invariance,
seeded reproducibility, common parameter draws across cases, correlated clone confidence
reduction, one-reviewer subset weight one, invalid-state exclusion, duplicate IDs, mixed
case IDs, missing calibration, all-invalid panels, and byte-for-byte agreement between
scalar and batched outputs for a fixed context. The scalar–matrix comparison includes
mixed timeout, invalid, refusal, and not-called executions and asserts per-case valid
count, lineage count, and ESS. A multi-lineage fixture proves fusion supplies enough
metadata for the decision layer to enforce lineage quorum.

**Step 2: Verify fusion RED, implement, then verify GREEN**

Run `uv run pytest tests/test_fusion.py -q` before and after implementation.

**Step 3: Write failing decision tests**

Cover every precedence branch, exact threshold boundaries, hard-gate override, unresolved
gate blocking an otherwise confident pass, quorum reasons, and all-abstain/all-invalid
defer behavior.

**Step 4: Verify decision RED, implement, then run combined quality checks**

```bash
uv run pytest tests/test_fusion.py tests/test_decision.py -q
uv run ruff check src/corum/fusion.py src/corum/decision.py tests/test_fusion.py tests/test_decision.py
uv run mypy src/corum/fusion.py src/corum/decision.py
uv run python scripts/benchmark_fusion.py --cases 10000 --reviewers 3 --draws 512 --max-seconds 5
```

The throughput command must process the registered shape in at most five seconds on the
development CPU and report peak working-array bytes below 512 MiB. If it fails, optimize
the shared vectorized kernel before later experiment work; do not lower cases, draws, or
the time threshold in the locked command.

**Step 5: Commit**

```bash
git add src/corum/fusion.py src/corum/decision.py scripts/benchmark_fusion.py tests/test_fusion.py tests/test_decision.py
git commit -m "feat: fuse evidence into risk-aware decisions"
```

---

## Task 5: Reproducible correlated-panel simulator

**Files:**

- Modify: `pyproject.toml` (add the declared SciPy runtime dependency only)
- Create: `src/corum/simulation.py`
- Create: `tests/test_simulation.py`

**Required public interface:**

```python
@dataclass(frozen=True, slots=True)
class ReviewerSpec:
    reviewer: Reviewer
    likelihoods: np.ndarray  # (2, 3)
    timeout_rate: float = 0.0
    invalid_rate: float = 0.0

@dataclass(frozen=True, slots=True)
class ScenarioPhase:
    reviewers: tuple[ReviewerSpec, ...]
    prior_pass: float
    lineage_error_correlation: Mapping[str, float]
    difficulty_rate: float = 0.0
    informative_missingness: float = 0.0
    adversarial_reviewer_id: str | None = None

@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    calibration: ScenarioPhase
    test: ScenarioPhase

@dataclass(frozen=True, slots=True)
class LineageCorrelationDiagnostic:
    reviewer_ids: tuple[str, ...]
    target_error_correlation: float
    solved_latent_correlation: float
    minimum_eigenvalue: float
    realized_error_correlation: float
    overlap_count: int

@dataclass(frozen=True, slots=True)
class SimulatedPanel:
    seed: int
    truths: Mapping[str, Truth]
    difficulty_by_case: Mapping[str, bool]
    reviews: tuple[Review, ...]
    gates: Mapping[str, tuple[HardGate, ...]]
    lineage_diagnostics: Mapping[str, LineageCorrelationDiagnostic]

def simulate_panel(phase: ScenarioPhase, n_cases: int, *, seed: int) -> SimulatedPanel: ...
def simulate_experiment(
    scenario: Scenario,
    *,
    n_calibration: int,
    n_test: int,
    seed: int,
) -> tuple[SimulatedPanel, SimulatedPanel]: ...
def builtin_scenarios() -> Mapping[str, Scenario]: ...
```

Generate correlated categorical observations with a Gaussian copula and fixed reviewer
marginals. Built-ins are `independent`, `clone_pair`, `majority_trap`,
`informative_missingness`, `drift`, and `cascade_cost`. Every generated review retains
an execution state; missingness must be capable of depending on truth/difficulty.
`SimulatedPanel` records its exact seed and case difficulty labels. Task 5 emits an
explicit empty gate tuple per case; hard-gate canary generation is not configurable in
this interface and remains a later experiment responsibility.

Semantic generation first samples the correlated binary error event, where `ABSTAIN`
counts as an error, then samples the configured wrong-vs-abstain category conditional on
error. This preserves each `(truth, observation)` marginal while targeting the quantity
used by dependence diagnostics. Timeout and invalid rates are unconditional base rates,
must sum to at most one, and are increased on difficult/FAIL cases by the declared
`informative_missingness` strength without ever attaching an observation to a non-valid
review. The adversarial reviewer ID is a validated declaration; its changed behavior is
encoded explicitly in that phase's likelihood matrix rather than silently inverted by
the simulator.

Correlation groups use the immutable reviewer `lineage`; `family` remains descriptive
metadata and is not a second grouping key. Each configured value is a target mean
*observed binary error correlation*, not a latent-normal correlation. For each lineage,
the simulator deterministically solves one valid equicorrelated Gaussian-copula parameter
whose expected pairwise observed error correlation matches the target, accounting for
class prevalence and reviewer marginals. It rejects infeasible targets and records both
the solved latent parameter and realized error correlation. Task 5 accepts non-negative
targets only; omitted lineages are independent, and correlation keys must name a lineage
containing at least two reviewers. Realized correlation uses overlapping `VALID` reviews,
with `ABSTAIN` counted as error, and records the overlap count. A zero-variance or
insufficient-overlap realized value is rejected for a registered diagnostic rather than
reported as fabricated evidence. `n_cases` may be zero, but registered correlation
diagnostics require enough cases to estimate a finite realized value.

For a lineage with heterogeneous reviewers, both the configured target and diagnostic are
the unweighted arithmetic mean over every unordered reviewer pair. The solver matches the
mean of the pair-specific expected correlations; the realized value averages separately
computed overlapping-`VALID` pair correlations and never pools vectors or selects a
favorable pair. A target below the truth-mixture correlation at latent zero, above the
reachable maximum, or involving a zero-variance error process is infeasible. The 100,000
case `0.03` target diagnostic uses no informative missingness; missingness selection bias
is tested separately rather than hidden inside the solver tolerance.

**Step 1: Write failing simulation tests**

Test seed reproducibility, different-seed divergence, approximate truth prevalence and
likelihood marginals, clone error correlation, informative missingness, valid state/
observation invariants, unique `(case_id, reviewer_id)` keys, scenario validation, and
the exact set of built-in scenarios. For `drift`, assert calibration and test differ in
prevalence, at least one reviewer's likelihoods, a lineage correlation, missingness, and
the declared adversarial reviewer behavior.
With 100,000 diagnostic cases, every registered realized lineage error correlation must
be within `0.03` of its configured target and the solved copula block must be PSD.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_simulation.py -q`

**Step 3: Implement the generator**

Use SciPy normal CDF only for copula uniforms and NumPy for all other sampling. Reject
non-PSD requested correlation structures before simulation. Keep text generation and
model calls entirely out of this module.

**Step 4: Verify GREEN and keep runtime under five seconds**

```bash
uv run pytest tests/test_simulation.py -q
uv run ruff check src/corum/simulation.py tests/test_simulation.py
uv run mypy src/corum/simulation.py
```

**Step 5: Commit**

```bash
git add src/corum/simulation.py tests/test_simulation.py
git commit -m "feat: simulate correlated reviewer panels"
```

---

## Task 6: Baselines, metrics, and paired uncertainty estimates

**Files:**

- Create: `src/corum/baselines.py`
- Create: `src/corum/metrics.py`
- Create: `tests/test_baselines.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_core_value.py`

**Required public interface:**

```python
@dataclass(frozen=True, slots=True)
class DecisionCosts:
    false_pass: float = 1.0
    false_fail: float = 0.2
    defer: float = 0.1

def majority_decision(reviews: Sequence[Review]) -> Action: ...
def best_single_reviewer(
    policy_rows: Sequence[CalibrationExample],
    calibrations: Mapping[str, ReviewerCalibration],
    *,
    prior_pass: float,
    pass_threshold: float,
    fail_threshold: float,
    costs: DecisionCosts,
) -> str: ...
def linear_pool_probability(
    reviews: Sequence[Review],
    calibrations: Mapping[str, ReviewerCalibration],
    *,
    prior_pass: float,
) -> float | None: ...

def evaluate_decisions(
    truths: Mapping[str, Truth],
    decisions: Mapping[str, Decision | Action],
    *,
    probabilities: Mapping[str, float] | None = None,
    costs: DecisionCosts = DecisionCosts(),
    sample_weights: Mapping[str, float] | None = None,
) -> dict[str, float]: ...

def target_prevalence_weights(
    truths: Mapping[str, Truth], *, target_fail_prevalence: float
) -> dict[str, float]: ...

@dataclass(frozen=True, slots=True)
class PolicySelection:
    policy: DecisionPolicy
    constraint_satisfied: bool
    decision_loss: float
    coverage: float

def policy_candidates() -> tuple[DecisionPolicy, ...]: ...

def select_decision_policy(
    truths: Mapping[str, Truth],
    posteriors: Mapping[str, FusedPosterior | None],
    gates: Mapping[str, Sequence[HardGate]],
    *,
    costs: DecisionCosts,
    min_coverage: float = 0.50,
) -> PolicySelection: ...

def stratified_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    metric: Callable[[Sequence[Mapping[str, object]]], float],
    *,
    strata: Sequence[str],
    draws: int = 2_000,
    seed: int,
) -> tuple[float, float, float]: ...
```

Metrics include coverage, defer rate, false-PASS, false-FAIL, selective risk, decision
loss, Brier, log loss, ECE, and mean reviewer/token cost. Undefined conditional metrics
return `math.nan` and are clearly serialized as `null`; they never silently become zero.
`linear_pool_probability` first converts every valid observation into a reviewer-specific
Bayes posterior using the declared class prior and that reviewer's posterior-mean
likelihoods, then averages those probabilities. Best-single selection uses only the
held-out policy partition, minimizes the published decision loss under the same prior and
thresholds, and breaks exact ties lexicographically. It never fits or selects on test data.

`policy_candidates()` is a fixed Cartesian set: pass thresholds `{0.80, 0.90, 0.95}`,
fail thresholds `{0.05, 0.10, 0.20}`, fixed reviewer/lineage quorum of two, and minimum
ESS `{1.0, 1.5}`. Selection considers only policy-partition IDs. Among candidates meeting
50% coverage, it minimizes decision loss and tie-breaks by lower false-PASS rate, higher
coverage, then the canonical policy tuple. If none meet coverage, it selects highest
coverage then lowest loss and returns `constraint_satisfied=False`; the run is not allowed
to label that outcome a feasibility pass. `selected_policy.json` records every candidate's
metrics and the exact winner.

**Step 1: Write failing baseline tests**

Cover majority tie/all-abstain defer, non-valid exclusion, calibration-only best-single
selection, deterministic tie breaking by reviewer ID, no access to test rows, and a
hand-calculated prior-aware linear pool.

**Step 2: Verify baseline RED and implement baseline functions**

Run `uv run pytest tests/test_baselines.py -q` before and after production code.

**Step 3: Write failing metric tests**

Use a four-case hand-worked table to assert every confusion/risk/loss result, then test
probability scoring, edge cases, deterministic bootstrap, paired resampling, stratum
preservation, target-prevalence post-stratification, and rejection of mismatched case IDs.
For a balanced sample and target `P(FAIL)=0.20`, assert normalized class weights are
proportional to `0.4` for FAIL and `1.6` for PASS, and report weighted and unweighted
metric namespaces separately.
Test policy selection on a hand-worked candidate table, all tie-break levels, no-feasible
fallback, and an access-guard mapping that contains only the policy partition. Experiment
integration later rejects any overlap among fit, policy, and test case IDs.

**Step 4: Verify metric RED, implement, and run quality checks**

```bash
uv run pytest tests/test_baselines.py tests/test_metrics.py -q
uv run ruff check src/corum/baselines.py src/corum/metrics.py tests/test_baselines.py tests/test_metrics.py
uv run mypy src/corum/baselines.py src/corum/metrics.py
```

**Step 5: Commit**

```bash
git add src/corum/baselines.py src/corum/metrics.py tests/test_baselines.py tests/test_metrics.py
git commit -m "feat: add consensus baselines and metrics"
```

**Step 6: Run the locked Core Value Gate before Task 7**

`tests/test_core_value.py` is an independent vertical judge. It uses the same reviews for
Corum, ordinary unweighted majority, and naive independent fusion; it may not tune on test
rows or rewrite the pre-registered seeds, losses, scenarios, policies, or thresholds.
Run:

```bash
uv run pytest tests/test_core_value.py -q
```

The exact machine criteria are Section 9.1 of the canonical design. Failure blocks Task 7
and all product-surface expansion. Allow at most three bounded core-repair cycles without
changing the judge; after that, record `CORE_VALUE_GATE_FAILED` for owner judgment.

---

## Task 6B: Fixed pair-block joint-likelihood pivot

**Entry condition:** Task 6A is permanently recorded as `CORE_VALUE_GATE_FAILED`, and the
owner has approved the prospective pivot in
`docs/sdd/0007-pair-block-consensus-pivot.md`. The old judge and failed result remain
unchanged.

**Documentation files:**

- Modify: `AGENTS.md`
- Modify: `docs/specs/corum-mvp-design.md`
- Modify: `docs/plans/corum-mvp.md`
- Create: `docs/sdd/0007-pair-block-consensus-pivot.md`

**Test-contract files, committed before implementation:**

- Modify: `tests/test_calibration.py`
- Modify: `tests/test_fusion.py`
- Create: `tests/test_pair_value.py`

**Implementation files:**

- Modify: `src/corum/__init__.py`
- Modify: `src/corum/calibration.py`
- Modify: `src/corum/fusion.py`
- Modify: `scripts/benchmark_fusion.py`

**Required public interface:**

```python
PairKey = tuple[str, str]

@dataclass(frozen=True, slots=True)
class ReviewerPairCalibration:
    reviewer_ids: PairKey
    alpha: np.ndarray             # (2, 3, 3)
    observed_counts: np.ndarray   # (2, 3, 3)
    prior_strength: float
    min_paired_per_truth: int = 30

    def mean_likelihoods(self) -> np.ndarray: ...
    def sample_likelihoods(
        self, draws: int, rng: np.random.Generator
    ) -> np.ndarray: ...          # (draws, 2, 3, 3)

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
    likelihoods: Mapping[str, np.ndarray],
    pair_likelihoods: Mapping[PairKey, np.ndarray],
    *,
    prior_pass: float,
) -> float: ...
```

Append an empty-by-default `pair_likelihood_draws` mapping to `FusionContext`, and append
an optional `pair_calibrations` keyword to `build_fusion_context`. Pair keys are canonical,
known, and globally disjoint. A both-valid pair contributes its joint likelihood once; an
exactly-one-valid pair falls back to the same singleton draw as naive Bayes; remaining
singletons have exponent one. With no pair mapping, the legacy power path is byte-for-byte
unchanged. The standalone known-pair oracle with an empty pair mapping is instead naive
exponent-one singleton fusion. ESS and lineage diagnostics retain their existing reviewer-
level definitions.

**Step 1: Commit and review the prospective documentation**

Commit exactly `docs: register pair-block consensus pivot`. Do not write production code
or run the new value judge before this commit.

**Step 2: Write RED unit tests and the independent locked judge**

Cover pair counts/prior/sparsity, hand-worked joint Bayes, no double counting, missing
fallback, invalid keys, immutable draws, seeded replay, legacy compatibility, and scalar/
matrix equivalence. Add the complete fresh two-gate judge, literal scenarios, constants,
and reference calculations from SDD 0007. Run only the focused unit files to capture RED;
do not execute `tests/test_pair_value.py`. Obtain independent read-only review of the
complete test diff, fix every Critical or Important finding without executing the judge,
then commit exactly `test: lock pair-block value gate`.

**Step 3: Implement the minimum pair-block path and verify GREEN**

Do not change the simulator, baselines, metrics, dependence model, decision policy, old
judge, or any gate literal. Run focused tests, the repository suite excluding both locked
judges, branch coverage at least 80%, Ruff, mypy, and the registered performance probe.
Commit exactly `feat: fuse calibrated reviewer pairs` after independent review reports no
open Critical or Important finding.

**Step 4: Execute the fresh judge once**

Run `uv run pytest tests/test_pair_value.py -q` only after all three commits and reviews.
Gate A admits the pair component; Gate B closes the core against ordinary majority. Both
must pass to unlock Task 7. After the first run, permit at most two bounded repairs only in
pair calibration/fusion production code plus additive or stronger regression tests; never
delete or relax an existing assertion, and never alter the judge in place.

**Frozen attempt 0 result (2026-08-29):** the judge completed all 64 runs on `14c363d`.
Gate A failed and Gate B passed, producing `PAIR_BLOCK_ADMISSION_FAILED`. Pair decision
loss was `22.10%` lower than majority with a strictly positive paired interval, but pair NLL
did not clear the frozen power baseline and breached the independent negative-control
guardrail. Three independent read-only postmortems found no implementation defect that can
be repaired without changing the registered model. No repair cycle was consumed; the
component is unadmitted and Task 7 remains blocked. Preserve the exact artifacts under
`docs/results/` and return the next pivot/stop decision to the owner.

---

## Task 6C: JudgeBench external vote value gate

**Entry condition:** Task 6A and Task 6B remain permanently failed under their original
judges. The owner has approved an external comparison of the unchanged legacy no-pair
core in `docs/sdd/0008-judgebench-external-vote-gate.md`. This task cannot repair either
failed mechanism or unlock Task 7.

**Documentation files:**

- Modify: `AGENTS.md`
- Modify: `docs/specs/corum-mvp-design.md`
- Modify: `docs/plans/corum-mvp.md`
- Create: `docs/sdd/0008-judgebench-external-vote-gate.md`

**Frozen judge files:**

- Create: `configs/judgebench-v1.json`
- Create: `tests/test_judgebench_value.py`

**Attempt-0 result files:**

- Create: `docs/results/task-6c-judgebench-attempt-0.json`
- Create: `docs/results/task-6c-judgebench-attempt-0.md`
- Create: `docs/results/task-6c-judgebench-attempt-0.txt`
- Modify only the status/result prose in the four documentation files above

**Step 1: Freeze and review the prospective documentation**

Record the exact upstream revision, eight file hashes, outcome-blind seven-reviewer panel,
three lineages,
order-reversal normalization, split digest, symmetric costs, unchanged core literals,
two voting baselines, anti-DEFER criteria, source-stratified bootstrap, verdict, and stop
rule. Do not calculate any held-out outcome. Obtain independent read-only review, resolve
every Critical or Important finding, and commit exactly
`docs: register JudgeBench external value gate`.

**Step 2: Lock the independent judge without running it**

Add a checked-in JSON registry and a self-contained external-data judge. The ordinary
suite skips it unless `CORUM_RUN_JUDGEBENCH_V1=1`. Synthetic parser/reference tests may
exercise structural code but must not read upstream held-out values. The formal judge
independently verifies the pinned Git tree and complete 11-candidate inventory, raw hashes,
output alignment, order normalization, exact split, empty pair registry, baseline
decisions, per-case losses, bootstrap inputs, and verdict logic. It must never write or
commit raw or normalized vote rows.

Run static checks only, obtain independent review without executing the formal test, fix
and re-review all findings, then commit exactly
`test: lock JudgeBench external value gate`. Do not modify `src/corum`.

**Step 3: Execute the frozen judge once**

Materialize the eight pinned blobs only under `.corum-work/judgebench-v1/raw/`, set the
explicit run switch and raw-directory environment variable, and run only:

```powershell
$env:CORUM_RUN_JUDGEBENCH_V1 = "1"
$env:CORUM_JUDGEBENCH_RAW_DIR = ".corum-work/judgebench-v1/raw"
$env:CORUM_JUDGEBENCH_UPSTREAM_REPO = ".corum-work/judgebench-v1/upstream"
.venv\Scripts\uv.exe run pytest tests/test_judgebench_value.py -q -s
```

Capture the exact output and wall time. Preserve the aggregate result, not the upstream
rows, under `docs/results/`. `PASS` authorizes only a minimal offline evaluator and a
fresh real developer-project/patch value gate. `FAIL` or `INCONCLUSIVE` stops component
and product expansion. `INVALID` requires a prospectively versioned judge fix before any
scientific claim. No same-data core tuning or threshold change is permitted.

**Step 4: Record and verify honestly**

Update only registered result/status prose, commit result artifacts, and run the ordinary
repository suite, Ruff, mypy, and `git diff --check`. Retain the Task 6A/6B failures and
state that JudgeBench is static answer-comparison evidence rather than project, patch, or
adoption validation. Commit exactly `docs: record JudgeBench external gate result` after
the artifacts and verification evidence are complete.

**Frozen attempt-0 result (2026-08-29):** the judge ran exactly once on
`6d03f4cf18c43decff3ae1bffde277279ff25d31` and returned `FAIL`. Corum's pooled
decision loss was `0.253247`, `11.36%` below both voting baselines at `0.285714`,
but the paired 95% intervals `[-0.025974, 0.092532]` crossed zero. Corum coverage was
only `3.90%` and useful resolution `2.60%`, versus `89.61%` and `63.64%` for
each baseline; 148 of 154 cases were `DEFER`. The policy constraint and the registered
pooled and coding anti-`DEFER`/utility guardrails failed. Preserve the complete aggregate
result and exact pytest capture under `docs/results/`. This core/panel result is final:
do not tune it on JudgeBench, and keep Task 7 plus all component and product expansion
blocked.

---

## Task 6D: Real-patch Daily Use Gate

**Entry condition:** Task 6C remains a final `FAIL`. The owner has approved a distinct,
prospective investment test under `docs/sdd/0009-daily-use-gate.md`, not a repair of any
old result. The accepted-base statistical core, rejected pair component, Task 6C judge,
and Task 7 entry condition remain frozen.

**Documentation files:**

- Modify: `AGENTS.md`
- Modify: `docs/specs/corum-mvp-design.md`
- Modify: `docs/plans/corum-mvp.md`
- Create: `docs/sdd/0009-daily-use-gate.md`

**Frozen judge files:**

- Create: `configs/daily-use-v1.json`
- Create: `tests/test_daily_use_value.py`
- Create after reviewer acquisition: `configs/daily-use-v1-seal.json`

**Attempt-0 result files:**

- Create: `docs/results/task-6d-daily-use-attempt-0.json`
- Create: `docs/results/task-6d-daily-use-attempt-0.md`
- Create: `docs/results/task-6d-daily-use-attempt-0.txt`
- Modify only registered status/result prose in the four documentation files above

**Step 1: Register and review the simple investment gate**

Freeze the 500-task SWE-bench Verified manifest, outcomes-free candidate contract, exact
three-reviewer limit, blind prompt/context boundary, equivalent perturbation, unchanged
no-pair Corum readout, ordinary and reliability-weighted votes, token ledger, metrics,
paired bootstrap, integrity checks, and one-run stop rule. Preserve every earlier failure.
Obtain independent read-only review, fix all Critical and Important findings, and commit
exactly `docs: register daily use value gate`.

**Step 2: Satisfy pre-acquisition prerequisites**

Freeze either a newly generated outcome-isolated 500-patch set or a complete public
candidate inventory with deterministic selection. Freeze the complete eligible endpoint
inventory and mechanical three-distinct-lineage model selection, exact prompt/retrieval
and 1,000 context packages, privacy/provenance clearances, container digests, and a
successful 500-case gold-patch environment check. Until every item exists, keep Task 6D
`BLOCKED`; do not write a placeholder config, acquire the formal 3,000-row reviewer
ledger, call a paid API, reduce the sample, substitute simulation, or start product work.
The already consumed preregistered synthetic transport/schema smoke is the sole exception;
it is closed and authorizes no further call, retry, repair, replacement, or reordering.

Checkpoint 2026-08-29: the one preregistered local panel smoke is consumed and formally
`BLOCKED / result_mismatch`. Its three preregistered digests each produced one HTTP-200,
exact-schema-valid mechanical record with zero retries, but PowerShell parameter binding
coerced the `$null` argument passed to `[AllowNull()][string]$FailureKind` into `""` before
JSON serialization instead of producing JSON `null`; therefore it establishes no eligible
panel or quality evidence. Do not rerun, repair, replace, or reorder this panel from the
smoke. The 500-patch commitment, judge, 3,000-row reviewer ledger, harness oracle, and
formal Task 6D attempt remain unstarted.

**Step 3: Lock the minimum independent judge through TDD**

Write synthetic RED tests first. Add only the JSON registry and self-contained judge.
Ordinary tests must validate schemas, blind joining, cross-repository fitting, both voting
baselines, full-coverage Corum readout, token accounting, perturbation flips, false-safe
incidence, repository-stratified paired bootstrap, verdict precedence, aggregate-only output, and
deterministic replay. The external path stays skipped unless
`CORUM_RUN_DAILY_USE_V1=1`. Do not modify `src/corum` or acquire real reviewer votes.
Verify at least 80% branch coverage for new helpers, the ordinary suite excluding only the
already-consumed failing Task 6A/6B judges, Ruff, mypy, and `git diff --check`; obtain fresh
independent review and commit exactly
`test: lock daily use value gate` without executing the formal gate.

**Step 4: Acquire and seal reviews before the oracle exists**

Acquire exactly 3,000 final reviewer records under the locked panel and packages. A
recorded timeout/refusal/invalid schema is a model outcome; a missing ledger row or hidden
call is an integrity failure. Count every bounded retry token. Reviewers must never receive
oracle material. Raw material and secrets remain under ignored local storage. Independently
review the acquisition audit, then commit only its hashes and token total in
`configs/daily-use-v1-seal.json` exactly as
`data: seal daily use reviewer ledger`. Do not generate or open candidate harness outcomes
until the seal commit and its review are complete.

**Step 5: Generate the oracle once, execute once, and obey the result**

Run every candidate once in the pinned harness, then run the locked external judge once.
Preserve exact aggregate output and wall time, and
commit only the registered result files plus status prose as
`docs: record daily use gate result`. `PASS` requires at least `+5pp` accuracy against
each vote with paired confidence above zero, shared token ratio exactly `1.0` (and thus at
most `1.20`), at least `30%`
fewer perturbation flips with paired confidence above zero, coverage at least `90%`, and
no worse false-safe incidence. A pass permits only an owner-reviewed plan for a minimal
human-input/BYO-LLM product pilot; it cannot admit a component or unlock the old Task 7.
`FAIL` or `INCONCLUSIVE` stops the current consensus path. A post-oracle `INVALID`
consumes the attempt and cannot rerun this candidate/panel/ledger/holdout combination.

---

## Task 6E: Prospective full-coverage convergence/resolution gate

**Entry condition:** Task 6D remains formally `BLOCKED`; its one panel smoke is consumed
and its formal attempt 0 is unconsumed. The owner approved this distinct prospective
synthetic qualification under `docs/sdd/0010-convergence-resolution-gate.md` before any
replacement acquisition SDD may be prepared. It does not reopen Tasks 6A--6D, change the
accepted-base core, admit a component, or satisfy Task 7's entry condition.

**Documentation files:**

- Modify: `AGENTS.md`
- Modify: `docs/specs/corum-mvp-design.md`
- Modify: `docs/plans/corum-mvp.md`
- Create: `docs/sdd/0010-convergence-resolution-gate.md`

**Frozen judge files:**

- Created in the reviewed judge milestone: `configs/convergence-resolution-v1.json`
- Created in the reviewed judge milestone: `tests/test_convergence_resolution_value.py`

**Attempt-0 result files:**

- Judge exclusively creates/appends during the one-shot run:
  `docs/results/task-6e-convergence-resolution-attempt-0.txt`
- Result recorder exclusively owns, creating or exact-byte-reusing after judge termination:
  `docs/results/task-6e-convergence-resolution-attempt-0.json`
- Result recorder also owns, creating or exact-byte-reusing after judge termination:
  `docs/results/task-6e-convergence-resolution-attempt-0.md`
- Modify only registered status/result prose in the four documentation files above

**Step 1: Register and review the synthetic qualification**

Freeze exactly the already registered Task 6D full-coverage readout over the accepted-base
legacy no-pair power fusion, six literal scenarios, 40 blocks per scenario, 8,000
independent fit cases and 10,000 holdout cases per block, independent enumerated seeds,
one accepted-base `simulate_experiment` call per block with pinned child-phase association,
the synthetic conditional action-dispersion/resolution stress operand, both voting
baselines, all-case metrics, whole-block paired bootstrap, canonical config/result bytes,
durable attempt lifecycle, anti-`DEFER` rules, verdict precedence, and one-shot stop rule.
Preserve every prior failure and Task 6D's `BLOCKED` state. Obtain independent
statistics, governance, and implementation review and commit exactly
`docs: register convergence resolution gate`. Do not create a config, judge, result
placeholder, or production file in this milestone.

**Step 2: Lock the independent synthetic judge through TDD**

In a separate reviewed milestone, add only the frozen JSON config and self-contained
judge. Synthetic unit fixtures must RED then GREEN schema/literal validation, seed
regeneration, literal NUL seed separators, tiny non-formal fixture phase splitting and
hashes, fit/test isolation, existing singleton calibration/dependence/fusion, all-invalid
`p=0.5` and mixed-invalid diagnostics, both baseline hand calculations, no test-truth
access in weights, row-order invariance, exact three-row rotation, multiset/truth
preservation, `DEFER`-as-wrong and probability metrics, whole-block bootstrap, mocked
fsynced-START then counted sequential 240-block orchestration, sufficient-statistic operand
hashes and pooled ECE, durable TXT-only judge lifecycle, recorder crash classes, every
closed reason/verdict class, exact PASS/FAIL/INCONCLUSIVE integrity/output fixtures,
deterministic replay, resumable recorder fault injection after JSON, Markdown, each status
update, and around commit, exact-byte reuse and mismatch refusal, canonical output, and
default external skip. Formal preflight is static and must never simulate a formal block
or rerun fixtures.
Pin the literal phase objects and
their canonical-JSON SHA-256 rather than deriving them from `builtin_scenarios()`;
enumerate all simulation, fusion, and perturbation seeds and bind the complete seed-table
digest. Keep `src/corum` frozen and commit exactly
`test: lock convergence resolution gate` without running the formal gate.

**Step 3: Execute once and obey the result**

Complete retryable preflight before creating any result file. The judge then exclusively
creates and fsyncs the registered TXT ledger's binding `START` record before the first
formal block. Existing TXT state refuses execution; post-`START` crash, partial work, or
integrity failure consumes the attempt as `INVALID`, and a `START`-only ledger is preserved.
Run all 240 blocks sequentially with no block drop, replacement, top-up, or rerun. Every
one of the 2,400,000 A-form holdout cases is scored; `DEFER` is incorrect and uncovered.
The judge embeds canonical deterministic JSON in TXT FINAL, appends/fsyncs exactly one
FINAL on normal or caught completion, and prints one final aggregate status line with no
partial metrics. It never creates registered JSON. After termination the recorder creates
JSON/MD/status from valid FINAL or a closed administrative crash result from preserved
START-only/partial/malformed TXT, without appending or rerunning. Publication is
resumable/idempotent: exact existing JSON/MD/status bytes are reused, missing stages are
completed, mismatches stop as preserved forensic conflicts, and a retry after any stage
converges to the same registered bytes and one result commit without touching TXT or
consuming a new attempt. Normal PASS/FAIL/INCONCLUSIVE results use integrity status
`PASS`, empty integrity reasons, deterministic replay `true`, and complete non-null
registered counts/maps/hashes.

Formal `PASS` requires, against both voting baselines, at least `+5pp` pooled accuracy with
positive paired confidence, candidate coverage at least `98%` and within one point of
each baseline, at least 30% lower synthetic conditional action-dispersion operand with
positive paired confidence, bounded false-safe incidence, every scenario guardrail, exact
shared A/B reviewer-row counts, zero model calls, all 240 blocks and 2.4M cases, and
deterministic integrity. A zero comparator dispersion-operand rate or
unmet confidence condition after all point checks yields `INCONCLUSIVE`; point or
guardrail failure yields `FAIL`; integrity failure yields `INVALID`.

A synthetic `PASS` authorizes only preparation and independent review of a new acquisition
SDD/version that fixes the serialization contract and re-establishes panel eligibility
before the still-frozen 500-case Daily Use Gate. It is not a Daily Use pass and does not
consume Task 6D attempt 0. `FAIL`, `INCONCLUSIVE`, or post-start `INVALID` ends the current
consensus path: do not add another synthetic candidate, shrinkage, pair averaging, model
averaging, threshold, model call, product surface, or Task 7 work; report root-cause slices
and return the continue/stop decision to the owner.

Task 6E is only a cheap synthetic kill/prequalification test against voting, not
state-of-practice or novelty proof. If and only if it passes, the replacement real-data
acquisition SDD must prospectively add human-labeled repository/code-review truth,
cross-vendor reviewer strata, and at least the strongest single judge, Dawid--Skene-class
EM aggregation, and registered conformal risk-control/cascade comparators. It must not use
synthetic dispersion as real stability evidence. Node-level provenance and active
experiment design remain future hypotheses, not authorized components.

After the consumed run, the result recorder derives registered JSON, Markdown, and status
prose from TXT without recomputing or reinterpreting a valid FINAL, then commits
only those registered artifacts and status prose exactly as
`docs: record convergence resolution gate result`. TDD injects publication faults after
JSON, Markdown, every status-document update, and around commit, and proves exact-byte
reuse, mismatch refusal, and convergence to one result commit.

**Recorded attempt-0 outcome:** Task 6E attempt 0 is final: `FAIL` with reason codes `FAIL_ACCURACY_POINT_WEIGHTED`, `FAIL_COVERAGE_FLOOR`, `FAIL_COVERAGE_GAP_WEIGHTED`, `FAIL_DISPERSION_POINT_ORDINARY`, `FAIL_DISPERSION_POINT_WEIGHTED`, `FAIL_SCENARIO_ACCURACY`, `FAIL_SCENARIO_COVERAGE`, `FAIL_SCENARIO_FALSE_SAFE`. Artifacts: `docs/results/task-6e-convergence-resolution-attempt-0.txt`, `docs/results/task-6e-convergence-resolution-attempt-0.json`, and `docs/results/task-6e-convergence-resolution-attempt-0.md`. The current consensus path is stopped; another synthetic candidate, Task 7, product work, and model calls remain unauthorized pending an owner decision.

---

## Task 7: Leakage-free adaptive cascade

**Entry condition:** both locked Task 6B gates pass. The favorable portions of the failed
Task 6A run, Task 6B Gate B, Task 6C, and a green unit suite are insufficient. Task 6C
and Task 6D cannot satisfy this entry condition; the cascade remains dormant unless a
future owner-approved prospective roadmap replaces it.

**Files:**

- Create: `src/corum/cascade.py`
- Create: `tests/test_cascade.py`

**Required public interface:**

```python
@dataclass(frozen=True, slots=True)
class CascadeResult:
    decision: Decision
    queried_reviewers: tuple[str, ...]
    execution_ledger: tuple[Review, ...]
    total_cost: float
    stop_reason: str

def order_reviewers(
    reviewers: Sequence[Reviewer],
    calibrations: Mapping[str, ReviewerCalibration],
    dependence: DependenceModel,
    *,
    prior_pass: float,
) -> tuple[str, ...]: ...

def replay_cascade(
    available_reviews: Mapping[str, Review],
    reviewers: Mapping[str, Reviewer],
    calibrations: Mapping[str, ReviewerCalibration],
    dependence: DependenceModel,
    policy: DecisionPolicy,
    gates: Sequence[HardGate],
    *,
    prior_pass: float,
    initial_reviewers: int = 2,
    max_cost: float | None = None,
    draws: int = 512,
    seed: int,
) -> CascadeResult: ...
```

Ordering uses calibration-only expected information gain, lineage novelty, and declared
cost. The initial set must span lineages when possible. Replay may reveal only the next
selected review; it cannot inspect unqueried observations to choose or stop. Budget
exhaustion returns `DEFER` without relaxing policy. Every registered reviewer appears in
the execution ledger: queried executions retain their original state, while untouched
reviewers are explicit `NOT_CALLED` records with no semantic observation. Fusion
recomputes dependence weights for only the accumulated valid subset at every step.

**Step 1: Write failing tests**

Test cheap informative reviewers rank early, same-lineage clones are separated, an easy
case stops after the initial panel, a hard case reaches the full panel, budget exhaustion,
deterministic hard-gate short circuit, full-budget equivalence to static fusion, token/
cost accounting, subset-conditioned weights, complete `NOT_CALLED` audit records, and a
sentinel mapping that raises if unqueried values are read before selection.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_cascade.py -q`

**Step 3: Implement ordering and replay**

Compute expected information from posterior-mean likelihoods and the explicit
`prior_pass` argument. Tie-break lexicographically by reviewer ID. Build one fusion
context per case replay and reuse its common parameter draws as the queried subset grows;
do not resample calibration parameters after observing whether the prior step deferred.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_cascade.py -q
uv run ruff check src/corum/cascade.py tests/test_cascade.py
uv run mypy src/corum/cascade.py
```

**Step 5: Commit**

```bash
git add src/corum/cascade.py tests/test_cascade.py
git commit -m "feat: add cost-aware adaptive cascade"
```

---

## Task 8: End-to-end experiment runner, CLI, and report renderer

**Files:**

- Create: `src/corum/experiment.py`
- Create: `src/corum/reporting.py`
- Create: `src/corum/cli.py`
- Create: `tests/test_experiment.py`
- Create: `tests/test_cli.py`
- Modify: `pyproject.toml`

**Required commands:**

```text
corum simulate --config CONFIG.json --output RUN_DIR
corum evaluate --config CONFIG.json --reviewers REVIEWERS.json \
  --calibration CALIBRATION.jsonl --truth TRUTH.jsonl --reviews REVIEWS.jsonl \
  --gates GATES.jsonl --output RUN_DIR
corum report --run RUN_DIR --output REPORT.md
```

**Required artifacts per run:**

```text
config.json
manifest.json
selected_policy.json
case_results.jsonl
execution_ledger.jsonl
metrics.json
report.md
```

The manifest records package version, git commit if available, Python/NumPy/SciPy
versions, UTC start/end, configuration SHA-256, explicit seeds, sample counts, and
completion state. Writes use a temporary sibling file plus `os.replace`; a failed run
must not leave `completion_state: complete`.
All external JSON/JSONL objects carry `schema_version: "1"`. The evaluation config pins
fit/policy case IDs, class prior, posterior draws, policy thresholds, quorum, loss matrix,
target prevalence, and seeds. Reviewer metadata supplies stable ID, vendor, family,
lineage, and cost; the gates stream may be empty but must be present so absence of hard
gates is explicit rather than inferred.

**Step 1: Write failing integration tests**

Run a tiny independent scenario through calibration, dependence fitting, naive Bayes,
full Corum, and cascade. Assert deterministic byte-identical metrics for a fixed
seed, schema-stable artifacts, honest gate outcome, complete manifest, and no network
access. Test malformed JSONL, duplicate keys, missing truth, output-directory collision,
split overlap, unknown reviewers, missing metadata, inconsistent schema versions, and
interrupted-write behavior.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_experiment.py tests/test_cli.py -q`

**Step 3: Implement the runner and renderer**

Use `argparse`; do not add a CLI framework. The report includes configuration, result
table, feasibility-gate verdicts, per-scenario caveats, cost, limitations, and a clear
statement that synthetic success is not external validation.

**Step 4: Verify GREEN and command smoke tests**

```bash
uv run pytest tests/test_experiment.py tests/test_cli.py -q
uv run corum --help
uv run corum simulate --config tests/fixtures/smoke_config.json --output .corum-work/smoke-run
uv run corum report --run .corum-work/smoke-run --output .corum-work/smoke-report.md
```

**Step 5: Commit**

```bash
git add pyproject.toml src/corum/experiment.py src/corum/reporting.py src/corum/cli.py tests/test_experiment.py tests/test_cli.py tests/fixtures/smoke_config.json
git commit -m "feat: run and report reproducible experiments"
```

---

## Task 9: HaluEval adapter and zero-cost Kaggle reviewer notebook

**Files:**

- Create: `src/corum/datasets/__init__.py`
- Create: `src/corum/datasets/halueval.py`
- Create: `scripts/download_halueval.py`
- Create: `notebooks/halueval_kaggle.ipynb`
- Create: `configs/halueval-sources.json`
- Create: `configs/halueval-models.json`
- Create: `schemas/reviewer-vote-v1.schema.json`
- Create: `tests/fixtures/halueval_tiny.json`
- Create: `tests/test_halueval.py`
- Create: `docs/halueval-protocol.md`

**Required public interface:**

```python
@dataclass(frozen=True, slots=True)
class HaluEvalCase:
    case_id: str
    source_id: str
    task: str
    truth: Truth
    context: str
    candidate: str

def load_halueval_file(path: Path, task: str) -> tuple[HaluEvalCase, ...]: ...
def make_locked_split(
    cases: Sequence[HaluEvalCase],
    *,
    smoke_per_task_class: int = 10,
    calibration_per_task_class: int = 50,
    test_per_task_class: int = 100,
    seed: int = 20260828,
) -> dict[str, tuple[HaluEvalCase, ...]]: ...
```

The loader accepts the official QA, dialogue, and summarization JSON schemas and maps
right answers to `PASS`, hallucinated answers to `FAIL`. The splitter groups by source
before selecting one answer variant, balances task/class, uses a stable SHA-256 ordering,
and proves no source crosses splits. The download script uses fixed official raw URLs,
streaming SHA-256, explicit `--output`, and refuses checksum mismatch. The checked-in
source registry pins upstream Git commit
`b7253db3cdaa0ab2c382f92b26b390109174f77e`, uses commit-addressed raw URLs, and records
the verified SHA-256 digests `89ed139e...57e88` (QA), `9c461df2...8bfd`
(dialogue), `86d2561e...2c758` (summarization), and `73302cd9...9219` (general), together
with exact byte sizes and the MIT license URL. The implementation stores the full digests,
not these shortened prose forms; tests reject a floating branch or absent digest.

The notebook must be valid JSON and executable on Kaggle with attached model-weight
datasets. The pre-registered fixed panel is:

- `Qwen/Qwen2.5-1.5B-Instruct` at
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Qwen2 lineage, Apache-2.0);
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` at
  `31b70e2e869a7173562077fd711b654946d38674` (SmolLM2 lineage, Apache-2.0);
- `microsoft/Phi-3.5-mini-instruct` at
  `2fe192450127e6a83f7441aef6e3ca586c338b77` (Phi-3 lineage, MIT).

The model registry records these revisions, parameter counts, licenses, lineage rationale,
quantization policy, prompt digest, deterministic decoding, minimum capability-smoke
criteria, and one pre-registered replacement per lineage. Any replacement creates a new
registry version before formal votes; no model is replaced using test performance.

The notebook reads the locked split, runs one model at a time, emits only
`PASS`, `FAIL`, or `ABSTAIN`, checkpoints JSONL every 20 cases, hashes model revision,
prompt, tokenizer, decoding config, and input into the cache key, and never collects
chain-of-thought. The vote schema requires case ID, reviewer ID and revision, prompt and
input hashes, semantic observation, execution state, token counts, latency, retry count,
UTC timestamp, and cache key. Local Kaggle attachment paths remain configuration cells;
the registry never claims that weights are already available in the user's account.

**Step 1: Commit an attributed tiny fixture and write failing adapter tests**

Cover all three schemas, label mapping, deterministic split, exact requested counts on a
generated fixture, source leakage prevention, insufficient-stratum error, malformed
records, fixture attribution, fully pinned source/model registries, and vote-schema
validation.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_halueval.py -q`

**Step 3: Implement adapter, downloader, notebook, and protocol**

The protocol locks 60 smoke, 300 calibration, and 600 test cases; three lineage-distinct
small open models; deterministic short output; raw-vote caching; no test-time tuning;
2,880 formal calls plus 10% retry reserve; and `PASS`/`FAIL`/`INCONCLUSIVE` reporting.
The calibration split deterministically allocates 40 of each task/class stratum to
likelihood/dependence fitting and 10 to policy/baseline selection. The balanced test
reports raw metrics plus fixed post-stratified target metrics: for primary
`P(FAIL)=0.20`, class weights are `0.4` for FAIL and `1.6` for PASS before normalization.

**Step 4: Verify GREEN without network dependence**

```bash
uv run pytest tests/test_halueval.py -q
uv run python -m json.tool notebooks/halueval_kaggle.ipynb >/dev/null
uv run ruff check src/corum/datasets scripts/download_halueval.py tests/test_halueval.py
uv run mypy src/corum/datasets scripts/download_halueval.py
```

**Step 5: Commit**

```bash
git add src/corum/datasets scripts/download_halueval.py notebooks/halueval_kaggle.ipynb configs/halueval-sources.json configs/halueval-models.json schemas/reviewer-vote-v1.schema.json tests/fixtures/halueval_tiny.json tests/test_halueval.py docs/halueval-protocol.md
git commit -m "feat: add HaluEval validation path"
```

---

## Task 10: Open-source release surface and automated quality gates

**Files:**

- Create: `README.md`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `AUTHORS.md`
- Create: `CITATION.cff`
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `docs/method.md`
- Create: `docs/references.md`
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Create: `tests/test_public_api.py`

**Step 1: Write failing release-surface tests**

Test that public imports match documented names, package metadata has author, Apache-2.0
license expression, Python range, typed marker, source and issue URLs, and the CLI version
matches `corum.__version__`.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_public_api.py -q`

**Step 3: Implement the public documentation and CI**

README sections are: problem, what Corum does, architecture, five-minute synthetic
quickstart, interpreting `DEFER`, evidence and current benchmark status, cost boundary,
limitations, roadmap, citation, contributing, and license. State that Franz Xu designed
and independently implemented this open-source project; distinguish code authorship from
ownership of general mathematical concepts. Do not mention employment history. References
attribute Dirichlet-multinomial calibration, Bayesian evidence fusion, correlated-design
effect ideas, selective classification, and HaluEval.

CI runs on Python 3.11 and 3.12:

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy src/corum
uv run pytest -q
uv build
```

**Step 4: Verify GREEN and distribution contents**

Run the same six commands locally, then inspect wheel contents to confirm `py.typed`,
license, and package modules are present.

**Step 5: Commit**

```bash
git add README.md LICENSE NOTICE AUTHORS.md CITATION.cff CHANGELOG.md CONTRIBUTING.md SECURITY.md docs/method.md docs/references.md .github/workflows/ci.yml pyproject.toml tests/test_public_api.py
git commit -m "docs: prepare Corum open-source release"
```

---

## Task 11: Execute locked MVP benchmark and publish the evaluation report

**Files:**

- Create: `configs/mvp-simulation.json`
- Create: `artifacts/mvp/manifest.json`
- Create: `artifacts/mvp/metrics.json`
- Create: `artifacts/mvp/case_results.jsonl`
- Create: `artifacts/mvp/report.md`
- Create: `tests/test_published_artifacts.py`

**Step 1: Write failing artifact-integrity tests**

Test that all published artifacts share the same configuration digest and completed
manifest, all six registered scenarios appear, sample/seed counts match the locked config,
metric values are finite or explicitly `null`, feasibility gates are machine-readable,
the report verdict matches `metrics.json`, and external HaluEval status is not presented
as complete without cached real-model votes.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_published_artifacts.py -q`

**Step 3: Run the locked zero-cost benchmark**

Use 50 seeds, 2,000 calibration cases, and 10,000 test cases per scenario. The config
locks 512 shared posterior parameter draws, a 4,096-case fusion chunk, the 1,600/400
fit/policy partition, policy thresholds, priors, bootstrap seed, bootstrap draws, and loss
matrix. The six-scenario run contains 3.6 million generated case panels but uses the
batched kernel and reuses parameter draws per seed; it must never invoke scalar fusion in
the case loop. Run:

```bash
uv run corum simulate --config configs/mvp-simulation.json --output artifacts/mvp
```

Before starting, rerun the registered 10,000-case throughput probe and estimate the upper
wall-clock bound from its measured p95 plus 25% margin. If projected runtime exceeds 45
minutes, optimize vectorized hot paths without changing the registered design. Do not
silently reduce seeds, cases, or posterior draws; a reduced diagnostic run belongs under
`.corum-work/`, not `artifacts/mvp/`.

**Step 4: Inspect failures before writing conclusions**

For each missed feasibility gate, reproduce with the smallest single-scenario seed,
classify it as implementation defect, estimator limitation, or design failure, and fix
only implementation defects. Statistical or design failures remain in the report as an
honest `FAIL` or `INCONCLUSIVE` result.

**Step 5: Render the report and verify artifacts**

```bash
uv run corum report --run artifacts/mvp --output artifacts/mvp/report.md
uv run pytest tests/test_published_artifacts.py -q
```

The report must lead with one of `PASS`, `FAIL`, or `INCONCLUSIVE`, separate simulation
validity from external validity, compare every baseline and ablation, report mean calls
and token-equivalent cost, disclose failures and uncertainty, and prescribe the cheapest
next experiment. If no real HaluEval vote cache exists, mark Stage 2 `PENDING` and do not
claim LLM-level effectiveness.

**Step 6: Commit**

```bash
git add configs/mvp-simulation.json artifacts/mvp tests/test_published_artifacts.py
git commit -m "bench: publish Corum MVP evaluation"
```

---

## Task 12: Independent final review, verification, and remote delivery

**Files:**

- Modify only files required by verified review findings.

**Step 1: Request final code and statistical review**

Provide reviewers the design spec, this plan, full diff from the design commit, test
results, and benchmark artifacts. Require severity-ranked findings with exact file/line
evidence. Address Critical and Important findings; document rejected findings with
technical reasons.

**Step 2: Run a fresh clean verification**

```bash
git diff --check
rg -n -i -f .corum-work/forbidden-patterns.txt .
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy src/corum
uv run pytest -q
uv build
uv run corum --version
uv run corum simulate --config tests/fixtures/smoke_config.json --output .corum-work/final-smoke
```

The ignored local pattern file is created before implementation from the prohibited
identifiers already known in the private working context; the expected result is zero
matches. Record every command, exit code, test count, and artifact digest in the final
handoff.

**Step 3: Merge the isolated implementation branch**

Use the finishing-development workflow to merge the reviewed branch into `main` without
rewriting user history. Tag the verified MVP `v0.1.0` only after all checks pass.

**Step 4: Deliver remotely when the empty repository exists**

Set `origin` to `https://github.com/IcantFind-a-username/Corum.git`, verify the exact
owner/repository, push `main` and `v0.1.0`, then confirm the remote commit and release
files match local hashes. If repository creation remains unavailable, stop at the
permission boundary, preserve all local commits, and report the single required owner
action instead of fabricating a successful push.
