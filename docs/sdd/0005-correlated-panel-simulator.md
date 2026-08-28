# SDD: Task 5 correlated-panel simulator

- Status: approved
- Accepted base: `91fa7ea`
- Exact commit: `feat: simulate correlated reviewer panels`
- Allowed files: `pyproject.toml`, `src/corum/simulation.py`,
  `tests/test_simulation.py`

## Outcome

Provide a deterministic, zero-cost simulator that can falsifiably test whether Corum's
dependence-aware consensus improves risk and cost under independent reviewers, clone
correlation, majority traps, informative missingness, drift, and cascade-oriented costs.
The simulator is accepted as test infrastructure only; it creates no standalone value
claim and must feed the locked core-vs-majority gate immediately after Task 6.

## Non-goals

No UI, repository reader, LLM/provider adapter, baseline comparison, cascade orchestration,
report renderer, network access, or real-model claim belongs in Task 5.

## Contract

Implement the exact Task 5 public interface from `docs/plans/corum-mvp.md`, including the
self-contained lineage correlation diagnostics. Add SciPy as a runtime dependency solely
for the normal CDF used by the Gaussian copula. All other sampling uses NumPy and every
random operation receives an explicit seed.

Configured lineage values are target observed binary-error correlations, not latent
Gaussian correlations. Solve a deterministic equicorrelated latent parameter, reject
infeasible or non-PSD configurations, and expose enough immutable metadata to compare the
target, solved parameter, and realized diagnostic correlation without hidden truth or
test-set tuning.

The panel records the exact seed, immutable truth and difficulty mappings, one review per
case/reviewer, explicit empty gate tuples, and lineage diagnostics containing reviewer
IDs, target, solved latent correlation, minimum eigenvalue, realized overlapping-VALID
error correlation, and overlap count. `ABSTAIN` is an error for correlation but remains a
valid categorical outcome. Base timeout plus invalid rate cannot exceed one;
`informative_missingness` increases missingness on difficult and FAIL cases. The
adversarial ID only identifies behavior already encoded in phase likelihoods.

One lineage target means the unweighted arithmetic mean across all unordered reviewer
pairs. Solve against the mean of pair-specific expected correlations and report the mean
of separately estimated overlapping-`VALID` pair correlations. Reject zero-variance or
unreachable targets. Run the 100,000-case `0.03` accuracy gate without informative
missingness and test informative selection separately.

## TDD evidence

- RED: `uv run pytest tests/test_simulation.py -q`
- GREEN: the same focused test command
- Static: Ruff and mypy on the two Task 5 modules
- Full: repository pytest, coverage at least 80%, Ruff, mypy, and `git diff --check`
- Statistical: 100,000 diagnostic cases, every registered lineage within `0.03` of its
  target and every solved copula block PSD
- Performance: registered focused Task 5 verification under five seconds

## Review and completion

Require independent read-only review of statistical semantics, seed reproducibility,
immutability, missingness, adversarial behavior, allowed-file scope, and runtime. Fix and
re-review every Critical or Important finding before the registered commit.
