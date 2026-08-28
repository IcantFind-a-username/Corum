# SDD: evidence-first component admission

- Status: verified
- Accepted base: `91fa7ea`
- Exact commit: `docs: gate components on baseline evidence`
- Allowed files: `AGENTS.md`, `DEVELOPMENT.md`, `docs/specs/corum-mvp-design.md`,
  `docs/plans/corum-mvp.md`, `docs/sdd/0001-component-value-gate.md`, and
  `docs/sdd/0005-correlated-panel-simulator.md`

## Outcome

Prevent Corum from becoming a theoretically elaborate but unusable collection of
components. Lock an early, machine-decidable comparison showing whether the existing core
beats ordinary majority voting and whether dependence correction beats its naive ablation
before the project builds the cascade or product surfaces.

## Non-goals

This change does not claim the gate passes, alter production algorithms, implement the
simulator/baselines, or treat synthetic evidence as real-world validation.

## Contract

The independent judge uses identical reviews, fixed scenario definitions, 20 fixed seeds,
2,000 calibration and 5,000 test cases per seed, the published asymmetric loss, locked
policy selection, at least 50% coverage, and paired uncertainty. The builder cannot edit
the judge to make a result pass.

Pass requires all of the following:

- pooled Corum decision loss at least 10% below majority with the paired 95% benefit
  interval strictly above zero;
- per-scenario loss no more than 0.01 worse, false-PASS no more than 0.02 worse, and zero
  hard-gate violations;
- dependence-aware NLL or Brier at least 5% better than naive independent fusion on the
  two correlated scenarios and no more than 1% worse on the independent scenario.

Failure blocks Task 7 and later product components. Three unchanged-judge repair cycles is
the hard retry cap; the owner decides whether to pivot or stop.

## Loop review

- Decidable: one deterministic test command returns pass/fail.
- Anti-Goodhart: baseline, cases, seeds, loss, coverage, risk boundaries, and tests are
  locked before results; deletion or weakening is forbidden.
- Independent judge: implementation agents do not grade or rewrite acceptance.
- Fallback: three bounded repairs, then `CORE_VALUE_GATE_FAILED` and owner judgment.
- Boundary: a synthetic PASS authorizes only external validation, never a usefulness or
  production-readiness claim.

## Review and completion

Verify all authority documents agree, the new gate blocks Task 7, no production file is
changed, and an independent reviewer finds no unresolved Critical or Important issue.
