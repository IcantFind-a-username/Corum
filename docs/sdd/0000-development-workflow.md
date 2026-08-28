# SDD: repository-native development workflow

- Status: verified
- Accepted base: `9e1c606`
- Exact commit: `docs: establish repository development workflow`
- Allowed files: `.gitignore`, `AGENTS.md`, `DEVELOPMENT.md`, `pyproject.toml`,
  `docs/specs/corum-mvp-design.md`, `docs/plans/corum-mvp.md`,
  `docs/sdd/0000-development-workflow.md`,
  `docs/sdd/0005-correlated-panel-simulator.md`, and `docs/sdd/TEMPLATE.md`

## Outcome

Make Corum's design, task planning, SDD, TDD, verification, review, and commit gates
portable within the repository without requiring any external workflow plugin.
Keep the process small enough for low-depth rapid implementation while preserving the
evidence needed for trustworthy statistical software.

## Non-goals

Do not implement Task 5 production behavior, CI/release automation, UI, repository
ingestion, or LLM API adapters in this change.

## Contract

Canonical specs and plans use neutral repository paths. Every non-trivial behavior task
has a tracked SDD, begins with RED, maintains at least 80% coverage, receives fresh full
verification and independent read-only review, and uses its registered commit message.
Local scratch output uses ignored `.corum-work/`.

The roadmap must record Tasks 1-4 as complete and Task 5 as current. The product guard
must preserve human-provided contracts as primary authority and keep safe project reading
plus BYO LLM analysis as optional later enrichment.

## TDD evidence

This is a documentation/configuration task with no production behavior. Verify by parsing
`pyproject.toml`, checking all tracked references resolve, confirming no external-plugin
workflow reference remains, running the full test/coverage/Ruff/mypy baseline, and
running `git diff --check`.

## Review and completion

Require an independent read-only review for stale paths, conflicting authority order,
unenforceable commands, accidental Task 5 implementation, and product-scope drift before
the registered commit. Independent review completed clean with no unresolved Critical or
Important finding.
