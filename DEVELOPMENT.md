# Corum Development Workflow

Corum uses a repository-native SDD + TDD workflow. No external workflow plugin is
required. The default execution style is low-depth, fast, and sequential; evidence gates
remain strict.

## Source of truth

Follow the read order and authority rules in `AGENTS.md` before changing production
behavior.

## One-task loop

1. **Scope:** record the goal, non-goals, accepted base, allowed files, public contract,
   test command, and exact commit message in a short SDD.
2. **RED:** add the smallest behavioral test and retain a failure proving the behavior is
   missing.
3. **GREEN:** implement only enough robust behavior to satisfy the test and contract.
4. **Refactor:** simplify without changing behavior; keep focused tests green.
5. **Verify:** run focused tests, the full suite, Ruff, mypy, coverage, required benchmark
   or replay, and `git diff --check`.
6. **Review:** inspect the whole diff, then obtain an independent read-only review. Fix
   every Critical and Important finding with a regression when behavior changes.
7. **Commit:** use the task's registered message. Do not mix the next task or push a
   milestone without its separate delivery gate.

## Fast-path rules

- Use low reasoning depth for clear local changes and one RED/GREEN cycle at a time.
- Do not skip typed validation, deterministic seeds, data-leakage checks, performance
  gates, or independent review to gain speed.
- Escalate only when the design conflicts, statistics are ambiguous, privacy/security is
  involved, a test fails repeatedly or flakes, or review remains unresolved.
- Parallel work is read-only unless task files are independent. Production modules remain
  sequential so evidence cannot race implementation.

## Component admission

Before implementing a component beyond a minimal test spike, record its user outcome,
existing-practice baseline, locked dataset/scenario split, metric, pass threshold,
non-regression boundaries, and removal/stop rule in the SDD. The builder cannot weaken
that acceptance condition. An independent deterministic judge runs it, with a maximum of
three repair cycles before owner escalation.

Infrastructure is justified by the comparison it enables, not by its architecture. Do
not begin the adaptive cascade, UI, project reader, LLM integration, quality score, or
another surface until its predecessor's value gate passes. Human judgment owns whether a
passing synthetic result is meaningful enough to continue.

## Baseline commands

```bash
uv run pytest -q
uv run pytest --cov=corum --cov-report=term-missing -q
uv run ruff check src tests scripts
uv run mypy src/corum scripts
git diff --check
```

Coverage must remain at least 80%. Task-specific statistical and performance gates outrank
the generic baseline. Network, paid inference, and hidden truth access are forbidden unless
the project owner explicitly authorizes them.

Local scratch output belongs in ignored `.corum-work/`. Product artifacts belong only in
the paths registered by their roadmap task.
