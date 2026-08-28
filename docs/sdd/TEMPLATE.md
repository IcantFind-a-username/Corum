# SDD: <task name>

- Status: proposed | approved | in progress | verified | committed
- Accepted base: `<commit>`
- Exact commit: `<type: message>`
- Allowed files: `<explicit paths>`

## Outcome

State the user-visible or experimentally measurable result in one paragraph.

## Non-goals

List adjacent behavior that this task must not implement.

## Contract

Record public interfaces, invariants, error behavior, determinism, performance limits,
and any approved amendment to the roadmap.

## TDD evidence

- RED command and expected missing-behavior failure
- GREEN focused command
- Full verification and coverage commands
- Benchmark, replay, or security checks when applicable

## Review and completion

Record Critical/Important findings, regression fixes, re-review verdict, final commit, and
any external validation that remains pending. A synthetic result must never be described
as real-world validation.
