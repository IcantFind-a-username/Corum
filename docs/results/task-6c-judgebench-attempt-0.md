# Task 6C JudgeBench attempt 0

## Verdict

**FAIL — final for the frozen legacy core and seven-reviewer panel.**

The result does not show that Corum practically beats voting. Its pooled decision-loss
point estimate was lower, but the confidence interval crossed zero and the policy
deferred on nearly every case. The frozen anti-`DEFER` guardrails correctly prevented
that lower loss from being counted as useful superiority.

## Reproducibility record

- Formal execution: exactly once
- UTC capture time: 2026-08-28T18:06:56Z
- Local date: 2026-08-29 (Asia/Singapore)
- Corum commit: `6d03f4cf18c43decff3ae1bffde277279ff25d31`
- Registry SHA-256: `2b45e2e9f064de225fcb55b37b17fa03a751c22f7d4b3ca98cc9071251e7a1af`
- JudgeBench commit: `e2c52c284e735e139b3daa61c206ee208f36c461`
- JudgeBench root tree: `f4439e456b91f93379d966c565b3ab283066f8b9`
- JudgeBench outputs tree: `5634a4f872eeca9665fb6355f6653b1158f8a3e4`
- Split SHA-256: `15725e487368ce26afd938d272985315d97a774d0ed44dd658a5f9619572815e`
- Runtime: Python `3.14.0`, NumPy `2.5.2`
- Command exit code: `1`, caused by the locked assertion that only `PASS` is green
- Pytest result: `1 failed, 10 passed in 1.10s`
- Command wall time: `1.5644136s`; outer capture wall time: `1.7s`

The actual non-secret environment values were:

```text
CORUM_RUN_JUDGEBENCH_V1=1
CORUM_JUDGEBENCH_RAW_DIR=C:\Users\user\Documents\Codex\2026-08-28\x20-https-github-com-icantfind-a\work\Corum\.worktrees\task-1-numeric-validation\.corum-work\judgebench-v1\raw
CORUM_JUDGEBENCH_UPSTREAM_REPO=C:\Users\user\Documents\Codex\2026-08-28\x20-https-github-com-icantfind-a\work\judgebench-metadata
```

The only formal command was:

```powershell
.venv\Scripts\uv.exe run pytest tests/test_judgebench_value.py -q -s
```

All eight local raw files matched the registered byte count, SHA-256, and Git blob OID
before execution. Raw and normalized rows remain under ignored local storage and are not
committed or redistributed.

## Aggregate result

The panel contains seven eligible reviewers from three lineages. Of 350 total cases, 128
were calibration fit, 68 policy selection, and 154 pooled test; the pooled test contains
the untouched 42-case coding slice plus 112 general cases.

| Metric | Corum | Ordinary majority | Lineage-balanced majority |
|---|---:|---:|---:|
| Pooled decision loss | 0.253247 | 0.285714 | 0.285714 |
| Pooled coverage | 3.90% (6/154) | 89.61% (138/154) | 89.61% (138/154) |
| Pooled useful resolution | 2.60% (4/154) | 63.64% (98/154) | 63.64% (98/154) |
| Pooled decisions | 6 | 138 | 138 |
| Pooled errors among decisions | 2 | 40 | 40 |
| Pooled `DEFER` | 148 | 16 | 16 |
| Coding decision loss | 0.267857 | 0.416667 | 0.410714 |
| Coding coverage | 2.38% (1/42) | 85.71% (36/42) | 88.10% (37/42) |
| Coding useful resolution | 0% | 47.62% | 50.00% |

The pooled absolute loss benefit against each baseline was
`0.032467532467532464`, or an `11.36%` relative point reduction. Both paired
source-stratified 95% benefit intervals were
`[-0.025974025974025976, 0.09253246753246754]`, which cross zero. This is not a
statistically reliable improvement claim and is not an accuracy increase: under the
frozen `DEFER=0.25` cost, Corum obtained the lower point loss by deferring on `96.10%`
of cases. Its decided-case accuracy was 4/6, below each baseline's 98/138.

On the coding slice, Corum made one decision and that decision was wrong; 41 of 42 cases
were deferred. Therefore the lower coding loss point estimate is not evidence of better
coding judgments.

## Gate audit

The following checks passed:

- full upstream, file, split, alignment, and numerical integrity;
- the 10% ordinary-majority and 5% lineage-majority relative point-loss targets;
- pooled and coding point no-harm checks under the frozen loss;
- false-`PASS` and false-`FAIL` directional guardrails.

The following checks failed:

- both paired-confidence lower bounds were not strictly positive;
- absolute pooled coverage and coverage parity against both baselines;
- useful-resolution parity against both baselines;
- the policy-selection coverage constraint: coverage was `0.0294118` and
  `constraint_satisfied=false`;
- absolute coding coverage and coding useful-resolution parity against both baselines.

The selected policy was `fail_threshold=0.2`, `pass_threshold=0.8`,
`min_effective_sample_size=1.0`, `min_lineages=2`, and
`min_valid_reviewers=2`. Its policy-partition constraint already failed before the
held-out scores were used.

## Decision and claim boundary

This is a design failure for the frozen core/panel, not an invalid run. No same-data
repair, threshold change, panel substitution, split change, or replacement attempt is
allowed. A future idea requires a new prospective SDD, a genuinely new evaluation, and a
mechanism hypothesis that addresses usable coverage rather than merely reducing loss by
abstention.

The result is static answer-comparison evidence only. It is not repository-patch,
developer-adoption, production-readiness, or universal-superiority validation. It keeps
the adaptive cascade, UI, repository ingestion, LLM adapters, quality scoring, reporting,
and all other component/product expansion blocked.

The complete aggregate JSON is in
`docs/results/task-6c-judgebench-attempt-0.json`; the exact captured command output is in
`docs/results/task-6c-judgebench-attempt-0.txt`.

## Post-result verification

All commands below ran from the result tree without the formal external-run switch:

- `.venv\Scripts\pytest.exe tests/test_judgebench_value.py -q`:
  `10 passed, 1 skipped in 0.11s`; the skip is the already-consumed formal gate.
- `.venv\Scripts\pytest.exe -q --ignore=tests/test_core_value.py --ignore=tests/test_pair_value.py`:
  `431 passed, 1 skipped in 3.94s`.
- The same ordinary suite with branch coverage and `--cov-fail-under=80`:
  `431 passed, 1 skipped`, total coverage `88.12%`.
- `.venv\Scripts\ruff.exe check src tests`: exit `0`, all checks passed.
- `.venv\Scripts\mypy.exe src/corum`: exit `0`, no issues in nine source files.
- `git diff --check`: exit `0`; only Git's configured LF-to-CRLF notices were emitted.
- Independent artifact comparison: the committed aggregate JSON and stdout capture both
  match the unique formal execution verbatim.
- Scope audit: exactly the four registered status/result documents and three registered
  result files changed; no `src/corum`, judge, registry, raw row, normalized row,
  `uv.lock`, or upstream content is included.

Artifact SHA-256 values:

- aggregate JSON:
  `6cd7708120be7c4ff63801db252005ac4e22b76f571d9f2de71a7a37c73d0027`
- exact stdout/stderr capture:
  `67cb82ccc6db076e560acb910cecf8daef950fa7a4b44851eadb8e08fd5a6c55`
