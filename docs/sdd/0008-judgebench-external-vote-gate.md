# SDD: JudgeBench external vote value gate

- Status: approved
- Accepted base: `079ccfc770ed71e398660b31a9179c01c9ee42ce`
- Exact documentation commit: `docs: register JudgeBench external value gate`
- Exact judge commit: `test: lock JudgeBench external value gate`
- Exact attempt-0 result commit: `docs: record JudgeBench external gate result`
- Allowed documentation files: `AGENTS.md`, `docs/specs/corum-mvp-design.md`,
  `docs/plans/corum-mvp.md`, and this SDD
- Allowed judge files: `configs/judgebench-v1.json` and
  `tests/test_judgebench_value.py`
- Allowed attempt-0 result files: `AGENTS.md`, `docs/specs/corum-mvp-design.md`,
  `docs/plans/corum-mvp.md`, this SDD, and
  `docs/results/task-6c-judgebench-attempt-0.{json,md,txt}`

## Outcome

Task 6C performs one zero-inference-cost external replay of the unchanged legacy Corum
power-likelihood core on 350 public JudgeBench answer comparisons. It asks whether Corum
turns the same seven static judge outputs into materially better decisions than both
ordinary reviewer-level majority voting and a stronger lineage-balanced voting practice.
The gate is prospective: no held-out judge performance may be calculated until the
configuration and independent judge are committed and independently reviewed.

A `PASS` is evidence that the existing static core deserves the next practical validation
stage. It does not admit the failed pair-block component, unlock the adaptive cascade,
prove repository-patch correctness, prove universal superiority, or establish that
developers already want the product. It authorizes only a minimal offline evaluator and a
fresh real developer-project/patch value gate. `FAIL` and `INCONCLUSIVE` authorize no new
statistical or product component.

## Why this pivot is allowed

Task 6A and Task 6B remain permanently recorded failures under their own frozen judges.
Task 6B Gate B nevertheless showed a decision-level signal against ordinary majority,
while Gate A rejected the new pair likelihood. On 2026-08-29 the owner directed Corum to
prioritize a demonstrable advantage over traditional voting and actual developer utility.
Task 6C therefore freezes the rejected pair path, adds no statistical mechanism, and tests
the already-implemented legacy power core against real cached reviewer behavior before
any cascade, UI, repository reader, provider adapter, or quality-scoring expansion.

## Non-goals

- Do not modify `src/corum`, the old locked judges, or any statistical literal.
- Do not fit or invoke a pair-block likelihood.
- Do not tune a reviewer panel, split, threshold, loss, baseline, or policy on test rows.
- Do not call an LLM, provider API, paid service, or live benchmark endpoint.
- Do not redistribute upstream pair data, judge outputs, prompts, or reasoning text.
- Do not implement a CLI, UI, cascade, project reader, LLM adapter, report renderer, or
  project-quality score.
- Do not claim repository-level, patch-level, adoption, or production validation.

## Upstream evidence and license boundary

The sole upstream revision is the official JudgeBench Git commit
`e2c52c284e735e139b3daa61c206ee208f36c461`. The official Hugging Face dataset card marks
the dataset MIT, but that GitHub revision has no root `LICENSE`, `COPYING`, or `NOTICE`
covering the cached `outputs/` files. Until the authors clarify the output license, Corum
may checksum and analyze those public files locally but must not commit or redistribute
raw or normalized judge outputs. Only aggregate metrics, provenance, and hashes may enter
the repository.

The judge requires these exact eight Git blob bytes:

| Role | Upstream path | Bytes | SHA-256 |
|---|---|---:|---|
| Pair/truth data | `data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl` | 1,960,188 | `d5106306b38dead1b2e964ecf5534a3e7b2fc4b8d3b8da327599edfcebd76042` |
| Claude Sonnet votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=claude-3-5-sonnet-20240620.jsonl` | 7,783,771 | `03c61ba5afaeef0476840212efb93aa4ca5c07505c46a4650a1fef225cf9d718` |
| Claude Haiku votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=claude-3-haiku-20240307.jsonl` | 7,372,693 | `c11818ca171588e7dcd3b238a4d3678def31b4eb4f64e03b5a7ff8b9b477710f` |
| Gemini Flash votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=gemini-1.5-flash-001.jsonl` | 7,400,061 | `f38f49b3b40c6260d3816f29f382a3586a6f7e96586f586b1fa8266a6a2fe6fd` |
| Gemini Pro votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=gemini-1.5-pro-001.jsonl` | 7,599,138 | `f9c9bc7ee7467e23308344035a9e60f862209210ad91cf56a348b1f0cc5b81ea` |
| Llama 8B votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=meta-llama_Meta-Llama-3.1-8B-Instruct.jsonl` | 8,703,281 | `01f4b18b828cdba03da25c3cb14843bfd63593b7d9b3ec5cafe79d4817cffedf` |
| Llama 70B votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=meta-llama_Meta-Llama-3.1-70B-Instruct.jsonl` | 8,114,787 | `d3ac1be5a4fedcb128280096ba7300c3faae5ef2b0512ad2b0224b62907d5c2d` |
| Llama 405B votes | `outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=meta-llama_Meta-Llama-3.1-405B-Instruct.jsonl` | 8,136,968 | `a8c0aa6871118ab00b2e5ac18cb51184a2cfac67f6a7375285d25b8da1d2afd8` |

The registry uses raw URLs pinned to that commit. Downloads live only under the ignored
`.corum-work/judgebench-v1/raw/` directory. The judge does not inspect, print, or persist
free-form judge responses; it uses only the declared decision fields and structural
metadata required for integrity checks.

## Fixed panel

The panel is generated by one outcome-blind rule: from the pinned commit, take every
`judge_name=arena_hard` output for the GPT-4o response subset whose judge model is not an
OpenAI model. This uniquely includes all seven available Anthropic, Google, and Meta
outputs. It excludes exactly GPT-4o, GPT-4o-mini, o1-mini, and o1-preview because the
compared response set is generated by an OpenAI model and a same-vendor judge could have
privileged/self-family behavior. No quality, size, leaderboard result, or observed vote
may include or exclude a model.

The anti-cherry-picking inventory is itself machine-verified from Git, not inferred from
the downloaded files. The pinned commit has root tree
`f4439e456b91f93379d966c565b3ab283066f8b9` and `outputs/` tree
`5634a4f872eeca9665fb6355f6653b1158f8a3e4`. Filter
`git ls-tree -r -l -z <commit> -- outputs` to the GPT-4o response subset and
`judge_name=arena_hard`. It must yield exactly these 11 candidates:

| Judge model | Git blob | Bytes | Eligibility |
|---|---|---:|---|
| `claude-3-5-sonnet-20240620` | `d63a81d6850f7f4bc808c553b275377f4720ece9` | 7,783,771 | include |
| `claude-3-haiku-20240307` | `23dfdd64354b63e78c606ff404314b6d03bb8e15` | 7,372,693 | include |
| `gemini-1.5-flash-001` | `b2f5a73a2bfbfc46f6bf939f055ee897d37d9d4b` | 7,400,061 | include |
| `gemini-1.5-pro-001` | `3f6f61e29d93e80079e15d7093a1e58ac49de130` | 7,599,138 | include |
| `gpt-4o-2024-05-13` | `6846366f41315ed70ca3c12ee181ff47b7cb829c` | 8,198,919 | exclude: OpenAI |
| `gpt-4o-mini-2024-07-18` | `cc89576fe4d641e6c9778c08b1442e401de36a21` | 7,875,448 | exclude: OpenAI |
| `meta-llama_Meta-Llama-3.1-405B-Instruct` | `cdee8b3e6b7fe59ff3cf3bec6450d447aa0b7c4d` | 8,136,968 | include |
| `meta-llama_Meta-Llama-3.1-70B-Instruct` | `ff4ce49c77bda4d991be269bf8159111224ba649` | 8,114,787 | include |
| `meta-llama_Meta-Llama-3.1-8B-Instruct` | `64f83c4dac0c8df6f3fb6ead2f5562935f155c89` | 8,703,281 | include |
| `o1-mini-2024-09-12` | `e155a6527af6b98cc4932124831c8c44c31057c8` | 3,879,585 | exclude: OpenAI |
| `o1-preview-2024-09-12` | `bf4381b7e8df6f2b0641881ab8f59065f12d91e9` | 6,220,610 | exclude: OpenAI |

For each candidate sorted by full upstream path, hash the UTF-8 bytes
`path + NUL + blob_oid + NUL + decimal_size + LF`. The SHA-256 of the resulting 1,929
bytes must be `170120b2373084ce93dcebc8cb23f47ebfc7ad1438d3ba8f594de39163067558`.
The formal judge requires a local clone through `CORUM_JUDGEBENCH_UPSTREAM_REPO` and
verifies the commit, both tree object IDs, the complete inventory, and this digest before
reading votes. The four excluded blobs need not be materialized.

| Reviewer ID | Vendor/family | Lineage |
|---|---|---|
| `claude-sonnet-20240620` | Anthropic / Claude 3 | `anthropic/claude-3` |
| `claude-haiku-20240307` | Anthropic / Claude 3 | `anthropic/claude-3` |
| `gemini-flash-001` | Google / Gemini 1.5 | `google/gemini-1.5` |
| `gemini-pro-001` | Google / Gemini 1.5 | `google/gemini-1.5` |
| `llama-3.1-8b` | Meta / Llama 3.1 | `meta/llama-3.1` |
| `llama-3.1-70b` | Meta / Llama 3.1 | `meta/llama-3.1` |
| `llama-3.1-405b` | Meta / Llama 3.1 | `meta/llama-3.1` |

Every case therefore has seven reviewer records and three predeclared lineage labels. All
declared reviewer costs are `1.0`; every method receives the same seven cached outputs.

## Decision normalization

Each upstream output row has two judgments: the first presents the original response A/B
order and the second presents the reversed order. Normalize without consulting truth:

1. In a judgment's displayed orientation, `A>B` and `B<A` mean displayed A wins;
   `B>A` and `A<B` mean displayed B wins; `A=B` and `B=A` mean tie.
2. Convert the first judgment directly to original-orientation `PASS=+1`, `FAIL=-1`, or
   tie/null/missing decision `=0`.
3. Reverse the sign of the second judgment to restore original orientation; zero remains
   zero.
4. Sum the two signs exactly as the pinned upstream metric does: a positive sum emits
   `Observation.PASS`, a negative sum emits `Observation.FAIL`, and zero emits
   `Observation.ABSTAIN`. Thus one directional result plus one tie/null retains the
   direction, while opposite directions, two ties, or two missing decisions abstain.
5. Any other decision token invalidates the attempt. No ambiguous value may be resolved
   using the label.

Upstream label `A>B` maps to `Truth.PASS`; `B>A` maps to `Truth.FAIL`. Any other truth
label invalidates the attempt. An abstention remains a valid semantic observation, not a
missing reviewer and not an extra vote.

## Integrity contract

Before metrics, the independent judge must establish all of the following or return
`INVALID` without a scientific verdict:

- every file byte count and SHA-256 matches the registry;
- the pair file contains exactly 350 unique `pair_id` values and each output contains the
  same set exactly once;
- `SHA256(source + NUL + question)` produces exactly 350 unique source groups;
- every output row declares the registered response model, judge name, and judge model;
- every row contains exactly two judgment objects; each decision is absent, null, or in
  the normalization vocabulary above, with no other token;
- the local upstream clone resolves the pinned commit and tree IDs, and its complete
  11-candidate Arena-Hard inventory matches the registered path/blob/size digest;
- the outcome-blind eligibility rule selects exactly the seven registered non-OpenAI
  candidates and the joined dataset has seven reviewer observations per case;
- fit, policy, general test, and coding test group sets are pairwise disjoint;
- posterior arrays, actions, per-case losses, bootstrap values, and every gate operand are
  finite and seeded; every gate denominator is strictly positive;
- `pair_likelihood_draws` is empty, proving the rejected pair path was not used.

An undefined non-gating ratio is not an integrity failure. For example, false-safe risk
with no `PASS` action or selective risk with zero source-slice coverage is recorded as
`{"numerator": ..., "denominator": 0, "value": null}`. The corresponding zero coverage
still fails any applicable utility guardrail; it never converts algorithm behavior into a
re-runnable `INVALID` result.

## Frozen split

The split unit is `source_group = SHA256(source + NUL + question)`, not `original_id`,
because 196 upstream GPT-4o rows have a blank `original_id`.

All 42 `source == "livecodebench"` cases form the untouched `coding_test` slice. They may
not fit calibration, dependence, or policy. For each `(source, label)` stratum among the
remaining 308 cases:

1. sort ascending by
   `SHA256("corum:judgebench:v1" + NUL + source_group)`;
2. assign repeated zero-based cycle
   `[fit, policy, test, fit, test, fit, policy, test, fit, test]`.

This yields exactly 128 fit, 68 policy, 112 general-test, and 42 coding-test cases. The
canonical digest of lines
`pair_id + NUL + split + NUL + source_group + LF`, sorted by `pair_id`, is
`15725e487368ce26afd938d272985315d97a774d0ed44dd658a5f9619572815e`.
The pooled locked test is the 154-case union of general test and coding test. Neither test
partition may influence the selected policy.

## Frozen Corum path

Use only the legacy no-pair power-likelihood path and these literals:

| Literal | Value |
|---|---:|
| singleton prior strength | `1.5` |
| dependence shrinkage | `0.25` |
| minimum dependence overlap | `10` |
| same-lineage fallback cap | `1.0` |
| prior `P(PASS)` | `0.5` |
| posterior draws | `512` |
| credible mass | `0.95` |
| fusion chunk size | `4096` |
| fusion seed | `20260829` |
| symmetric false-PASS cost | `1.0` |
| symmetric false-FAIL cost | `1.0` |
| DEFER cost | `0.25` |
| policy-selection minimum coverage | `0.70` |
| paired bootstrap draws | `5000` |
| paired bootstrap seed | `20260830` |

Fit singleton calibration and dependence only on the 128 fit cases. Select exactly one
policy on the 68 policy cases with the repository's existing frozen candidate grid and
tie-breaks. An unsatisfied policy coverage constraint is a gate failure. Build one fusion
context with no pair calibrations and use it unchanged for policy and test cases.

## Voting baselines

The primary traditional baseline is ordinary unweighted majority over all seven reviewer
observations. `PASS` and `FAIL` count one each; abstentions count zero; a tie returns
`DEFER`.

The stronger practice baseline is lineage-balanced majority:

1. Within each of the Anthropic, Google, and Meta lineages, take an unweighted majority of
   non-abstaining member votes; an internal tie or no semantic vote becomes a lineage
   abstention.
2. Take an unweighted majority of the resulting three lineage votes; a tie returns
   `DEFER`.

It receives no learned reliability, truth, or test-selected threshold. Corum must beat
both fixed baselines, not a favorable one selected after inspection. Naive independent
fusion, posterior-mean linear pooling, and best-single reviewer may be reported only as
diagnostics and cannot rescue the gate.

## Metrics and anti-DEFER utility

Evaluate the pooled locked test and the coding slice with the symmetric costs above.
Report decision loss, coverage, false-PASS rate, false-FAIL rate, false-safe risk,
selective risk, useful resolution, abstention reasons, per-source values, and per-reviewer
normalization counts for the decision methods. Brier, NLL, and ECE apply only to Corum and
explicit probability-producing diagnostics such as naive independent fusion or linear
pooling; voting baselines have no invented probability score.

`useful_resolution` is the fraction of all cases receiving a correct non-`DEFER`
decision. For each voting baseline, run a paired source-stratified bootstrap of
`baseline_case_loss - corum_case_loss`; resample cases with replacement within each
source, use 5,000 draws, and report the point estimate and percentile 95% interval.

The bootstrap implementation is frozen. Sort source strings ascending, then sort cases in
each source by `pair_id`. Initialize one `numpy.random.default_rng(20260830)`. For each
replicate and each source in that order, call `integers(0, source_size,
size=source_size)` once and use the same sampled case indices for both voting baselines.
Compute every mean with `math.fsum(values) / len(values)`. The interval endpoints are
`numpy.quantile(replicate_values, [0.025, 0.975], method="linear")`; no rounding occurs
before gate comparison, and the lower-bound test is strict `> 0.0`.

## Frozen verdict

The attempt is `PASS` only when every condition below is true on the 154-case pooled test,
unless a condition explicitly names the 42-case coding slice:

1. Corum decision loss is at most `0.90 * ordinary_majority_loss`.
2. Corum decision loss is at most `0.95 * lineage_balanced_loss`.
3. The paired 95% benefit interval lower bound is greater than zero against each voting
   baseline.
4. Corum coverage is at least `0.70` and no more than `0.10` below either voting baseline.
5. Corum useful resolution is no more than `0.02` below either voting baseline.
6. Corum false-PASS and false-FAIL rates are each no more than `0.02` above either voting
   baseline's corresponding rate.
7. On `coding_test`, Corum decision loss is no more than one full-error unit (`1 / 42`)
   above either voting baseline, coverage is at least `0.60`, and useful resolution is no
   more than `0.10` below either voting baseline.
8. The policy-selection coverage constraint and every integrity invariant pass.

If integrity fails, the result is `INVALID` and the defect must be fixed prospectively
before a newly versioned attempt. If integrity passes but any coverage, useful-resolution,
directional-error, coding-slice, policy-constraint, or point-harm guardrail fails, the
result is `FAIL`. If all no-harm and utility guardrails pass but a relative-improvement
target or confidence lower bound misses, the result is `INCONCLUSIVE`. `FAIL` and
`INCONCLUSIVE` both keep downstream development blocked.

For this rule, pooled point harm means Corum decision loss is greater than either voting
baseline; coding-slice point harm means it exceeds either baseline by more than `1 / 42`.
Missing a stricter 10% or 5% relative-improvement target without point harm is
`INCONCLUSIVE`, not `FAIL`.

The first frozen result is final for this core and panel. No same-data repair, threshold
change, panel substitution, split change, or algorithm tuning is allowed. Only a proven
parser or judge defect can invalidate the run, and its fix requires a new SDD version and
a fresh result path rather than editing attempt 0.

## TDD evidence

This task adds no production behavior. The committed test is itself the prospective,
independent evaluation contract, so it must not be executed as a RED probe. Before its
first formal execution:

1. commit and independently review this documentation;
2. add the registry and complete judge without modifying `src/corum`;
3. run static checks and synthetic parser/reference fixtures that cannot access upstream
   held-out values;
4. independently review the judge, reference majority, split logic, loss rows,
   bootstrap, and verdict logic;
5. commit exactly `test: lock JudgeBench external value gate`;
6. run exactly once with `CORUM_RUN_JUDGEBENCH_V1=1`, the pinned raw directory, and the
   pinned local upstream repository.

The normal repository suite must skip this external-data judge unless that explicit
environment switch is set. The formal command, stdout/stderr, wall time, commit, registry
digest, environment, aggregate result JSON, and final verdict are retained verbatim under
`docs/results/`.

## Review and completion

Documentation and judge reviews are read-only and independent. Every Critical or
Important finding is fixed and re-reviewed before the one-time run. Completion requires
the result artifacts, honest roadmap status, a clean ordinary suite, Ruff, mypy,
`git diff --check`, and no committed upstream content. Whatever the verdict, the failed
Task 6A/6B evidence remains visible and no claim exceeds this benchmark's answer-comparison
scope.
