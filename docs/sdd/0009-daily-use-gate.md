# SDD: Daily Use Gate

- Status: committed — panel smoke formally `BLOCKED`; formal attempt unconsumed
- Accepted base: `29dbc129d13f307ac0633510feefb1e0c2b6684b`
- Exact documentation commit: `docs: register daily use value gate`
- Exact judge commit: `test: lock daily use value gate`
- Exact reviewer-ledger seal commit: `data: seal daily use reviewer ledger`
- Exact attempt-0 result commit: `docs: record daily use gate result`
- Allowed documentation files: `AGENTS.md`, `docs/specs/corum-mvp-design.md`,
  `docs/plans/corum-mvp.md`, and this SDD
- Allowed judge files: `configs/daily-use-v1.json` and
  `tests/test_daily_use_value.py`
- Allowed seal file: `configs/daily-use-v1-seal.json`
- Allowed attempt-0 result files: the four documentation files above and
  `docs/results/task-6d-daily-use-attempt-0.{json,md,txt}`

## Outcome

Task 6D asks one deliberately narrow product question: on 500 objectively scored real
development patches, can the unchanged no-pair Corum fusion readout turn the same three
LLM reviews into meaningfully more accurate and more stable accept/reject decisions than
ordinary and reliability-weighted voting, without using materially more tokens?

This is an owner-approved prospective exception after Task 6C's final `FAIL`. It does not
rewrite Task 6A, 6B, or 6C; repair the rejected pair component; change the JudgeBench
judge; or claim that the current core already beats voting. It exists to decide whether
further investment is justified before Corum grows product surfaces.

A `PASS` requires every registered accuracy, token, stability, coverage, and false-safe
condition below. It authorizes only a new owner-reviewed plan for a minimal human-input
project form, bring-your-own LLM flow, safe optional repository reading, and practical
quality feedback. It does not admit a statistical component or automatically unlock the
old Task 7 cascade. `FAIL` or `INCONCLUSIVE` stops work on the current consensus path and
keeps all product expansion blocked. The exact pre-join and post-join `INVALID` behavior
is registered below; neither is a favorable scientific result.

## Non-goals

- Do not modify `src/corum`, any Task 6A/6B/6C judge, or any recorded result.
- Do not add or tune a statistical component, threshold, model, prompt, candidate patch,
  baseline, or exclusion after formal outcomes are inspected.
- Do not implement the cascade, CLI, UI, repository reader, provider adapter, secret
  handling, project score, or report product.
- Do not call a paid API or assume an owner API budget. Reviewer inference happens
  outside the repository and only after exact models, prompts, limits, and hashes are
  frozen.
- Do not expose a gold patch, test patch, hidden test identifier, harness output, result
  label, future issue comment, or fixing-commit history to a reviewer.
- Do not redistribute upstream repositories, issue text, candidate patches, containers,
  raw LLM responses, or per-task outcomes. Only hashes, provenance, aggregate metrics,
  and non-identifying slices may be committed.
- Do not present this Python-only patch-review gate as proof of multilingual project
  understanding, patch generation, developer adoption, or production safety.

## Frozen question and systems

Each case asks whether one candidate patch correctly and safely resolves its issue at the
registered base commit. The executable oracle is the pinned SWE-bench harness outcome:
`PASS` only when every declared `FAIL_TO_PASS` and `PASS_TO_PASS` test succeeds; every
other completed outcome is `FAIL`.

The panel contains exactly three distinct reviewer model IDs from three declared base
lineages. Each valid reviewer response emits only `PASS`, `FAIL`, or `ABSTAIN`; no unused
confidence or reasoning field is collected. Original and perturbation prompts are
acquired once and cached. Corum and all baselines consume those identical cached
observations; aggregation itself makes no LLM call.

The candidate Corum system is the accepted-base, legacy no-pair power-likelihood fusion.
For each target repository, reviewer calibration and dependence are fit only from the
original-prompt records belonging to the other repositories. Pair likelihoods are empty.
The literals remain `prior_strength=1.5`, `dependence_shrinkage=0.25`,
`minimum_overlap=10`, `lineage_cap=1.0`, `prior_pass=0.5`, `posterior_draws=512`,
`credible_mass=0.95`, and `chunk_size=4096`. Repository is the NFC-normalized value of the
Verified Parquet `repo` field. Its fusion seed is exactly
`int.from_bytes(hashlib.sha256(b"corum:daily-use:v1\x00" +
repository.encode("utf-8")).digest()[:8], "big", signed=False)`. The locked config records the exact Python, NumPy, and Corum
versions; NumPy `Generator(PCG64(seed))` behavior is required, and cases and reviewers are
always Unicode-code-point sorted before arrays are built.

The experimental full-coverage readout is prospective and exists only in the independent
judge. `non_abstain` counts only records with `state=VALID` and observation `PASS` or
`FAIL`; other states have no observation and contribute no evidence. The posterior still
uses every `VALID` observation, including `ABSTAIN`. Fewer than two non-abstentions returns
`DEFER`; otherwise posterior mean greater than `0.5` returns `PASS`, less than `0.5`
returns `FAIL`, and exactly `0.5` returns `DEFER`. The readout never calls the production
`DecisionPolicy` and never uses the credible interval to break a tie. It does not change
production behavior or retroactively alter JudgeBench.

Corum must beat both frozen voting practices, so no result-dependent baseline selection
is needed:

1. **ordinary majority**: `PASS=+1`, `FAIL=-1`, `ABSTAIN=0`; the sign of the sum is the
   action and zero is `DEFER`;
2. **leave-one-repository-out reliability-weighted vote**: for each target repository and
   reviewer, use only other-repository, original-form rows with `state=VALID` and
   observation in `{PASS, FAIL}`. A row is correct exactly when observation equals truth.
   With `n` eligible rows and `c` correct rows, compute `a=(c+1)/(n+2)` and
   `w=math.log(a/(1-a))`; thus `n=0` gives `w=0` and accuracy below one half produces a
   negative weight. For the target row take the sign of the reviewer-ID-sorted
   `math.fsum(w * observation_sign)`; non-valid and `ABSTAIN` rows contribute exact zero,
   and an exact zero sum is `DEFER`.

The judge also reports each single reviewer and a leave-one-repository-out majority-class
predictor as diagnostics. They are not gate comparators and cannot rescue a failed gate.

## Dataset and candidate registry

The task manifest is exactly the `test` split of `SWE-bench/SWE-bench_Verified` at Git
revision `78f471bf655a3137b2e8a75af1501690ec009ec3`. Its sole Parquet LFS object has SHA-256
`030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25` and size
`6,304,616` bytes. It must contain exactly 500 rows, 500 unique `instance_id` values, and
12 repositories. Sorting the IDs as Unicode strings and hashing the UTF-8 bytes of each
`instance_id + LF` yields
`7e094f04bef443937b420faa59f950d81b28394b9da9d7562da70f02d79a59c2`.

The official harness is pinned to `SWE-bench/SWE-bench` commit
`7a21e05772954cc81471ae19d56f436cecf43c54`. The later locked registry must additionally
record every materialized image digest. Before reviewer acquisition, all 500 registered
gold patches must run once on those exact images and produce complete reports with
`resolved=true`, no environment-tier infrastructure signature, and a sealed aggregate
report/log manifest digest. Any miss, infrastructure failure, unpinned image, or
non-reproducible result is a failed pre-acquisition prerequisite and leaves the task
`BLOCKED`; it consumes no attempt. After infrastructure correction, all 500 gold checks
and the complete environment manifest must be regenerated and independently re-reviewed
before the judge milestone or reviewer acquisition.

The public `SWE-bench/experiments` metadata tree at commit
`1faa91cade0562ba62b66c1c99e71f7b72d96f13` does not contain its advertised prediction
JSON files, so it cannot currently supply an outcome-blind 500-patch candidate. Task 6D
therefore remains evidence-blocked until an outcome-blind candidate commitment passes one
of two routes:

1. **fresh generation**: after this documentation commit, one fully registered generator
   job produces a patch for every ID while network access, gold/test patches, harness
   output, fixing history, and leaderboard material are unavailable; or
2. **complete public inventory**: a pinned source exposes every eligible prediction file,
   the config records the complete metadata-only inventory, and the mechanically eligible
   file with lexicographically smallest `(source_commit, relative_path, sha256)` is chosen
   without opening any score, report, test output, trajectory, or patch content.

The complete inventory or fresh-job manifest and its independent oracle-isolation audit
must be committed in the locked config before reviewer acquisition. A provider-supplied
single file, a curated subset, a result-linked source, or a source whose complete inventory
cannot be verified is not outcome-blind and leaves Task 6D `BLOCKED`.

The selected predictions JSONL pins source repository, commit, relative path, Git blob
when applicable, byte size, SHA-256, single `model_name_or_path`, and provenance. It has
exactly one non-empty `model_patch` for every registered ID and no result, report, test
output, gold patch, or oracle field. At oracle join, byte-identical or canonical-diff-
identical copies of a gold patch, test patch, or fixing patch make the consumed attempt
`INVALID`; the candidate cannot then be replaced. Zero, incomplete, duplicate, or
result-derived candidates fail closed before reviewer calls and do not consume attempt 0.

Canonical diff text is formed only by replacing CRLF and lone CR with LF, removing ASCII
space and tab at each line end, removing leading/trailing empty lines, then joining with LF
and one final LF; no hunk, path, or content line is reordered or otherwise normalized.

SWE-bench harness code is MIT, but the Verified dataset and experiments metadata do not
establish a blanket license for copied issue, repository, candidate, or output data.
All raw material remains local research data pending a provenance and redistribution
review. Corum must not label the entire benchmark corpus MIT.

## Artifact contract and seal

All immutable evidence files live below the read-only root named by
`CORUM_DAILY_USE_ROOT`; the formal relative paths are `dataset.parquet`,
`candidate.jsonl`, `contexts.jsonl`, `reviews.jsonl`, `harness/oracle.jsonl`, and
`audit/acquisition.json`. The auditor-owned append-only attempt log is a separate file
named by `CORUM_DAILY_USE_ATTEMPT_LOG` and must not be below that root. JSONL is UTF-8
without BOM, uses LF including one final LF, and is
sorted by the compound key registered below. Every line is a JSON object serialized with
lexicographically sorted keys, `ensure_ascii=false`, and separators `(",", ":")`.
Identifier fields are Unicode NFC; free-form issue, patch, code, and response-derived text
is never normalized. Extra keys, duplicate keys, non-canonical bytes, non-finite numbers,
or a count/key mismatch are `INVALID`.

The schemas and sort/unique keys are:

| File | Rows and unique sort key | Exact allowed top-level keys |
|---|---|---|
| `candidate.jsonl` | 500; `(instance_id)` | `instance_id`, `model_name_or_path`, `model_patch` |
| `contexts.jsonl` | 1,000; `(instance_id, form)` | `instance_id`, `form`, `issue_text`, `candidate_blocks`, `context_blocks`, `rendered_prompt_sha256`, `rendered_input_bytes` |
| `reviews.jsonl` | 3,000; `(instance_id, form, reviewer_id)` | `instance_id`, `form`, `reviewer_id`, `model_id`, `lineage`, `state`, `observation`, `attempts`, `prompt_sha256`, `context_sha256`, `stop_reason`, `acquired_at_utc` |
| `harness/oracle.jsonl` | 500; `(instance_id)` | `instance_id`, `state`, `resolved`, `report_sha256`, `test_output_sha256`, `instance_log_sha256`, `image_digest`, `failure_class` |

Each context block has exactly `block_id`, `path`, `bytes_sha256`, and `text`; block IDs
and paths are unique within their array. Each review `attempts` array has one or two rows,
each with exactly `attempt_index`, `state`, `input_tokens`, `output_tokens`,
`response_sha256`, and `stop_reason`. The top-level final state is one of the existing
Corum execution states `VALID`, `TIMEOUT`, `INVALID`, or `REFUSAL`; `NOT_CALLED` is
forbidden. `VALID` requires an observation in `{PASS, FAIL, ABSTAIN}` and every other
state requires `observation=null`. A timeout, refusal, or schema-invalid provider response
is therefore a real, counted model outcome, not a malformed ledger. A missing expected
row, malformed row, identity/hash mismatch, or undeclared call invalidates the attempt.

The locked config embeds Draft 2020-12 JSON Schemas for candidate, context block,
context row, call attempt, review row, acquisition audit and nested clearances, oracle row,
seal, and attempt-log row. Every property is required unless explicitly nullable,
`additionalProperties=false` applies at every object level, strings/arrays/integers have
registered length and byte caps, forms are exactly `{A, B}`, hashes match
`^[0-9a-f]{64}$`, timestamps match whole-second RFC 3339 UTC, and every enum is closed.
Synthetic tests cover every enum and boundary; prose is not a substitute for those schemas.

The judge enforces these cross-file relations before opening the oracle: dataset,
candidate, and context ID sets are identical; every ID has exactly A and B contexts; every
context has exactly the three config reviewers; reviewer model/lineage values equal the
panel; review prompt/context hashes equal judge-reconstructed bytes; and every registered
candidate/context/review/audit/seal digest matches. Attempt indices start at zero and are
contiguous. A first `TIMEOUT` or `INVALID` call is retried exactly once; a first `VALID` or
`REFUSAL` is never retried. The top-level state equals the last attempt state, and only a
final `VALID` may have an observation. After oracle opening, its ID set must also equal the
same 500 IDs; any mismatch is a consumed `INVALID`.

`audit/acquisition.json` has exactly `schema_version`, `gate_id`, `config_sha256`,
`candidate_sha256`, `contexts_sha256`, `reviews_sha256`, `reviewer_attempt_ids_sha256`,
`privacy_clearance`, `provenance_clearance`, `oracle_isolation_attestation`, `auditor`, and
`sealed_at_utc`. The two clearance objects record reviewer-provider terms, public-data
processing authority, secret/credential/PII scan tool and digest, rejected path classes,
and independent reviewer identity. These facts require an independent acquisition audit;
the judge verifies their schema, identity, and sealed digests but must not claim that a
hash alone proves no human or provider saw oracle data.

The attempt log uses the same canonical JSON encoding but preserves physical append order.
An initially empty log is exactly zero bytes. Every row has exactly `schema_version`,
`gate_id`, `gate_version`, `attempt_uuid`, `event_index`, `event`, `code_commit`,
`config_sha256`, `seal_sha256`, `candidate_sha256`, `panel_sha256`, `holdout_sha256`,
`reviewer_ledger_sha256`, `verdict`, `output_sha256`, `event_at_utc`, `auditor`,
`previous_record_sha256`, and `record_sha256`. Hashes are 64 lowercase hexadecimal
characters and UTC timestamps are RFC 3339 with `Z` and whole seconds.
`(attempt_uuid, event_index)` is unique.

`START` has `event_index=0` and null verdict/output; `FINAL` has `event_index=1`, a verdict
in `{PASS, FAIL, INCONCLUSIVE, INVALID}`, and a non-null output hash. All binding fields
must be identical across the two events. `previous_record_sha256` equals the preceding
physical row's `record_sha256`, or 64 zeroes for the first row. `record_sha256` is SHA-256
of that row's canonical JSON with the `record_sha256` key omitted, followed by LF. The
named auditor holds an exclusive file lock, verifies the entire chain, appends exactly one
canonical row, flushes and fsyncs before releasing the lock. The judge is read-only: it
verifies the chain and requires exactly one matching `START` with no `FINAL`. The seal
binds the pre-`START` log-head digest; the auditor appends `FINAL` after captured output.

Replay identity is independent of UUID. Before appending `START`, both auditor and judge
must reject any earlier `START` or `FINAL` with the same `(gate_id, gate_version,
config_sha256, candidate_sha256, panel_sha256, holdout_sha256,
reviewer_ledger_sha256)` tuple, even if code, seal, or UUID differs. While holding the
exclusive lock, the auditor must also prove that the current chain head equals the seal's
`previous_attempt_log_head_sha256`; the new `START.previous_record_sha256` must equal that
same value. A mismatch is `INVALID_PREJOIN` and cannot be bypassed with another UUID.

`configs/daily-use-v1.json` pins the exact schema version, candidate commitment and
complete selection inventory, three model IDs/revisions/providers/lineages, prompt bytes,
both template hashes, retrieval implementation/config digest, path allow/deny rules,
per-file and total byte caps, stable ordering/truncation behavior, context-package hashes,
container digests, runtime versions, and every judge literal before reviewer acquisition.
The judge milestone cannot be committed while any of those fields is unknown.

The model panel is also outcome-blind. The locked config first enumerates the complete
provider/endpoint universe, which for v1 is limited to existing endpoints that create no
paid charge. The operator then supplies a timestamped, complete inventory of every model
available through exactly those endpoints.
Eligibility is limited to an immutable model revision, sufficient registered context
window, deterministic structured-output support, permitted data-processing terms, and a
declared base lineage. Sort eligible entries by `(provider, model_id, revision)` and choose
the lexicographically first triple whose lineages are all distinct. No coding score,
candidate content, case-level vote, or Task 6D outcome may affect eligibility or order.
An incomplete endpoint inventory or an unauditable lineage leaves the gate `BLOCKED`.
Any future paid inference requires a separate explicit owner budget amendment to the
authoritative documents before the endpoint universe may change; this SDD grants none.

After the 3,000 review rows are acquired but before the harness oracle is generated or
opened, independent review verifies the acquisition audit. Then
`configs/daily-use-v1-seal.json` commits the exact byte size and SHA-256 of
`reviews.jsonl` and `audit/acquisition.json`, the retry-inclusive non-zero token total,
the 3,000-key digest, an attempt UUID, and the previous auditor-owned attempt-log head
digest under the exact field `previous_attempt_log_head_sha256`.
Only that hash-only seal is committed as `data: seal daily use reviewer ledger`; raw data
stays outside Git. Changing the candidate, panel, prompts, contexts, semantic observations,
or token ledger after this seal requires a new SDD/gate version and cannot reuse Task 6D.

## Blind prompt package and perturbation

Before any reviewer call, the locked config freezes all three exact model IDs and
revisions, distinct lineage declarations, prompt-template bytes and SHA-256, decoding
settings, context limit, timeout, retry limit, provider/token-accounting fields, and all
1,000 context packages. A package may contain only the issue text, candidate diff, and
repository files at the base commit selected by the registered outcome-blind retrieval
implementation. That implementation's exact digest, byte limits, stable path order,
truncation rule, and fail-closed behavior live in the config. It may not read an oracle
artifact named in the non-goals. Both rendered forms must fit every reviewer's context
limit in full; truncation by a provider is `INVALID`.

Every case has one mechanically equivalent order perturbation. Forms A and B use identical
instruction, issue, schema, delimiters, and all other non-block bytes. B reverses only the
sequence of uniquely identified complete repository-context blocks and complete
candidate-diff file blocks. It changes no byte inside a block. The judge reconstructs both
renderings, verifies that their only serialized difference is the registered permutation,
and rejects a content change. For sorted zero-based `instance_id` index `i` and sorted
zero-based reviewer index `j`, A is acquired first exactly when `(i+j) % 2 == 0`; otherwise
B is first. The config pins both full rendered-package manifest hashes.

Reviewer acquisition produces exactly 3 reviewers x 500 cases x 2 prompt forms = 3,000
final records. Each record binds the registered identity, execution state, optional
normalized observation, every attempt's input/output tokens, prompt/context/raw-response
hashes, stop reason, and acquisition timestamp. A first `TIMEOUT` or `INVALID` transport/
schema attempt is retried exactly once using the same model and exact prompt; `VALID` and
`REFUSAL` are never retried, and both attempts' tokens count. The three registered reviewer
calls and that bounded retry are the only
model calls. Any model verifier, fallback, router, summarizer, repairer, or provider-side
call not represented in `attempts` is `INVALID`; deterministic local parsing spends no
tokens. No manual per-case intervention or outcome-dependent retry is allowed. Raw
responses and secrets remain outside Git.

The reviewer ledger and token totals are sealed before the oracle outcome file is joined.
Missing or malformed records, mismatched hashes, unexpected model IDs, hidden calls,
non-finite values, or any indication that an oracle field reached a reviewer makes the
pre-join validation `INVALID_PREJOIN`; it cannot be turned into a scientific result.

## Harness truth mapping

After the reviewer-ledger seal, each candidate runs exactly once through the pinned
harness; no selective task retry is allowed. For classification, a missing log file is
empty bytes; otherwise decode its bytes as UTF-8 with `errors="replace"`, then set
`text = decoded_test_output + "\n---CORUM-LOG-BOUNDARY---\n" + decoded_instance_log`.
Every regex predicate is exactly
`re.search(pattern, text, flags=re.MULTILINE) is not None`; literals use `literal in text`.
Oracle state is derived from the pinned report and that exact text in this precedence:

1. `ENVIRONMENT_FAILURE` if concatenated test output plus instance log matches any of
   these exact Python `re.MULTILINE` patterns:
   `Failed to launch|Failed to connect to the bus`,
   `cannot open display|Missing X server|unable to open X display`,
   `Cannot allocate memory|OutOfMemoryError|^Killed$`,
   `Error response from daemon|Cannot connect to the Docker daemon`, or
   `Could not resolve host|Temporary failure in name resolution`;
2. `COMPLETED` when the per-instance report is structurally valid JSON containing exactly the
   registered instance key and the pinned boolean fields `patch_is_None`, `patch_exists`,
   `patch_successfully_applied`, `resolved`, and `infra_failure`, plus only the optional
   string `infra_failure_reason`, and `infra_failure=false`;
3. `CANDIDATE_FAILURE` when no complete report exists and concatenated logs contain one
   of the exact literals `>>>>> Patch Apply Failed`, `>>>>> Tests Errored`, or
   `>>>>> Tests Timed Out`, or match one of the exact `re.MULTILINE` patterns
   `Cannot find module|MODULE_NOT_FOUND|ModuleNotFoundError`,
   `no tests ran|collected 0 items`, or `Timeout error: \d+ seconds exceeded`;
4. `UNCLASSIFIED_ERROR` for every other missing, truncated, contradictory, or unreadable
   report/log state.

Truth is `PASS` only for `COMPLETED` with `resolved=true`. `COMPLETED` with
`resolved=false` and every `CANDIDATE_FAILURE` are truth `FAIL` because the submitted
patch did not satisfy the executable contract. Any `ENVIRONMENT_FAILURE` or
`UNCLASSIFIED_ERROR` consumes the post-join attempt as `INVALID`; it is never relabeled as
a bad patch and no per-case rerun is permitted. The judge independently reproduces the
pinned classifier, report schema, marker precedence, and every report/log/image digest.
The locked config pins Git blob OIDs and SHA-256 for the pinned commit's
`swebench/harness/constants/__init__.py`, `grading.py`, `infra_failure.py`, and
`reporting.py`, plus the complete regex/literal arrays above and versioned positive,
candidate-failure, environment-failure, conflict-precedence, and unknown-error fixtures.
No unregistered marker may be classified manually; environment patterns always win,
then a complete report, then candidate-failure patterns, then unclassified error.
Fixtures must prove search-not-match behavior and that a token split across the registered
log-boundary delimiter cannot match.

## Metrics and gate

All 500 original cases are assigned. A recorded non-valid reviewer call contributes no
evidence; the resulting system action is scored normally. Every `DEFER` action counts as
incorrect and uncovered. Coverage is the fraction with a final `PASS` or `FAIL` action.
False-safe incidence is the fraction of all 500 cases whose action is `PASS` while oracle
truth is `FAIL`; conditional false-safe risk among `PASS` actions is diagnostic and is
`null` when a method has no `PASS` action.

For stability, compare each original action with its equivalent perturbation action. Any
change among `PASS`, `FAIL`, and `DEFER` is a flip. The flip rate is the share of 500 pairs
that change. The shared token total is the integer sum of all `input_tokens` and
`output_tokens` in every registered attempt, including the bounded retries, and must be
strictly positive. All methods reference the same 3,000 ledger keys and aggregation has
zero model calls, so token equality and ratio exactly `1.0` are integrity invariants. The
registered `<=1.20` ceiling therefore checks no hidden extra call; results must not present
it as an empirically discovered cost advantage.

Accuracy, coverage, flip rate, and false-safe incidence are 500-case micro-averages. Use
10,000 deterministic paired repository-stratified bootstrap draws conditional on the
already fitted out-of-fold actions; cross-fits are not refit inside a draw. Set
`seed=20260831` and `rng=numpy.random.Generator(numpy.random.PCG64(seed))`. Sort NFC
repository strings and, within each, `instance_id` strings by Unicode code point. On every
draw and for every repository in that order, draw exactly its observed `n_r` integer case
indices with replacement using one `rng.integers(0, n_r, size=n_r)` call. Concatenate the
12 samples into exactly 500 paired rows. The same row indices compute every method and
metric operand in that draw.

Use `numpy.quantile` at `0.025` and `0.975` with `method="linear"`; CI lower-bound tests
are strict `>0` on unrounded values. Point estimates use the original 500 rows, not
bootstrap means. Every operand must be finite before sampling. Undefined diagnostic
denominators stay `null`; they never become a favorable zero. Report raw paired counts,
two-sided 95% intervals, and every per-repository slice. These intervals are conditional
on this fixed 12-repository suite and do not claim a repository-superpopulation result.

Task 6D is `PASS` only when every condition below holds against **each** voting baseline:

1. `accuracy_Corum - accuracy_baseline >= 0.05` on original cases, and the paired 95%
   bootstrap lower bound for that difference is strictly above zero;
2. shared total-token ratio is exactly `1.0` and therefore at most `1.20`;
3. `flip_rate_Corum <= 0.70 * flip_rate_baseline`, using unrounded values, and the paired
   95% lower bound for `flip_rate_baseline - flip_rate_Corum` is strictly above zero;
4. Corum coverage is at least `90%`; `DEFER` never counts as success;
5. Corum false-safe incidence is no greater than the baseline's; nullable conditional
   false-safe risk remains diagnostic only;
6. every integrity, blindness, provenance, deterministic replay, and finite-value check
   passes.

After integrity succeeds, verdict precedence is deterministic. First, any missed accuracy
or stability point threshold, coverage floor, token equality/ceiling, or false-safe
guardrail yields `FAIL`. Otherwise, a zero flip rate for either baseline yields
`INCONCLUSIVE` because no 30% lead can be demonstrated. Otherwise, any accuracy or
stability confidence lower bound that is not strictly above zero yields `INCONCLUSIVE`.
Only the remaining state is `PASS`. `FAIL` and `INCONCLUSIVE` have the same roadmap
consequence: stop the current consensus path rather than add components or rerun the same
data.

State and attempt precedence is exact:

1. missing candidate/panel/config/clearance/seal files before the formal switch is
   `BLOCKED`; no attempt record is written;
2. a mandatory no-oracle preflight runs before any `START` record. A byte/schema/seal
   failure is `INVALID_PREJOIN`, writes no scientific metric or attempt event, and consumes
   nothing; only a transfer/parser fix that leaves every sealed semantic byte unchanged
   may retry after independent review;
3. after a clean preflight and immediately before generating the oracle, the auditor
   requires no prior matching attempt UUID and atomically appends the `START` record
   binding gate, code, config, seal, candidate, panel, and holdout digests; the read-only
   judge verifies it;
4. from `START`, `PASS`, `FAIL`, `INCONCLUSIVE`, or any integrity/oracle/replay error is
   a consumed final attempt. A post-join error is recorded as `INVALID`, emits no partial
   scientific metrics, and cannot rerun this candidate/panel/ledger/holdout combination;
5. any later integrity repair requires a new SDD, judge/config version, result path, fresh
   sealed reviewer ledger, and independent review. If scientific operands were exposed,
   the same candidate/panel/holdout cannot support the replacement claim.

The local judge can verify but never write the attempt log. Append-only retention and
oracle isolation ultimately depend on the named independent auditor. The result must say
so rather than claiming cryptographic proof it does not possess.

## TDD evidence and execution order

1. Commit and independently review this prospective documentation before writing the
   judge or acquiring formal Task 6D benchmark reviewer outputs.
2. Satisfy the pre-acquisition prerequisites: commit one complete outcome-blind candidate
   commitment, the mechanically selected exact three-model panel, prompt/retrieval/context
   manifests, privacy/provenance clearances, container digests, and the 500-case gold-patch
   environment check. Until these exist, remain `BLOCKED` and do not write a placeholder
   config or acquire the formal 3,000-row reviewer ledger. The already consumed
   preregistered synthetic transport/schema smoke is the sole exception; it is closed and
   authorizes no further call, retry, repair, model replacement, or reordering.
3. RED: synthetic contract tests must first fail because the registry validator, blind
   join, baselines, metrics, bootstrap, and verdict do not exist.
4. GREEN: implement only `configs/daily-use-v1.json` and the self-contained independent
   judge in `tests/test_daily_use_value.py`. Ordinary test runs exercise synthetic fixtures
   only and skip the external gate unless `CORUM_RUN_DAILY_USE_V1=1`.
5. Verify deterministic replay and at least 80% branch coverage of the judge file with:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_daily_use_value.py --cov=test_daily_use_value --cov-branch --cov-report=term-missing --cov-fail-under=80 -q
   ```

   Then run the ordinary suite exactly as
   `.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_core_value.py
   --ignore=tests/test_pair_value.py`; those two already-consumed failing judges remain
   immutable and are not ordinary regression tests. Run Ruff, mypy, and `git diff --check`.
   Obtain fresh independent review and commit exactly
   `test: lock daily use value gate` without acquiring reviews or opening the candidate
   oracle.
6. Acquire only the 3,000 reviewer rows, complete the external acquisition audit, and
   commit `configs/daily-use-v1-seal.json` exactly as
   `data: seal daily use reviewer ledger`. Obtain fresh independent seal review. Do not
   generate or open `harness/oracle.jsonl` before this commit. Then run the no-oracle
   preflight exactly once per unchanged byte transfer:

   ```powershell
   $env:CORUM_PREFLIGHT_DAILY_USE_V1 = "1"
   $env:CORUM_DAILY_USE_ROOT = ".corum-work/daily-use-v1"
   $env:CORUM_DAILY_USE_ATTEMPT_LOG = ".corum-work/daily-use-attempts.jsonl"
   .venv\Scripts\python.exe -m pytest tests/test_daily_use_value.py -q -s -k external_preflight
   ```

   It must reject an existing oracle and never read or write an attempt event. Clear the
   preflight switch after success.
7. Have the named auditor atomically append the matching `START` event before any
   candidate oracle is generated. Then generate the 500 oracle rows once with the pinned
   harness, make the evidence root read-only, and execute only:

   ```powershell
   $env:CORUM_RUN_DAILY_USE_V1 = "1"
   $env:CORUM_DAILY_USE_ROOT = ".corum-work/daily-use-v1"
   $env:CORUM_DAILY_USE_ATTEMPT_LOG = ".corum-work/daily-use-attempts.jsonl"
   .venv\Scripts\python.exe -m pytest tests/test_daily_use_value.py -q -s
   ```

   Capture exact aggregate output and wall time, have the auditor append `FINAL`, commit
   only registered aggregate results, and never tune or rerun this
   candidate/panel/ledger/holdout combination.

## Review and completion

Documentation review must challenge whether the test actually distinguishes Corum from
voting, whether token accounting is symmetric, whether abstention can game accuracy,
whether any oracle information can leak into prompts, and whether the candidate-patch
choice is outcome-blind. Judge review must independently recompute registry hashes,
actions, cross-fits, paired operands, bootstrap inputs, and verdict from synthetic data
without trusting production metric helpers.

Three independent read-only registration reviews covered statistics, repository
governance/privacy, and judge implementability. Their first pass found outcome-selectable
candidate/panel inputs, ambiguous model-failure scoring, incompletely frozen bootstrap and
seed behavior, unverifiable artifacts, a mixed wording/order perturbation, oracle-state
ambiguity, and replayable attempt IDs. The contract now uses complete prospective
inventories, explicit execution states, exact schemas and cross-file relations, fixed
PCG64 sampling, order-only perturbation, pinned oracle classification, a pre-oracle
reviewer-ledger seal, and semantic hash-chain replay protection. All three re-reviews
returned `APPROVE` with no open Critical or Important finding.

At the current checkpoint, external validation and judge implementation remain blocked:
there is no compliant outcome-blind 500-patch candidate commitment, exact eligible
three-model panel, privacy/provenance clearance, reviewer/token ledger, or harness outcome
registry. This status is an honest prerequisite, not an invitation to write placeholders,
replace real data with simulation, or start product work.

## Pre-acquisition panel-smoke outcome

On 2026-08-29 the one preregistered, zero-cost synthetic transport/schema smoke was
consumed under an external START/FINAL hash-chain audit. The formal outcome is
`BLOCKED / result_mismatch`; this smoke must not be rerun as v2.1, v2.2, or v3. Its
immutable runner artifact records exactly three calls, zero retries, zero benchmark cases,
zero formal reviewer rows, and one `HTTP 200` plus exact enum-schema-valid record for each
of the three preregistered digests. It persists no observation value, response body/hash,
token count, or duration diagnostic.

The external validator rejected the runner's internal `PASS` because PowerShell parameter
binding coerced the `$null` argument passed to `[AllowNull()][string]$FailureKind` into
`""` before JSON serialization, while the registered result contract required JSON
`null`. The rejection is correct and is not a parser relaxation opportunity. The `3/3`
mechanical records prove only that those exact one-time calls crossed the registered
transport/schema boundary; they prove neither panel eligibility nor model accuracy,
stability, independence, context capacity, or quality. No model was selected, replaced,
or reordered from the outcome. Candidate generation, the judge, context packages, the
3,000-row reviewer ledger, and the harness oracle were not started, so formal Task 6D
attempt 0 remains unconsumed.
