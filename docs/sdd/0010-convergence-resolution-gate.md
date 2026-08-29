# SDD: Prospective Full-Coverage Convergence/Resolution Gate

- Status: attempt 0 final — `FAIL`; reason codes: FAIL_ACCURACY_POINT_WEIGHTED, FAIL_COVERAGE_FLOOR, FAIL_COVERAGE_GAP_WEIGHTED, FAIL_DISPERSION_POINT_ORDINARY, FAIL_DISPERSION_POINT_WEIGHTED, FAIL_SCENARIO_ACCURACY, FAIL_SCENARIO_COVERAGE, FAIL_SCENARIO_FALSE_SAFE
- Accepted base: `b34e089`
- Exact documentation commit: `docs: register convergence resolution gate`
- Exact judge commit: `test: lock convergence resolution gate`
- Exact attempt-0 result commit: `docs: record convergence resolution gate result`
- Allowed documentation files: `AGENTS.md`, `docs/specs/corum-mvp-design.md`,
  `docs/plans/corum-mvp.md`, and this SDD
- Allowed judge files: `configs/convergence-resolution-v1.json` and
  `tests/test_convergence_resolution_value.py`
- Allowed attempt-0 result files: the four documentation files above and
  `docs/results/task-6e-convergence-resolution-attempt-0.{json,md,txt}`
- Artifact ownership: the judge writes only the registered TXT attempt ledger and embeds
  its deterministic result object in FINAL; after termination, the recorder exclusively
  publishes registered JSON/MD/status from TXT without reinterpretation or rerun

## Authority and outcome

On 2026-08-29 the owner instructed Corum to test prospectively whether consensus can
converge better than voting on a substantial sample without purchasing attractive metrics
through `DEFER`. Task 6E is one narrow, owner-approved exception while Task 6D remains
formally `BLOCKED`, its panel smoke remains consumed, and its formal attempt 0 remains
unconsumed. It is a zero-network synthetic qualification of exactly one frozen candidate
before any replacement acquisition SDD may be prepared.

Task 6E does not reopen, repair, overwrite, or rerun Task 6A, Task 6B, Task 6C, the
consumed Task 6D smoke, or formal Task 6D attempt 0. It adds no statistical component,
changes no production behavior, and cannot unlock Task 7 or product work. The still-frozen
ultimate 500-case Daily Use Gate requires at least `+5pp` accuracy against ordinary and
reliability-weighted voting, shared token ratio exactly `1.00`, at least `90%` coverage,
at least `30%` fewer harmless-perturbation flips, no worse false-safe incidence, and the
registered paired confidence evidence. This synthetic gate neither weakens nor satisfies
that real-patch contract.

A synthetic `PASS` authorizes only preparation and independent review of a new acquisition
SDD/version that fixes the serialization contract and re-establishes panel eligibility
before the still-frozen 500-case Daily Use Gate. It is not a Daily Use `PASS`, consumes no
Task 6D attempt, and authorizes no model call, acquisition, product work, component, or
Task 7 work. `FAIL`, `INCONCLUSIVE`, or post-start `INVALID` stops the current consensus
path: report root-cause slices and return the continue/stop decision to the owner.

Task 6E is only a cheap synthetic kill/prequalification test against voting, not evidence
of state-of-practice superiority or novelty. If and only if it passes, the replacement
real-data acquisition SDD must prospectively require human-labeled repository/code-review
ground truth, cross-vendor reviewer strata, and stronger comparators including at minimum
the strongest single judge, Dawid--Skene-class EM aggregation, and registered conformal
risk-control/cascade methods. It must not treat synthetic dispersion as real stability.
Node-level provenance and active experiment design remain future hypotheses, not authorized
components.

## Candidate and non-goals

The sole candidate is the already registered Task 6D experimental full-coverage readout
over accepted-base legacy no-pair power fusion:

1. fit the existing singleton calibration and dependence estimators on fit rows only;
2. keep the pair registry empty;
3. use the existing posterior draws and their mean pass probability `p`;
4. count only `VALID` observations in `{PASS, FAIL}` as directional;
5. with at least two directional observations, return `PASS` when `p > 0.5`, `FAIL` when
   `p < 0.5`, and `DEFER` when `p == 0.5` exactly;
6. with fewer than two directional observations, return `DEFER`;
7. never call production `DecisionPolicy` and never use interval bounds to decide.

There is no policy-selection split because the candidate has no selected parameter or
threshold. As in the registered Task 6D readout, every `VALID` observation, including
`ABSTAIN`, enters the existing fusion likelihood; the directional count controls only the
minimum-two readout rule. For diagnostics, when accepted-base batch fusion returns no
posterior because all three reviews are non-`VALID`, candidate probability is exactly
`0.5`; its action remains `DEFER` under the fewer-than-two-directional rule. If at least one
`VALID` observation contributes, diagnostics use the finite accepted-base posterior mean
even when the action remains `DEFER`. All-invalid and mixed-valid/non-valid cases remain in
all full-case probability metrics. Do not add pair-power mixing, naive-power mixing,
adaptive shrinkage, a learned threshold, a production module, a cascade, UI, repository
reader, provider adapter, or model call. `src/corum` is frozen. Existing statistical review found
no fresh evidence for those additions, and Task 6B's rejected pair path is mismatched to
Task 6D's target of three distinct lineages.

No network, external dataset, LLM, paid API, benchmark case, or Task 6D artifact may be
read. No config, judge, result, or placeholder is permitted in this documentation
milestone. After this documentation is committed and independently reviewed, a separate
TDD milestone may create only the two registered judge files. Formal results may be
created only after that separate milestone and only at the registered result paths.

## Frozen sample and execution

The formal holdout contains exactly:

`6 scenarios * 40 replicate blocks * 10,000 test cases = 2,400,000 test cases`.

Each block also contains exactly 8,000 independent fit cases. Replicate indices are
exactly `0..39`. Formal execution processes blocks sequentially and may not drop, replace,
top up, or rerun a block. Every A-form test case is assigned. There is no policy-selection
split.

For each scenario-replicate block, the judge constructs the literal `Scenario` and makes
exactly one accepted-base call:

```python
simulate_experiment(
    scenario,
    n_calibration=8_000,
    n_test=10_000,
    seed=enumerated_simulation_seed,
)
```

At accepted base, this call evaluates `SeedSequence(seed).spawn(2)` once, converts each
child through `int(child.generate_state(1, dtype=np.uint64)[0])`, associates child index
`0` with the calibration/fit phase and child index `1` with the test phase, then passes the
two uint64 values to the existing internal panel simulator. That operation and association
are frozen. Direct `simulate_panel` calls, repeated use of the parent seed, offsets,
additional hashes, reversed children, or any other fit/test seed split are forbidden.

Every scenario contains exactly three reviewers. For a declared semantic accuracy `a`
and abstention probability `s`, likelihood rows are symmetric: truth `PASS` uses
`(a, 1-a-s, s)` and truth `FAIL` uses `(1-a-s, a, s)` over semantic observations
`(PASS, FAIL, ABSTAIN)`. Timeout and invalid rates are separate execution probabilities.
The tuple notation below is `(accuracy, abstain, timeout_rate, invalid_rate)`.

The later config must contain the literal calibration and test phase objects and the
SHA-256 of canonical JSON for those complete objects. It must not derive them from
`builtin_scenarios()`.

Every config reviewer object has exactly `reviewer_id`, `lineage`, `accuracy`, `abstain`,
`timeout_rate`, and `invalid_rate`. Every phase object has exactly `prior_pass`,
`difficulty_rate`, `informative_missingness`, `lineage_error_correlation`, and `reviewers`.
Unmentioned difficulty, informative-missingness, timeout, invalid, and correlation values
below are literal `0.0` or `{}`, not omitted defaults. Construction supplies fixed
non-statistical reviewer metadata `vendor="simulated"`, `family="general"`, `cost=1.0`
and `adversarial_reviewer_id=None`; these values cannot vary by scenario or phase.

### Scenario 1: `independent-balanced-v1`

- calibration and test: `prior_pass=0.55`; no lineage correlation, missingness, or
  difficulty;
- `ind-a`, lineage `ind-a`: `(0.83, 0.03, 0, 0)`;
- `ind-b`, lineage `ind-b`: `(0.76, 0.05, 0, 0)`;
- `ind-c`, lineage `ind-c`: `(0.69, 0.07, 0, 0)`.

### Scenario 2: `clone-pressure-v1`

- calibration and test: `prior_pass=0.50`; no missingness or difficulty;
- lineage error correlation: `clone=0.88`;
- `clone-a`, lineage `clone`: `(0.72, 0.04, 0, 0)`;
- `clone-b`, lineage `clone`: `(0.74, 0.04, 0, 0)`;
- `clone-strong`, lineage `clone-independent`: `(0.86, 0.03, 0, 0)`.

### Scenario 3: `majority-trap-v1`

- calibration and test: `prior_pass=0.50`; no missingness or difficulty;
- lineage error correlation: `trap-weak=0.90`;
- `trap-a`, lineage `trap-weak`: `(0.59, 0.03, 0, 0)`;
- `trap-b`, lineage `trap-weak`: `(0.61, 0.03, 0, 0)`;
- `trap-strong`, lineage `trap-strong`: `(0.90, 0.02, 0, 0)`.

### Scenario 4: `informative-missing-v1`

- calibration: `prior_pass=0.60`, `difficulty_rate=0.30`,
  `informative_missingness=0.20`, no lineage correlation;
- test: `prior_pass=0.48`, `difficulty_rate=0.55`,
  `informative_missingness=0.60`, no lineage correlation;
- `miss-a`, lineage `miss-a`: calibration `(0.80, 0.10, 0.03, 0.02)`; test
  `(0.76, 0.12, 0.05, 0.03)`;
- `miss-b`, lineage `miss-b`: calibration `(0.85, 0.05, 0.05, 0.02)`; test
  `(0.81, 0.07, 0.08, 0.03)`;
- `miss-c`, lineage `miss-c`: calibration `(0.74, 0.18, 0.02, 0.04)`; test
  `(0.68, 0.22, 0.04, 0.05)`.

### Scenario 5: `adversarial-shift-v1`

- calibration: `prior_pass=0.65`; test: `prior_pass=0.40`; no lineage correlation or
  difficulty;
- `shift-a`, lineage `shift-a`: calibration `(0.84, 0.03, 0, 0)`; test
  `(0.74, 0.05, 0.02, 0)`;
- `shift-b`, lineage `shift-b`: calibration `(0.78, 0.04, 0, 0)`; test
  `(0.32, 0.04, 0.02, 0)`;
- `shift-c`, lineage `shift-c`: calibration `(0.76, 0.05, 0, 0)`; test
  `(0.70, 0.06, 0.02, 0)`.

### Scenario 6: `dependence-shift-v1`

- calibration: `prior_pass=0.60`, lineage error correlation `dep-shared=0.20`;
- test: `prior_pass=0.45`, lineage error correlation `dep-shared=0.72`;
- no difficulty;
- `dep-a`, lineage `dep-shared`: calibration `(0.79, 0.05, 0, 0)`; test
  `(0.72, 0.07, 0.02, 0)`;
- `dep-b`, lineage `dep-shared`: calibration `(0.77, 0.05, 0, 0)`; test
  `(0.71, 0.07, 0.02, 0)`;
- `dep-c`, lineage `dep-independent`: calibration `(0.82, 0.04, 0, 0)`; test
  `(0.80, 0.05, 0.02, 0)`.

## Seeds, fusion, and runtime

For every scenario and replicate, derive simulation, fusion, and perturbation seeds
independently. The unsigned 64-bit seed is the first eight SHA-256 digest bytes interpreted
big-endian from the UTF-8 bytes of:

`corum:convergence-resolution:v1\0{scenario_name}\0{replicate_decimal}\0{purpose}`

`purpose` is exactly `simulation`, `fusion`, or `perturbation`; `replicate_decimal` has
the ordinary base-10 representation of its index. Each displayed `\0` is exactly one
literal NUL octet `0x00` in the hashed UTF-8 byte sequence, not the two printable
characters backslash and zero. The later config must enumerate all 720 derived numeric
values, include the SHA-256 of canonical JSON for the complete seed table, and require
formal runtime regeneration to match the table exactly.

The seed-table value is a 240-element array in scenario-name Unicode-code-point order,
then replicate integer order. Each element has exactly `scenario`, `replicate`,
`simulation`, `fusion`, and `perturbation`, for 720 numeric seeds total; the digest is over
that array alone and never includes its digest field.

Fusion literals are exactly:

| Literal | Value |
|---|---:|
| `prior_strength` | `1.5` |
| `dependence_shrinkage` | `0.25` |
| `minimum_overlap` | `10` |
| `lineage_cap` | `1.0` |
| `prior_pass` | `0.5` |
| `posterior_draws` | `512` |
| `credible_mass` | `0.95` |
| `chunk_size` | `4096` |

The later config pins the exact Python, NumPy, and Corum runtime versions. Formal
simulation and bootstrap use NumPy `Generator(PCG64(seed))` under those pinned versions.

## Canonical JSON and config schema

The config and deterministic result are standalone canonical JSON documents. Decode
strict UTF-8 with no BOM and require no terminal newline or trailing byte. Parse with
duplicate-key detection at every object level, normalize every key and string value to
Unicode NFC, reject any string that changes under that normalization, and reject duplicate
keys after NFC. Reject extra or missing keys against the exact schemas below. Booleans are
not numbers. Integers must be mathematical integers within their registered non-negative
ranges. Floating values must be finite binary64 values; `NaN`, infinities, and every JSON
number token whose numeric value is negative zero are forbidden.

After parsing, the original bytes must equal Python `json.dumps` under the runtime version
pinned in the config with `sort_keys=True`, `ensure_ascii=False`,
`separators=(",", ":")`, and `allow_nan=False`, encoded as UTF-8 with no following
newline. Python's exact numeric spelling in that pinned runtime is therefore part of the
contract; alternative but numerically equal spellings are noncanonical. Object keys sort
by Unicode code point. Array order is semantic and never sorted by the serializer.

The config top-level object has exactly these keys:

`accepted_base`, `bootstrap`, `external_switch`, `fusion`, `gate_id`, `metrics`,
`perturbation`, `phase_sha256`, `runtime`, `sample_design`, `scenario_sha256`, `scenarios`,
`schema_version`, `seed_table_sha256`, `seeds`, and `verdict`.

- `schema_version` is `"1"`; `gate_id` is `"convergence-resolution-v1"`;
  `accepted_base` is the full accepted-base commit; and `external_switch` is exactly
  `{"name":"CORUM_RUN_CONVERGENCE_V1","required_value":"1"}`.
- `runtime` has exactly `python`, `numpy`, and `corum`, each a non-empty exact version
  string.
- `sample_design` has exactly `scenario_count`, `replicates_per_scenario`,
  `fit_cases_per_block`, `test_cases_per_block`, `total_blocks`, `total_fit_cases`, and
  `total_test_cases`, with the registered values `6`, `40`, `8000`, `10000`, `240`,
  `1920000`, and `2400000`.
- `fusion` has exactly the eight literal keys in the fusion table above.
- `scenarios` is a six-element array in scenario-name Unicode-code-point order. Each
  scenario has exactly `name`, `calibration`, and `test`; phase and reviewer schemas are
  the exact objects registered above. Reviewer array order is the displayed order and is
  identical across phases.
- `phase_sha256` is an object with exactly 12 keys of the form
  `{scenario_name}:calibration` or `{scenario_name}:test`. Each value is the lowercase
  SHA-256 of the canonical standalone phase object bytes. `scenario_sha256` is the
  lowercase SHA-256 of the canonical `scenarios` array alone. Neither hashed value contains
  a digest field.
- `seeds` is the registered 240-element seed-table array;
  `seed_table_sha256` hashes that array alone.
- `perturbation` has exactly `algorithm`, `selection_fraction`, and `rotation`, with
  values `"same-truth-row-rotation-v1"`, `0.15`, and
  `"B[selected[(k+1)%m]]=A[selected[k]]"`.
- `bootstrap` has exactly `draws`, `seed`, `quantiles`, and `quantile_method`, with values
  `10000`, `20260901`, `[0.025,0.975]`, and `"linear"`.
- `metrics` has exactly `false_pass_cost`, `false_fail_cost`, `defer_cost`,
  `correct_cost`, `probability_clip`, and `ece_bins`, with values `1.0`, `0.2`, `1.0`,
  `0.0`, `1e-15`, and `10` under pinned Python spelling.
- `verdict` has exactly `accuracy_advantage_min`, `coverage_min`,
  `coverage_baseline_gap_max`, `dispersion_ratio_max`, `scenario_accuracy_gap_max`,
  `scenario_coverage_min`, `false_safe_ci_upper_max`, and
  `scenario_false_safe_gap_max`, with values `0.05`, `0.98`, `0.01`, `0.70`, `0.01`,
  `0.97`, `0.005`, and `0.005`.

The judge computes `config_sha256` over the complete canonical config bytes; the config
does not contain its own digest. Independent literal serializer fixtures are mandatory:

1. parsed object `{"n":1,"p":0.5,"z":null,"é":"雪"}` has byte sequence
   `7b226e223a312c2270223a302e352c227a223a6e756c6c2c22c3a9223a22e99baa227d`
   have length `35` and SHA-256
   `e8582054d4ca562a1fbdd1bf21c6eaee7b2192859275be3730989c96c52067f7`;
2. parsed object
   `{"array":["PASS","FAIL","DEFER"],"nested":{"a":0.0,"b":2}}` has byte sequence
   `7b226172726179223a5b2250415353222c224641494c222c224445464552225d2c226e6573746564223a7b2261223a302e302c2262223a327d7d`
   have length `58` and SHA-256
   `89f3bf53dad58f7f71ccc8cabd9133f7f99d4aa7fe7835567c4a50125e8eaf3d`.

The test must compare these literal bytes/digests independently of the config's claimed
hashes; recomputing expected values through the serializer under test is insufficient.

### Non-formal phase-generation development fixture

Ordinary tests use only scenario `fixture-phase-split-v1`, which is not one of the six
registered scenarios. It uses fixture-only domain bytes
`corum:convergence-resolution:fixture:v1\0phase-split`, where `\0` is NUL, yielding parent
seed `172339166934708224`; this domain/value is absent from the 720 formal seeds. It calls
`simulate_experiment` with `n_calibration=7` and `n_test=9`.

Its calibration phase has `prior_pass=0.52`, `difficulty_rate=0.25`,
`informative_missingness=0.10`; its test phase has `prior_pass=0.47`,
`difficulty_rate=0.40`, `informative_missingness=0.20`; both have no correlation. Reviewer
tuples `(accuracy, abstain, timeout, invalid)` are calibration/test respectively:
`fixture-a` `(0.73,0.11,0,0)` / `(0.70,0.12,0,0)`, `fixture-b`
`(0.68,0.07,0.05,0)` / `(0.64,0.08,0.04,0.01)`, and `fixture-c`
`(0.81,0.03,0,0.04)` / `(0.79,0.04,0.01,0.03)`, each with its own same-named lineage.

Child `0` is fit seed `15680819540018043498`; child `1` is test seed
`12188076203272908518`. The phase fixture object schema remains exactly `difficulty`,
`reviews`, and `truths` as previously defined: sorted `[case_id,value]` arrays for the
first/last and accepted-base panel tuple order for
`[case_id,reviewer_id,state,observation_or_null]`. Canonical fit bytes have length `1504`
and SHA-256 `8cb3a4892b4a9cf381fb5463b3e4b57a9fbc354b0c2ef0e9b9de4e129453004d`;
test bytes have length `1611` and SHA-256
`744e35aae4b7e7d92bd77805c2b734a49e330fe7b2be778a7d4386f62fba5a88`.

This fixture is development evidence only and never runs in formal preflight or execution.
No ordinary or preflight test may construct a registered formal scenario, use a formal
seed, or generate a formal block. A mocked/counted orchestration test must prove the formal
loop invokes all 240 `(scenario,replicate)` blocks once, in registered order, only after a
mocked fsynced START, with no pre-START call and no duplicate.

## Synthetic conditional action-dispersion/resolution operand

This operand moves complete response rows between different cases conditional on the same
truth. It measures how dispersed each method's actions are over those row assignments and
whether the method resolves them differently. It is not a same-case A/B response
transition, does not estimate reviewer instability, and is no evidence of prompt-order or
real-case stability. Even a 30% lead cannot satisfy Task 6D's real stability condition.

For each block and each truth stratum separately, construct a fresh
`Generator(PCG64(block_perturbation_seed))` and pass that stratum's ascending case indices
to `rng.permutation`. Select exactly `floor(0.15 * stratum_size)` indices and require at
least two. Cyclically rotate the complete three-reviewer A-form rows among only the
selected same-truth cases. All unselected rows are byte-identical between A and B.
The transformation preserves truth and the exact per-truth joint reviewer-row multiset,
including correlation, abstention, timeout, and invalid states. Synthetic tests must prove
both properties.

Let the permuted selected-index sequence be `selected[0:m]`, with `m >= 2`. Direction is
exactly `B[selected[(k+1) mod m]] = A[selected[k]]` for every integer `k` in `0..m-1`.
No reverse rotation is permitted. A hand fixture with at least three distinct rows must
assert this exact destination/source mapping. Any A/B action change among `PASS`, `FAIL`,
and `DEFER` is one synthetic dispersion-change event.

Fresh per-stratum generators make truth-stratum processing order irrelevant. The
permutation selects and reassigns whole rows only: it never permutes reviewer columns,
mutates a reviewer value, or moves a row across truth strata.

## Baselines and full-case probabilities

All methods receive the exact same A and B rows. Reviewer filtering or method-specific
input is forbidden.

1. **Ordinary majority:** each valid `PASS` contributes `+1`, each valid `FAIL`
   contributes `-1`, and every other row contributes zero. The sign decides; exact zero
   is `DEFER`.
2. **Reliability-weighted vote:** use only the block's 8,000 fit rows. For each reviewer,
   consider only valid directional observations; let `n` be eligible rows and `c` those
   equal to truth. Compute `a=(c+1)/(n+2)` and `w=log(a/(1-a))`. On each test case, compute
   the reviewer-ID-sorted `math.fsum(w*sign)` and take its sign; exact zero is `DEFER`.
   Test truth never enters weight fitting or action calculation.

For full-case probability diagnostics only:

- the candidate uses posterior mean `p`, or exactly `0.5` when accepted-base batch fusion
  has no posterior because all contributing reviews are non-`VALID`;
- ordinary majority uses Laplace-smoothed directional share
  `(PASS+1)/(PASS+FAIL+2)`, or `0.5` when there is no directional vote;
- weighted vote uses `sigmoid(sum(w*sign))`, or `0.5` on exact zero.

Probability diagnostics use the A-form probabilities and truths for all 2,400,000
holdout cases. With `y=1` for truth `PASS` and `0` for `FAIL`:

- Brier score is the arithmetic mean of `(p-y)**2`;
- NLL uses the natural logarithm and is the arithmetic mean of
  `-log(clip(p if y==1 else 1-p, 1e-15, 1-1e-15))`;
- ECE uses ten bins: `[k/10,(k+1)/10)` for `k=0..8` and `[0.9,1.0]` for the final bin,
  so `p=1.0` enters bin 9. Each nonempty bin contributes its case-count fraction times
  `abs(mean_probability - PASS_frequency)`.

These are exactly the accepted-base `corum.metrics` semantics with equal case weights.
TDD requires independent hand cases covering `p=0`, every bin edge, `p=1`, clipping, and
empty bins, plus equivalence against the accepted-base metric API on a deterministic
fixture. NLL, Brier, and ECE cannot select the candidate, tune a threshold, rescue a
failed gate, or change its verdict.

## Metrics and anti-`DEFER` contract

Every one of the 2,400,000 A-form test cases is assigned. `DEFER` is always incorrect for
accuracy and uncovered for coverage. Report exact `PASS`, `FAIL`, and `DEFER` counts for
every method:

- **accuracy:** fraction whose directional action equals truth; `DEFER` contributes zero;
- **coverage:** fraction whose action is not `DEFER`;
- **false-safe incidence:** fraction of all cases with action `PASS` and truth `FAIL`;
- **dispersion-change rate:** fraction whose A and B actions differ under the registered
  row rotation, with `DEFER` a normal action;
- **diagnostic decision loss:** `false_pass=1.0`, `false_fail=0.2`, `defer=1.0`, and
  correct `=0`, so deferral can never be cheaper than an error;
- **NLL/Brier/ECE:** A-form full-case diagnostics, never conditioned on coverage.

Structural fairness uses direct equality invariants rather than a ratio. Each method sees
exactly 2,400,000 case rows and 7,200,000 three-reviewer rows per form, hence 4,800,000 case
rows and 14,400,000 holdout reviewer rows across A and B. Per block those counts are
exactly 10,000 cases and 30,000 reviewer rows per form. Candidate and reliability-weighted
fitting each receive the same 5,760,000 raw fit reviewer rows; ordinary majority receives
zero fit rows. Candidate and weighted eligibility semantics differ as registered, so this
is not equality of total method inputs. Each model-call count is exactly zero. These are
synthetic accounting facts only, not token-cost evidence or satisfaction of Task 6D's
token claim.

Point metrics micro-average all equal-size blocks and scenarios. The result also reports
every individual block and each scenario aggregate. The primary paired unit is one
complete scenario-replicate block, never an individual case.

## Paired uncertainty

Use exactly 10,000 paired bootstrap draws with
`numpy.random.Generator(numpy.random.PCG64(20260901))`. For each bootstrap draw, process
scenario names in Unicode sorted order. Within each scenario, resample exactly 40 whole
replicate blocks with replacement through one `rng.integers(0, 40, size=40)` call. All
methods and all paired operands use the identical sampled indices. Compute each scenario
mean, then average the six scenario means equally.

Use NumPy quantiles `(0.025, 0.975)` with `method="linear"`. Point estimates use the
original 240 blocks, not bootstrap means. Every strict comparison uses unrounded values.
Intervals are paired evidence conditional on these six frozen synthetic scenarios; they
are not individual-case confidence intervals or a general real-project claim.

## Durable one-shot lifecycle

Formal preflight performs only static validation of the reviewed accepted-base,
documentation-head, judge-commit, config/scenario/seed hashes, pinned runtime, external
switch, absence of all three registered result paths, and regeneration of the 720-entry
seed table. It never calls `simulate_experiment`, constructs a formal `Scenario`, generates
a registered block, or reruns ordinary/heavy fixture tests. A preflight failure creates no
file and is retryable without changing registered semantics.

The registered TXT path is the attempt-0 append-only ledger. The judge, not a shell
redirect, wrapper, or recorder, is its sole writer. After clean preflight and before any formal
simulation or block, the judge opens that path with create-new/exclusive semantics
(`O_CREAT|O_EXCL` or the exact platform equivalent), writes one canonical `START` JSON
record plus LF, flushes, and fsyncs the file. Any existing TXT path, including an empty,
`START`-only, or finalized file, refuses execution and must never be deleted, truncated,
or replaced to rerun the gate.

`START` has exactly these keys: `accepted_base`, `bootstrap_draws`, `bootstrap_seed`,
`config_sha256`, `documentation_commit`, `external_switch`, `fit_cases_per_block`,
`gate_id`, `judge_commit`, `record_type`, `replicates_per_scenario`, `runtime`,
`scenario_count`, `scenario_sha256`, `schema_version`, `seed_table_sha256`,
`test_cases_per_block`, `total_blocks`, and `total_test_cases`. Values bind the full
accepted base, amended documentation head, reviewed judge commit, complete config and its
scenario/seed subhashes, exact pinned runtime versions, switch name/value, 10,000 draws,
bootstrap seed `20260901`, and the registered block/sample counts. `record_type` is
`"START"`; `schema_version` and `gate_id` match the config.

Once the fsynced `START` exists, the attempt is consumed. A caught exception, integrity or
replay failure, partial computation, or interrupt is `INVALID`; the judge writes no
partial scientific metrics. On caught or normal completion it constructs deterministic
result JSON bytes in memory but creates no registered JSON file. It appends exactly one
canonical `FINAL` record plus LF to TXT and flushes/fsyncs again. `FINAL` has exactly
`gate_id`, `reason_codes`, `record_type`, `result`, `result_sha256`, `schema_version`,
`start_sha256`, `verdict`, and `wall_time_seconds`. `result` is the entire parsed
deterministic result object; `result_sha256` hashes its canonical standalone bytes;
`start_sha256` hashes canonical START bytes without LF; verdict/reasons equal
`result.verdict`; wall time is finite and non-negative. No temp or unregistered artifact is
allowed. An uncatchable crash may leave START or a partial second record; that preserved
TXT is consumed `INVALID`, refuses execution, and is never appended, repaired, or rerun.
No third record is valid.

The judge emits no block or partial-metric stdout. After a fsynced normal/caught `FINAL`,
stdout is exactly one final line
`CORUM_CONVERGENCE_V1 verdict={VERDICT} result_sha256={SHA256}`. A START-only crash emits
no authoritative line. Wall time exists only in FINAL; it is excluded from deterministic
JSON and cannot affect replay or verdict.

## Deterministic result schema

The in-memory result and recorder-created registered JSON use the standalone canonical
rules above and have exactly `blocks`,
`gate_id`, `identity`, `integrity`, `paired`, `pooled`, `scenarios`, `schema_version`, and
`verdict`. It contains no timestamp, duration, host path, or unordered map.

- `identity` has exactly `accepted_base`, `bootstrap_draws`, `bootstrap_seed`,
  `config_sha256`, `documentation_commit`, `judge_commit`, `runtime`, `scenario_sha256`,
  and `seed_table_sha256`, equal to START. Only an administrative result for a zero or
  malformed first ledger record may set the entire `identity` value to JSON `null`.
- `integrity` has exactly `case_count_per_form`, `deterministic_replay`, `fit_case_count`,
  `method_ab_reviewer_rows`, `method_fit_reviewer_rows`, `model_call_counts`,
  `operands_sha256`, `reason_codes`, `reviewer_row_count_per_form`, `status`,
  `total_blocks`, and `test_case_count`.
  The normal values are 2,400,000 cases per form, 1,920,000 fit cases, 7,200,000 reviewer
  rows per form, 240 blocks, and 2,400,000 test cases. Method-count objects have exact key
  order `candidate`, `ordinary_majority`, `reliability_weighted` in the schema (canonical
  object serialization still sorts keys). Every `method_ab_reviewer_rows` value is
  14,400,000. `method_fit_reviewer_rows` is 5,760,000 for candidate and reliability
  weighted, and zero for ordinary majority. The same raw fit rows are available to both
  fitted methods, but candidate calibration/dependence admits every valid semantic row
  while reliability fitting admits only valid directional rows; this is not token or
  total-input equality. Every model-call count is zero. For every normal scientific
  result (`PASS`, `FAIL`, or `INCONCLUSIVE`), every integrity value is non-null,
  `deterministic_replay` is the JSON Boolean `true`, `status` is exactly `"PASS"`, and
  `reason_codes` is exactly `[]`, independent of the separate scientific verdict. All
  counts and method maps equal the normal values above. `integrity.operands_sha256` and
  every block `operands_sha256` are strings of exactly 64 lowercase hexadecimal
  characters matching `[0-9a-f]{64}`.
- `blocks` is a 240-element array sorted by scenario then replicate. Each element has
  exactly `fit_case_count`, `methods`, `operands_sha256`, `replicate`, `scenario`, and
  `test_case_count`, and `truth_counts_a`. `truth_counts_a` has exact integer keys `FAIL`
  and `PASS` summing to 10,000. `methods` is an array in exact order `candidate`,
  `ordinary_majority`, `reliability_weighted`.
- Each block method record has exactly `action_counts_a`, `action_counts_b`, `brier_sum_a`,
  `correct_count_a`, `covered_count_a`, `defer_count_a`, `dispersion_change_count`,
  `ece_bins_a`, `false_fail_count_a`, `false_pass_count_a`, `method_id`, `metrics`, and
  `nll_sum_a`. Action-count objects have exact integer keys `DEFER`, `FAIL`, `PASS`, each
  summing to 10,000. Counts are non-negative integers;
  `correct+false_fail+false_pass+defer=10000`; `covered=10000-defer`.
- `ece_bins_a` is a ten-element array in bin-index order. Each record has exactly
  `bin_index`, `count`, `pass_count`, and `probability_sum_a`; counts are integers,
  `0<=pass_count<=count`, and the ten counts sum to 10,000. `brier_sum_a`, `nll_sum_a`, and
  every probability sum are finite non-negative binary64 values.
- `metrics` has exactly `accuracy_a`, `brier_a`, `coverage_a`, `decision_loss_a`,
  `dispersion_change_rate`, `ece_a`, `false_safe_incidence_a`, and `nll_a`.
- `scenarios` is a six-element array sorted by scenario name. Each record has exactly
  `methods`, `scenario`, `test_case_count`, and `truth_counts_a`; method records use the
  same sufficient-statistic/bin/metrics schema with counts summing to 400,000.
- `pooled` has exactly `methods`, `test_case_count`, and `truth_counts_a`; methods use the
  same schema with counts summing to 2,400,000.
- `paired` is a six-element array sorted first by operand order `accuracy_advantage`,
  `dispersion_advantage`, `false_safe_delta`, then baseline ID. Each record has exactly
  `baseline`, `ci_lower`, `ci_upper`, `operand`, and `point`. Directions are candidate
  minus baseline for accuracy, baseline minus candidate for dispersion, and candidate
  minus baseline for false-safe incidence.
- `verdict` has exactly `reason_codes` and `status`; status is one of `PASS`, `FAIL`,
  `INCONCLUSIVE`, or `INVALID`, and reason codes are sorted. `PASS` has an empty reason
  array. For caught or administrative `INVALID`, `blocks`, `paired`, `pooled`, and
  `scenarios` are null; no partial science is serialized. Its `integrity.status` is
  `"INVALID"`, `integrity.reason_codes` equals `verdict.reason_codes`, and every other
  `integrity` value is JSON `null`. Its identity is the complete START-bound identity for
  caught failures and valid-START recorder failures, or JSON `null` only when the first
  record cannot be validated.

For each block and method, cases accumulate in accepted test-case order. `math.fsum` in
that order computes Brier/NLL sums and each bin probability sum. Scenario aggregation uses
replicate order `0..39`; pooled aggregation uses scenario-name order; each higher level
uses `math.fsum` over the immediately lower-level sums in that order. Counts add exactly.
Derived metrics are recomputed from sufficient statistics: accuracy `correct/N`, coverage
`covered/N`, false-safe incidence `false_pass/N`, dispersion rate `changes/N`, decision
loss `(false_pass + 0.2*false_fail + defer)/N`, Brier/NLL `sum/N`, and ECE
`sum_bin (count/N)*abs(probability_sum/count-pass_count/count)` over nonempty bins.

`operands_sha256` hashes canonical bytes of an object with exactly `methods`, `replicate`,
`scenario`, and `truth_counts_a`; method records omit derived `metrics` but contain every
sufficient-statistic field above. The integrity operand hash covers the ordered array of
all 240 such objects. No raw case, hidden probability, action, or truth is claimed to be
hashed. Fixtures pin one complete operand object's bytes/hash and reconstruct scenario and
pooled ECE solely from emitted bins/sums.

The non-formal operand fixture `fixture-operands-v1`, replicate `0`, has truths
`[PASS,FAIL,PASS]`. Candidate A/B actions are `[PASS,FAIL,DEFER]` /
`[FAIL,FAIL,DEFER]` with probabilities `[0.8,0.2,0.5]`; ordinary actions are all `PASS`
on both forms with probabilities `[0.75,0.75,0.75]`; weighted A/B actions are
`[FAIL,FAIL,PASS]` / `[FAIL,PASS,PASS]` with probabilities `[0.4,0.3,0.7]`. Under the
registered schema its canonical operand bytes have length `3039` and SHA-256
`faeb1f5df17b89bdfd2dd37dff27ec8d980b1d7857c858afcf012d63add0be45`.
Candidate/ordinary/weighted Brier sums are respectively
`0.32999999999999996`, `0.6875`, `0.54`; NLL sums are
`1.1394342831883648`, `1.9616585060234524`, `1.62964061975162`. The ten bins follow
directly from the listed probabilities and must reconstruct the fixture's pooled ECE.

## Closed reason codes and precedence

The complete closed reason-code set is:

- invalid judge codes: `INVALID_EXCEPTION`, `INVALID_SEED_REGENERATION`,
  `INVALID_SIMULATION_ORDER`, `INVALID_PERTURBATION_MULTISET`, `INVALID_COUNTS`,
  `INVALID_NONFINITE`, `INVALID_SHARED_AB_ROWS`, `INVALID_FIT_ROWS`,
  `INVALID_MODEL_CALLS`, `INVALID_REPLAY`, `INVALID_RESULT_CANONICAL`;
- recorder crash codes: `RECORDER_START_ONLY`, `RECORDER_PARTIAL_FINAL`,
  `RECORDER_MALFORMED_FINAL`;
- point/guardrail codes: `FAIL_ACCURACY_POINT_ORDINARY`,
  `FAIL_ACCURACY_POINT_WEIGHTED`, `FAIL_COVERAGE_FLOOR`,
  `FAIL_COVERAGE_GAP_ORDINARY`, `FAIL_COVERAGE_GAP_WEIGHTED`,
  `FAIL_DISPERSION_POINT_ORDINARY`, `FAIL_DISPERSION_POINT_WEIGHTED`,
  `FAIL_FALSE_SAFE_POINT_ORDINARY`, `FAIL_FALSE_SAFE_POINT_WEIGHTED`,
  `FAIL_SCENARIO_ACCURACY`, `FAIL_SCENARIO_COVERAGE`,
  `FAIL_SCENARIO_FALSE_SAFE`;
- inconclusive codes: `INCONCLUSIVE_ZERO_DISPERSION_ORDINARY`,
  `INCONCLUSIVE_ZERO_DISPERSION_WEIGHTED`, `INCONCLUSIVE_ACCURACY_CI_ORDINARY`,
  `INCONCLUSIVE_ACCURACY_CI_WEIGHTED`, `INCONCLUSIVE_DISPERSION_CI_ORDINARY`,
  `INCONCLUSIVE_DISPERSION_CI_WEIGHTED`, `INCONCLUSIVE_FALSE_SAFE_CI_ORDINARY`,
  `INCONCLUSIVE_FALSE_SAFE_CI_WEIGHTED`.

Judge invalid codes map respectively to an unexpected caught exception; seed-table/runtime
regeneration mismatch; wrong/duplicate/missing/out-of-order block calls; row-rotation truth
or multiset failure; sample/action/truth/bin/count mismatch; any nonfinite operand; A/B
row-count mismatch; fit-row count/exposure mismatch; nonzero model call; duplicate
aggregation mismatch; or noncanonical/result-hash mismatch. Any such trigger has first
precedence, yields `INVALID`, null science, and all triggered invalid codes sorted.

After integrity, every missed point or scenario condition emits its precise `FAIL_*` code;
generic scenario codes apply when one or more displayed scenario slices miss that named
guardrail. Any point code yields `FAIL`. Otherwise zero comparator dispersion emits its
precise zero code. An accuracy or dispersion CI miss means its lower bound is not strictly
greater than zero; a false-safe CI miss means its upper bound is greater than `0.005`.
Each emits its operand/baseline CI code and yields `INCONCLUSIVE`. Only the remainder is
`PASS` with `reason_codes=[]`.

## Recorder terminalization and artifact ownership

The recorder runs only after judge termination and never writes TXT. Missing TXT means no
attempt and produces no result. It validates TXT as strict UTF-8/no BOM with canonical JSON
records each terminated by LF:

1. zero bytes, an incomplete/noncanonical first record, or any first record other than the
   exact START schema -> preserve TXT and administrative `INVALID` with
   `RECORDER_MALFORMED_FINAL`;
2. one complete valid START and exact EOF -> `RECORDER_START_ONLY`;
3. valid START followed by nonempty bytes lacking a terminal LF ->
   `RECORDER_PARTIAL_FINAL`;
4. valid START plus one LF-terminated second record that fails FINAL schema, canonical
   bytes, START/result hash, verdict/reason equality, or has any trailing/third bytes ->
   `RECORDER_MALFORMED_FINAL`;
5. exactly valid START+FINAL -> extract `FINAL.result`, canonicalize it, verify
   `result_sha256`, and preserve its verdict without reinterpretation.

For cases 1--4 the recorder derives the exact administrative result schema above with
null science and the one mapped recorder code; identity comes from valid START when
available and is null for case 1. The fixed registered `gate_id` and `schema_version`
remain present. It never appends FINAL or reruns.

Publication is resumable and content-idempotent. On every invocation the recorder first
revalidates preserved TXT and derives in memory the one expected registered JSON byte
sequence, the deterministic MD byte sequence, and the exact registered status-prose
transformations. MD is UTF-8 with LF line endings and no BOM, timestamp, host path, or
unregistered input; its complete expected bytes are a pure rendering of the validated
TXT/expected JSON and are pinned by the exact verdict/crash fixtures. At the JSON and then
MD stages, an absent path is opened with
create-new/exclusive semantics, written, flushed, and fsynced. A present path is reused
only when its complete bytes equal the expected bytes; byte mismatch is a forensic
artifact conflict. The recorder then visits status documents in exact path order
`AGENTS.md`, `docs/plans/corum-mvp.md`, `docs/sdd/0010-convergence-resolution-gate.md`,
`docs/specs/corum-mvp-design.md`. Each must equal either the bound pre-result bytes or its
single exact expected post-result bytes. The pre-result state receives the registered
deterministic transformation and is flushed/fsynced; the post-result state is reused.
Any other bytes are a forensic artifact conflict.

A forensic artifact conflict refuses publication without deleting, truncating,
overwriting, appending, changing the ledger-derived verdict, or rerunning science. The
recorder never opens TXT for writing. A recorder/publication interruption consumes no new
attempt: it resumes publication for the same already-consumed attempt. A retry after a
fault at JSON, MD, any status-document boundary, or immediately before/during commit must
converge to the same registered bytes and one result commit. If the exact result commit
with subject `docs: record convergence resolution gate result`, expected parent/tree, and
only registered path changes is already HEAD, reuse it; otherwise require the bound judge
head and create that one commit only after all expected bytes validate. The expected
parent is exactly `START.judge_commit`; the expected tree is that parent plus only the
preserved TXT, expected JSON/MD, and four expected status-document bytes. A different
HEAD, tree, parent, subject, or path set is a forensic conflict. No temp or unregistered
artifact is allowed.

The deterministic MD records the lowercase SHA-256 of the complete preserved TXT bytes
and of the exact registered JSON bytes; for case 5 the latter also equals
`FINAL.result_sha256`. The result commit contains preserved TXT, recorder-owned JSON/MD,
and registered status prose. Fixtures cover PASS, FAIL, INCONCLUSIVE, caught INVALID,
START-only, partial FINAL, malformed FINAL, zero ledger, valid FINAL, and crash after
fsynced FINAL before stdout.

Deterministic replay rebuilds operands, aggregates, intervals, result, and verdict twice
from one in-memory formal run; it never resimulates. The judge places the result only in
FINAL. The recorder is sole creator of registered JSON/MD/status and never recomputes
science or reinterprets a valid FINAL.

## Attempt-0 recorded result

Task 6E attempt 0 is final: `FAIL` with reason codes `FAIL_ACCURACY_POINT_WEIGHTED`, `FAIL_COVERAGE_FLOOR`, `FAIL_COVERAGE_GAP_WEIGHTED`, `FAIL_DISPERSION_POINT_ORDINARY`, `FAIL_DISPERSION_POINT_WEIGHTED`, `FAIL_SCENARIO_ACCURACY`, `FAIL_SCENARIO_COVERAGE`, `FAIL_SCENARIO_FALSE_SAFE`. Artifacts: `docs/results/task-6e-convergence-resolution-attempt-0.txt`, `docs/results/task-6e-convergence-resolution-attempt-0.json`, and `docs/results/task-6e-convergence-resolution-attempt-0.md`. The current consensus path is stopped; another synthetic candidate, Task 7, product work, and model calls remain unauthorized pending an owner decision.

## Synthetic verdict

An integrity failure is `INVALID`. Formal execution is one-shot from the durable `START`
boundary. After START there is no same-scenario/seed repair, threshold change, block drop,
block replacement, top-up, or rerun, whether or not JSON, FINAL, or stdout exists. A
future candidate requires SDD/config/judge v2, entirely new scenario literals, and a new
seed domain.

Synthetic `PASS` requires every condition below:

1. Against each voting baseline, pooled candidate accuracy advantage is at least `0.05`,
   and its paired 95% CI lower bound is strictly greater than zero.
2. Candidate pooled coverage is at least `0.98` and no more than `0.01` below either
   voting baseline.
3. Against each voting baseline, candidate dispersion-change rate is at most `0.70` times
   baseline dispersion-change rate, and the paired 95% CI lower bound for
   `baseline_dispersion_change - candidate_dispersion_change` is strictly greater than
   zero. If either baseline rate is zero after point checks, the verdict is
   `INCONCLUSIVE`. This synthetic operand cannot satisfy Task 6D's real stability gate.
4. Candidate pooled false-safe incidence is no greater than either baseline point value,
   and the paired 95% CI upper bound for `candidate_false_safe - baseline_false_safe` is
   at most `0.005` against each.
5. In every scenario, candidate accuracy is no more than `0.01` below either baseline,
   candidate coverage is at least `0.97`, and candidate false-safe incidence is no more
   than `0.005` above either baseline.
6. Every method has the exact shared A/B counts, each fitted method has its registered fit
   exposure and ordinary majority has zero fit exposure, every model-call count equals
   zero, all 240 blocks and 2,400,000 test cases are present, and deterministic replay and
   every integrity check pass.

After integrity, verdict precedence is exact: any point or scenario guardrail failure is
`FAIL`; otherwise a zero comparator dispersion-change rate or any required confidence
condition not met is `INCONCLUSIVE`; only the remainder is `PASS`. NLL, Brier, ECE,
diagnostic decision loss, or any attractive slice never overrides this verdict.

`PASS` has only the narrow authorization stated above. `FAIL`, `INCONCLUSIVE`, or
post-start `INVALID` forbids added shrinkage, pair averaging, model averaging, thresholds,
or another synthetic candidate on the current consensus path. Preserve and report all
root-cause slices; the owner decides whether the project continues or stops.

## TDD, review, and execution order

1. Commit and independently review these four prospective documentation files before any
   config or judge work.
2. In a separate milestone, synthetic unit fixtures must RED before the config validator
   and independent judge exist, then GREEN schema/literal validation, canonical digest
   literals, NUL-octet seed regeneration, the tiny fixture's parent/child seeds and panel
   hashes, fit/test isolation, reference singleton calibration/dependence/fusion, empty pair
   registry, all-invalid `p=0.5` and mixed-invalid diagnostic hand cases, both baseline hand
   calculations, no test-truth access in weights, row-order invariance, exact three-row
   rotation, truth/multiset preservation, `DEFER`-as-wrong metrics, probability hand/API
   equivalence, whole-block bootstrap, mocked fsynced-START then exactly-once sequential
   240-block orchestration, operand hash and pooled-ECE reconstruction, exclusive ledger,
   static-only preflight, recorder zero/START-only/partial/malformed/valid FINAL cases,
   every verdict/reason class, exact normal PASS/FAIL/INCONCLUSIVE and administrative
   result bytes, recorder publication fault injection after JSON fsync, after MD fsync,
   after every ordered status-document update, immediately before commit, and after commit
   ref update, same-byte/single-commit resumption, mismatch refusal, one-line stdout,
   deterministic replay, and default external skip.
3. Add only `configs/convergence-resolution-v1.json` and
   `tests/test_convergence_resolution_value.py`. Keep formal execution skipped unless
   `CORUM_RUN_CONVERGENCE_V1=1`. Commit exactly
   `test: lock convergence resolution gate` after independent statistics, governance,
   and implementation review. Do not modify `src/corum`.
4. Run the ordinary regression suite excluding only the already consumed failing Task 6A
   and Task 6B judges. Run the focused synthetic fixtures, Ruff, mypy, deterministic
   replay, and `git diff --check`. No Task 6D artifact may be read.
5. After the config and judge milestone is independently accepted, complete preflight,
   durably create START, and run the formal 240 blocks sequentially once. The judge embeds
   deterministic result JSON in FINAL, appends/fsyncs it, and emits only the registered
   final stdout line. It never creates registered JSON. Do not expose a partial metric as
   a verdict or use it to tune anything.
6. After termination, the recorder validates TXT, creates or exact-byte-reuses the
   extracted/administrative JSON and MD, idempotently completes result/status prose, and
   refuses any forensic mismatch. Commit registered JSON, TXT, MD, and prose exactly as
   `docs: record convergence resolution gate result`. Preserve START-only, negative, and
   inconclusive evidence without reinterpretation; publication retry never touches TXT or
   consumes another attempt.

Documentation, judge/config, and formal result each require independent statistics,
governance, and implementation review. Documentation review must challenge statistical
wording, candidate leakage, `DEFER` gaming, exact sample arithmetic, whole-block pairing,
conditional-dispersion limitations, lifecycle durability, result schema, verdict
precedence, and every authorization consequence. Judge review must recompute literals,
hashes, parent/child seeds, panel fixtures, baseline and probability hand cases, paired
operands, sufficient statistics, canonical output, intervals, reason triggers, judge/
recorder ledger transitions, and verdict without trusting the candidate's metric helpers.
