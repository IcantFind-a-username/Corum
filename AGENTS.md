# Corum AI Construction Contract

This file governs AI-assisted work in the entire repository. It is an execution contract,
not a replacement for the approved architecture or implementation plan.

## 1. Read before changing anything

After discovering this file, every implementation or review agent must read, in order:

1. `DEVELOPMENT.md`
2. `docs/specs/corum-mvp-design.md`
3. the relevant task in `docs/plans/corum-mvp.md`
4. the matching tracked SDD under `docs/sdd/`, when present
5. the current implementation and tests for the affected modules
6. recent Git history and `git status`

Conflict priority is: direct project-owner instruction, approved design, implementation
plan or owner-approved task SDD, this operational contract, then README prose. Do not
silently resolve a material conflict. Stop and ask the owner, or record an approved plan
amendment before implementation.

## 2. Project identity and sovereignty boundary

- Project: `Corum`
- Python distribution and import package: `corum`
- Owner and original author: Franz Xu
- License: Apache-2.0
- Canonical repository: `IcantFind-a-username/Corum`

Corum is an independent personal open-source project. Keep it clean-room:

- Never introduce former-employer or non-public project names, code, prompts, schemas,
  interfaces, datasets, logs, screenshots, results, or internal terminology.
- Do not adapt private artifacts by renaming identifiers. Re-derive behavior from this
  repository's public design, public sources, and tests.
- Attribute public datasets, papers, mathematical methods, and reused code to their
  original sources.
- Describe Corum's novelty as its system design and engineering combination. Do not claim
  ownership of prior mathematics or prevent independent implementations of ideas.
- Do not expose private email addresses or credentials. Use the configured GitHub no-reply
  identity for public Git metadata.
- No paid API or paid compute call is authorized without an explicit owner-approved budget.

## 3. Product objective

Corum tests whether evidence-aware, dependence-aware consensus among imperfect reviewers
can reduce decision loss and reviewer cost without hiding uncertainty behind majority
voting.

The MVP must:

- calibrate reviewer behavior from labeled data;
- propagate calibration uncertainty;
- discount correlated or same-lineage evidence;
- enforce deterministic hard gates and reviewer/lineage/ESS quorum;
- return `PASS`, `FAIL`, or `DEFER` with explicit reasons;
- support a leakage-free adaptive cascade;
- measure quality, uncertainty, coverage, and cost reproducibly;
- validate with zero-cost simulation before optional public-data model inference.

The statistical MVP remains a small library and evaluation harness. Productization comes
after its usefulness gates and follows a human-contract-first boundary:

- a user can provide project description, critical checkpoints, FAIL conditions, and
  requirement/evidence documents through a simple local UI;
- safe project reading and LLM-assisted checkpoint suggestions are optional enrichment,
  never the primary authority;
- developers bring their own LLM endpoint, key, and model; Corum owns validation,
  bounded context, traceability, consensus, audit, recommendations, and quality scoring;
- do not build hosted secret management, a universal provider SDK, or paid inference into
  the statistical core.

### 3.1 Evidence before components

Corum is useful software, not a research-component collection. A new component is admitted
only when it serves a pre-registered user outcome and has a cheaper existing-practice
baseline or ablation to beat. Infrastructure such as simulation is allowed only as the
minimum machinery for the next locked comparison; it is not user value by itself.

Before Task 7 or any product-surface expansion, the full-static Corum core must pass the
locked Core Value Gate after Task 6 against ordinary unweighted majority voting. The gate
also compares dependence-aware fusion with naive independent fusion. Builders may not
change scenarios, loss weights, thresholds, test seeds, coverage requirements, or the
baseline after seeing gate results. Synthetic success permits external validation; it
does not prove real-world usefulness.

If the gate fails, stop new-component work. Permit at most three bounded repair iterations
within the existing core, each with a regression and the unchanged independent judge. If
it still fails, record `CORE_VALUE_GATE_FAILED` and return the pivot/stop decision to the
project owner.

## 4. Architecture and ownership

The statistical core is a small typed Python library. Network access, dataset download,
and model inference stay outside the core.

| Unit | Sole responsibility |
|---|---|
| `models.py` | Immutable enums and domain records |
| `calibration.py` | Dirichlet fitting and sampling for reviewer likelihoods |
| `dependence.py` | Error dependence, lineage fallback, subset weights, and ESS |
| `fusion.py` | Log-space posterior fusion over shared likelihood draws |
| `decision.py` | Hard-gate precedence, quorum, and risk-aware action policy |
| `simulation.py` | Reproducible correlated panels, drift, and missingness |
| `baselines.py` | Leakage-free comparison methods |
| `metrics.py` | Selective performance, calibration, dependence, and cost metrics |
| `cascade.py` | No-look-ahead reviewer ordering, acquisition, and early stop |
| `experiment.py` | Split-safe orchestration and artifact production |
| `reporting.py` | Honest machine-readable and Markdown results |
| `datasets/halueval.py` | Pinned public-data validation and deterministic splits |
| `cli.py` | Minimal `argparse` commands only |

Keep each unit independently understandable and testable. Do not move responsibilities
between modules merely for convenience. Do not introduce a framework when a focused
function or dataclass satisfies the registered interface.

The intended data flow is:

```text
cases/reviews -> schema validation + hard gates -> calibration -> dependence
              -> posterior fusion -> risk decision -> PASS / FAIL / DEFER
                                             \-> next reviewer -> refusion
```

## 5. Non-negotiable domain and statistical invariants

### 5.1 Data contract

- Latent truth is binary: `Truth.PASS` or `Truth.FAIL`.
- Semantic observation is separate: `PASS`, `FAIL`, or `ABSTAIN`.
- Execution state is separate: `VALID`, `TIMEOUT`, `INVALID`, `REFUSAL`, or
  `NOT_CALLED`.
- Only `VALID` reviews contribute a semantic likelihood. Other states remain in the audit
  ledger and reduce coverage or quorum.
- A `VALID` review requires an observation; a non-`VALID` review forbids one.
- Duplicate `(reviewer_id, case_id)` rows, conflicting truth for one case, unknown
  reviewers, impossible probabilities, negative costs, and non-finite values fail with
  typed, actionable errors.
- Public records are immutable. NumPy arrays must be defensively owned and irreversibly
  read-only, not merely views with a reversible write flag.

### 5.2 Calibration

- Model `P(observation | truth)`, never `P(truth | observation)`.
- Truth row order is `(PASS, FAIL)`.
- Observation column order is `(PASS, FAIL, ABSTAIN)`.
- Fit counts from the calibration split only. Non-valid executions contribute no
  semantic counts.
- The pooled parent prior is symmetrically smoothed, row-normalized, and contributes
  exactly the declared pseudo-count strength to each truth row.
- Cold-start reviewers shrink toward the pooled parent and retain wider uncertainty than
  large-sample reviewers at the same empirical rate.
- Fusion uses Dirichlet draws, not only posterior means.

### 5.3 Dependence

- Calibration likelihoods already encode reliability. Never multiply evidence by an
  accuracy or quality weight again.
- Entropy and Jensen-Shannon divergence are diagnostic or routing signals only. They do
  not independently prove correctness and must never become an additional reliability
  weight.
- Dependence is estimated from overlapping binary semantic errors. A valid `ABSTAIN` is
  an error for this diagnostic because it did not recover truth.
- Use `lineage`, not descriptive `family`, as the conservative grouping key.
- Weight only the actually queried subset `S`:

  `w_i(S) = 1 / (1 + sum(max(rho_ij, 0) for j in S if j != i))`

- Negative error correlation is retained for diagnosis but never creates extra weight.
- Sparse or unestimable same-lineage pairs use the registered conservative fallback;
  unrelated pairs default to zero. PSD projection of the diagnostic matrix must not
  overwrite these exact weighting fallbacks.
- A singleton queried reviewer has weight `1`, regardless of unqueried clones.
- Empty-subset ESS is `0`; non-empty ESS is finite and bounded in `[1, n]`.

### 5.4 Fusion and decision

- Accumulate weighted class likelihoods in log space with bounded probabilities.
- Sample calibration parameters once per `FusionContext` and reuse the same draws across
  cases and across growing cascade subsets.
- Recompute dependence weights for each case's valid queried subset.
- Missing executions contribute nothing. An empty valid panel returns no posterior.
- The batched path and scalar path must share a tested kernel and agree byte-for-byte for
  a fixed context. In the matrix path, `valid_mask` is the sole authority on whether a
  cell contributes.
- The posterior interval propagates Dirichlet likelihood uncertainty conditional on a
  point-estimated dependence adjustment. Never call it a full correlated-output credible
  interval or a formal risk guarantee.
- Decision precedence is fixed:
  1. trusted deterministic `FAIL` gate -> `FAIL`;
  2. trusted `UNRESOLVED` gate prevents `PASS` -> ordinarily `DEFER`;
  3. missing posterior or reviewer/lineage/ESS quorum failure -> `DEFER`;
  4. lower conditional bound at or above pass threshold -> `PASS`;
  5. upper conditional bound at or below fail threshold -> `FAIL`;
  6. otherwise -> `DEFER`.
- Hard gates always outrank statistical fusion.

### 5.5 Cascade and split safety

- Reviewer ordering and utility use likelihood-fitting calibration data only. Held-out
  policy data may select a baseline and action policy, but must not influence reviewer
  ordering.
- The cascade may reveal only the next selected review. It must not inspect an unqueried
  observation to select, stop, or reorder.
- Every registered reviewer appears in the final execution ledger; untouched reviewers
  are explicit `NOT_CALLED` records.
- Budget exhaustion returns `DEFER`; it never relaxes a threshold or quorum.
- Likelihood-fit, policy-selection, smoke, and locked-test case IDs must be disjoint.
- Never select reviewers, thresholds, baselines, prevalence, or algorithms on test data.

## 6. Roadmap and current checkpoint

Tasks execute strictly in order. A later task must not begin until the current task has
fresh verification and independent review with no open Critical or Important finding.

| Task | Deliverable | Status at this checkpoint |
|---:|---|---|
| 1 | Package scaffold and immutable domain contract | Complete, including numeric repair |
| 2 | Dirichlet reviewer calibration and uncertainty | Complete |
| 3 | Dependence estimation and duplicate-evidence control | Complete |
| 4 | Posterior fusion, hard gates, and risk-aware policy | Complete (`9e1c606`) |
| 5 | Reproducible correlated-panel simulator | Current |
| 6 | Baselines, metrics, and paired uncertainty | Pending |
| 6A | Locked core-vs-majority value gate | Pending; blocks Task 7 |
| 7 | Leakage-free adaptive cascade | Pending |
| 8 | End-to-end runner, CLI, and report renderer | Pending |
| 9 | HaluEval adapter and zero-cost Kaggle notebook | Pending |
| 10 | Open-source release surface and CI quality gates | Pending |
| 11 | Execute the locked MVP benchmark and publish report | Pending |
| 12 | Independent final review, verification, and delivery | Pending |

The coordinator may update this checkpoint in a repository-level documentation commit.
Task implementers must not expand their allowed-file list solely to edit status prose.

### Task 5 handoff

The next implementation task is exactly Task 5 in the tracked plan and
`docs/sdd/0005-correlated-panel-simulator.md`. It may modify only `pyproject.toml` to add
the declared SciPy runtime dependency and create `src/corum/simulation.py` plus
`tests/test_simulation.py`. Keep UI, repository ingestion, LLM adapters, baselines, and
reporting out of this task. Task 5 is the zero-cost testbed that must make later claims
about consensus usefulness measurable.

## 7. Mandatory development workflow

Use the detailed steps and exact commit message registered in the current roadmap task.
Default to low reasoning depth and small TDD increments. Escalate reasoning only for a
contract conflict, numerical/statistical ambiguity, security/privacy boundary, repeated
test failure, flaky evidence, or an unresolved independent-review finding.

1. **Establish scope**
   - Derive the accepted base and branch from the current checkpoint, Git history, and
     owner/coordinator handoff; do not hard-code an environment-specific SHA here.
   - Read the complete task and affected modules.
   - Preserve all unrelated and pre-existing worktree changes.
   - Modify only the files listed by the task unless an owner-approved plan amendment is
     committed first.
2. **RED**
   - Write behavioral tests before production behavior.
   - Run the focused test and retain the failure evidence.
   - The failure must demonstrate the missing behavior, not a typo or broken fixture.
3. **GREEN**
   - Implement the smallest robust behavior satisfying the contract.
   - Avoid speculative abstractions, compatibility layers, provider SDKs, and unrelated
     refactors.
4. **Self-review**
   - Inspect the complete diff, public API, numerical edge cases, immutability, seed use,
     data leakage, error messages, and allowed-file scope.
5. **Fresh verification**
   - Run the task's exact targeted tests, Ruff, and mypy commands.
   - Run the complete repository test suite.
   - Run performance, deterministic replay, package, CLI, or notebook gates when the task
     requires them.
6. **Independent review**
   - A fresh read-only reviewer checks the diff against the design and task.
   - Fix every Critical and Important finding. Behavioral code defects require a new
     failing regression; documentation, provenance, performance-environment, Git, or
     instruction defects require the smallest appropriate executable or manual evidence.
   - Re-review each fix. Do not advance on an unresolved finding.
7. **Commit and milestone delivery**
   - Use the task's exact registered commit message.
   - Never force-push or rewrite public history.
   - Local reviewed task completion and public milestone delivery are separate gates. Push
     and verify only when the owner/coordinator designates the task as a stable milestone,
     and complete final remote verification in Task 12.
   - If ordinary push is unavailable and no authorized repository integration is
     available, stop at the permission boundary and report the exact blocker. Never bypass
     authentication or broaden repository permissions.

Do not implement the next task while review or repair of the current task is running.
Parallel agents may perform independent read-only analysis, but production implementers
remain sequential.

## 8. Verification baseline

Use task-specific commands from the plan first. The general repository gate is:

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src/corum
git diff --check
```

Add `scripts` to Ruff/mypy commands once Python scripts exist. If any generated tool cache
is corrupt, rerun with a fresh temporary cache directory; cache corruption is not a
source-code failure. Never delete or commit an unexplained local cache or lock file to
make a check green.

When required by the task, also run:

- `uv build` and wheel-install smoke tests;
- CLI help and deterministic tiny experiment replay;
- the locked fusion throughput command without reducing its dimensions or threshold;
- notebook validation without executing paid or network-dependent inference;
- dataset fixture tests without live network access.

Evidence must be fresh. An earlier run, agent report, or green partial suite is not proof
of current completion.

## 9. Reproducibility, data, and cost gates

- Every random operation accepts an explicit seed.
- Every result records package version, Git commit, configuration digest, split seed,
  posterior draws, runtime versions, and dataset/model revisions when applicable. Fields
  that do not apply remain explicit `null` or status values rather than disappearing.
- Experiment writes are atomic; interrupted output cannot claim `complete`.
- Raw public datasets are downloaded from pinned official revisions, checksum-verified,
  attributed, and never committed unless a tiny fixture is explicitly license-safe.
- HaluEval source records, not answer variants, are the split unit.
- Raw locked-test reviewer votes are cached before test analysis.
- Do not collect hidden chain-of-thought. Store only concise structured votes and declared
  metadata.
- Zero-cost deterministic tests and simulation gates run before any real-model stage.
- If free compute is unavailable, ship a runnable notebook and mark external validation
  pending. Do not substitute synthetic success for real-model evidence.
- A zero-cost gate failure stops escalation and is reported honestly.

## 10. Evaluation and reporting discipline

- Use the pre-registered scenarios, seeds, sample sizes, costs, target prevalence,
  baselines, policy candidate grid, calibration-only selection objective and tie-breaks,
  and bootstrap design. Lock the selected policy before test access. Changes require a
  prospective design/plan amendment before locked-test inspection.
- Report synthetic and real-model evidence separately.
- Compare methods at equal or explicitly reported coverage.
- Report false-PASS, false-FAIL, selective risk, coverage, decision loss, Brier, NLL,
  calibration, interval width, dependence, ESS, reviewer calls, tokens, and cost as
  registered.
- `PASS`, `FAIL`, and `INCONCLUSIVE` are all legitimate MVP outcomes.
- Do not convert non-significance into success or failure.
- Do not describe heuristic dependence correction as a formal correlated-output posterior
  model.
- Do not claim production readiness, universal model superiority, or general external
  validity from three fixed models or one public benchmark.

## 11. Git, dependency, and repository hygiene

- Preserve a dirty worktree. Existing changes belong to the owner unless proven otherwise.
- Never use destructive reset/checkout commands or broad recursive deletion.
- Use focused patches and explicit paths.
- Do not commit `uv.lock` unless the owner explicitly adopts a lock-file policy for this
  library.
- Add a production dependency only at first runtime use. Prefer NumPy; add SciPy only when
  a registered numerical primitive actually requires it.
- Keep generated results, downloaded data, model caches, secrets, and scratch artifacts
  out of Git unless the plan explicitly registers a public artifact.
- License, authorship, citation, and provenance files are release contracts. Do not alter
  them incidentally during feature tasks.
- If ordinary `git push` is unavailable, use only an authorized repository integration.
  Preserve fast-forward history, never force the ref, and verify the resulting remote SHA
  and changed files. If neither path is authorized, stop and report the blocker.

## 12. Definition of Done

A task is complete only when all of the following are evidenced:

- every registered behavior and edge case is implemented;
- the RED failure and final GREEN result are recorded;
- targeted and full tests pass from the final tree;
- Ruff, mypy, formatting/diff, and task-specific gates pass;
- deterministic and performance constraints remain unchanged;
- no data leakage, proprietary material, credential, or paid call was introduced;
- an independent reviewer reports no open Critical or Important issue;
- behavioral review fixes have regression coverage; other fixes have appropriate recorded
  verification evidence; all fixes receive a clean re-review;
- the commit contains only task-scoped files and uses the registered message;
- if designated as a public stable milestone, it is delivered and verified on the remote
  repository; otherwise local completion is explicitly distinguished from delivery;
- reports state limitations and negative/inconclusive results honestly.

When any item is missing, report the task as in progress or blocked. Never lower a gate,
hide a failure, or broaden a claim to make the MVP appear successful.
