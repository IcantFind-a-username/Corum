"""Frozen JudgeBench gate for the external value of Corum's legacy core.

User journey: as a developer, I want Corum compared with ordinary voting on the same
real reviewer outputs, so that a product investment is backed by prospective evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha1, sha256
from math import fsum, isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from corum.baselines import DecisionCosts, majority_decision
from corum.calibration import OBSERVATION_ORDER, fit_panel_calibrations
from corum.decision import DecisionPolicy, decide
from corum.dependence import fit_dependence
from corum.fusion import BatchFusedPosterior, build_fusion_context, fuse_review_matrix
from corum.metrics import evaluate_decisions, select_decision_policy
from corum.models import (
    Action,
    CalibrationExample,
    ExecutionState,
    FusedPosterior,
    Observation,
    Review,
    Reviewer,
    Truth,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _REPOSITORY_ROOT / "configs" / "judgebench-v1.json"
_EXPECTED_REGISTRY_SHA256 = (
    "2b45e2e9f064de225fcb55b37b17fa03a751c22f7d4b3ca98cc9071251e7a1af"
)
_EXPECTED_CASES = 350
_NLL_EPSILON = 1e-15
_COSTS = DecisionCosts(false_pass=1.0, false_fail=1.0, defer=0.25)


@dataclass(frozen=True, slots=True)
class _PairRecord:
    case_id: str
    source: str
    question: str
    label: str
    truth: Truth
    source_group: str


@dataclass(frozen=True, slots=True)
class _GateCase:
    case_id: str
    source: str
    source_group: str
    truth: Truth
    split: str
    reviews: tuple[Review, ...]


def _normalize_displayed_decision(decision: object) -> int:
    if decision is None:
        return 0
    if decision in {"A>B", "B<A"}:
        return 1
    if decision in {"B>A", "A<B"}:
        return -1
    if decision in {"A=B", "B=A"}:
        return 0
    raise ValueError("unknown JudgeBench decision token")


def _normalize_decision_pair(first: object, second: object) -> Observation:
    sign = _normalize_displayed_decision(first) - _normalize_displayed_decision(second)
    if sign > 0:
        return Observation.PASS
    if sign < 0:
        return Observation.FAIL
    return Observation.ABSTAIN


def _normalization_kind(first: object, second: object) -> str:
    first_sign = _normalize_displayed_decision(first)
    second_sign = -_normalize_displayed_decision(second)
    total = first_sign + second_sign
    if abs(total) == 2:
        return "directional_agreement"
    if abs(total) == 1:
        return "direction_plus_zero"
    if first_sign == 0 and second_sign == 0:
        return "double_zero"
    return "order_conflict"


def _reference_majority_observations(
    observations: Iterable[Observation],
) -> Action:
    rows = tuple(observations)
    pass_votes = sum(value is Observation.PASS for value in rows)
    fail_votes = sum(value is Observation.FAIL for value in rows)
    if pass_votes > fail_votes:
        return Action.PASS
    if fail_votes > pass_votes:
        return Action.FAIL
    return Action.DEFER


def _reference_lineage_majority(
    observations: Mapping[str, Observation],
    lineage_by_reviewer: Mapping[str, str],
) -> Action:
    if set(observations) != set(lineage_by_reviewer):
        raise ValueError("lineage majority requires exactly one lineage per reviewer")
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for reviewer_id, observation in observations.items():
        grouped[lineage_by_reviewer[reviewer_id]].append(observation)
    lineage_observations: list[Observation] = []
    for lineage in sorted(grouped):
        action = _reference_majority_observations(grouped[lineage])
        if action is Action.PASS:
            lineage_observations.append(Observation.PASS)
        elif action is Action.FAIL:
            lineage_observations.append(Observation.FAIL)
        else:
            lineage_observations.append(Observation.ABSTAIN)
    return _reference_majority_observations(lineage_observations)


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def _aggregate_reason_counts(
    reason_rows: Iterable[Sequence[str]],
) -> dict[str, int]:
    counts = Counter(reason for reasons in reason_rows for reason in reasons)
    return dict(sorted(counts.items()))


def _case_loss(truth: Truth, action: Action) -> float:
    if action is Action.DEFER:
        return _COSTS.defer
    if action is Action.PASS and truth is Truth.FAIL:
        return _COSTS.false_pass
    if action is Action.FAIL and truth is Truth.PASS:
        return _COSTS.false_fail
    return 0.0


def _reference_score(
    case_ids: Sequence[str],
    truths: Sequence[Truth],
    actions: Sequence[Action],
) -> tuple[dict[str, object], dict[str, float]]:
    if not case_ids or not (len(case_ids) == len(truths) == len(actions)):
        raise ValueError("score rows must be aligned and non-empty")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("score case IDs must be unique")

    losses = {
        case_id: _case_loss(truth, action)
        for case_id, truth, action in zip(case_ids, truths, actions, strict=True)
    }
    decided = sum(action is not Action.DEFER for action in actions)
    correct_decided = sum(
        (action is Action.PASS and truth is Truth.PASS)
        or (action is Action.FAIL and truth is Truth.FAIL)
        for truth, action in zip(truths, actions, strict=True)
    )
    fail_truths = sum(truth is Truth.FAIL for truth in truths)
    pass_truths = len(truths) - fail_truths
    false_passes = sum(
        truth is Truth.FAIL and action is Action.PASS
        for truth, action in zip(truths, actions, strict=True)
    )
    false_fails = sum(
        truth is Truth.PASS and action is Action.FAIL
        for truth, action in zip(truths, actions, strict=True)
    )
    pass_actions = sum(action is Action.PASS for action in actions)
    errors = false_passes + false_fails
    action_counts = Counter(action.value for action in actions)
    score: dict[str, object] = {
        "cases": len(case_ids),
        "truth_counts": {
            "PASS": pass_truths,
            "FAIL": fail_truths,
        },
        "action_counts": {
            action.value: action_counts[action.value] for action in Action
        },
        "decision_loss": fsum(losses.values()) / len(case_ids),
        "coverage": decided / len(case_ids),
        "useful_resolution": correct_decided / len(case_ids),
        "false_pass_rate": _ratio(false_passes, fail_truths),
        "false_fail_rate": _ratio(false_fails, pass_truths),
        "false_safe_risk": _ratio(false_passes, pass_actions),
        "selective_risk": _ratio(errors, decided),
    }
    return score, losses


def _paired_source_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    draws: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    if not rows or draws <= 0:
        raise ValueError("bootstrap requires rows and positive draws")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[cast(str, row["source"])].append(row)
    ordered_groups = tuple(
        tuple(sorted(grouped[source], key=lambda row: cast(str, row["case_id"])))
        for source in sorted(grouped)
    )

    ordinary_point = fsum(
        cast(float, row["ordinary_loss"]) - cast(float, row["corum_loss"])
        for row in rows
    ) / len(rows)
    lineage_point = fsum(
        cast(float, row["lineage_loss"]) - cast(float, row["corum_loss"])
        for row in rows
    ) / len(rows)
    ordinary_replicates = np.empty(draws, dtype=float)
    lineage_replicates = np.empty(draws, dtype=float)
    generator = np.random.default_rng(seed)
    for draw_index in range(draws):
        sampled: list[Mapping[str, object]] = []
        for group in ordered_groups:
            indices = generator.integers(0, len(group), size=len(group))
            sampled.extend(group[int(index)] for index in indices)
        ordinary_replicates[draw_index] = fsum(
            cast(float, row["ordinary_loss"]) - cast(float, row["corum_loss"])
            for row in sampled
        ) / len(sampled)
        lineage_replicates[draw_index] = fsum(
            cast(float, row["lineage_loss"]) - cast(float, row["corum_loss"])
            for row in sampled
        ) / len(sampled)

    ordinary_interval = np.quantile(
        ordinary_replicates,
        (0.025, 0.975),
        method="linear",
    )
    lineage_interval = np.quantile(
        lineage_replicates,
        (0.025, 0.975),
        method="linear",
    )
    return {
        "ordinary": (
            ordinary_point,
            float(ordinary_interval[0]),
            float(ordinary_interval[1]),
        ),
        "lineage": (
            lineage_point,
            float(lineage_interval[0]),
            float(lineage_interval[1]),
        ),
    }


def _classify_verdict(
    *,
    integrity_ok: bool,
    guardrails_ok: bool,
    relative_targets_ok: bool,
    confidence_ok: bool,
) -> str:
    if not integrity_ok:
        return "INVALID"
    if not guardrails_ok:
        return "FAIL"
    if not relative_targets_ok or not confidence_ok:
        return "INCONCLUSIVE"
    return "PASS"


def _load_registry() -> dict[str, Any]:
    raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("JudgeBench registry must be a JSON object")
    return cast(dict[str, Any], raw)


def _registry_digest(registry: Mapping[str, object]) -> str:
    canonical = json.dumps(
        registry,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


def _safe_registry_digest(
    loader: Callable[[], Mapping[str, object]] = _load_registry,
) -> str | None:
    try:
        return _registry_digest(loader())
    except Exception:  # noqa: BLE001 - INVALID fallback must never raise
        return None


def _result_json(result: Mapping[str, object]) -> str:
    return json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"pinned upstream Git check failed: {detail[:300]}")
    return completed.stdout


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("utf-8").strip()


def _candidate_manifest_bytes(
    candidates: Sequence[Mapping[str, object]],
) -> bytes:
    return "".join(
        f"{candidate['path']}\0{candidate['blob_oid']}\0{candidate['bytes']}\n"
        for candidate in sorted(candidates, key=lambda row: cast(str, row["path"]))
    ).encode("utf-8")


def _verify_upstream_inventory(
    repository: Path,
    registry: Mapping[str, Any],
) -> dict[str, object]:
    upstream = cast(Mapping[str, Any], registry["upstream"])
    commit = cast(str, upstream["commit"])
    if _git_text(repository, "rev-parse", f"{commit}^{{commit}}") != commit:
        raise ValueError("local JudgeBench clone does not resolve the pinned commit")
    root_tree = _git_text(repository, "rev-parse", f"{commit}^{{tree}}")
    outputs_tree = _git_text(repository, "rev-parse", f"{commit}:outputs")
    if root_tree != upstream["root_tree"] or outputs_tree != upstream["outputs_tree"]:
        raise ValueError("pinned JudgeBench tree object mismatch")

    raw_tree = _git_bytes(
        repository,
        "ls-tree",
        "-r",
        "-l",
        "-z",
        commit,
        "--",
        "outputs",
    )
    response_model = cast(str, upstream["response_model"])
    judge_name = cast(str, upstream["judge_name"])
    marker = (
        "outputs/dataset=judgebench,"
        f"response_model={response_model},judge_name={judge_name},judge_model="
    )
    discovered: list[dict[str, object]] = []
    for entry in raw_tree.split(b"\0"):
        if not entry:
            continue
        metadata, encoded_path = entry.split(b"\t", 1)
        path = encoded_path.decode("utf-8")
        if not path.startswith(marker) or not path.endswith(".jsonl"):
            continue
        mode, object_type, encoded_oid, encoded_size = metadata.split()
        if mode != b"100644" or object_type != b"blob":
            raise ValueError("JudgeBench candidate must be a regular blob")
        judge_model = path[len(marker) : -len(".jsonl")]
        discovered.append(
            {
                "path": path,
                "judge_model": judge_model,
                "blob_oid": encoded_oid.decode("ascii"),
                "bytes": int(encoded_size),
            }
        )
    discovered.sort(key=lambda row: cast(str, row["path"]))

    registered = tuple(cast(Sequence[Mapping[str, object]], upstream["candidates"]))
    registered_structural = [
        {
            "path": candidate["path"],
            "judge_model": candidate["judge_model"],
            "blob_oid": candidate["blob_oid"],
            "bytes": candidate["bytes"],
        }
        for candidate in registered
    ]
    if discovered != registered_structural:
        raise ValueError("complete JudgeBench Arena-Hard candidate inventory mismatch")
    canonical = _candidate_manifest_bytes(discovered)
    if len(canonical) != upstream["candidate_manifest_bytes"]:
        raise ValueError("JudgeBench candidate manifest byte count mismatch")
    if sha256(canonical).hexdigest() != upstream["candidate_manifest_sha256"]:
        raise ValueError("JudgeBench candidate manifest digest mismatch")

    prefixes = tuple(
        cast(Sequence[str], upstream["eligibility"]["openai_model_prefixes"])
    )
    eligible_paths: set[str] = set()
    for discovered_row, registered_row in zip(
        discovered,
        registered,
        strict=True,
    ):
        judge_model = cast(str, discovered_row["judge_model"])
        eligible = not judge_model.startswith(prefixes)
        if eligible is not registered_row["eligible"]:
            raise ValueError("JudgeBench eligibility rule disagrees with registry")
        if eligible:
            eligible_paths.add(cast(str, discovered_row["path"]))
    vote_paths = {
        cast(str, file_row["path"])
        for file_row in cast(Sequence[Mapping[str, object]], upstream["files"])
        if file_row["role"] == "votes"
    }
    if eligible_paths != vote_paths:
        raise ValueError("materialized vote registry must equal every eligible candidate")
    for file_row in cast(Sequence[Mapping[str, object]], upstream["files"]):
        path = cast(str, file_row["path"])
        expected_oid = cast(str, file_row["blob_oid"])
        if _git_text(repository, "rev-parse", f"{commit}:{path}") != expected_oid:
            raise ValueError("registered raw path does not resolve to its pinned Git blob")
    return {
        "commit": commit,
        "root_tree": root_tree,
        "outputs_tree": outputs_tree,
        "candidate_count": len(discovered),
        "eligible_count": len(eligible_paths),
        "candidate_manifest_sha256": sha256(canonical).hexdigest(),
    }


def _stream_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    size = path.stat().st_size
    digest = sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_raw_files(
    raw_directory: Path,
    registry: Mapping[str, Any],
) -> dict[str, Path]:
    upstream = cast(Mapping[str, Any], registry["upstream"])
    files = cast(Sequence[Mapping[str, object]], upstream["files"])
    resolved: dict[str, Path] = {}
    for row in files:
        local_name = cast(str, row["local_name"])
        if Path(local_name).name != local_name:
            raise ValueError("registry local_name must be a plain file name")
        path = raw_directory / local_name
        if not path.is_file():
            raise ValueError(f"missing pinned JudgeBench raw file: {local_name}")
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"JudgeBench byte count mismatch: {local_name}")
        if _git_blob_oid(path) != row["blob_oid"]:
            raise ValueError(f"JudgeBench Git blob identity mismatch: {local_name}")
        if _stream_sha256(path) != row["sha256"]:
            raise ValueError(f"JudgeBench SHA-256 mismatch: {local_name}")
        key = "pairs" if row["role"] == "pairs" else cast(str, row["reviewer_id"])
        if key in resolved:
            raise ValueError(f"duplicate JudgeBench raw role: {key}")
        resolved[key] = path
    if len(resolved) != 8 or "pairs" not in resolved:
        raise ValueError("JudgeBench raw registry must resolve one pair file and seven votes")
    return resolved


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL in {path.name} at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL row in {path.name} at line {line_number} must be an object"
                )
            yield cast(dict[str, Any], value)


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"JudgeBench {field} must be a non-empty string")
    return value


def _load_pairs(
    path: Path,
    registry: Mapping[str, Any],
) -> dict[str, _PairRecord]:
    response_model = cast(str, registry["upstream"]["response_model"])
    records: dict[str, _PairRecord] = {}
    groups: set[str] = set()
    for row in _iter_jsonl(path):
        case_id = _required_string(row, "pair_id")
        if case_id in records:
            raise ValueError("duplicate JudgeBench pair_id in pair file")
        if row.get("response_model") != response_model:
            raise ValueError("JudgeBench pair response_model mismatch")
        source = _required_string(row, "source")
        question = _required_string(row, "question")
        label = _required_string(row, "label")
        if label == "A>B":
            truth = Truth.PASS
        elif label == "B>A":
            truth = Truth.FAIL
        else:
            raise ValueError("unknown JudgeBench truth label")
        source_group = sha256(f"{source}\0{question}".encode()).hexdigest()
        if source_group in groups:
            raise ValueError("JudgeBench source_group must be unique in attempt 0")
        groups.add(source_group)
        records[case_id] = _PairRecord(
            case_id=case_id,
            source=source,
            question=question,
            label=label,
            truth=truth,
            source_group=source_group,
        )
    if len(records) != _EXPECTED_CASES:
        raise ValueError("JudgeBench pair file must contain exactly 350 rows")
    return records


def _assign_splits(
    pairs: Mapping[str, _PairRecord],
    registry: Mapping[str, Any],
) -> dict[str, str]:
    split_config = cast(Mapping[str, Any], registry["split"])
    coding_source = cast(str, split_config["coding_source"])
    salt = cast(str, split_config["sort_salt"])
    cycle = tuple(cast(Sequence[str], split_config["cycle"]))
    strata: dict[tuple[str, str], list[_PairRecord]] = defaultdict(list)
    assigned: dict[str, str] = {}
    for pair in pairs.values():
        if pair.source == coding_source:
            assigned[pair.case_id] = "coding_test"
        else:
            strata[(pair.source, pair.label)].append(pair)
    for rows in strata.values():
        ordered = sorted(
            rows,
            key=lambda pair: sha256(
                f"{salt}\0{pair.source_group}".encode()
            ).hexdigest(),
        )
        for index, pair in enumerate(ordered):
            assigned[pair.case_id] = cycle[index % len(cycle)]
    counts = Counter(assigned.values())
    if dict(counts) != dict(split_config["expected_counts"]):
        raise ValueError("JudgeBench split counts mismatch")
    canonical = "".join(
        f"{case_id}\0{assigned[case_id]}\0{pairs[case_id].source_group}\n"
        for case_id in sorted(assigned)
    ).encode("utf-8")
    if len(canonical) != split_config["manifest_bytes"]:
        raise ValueError("JudgeBench split manifest byte count mismatch")
    if sha256(canonical).hexdigest() != split_config["manifest_sha256"]:
        raise ValueError("JudgeBench split manifest digest mismatch")
    return assigned


def _load_reviewer_votes(
    path: Path,
    reviewer_row: Mapping[str, Any],
    pairs: Mapping[str, _PairRecord],
    registry: Mapping[str, Any],
) -> tuple[dict[str, Review], dict[str, int]]:
    reviewer_id = cast(str, reviewer_row["reviewer_id"])
    judge_model = cast(str, reviewer_row["declared_judge_model"])
    response_model = cast(str, registry["upstream"]["response_model"])
    judge_name = cast(str, registry["upstream"]["judge_name"])
    reviews: dict[str, Review] = {}
    kinds: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        case_id = _required_string(row, "pair_id")
        if case_id in reviews:
            raise ValueError(f"duplicate JudgeBench pair_id for {reviewer_id}")
        pair = pairs.get(case_id)
        if pair is None:
            raise ValueError(f"unknown JudgeBench pair_id for {reviewer_id}")
        if row.get("response_model") != response_model:
            raise ValueError(f"response_model mismatch for {reviewer_id}")
        if row.get("judge_name") != judge_name:
            raise ValueError(f"judge_name mismatch for {reviewer_id}")
        if (
            row.get("source") != pair.source
            or row.get("question") != pair.question
            or row.get("label") != pair.label
        ):
            raise ValueError(f"pair alignment mismatch for {reviewer_id}")
        judgments = row.get("judgments")
        if not isinstance(judgments, list) or len(judgments) != 2:
            raise ValueError(f"{reviewer_id} must contain exactly two judgments")
        decisions: list[object] = []
        for judgment in judgments:
            if not isinstance(judgment, dict):
                raise TypeError(f"{reviewer_id} judgment must be an object")
            payload = judgment.get("judgment")
            if not isinstance(payload, dict) or payload.get("judge_model") != judge_model:
                raise ValueError(f"nested judge_model mismatch for {reviewer_id}")
            decision = judgment.get("decision")
            _normalize_displayed_decision(decision)
            decisions.append(decision)
        observation = _normalize_decision_pair(decisions[0], decisions[1])
        kind = _normalization_kind(decisions[0], decisions[1])
        kinds[kind] += 1
        kinds[observation.value] += 1
        reviews[case_id] = Review(
            case_id=case_id,
            reviewer_id=reviewer_id,
            observation=observation,
            state=ExecutionState.VALID,
        )
    if set(reviews) != set(pairs):
        raise ValueError(f"vote pair_id coverage mismatch for {reviewer_id}")
    return reviews, dict(sorted(kinds.items()))


def _reviewer_panel(registry: Mapping[str, Any]) -> tuple[Reviewer, ...]:
    rows = cast(Sequence[Mapping[str, Any]], registry["reviewers"])
    reviewers = tuple(
        Reviewer(
            reviewer_id=cast(str, row["reviewer_id"]),
            vendor=cast(str, row["vendor"]),
            family=cast(str, row["family"]),
            lineage=cast(str, row["lineage"]),
            cost=cast(float, row["cost"]),
        )
        for row in rows
    )
    if len(reviewers) != 7 or len({row.reviewer_id for row in reviewers}) != 7:
        raise ValueError("JudgeBench panel must contain seven unique reviewers")
    if len({row.lineage for row in reviewers}) != 3:
        raise ValueError("JudgeBench panel must contain three declared lineages")
    return reviewers


def _assemble_cases(
    pairs: Mapping[str, _PairRecord],
    splits: Mapping[str, str],
    reviewers: Sequence[Reviewer],
    raw_files: Mapping[str, Path],
    registry: Mapping[str, Any],
) -> tuple[tuple[_GateCase, ...], dict[str, dict[str, int]]]:
    reviewer_rows = {
        cast(str, row["reviewer_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], registry["reviewers"])
    }
    votes: dict[str, dict[str, Review]] = {}
    normalization_counts: dict[str, dict[str, int]] = {}
    for reviewer in reviewers:
        reviewer_votes, counts = _load_reviewer_votes(
            raw_files[reviewer.reviewer_id],
            reviewer_rows[reviewer.reviewer_id],
            pairs,
            registry,
        )
        votes[reviewer.reviewer_id] = reviewer_votes
        normalization_counts[reviewer.reviewer_id] = counts
    cases = tuple(
        _GateCase(
            case_id=case_id,
            source=pairs[case_id].source,
            source_group=pairs[case_id].source_group,
            truth=pairs[case_id].truth,
            split=splits[case_id],
            reviews=tuple(
                votes[reviewer.reviewer_id][case_id] for reviewer in reviewers
            ),
        )
        for case_id in sorted(pairs)
    )
    if any(len(case.reviews) != 7 for case in cases):
        raise ValueError("every JudgeBench case must have seven reviews")
    return cases, normalization_counts


def _review_matrix(
    cases: Sequence[_GateCase],
    reviewer_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    column_by_reviewer = {
        reviewer_id: index for index, reviewer_id in enumerate(reviewer_ids)
    }
    observations = np.full((len(cases), len(reviewer_ids)), -1, dtype=np.int64)
    valid_mask = np.zeros_like(observations, dtype=bool)
    for row_index, case in enumerate(cases):
        seen: set[str] = set()
        for review in case.reviews:
            if review.reviewer_id in seen:
                raise ValueError("duplicate reviewer within JudgeBench case")
            seen.add(review.reviewer_id)
            if review.state is not ExecutionState.VALID or review.observation is None:
                raise ValueError("JudgeBench cached votes must be valid observations")
            column = column_by_reviewer[review.reviewer_id]
            observations[row_index, column] = OBSERVATION_ORDER.index(
                review.observation
            )
            valid_mask[row_index, column] = True
        if seen != set(reviewer_ids):
            raise ValueError("JudgeBench case reviewer set mismatch")
    if not np.all(valid_mask):
        raise ValueError("JudgeBench abstentions must remain valid observations")
    return observations, valid_mask


def _posterior_at(
    batch: BatchFusedPosterior,
    index: int,
) -> FusedPosterior | None:
    if int(batch.valid_reviewers[index]) == 0:
        return None
    return FusedPosterior(
        pass_probability=float(batch.pass_probability[index]),
        lower=float(batch.lower[index]),
        upper=float(batch.upper[index]),
        valid_reviewers=int(batch.valid_reviewers[index]),
        lineage_count=int(batch.lineage_count[index]),
        effective_sample_size=float(batch.effective_sample_size[index]),
        samples=(),
    )


def _posterior_mapping(
    batch: BatchFusedPosterior,
    cases: Sequence[_GateCase],
) -> dict[str, FusedPosterior | None]:
    return {
        case.case_id: _posterior_at(batch, index)
        for index, case in enumerate(cases)
    }


def _assert_finite_batch(batch: BatchFusedPosterior, expected_rows: int) -> None:
    arrays = (
        batch.pass_probability,
        batch.lower,
        batch.upper,
        batch.valid_reviewers,
        batch.lineage_count,
        batch.effective_sample_size,
    )
    if any(len(array) != expected_rows for array in arrays):
        raise ValueError("JudgeBench posterior row count mismatch")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("JudgeBench posterior contains a non-finite gate operand")
    if not np.all(batch.valid_reviewers == 7):
        raise ValueError("JudgeBench posterior must retain all seven semantic votes")
    if not np.all(batch.lineage_count == 3):
        raise ValueError("JudgeBench posterior must retain all three lineages")


def _canonical_policy(policy: DecisionPolicy) -> dict[str, int | float]:
    return {
        "pass_threshold": policy.pass_threshold,
        "fail_threshold": policy.fail_threshold,
        "min_valid_reviewers": policy.min_valid_reviewers,
        "min_lineages": policy.min_lineages,
        "min_effective_sample_size": policy.min_effective_sample_size,
    }


def _score_actions(
    cases: Sequence[_GateCase],
    actions: Mapping[str, Action],
) -> tuple[dict[str, object], dict[str, float]]:
    case_ids = tuple(case.case_id for case in cases)
    if set(actions) != set(case_ids):
        raise ValueError("action mapping must exactly cover the scored cases")
    return _reference_score(
        case_ids,
        tuple(case.truth for case in cases),
        tuple(actions[case_id] for case_id in case_ids),
    )


def _required_ratio_value(score: Mapping[str, object], name: str) -> float:
    ratio = cast(Mapping[str, object], score[name])
    value = ratio["value"]
    if value is None:
        raise ValueError(f"required JudgeBench ratio has zero denominator: {name}")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"required JudgeBench ratio must be numeric: {name}")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"required JudgeBench ratio is not finite: {name}")
    return numeric


def _method_scores(
    cases: Sequence[_GateCase],
    corum_actions: Mapping[str, Action],
    ordinary_actions: Mapping[str, Action],
    lineage_actions: Mapping[str, Action],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, float]],
]:
    scores: dict[str, dict[str, object]] = {}
    losses: dict[str, dict[str, float]] = {}
    for name, actions in (
        ("corum", corum_actions),
        ("ordinary_majority", ordinary_actions),
        ("lineage_balanced", lineage_actions),
    ):
        score, method_losses = _score_actions(cases, actions)
        scores[name] = score
        losses[name] = method_losses
    return scores, losses


def _run_locked_gate_impl() -> dict[str, object]:
    registry = _load_registry()
    registry_digest = _registry_digest(registry)
    if registry_digest != _EXPECTED_REGISTRY_SHA256:
        raise ValueError("JudgeBench registry snapshot digest mismatch")
    raw_value = os.environ.get("CORUM_JUDGEBENCH_RAW_DIR")
    upstream_value = os.environ.get("CORUM_JUDGEBENCH_UPSTREAM_REPO")
    if not raw_value or not upstream_value:
        raise ValueError(
            "formal JudgeBench run requires raw and upstream repository paths"
        )
    raw_directory = Path(raw_value).resolve()
    upstream_repository = Path(upstream_value).resolve()
    if not upstream_repository.is_dir():
        raise ValueError("pinned JudgeBench upstream repository is missing")

    inventory = _verify_upstream_inventory(upstream_repository, registry)
    raw_files = _verify_raw_files(raw_directory, registry)
    pairs = _load_pairs(raw_files["pairs"], registry)
    splits = _assign_splits(pairs, registry)
    reviewers = _reviewer_panel(registry)
    cases, normalization_counts = _assemble_cases(
        pairs,
        splits,
        reviewers,
        raw_files,
        registry,
    )
    split_counts = Counter(case.split for case in cases)
    expected_counts = dict(registry["split"]["expected_counts"])
    if dict(split_counts) != expected_counts:
        raise ValueError("assembled JudgeBench split counts mismatch")
    split_groups: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        split_groups[case.split].add(case.source_group)
    split_names = tuple(sorted(split_groups))
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            if not split_groups[first].isdisjoint(split_groups[second]):
                raise ValueError("JudgeBench source groups crossed split boundaries")

    fit_cases = tuple(case for case in cases if case.split == "fit")
    policy_cases = tuple(case for case in cases if case.split == "policy")
    test_cases = tuple(
        case for case in cases if case.split in {"test", "coding_test"}
    )
    coding_cases = tuple(case for case in cases if case.split == "coding_test")
    if (len(fit_cases), len(policy_cases), len(test_cases), len(coding_cases)) != (
        128,
        68,
        154,
        42,
    ):
        raise ValueError("JudgeBench fit/policy/test partition size mismatch")
    if {case.truth for case in policy_cases} != {Truth.PASS, Truth.FAIL}:
        raise ValueError("JudgeBench policy split must contain both truth classes")
    if {case.truth for case in test_cases} != {Truth.PASS, Truth.FAIL}:
        raise ValueError("JudgeBench pooled test must contain both truth classes")
    if {case.truth for case in coding_cases} != {Truth.PASS, Truth.FAIL}:
        raise ValueError("JudgeBench coding test must contain both truth classes")

    core = cast(Mapping[str, Any], registry["core"])
    fit_examples = tuple(
        CalibrationExample(case.truth, review)
        for case in fit_cases
        for review in case.reviews
    )
    calibrations = fit_panel_calibrations(
        reviewers,
        fit_examples,
        prior_strength=cast(float, core["prior_strength"]),
    )
    dependence = fit_dependence(
        reviewers,
        fit_examples,
        shrinkage=cast(float, core["dependence_shrinkage"]),
        min_overlap=cast(int, core["minimum_overlap"]),
        lineage_cap=cast(float, core["lineage_cap"]),
    )
    context = build_fusion_context(
        calibrations,
        dependence,
        prior_pass=cast(float, core["prior_pass"]),
        draws=cast(int, core["posterior_draws"]),
        credible_mass=cast(float, core["credible_mass"]),
        seed=cast(int, core["fusion_seed"]),
    )
    if dict(context.pair_likelihood_draws):
        raise ValueError("rejected pair-block path was active in JudgeBench gate")
    reviewer_ids = dependence.reviewer_ids

    policy_observations, policy_mask = _review_matrix(policy_cases, reviewer_ids)
    policy_batch = fuse_review_matrix(
        policy_observations,
        policy_mask,
        reviewer_ids,
        context,
        chunk_size=cast(int, core["chunk_size"]),
    )
    _assert_finite_batch(policy_batch, len(policy_cases))
    policy_posteriors = _posterior_mapping(policy_batch, policy_cases)
    selection = select_decision_policy(
        {case.case_id: case.truth for case in policy_cases},
        policy_posteriors,
        {case.case_id: () for case in policy_cases},
        costs=_COSTS,
        min_coverage=cast(float, core["policy_minimum_coverage"]),
    )

    test_observations, test_mask = _review_matrix(test_cases, reviewer_ids)
    test_batch = fuse_review_matrix(
        test_observations,
        test_mask,
        reviewer_ids,
        context,
        chunk_size=cast(int, core["chunk_size"]),
    )
    _assert_finite_batch(test_batch, len(test_cases))
    test_posteriors = _posterior_mapping(test_batch, test_cases)
    corum_decisions = {
        case.case_id: decide(
            test_posteriors[case.case_id],
            (),
            selection.policy,
        )
        for case in test_cases
    }
    corum_actions = {
        case_id: decision.action for case_id, decision in corum_decisions.items()
    }
    ordinary_actions: dict[str, Action] = {}
    lineage_actions: dict[str, Action] = {}
    lineage_by_reviewer = dependence.lineage_by_reviewer
    for case in test_cases:
        observations = {
            review.reviewer_id: cast(Observation, review.observation)
            for review in case.reviews
        }
        reference_ordinary = _reference_majority_observations(observations.values())
        production_ordinary = majority_decision(case.reviews)
        if production_ordinary is not reference_ordinary:
            raise ValueError("ordinary majority disagrees with independent reference")
        ordinary_actions[case.case_id] = reference_ordinary
        lineage_actions[case.case_id] = _reference_lineage_majority(
            observations,
            lineage_by_reviewer,
        )

    pooled_scores, pooled_losses = _method_scores(
        test_cases,
        corum_actions,
        ordinary_actions,
        lineage_actions,
    )
    coding_ids = {case.case_id for case in coding_cases}
    coding_scores, _ = _method_scores(
        coding_cases,
        {case_id: action for case_id, action in corum_actions.items() if case_id in coding_ids},
        {
            case_id: action
            for case_id, action in ordinary_actions.items()
            if case_id in coding_ids
        },
        {
            case_id: action
            for case_id, action in lineage_actions.items()
            if case_id in coding_ids
        },
    )

    truth_mapping = {case.case_id: case.truth for case in test_cases}
    probabilities = {
        case.case_id: float(test_batch.pass_probability[index])
        for index, case in enumerate(test_cases)
    }
    production_metrics = evaluate_decisions(
        truth_mapping,
        corum_actions,
        probabilities=probabilities,
        costs=_COSTS,
    )
    for field in ("decision_loss", "coverage"):
        reference_value = cast(float, pooled_scores["corum"][field])
        if abs(production_metrics[field] - reference_value) > 1e-12:
            raise ValueError(f"production metric disagrees with reference: {field}")
    probability_diagnostics = {
        "brier": production_metrics["brier"],
        "nll": production_metrics["log_loss"],
        "ece": production_metrics["ece"],
        "mean_interval_width": fsum(
            float(upper - lower)
            for lower, upper in zip(test_batch.lower, test_batch.upper, strict=True)
        )
        / len(test_cases),
    }
    if not all(isfinite(value) for value in probability_diagnostics.values()):
        raise ValueError("JudgeBench probability diagnostic is non-finite")

    source_scores: dict[str, dict[str, dict[str, object]]] = {}
    for source in sorted({case.source for case in test_cases}):
        source_cases = tuple(case for case in test_cases if case.source == source)
        source_ids = {case.case_id for case in source_cases}
        scores, _ = _method_scores(
            source_cases,
            {
                case_id: action
                for case_id, action in corum_actions.items()
                if case_id in source_ids
            },
            {
                case_id: action
                for case_id, action in ordinary_actions.items()
                if case_id in source_ids
            },
            {
                case_id: action
                for case_id, action in lineage_actions.items()
                if case_id in source_ids
            },
        )
        source_scores[source] = scores

    bootstrap_rows = tuple(
        {
            "case_id": case.case_id,
            "source": case.source,
            "corum_loss": pooled_losses["corum"][case.case_id],
            "ordinary_loss": pooled_losses["ordinary_majority"][case.case_id],
            "lineage_loss": pooled_losses["lineage_balanced"][case.case_id],
        }
        for case in test_cases
    )
    bootstrap_config = cast(Mapping[str, Any], registry["bootstrap"])
    intervals = _paired_source_bootstrap(
        bootstrap_rows,
        draws=cast(int, bootstrap_config["draws"]),
        seed=cast(int, bootstrap_config["seed"]),
    )

    gate = cast(Mapping[str, Any], registry["gate"])
    corum = pooled_scores["corum"]
    ordinary = pooled_scores["ordinary_majority"]
    lineage = pooled_scores["lineage_balanced"]
    coding_corum = coding_scores["corum"]
    coding_ordinary = coding_scores["ordinary_majority"]
    coding_lineage = coding_scores["lineage_balanced"]
    corum_loss = cast(float, corum["decision_loss"])
    ordinary_loss = cast(float, ordinary["decision_loss"])
    lineage_loss = cast(float, lineage["decision_loss"])
    corum_coverage = cast(float, corum["coverage"])
    corum_useful = cast(float, corum["useful_resolution"])
    guardrails = {
        "policy_constraint_satisfied": selection.constraint_satisfied,
        "no_point_harm_vs_ordinary": corum_loss <= ordinary_loss,
        "no_point_harm_vs_lineage": corum_loss <= lineage_loss,
        "coverage_absolute": corum_coverage >= gate["coverage_min"],
        "coverage_vs_ordinary": corum_coverage
        >= cast(float, ordinary["coverage"]) - gate["coverage_gap_max"],
        "coverage_vs_lineage": corum_coverage
        >= cast(float, lineage["coverage"]) - gate["coverage_gap_max"],
        "useful_resolution_vs_ordinary": corum_useful
        >= cast(float, ordinary["useful_resolution"])
        - gate["useful_resolution_gap_max"],
        "useful_resolution_vs_lineage": corum_useful
        >= cast(float, lineage["useful_resolution"])
        - gate["useful_resolution_gap_max"],
        "false_pass_vs_ordinary": _required_ratio_value(corum, "false_pass_rate")
        <= _required_ratio_value(ordinary, "false_pass_rate")
        + gate["directional_error_gap_max"],
        "false_pass_vs_lineage": _required_ratio_value(corum, "false_pass_rate")
        <= _required_ratio_value(lineage, "false_pass_rate")
        + gate["directional_error_gap_max"],
        "false_fail_vs_ordinary": _required_ratio_value(corum, "false_fail_rate")
        <= _required_ratio_value(ordinary, "false_fail_rate")
        + gate["directional_error_gap_max"],
        "false_fail_vs_lineage": _required_ratio_value(corum, "false_fail_rate")
        <= _required_ratio_value(lineage, "false_fail_rate")
        + gate["directional_error_gap_max"],
        "coding_loss_vs_ordinary": cast(float, coding_corum["decision_loss"])
        <= cast(float, coding_ordinary["decision_loss"])
        + gate["coding_loss_gap_max"],
        "coding_loss_vs_lineage": cast(float, coding_corum["decision_loss"])
        <= cast(float, coding_lineage["decision_loss"])
        + gate["coding_loss_gap_max"],
        "coding_coverage_absolute": cast(float, coding_corum["coverage"])
        >= gate["coding_coverage_min"],
        "coding_useful_resolution_vs_ordinary": cast(
            float,
            coding_corum["useful_resolution"],
        )
        >= cast(float, coding_ordinary["useful_resolution"])
        - gate["coding_useful_resolution_gap_max"],
        "coding_useful_resolution_vs_lineage": cast(
            float,
            coding_corum["useful_resolution"],
        )
        >= cast(float, coding_lineage["useful_resolution"])
        - gate["coding_useful_resolution_gap_max"],
    }
    relative_targets = {
        "ordinary_loss_reduction": corum_loss
        <= gate["ordinary_loss_ratio_max"] * ordinary_loss,
        "lineage_loss_reduction": corum_loss
        <= gate["lineage_loss_ratio_max"] * lineage_loss,
    }
    confidence = {
        "ordinary_positive_lower_bound": intervals["ordinary"][1] > 0.0,
        "lineage_positive_lower_bound": intervals["lineage"][1] > 0.0,
    }
    gate_operands = (
        corum_loss,
        ordinary_loss,
        lineage_loss,
        corum_coverage,
        corum_useful,
        *intervals["ordinary"],
        *intervals["lineage"],
    )
    if not all(isfinite(float(value)) for value in gate_operands):
        raise ValueError("JudgeBench gate operand is non-finite")
    verdict = _classify_verdict(
        integrity_ok=True,
        guardrails_ok=all(guardrails.values()),
        relative_targets_ok=all(relative_targets.values()),
        confidence_ok=all(confidence.values()),
    )

    return {
        "schema_version": "1",
        "gate_id": registry["gate_id"],
        "verdict": verdict,
        "registry_sha256": registry_digest,
        "corum_commit": _git_text(_REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "upstream": inventory,
        "counts": {
            "total": len(cases),
            "fit": len(fit_cases),
            "policy": len(policy_cases),
            "pooled_test": len(test_cases),
            "general_test": len(test_cases) - len(coding_cases),
            "coding_test": len(coding_cases),
            "reviewers": len(reviewers),
            "lineages": len({reviewer.lineage for reviewer in reviewers}),
        },
        "selected_policy": _canonical_policy(selection.policy),
        "policy_selection": {
            "constraint_satisfied": selection.constraint_satisfied,
            "decision_loss": selection.decision_loss,
            "coverage": selection.coverage,
        },
        "normalization_counts": normalization_counts,
        "decision_reason_counts": {
            "corum": _aggregate_reason_counts(
                decision.reasons for decision in corum_decisions.values()
            ),
            "ordinary_majority": {
                "tie_or_no_semantic_vote": sum(
                    action is Action.DEFER for action in ordinary_actions.values()
                )
            },
            "lineage_balanced": {
                "lineage_tie_or_no_semantic_vote": sum(
                    action is Action.DEFER for action in lineage_actions.values()
                )
            },
        },
        "pooled": pooled_scores,
        "coding": coding_scores,
        "per_source": source_scores,
        "corum_probability": probability_diagnostics,
        "paired_benefit_intervals": {
            name: {
                "point": values[0],
                "lower": values[1],
                "upper": values[2],
            }
            for name, values in intervals.items()
        },
        "checks": {
            "integrity": True,
            "guardrails": guardrails,
            "relative_targets": relative_targets,
            "confidence": confidence,
        },
        "claim_boundary": (
            "static answer-comparison evidence only; not repository-patch, adoption, "
            "production, or universal-superiority validation"
        ),
    }


def _run_locked_gate() -> dict[str, object]:
    try:
        return _run_locked_gate_impl()
    except Exception as error:  # noqa: BLE001 - INVALID is a registered gate outcome
        return {
            "schema_version": "1",
            "gate_id": "judgebench-external-v1",
            "verdict": "INVALID",
            "registry_sha256": _safe_registry_digest(),
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:500],
            },
        }


def test_synthetic_order_normalization_matches_upstream_sign_rule() -> None:
    assert _normalize_decision_pair("A>B", "B>A") is Observation.PASS
    assert _normalize_decision_pair("A>B", "A=B") is Observation.PASS
    assert _normalize_decision_pair("A>B", "A>B") is Observation.ABSTAIN
    assert _normalize_decision_pair("B>A", "A>B") is Observation.FAIL
    assert _normalize_decision_pair(None, "A>B") is Observation.FAIL
    assert _normalize_decision_pair("B<A", "A<B") is Observation.PASS
    assert _normalize_decision_pair(None, None) is Observation.ABSTAIN
    with pytest.raises(ValueError, match="unknown JudgeBench decision") as error:
        _normalize_decision_pair("SECRET_RAW_DECISION", "A>B")
    assert "SECRET_RAW_DECISION" not in str(error.value)


def test_synthetic_invalid_fallback_never_reraises_registry_error() -> None:
    def broken_loader() -> dict[str, Any]:
        raise ValueError("SECRET_RAW_REGISTRY")

    assert _safe_registry_digest(broken_loader) is None


def test_synthetic_materialized_bytes_match_git_blob_identity(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "fixture.jsonl"
    materialized.write_bytes(b'{"decision":"A>B"}\n')
    expected = subprocess.run(
        ("git", "hash-object", "--no-filters", "--stdin"),
        input=materialized.read_bytes(),
        check=True,
        capture_output=True,
    ).stdout.decode("ascii").strip()

    assert _git_blob_oid(materialized) == expected


def test_synthetic_vote_jsonl_preserves_declared_model_and_valid_abstain(
    tmp_path: Path,
) -> None:
    declared_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pair = _PairRecord(
        case_id="pair-1",
        source="synthetic",
        question="Which response is better?",
        label="A>B",
        truth=Truth.PASS,
        source_group="group-1",
    )
    row = {
        "pair_id": pair.case_id,
        "source": pair.source,
        "question": pair.question,
        "label": pair.label,
        "response_model": "gpt-4o-2024-05-13",
        "judge_name": "arena_hard",
        "judgments": [
            {
                "decision": "A>B",
                "judgment": {"judge_model": declared_model},
            },
            {
                "decision": "A>B",
                "judgment": {"judge_model": declared_model},
            },
        ],
    }
    fixture = tmp_path / "votes.jsonl"
    fixture.write_text(json.dumps(row) + "\n", encoding="utf-8")

    reviews, counts = _load_reviewer_votes(
        fixture,
        {
            "reviewer_id": "llama-3.1-8b",
            "declared_judge_model": declared_model,
        },
        {pair.case_id: pair},
        {
            "upstream": {
                "response_model": "gpt-4o-2024-05-13",
                "judge_name": "arena_hard",
            }
        },
    )

    assert reviews[pair.case_id].state is ExecutionState.VALID
    assert reviews[pair.case_id].observation is Observation.ABSTAIN
    assert counts == {"ABSTAIN": 1, "order_conflict": 1}


def test_synthetic_lineage_vote_prevents_one_family_from_counting_three_times() -> None:
    observations = {
        "anthropic-a": Observation.FAIL,
        "anthropic-b": Observation.FAIL,
        "google-a": Observation.ABSTAIN,
        "google-b": Observation.ABSTAIN,
        "meta-a": Observation.PASS,
        "meta-b": Observation.PASS,
        "meta-c": Observation.PASS,
    }
    lineages = {
        "anthropic-a": "anthropic",
        "anthropic-b": "anthropic",
        "google-a": "google",
        "google-b": "google",
        "meta-a": "meta",
        "meta-b": "meta",
        "meta-c": "meta",
    }

    assert _reference_majority_observations(observations.values()) is Action.PASS
    assert _reference_lineage_majority(observations, lineages) is Action.DEFER


def test_synthetic_score_preserves_zero_denominator_as_null() -> None:
    score, losses = _reference_score(
        ("pass-case", "fail-case"),
        (Truth.PASS, Truth.FAIL),
        (Action.DEFER, Action.DEFER),
    )

    assert losses == {"pass-case": 0.25, "fail-case": 0.25}
    assert score["coverage"] == 0.0
    assert score["decision_loss"] == 0.25
    assert score["false_safe_risk"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }
    assert score["selective_risk"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }


def test_synthetic_decision_reasons_are_aggregate_only() -> None:
    assert _aggregate_reason_counts(
        (
            ("posterior_uncertain",),
            ("insufficient_effective_sample_size", "posterior_uncertain"),
            ("pass_threshold_met",),
        )
    ) == {
        "insufficient_effective_sample_size": 1,
        "pass_threshold_met": 1,
        "posterior_uncertain": 2,
    }


def test_synthetic_bootstrap_is_deterministic_and_reuses_samples() -> None:
    rows = (
        {
            "case_id": "a-1",
            "source": "a",
            "corum_loss": 0.1,
            "ordinary_loss": 0.2,
            "lineage_loss": 0.3,
        },
        {
            "case_id": "a-2",
            "source": "a",
            "corum_loss": 0.2,
            "ordinary_loss": 0.3,
            "lineage_loss": 0.4,
        },
        {
            "case_id": "b-1",
            "source": "b",
            "corum_loss": 0.4,
            "ordinary_loss": 0.5,
            "lineage_loss": 0.6,
        },
    )

    first = _paired_source_bootstrap(rows, draws=40, seed=7)
    second = _paired_source_bootstrap(tuple(reversed(rows)), draws=40, seed=7)

    assert first == second
    assert first["ordinary"] == pytest.approx((0.1, 0.1, 0.1), abs=1e-15)
    assert first["lineage"] == pytest.approx((0.2, 0.2, 0.2), abs=1e-15)


def test_synthetic_verdict_has_four_unambiguous_outcomes() -> None:
    assert (
        _classify_verdict(
            integrity_ok=False,
            guardrails_ok=False,
            relative_targets_ok=False,
            confidence_ok=False,
        )
        == "INVALID"
    )
    assert (
        _classify_verdict(
            integrity_ok=True,
            guardrails_ok=False,
            relative_targets_ok=True,
            confidence_ok=True,
        )
        == "FAIL"
    )
    assert (
        _classify_verdict(
            integrity_ok=True,
            guardrails_ok=True,
            relative_targets_ok=False,
            confidence_ok=True,
        )
        == "INCONCLUSIVE"
    )
    assert (
        _classify_verdict(
            integrity_ok=True,
            guardrails_ok=True,
            relative_targets_ok=True,
            confidence_ok=True,
        )
        == "PASS"
    )


def test_registry_snapshot_is_locked_and_outcome_blind() -> None:
    registry = _load_registry()
    candidates = registry["upstream"]["candidates"]
    files = registry["upstream"]["files"]
    raw_url_base = registry["upstream"]["raw_url_base"]

    assert _registry_digest(registry) == _EXPECTED_REGISTRY_SHA256
    assert len(candidates) == 11
    assert sum(bool(candidate["eligible"]) for candidate in candidates) == 7
    assert len(files) == 8
    assert all((raw_url_base + row["path"]).startswith("https://") for row in files)
    assert registry["split"]["expected_counts"] == {
        "fit": 128,
        "policy": 68,
        "test": 112,
        "coding_test": 42,
    }
    assert registry["costs"] == {
        "false_pass": 1.0,
        "false_fail": 1.0,
        "defer": 0.25,
    }
    assert registry["bootstrap"] == {
        "draws": 5000,
        "seed": 20260830,
        "lower_quantile": 0.025,
        "upper_quantile": 0.975,
        "quantile_method": "linear",
        "strict_positive_lower_bound": True,
    }
    assert registry["gate"]["coding_loss_gap_max"] == 1.0 / 42.0
    llama_reviewers = [
        row for row in registry["reviewers"] if row["vendor"] == "meta"
    ]
    assert {
        row["declared_judge_model"] for row in llama_reviewers
    } == {
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
    }


@pytest.mark.skipif(
    os.environ.get("CORUM_RUN_JUDGEBENCH_V1") != "1",
    reason="locked external gate requires the explicit one-time run switch",
)
def test_locked_judgebench_external_value_gate() -> None:
    result = _run_locked_gate()
    print("CORUM_JUDGEBENCH_RESULT=" + _result_json(result))
    assert result["verdict"] == "PASS", result["verdict"]
