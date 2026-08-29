"""Task 6E's independent judge; ordinary tests use only tiny non-formal fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from corum.baselines import DecisionCosts
from corum.calibration import fit_panel_calibrations
from corum.dependence import fit_dependence
from corum.fusion import build_fusion_context, fuse_review_matrix, fuse_reviews
from corum.metrics import evaluate_decisions
from corum.models import (
    Action,
    CalibrationExample,
    ExecutionState,
    Observation,
    Review,
    Reviewer,
    Truth,
)
from corum.simulation import ReviewerSpec, Scenario, ScenarioPhase, simulate_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/convergence-resolution-v1.json"
RESULTS = tuple(
    ROOT / f"docs/results/task-6e-convergence-resolution-attempt-0.{x}"
    for x in ("txt", "json", "md")
)
RESULT_RELATIVE_PATHS = tuple(
    Path(f"docs/results/task-6e-convergence-resolution-attempt-0.{extension}")
    for extension in ("txt", "json", "md")
)
STATUS_PATHS = (
    Path("AGENTS.md"),
    Path("docs/plans/corum-mvp.md"),
    Path("docs/sdd/0010-convergence-resolution-gate.md"),
    Path("docs/specs/corum-mvp-design.md"),
)
RESULT_COMMIT_SUBJECT = "docs: record convergence resolution gate result"
RECORDER_FAULTS = frozenset(
    {
        "after_json_fsync",
        "after_md_fsync",
        "after_status_agents",
        "after_status_plan",
        "after_status_sdd",
        "after_status_design",
        "before_commit",
        "after_commit_ref_update",
    }
)
STATUS_PRE_SHA256 = {
    Path("AGENTS.md"): "930c5512eb27b3eab70f2c13d6461153f7913ba53c9ece6990bc0410153f7d3e",
    Path("docs/plans/corum-mvp.md"): "775109ce94813a13ecbfc0b2806c0c1e3c027c666df9ce192240d4c120145d20",
    Path("docs/sdd/0010-convergence-resolution-gate.md"): "3991832ce08b157a1875b4f897363d1a48360697549de75f0f613307f2a3ffe5",
    Path("docs/specs/corum-mvp-design.md"): "2909449580f2a4eb3480b38ca7d7eb955ab678c984ca99cae79bf667f376092d",
}
SWITCH = "CORUM_RUN_CONVERGENCE_V1"
METHODS = ("candidate", "ordinary_majority", "reliability_weighted")
REASONS = frozenset(
    ["INVALID_EXCEPTION", "INVALID_SEED_REGENERATION", "INVALID_SIMULATION_ORDER", "INVALID_PERTURBATION_MULTISET", "INVALID_COUNTS", "INVALID_NONFINITE", "INVALID_SHARED_AB_ROWS", "INVALID_FIT_ROWS", "INVALID_MODEL_CALLS", "INVALID_REPLAY", "INVALID_RESULT_CANONICAL", "RECORDER_START_ONLY", "RECORDER_PARTIAL_FINAL", "RECORDER_MALFORMED_FINAL", "FAIL_ACCURACY_POINT_ORDINARY", "FAIL_ACCURACY_POINT_WEIGHTED", "FAIL_COVERAGE_FLOOR", "FAIL_COVERAGE_GAP_ORDINARY", "FAIL_COVERAGE_GAP_WEIGHTED", "FAIL_DISPERSION_POINT_ORDINARY", "FAIL_DISPERSION_POINT_WEIGHTED", "FAIL_FALSE_SAFE_POINT_ORDINARY", "FAIL_FALSE_SAFE_POINT_WEIGHTED", "FAIL_SCENARIO_ACCURACY", "FAIL_SCENARIO_COVERAGE", "FAIL_SCENARIO_FALSE_SAFE", "INCONCLUSIVE_ZERO_DISPERSION_ORDINARY", "INCONCLUSIVE_ZERO_DISPERSION_WEIGHTED", "INCONCLUSIVE_ACCURACY_CI_ORDINARY", "INCONCLUSIVE_ACCURACY_CI_WEIGHTED", "INCONCLUSIVE_DISPERSION_CI_ORDINARY", "INCONCLUSIVE_DISPERSION_CI_WEIGHTED", "INCONCLUSIVE_FALSE_SAFE_CI_ORDINARY", "INCONCLUSIVE_FALSE_SAFE_CI_WEIGHTED"]
)

PINNED_SCENARIO_SHA256 = "5126f4a1d4c0d7cd97dccd6a860ed7dea45e69c23ecd2da6692b188a6198619c"
PINNED_SEED_SHA256 = "3d7cfb42bb5f48a11410a5187d13213e6e30446a40f00832e00aa19700c0ea29"
PINNED_CONFIG_SHA256 = "30185a170545f8582d35a5f041dd902002396ec17a9dce79cf3d69790a2d2ece"
DOCUMENTATION_COMMIT = "044edd5c29fe81d4d0cefe45a45e445273bf4738"
PINNED_PHASE_SHA256 = {"adversarial-shift-v1:calibration":"9d5338463748d61be8e6aab5cc88f92bbd015c2d0ccf625f07cdad89a29019f8","adversarial-shift-v1:test":"8332752e44bbb706f8bdd9f9c0b5ec71407aec04a711b0692fa17748191dce35","clone-pressure-v1:calibration":"950de4057f4fdbbdfc95973837e5220482337866016eacb67d1ed68579ae78aa","clone-pressure-v1:test":"950de4057f4fdbbdfc95973837e5220482337866016eacb67d1ed68579ae78aa","dependence-shift-v1:calibration":"08e72e3d4211548660130f8e4fd8fcd567bebf8fec75fc519dd08bed09bb2f0d","dependence-shift-v1:test":"d95e3cd22ffc4b70a973d07bd85ecc58e4c6d9e992a214ad3ac7c0698fc09ae5","independent-balanced-v1:calibration":"878fa5bd851a07388bcd01aeb983ed64dd6eae07506f2dd449aa49fa647f6ffa","independent-balanced-v1:test":"878fa5bd851a07388bcd01aeb983ed64dd6eae07506f2dd449aa49fa647f6ffa","informative-missing-v1:calibration":"164803c5fec2d77d26b9e5c45c1a1d67a33687fa9c7d87377fa68f85664070ba","informative-missing-v1:test":"5ee04a69c309cc98f5ebeb252fea0be136fbbf352d5d4d462074701a3fd7143d","majority-trap-v1:calibration":"05720793599a20cd21be01c90ee910cc68b3046076d8ed4dde3b84b89dfc9ffb","majority-trap-v1:test":"05720793599a20cd21be01c90ee910cc68b3046076d8ed4dde3b84b89dfc9ffb"}


class JudgeInvalid(RuntimeError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(codes)))
        super().__init__(",".join(self.codes))


class RecorderConflict(RuntimeError):
    """A preserved artifact or Git state differs from the bound attempt."""


class RecorderFault(RuntimeError):
    """Deterministic test-only interruption at a durable publication boundary."""


@dataclass(frozen=True)
class ParsedLedger:
    start: dict[str, Any] | None
    result: dict[str, Any]


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def exact_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(cast(dict[Any, Any], value)) == set(expected) and all(exact_equal(cast(dict[Any, Any], value)[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(cast(list[Any], value)) == len(expected) and all(exact_equal(a, b) for a, b in zip(cast(list[Any], value), expected, strict=True))
    return value == expected


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out = {}
    normalized = set()
    for key, value in pairs:
        n = unicodedata.normalize("NFC", key)
        if n != key or n in normalized:
            raise ValueError("duplicate/non-NFC key")
        normalized.add(n)
        out[key] = value
    return out


def strict(raw: bytes) -> object:
    if raw.startswith(b"\xef\xbb\xbf") or raw.endswith(b"\n"):
        raise ValueError("BOM/newline")
    text = raw.decode("utf-8")
    if re.search(r"(?<![A-Za-z])-(?:0(?:\.0*)?|0[eE][+-]?\d+)(?![\d.])", text):
        raise ValueError("negative zero")
    value = json.loads(
        text,
        object_pairs_hook=_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )

    def walk(x: object) -> None:
        if isinstance(x, str) and unicodedata.normalize("NFC", x) != x:
            raise ValueError("non-NFC")
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("nonfinite")
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for k, y in x.items():
                walk(k)
                walk(y)

    walk(value)
    if canonical(value) != raw:
        raise ValueError("noncanonical")
    return value


def load_config() -> dict[str, Any]:
    value = strict(CONFIG.read_bytes())
    assert isinstance(value, dict)
    keys = {
        "accepted_base",
        "bootstrap",
        "external_switch",
        "fusion",
        "gate_id",
        "metrics",
        "perturbation",
        "phase_sha256",
        "runtime",
        "sample_design",
        "scenario_sha256",
        "scenarios",
        "schema_version",
        "seed_table_sha256",
        "seeds",
        "verdict",
    }
    if set(value) != keys:
        raise ValueError("config schema")
    if value["accepted_base"] != "b34e0896c3cb80c325288d7057247f8b25fa72ab":
        raise ValueError("accepted base")
    exact = {
        "schema_version": "1", "gate_id": "convergence-resolution-v1",
        "external_switch": {"name": SWITCH, "required_value": "1"},
        "sample_design": {"scenario_count": 6, "replicates_per_scenario": 40, "fit_cases_per_block": 8000, "test_cases_per_block": 10000, "total_blocks": 240, "total_fit_cases": 1920000, "total_test_cases": 2400000},
        "fusion": {"prior_strength": 1.5, "dependence_shrinkage": .25, "minimum_overlap": 10, "lineage_cap": 1.0, "prior_pass": .5, "posterior_draws": 512, "credible_mass": .95, "chunk_size": 4096},
        "perturbation": {"algorithm": "same-truth-row-rotation-v1", "selection_fraction": .15, "rotation": "B[selected[(k+1)%m]]=A[selected[k]]"},
        "bootstrap": {"draws": 10000, "seed": 20260901, "quantiles": [.025, .975], "quantile_method": "linear"},
        "metrics": {"false_pass_cost": 1.0, "false_fail_cost": .2, "defer_cost": 1.0, "correct_cost": 0.0, "probability_clip": 1e-15, "ece_bins": 10},
        "verdict": {"accuracy_advantage_min": .05, "coverage_min": .98, "coverage_baseline_gap_max": .01, "dispersion_ratio_max": .70, "scenario_accuracy_gap_max": .01, "scenario_coverage_min": .97, "false_safe_ci_upper_max": .005, "scenario_false_safe_gap_max": .005},
    }
    for key, expected in exact.items():
        if not exact_equal(value[key], expected):
            raise ValueError(f"config literal: {key}")
    if not exact_equal(value["runtime"], {"corum": "0.1.0", "numpy": "2.5.2", "python": "3.14.0"}):
        raise ValueError("runtime schema")
    if len(value["scenarios"]) != 6 or len(value["seeds"]) != 240 or set(value["phase_sha256"]) != {f"{s['name']}:{p}" for s in value["scenarios"] for p in ("calibration", "test")}:
        raise ValueError("registered cardinality")
    for index, row in enumerate(value["seeds"]):
        if set(row) != {"scenario", "replicate", "simulation", "fusion", "perturbation"} or type(row["replicate"]) is not int or row["replicate"] != index % 40 or any(type(row[p]) is not int or not 0 <= row[p] < 2**64 for p in ("simulation", "fusion", "perturbation")):
            raise ValueError("seed schema")
    if value["scenario_sha256"] != PINNED_SCENARIO_SHA256 or value["seed_table_sha256"] != PINNED_SEED_SHA256 or value["phase_sha256"] != PINNED_PHASE_SHA256:
        raise ValueError("independent digest")
    if digest(value["scenarios"]) != PINNED_SCENARIO_SHA256 or digest(value["seeds"]) != PINNED_SEED_SHA256:
        raise ValueError("digest")
    for scenario in value["scenarios"]:
        if set(scenario) != {"name", "calibration", "test"}:
            raise ValueError("scenario schema")
        for p in ("calibration", "test"):
            if digest(scenario[p]) != value["phase_sha256"][f"{scenario['name']}:{p}"]:
                raise ValueError("phase digest")
            phase_value = scenario[p]
            if set(phase_value) != {"prior_pass", "difficulty_rate", "informative_missingness", "lineage_error_correlation", "reviewers"}:
                raise ValueError("phase schema")
            if len(phase_value["reviewers"]) != 3:
                raise ValueError("reviewer cardinality")
            for reviewer_value in phase_value["reviewers"]:
                if set(reviewer_value) != {"reviewer_id", "lineage", "accuracy", "abstain", "timeout_rate", "invalid_rate"}:
                    raise ValueError("reviewer schema")
                if any(isinstance(reviewer_value[k], bool) or not isinstance(reviewer_value[k], (int, float)) or not math.isfinite(reviewer_value[k]) for k in ("accuracy", "abstain", "timeout_rate", "invalid_rate")):
                    raise ValueError("reviewer numeric")
                if reviewer_value["accuracy"] + reviewer_value["abstain"] > 1 or any(not 0 <= reviewer_value[k] <= 1 for k in ("accuracy", "abstain", "timeout_rate", "invalid_rate")):
                    raise ValueError("reviewer range")
    return value


def seed(name: str, replicate: int, purpose: str) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"corum:convergence-resolution:v1\0{name}\0{replicate}\0{purpose}".encode()
        ).digest()[:8],
        "big",
    )


def seed_table(config: Mapping[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "scenario": s["name"],
            "replicate": r,
            **{
                p: seed(s["name"], r, p)
                for p in ("simulation", "fusion", "perturbation")
            },
        }
        for s in config["scenarios"]
        for r in range(40)
    ]


def reviewer(x: Mapping[str, Any]) -> ReviewerSpec:
    a, s = float(x["accuracy"]), float(x["abstain"])
    return ReviewerSpec(
        Reviewer(x["reviewer_id"], "simulated", "general", x["lineage"], 1.0),
        np.array([[a, 1 - a - s, s], [1 - a - s, a, s]]),
        timeout_rate=x["timeout_rate"],
        invalid_rate=x["invalid_rate"],
    )


def phase(x: Mapping[str, Any]) -> ScenarioPhase:
    return ScenarioPhase(
        tuple(reviewer(r) for r in x["reviewers"]),
        x["prior_pass"],
        x["lineage_error_correlation"],
        difficulty_rate=x["difficulty_rate"],
        informative_missingness=x["informative_missingness"],
    )


def fixture() -> Scenario:
    def p(
        prior: float,
        diff: float,
        miss: float,
        rows: Sequence[tuple[str, float, float, float, float]],
    ) -> ScenarioPhase:
        return ScenarioPhase(
            tuple(
                reviewer(
                    {
                        "reviewer_id": i,
                        "lineage": i,
                        "accuracy": a,
                        "abstain": s,
                        "timeout_rate": t,
                        "invalid_rate": v,
                    }
                )
                for i, a, s, t, v in rows
            ),
            prior,
            {},
            difficulty_rate=diff,
            informative_missingness=miss,
        )

    return Scenario(
        "fixture-phase-split-v1",
        p(
            0.52,
            0.25,
            0.10,
            (
                ("fixture-a", 0.73, 0.11, 0, 0),
                ("fixture-b", 0.68, 0.07, 0.05, 0),
                ("fixture-c", 0.81, 0.03, 0, 0.04),
            ),
        ),
        p(
            0.47,
            0.40,
            0.20,
            (
                ("fixture-a", 0.70, 0.12, 0, 0),
                ("fixture-b", 0.64, 0.08, 0.04, 0.01),
                ("fixture-c", 0.79, 0.04, 0.01, 0.03),
            ),
        ),
    )


def panel_object(panel: Any) -> dict[str, object]:
    return {
        "difficulty": sorted((k, v) for k, v in panel.difficulty_by_case.items()),
        "reviews": [
            [
                r.case_id,
                r.reviewer_id,
                r.state.value,
                r.observation.value if r.observation else None,
            ]
            for r in panel.reviews
        ],
        "truths": sorted((k, v.value) for k, v in panel.truths.items()),
    }


def sign(r: Review) -> int:
    return (
        1
        if r.state is ExecutionState.VALID and r.observation is Observation.PASS
        else -1
        if r.state is ExecutionState.VALID and r.observation is Observation.FAIL
        else 0
    )


def fit_weights(rows: Sequence[CalibrationExample]) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for x in rows:
        if not sign(x.review):
            continue
        n, c = counts.setdefault(x.review.reviewer_id, [0, 0])
        counts[x.review.reviewer_id] = [
            n + 1,
            c + int((sign(x.review) > 0) == (x.truth is Truth.PASS)),
        ]
    return {
        r: math.log(((c + 1) / (n + 2)) / (1 - (c + 1) / (n + 2)))
        for r, (n, c) in counts.items()
    }


def vote(
    rows: Sequence[Review], weights: Mapping[str, float] | None = None
) -> tuple[Action, float]:
    score = math.fsum(
        (1.0 if weights is None else weights.get(r.reviewer_id, 0.0)) * sign(r)
        for r in sorted(rows, key=lambda x: x.reviewer_id)
    )
    action = Action.PASS if score > 0 else Action.FAIL if score < 0 else Action.DEFER
    if weights is None:
        p = sum(sign(r) > 0 for r in rows)
        f = sum(sign(r) < 0 for r in rows)
        prob = (p + 1) / (p + f + 2)
    else:
        prob = 1 / (1 + math.exp(-score)) if score else 0.5
    return action, prob


def candidate_action(prob: float, directional_count: int) -> Action:
    if directional_count < 2:
        return Action.DEFER
    return Action.PASS if prob > 0.5 else Action.FAIL if prob < 0.5 else Action.DEFER


def candidate(prob: float, rows: Sequence[Review]) -> Action:
    return candidate_action(prob, sum(sign(r) != 0 for r in rows))


def rotate(
    rows: Sequence[tuple[object, ...]],
    truths: Sequence[Truth],
    block_seed: int,
    fraction: float = 0.15,
) -> tuple[tuple[object, ...], ...]:
    if len(rows) != len(truths) or not rows or not 0 <= fraction <= 1:
        raise JudgeInvalid("INVALID_PERTURBATION_MULTISET")
    out = list(rows)
    for truth in (Truth.PASS, Truth.FAIL):
        ids = np.array([i for i, t in enumerate(truths) if t is truth])
        selected = np.random.Generator(np.random.PCG64(block_seed)).permutation(ids)[
            : math.floor(fraction * len(ids))
        ]
        if len(selected) < 2:
            raise JudgeInvalid("INVALID_PERTURBATION_MULTISET")
        if len(selected) != math.floor(fraction * len(ids)) or len({int(x) for x in selected}) != len(selected):
            raise JudgeInvalid("INVALID_PERTURBATION_MULTISET")
        for k, source in enumerate(selected):
            out[int(selected[(k + 1) % len(selected)])] = rows[int(source)]
        selected_set = {int(x) for x in selected}
        if any(out[int(i)] != rows[int(i)] for i in ids if int(i) not in selected_set):
            raise JudgeInvalid("INVALID_PERTURBATION_MULTISET")
    return tuple(out)


def stats(
    truths: Sequence[Truth],
    aa: Sequence[Action],
    ab: Sequence[Action],
    probs: Sequence[float],
    method: str,
) -> dict[str, Any]:
    n = len(truths)
    ca = Counter(x.value for x in aa)
    cb = Counter(x.value for x in ab)
    correct = sum(
        a is not Action.DEFER and ((a is Action.PASS) == (t is Truth.PASS))
        for a, t in zip(aa, truths, strict=True)
    )
    fp = sum(
        a is Action.PASS and t is Truth.FAIL for a, t in zip(aa, truths, strict=True)
    )
    ff = sum(
        a is Action.FAIL and t is Truth.PASS for a, t in zip(aa, truths, strict=True)
    )
    bins = []
    for k in range(10):
        ids = [i for i, p in enumerate(probs) if min(int(p * 10), 9) == k]
        bins.append(
            {
                "bin_index": k,
                "count": len(ids),
                "pass_count": sum(truths[i] is Truth.PASS for i in ids),
                "probability_sum_a": math.fsum(probs[i] for i in ids),
            }
        )
    out = {
        "action_counts_a": {x: ca[x] for x in ("DEFER", "FAIL", "PASS")},
        "action_counts_b": {x: cb[x] for x in ("DEFER", "FAIL", "PASS")},
        "brier_sum_a": math.fsum(
            (p - (t is Truth.PASS)) ** 2 for p, t in zip(probs, truths, strict=True)
        ),
        "correct_count_a": correct,
        "covered_count_a": n - ca["DEFER"],
        "defer_count_a": ca["DEFER"],
        "dispersion_change_count": sum(a is not b for a, b in zip(aa, ab, strict=True)),
        "ece_bins_a": bins,
        "false_fail_count_a": ff,
        "false_pass_count_a": fp,
        "method_id": method,
        "nll_sum_a": math.fsum(
            -math.log(min(max(p if t is Truth.PASS else 1 - p, 1e-15), 1 - 1e-15))
            for p, t in zip(probs, truths, strict=True)
        ),
    }
    out["metrics"] = metrics(out, n)
    return out


def metrics(x: Mapping[str, Any], n: int) -> dict[str, float]:
    ece = sum(
        (b["count"] / n)
        * abs(b["probability_sum_a"] / b["count"] - b["pass_count"] / b["count"])
        for b in x["ece_bins_a"]
        if b["count"]
    )
    return {
        "accuracy_a": x["correct_count_a"] / n,
        "brier_a": x["brier_sum_a"] / n,
        "coverage_a": x["covered_count_a"] / n,
        "decision_loss_a": (
            x["false_pass_count_a"] + 0.2 * x["false_fail_count_a"] + x["defer_count_a"]
        )
        / n,
        "dispersion_change_rate": x["dispersion_change_count"] / n,
        "ece_a": ece,
        "false_safe_incidence_a": x["false_pass_count_a"] / n,
        "nll_a": x["nll_sum_a"] / n,
    }


def bootstrap(
    blocks: Mapping[str, np.ndarray], draws: int = 10000
) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(20260901))
    values = []
    for _ in range(draws):
        values.append(
            float(
                np.mean(
                    [
                        np.mean(blocks[s][rng.integers(0, 40, size=40)])
                        for s in sorted(blocks)
                    ]
                )
            )
        )
    quantiles = np.quantile(values, (0.025, 0.975), method="linear")
    return float(quantiles[0]), float(quantiles[1])


def verdict(
    points: Sequence[str] = (), cis: Sequence[str] = (), invalid: Sequence[str] = ()
) -> dict[str, object]:
    reasons = sorted(set(invalid or points or cis))
    if set(reasons) - REASONS:
        raise ValueError("reason")
    return {
        "reason_codes": reasons,
        "status": "INVALID"
        if invalid
        else "FAIL"
        if points
        else "INCONCLUSIVE"
        if cis
        else "PASS",
    }


def ordered(
    config: Mapping[str, Any],
    start: Callable[[], None],
    block: Callable[[str, int], object],
) -> list[object]:
    start()
    return [block(s["name"], r) for s in config["scenarios"] for r in range(40)]


def exclusive(path: Path, value: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "wb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def invalid_result(identity: object, reason: str | Sequence[str]) -> dict[str, Any]:
    reasons = [reason] if isinstance(reason, str) else sorted(set(reason))
    if (
        not reasons
        or reasons != sorted(set(reasons))
        or any(
            code not in REASONS
            or not code.startswith(("INVALID_", "RECORDER_"))
            for code in reasons
        )
    ):
        raise ValueError("invalid administrative reason")
    integrity: dict[str, Any] = {
        k: None
        for k in (
            "case_count_per_form",
            "deterministic_replay",
            "fit_case_count",
            "method_ab_reviewer_rows",
            "method_fit_reviewer_rows",
            "model_call_counts",
            "operands_sha256",
            "reviewer_row_count_per_form",
            "total_blocks",
            "test_case_count",
        )
    }
    integrity.update(reason_codes=reasons, status="INVALID")
    return {
        "blocks": None,
        "gate_id": "convergence-resolution-v1",
        "identity": identity,
        "integrity": integrity,
        "paired": None,
        "pooled": None,
        "scenarios": None,
        "schema_version": "1",
        "verdict": {"reason_codes": reasons, "status": "INVALID"},
    }


IDENTITY_KEYS = (
    "accepted_base",
    "bootstrap_draws",
    "bootstrap_seed",
    "config_sha256",
    "documentation_commit",
    "judge_commit",
    "runtime",
    "scenario_sha256",
    "seed_table_sha256",
)
START_KEYS = frozenset(
    {
        *IDENTITY_KEYS,
        "external_switch",
        "fit_cases_per_block",
        "gate_id",
        "record_type",
        "replicates_per_scenario",
        "scenario_count",
        "schema_version",
        "test_cases_per_block",
        "total_blocks",
        "total_test_cases",
    }
)
FINAL_KEYS = frozenset(
    {
        "gate_id",
        "reason_codes",
        "record_type",
        "result",
        "result_sha256",
        "schema_version",
        "start_sha256",
        "verdict",
        "wall_time_seconds",
    }
)
RESULT_KEYS = frozenset(
    {
        "blocks",
        "gate_id",
        "identity",
        "integrity",
        "paired",
        "pooled",
        "scenarios",
        "schema_version",
        "verdict",
    }
)
INTEGRITY_KEYS = frozenset(
    {
        "case_count_per_form",
        "deterministic_replay",
        "fit_case_count",
        "method_ab_reviewer_rows",
        "method_fit_reviewer_rows",
        "model_call_counts",
        "operands_sha256",
        "reason_codes",
        "reviewer_row_count_per_form",
        "status",
        "total_blocks",
        "test_case_count",
    }
)
METHOD_KEYS = frozenset(
    {
        "action_counts_a",
        "action_counts_b",
        "brier_sum_a",
        "correct_count_a",
        "covered_count_a",
        "defer_count_a",
        "dispersion_change_count",
        "ece_bins_a",
        "false_fail_count_a",
        "false_pass_count_a",
        "method_id",
        "metrics",
        "nll_sum_a",
    }
)
METRIC_KEYS = frozenset(
    {
        "accuracy_a",
        "brier_a",
        "coverage_a",
        "decision_loss_a",
        "dispersion_change_rate",
        "ece_a",
        "false_safe_incidence_a",
        "nll_a",
    }
)


def start_identity(start: Mapping[str, Any]) -> dict[str, Any]:
    return {key: start[key] for key in IDENTITY_KEYS}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_start(start: object) -> dict[str, Any]:
    if not isinstance(start, dict) or set(start) != START_KEYS:
        raise ValueError("START schema")
    config = load_config()
    design = config["sample_design"]
    expected = {
        "accepted_base": config["accepted_base"],
        "bootstrap_draws": config["bootstrap"]["draws"],
        "bootstrap_seed": config["bootstrap"]["seed"],
        "config_sha256": PINNED_CONFIG_SHA256,
        "documentation_commit": DOCUMENTATION_COMMIT,
        "external_switch": config["external_switch"],
        "fit_cases_per_block": design["fit_cases_per_block"],
        "gate_id": config["gate_id"],
        "record_type": "START",
        "replicates_per_scenario": design["replicates_per_scenario"],
        "runtime": config["runtime"],
        "scenario_count": design["scenario_count"],
        "scenario_sha256": config["scenario_sha256"],
        "schema_version": config["schema_version"],
        "seed_table_sha256": config["seed_table_sha256"],
        "test_cases_per_block": design["test_cases_per_block"],
        "total_blocks": design["total_blocks"],
        "total_test_cases": design["total_test_cases"],
    }
    for key, value in expected.items():
        if not exact_equal(start[key], value):
            raise ValueError(f"START literal: {key}")
    if (
        not isinstance(start["judge_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", start["judge_commit"]) is None
    ):
        raise ValueError("START judge commit")
    return cast(dict[str, Any], start)


def _validate_verdict(value: object, *, embedded: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"reason_codes", "status"}:
        raise ValueError("verdict schema")
    reasons = value["reason_codes"]
    status = value["status"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
        or reasons != sorted(set(reasons))
        or any(reason not in REASONS for reason in reasons)
        or status not in {"PASS", "FAIL", "INCONCLUSIVE", "INVALID"}
    ):
        raise ValueError("verdict value")
    required_prefix = {
        "FAIL": "FAIL_",
        "INCONCLUSIVE": "INCONCLUSIVE_",
        "INVALID": "INVALID_" if embedded else ("INVALID_", "RECORDER_"),
    }
    if status == "PASS":
        if reasons:
            raise ValueError("PASS reasons")
    elif not reasons or any(
        not reason.startswith(cast(str | tuple[str, ...], required_prefix[status]))
        for reason in reasons
    ):
        raise ValueError("verdict/reason class")
    return cast(dict[str, Any], value)


def _validate_truth_counts(value: object, n: int) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"FAIL", "PASS"}
        or any(type(count) is not int or count < 0 for count in value.values())
        or sum(value.values()) != n
    ):
        raise ValueError("truth counts")


def _validate_method_schema(value: object, n: int, method_id: str) -> None:
    if not isinstance(value, dict) or set(value) != METHOD_KEYS:
        raise ValueError("method schema")
    if value["method_id"] != method_id:
        raise ValueError("method order")
    action_counts = []
    for key in ("action_counts_a", "action_counts_b"):
        counts = value[key]
        if (
            not isinstance(counts, dict)
            or set(counts) != {"DEFER", "FAIL", "PASS"}
            or any(type(count) is not int or count < 0 for count in counts.values())
            or sum(counts.values()) != n
        ):
            raise ValueError("action counts")
        action_counts.append(counts)
    count_fields = (
        "correct_count_a",
        "covered_count_a",
        "defer_count_a",
        "dispersion_change_count",
        "false_fail_count_a",
        "false_pass_count_a",
    )
    if any(type(value[key]) is not int or not 0 <= value[key] <= n for key in count_fields):
        raise ValueError("method counts")
    if (
        value["defer_count_a"] != action_counts[0]["DEFER"]
        or value["covered_count_a"] != n - value["defer_count_a"]
        or value["correct_count_a"]
        + value["false_fail_count_a"]
        + value["false_pass_count_a"]
        + value["defer_count_a"]
        != n
    ):
        raise ValueError("method count algebra")
    bins = value["ece_bins_a"]
    if not isinstance(bins, list) or len(bins) != 10:
        raise ValueError("ECE bins")
    for index, row in enumerate(bins):
        if (
            not isinstance(row, dict)
            or set(row) != {"bin_index", "count", "pass_count", "probability_sum_a"}
            or row["bin_index"] != index
            or type(row["count"]) is not int
            or type(row["pass_count"]) is not int
            or not 0 <= row["pass_count"] <= row["count"]
            or not _is_nonnegative_number(row["probability_sum_a"])
        ):
            raise ValueError("ECE bin schema")
    if sum(row["count"] for row in bins) != n:
        raise ValueError("ECE count")
    if not _is_nonnegative_number(value["brier_sum_a"]) or not _is_nonnegative_number(
        value["nll_sum_a"]
    ):
        raise ValueError("probability sums")
    metrics_value = value["metrics"]
    if (
        not isinstance(metrics_value, dict)
        or set(metrics_value) != METRIC_KEYS
        or any(not _is_nonnegative_number(item) for item in metrics_value.values())
    ):
        raise ValueError("metric schema")


def _validate_summary_schema(value: object, n: int, scenario: str | None) -> None:
    expected_keys = (
        {"methods", "scenario", "test_case_count", "truth_counts_a"}
        if scenario is not None
        else {"methods", "test_case_count", "truth_counts_a"}
    )
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("summary schema")
    if value["test_case_count"] != n or (scenario is not None and value["scenario"] != scenario):
        raise ValueError("summary identity")
    _validate_truth_counts(value["truth_counts_a"], n)
    methods_value = value["methods"]
    if not isinstance(methods_value, list) or len(methods_value) != 3:
        raise ValueError("summary methods")
    for method_id, method_value in zip(METHODS, methods_value, strict=True):
        _validate_method_schema(method_value, n, method_id)


def _validate_embedded_result_schema(
    value: object, identity: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESULT_KEYS:
        raise ValueError("result schema")
    if value["gate_id"] != "convergence-resolution-v1" or value["schema_version"] != "1":
        raise ValueError("result gate")
    if not exact_equal(value["identity"], dict(identity)):
        raise ValueError("result identity")
    verdict_value = _validate_verdict(value["verdict"], embedded=True)
    integrity = value["integrity"]
    if not isinstance(integrity, dict) or set(integrity) != INTEGRITY_KEYS:
        raise ValueError("integrity schema")
    if verdict_value["status"] == "INVALID":
        if any(value[key] is not None for key in ("blocks", "paired", "pooled", "scenarios")):
            raise ValueError("INVALID science")
        if (
            integrity["status"] != "INVALID"
            or integrity["reason_codes"] != verdict_value["reason_codes"]
            or any(
                integrity[key] is not None
                for key in INTEGRITY_KEYS - {"status", "reason_codes"}
            )
        ):
            raise ValueError("INVALID integrity")
        return cast(dict[str, Any], value)
    expected_integrity = {
        "case_count_per_form": 2_400_000,
        "deterministic_replay": True,
        "fit_case_count": 1_920_000,
        "method_ab_reviewer_rows": {method: 14_400_000 for method in METHODS},
        "method_fit_reviewer_rows": {
            "candidate": 5_760_000,
            "ordinary_majority": 0,
            "reliability_weighted": 5_760_000,
        },
        "model_call_counts": {method: 0 for method in METHODS},
        "reason_codes": [],
        "reviewer_row_count_per_form": 7_200_000,
        "status": "PASS",
        "total_blocks": 240,
        "test_case_count": 2_400_000,
    }
    if any(not exact_equal(integrity[key], item) for key, item in expected_integrity.items()):
        raise ValueError("normal integrity")
    if not _is_sha256(integrity["operands_sha256"]):
        raise ValueError("integrity operand hash")
    config = load_config()
    blocks = value["blocks"]
    expected_block_ids = [
        (row["scenario"], row["replicate"]) for row in config["seeds"]
    ]
    if not isinstance(blocks, list) or len(blocks) != 240:
        raise ValueError("block cardinality")
    for block, (scenario_name, replicate) in zip(blocks, expected_block_ids, strict=True):
        if (
            not isinstance(block, dict)
            or set(block)
            != {
                "fit_case_count",
                "methods",
                "operands_sha256",
                "replicate",
                "scenario",
                "test_case_count",
                "truth_counts_a",
            }
            or block["fit_case_count"] != 8_000
            or block["test_case_count"] != 10_000
            or block["scenario"] != scenario_name
            or block["replicate"] != replicate
            or not _is_sha256(block["operands_sha256"])
        ):
            raise ValueError("block schema")
        _validate_truth_counts(block["truth_counts_a"], 10_000)
        methods_value = block["methods"]
        if not isinstance(methods_value, list) or len(methods_value) != 3:
            raise ValueError("block methods")
        for method_id, method_value in zip(METHODS, methods_value, strict=True):
            _validate_method_schema(method_value, 10_000, method_id)
    scenarios = value["scenarios"]
    scenario_names = sorted(row["name"] for row in config["scenarios"])
    if not isinstance(scenarios, list) or len(scenarios) != 6:
        raise ValueError("scenario cardinality")
    for scenario_value, scenario_name in zip(scenarios, scenario_names, strict=True):
        _validate_summary_schema(scenario_value, 400_000, scenario_name)
    _validate_summary_schema(value["pooled"], 2_400_000, None)
    paired = value["paired"]
    expected_paired = [
        (operand, baseline)
        for operand in ("accuracy_advantage", "dispersion_advantage", "false_safe_delta")
        for baseline in ("ordinary_majority", "reliability_weighted")
    ]
    if not isinstance(paired, list) or len(paired) != 6:
        raise ValueError("paired cardinality")
    for row, (operand, baseline) in zip(paired, expected_paired, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"baseline", "ci_lower", "ci_upper", "operand", "point"}
            or row["operand"] != operand
            or row["baseline"] != baseline
            or any(
                type(row[key]) not in (int, float) or not math.isfinite(float(row[key]))
                for key in ("ci_lower", "ci_upper", "point")
            )
            or row["ci_lower"] > row["ci_upper"]
        ):
            raise ValueError("paired schema")
    return cast(dict[str, Any], value)


def _parse_ledger(raw: bytes) -> ParsedLedger:
    malformed = lambda identity: invalid_result(identity, "RECORDER_MALFORMED_FINAL")
    if not raw:
        return ParsedLedger(None, malformed(None))
    first_lf = raw.find(b"\n")
    if first_lf < 0:
        return ParsedLedger(None, malformed(None))
    try:
        start = _validate_start(strict(raw[:first_lf]))
    except (KeyError, TypeError, UnicodeError, ValueError):
        return ParsedLedger(None, malformed(None))
    identity = start_identity(start)
    remainder = raw[first_lf + 1 :]
    if not remainder:
        return ParsedLedger(start, invalid_result(identity, "RECORDER_START_ONLY"))
    if not raw.endswith(b"\n"):
        if b"\n" in remainder:
            return ParsedLedger(start, malformed(identity))
        return ParsedLedger(start, invalid_result(identity, "RECORDER_PARTIAL_FINAL"))
    second = remainder[:-1]
    if not second or b"\n" in second:
        return ParsedLedger(start, malformed(identity))
    try:
        final = strict(second)
        if not isinstance(final, dict) or set(final) != FINAL_KEYS:
            raise ValueError("FINAL schema")
        if (
            final["record_type"] != "FINAL"
            or final["gate_id"] != "convergence-resolution-v1"
            or final["schema_version"] != "1"
            or final["start_sha256"] != digest(start)
            or not _is_sha256(final["result_sha256"])
            or not _is_nonnegative_number(final["wall_time_seconds"])
        ):
            raise ValueError("FINAL literal")
        result = _validate_embedded_result_schema(final["result"], identity)
        if (
            digest(result) != final["result_sha256"]
            or not exact_equal(final["verdict"], result["verdict"])
            or not exact_equal(final["reason_codes"], result["verdict"]["reason_codes"])
        ):
            raise ValueError("FINAL binding")
    except (KeyError, TypeError, UnicodeError, ValueError):
        return ParsedLedger(start, malformed(identity))
    return ParsedLedger(start, result)


def parse_ledger(raw: bytes) -> dict[str, Any]:
    return _parse_ledger(raw).result


def publish(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RecorderConflict("forensic artifact conflict")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def render_markdown(txt_bytes: bytes, result: Mapping[str, Any]) -> bytes:
    json_bytes = canonical(result)
    reasons = cast(list[str], result["verdict"]["reason_codes"])
    reason_text = "none" if not reasons else ", ".join(f"`{reason}`" for reason in reasons)
    identity = result["identity"]
    judge_commit = "unavailable" if identity is None else identity["judge_commit"]
    lines = [
        "# Corum Task 6E Convergence/Resolution Gate — Attempt 0",
        "",
        f"- Verdict: `{result['verdict']['status']}`",
        f"- Reason codes: {reason_text}",
        f"- Gate ID: `{result['gate_id']}`",
        f"- Judge commit: `{judge_commit}`",
        f"- TXT SHA-256: `{_sha256_bytes(txt_bytes)}`",
        f"- JSON SHA-256: `{_sha256_bytes(json_bytes)}`",
        "",
    ]
    for heading, key in (
        ("Integrity", "integrity"),
        ("Pooled", "pooled"),
        ("Paired", "paired"),
        ("Scenarios", "scenarios"),
    ):
        lines.extend((f"## {heading}", ""))
        if result[key] is None:
            lines.extend(("Unavailable for this `INVALID` result.", ""))
        else:
            lines.extend(("```json", canonical(result[key]).decode("utf-8"), "```", ""))
    return "\n".join(lines).encode("utf-8")


def _status_summary(result: Mapping[str, Any]) -> str:
    status = str(result["verdict"]["status"])
    reasons = cast(list[str], result["verdict"]["reason_codes"])
    reason_text = "none" if not reasons else ", ".join(f"`{reason}`" for reason in reasons)
    consequence = (
        "The only authorized next step is preparation and independent review of a new "
        "acquisition SDD/version; Task 6D execution, Task 7, product work, and model "
        "calls remain blocked."
        if status == "PASS"
        else "The current consensus path is stopped; another synthetic candidate, Task "
        "7, product work, and model calls remain unauthorized pending an owner decision."
    )
    return (
        f"Task 6E attempt 0 is final: `{status}` with reason codes {reason_text}. "
        "Artifacts: `docs/results/task-6e-convergence-resolution-attempt-0.txt`, "
        "`docs/results/task-6e-convergence-resolution-attempt-0.json`, and "
        "`docs/results/task-6e-convergence-resolution-attempt-0.md`. "
        f"{consequence}"
    )


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RecorderConflict("forensic status sentinel conflict")
    return text.replace(old, new)


def transform_status_document(
    relative: Path, pre_bytes: bytes, result: Mapping[str, Any]
) -> bytes:
    try:
        text = pre_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecorderConflict("forensic status UTF-8 conflict") from error
    summary = _status_summary(result)
    status = str(result["verdict"]["status"])
    reasons = cast(list[str], result["verdict"]["reason_codes"])
    reason_text = "none" if not reasons else ", ".join(reasons)
    if relative == Path("AGENTS.md"):
        text = _replace_once(
            text,
            "| 6E | Prospective full-coverage convergence/resolution gate | Prospective documentation registered; synthetic judge not yet implemented or run |",
            f"| 6E | Full-coverage convergence/resolution gate | Attempt 0 `{status}`; reasons: {reason_text} |",
        )
        text = _replace_once(
            text,
            "The documentation milestone may change only the four files registered in SDD 0010. A\n"
            "later reviewed TDD milestone may add only `configs/convergence-resolution-v1.json` and\n"
            "`tests/test_convergence_resolution_value.py`; no config or judge exists yet.",
            "The documentation milestone changed only the four files registered in SDD 0010. The\n"
            "reviewed judge milestone then added only `configs/convergence-resolution-v1.json` and\n"
            "`tests/test_convergence_resolution_value.py` before attempt 0.",
        )
        text = _replace_once(
            text,
            "\n## 7. Mandatory development workflow",
            f"\n### Task 6E attempt-0 recorded result\n\n{summary}\n\n## 7. Mandatory development workflow",
        )
    elif relative == Path("docs/plans/corum-mvp.md"):
        text = _replace_once(
            text,
            "- Create later: `configs/convergence-resolution-v1.json`",
            "- Created in the reviewed judge milestone: `configs/convergence-resolution-v1.json`",
        )
        text = _replace_once(
            text,
            "- Create later: `tests/test_convergence_resolution_value.py`",
            "- Created in the reviewed judge milestone: `tests/test_convergence_resolution_value.py`",
        )
        text = _replace_once(
            text,
            "\n---\n\n## Task 7: Leakage-free adaptive cascade",
            f"\n**Recorded attempt-0 outcome:** {summary}\n\n---\n\n## Task 7: Leakage-free adaptive cascade",
        )
    elif relative == Path("docs/sdd/0010-convergence-resolution-gate.md"):
        text = _replace_once(
            text,
            "- Status: prospective documentation registered; judge and formal attempt not started",
            f"- Status: attempt 0 final — `{status}`; reason codes: {reason_text}",
        )
        text = _replace_once(
            text,
            "\n## Synthetic verdict",
            f"\n## Attempt-0 recorded result\n\n{summary}\n\n## Synthetic verdict",
        )
    elif relative == Path("docs/specs/corum-mvp-design.md"):
        old = (
            "Task 6E has no formal result at this checkpoint. Its judge and config do not yet exist,\n"
            "and the documentation milestone must not create placeholders. After independent review,\n"
            "a separate TDD milestone may create only `configs/convergence-resolution-v1.json` and\n"
            "`tests/test_convergence_resolution_value.py`; the external path remains skipped unless\n"
            "`CORUM_RUN_CONVERGENCE_V1=1`."
        )
        text = _replace_once(text, old, summary)
    else:
        raise RecorderConflict("forensic unregistered status path")
    return text.encode("utf-8")


def _recorder_git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=not binary
        )
    except subprocess.CalledProcessError as error:
        raise RecorderConflict("forensic Git conflict") from error
    return completed.stdout if binary else cast(str, completed.stdout).rstrip("\r\n")


def _git_blob(repo: Path, commit: str, relative: Path) -> bytes:
    return cast(
        bytes,
        _recorder_git(repo, "show", f"{commit}:{relative.as_posix()}", binary=True),
    )


def validate_bound_judge_commit(repo: Path, judge_commit: str) -> None:
    try:
        parents = cast(
            str,
            _recorder_git(repo, "rev-list", "--parents", "-n", "1", judge_commit),
        ).split()
        if len(parents) != 2 or parents[0] != judge_commit or parents[1] != DOCUMENTATION_COMMIT:
            raise RecorderConflict("forensic bound judge parent conflict")
        if (
            cast(
                str,
                _recorder_git(repo, "show", "-s", "--format=%s", judge_commit),
            )
            != "test: lock convergence resolution gate"
        ):
            raise RecorderConflict("forensic bound judge subject conflict")
        changed = cast(
            str,
            _recorder_git(
                repo,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                judge_commit,
            ),
        ).splitlines()
        if sorted(changed) != [
            "configs/convergence-resolution-v1.json",
            "tests/test_convergence_resolution_value.py",
        ]:
            raise RecorderConflict("forensic bound judge path conflict")
        if (
            _sha256_bytes(
                _git_blob(
                    repo,
                    judge_commit,
                    Path("configs/convergence-resolution-v1.json"),
                )
            )
            != PINNED_CONFIG_SHA256
        ):
            raise RecorderConflict("forensic bound judge config conflict")
        for relative in STATUS_PATHS:
            if _sha256_bytes(_git_blob(repo, judge_commit, relative)) != STATUS_PRE_SHA256[
                relative
            ]:
                raise RecorderConflict("forensic bound judge status conflict")
    except RecorderConflict:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise RecorderConflict("forensic bound judge identity conflict") from error


def _status_expected(
    repo: Path, judge_commit: str, result: Mapping[str, Any]
) -> dict[Path, bytes]:
    expected = {}
    for relative in STATUS_PATHS:
        pre = _git_blob(repo, judge_commit, relative)
        if _sha256_bytes(pre) != STATUS_PRE_SHA256[relative]:
            raise RecorderConflict("forensic bound status blob conflict")
        expected[relative] = transform_status_document(relative, pre, result)
    return expected


def _allowed_status(repo: Path) -> None:
    output = cast(str, _recorder_git(repo, "status", "--porcelain=v1", "--untracked-files=all"))
    allowed = {path.as_posix() for path in (*RESULT_RELATIVE_PATHS, *STATUS_PATHS)}
    for line in output.splitlines():
        if len(line) < 4 or " -> " in line or line[3:] not in allowed:
            raise RecorderConflict("forensic unregistered worktree artifact")


def _matches_bound_pre(repo: Path, judge_commit: str, relative: Path) -> bool:
    actual = cast(
        str,
        _recorder_git(repo, "hash-object", f"--path={relative.as_posix()}", relative.as_posix()),
    )
    expected = cast(str, _recorder_git(repo, "rev-parse", f"{judge_commit}:{relative.as_posix()}"))
    return actual == expected


def _publish_status(
    repo: Path, judge_commit: str, relative: Path, expected: bytes
) -> None:
    path = repo / relative
    current = path.read_bytes()
    if current == expected:
        return
    if not _matches_bound_pre(repo, judge_commit, relative):
        raise RecorderConflict("forensic status artifact conflict")
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(expected)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def _inject_fault(fault_at: str | None, boundary: str) -> None:
    if fault_at == boundary:
        raise RecorderFault(boundary)


def _after_commit_ref_update(_repo: Path) -> None:
    """Test seam for ref-update-time worktree mutation; production is a no-op."""


def _verify_post_commit_worktree(
    repo: Path, expected_files: Mapping[Path, bytes]
) -> None:
    _allowed_status(repo)
    for relative, expected in expected_files.items():
        if not (repo / relative).exists() or (repo / relative).read_bytes() != expected:
            raise RecorderConflict("forensic post-commit worktree conflict")
    if cast(str, _recorder_git(repo, "status", "--porcelain=v1")):
        raise RecorderConflict("forensic dirty result commit")


def _validate_publication_state(
    repo: Path,
    judge_commit: str,
    expected_files: Mapping[Path, bytes],
) -> None:
    for relative in RESULT_RELATIVE_PATHS:
        path = repo / relative
        if not path.exists():
            if relative == RESULT_RELATIVE_PATHS[0]:
                raise RecorderConflict("forensic missing TXT ledger")
            continue
        if path.read_bytes() != expected_files[relative]:
            raise RecorderConflict("forensic result artifact conflict")
    for relative in STATUS_PATHS:
        path = repo / relative
        if not path.exists():
            raise RecorderConflict("forensic missing status artifact")
        if path.read_bytes() != expected_files[relative] and not _matches_bound_pre(
            repo, judge_commit, relative
        ):
            raise RecorderConflict("forensic status artifact conflict")


def _verify_result_commit(
    repo: Path,
    commit: str,
    judge_commit: str,
    expected_files: Mapping[Path, bytes],
) -> None:
    if (
        cast(str, _recorder_git(repo, "rev-parse", f"{commit}^")) != judge_commit
        or cast(str, _recorder_git(repo, "show", "-s", "--format=%s", commit))
        != RESULT_COMMIT_SUBJECT
    ):
        raise RecorderConflict("forensic result commit identity conflict")
    changed = cast(
        str,
        _recorder_git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ),
    ).splitlines()
    if sorted(changed) != sorted(path.as_posix() for path in expected_files):
        raise RecorderConflict("forensic result commit path conflict")
    for relative, expected in expected_files.items():
        if _git_blob(repo, commit, relative) != expected:
            raise RecorderConflict("forensic result commit tree conflict")


def record_consumed_attempt(repo: Path, fault_at: str | None = None) -> str:
    repo = repo.resolve()
    if fault_at is not None and fault_at not in RECORDER_FAULTS:
        raise ValueError("unknown recorder fault boundary")
    ledger_path = repo / RESULT_RELATIVE_PATHS[0]
    if not ledger_path.exists():
        raise FileNotFoundError("Task 6E TXT ledger is absent; no attempt was consumed")
    txt_bytes = ledger_path.read_bytes()
    parsed = _parse_ledger(txt_bytes)
    head = cast(str, _recorder_git(repo, "rev-parse", "HEAD"))
    if parsed.start is None:
        subject = cast(str, _recorder_git(repo, "show", "-s", "--format=%s", head))
        if subject == RESULT_COMMIT_SUBJECT:
            judge_commit = cast(str, _recorder_git(repo, "rev-parse", f"{head}^"))
        elif subject == "test: lock convergence resolution gate":
            judge_commit = head
        else:
            raise RecorderConflict("forensic unbound malformed-ledger HEAD")
    else:
        judge_commit = str(parsed.start["judge_commit"])
    validate_bound_judge_commit(repo, judge_commit)
    json_bytes = canonical(parsed.result)
    md_bytes = render_markdown(txt_bytes, parsed.result)
    status_bytes = _status_expected(repo, judge_commit, parsed.result)
    expected_files = {
        RESULT_RELATIVE_PATHS[0]: txt_bytes,
        RESULT_RELATIVE_PATHS[1]: json_bytes,
        RESULT_RELATIVE_PATHS[2]: md_bytes,
        **status_bytes,
    }
    _allowed_status(repo)
    _validate_publication_state(repo, judge_commit, expected_files)
    if head != judge_commit:
        _verify_result_commit(repo, head, judge_commit, expected_files)
        _verify_post_commit_worktree(repo, expected_files)
        return head
    publish(repo / RESULT_RELATIVE_PATHS[1], json_bytes)
    _inject_fault(fault_at, "after_json_fsync")
    publish(repo / RESULT_RELATIVE_PATHS[2], md_bytes)
    _inject_fault(fault_at, "after_md_fsync")
    status_faults = (
        "after_status_agents",
        "after_status_plan",
        "after_status_sdd",
        "after_status_design",
    )
    for relative, boundary in zip(STATUS_PATHS, status_faults, strict=True):
        _publish_status(repo, judge_commit, relative, status_bytes[relative])
        _inject_fault(fault_at, boundary)
    if ledger_path.read_bytes() != txt_bytes:
        raise RecorderConflict("forensic TXT mutation")
    for relative, expected in expected_files.items():
        if (repo / relative).read_bytes() != expected:
            raise RecorderConflict("forensic publication byte conflict")
    _allowed_status(repo)
    _recorder_git(repo, "add", "--", *(path.as_posix() for path in expected_files))
    expected_tree = cast(str, _recorder_git(repo, "write-tree"))
    _inject_fault(fault_at, "before_commit")
    _recorder_git(repo, "commit", "-m", RESULT_COMMIT_SUBJECT)
    result_commit = cast(str, _recorder_git(repo, "rev-parse", "HEAD"))
    _verify_result_commit(repo, result_commit, judge_commit, expected_files)
    if cast(str, _recorder_git(repo, "rev-parse", f"{result_commit}^{{tree}}")) != expected_tree:
        raise RecorderConflict("forensic result commit expected-tree conflict")
    _after_commit_ref_update(repo)
    _verify_post_commit_worktree(repo, expected_files)
    _inject_fault(fault_at, "after_commit_ref_update")
    return result_commit


def recorder_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish one consumed Task 6E attempt")
    parser.add_argument("--record-consumed-attempt", action="store_true")
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if not args.record_consumed_attempt:
        parser.error("--record-consumed-attempt is required")
    print(record_consumed_attempt(args.repo))
    return 0


def block_operand(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "methods": [
            {k: v for k, v in method.items() if k != "metrics"}
            for method in block["methods"]
        ],
        "replicate": block["replicate"],
        "scenario": block["scenario"],
        "truth_counts_a": block["truth_counts_a"],
    }


def block_operand_hash(block: Mapping[str, Any]) -> str:
    return digest(block_operand(block))


def registered_accounting() -> dict[str, dict[str, int]]:
    return {
        "method_ab_reviewer_rows": {m: 14_400_000 for m in METHODS},
        "method_fit_reviewer_rows": {"candidate": 5_760_000, "ordinary_majority": 0, "reliability_weighted": 5_760_000},
        "model_call_counts": {m: 0 for m in METHODS},
    }


def validate_accounting(accounting: Mapping[str, Any]) -> None:
    expected = registered_accounting()
    codes = []
    if not exact_equal(accounting.get("method_ab_reviewer_rows"), expected["method_ab_reviewer_rows"]):
        codes.append("INVALID_SHARED_AB_ROWS")
    if not exact_equal(accounting.get("method_fit_reviewer_rows"), expected["method_fit_reviewer_rows"]):
        codes.append("INVALID_FIT_ROWS")
    if not exact_equal(accounting.get("model_call_counts"), expected["model_call_counts"]):
        codes.append("INVALID_MODEL_CALLS")
    if codes:
        raise JudgeInvalid(*codes)


ACCOUNTING_REASON_BY_FAMILY = {
    "method_ab_reviewer_rows": "INVALID_SHARED_AB_ROWS",
    "method_fit_reviewer_rows": "INVALID_FIT_ROWS",
    "model_call_counts": "INVALID_MODEL_CALLS",
}


def validate_accounting_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    codes = []
    if not isinstance(evidence, Mapping):
        raise JudgeInvalid(*ACCOUNTING_REASON_BY_FAMILY.values())
    for family, reason in ACCOUNTING_REASON_BY_FAMILY.items():
        rows = evidence.get(family)
        if (
            not isinstance(rows, Mapping)
            or set(rows) != set(METHODS)
            or any(type(rows.get(method)) is not int or rows[method] < 0 for method in METHODS)
        ):
            codes.append(reason)
    if codes:
        raise JudgeInvalid(*codes)
    return {
        family: {method: cast(int, evidence[family][method]) for method in METHODS}
        for family in ACCOUNTING_REASON_BY_FAMILY
    }


def _aggregate_methods(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    n = sum(int(block["test_case_count"]) for block in blocks)
    result = []
    for method_id in METHODS:
        rows = [next(x for x in block["methods"] if x["method_id"] == method_id) for block in blocks]
        item: dict[str, Any] = {"method_id": method_id}
        for key in ("correct_count_a", "covered_count_a", "defer_count_a", "dispersion_change_count", "false_fail_count_a", "false_pass_count_a"):
            item[key] = sum(int(row[key]) for row in rows)
        for key in ("brier_sum_a", "nll_sum_a"):
            item[key] = math.fsum(float(row[key]) for row in rows)
        for key in ("action_counts_a", "action_counts_b"):
            item[key] = {action: sum(int(row[key][action]) for row in rows) for action in ("DEFER", "FAIL", "PASS")}
        item["ece_bins_a"] = [
            {
                "bin_index": index,
                "count": sum(int(row["ece_bins_a"][index]["count"]) for row in rows),
                "pass_count": sum(int(row["ece_bins_a"][index]["pass_count"]) for row in rows),
                "probability_sum_a": math.fsum(float(row["ece_bins_a"][index]["probability_sum_a"]) for row in rows),
            }
            for index in range(10)
        ]
        item["metrics"] = metrics(item, n)
        result.append(item)
    return result


def _paired_records(blocks: Sequence[Mapping[str, Any]], draws: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for block in blocks:
        grouped.setdefault(str(block["scenario"]), []).append(block)
    if len(blocks) == 240 and (len(grouped) != 6 or any(len(rows) != 40 for rows in grouped.values())):
        raise JudgeInvalid("INVALID_SIMULATION_ORDER")
    rng = np.random.Generator(np.random.PCG64(20260901))
    operands = (("accuracy_advantage", "accuracy_a", 1.0), ("dispersion_advantage", "dispersion_change_rate", -1.0), ("false_safe_delta", "false_safe_incidence_a", 1.0))
    baselines = ("ordinary_majority", "reliability_weighted")
    samples: dict[tuple[str, str], list[float]] = {(operand, baseline): [] for operand, _, _ in operands for baseline in baselines}
    points: dict[tuple[str, str], float] = {}
    def method(block: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        return next(x for x in block["methods"] if x["method_id"] == name)
    for operand, metric, direction in operands:
        for baseline in baselines:
            deltas = {name: np.array([direction * (method(b, "candidate")["metrics"][metric] - method(b, baseline)["metrics"][metric]) for b in rows]) for name, rows in grouped.items()}
            points[(operand, baseline)] = float(np.mean([np.mean(x) for x in deltas.values()]))
    for _ in range(draws):
        indices = {name: rng.integers(0, len(rows), size=len(rows)) for name, rows in sorted(grouped.items())}
        for key, values in samples.items():
            operand, baseline = key
            metric, direction = next((m, d) for o, m, d in operands if o == operand)
            values.append(float(np.mean([np.mean([direction * (method(rows[int(i)], "candidate")["metrics"][metric] - method(rows[int(i)], baseline)["metrics"][metric]) for i in indices[name]]) for name, rows in sorted(grouped.items())])))
    return [
        {"baseline": baseline, "ci_lower": float(np.quantile(samples[(operand, baseline)], .025, method="linear")), "ci_upper": float(np.quantile(samples[(operand, baseline)], .975, method="linear")), "operand": operand, "point": points[(operand, baseline)]}
        for operand, _, _ in operands for baseline in baselines
    ]


def validate_blocks(blocks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    expected = [(row["scenario"], row["replicate"]) for row in config["seeds"]]
    actual = [(block.get("scenario"), block.get("replicate")) if isinstance(block, Mapping) else (None, None) for block in blocks]
    if actual != expected or len(set(actual)) != 240:
        raise JudgeInvalid("INVALID_SIMULATION_ORDER")
    codes: set[str] = set()
    block_keys = {"fit_case_count", "methods", "operands_sha256", "replicate", "scenario", "test_case_count", "truth_counts_a"}
    method_keys = {"action_counts_a", "action_counts_b", "brier_sum_a", "correct_count_a", "covered_count_a", "defer_count_a", "dispersion_change_count", "ece_bins_a", "false_fail_count_a", "false_pass_count_a", "method_id", "metrics", "nll_sum_a"}
    for block in blocks:
        if not isinstance(block, Mapping) or set(block) != block_keys or block.get("fit_case_count") != 8000 or block.get("test_case_count") != 10000:
            codes.add("INVALID_COUNTS"); continue
        truth = block["truth_counts_a"]
        if not isinstance(truth, Mapping) or set(truth) != {"FAIL", "PASS"} or any(type(v) is not int or v < 0 for v in truth.values()) or sum(truth.values()) != 10000:
            codes.add("INVALID_COUNTS"); continue
        methods = block["methods"]
        if not isinstance(methods, list) or not all(isinstance(m, Mapping) for m in methods) or [m.get("method_id") for m in methods] != list(METHODS):
            codes.add("INVALID_COUNTS"); continue
        for method in methods:
            if set(method) != method_keys:
                codes.add("INVALID_COUNTS"); continue
            aa, ab = method["action_counts_a"], method["action_counts_b"]
            metric_values = method["metrics"]
            bins = method["ece_bins_a"]
            if not isinstance(aa, Mapping) or not isinstance(ab, Mapping) or not isinstance(metric_values, Mapping) or not isinstance(bins, list) or not all(isinstance(x, Mapping) for x in bins):
                codes.add("INVALID_COUNTS"); continue
            count_fields = ("correct_count_a", "covered_count_a", "defer_count_a", "dispersion_change_count", "false_fail_count_a", "false_pass_count_a")
            action_values = [*aa.values(), *ab.values()]
            if set(aa) != {"DEFER", "FAIL", "PASS"} or set(ab) != set(aa) or any(type(x) is not int or x < 0 for x in action_values) or sum(aa.values()) != 10000 or sum(ab.values()) != 10000 or any(type(method[k]) is not int or not 0 <= method[k] <= 10000 for k in count_fields):
                codes.add("INVALID_COUNTS"); continue
            if method["covered_count_a"] != 10000 - method["defer_count_a"] or method["defer_count_a"] != aa["DEFER"] or method["correct_count_a"] + method["false_fail_count_a"] + method["false_pass_count_a"] + method["defer_count_a"] != 10000 or method["correct_count_a"] != aa["PASS"] + aa["FAIL"] - method["false_pass_count_a"] - method["false_fail_count_a"] or method["false_pass_count_a"] > min(truth["FAIL"], aa["PASS"]) or method["false_fail_count_a"] > min(truth["PASS"], aa["FAIL"]):
                codes.add("INVALID_COUNTS")
            bin_schema = {"bin_index", "count", "pass_count", "probability_sum_a"}
            if len(bins) != 10 or any(set(x) != bin_schema for x in bins):
                codes.add("INVALID_COUNTS"); continue
            if [x.get("bin_index") for x in bins] != list(range(10)) or any(type(x["count"]) is not int or type(x["pass_count"]) is not int or not 0 <= x["pass_count"] <= x["count"] for x in bins) or sum(x["count"] for x in bins) != 10000 or sum(x["pass_count"] for x in bins) != truth["PASS"]:
                codes.add("INVALID_COUNTS")
            else:
                for index, bin_row in enumerate(bins):
                    probability_sum = bin_row["probability_sum_a"]
                    lower, upper = index / 10, (index + 1) / 10
                    if not isinstance(probability_sum, (int, float)) or isinstance(probability_sum, bool) or not math.isfinite(probability_sum):
                        codes.add("INVALID_NONFINITE")
                    elif probability_sum < lower * bin_row["count"] - 1e-10 or probability_sum > upper * bin_row["count"] + 1e-10:
                        codes.add("INVALID_COUNTS")
            numbers = [method["brier_sum_a"], method["nll_sum_a"], *metric_values.values(), *(x.get("probability_sum_a") for x in bins)]
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in numbers):
                codes.add("INVALID_NONFINITE")
            elif set(metric_values) != {"accuracy_a", "brier_a", "coverage_a", "decision_loss_a", "dispersion_change_rate", "ece_a", "false_safe_incidence_a", "nll_a"} or any(not math.isclose(float(metric_values[k]), v, rel_tol=1e-15, abs_tol=1e-15) for k, v in metrics(method, 10000).items()):
                codes.add("INVALID_REPLAY")
        try:
            if block.get("operands_sha256") != block_operand_hash(block):
                codes.add("INVALID_REPLAY")
        except (AttributeError, TypeError, ValueError):
            codes.add("INVALID_COUNTS")
    if codes:
        raise JudgeInvalid(*codes)


def _build_result_unchecked(blocks: Sequence[Mapping[str, Any]], identity: Mapping[str, Any], *, draws: int, accounting: Mapping[str, Any] | None = None) -> dict[str, Any]:
    ordered_blocks = sorted(blocks, key=lambda x: (x["scenario"], x["replicate"]))
    scenarios = []
    for name in sorted({str(x["scenario"]) for x in ordered_blocks}):
        rows = [x for x in ordered_blocks if x["scenario"] == name]
        scenarios.append({"methods": _aggregate_methods(rows), "scenario": name, "test_case_count": sum(x["test_case_count"] for x in rows), "truth_counts_a": {t: sum(x["truth_counts_a"][t] for x in rows) for t in ("FAIL", "PASS")}})
    pooled = {"methods": _aggregate_methods(ordered_blocks), "test_case_count": sum(x["test_case_count"] for x in ordered_blocks), "truth_counts_a": {t: sum(x["truth_counts_a"][t] for x in ordered_blocks) for t in ("FAIL", "PASS")}}
    paired = _paired_records(ordered_blocks, draws)
    observed = accounting or {"method_ab_reviewer_rows": {m: 6 * pooled["test_case_count"] for m in METHODS}, "method_fit_reviewer_rows": {"candidate": 3 * sum(x["fit_case_count"] for x in ordered_blocks), "ordinary_majority": 0, "reliability_weighted": 3 * sum(x["fit_case_count"] for x in ordered_blocks)}, "model_call_counts": {m: 0 for m in METHODS}}
    integrity = {"case_count_per_form": pooled["test_case_count"], "deterministic_replay": True, "fit_case_count": sum(x["fit_case_count"] for x in ordered_blocks), **observed, "operands_sha256": digest([block_operand(x) for x in ordered_blocks]), "reason_codes": [], "reviewer_row_count_per_form": 3 * pooled["test_case_count"], "status": "PASS", "total_blocks": len(ordered_blocks), "test_case_count": pooled["test_case_count"]}
    scientific = verdict_from_result(pooled, scenarios, paired)
    return {"blocks": ordered_blocks, "gate_id": "convergence-resolution-v1", "identity": dict(identity), "integrity": integrity, "paired": paired, "pooled": pooled, "scenarios": scenarios, "schema_version": "1", "verdict": scientific}


def build_nonformal_result(blocks: Sequence[Mapping[str, Any]], identity: Mapping[str, Any], *, draws: int = 10000) -> dict[str, Any]:
    return _build_result_unchecked(blocks, identity, draws=draws)


def validate_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if set(result) != {"blocks", "gate_id", "identity", "integrity", "paired", "pooled", "scenarios", "schema_version", "verdict"} or result["gate_id"] != config["gate_id"] or result["schema_version"] != config["schema_version"]:
        raise JudgeInvalid("INVALID_RESULT_CANONICAL")
    blocks = result["blocks"]
    if not isinstance(blocks, list):
        raise JudgeInvalid("INVALID_COUNTS")
    validate_blocks(blocks, config)
    design = config["sample_design"]
    expected_integrity = {"case_count_per_form": design["total_test_cases"], "deterministic_replay": True, "fit_case_count": design["total_fit_cases"], "method_ab_reviewer_rows": {m: 2 * 3 * design["total_test_cases"] for m in METHODS}, "method_fit_reviewer_rows": {"candidate": 3 * design["total_fit_cases"], "ordinary_majority": 0, "reliability_weighted": 3 * design["total_fit_cases"]}, "model_call_counts": {m: 0 for m in METHODS}, "operands_sha256": digest([block_operand(x) for x in blocks]), "reason_codes": [], "reviewer_row_count_per_form": 3 * design["total_test_cases"], "status": "PASS", "total_blocks": 240, "test_case_count": design["total_test_cases"]}
    if not exact_equal(result["integrity"], expected_integrity):
        raise JudgeInvalid("INVALID_COUNTS")
    if not isinstance(result["scenarios"], list) or len(result["scenarios"]) != 6 or not isinstance(result["paired"], list) or len(result["paired"]) != 6 or result["pooled"]["test_case_count"] != design["total_test_cases"]:
        raise JudgeInvalid("INVALID_COUNTS")
    if [x.get("operand") for x in result["paired"]] != [o for o in ("accuracy_advantage", "dispersion_advantage", "false_safe_delta") for _ in range(2)]:
        raise JudgeInvalid("INVALID_REPLAY")


def _reference_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = sum(int(row["test_case_count"]) for row in rows)
    output = []
    for method_id in METHODS:
        source = [next(m for m in row["methods"] if m["method_id"] == method_id) for row in rows]
        item: dict[str, Any] = {"method_id": method_id}
        for key in ("correct_count_a", "covered_count_a", "defer_count_a", "dispersion_change_count", "false_fail_count_a", "false_pass_count_a"):
            item[key] = sum(int(m[key]) for m in source)
        for key in ("brier_sum_a", "nll_sum_a"):
            item[key] = math.fsum(float(m[key]) for m in source)
        for key in ("action_counts_a", "action_counts_b"):
            item[key] = {a: sum(int(m[key][a]) for m in source) for a in ("DEFER", "FAIL", "PASS")}
        item["ece_bins_a"] = []
        for index in range(10):
            item["ece_bins_a"].append({"bin_index": index, "count": sum(int(m["ece_bins_a"][index]["count"]) for m in source), "pass_count": sum(int(m["ece_bins_a"][index]["pass_count"]) for m in source), "probability_sum_a": math.fsum(float(m["ece_bins_a"][index]["probability_sum_a"]) for m in source)})
        reference_ece = math.fsum(
            (b["count"] / n)
            * abs(b["probability_sum_a"] / b["count"] - b["pass_count"] / b["count"])
            for b in item["ece_bins_a"]
            if b["count"]
        )
        item["metrics"] = {
            "accuracy_a": item["correct_count_a"] / n,
            "brier_a": item["brier_sum_a"] / n,
            "coverage_a": item["covered_count_a"] / n,
            "decision_loss_a": (item["false_pass_count_a"] + .2 * item["false_fail_count_a"] + item["defer_count_a"]) / n,
            "dispersion_change_rate": item["dispersion_change_count"] / n,
            "ece_a": reference_ece,
            "false_safe_incidence_a": item["false_pass_count_a"] / n,
            "nll_a": item["nll_sum_a"] / n,
        }
        output.append(item)
    return {"methods": output, "test_case_count": n, "truth_counts_a": {t: sum(int(row["truth_counts_a"][t]) for row in rows) for t in ("FAIL", "PASS")}}


def _reference_paired(blocks: Sequence[Mapping[str, Any]], draws: int) -> list[dict[str, Any]]:
    groups = {name: [b for b in blocks if b["scenario"] == name] for name in sorted({str(b["scenario"]) for b in blocks})}
    definitions = (("accuracy_advantage", "accuracy_a", 1.0), ("dispersion_advantage", "dispersion_change_rate", -1.0), ("false_safe_delta", "false_safe_incidence_a", 1.0))
    baselines = ("ordinary_majority", "reliability_weighted")
    delta: dict[tuple[str, str, str], np.ndarray] = {}
    def raw_metric(block: Mapping[str, Any], method_id: str, metric: str) -> float:
        method = next(m for m in block["methods"] if m["method_id"] == method_id)
        n = int(block["test_case_count"])
        if metric == "accuracy_a":
            return float(method["correct_count_a"]) / n
        if metric == "dispersion_change_rate":
            return float(method["dispersion_change_count"]) / n
        if metric == "false_safe_incidence_a":
            return float(method["false_pass_count_a"]) / n
        raise JudgeInvalid("INVALID_REPLAY")
    for operand, metric, direction in definitions:
        for baseline in baselines:
            for name, rows in groups.items():
                values = []
                for block in rows:
                    values.append(direction * (raw_metric(block, "candidate", metric) - raw_metric(block, baseline, metric)))
                delta[(operand, baseline, name)] = np.asarray(values)
    rng = np.random.Generator(np.random.PCG64(20260901))
    samples: dict[tuple[str, str], list[float]] = {(o, b): [] for o, _, _ in definitions for b in baselines}
    for _ in range(draws):
        sampled = {name: rng.integers(0, 40, size=40) for name in groups}
        for operand, _, _ in definitions:
            for baseline in baselines:
                samples[(operand, baseline)].append(float(np.mean([np.mean(delta[(operand, baseline, name)][sampled[name]]) for name in groups])))
    output = []
    for operand, _, _ in definitions:
        for baseline in baselines:
            values = samples[(operand, baseline)]
            point = float(np.mean([np.mean(delta[(operand, baseline, name)]) for name in groups]))
            q = np.quantile(values, (.025, .975), method="linear")
            output.append({"baseline": baseline, "ci_lower": float(q[0]), "ci_upper": float(q[1]), "operand": operand, "point": point})
    return output


def _reference_verdict(pooled: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pm = {m["method_id"]: m["metrics"] for m in pooled["methods"]}
    labels = {"ordinary_majority": "ORDINARY", "reliability_weighted": "WEIGHTED"}
    fail: set[str] = set()
    inconclusive: set[str] = set()
    for baseline, label in labels.items():
        if pm["candidate"]["accuracy_a"] - pm[baseline]["accuracy_a"] < .05: fail.add(f"FAIL_ACCURACY_POINT_{label}")
        if pm["candidate"]["coverage_a"] < pm[baseline]["coverage_a"] - .01: fail.add(f"FAIL_COVERAGE_GAP_{label}")
        if pm["candidate"]["dispersion_change_rate"] > .7 * pm[baseline]["dispersion_change_rate"]: fail.add(f"FAIL_DISPERSION_POINT_{label}")
        if pm["candidate"]["false_safe_incidence_a"] > pm[baseline]["false_safe_incidence_a"]: fail.add(f"FAIL_FALSE_SAFE_POINT_{label}")
        if pm[baseline]["dispersion_change_rate"] == 0: inconclusive.add(f"INCONCLUSIVE_ZERO_DISPERSION_{label}")
        for operand, suffix, upper in (("accuracy_advantage", "ACCURACY", False), ("dispersion_advantage", "DISPERSION", False), ("false_safe_delta", "FALSE_SAFE", True)):
            record = next(x for x in paired if x["operand"] == operand and x["baseline"] == baseline)
            if (record["ci_upper"] > .005) if upper else (record["ci_lower"] <= 0): inconclusive.add(f"INCONCLUSIVE_{suffix}_CI_{label}")
    if pm["candidate"]["coverage_a"] < .98: fail.add("FAIL_COVERAGE_FLOOR")
    for scenario in scenarios:
        sm = {m["method_id"]: m["metrics"] for m in scenario["methods"]}
        if any(sm["candidate"]["accuracy_a"] < sm[b]["accuracy_a"] - .01 for b in labels): fail.add("FAIL_SCENARIO_ACCURACY")
        if sm["candidate"]["coverage_a"] < .97: fail.add("FAIL_SCENARIO_COVERAGE")
        if any(sm["candidate"]["false_safe_incidence_a"] > sm[b]["false_safe_incidence_a"] + .005 for b in labels): fail.add("FAIL_SCENARIO_FALSE_SAFE")
    reasons = sorted(fail or inconclusive)
    return {"reason_codes": reasons, "status": "FAIL" if fail else "INCONCLUSIVE" if inconclusive else "PASS"}


def validate_independent_replay(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    blocks = cast(list[Mapping[str, Any]], result["blocks"])
    scenarios = []
    for name in sorted({str(b["scenario"]) for b in blocks}):
        aggregate = _reference_aggregate([b for b in blocks if b["scenario"] == name])
        scenarios.append({**aggregate, "scenario": name})
    pooled = _reference_aggregate(blocks)
    paired = _reference_paired(blocks, config["bootstrap"]["draws"])
    reference = {"scenarios": scenarios, "pooled": pooled, "paired": paired, "verdict": _reference_verdict(pooled, scenarios, paired)}
    if any(canonical(result[key]) != canonical(reference[key]) for key in reference):
        raise JudgeInvalid("INVALID_REPLAY")


def build_scientific_result(blocks: Sequence[Mapping[str, Any]], identity: Mapping[str, Any], config: Mapping[str, Any], accounting: Mapping[str, Any]) -> dict[str, Any]:
    validate_blocks(blocks, config)
    validate_accounting(accounting)
    result = _build_result_unchecked(blocks, identity, draws=config["bootstrap"]["draws"], accounting=accounting)
    # Independent replay validates serialized sufficient statistics, hashes, counts, and
    # recomputed derived metrics before accepting the separately rendered result.
    replay_blocks = strict(canonical(result["blocks"]))
    if not isinstance(replay_blocks, list):
        raise JudgeInvalid("INVALID_REPLAY")
    validate_blocks(replay_blocks, config)
    replay = _build_result_unchecked(replay_blocks, identity, draws=config["bootstrap"]["draws"], accounting=accounting)
    if canonical(result) != canonical(replay):
        raise JudgeInvalid("INVALID_REPLAY")
    validate_result(result, config)
    validate_result(replay, config)
    validate_independent_replay(result, config)
    if strict(canonical(result)) != result:
        raise JudgeInvalid("INVALID_RESULT_CANONICAL")
    return result


def verdict_from_result(pooled: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    methods = {x["method_id"]: x["metrics"] for x in pooled["methods"]}
    labels = {"ordinary_majority": "ORDINARY", "reliability_weighted": "WEIGHTED"}
    points: list[str] = []
    cis: list[str] = []
    for baseline, label in labels.items():
        if methods["candidate"]["accuracy_a"] - methods[baseline]["accuracy_a"] < .05: points.append(f"FAIL_ACCURACY_POINT_{label}")
        if methods["candidate"]["coverage_a"] < methods[baseline]["coverage_a"] - .01: points.append(f"FAIL_COVERAGE_GAP_{label}")
        if methods["candidate"]["dispersion_change_rate"] > .70 * methods[baseline]["dispersion_change_rate"]: points.append(f"FAIL_DISPERSION_POINT_{label}")
        if methods["candidate"]["false_safe_incidence_a"] > methods[baseline]["false_safe_incidence_a"]: points.append(f"FAIL_FALSE_SAFE_POINT_{label}")
        for operand, suffix, upper in (("accuracy_advantage", "ACCURACY", False), ("dispersion_advantage", "DISPERSION", False), ("false_safe_delta", "FALSE_SAFE", True)):
            row = next(x for x in paired if x["operand"] == operand and x["baseline"] == baseline)
            if (row["ci_upper"] > .005) if upper else (row["ci_lower"] <= 0): cis.append(f"INCONCLUSIVE_{suffix}_CI_{label}")
        if methods[baseline]["dispersion_change_rate"] == 0: cis.append(f"INCONCLUSIVE_ZERO_DISPERSION_{label}")
    if methods["candidate"]["coverage_a"] < .98: points.append("FAIL_COVERAGE_FLOOR")
    for scenario in scenarios:
        sm = {x["method_id"]: x["metrics"] for x in scenario["methods"]}
        if any(sm["candidate"]["accuracy_a"] < sm[b]["accuracy_a"] - .01 for b in labels): points.append("FAIL_SCENARIO_ACCURACY")
        if sm["candidate"]["coverage_a"] < .97: points.append("FAIL_SCENARIO_COVERAGE")
        if any(sm["candidate"]["false_safe_incidence_a"] > sm[b]["false_safe_incidence_a"] + .005 for b in labels): points.append("FAIL_SCENARIO_FALSE_SAFE")
    return verdict(points=points, cis=cis)


def construct_formal_scenario(value: Mapping[str, Any]) -> Scenario:  # pragma: no cover - formal switch only
    """Construct literals only after the formal START boundary."""
    return Scenario(str(value["name"]), phase(value["calibration"]), phase(value["test"]))


def _matrix(panel: Any, reviewer_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[Truth, ...], np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover - formal switch only
    case_ids = tuple(panel.truths)
    row = {case_id: i for i, case_id in enumerate(case_ids)}
    col = {reviewer_id: i for i, reviewer_id in enumerate(reviewer_ids)}
    observations = np.full((len(case_ids), len(reviewer_ids)), -1, dtype=np.int64)
    valid = np.zeros_like(observations, dtype=bool)
    state_codes = {state: index for index, state in enumerate(ExecutionState)}
    states = np.full_like(observations, state_codes[ExecutionState.INVALID])
    codes = {Observation.PASS: 0, Observation.FAIL: 1, Observation.ABSTAIN: 2}
    for review_row in panel.reviews:
        states[row[review_row.case_id], col[review_row.reviewer_id]] = state_codes[review_row.state]
        if review_row.state is ExecutionState.VALID:
            observations[row[review_row.case_id], col[review_row.reviewer_id]] = codes[review_row.observation]
            valid[row[review_row.case_id], col[review_row.reviewer_id]] = True
    return case_ids, tuple(panel.truths[x] for x in case_ids), observations, valid, states


def execute_scientific_block(config: Mapping[str, Any], scenario_value: Mapping[str, Any], seed_row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:  # pragma: no cover - formal switch only
    scenario = construct_formal_scenario(scenario_value)
    fit_panel, test_panel = simulate_experiment(scenario, n_calibration=8000, n_test=10000, seed=int(seed_row["simulation"]))
    if len(fit_panel.truths) != 8000 or len(fit_panel.reviews) != 24000:
        raise JudgeInvalid("INVALID_FIT_ROWS")
    if len(test_panel.truths) != 10000 or len(test_panel.reviews) != 30000:
        raise JudgeInvalid("INVALID_COUNTS", "INVALID_SHARED_AB_ROWS")
    reviewers = tuple(x.reviewer for x in scenario.calibration.reviewers)
    examples = tuple(CalibrationExample(fit_panel.truths[r.case_id], r) for r in fit_panel.reviews)
    fusion = config["fusion"]
    calibrations = fit_panel_calibrations(reviewers, examples, prior_strength=fusion["prior_strength"])
    dependence = fit_dependence(reviewers, examples, shrinkage=fusion["dependence_shrinkage"], min_overlap=fusion["minimum_overlap"], lineage_cap=fusion["lineage_cap"])
    context = build_fusion_context(calibrations, dependence, prior_pass=fusion["prior_pass"], draws=fusion["posterior_draws"], credible_mass=fusion["credible_mass"], seed=int(seed_row["fusion"]), pair_calibrations={})
    reviewer_ids = tuple(r.reviewer_id for r in reviewers)
    _, truths, obs_a, valid_a, states_a = _matrix(test_panel, reviewer_ids)
    if obs_a.shape != (10000, 3) or valid_a.shape != obs_a.shape:
        raise JudgeInvalid("INVALID_SHARED_AB_ROWS")
    combined = tuple((*states_a[i].tolist(), *obs_a[i].tolist()) for i in range(len(truths)))
    rotated = rotate(combined, truths, int(seed_row["perturbation"]), config["perturbation"]["selection_fraction"])
    width = len(reviewer_ids)
    states_b = np.array([x[:width] for x in rotated], dtype=np.int64)
    valid_code = tuple(ExecutionState).index(ExecutionState.VALID)
    valid_b = states_b == valid_code
    obs_b = np.array([x[width:] for x in rotated], dtype=np.int64)
    if obs_b.shape != obs_a.shape or valid_b.shape != valid_a.shape:
        raise JudgeInvalid("INVALID_SHARED_AB_ROWS")
    if any(Counter(x for x, t in zip(combined, truths, strict=True) if t is truth) != Counter(x for x, t in zip(rotated, truths, strict=True) if t is truth) for truth in (Truth.PASS, Truth.FAIL)):
        raise JudgeInvalid("INVALID_PERTURBATION_MULTISET")
    batch_a = fuse_review_matrix(obs_a, valid_a, reviewer_ids, context, chunk_size=fusion["chunk_size"])
    batch_b = fuse_review_matrix(obs_b, valid_b, reviewer_ids, context, chunk_size=fusion["chunk_size"])
    probability_a = np.where(batch_a.valid_reviewers == 0, .5, batch_a.pass_probability)
    probability_b = np.where(batch_b.valid_reviewers == 0, .5, batch_b.pass_probability)
    directional_a = valid_a & (obs_a < 2)
    directional_b = valid_b & (obs_b < 2)
    def candidate_actions(probabilities: np.ndarray, directional: np.ndarray) -> tuple[Action, ...]:
        return tuple(
            candidate_action(float(probabilities[i]), int(directional[i].sum()))
            for i in range(len(probabilities))
        )
    weights = fit_weights(examples)
    def baseline_rows(observations: np.ndarray, validity: np.ndarray, weighted: bool) -> tuple[tuple[Action, ...], tuple[float, ...]]:
        actions, probabilities = [], []
        for i in range(len(truths)):
            reviews = tuple(Review(str(i), reviewer_id, ({0: Observation.PASS, 1: Observation.FAIL, 2: Observation.ABSTAIN}[int(observations[i, j])] if validity[i, j] else None), (ExecutionState.VALID if validity[i, j] else ExecutionState.INVALID)) for j, reviewer_id in enumerate(reviewer_ids))
            action, probability = vote(reviews, weights if weighted else None)
            actions.append(action); probabilities.append(probability)
        return tuple(actions), tuple(probabilities)
    ordinary_a, ordinary_p = baseline_rows(obs_a, valid_a, False); ordinary_b, _ = baseline_rows(obs_b, valid_b, False)
    weighted_a, weighted_p = baseline_rows(obs_a, valid_a, True); weighted_b, _ = baseline_rows(obs_b, valid_b, True)
    methods = [stats(truths, candidate_actions(probability_a, directional_a), candidate_actions(probability_b, directional_b), cast(Sequence[float], probability_a.tolist()), "candidate"), stats(truths, ordinary_a, ordinary_b, ordinary_p, "ordinary_majority"), stats(truths, weighted_a, weighted_b, weighted_p, "reliability_weighted")]
    block = {"fit_case_count": len(fit_panel.truths), "methods": methods, "replicate": seed_row["replicate"], "scenario": seed_row["scenario"], "test_case_count": len(test_panel.truths), "truth_counts_a": {"FAIL": sum(x is Truth.FAIL for x in truths), "PASS": sum(x is Truth.PASS for x in truths)}}
    block["operands_sha256"] = block_operand_hash(block)
    evidence = {"method_ab_reviewer_rows": {m: int(obs_a.size + obs_b.size) for m in METHODS}, "method_fit_reviewer_rows": {"candidate": len(fit_panel.reviews), "ordinary_majority": 0, "reliability_weighted": len(fit_panel.reviews)}, "model_call_counts": {m: 0 for m in METHODS}}
    return block, evidence


def append_final(path: Path, final: Mapping[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(canonical(final) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def run_scientific_gate(config: Mapping[str, Any], ledger: Path, identity: Mapping[str, Any], block_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    design = config["sample_design"]
    start = {**identity, "external_switch": config["external_switch"], "fit_cases_per_block": design["fit_cases_per_block"], "gate_id": config["gate_id"], "record_type": "START", "replicates_per_scenario": design["replicates_per_scenario"], "scenario_count": design["scenario_count"], "schema_version": config["schema_version"], "test_cases_per_block": design["test_cases_per_block"], "total_blocks": design["total_blocks"], "total_test_cases": design["total_test_cases"]}
    exclusive(ledger, start)
    started = time.monotonic()
    try:
        by_name = {x["name"]: x for x in config["scenarios"]}
        blocks = []
        accounting = {name: {m: 0 for m in METHODS} for name in ("method_ab_reviewer_rows", "method_fit_reviewer_rows", "model_call_counts")}
        for row in config["seeds"]:
            block, evidence = block_fn(by_name[row["scenario"]], row)
            if (block.get("scenario"), block.get("replicate")) != (row["scenario"], row["replicate"]):
                raise JudgeInvalid("INVALID_SIMULATION_ORDER")
            blocks.append(block)
            validated_evidence = validate_accounting_evidence(evidence)
            for name, observed_methods in accounting.items():
                for method_id in METHODS:
                    observed_methods[method_id] += validated_evidence[name][method_id]
        result = build_scientific_result(blocks, identity, config, accounting)
    except JudgeInvalid as error:
        result = invalid_result(identity, error.codes)
    except BaseException:  # noqa: BLE001 - interrupts after START must terminalize INVALID
        result = invalid_result(identity, "INVALID_EXCEPTION")
    final = {"gate_id": config["gate_id"], "reason_codes": result["verdict"]["reason_codes"], "record_type": "FINAL", "result": result, "result_sha256": digest(result), "schema_version": config["schema_version"], "start_sha256": digest(start), "verdict": result["verdict"], "wall_time_seconds": time.monotonic() - started}
    append_final(ledger, final)
    print(f"CORUM_CONVERGENCE_V1 verdict={result['verdict']['status']} result_sha256={digest(result)}")
    return result


FORMAL_PYTEST_NODE = (
    "tests/test_convergence_resolution_value.py::test_locked_formal_gate"
)


def validate_formal_pytest_argv(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    nodes = [arg for arg in args if not arg.startswith("-")]
    options = [arg for arg in args if arg.startswith("-")]
    if nodes != [FORMAL_PYTEST_NODE] or any(
        re.fullmatch(r"-[qs]+", option) is None
        or len(option[1:]) != len(set(option[1:]))
        for option in options
    ):
        raise RuntimeError("single-node formal invocation required")


def static_formal_preflight() -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover - formal switch only
    validate_formal_pytest_argv()
    config = load_config()
    if os.environ.get(SWITCH) != "1" or any(path.exists() for path in RESULTS): raise RuntimeError("formal preflight")
    if seed_table(config) != config["seeds"]: raise RuntimeError("INVALID_SEED_REGENERATION")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "corum": "0.1.0"}
    if runtime != config["runtime"]: raise RuntimeError("runtime mismatch")
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    git_state = {"clean": git("status", "--porcelain") == "", "documentation_parent": git("rev-parse", f"{DOCUMENTATION_COMMIT}^"), "head": head, "parent": git("rev-parse", "HEAD^"), "subject": git("show", "-s", "--format=%s", "HEAD"), "paths": git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()}
    validate_git_identity(git_state)
    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if config_sha != PINNED_CONFIG_SHA256: raise RuntimeError("config hash")
    identity = {"accepted_base": config["accepted_base"], "bootstrap_draws": config["bootstrap"]["draws"], "bootstrap_seed": config["bootstrap"]["seed"], "config_sha256": config_sha, "documentation_commit": DOCUMENTATION_COMMIT, "judge_commit": head, "runtime": runtime, "scenario_sha256": config["scenario_sha256"], "seed_table_sha256": config["seed_table_sha256"]}
    return config, identity


def validate_git_identity(state: Mapping[str, Any]) -> None:
    if not state.get("clean") or state.get("documentation_parent") != "b34e0896c3cb80c325288d7057247f8b25fa72ab" or state.get("parent") != DOCUMENTATION_COMMIT or state.get("subject") != "test: lock convergence resolution gate" or state.get("paths") != ["configs/convergence-resolution-v1.json", "tests/test_convergence_resolution_value.py"]:
        raise RuntimeError("unreviewed judge identity")


def test_canonical_literals_and_rejections() -> None:
    a = {"n": 1, "p": 0.5, "z": None, "é": "雪"}
    b = {"array": ["PASS", "FAIL", "DEFER"], "nested": {"a": 0.0, "b": 2}}
    assert (
        canonical(a).hex()
        == "7b226e223a312c2270223a302e352c227a223a6e756c6c2c22c3a9223a22e99baa227d"
        and len(canonical(a)) == 35
        and digest(a)
        == "e8582054d4ca562a1fbdd1bf21c6eaee7b2192859275be3730989c96c52067f7"
    )
    assert (
        len(canonical(b)) == 58
        and digest(b)
        == "89f3bf53dad58f7f71ccc8cabd9133f7f99d4aa7fe7835567c4a50125e8eaf3d"
    )
    for raw in (
        b'{"a":1,"a":2}',
        b'{"x":-0.0}',
        b"{}\n",
        b"\xef\xbb\xbf{}",
        b'{"x":NaN}',
    ):
        with pytest.raises(ValueError):
            strict(raw)


def test_config_digests_literals_and_720_seeds() -> None:
    c = load_config()
    assert (
        c["accepted_base"] == "b34e0896c3cb80c325288d7057247f8b25fa72ab"
        and len(c["seeds"]) == 240
        and len(
            {r[p] for r in c["seeds"] for p in ("simulation", "fusion", "perturbation")}
            )
            == 720
        and c["scenario_sha256"] == PINNED_SCENARIO_SHA256
        and c["seed_table_sha256"] == PINNED_SEED_SHA256
    )
    assert (len(canonical(c["scenarios"])), c["scenario_sha256"]) == (
        5975,
        "5126f4a1d4c0d7cd97dccd6a860ed7dea45e69c23ecd2da6692b188a6198619c",
    )
    assert (len(canonical(c["seeds"])), c["seed_table_sha256"]) == (
        35646,
        "3d7cfb42bb5f48a11410a5187d13213e6e30446a40f00832e00aa19700c0ea29",
    )


def test_nonformal_phase_split() -> None:
    parent = int.from_bytes(
        hashlib.sha256(
            b"corum:convergence-resolution:fixture:v1\0phase-split"
        ).digest()[:8],
        "big",
    )
    assert parent == 172339166934708224
    assert tuple(
        int(x.generate_state(1, dtype=np.uint64)[0])
        for x in np.random.SeedSequence(parent).spawn(2)
    ) == (15680819540018043498, 12188076203272908518)
    fit, test = simulate_experiment(fixture(), n_calibration=7, n_test=9, seed=parent)
    assert (len(canonical(panel_object(fit))), digest(panel_object(fit))) == (
        1504,
        "8cb3a4892b4a9cf381fb5463b3e4b57a9fbc354b0c2ef0e9b9de4e129453004d",
    )
    assert (len(canonical(panel_object(test))), digest(panel_object(test))) == (
        1611,
        "744e35aae4b7e7d92bd77805c2b734a49e330fe7b2be778a7d4386f62fba5a88",
    )


def test_existing_core_empty_pair_and_all_invalid_fallback() -> None:
    fit, _test = simulate_experiment(
        fixture(), n_calibration=30, n_test=3, seed=172339166934708224
    )
    reviewers = tuple(x.reviewer for x in fixture().calibration.reviewers)
    examples = [CalibrationExample(fit.truths[r.case_id], r) for r in fit.reviews]
    cal = fit_panel_calibrations(reviewers, examples)
    dep = fit_dependence(
        reviewers, examples, shrinkage=0.25, min_overlap=10, lineage_cap=1.0
    )
    ctx = build_fusion_context(
        cal, dep, prior_pass=0.5, draws=32, seed=2, pair_calibrations={}
    )
    invalid = [
        Review("x", r.reviewer_id, None, ExecutionState.INVALID) for r in reviewers
    ]
    assert (
        fuse_reviews(invalid, ctx) is None and candidate(0.5, invalid) is Action.DEFER
    )


def test_baselines_fit_only_order_invariant() -> None:
    rows = [
        CalibrationExample(
            Truth.PASS,
            Review(
                str(i),
                "r",
                Observation.PASS if i < 3 else Observation.FAIL,
                ExecutionState.VALID,
            ),
        )
        for i in range(4)
    ]
    assert fit_weights(rows)["r"] == pytest.approx(math.log(2))
    test = [
        Review("x", "r", Observation.FAIL, ExecutionState.VALID),
        Review("x", "z", Observation.PASS, ExecutionState.VALID),
    ]
    assert vote(test) == vote(tuple(reversed(test))) == (Action.DEFER, 0.5)


def test_rotation_direction_truth_and_multiset() -> None:
    rows = tuple((x,) for x in "abcdef")
    truths = (Truth.PASS,) * 3 + (Truth.FAIL,) * 3
    out = rotate(rows, truths, 8, 1.0)
    for t in (Truth.PASS, Truth.FAIL):
        assert Counter(
            x for x, y in zip(rows, truths, strict=True) if y is t
        ) == Counter(x for x, y in zip(out, truths, strict=True) if y is t)
    selected = np.random.Generator(np.random.PCG64(8)).permutation(np.array([0, 1, 2]))
    assert all(
        out[int(selected[(k + 1) % 3])] == rows[int(selected[k])] for k in range(3)
    )


def test_rotation_shape_and_small_selection_are_precise_invalid() -> None:
    with pytest.raises(JudgeInvalid, match="INVALID_PERTURBATION_MULTISET"):
        rotate(((1,),), (Truth.PASS, Truth.FAIL), 1, 1.0)
    with pytest.raises(JudgeInvalid, match="INVALID_PERTURBATION_MULTISET"):
        rotate(((1,), (2,)), (Truth.PASS, Truth.FAIL), 1, .15)


def test_rotation_matrix_preserves_distinct_nonvalid_states() -> None:
    panel = SimpleNamespace(
        truths={"x": Truth.PASS},
        reviews=(
            Review("x", "a", None, ExecutionState.TIMEOUT),
            Review("x", "b", None, ExecutionState.INVALID),
            Review("x", "c", Observation.ABSTAIN, ExecutionState.VALID),
        ),
    )
    _, _, observations, valid, states = _matrix(panel, ("a", "b", "c"))
    assert states[0, 0] != states[0, 1] and valid.tolist() == [[False, False, True]]
    assert observations.tolist() == [[-1, -1, 2]]


def test_metrics_fixture_operand_and_api() -> None:
    truths = (Truth.PASS, Truth.FAIL, Truth.PASS)
    specs = (
        (
            "candidate",
            (Action.PASS, Action.FAIL, Action.DEFER),
            (Action.FAIL, Action.FAIL, Action.DEFER),
            (0.8, 0.2, 0.5),
        ),
        ("ordinary_majority", (Action.PASS,) * 3, (Action.PASS,) * 3, (0.75,) * 3),
        (
            "reliability_weighted",
            (Action.FAIL, Action.FAIL, Action.PASS),
            (Action.FAIL, Action.PASS, Action.PASS),
            (0.4, 0.3, 0.7),
        ),
    )
    records = [stats(truths, a, b, p, m) for m, a, b, p in specs]
    operand = {
        "methods": [{k: v for k, v in x.items() if k != "metrics"} for x in records],
        "replicate": 0,
        "scenario": "fixture-operands-v1",
        "truth_counts_a": {"FAIL": 1, "PASS": 2},
    }
    assert (
        len(canonical(operand)) == 3039
        and digest(operand)
        == "faeb1f5df17b89bdfd2dd37dff27ec8d980b1d7857c858afcf012d63add0be45"
    )
    assert (
        records[0]["brier_sum_a"] == 0.32999999999999996
        and records[0]["nll_sum_a"] == 1.1394342831883648
        and records[0]["metrics"]["accuracy_a"] == 2 / 3
    )
    api = evaluate_decisions(
        {str(i): t for i, t in enumerate(truths)},
        {str(i): a for i, a in enumerate(specs[0][1])},
        probabilities={str(i): p for i, p in enumerate(specs[0][3])},
        costs=DecisionCosts(false_pass=1.0, false_fail=0.2, defer=1.0),
    )
    assert api["brier"] == pytest.approx(records[0]["metrics"]["brier_a"]) and api[
        "ece"
    ] == pytest.approx(records[0]["metrics"]["ece_a"])


def test_probability_edges_clipping_and_empty_bins() -> None:
    p = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    r = stats(
        (Truth.PASS,) * 11, (Action.PASS,) * 11, (Action.PASS,) * 11, p, "candidate"
    )
    assert [x["count"] for x in r["ece_bins_a"]] == [1] * 9 + [2] and math.isfinite(
        r["nll_sum_a"]
    )


def test_whole_block_bootstrap_replay() -> None:
    blocks = {str(i): np.arange(40, dtype=float) + i for i in range(6)}
    assert bootstrap(blocks, 2) == (21.421041666666667, 22.220625)
    assert bootstrap(blocks, 100) == bootstrap(blocks, 100)


def test_git_identity_contract_with_mock_state() -> None:
    good = {"clean": True, "documentation_parent": "b34e0896c3cb80c325288d7057247f8b25fa72ab", "head": "judge", "parent": DOCUMENTATION_COMMIT, "subject": "test: lock convergence resolution gate", "paths": ["configs/convergence-resolution-v1.json", "tests/test_convergence_resolution_value.py"]}
    validate_git_identity(good)
    for key, bad in (("clean", False), ("documentation_parent", "wrong"), ("parent", "wrong"), ("subject", "wrong"), ("paths", [])):
        with pytest.raises(RuntimeError, match="unreviewed"):
            validate_git_identity({**good, key: bad})


@pytest.mark.parametrize("reason", sorted(REASONS))
def test_every_closed_reason_is_classified(reason: str) -> None:
    if reason.startswith(("INVALID_", "RECORDER_")):
        assert verdict(invalid=[reason]) == {"reason_codes": [reason], "status": "INVALID"}
    elif reason.startswith("FAIL_"):
        assert verdict(points=[reason], cis=["INCONCLUSIVE_ACCURACY_CI_ORDINARY"])["status"] == "FAIL"
    else:
        assert verdict(cis=[reason])["status"] == "INCONCLUSIVE"


def test_scientific_result_replay_from_retained_block_statistics() -> None:
    block = {
        "fit_case_count": 1,
        "methods": [
            stats((Truth.PASS,), (Action.PASS,), (Action.PASS,), (0.8,), method)
            for method in METHODS
        ],
        "replicate": 0,
        "scenario": "fixture-operands-v1",
        "test_case_count": 1,
        "truth_counts_a": {"FAIL": 0, "PASS": 1},
    }
    block["operands_sha256"] = block_operand_hash(block)
    first = build_nonformal_result([block], {"accepted_base": "fixture"}, draws=10)
    second = build_nonformal_result([block], {"accepted_base": "fixture"}, draws=10)
    assert canonical(first) == canonical(second)


def test_formal_builder_rejects_incomplete_and_corrupt_blocks() -> None:
    block = {
        "fit_case_count": 1,
        "methods": [stats((Truth.PASS,), (Action.PASS,), (Action.PASS,), (.8,), m) for m in METHODS],
        "operands_sha256": "0" * 64,
        "replicate": 0,
        "scenario": "fixture-operands-v1",
        "test_case_count": 1,
        "truth_counts_a": {"FAIL": 0, "PASS": 1},
    }
    with pytest.raises(JudgeInvalid, match="INVALID_SIMULATION_ORDER"):
        validate_blocks([block], load_config())


@pytest.mark.parametrize(
    ("field", "reason"),
    [("method_ab_reviewer_rows", "INVALID_SHARED_AB_ROWS"), ("method_fit_reviewer_rows", "INVALID_FIT_ROWS"), ("model_call_counts", "INVALID_MODEL_CALLS")],
)
def test_run_accounting_precise_invalid_reasons(field: str, reason: str) -> None:
    accounting = registered_accounting()
    accounting[field]["candidate"] += 1
    with pytest.raises(JudgeInvalid, match=reason):
        validate_accounting(accounting)


def test_run_accounting_collects_sorted_multiple_reasons() -> None:
    accounting = registered_accounting()
    for field in accounting.values():
        field["candidate"] += 1
    with pytest.raises(JudgeInvalid) as caught:
        validate_accounting(accounting)
    assert caught.value.codes == ("INVALID_FIT_ROWS", "INVALID_MODEL_CALLS", "INVALID_SHARED_AB_ROWS")


def test_formal_builder_validates_complete_retained_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    config = {**config, "bootstrap": {**config["bootstrap"], "draws": 2}}
    blocks = []
    for row in config["seeds"]:
        methods = []
        for method_id in METHODS:
            method = stats((Truth.PASS,), (Action.PASS,), (Action.PASS,), (.8,), method_id)
            for key in ("correct_count_a", "covered_count_a"):
                method[key] = 10000
            for key in ("brier_sum_a", "nll_sum_a"):
                method[key] *= 10000
            method["action_counts_a"]["PASS"] = 10000
            method["action_counts_b"]["PASS"] = 10000
            method["ece_bins_a"][8]["count"] = 10000
            method["ece_bins_a"][8]["pass_count"] = 10000
            method["ece_bins_a"][8]["probability_sum_a"] = 8000.0
            method["metrics"] = metrics(method, 10000)
            methods.append(method)
        block: dict[str, Any] = {"fit_case_count": 8000, "methods": methods, "replicate": row["replicate"], "scenario": row["scenario"], "test_case_count": 10000, "truth_counts_a": {"FAIL": 0, "PASS": 10000}}
        block["operands_sha256"] = block_operand_hash(block)
        blocks.append(block)
    result = build_scientific_result(blocks, {"accepted_base": "fixture"}, config, registered_accounting())
    assert result["integrity"]["status"] == "PASS" and len(result["paired"]) == 6
    for corruptor in (
        lambda value: value["pooled"]["methods"][0].__setitem__("correct_count_a", 0),
        lambda value: value["paired"][0].__setitem__("point", .123),
        lambda value: value.__setitem__("verdict", {"reason_codes": [], "status": "PASS"}),
    ):
        corrupted = cast(dict[str, Any], strict(canonical(result)))
        corruptor(corrupted)
        with pytest.raises(JudgeInvalid, match="INVALID_REPLAY"):
            validate_independent_replay(corrupted, config)
    corruptions = (
        lambda value: value[0]["methods"][0]["action_counts_a"].update({"FAIL": -1, "PASS": 10001}),
        lambda value: value[0]["methods"][0].__setitem__("false_pass_count_a", 1),
        lambda value: value[0]["methods"][0]["ece_bins_a"][8].__setitem__("pass_count", 10001),
    )
    for corruptor in corruptions:
        damaged = cast(list[dict[str, Any]], strict(canonical(blocks)))
        corruptor(damaged)
        with pytest.raises(JudgeInvalid, match="INVALID_COUNTS"):
            validate_blocks(damaged, config)
    malformed = (
        lambda value: value[0]["methods"][0].__setitem__("ece_bins_a", {}),
        lambda value: value[0]["methods"].__setitem__(0, "not-a-method"),
        lambda value: value[0]["methods"][0].__setitem__("metrics", []),
    )
    for corruptor in malformed:
        damaged = cast(list[dict[str, Any]], strict(canonical(blocks)))
        corruptor(damaged)
        with pytest.raises(JudgeInvalid, match="INVALID_COUNTS"):
            validate_blocks(damaged, config)
    paired_damaged = cast(list[dict[str, Any]], strict(canonical(blocks)))
    paired_damaged[0]["methods"][0]["metrics"]["accuracy_a"] += .1
    paired_damaged[1]["methods"][0]["metrics"]["accuracy_a"] -= .1
    broken_paired = _build_result_unchecked(paired_damaged, {"accepted_base": "fixture"}, draws=2, accounting=registered_accounting())
    assert broken_paired["paired"][0]["point"] == pytest.approx(result["paired"][0]["point"], abs=1e-18)
    with pytest.raises(JudgeInvalid, match="INVALID_REPLAY"):
        validate_independent_replay(broken_paired, config)
    monkeypatch.setitem(globals(), "metrics", lambda _item, _n: {key: 0.0 for key in ("accuracy_a", "brier_a", "coverage_a", "decision_loss_a", "dispersion_change_rate", "ece_a", "false_safe_incidence_a", "nll_a")})
    broken_primary = _build_result_unchecked(blocks, {"accepted_base": "fixture"}, draws=2, accounting=registered_accounting())
    with pytest.raises(JudgeInvalid, match="INVALID_REPLAY"):
        validate_independent_replay(broken_primary, config)


@pytest.mark.parametrize(
    ("status", "kwargs"),
    [
        ("PASS", {}),
        ("FAIL", {"points": ["FAIL_COVERAGE_FLOOR"]}),
        ("INCONCLUSIVE", {"cis": ["INCONCLUSIVE_ACCURACY_CI_ORDINARY"]}),
        ("INVALID", {"invalid": ["INVALID_COUNTS"]}),
    ],
)
def test_verdict_classes(status: str, kwargs: dict[str, Any]) -> None:
    assert verdict(**kwargs)["status"] == status


def test_start_before_exact_ordered_240_blocks() -> None:
    events: list[object] = []
    result = ordered(
        load_config(),
        lambda: events.append("fsynced START"),
        lambda s, r: events.append((s, r)),
    )
    assert (
        events[0] == "fsynced START"
        and len(result) == 240
        and len(set(events[1:])) == 240
    )


def test_end_to_end_runner_fsyncs_start_then_calls_240_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = load_config()
    config = {**config, "bootstrap": {**config["bootstrap"], "draws": 2}}
    calls: list[tuple[str, int]] = []
    def fake(_scenario: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        calls.append((str(row["scenario"]), int(row["replicate"])))
        assert (tmp_path / "attempt.txt").read_bytes().endswith(b"\n")
        block: dict[str, Any] = {"fit_case_count": 1, "methods": [stats((Truth.PASS,), (Action.PASS,), (Action.PASS,), (.8,), method) for method in METHODS], "replicate": row["replicate"], "scenario": row["scenario"], "test_case_count": 1, "truth_counts_a": {"FAIL": 0, "PASS": 1}}
        block["operands_sha256"] = block_operand_hash(block)
        evidence = {"method_ab_reviewer_rows": {m: 60000 for m in METHODS}, "method_fit_reviewer_rows": {"candidate": 24000, "ordinary_majority": 0, "reliability_weighted": 24000}, "model_call_counts": {m: 0 for m in METHODS}}
        return block, evidence
    identity = {"accepted_base": config["accepted_base"], "bootstrap_draws": 2, "bootstrap_seed": 20260901, "config_sha256": "0" * 64, "documentation_commit": "d", "judge_commit": "j", "runtime": config["runtime"], "scenario_sha256": config["scenario_sha256"], "seed_table_sha256": config["seed_table_sha256"]}
    run_scientific_gate(config, tmp_path / "attempt.txt", identity, fake)
    assert calls == [(row["scenario"], row["replicate"]) for row in config["seeds"]]
    assert re.fullmatch(r"CORUM_CONVERGENCE_V1 verdict=(?:PASS|FAIL|INCONCLUSIVE|INVALID) result_sha256=[0-9a-f]{64}\n", capsys.readouterr().out)


def test_exclusive_ledger_and_publication_recovery(tmp_path: Path) -> None:
    path = tmp_path / "ledger"
    exclusive(path, {"record_type": "START"})
    with pytest.raises(FileExistsError):
        exclusive(path, {"record_type": "START"})
    out = tmp_path / "result"
    publish(out, b"same")
    publish(out, b"same")
    with pytest.raises(RuntimeError, match="forensic"):
        publish(out, b"different")


def test_recorder_terminalization() -> None:
    start = recorder_start()
    sb = canonical(start)
    assert (
        parse_ledger(b"")["verdict"]["reason_codes"] == ["RECORDER_MALFORMED_FINAL"]
        and parse_ledger(sb + b"\n")["verdict"]["reason_codes"]
        == ["RECORDER_START_ONLY"]
        and parse_ledger(sb + b"\n{")["verdict"]["reason_codes"]
        == ["RECORDER_PARTIAL_FINAL"]
    )
    result = invalid_result(start_identity(start), "INVALID_EXCEPTION")
    final = recorder_final(start, result)
    assert parse_ledger(sb + b"\n" + canonical(final) + b"\n") == result


def test_static_preflight_default_external_skip() -> None:
    if os.environ.get(SWITCH) == "1":
        pytest.skip("formal switch is guarded by the registered single-node command")
    assert not any(path.exists() for path in RESULTS)


@pytest.mark.skipif(os.environ.get(SWITCH) != "1", reason="formal switch absent")
def test_locked_formal_gate() -> None:
    config, identity = static_formal_preflight()
    run_scientific_gate(config, RESULTS[0], identity, lambda scenario, row: execute_scientific_block(config, scenario, row))


def recorder_start(judge_commit: str = "a" * 40) -> dict[str, Any]:
    config = load_config()
    design = config["sample_design"]
    return {
        "accepted_base": config["accepted_base"],
        "bootstrap_draws": config["bootstrap"]["draws"],
        "bootstrap_seed": config["bootstrap"]["seed"],
        "config_sha256": PINNED_CONFIG_SHA256,
        "documentation_commit": DOCUMENTATION_COMMIT,
        "external_switch": config["external_switch"],
        "fit_cases_per_block": design["fit_cases_per_block"],
        "gate_id": config["gate_id"],
        "judge_commit": judge_commit,
        "record_type": "START",
        "replicates_per_scenario": design["replicates_per_scenario"],
        "runtime": config["runtime"],
        "scenario_count": design["scenario_count"],
        "scenario_sha256": config["scenario_sha256"],
        "schema_version": config["schema_version"],
        "seed_table_sha256": config["seed_table_sha256"],
        "test_cases_per_block": design["test_cases_per_block"],
        "total_blocks": design["total_blocks"],
        "total_test_cases": design["total_test_cases"],
    }


def recorder_final(
    start: Mapping[str, Any], result: Mapping[str, Any], wall_time: float = 1.25
) -> dict[str, Any]:
    return {
        "gate_id": "convergence-resolution-v1",
        "reason_codes": result["verdict"]["reason_codes"],
        "record_type": "FINAL",
        "result": result,
        "result_sha256": digest(result),
        "schema_version": "1",
        "start_sha256": digest(start),
        "verdict": result["verdict"],
        "wall_time_seconds": wall_time,
    }


def recorder_ledger(
    start: Mapping[str, Any], result: Mapping[str, Any]
) -> bytes:
    return canonical(start) + b"\n" + canonical(recorder_final(start, result)) + b"\n"


def recorder_normal_method(n: int, method_id: str) -> dict[str, Any]:
    bins = [
        {
            "bin_index": index,
            "count": n if index == 9 else 0,
            "pass_count": n if index == 9 else 0,
            "probability_sum_a": float(n) if index == 9 else 0.0,
        }
        for index in range(10)
    ]
    return {
        "action_counts_a": {"DEFER": 0, "FAIL": 0, "PASS": n},
        "action_counts_b": {"DEFER": 0, "FAIL": 0, "PASS": n},
        "brier_sum_a": 0.0,
        "correct_count_a": n,
        "covered_count_a": n,
        "defer_count_a": 0,
        "dispersion_change_count": 0,
        "ece_bins_a": bins,
        "false_fail_count_a": 0,
        "false_pass_count_a": 0,
        "method_id": method_id,
        "metrics": {
            "accuracy_a": 1.0,
            "brier_a": 0.0,
            "coverage_a": 1.0,
            "decision_loss_a": 0.0,
            "dispersion_change_rate": 0.0,
            "ece_a": 0.0,
            "false_safe_incidence_a": 0.0,
            "nll_a": 0.0,
        },
        "nll_sum_a": 0.0,
    }


def recorder_summary(n: int, scenario: str | None = None) -> dict[str, Any]:
    value = {
        "methods": [recorder_normal_method(n, method) for method in METHODS],
        "test_case_count": n,
        "truth_counts_a": {"FAIL": 0, "PASS": n},
    }
    if scenario is not None:
        value["scenario"] = scenario
    return value


def recorder_normal_result(start: Mapping[str, Any], status: str) -> dict[str, Any]:
    reason_codes = {
        "PASS": [],
        "FAIL": ["FAIL_COVERAGE_FLOOR"],
        "INCONCLUSIVE": ["INCONCLUSIVE_ACCURACY_CI_ORDINARY"],
    }[status]
    config = load_config()
    blocks = []
    for row in config["seeds"]:
        blocks.append(
            {
                "fit_case_count": 8_000,
                "methods": [
                    recorder_normal_method(10_000, method) for method in METHODS
                ],
                "operands_sha256": "a" * 64,
                "replicate": row["replicate"],
                "scenario": row["scenario"],
                "test_case_count": 10_000,
                "truth_counts_a": {"FAIL": 0, "PASS": 10_000},
            }
        )
    integrity = {
        "case_count_per_form": 2_400_000,
        "deterministic_replay": True,
        "fit_case_count": 1_920_000,
        "method_ab_reviewer_rows": {method: 14_400_000 for method in METHODS},
        "method_fit_reviewer_rows": {
            "candidate": 5_760_000,
            "ordinary_majority": 0,
            "reliability_weighted": 5_760_000,
        },
        "model_call_counts": {method: 0 for method in METHODS},
        "operands_sha256": "b" * 64,
        "reason_codes": [],
        "reviewer_row_count_per_form": 7_200_000,
        "status": "PASS",
        "total_blocks": 240,
        "test_case_count": 2_400_000,
    }
    paired = [
        {
            "baseline": baseline,
            "ci_lower": 0.01,
            "ci_upper": 0.2,
            "operand": operand,
            "point": 0.1,
        }
        for operand in (
            "accuracy_advantage",
            "dispersion_advantage",
            "false_safe_delta",
        )
        for baseline in ("ordinary_majority", "reliability_weighted")
    ]
    return {
        "blocks": blocks,
        "gate_id": "convergence-resolution-v1",
        "identity": start_identity(start),
        "integrity": integrity,
        "paired": paired,
        "pooled": recorder_summary(2_400_000),
        "scenarios": [
            recorder_summary(400_000, name)
            for name in sorted(row["name"] for row in config["scenarios"])
        ],
        "schema_version": "1",
        "verdict": {"reason_codes": reason_codes, "status": status},
    }


def test_recorder_strict_ledger_state_machine() -> None:
    start = recorder_start()
    identity = start_identity(start)
    caught = invalid_result(identity, "INVALID_EXCEPTION")
    valid = recorder_ledger(start, caught)
    assert parse_ledger(valid) == caught
    assert parse_ledger(canonical(start) + b"\n")["verdict"] == {
        "reason_codes": ["RECORDER_START_ONLY"],
        "status": "INVALID",
    }
    assert parse_ledger(canonical(start) + b"\n{")["verdict"] == {
        "reason_codes": ["RECORDER_PARTIAL_FINAL"],
        "status": "INVALID",
    }
    malformed = parse_ledger(b"")
    assert malformed["identity"] is None and malformed["verdict"]["reason_codes"] == [
        "RECORDER_MALFORMED_FINAL"
    ]
    bad_start = {**start, "total_blocks": 239}
    assert parse_ledger(canonical(bad_start) + b"\n")["identity"] is None
    assert parse_ledger(valid + b"{}\n")["verdict"]["reason_codes"] == [
        "RECORDER_MALFORMED_FINAL"
    ]
    assert parse_ledger(valid + b"trailing")["verdict"]["reason_codes"] == [
        "RECORDER_MALFORMED_FINAL"
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda final, _result: final.update(start_sha256="0" * 64),
        lambda final, _result: final.update(result_sha256="0" * 64),
        lambda final, _result: final.update(wall_time_seconds=-1.0),
        lambda final, _result: final.update(reason_codes=[]),
        lambda final, _result: final.update(extra=True),
        lambda final, result: result.update(extra=True),
    ],
)
def test_recorder_rejects_malformed_final_without_reinterpreting(
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    start = recorder_start()
    result = invalid_result(start_identity(start), "INVALID_EXCEPTION")
    final = recorder_final(start, result)
    mutate(final, result)
    raw = canonical(start) + b"\n" + canonical(final) + b"\n"
    parsed = parse_ledger(raw)
    assert parsed["identity"] == start_identity(start)
    assert parsed["verdict"]["reason_codes"] == ["RECORDER_MALFORMED_FINAL"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_recorder_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    judge_subject: str = "test: lock convergence resolution gate",
    include_config: bool = True,
    include_test: bool = True,
    extra_judge_path: bool = False,
    wrong_parent: bool = False,
    config_bytes: bytes | None = None,
) -> tuple[Path, dict[str, Any], bytes]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Corum Test")
    _git(repo, "config", "user.email", "corum@example.invalid")
    _git(repo, "config", "core.autocrlf", "true")
    for relative in STATUS_PATHS:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "documentation fixture")
    documentation_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setitem(globals(), "DOCUMENTATION_COMMIT", documentation_commit)
    if wrong_parent:
        (repo / "intermediate.txt").write_text("wrong parent\n", encoding="utf-8")
        _git(repo, "add", "intermediate.txt")
        _git(repo, "commit", "-m", "intermediate fixture")
    if include_config:
        (repo / "configs").mkdir()
        (repo / "configs/convergence-resolution-v1.json").write_bytes(
            CONFIG.read_bytes() if config_bytes is None else config_bytes
        )
    if include_test:
        (repo / "tests").mkdir()
        (repo / "tests/test_convergence_resolution_value.py").write_text(
            "# reviewed judge fixture\n", encoding="utf-8", newline=""
        )
    if extra_judge_path:
        (repo / "extra-judge.txt").write_text("extra\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", judge_subject)
    judge_commit = _git(repo, "rev-parse", "HEAD")
    start = recorder_start(judge_commit)
    result = invalid_result(start_identity(start), "INVALID_EXCEPTION")
    raw = recorder_ledger(start, result)
    ledger = repo / RESULT_RELATIVE_PATHS[0]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(raw)
    return repo, result, raw


@pytest.mark.parametrize(
    "fault_at",
    [
        "after_json_fsync",
        "after_md_fsync",
        "after_status_agents",
        "after_status_plan",
        "after_status_sdd",
        "after_status_design",
        "before_commit",
        "after_commit_ref_update",
    ],
)
def test_recorder_publication_faults_resume_to_one_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_at: str
) -> None:
    repo, result, raw = _init_recorder_repo(tmp_path, monkeypatch)
    judge = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(RecorderFault, match=fault_at):
        record_consumed_attempt(repo, fault_at=fault_at)
    commit = record_consumed_attempt(repo)
    assert commit == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "rev-parse", "HEAD^") == judge
    assert _git(repo, "show", "-s", "--format=%s", "HEAD") == RESULT_COMMIT_SUBJECT
    assert (repo / RESULT_RELATIVE_PATHS[0]).read_bytes() == raw
    assert (repo / RESULT_RELATIVE_PATHS[1]).read_bytes() == canonical(result)
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "rev-list", "--count", f"{judge}..HEAD") == "1"
    assert sorted(
        _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    ) == sorted(path.as_posix() for path in (*RESULT_RELATIVE_PATHS, *STATUS_PATHS))
    for relative in STATUS_PATHS:
        status_text = (repo / relative).read_text(encoding="utf-8")
        assert "Task 6E attempt 0 is final: `INVALID`" in status_text
        assert "task-6e-convergence-resolution-attempt-0.json" in status_text
        assert "INVALID_EXCEPTION" in status_text
        assert "synthetic judge not yet implemented" not in status_text
        assert "no config or judge exists yet" not in status_text
        assert "Create later: `configs/convergence-resolution-v1.json`" not in status_text
        assert "Create later: `tests/test_convergence_resolution_value.py`" not in status_text


def test_recorder_mismatch_is_forensic_and_txt_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _result, raw = _init_recorder_repo(tmp_path, monkeypatch)
    with pytest.raises(RecorderFault, match="after_json_fsync"):
        record_consumed_attempt(repo, fault_at="after_json_fsync")
    json_path = repo / RESULT_RELATIVE_PATHS[1]
    json_path.write_bytes(b"forensic mismatch")
    with pytest.raises(RecorderConflict, match="forensic"):
        record_consumed_attempt(repo)
    assert json_path.read_bytes() == b"forensic mismatch"
    assert (repo / RESULT_RELATIVE_PATHS[0]).read_bytes() == raw


RECORDER_NORMAL_FIXTURE_PINS = {
    "PASS": (
        912255,
        "31e750d3f3a41aa764dc06007f06b1b9c3e357d837717ef47a55fb721c47a086",
        26816,
        "74c285ac7178640deeb793c06e0d673a32c9e4bfd586474b873a297287ee2087",
    ),
    "FAIL": (
        912276,
        "250ece6ae0b974433309380be771f81869c22924189f686a29d817fffd35fb4c",
        26833,
        "843edf4d686e5ea253f86427c602a8034be0d88df76df02ea010d9ebcf8d70b4",
    ),
    "INCONCLUSIVE": (
        912298,
        "65b951d3d937f5fbb767c4565729e434c8ee97f0cdf1b9e95adadbb25ff5ff16",
        26855,
        "6a8984d3867030805079c05db28c6fb10dc7f5a5802455f993649bf6e23a164d",
    ),
    "INVALID": (
        1086,
        "d697ed7c2a02ff173d45e4ef0f948af79786fc8d1324e8a465087ffb8628364a",
        886,
        "fd0699ade5c041cdac306d3a3ed280409977cb3e9b5fcd68db001ccbd95c23ff",
    ),
}


@pytest.mark.parametrize("status", ["PASS", "FAIL", "INCONCLUSIVE", "INVALID"])
def test_recorder_pins_normal_and_caught_result_and_markdown_bytes(status: str) -> None:
    start = recorder_start()
    result = (
        invalid_result(start_identity(start), "INVALID_EXCEPTION")
        if status == "INVALID"
        else recorder_normal_result(start, status)
    )
    raw = recorder_ledger(start, result)
    parsed = parse_ledger(raw)
    json_bytes = canonical(parsed)
    md_bytes = render_markdown(raw, parsed)
    expected_json_length, expected_json_sha, expected_md_length, expected_md_sha = (
        RECORDER_NORMAL_FIXTURE_PINS[status]
    )
    assert parsed == result
    assert (len(json_bytes), _sha256_bytes(json_bytes)) == (
        expected_json_length,
        expected_json_sha,
    )
    assert (len(md_bytes), _sha256_bytes(md_bytes)) == (
        expected_md_length,
        expected_md_sha,
    )
    assert b"TXT SHA-256" in md_bytes and b"JSON SHA-256" in md_bytes
    assert all(
        heading in md_bytes
        for heading in (b"## Integrity", b"## Pooled", b"## Paired", b"## Scenarios")
    )


RECORDER_CRASH_FIXTURE_PINS = {
    "zero": (
        553,
        "8136fbaac37872d7093d107d98ff4b414aca535af3d74c2cdf81c0b428ac822b",
        871,
        "862d8ff93ec49a0d48e054d9bfedcea36afe3c38f3f8b886089b2de6c3c22653",
    ),
    "malformed_first": (
        553,
        "8136fbaac37872d7093d107d98ff4b414aca535af3d74c2cdf81c0b428ac822b",
        871,
        "4b85248b18b2f8c581cfa5973e921478a1f7f7b8feb5d93672256ca1c5029373",
    ),
    "start_only": (
        1090,
        "2646b3b684a570fbcd86eec9b862f2c0d446395ce34e113defd8df904480eb7b",
        890,
        "b05e1d9b3d936f86b977308501f901870542ca44c679e5376cff142e12f7ef49",
    ),
    "partial_final": (
        1096,
        "c7396e0e7958781ff23f179d531d4cc8048ebb285a9d549db6dd6df4e955beef",
        896,
        "07073fa1e72b9260748e98eb623a52adf74dae3311d42f2b19c8a42f3fd4daea",
    ),
    "malformed_final": (
        1100,
        "40e176bedb5d61b969cbc766c115af97786d9bc3c3792b2bbf413f77db4e81cb",
        900,
        "7871625d12e51d77fd1d928e41b9bf1f775f47940d0fa8382d0711a396988a31",
    ),
    "after_final_before_stdout": (
        1086,
        "d697ed7c2a02ff173d45e4ef0f948af79786fc8d1324e8a465087ffb8628364a",
        886,
        "fd0699ade5c041cdac306d3a3ed280409977cb3e9b5fcd68db001ccbd95c23ff",
    ),
}


def test_recorder_pins_every_crash_fixture_bytes() -> None:
    start = recorder_start()
    start_bytes = canonical(start)
    caught = invalid_result(start_identity(start), "INVALID_EXCEPTION")
    fixtures = {
        "zero": b"",
        "malformed_first": b"{}\n",
        "start_only": start_bytes + b"\n",
        "partial_final": start_bytes + b"\n{",
        "malformed_final": start_bytes + b"\n{}\n",
        "after_final_before_stdout": recorder_ledger(start, caught),
    }
    for name, raw in fixtures.items():
        result = parse_ledger(raw)
        json_bytes = canonical(result)
        md_bytes = render_markdown(raw, result)
        expected_json_length, expected_json_sha, expected_md_length, expected_md_sha = (
            RECORDER_CRASH_FIXTURE_PINS[name]
        )
        assert (len(json_bytes), _sha256_bytes(json_bytes)) == (
            expected_json_length,
            expected_json_sha,
        )
        assert (len(md_bytes), _sha256_bytes(md_bytes)) == (
            expected_md_length,
            expected_md_sha,
        )


def test_zero_ledger_result_commit_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _result, _raw = _init_recorder_repo(tmp_path, monkeypatch)
    (repo / RESULT_RELATIVE_PATHS[0]).write_bytes(b"")
    commit = record_consumed_attempt(repo)
    assert record_consumed_attempt(repo) == commit
    result = strict((repo / RESULT_RELATIVE_PATHS[1]).read_bytes())
    assert cast(dict[str, Any], result)["identity"] is None


def test_status_conflict_is_detected_before_any_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _result, raw = _init_recorder_repo(tmp_path, monkeypatch)
    agents = repo / STATUS_PATHS[0]
    agents.write_bytes(agents.read_bytes() + b"forensic mutation")
    with pytest.raises(RecorderConflict, match="forensic"):
        record_consumed_attempt(repo)
    assert not (repo / RESULT_RELATIVE_PATHS[1]).exists()
    assert not (repo / RESULT_RELATIVE_PATHS[2]).exists()
    assert (repo / RESULT_RELATIVE_PATHS[0]).read_bytes() == raw


def test_recorder_cli_is_explicit_and_missing_ledger_creates_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        recorder_cli([])
    repo = tmp_path / "empty"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="no attempt"):
        record_consumed_attempt(repo)
    assert list(repo.iterdir()) == []


def test_recorder_validates_bound_judge_commit_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _result, _raw = _init_recorder_repo(tmp_path, monkeypatch)
    validate_bound_judge_commit(repo, _git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize(
    "judge_variant",
    [
        {"wrong_parent": True},
        {"judge_subject": "wrong judge subject"},
        {"include_config": False},
        {"include_test": False},
        {"extra_judge_path": True},
        {"config_bytes": b"{}"},
    ],
)
def test_recorder_rejects_unbound_judge_histories_before_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    judge_variant: dict[str, Any],
) -> None:
    repo, _result, raw = _init_recorder_repo(
        tmp_path, monkeypatch, **judge_variant
    )
    with pytest.raises(RecorderConflict, match="forensic"):
        record_consumed_attempt(repo)
    assert (repo / RESULT_RELATIVE_PATHS[0]).read_bytes() == raw
    assert not (repo / RESULT_RELATIVE_PATHS[1]).exists()
    assert not (repo / RESULT_RELATIVE_PATHS[2]).exists()


@pytest.mark.parametrize(
    "polluted_path",
    [
        Path("configs/convergence-resolution-v1.json"),
        Path("tests/test_convergence_resolution_value.py"),
    ],
)
def test_recorder_rejects_working_judge_file_pollution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    polluted_path: Path,
) -> None:
    repo, _result, _raw = _init_recorder_repo(tmp_path, monkeypatch)
    path = repo / polluted_path
    path.write_bytes(path.read_bytes() + b"polluted")
    with pytest.raises(RecorderConflict, match="forensic"):
        record_consumed_attempt(repo)
    assert not (repo / RESULT_RELATIVE_PATHS[1]).exists()


@pytest.mark.parametrize("mutation", ["txt", "status", "unregistered"])
def test_post_commit_mutation_is_forensic_before_normal_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repo, _result, raw = _init_recorder_repo(tmp_path, monkeypatch)

    def mutate_after_ref(target_repo: Path) -> None:
        path = (
            target_repo / RESULT_RELATIVE_PATHS[0]
            if mutation == "txt"
            else target_repo / STATUS_PATHS[0]
            if mutation == "status"
            else target_repo / "unregistered-after-commit.txt"
        )
        path.write_bytes(path.read_bytes() + b"mutation" if path.exists() else b"mutation")

    monkeypatch.setitem(globals(), "_after_commit_ref_update", mutate_after_ref)
    with pytest.raises(RecorderConflict, match="forensic"):
        record_consumed_attempt(repo)
    assert _git(repo, "show", "-s", "--format=%s", "HEAD") == RESULT_COMMIT_SUBJECT
    if mutation != "txt":
        assert (repo / RESULT_RELATIVE_PATHS[0]).read_bytes() == raw


def per_block_accounting_fixture() -> dict[str, Any]:
    return {
        "method_ab_reviewer_rows": {method: 60_000 for method in METHODS},
        "method_fit_reviewer_rows": {
            "candidate": 24_000,
            "ordinary_majority": 0,
            "reliability_weighted": 24_000,
        },
        "model_call_counts": {method: 0 for method in METHODS},
    }


@pytest.mark.parametrize(
    ("family", "reason"),
    [
        ("method_ab_reviewer_rows", "INVALID_SHARED_AB_ROWS"),
        ("method_fit_reviewer_rows", "INVALID_FIT_ROWS"),
        ("model_call_counts", "INVALID_MODEL_CALLS"),
    ],
)
def test_accounting_evidence_rejects_every_noninteger_shape(
    family: str, reason: str
) -> None:
    malformed_values: tuple[object, ...] = (
        None,
        {},
        {**{method: 0 for method in METHODS}, "extra": 0},
        {"candidate": True, "ordinary_majority": 0, "reliability_weighted": 0},
        {"candidate": -1, "ordinary_majority": 0, "reliability_weighted": 0},
        {"candidate": 1.5, "ordinary_majority": 0, "reliability_weighted": 0},
        {"candidate": "1", "ordinary_majority": 0, "reliability_weighted": 0},
    )
    for malformed in malformed_values:
        evidence: dict[str, object] = per_block_accounting_fixture()
        evidence[family] = malformed
        with pytest.raises(JudgeInvalid) as caught:
            validate_accounting_evidence(evidence)
        assert caught.value.codes == (reason,)


@pytest.mark.parametrize(
    ("family", "reason"),
    [
        ("method_ab_reviewer_rows", "INVALID_SHARED_AB_ROWS"),
        ("method_fit_reviewer_rows", "INVALID_FIT_ROWS"),
        ("model_call_counts", "INVALID_MODEL_CALLS"),
    ],
)
def test_runner_final_uses_precise_code_for_malformed_accounting_family(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    family: str,
    reason: str,
) -> None:
    config = load_config()
    identity = start_identity(recorder_start())

    def malformed(
        _scenario: Mapping[str, Any], row: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        evidence: dict[str, Any] = per_block_accounting_fixture()
        evidence.pop(family)
        return {"scenario": row["scenario"], "replicate": row["replicate"]}, evidence

    ledger = tmp_path / "attempt.txt"
    result = run_scientific_gate(config, ledger, identity, malformed)
    assert result["verdict"] == {"reason_codes": [reason], "status": "INVALID"}
    records = ledger.read_bytes().splitlines()
    assert len(records) == 2
    final = strict(records[1])
    assert cast(dict[str, Any], final)["result"] == result
    assert parse_ledger(ledger.read_bytes()) == result
    assert f"verdict=INVALID result_sha256={digest(result)}" in capsys.readouterr().out


def test_runner_final_sorts_multiple_malformed_accounting_codes(
    tmp_path: Path,
) -> None:
    config = load_config()
    identity = start_identity(recorder_start())

    def malformed(
        _scenario: Mapping[str, Any], row: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return {
            "scenario": row["scenario"],
            "replicate": row["replicate"],
        }, {family: None for family in per_block_accounting_fixture()}

    result = run_scientific_gate(config, tmp_path / "attempt.txt", identity, malformed)
    assert result["verdict"] == {
        "reason_codes": [
            "INVALID_FIT_ROWS",
            "INVALID_MODEL_CALLS",
            "INVALID_SHARED_AB_ROWS",
        ],
        "status": "INVALID",
    }


def test_candidate_hand_cases_lock_two_directional_mixed_state_readout() -> None:
    pass_review = Review("x", "a", Observation.PASS, ExecutionState.VALID)
    fail_review = Review("x", "b", Observation.FAIL, ExecutionState.VALID)
    invalid_review = Review("x", "c", None, ExecutionState.INVALID)
    abstain_review = Review("x", "c", Observation.ABSTAIN, ExecutionState.VALID)
    directional_with_invalid = (pass_review, fail_review, invalid_review)
    directional_with_abstain = (pass_review, fail_review, abstain_review)
    assert candidate_action(0.75, 2) is Action.PASS
    assert candidate_action(0.25, 2) is Action.FAIL
    assert candidate_action(0.5, 2) is Action.DEFER
    assert candidate_action(0.9, 1) is Action.DEFER
    assert candidate(0.75, directional_with_invalid) is Action.PASS
    assert candidate(0.25, directional_with_invalid) is Action.FAIL
    assert candidate(0.5, directional_with_abstain) is Action.DEFER
    assert candidate(0.9, (pass_review, abstain_review, invalid_review)) is Action.DEFER


def test_weighted_vote_hand_logit_sigmoid_and_reversal() -> None:
    rows = (
        Review("x", "a", Observation.PASS, ExecutionState.VALID),
        Review("x", "b", Observation.FAIL, ExecutionState.VALID),
        Review("x", "c", None, ExecutionState.TIMEOUT),
    )
    positive_weights = {"a": math.log(4), "b": math.log(2), "c": 100.0}
    positive = vote(rows, positive_weights)
    assert positive == vote(tuple(reversed(rows)), positive_weights)
    assert positive[0] is Action.PASS
    assert positive[1] == pytest.approx(1 / (1 + math.exp(-math.log(2))))
    negative_weights = {"a": math.log(2), "b": math.log(4), "c": 100.0}
    negative = vote(rows, negative_weights)
    assert negative == vote(tuple(reversed(rows)), negative_weights)
    assert negative[0] is Action.FAIL
    assert negative[1] == pytest.approx(1 / (1 + math.exp(math.log(2))))
    tie_weights = {"a": math.log(3), "b": math.log(3), "c": 100.0}
    assert vote(rows, tie_weights) == vote(tuple(reversed(rows)), tie_weights) == (
        Action.DEFER,
        0.5,
    )


def test_formal_pytest_argv_guard_accepts_only_registered_single_node() -> None:
    node = "tests/test_convergence_resolution_value.py::test_locked_formal_gate"
    for argv in ([node], ["-q", node, "-s"], [node, "-qs"]):
        validate_formal_pytest_argv(argv)
    for argv in (
        ["tests/test_convergence_resolution_value.py"],
        [node, "tests/test_models.py"],
        [node, "-x"],
        [node, node],
        ["-q"],
    ):
        with pytest.raises(RuntimeError, match="single-node formal invocation"):
            validate_formal_pytest_argv(argv)


def test_static_preflight_rejects_switched_whole_file_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["pytest", "tests/test_convergence_resolution_value.py", "-q"],
    )
    with pytest.raises(RuntimeError, match="single-node formal invocation"):
        static_formal_preflight()
    assert not any(path.exists() for path in RESULTS)


def test_default_skip_assertion_is_safe_when_formal_switch_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(pytest.skip.Exception):
        test_static_preflight_default_external_skip()


if __name__ == "__main__":  # pragma: no cover - explicit post-judge operator entrypoint
    raise SystemExit(recorder_cli())
