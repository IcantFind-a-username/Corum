from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from corum.calibration import PairKey
from corum.dependence import DependenceModel
from corum.fusion import FusionContext, fuse_review_matrix

_DEFAULT_CHUNK_SIZE = 4_096
_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Corum's locked matrix-fusion workload.",
    )
    parser.add_argument("--cases", type=_positive_int, default=10_000)
    parser.add_argument("--reviewers", type=_positive_int, default=3)
    parser.add_argument("--draws", type=_positive_int, default=512)
    parser.add_argument("--max-seconds", type=_positive_float, default=5.0)
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=_DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--pair-block",
        action="store_true",
        help="Activate adjacent disjoint reviewer-pair likelihood blocks.",
    )
    return parser.parse_args(argv)


def _build_context(
    reviewer_ids: tuple[str, ...],
    draws: int,
    seed: int,
    *,
    pair_block: bool,
) -> FusionContext:
    base_likelihoods = np.array(
        [[0.78, 0.17, 0.05], [0.16, 0.79, 0.05]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    likelihood_draws = {
        reviewer_id: np.stack(
            tuple(
                rng.dirichlet(truth_row * 200.0, size=draws)
                for truth_row in base_likelihoods
            ),
            axis=1,
        )
        for reviewer_id in reviewer_ids
    }
    pair_keys: tuple[PairKey, ...] = (
        tuple(
            (reviewer_ids[index], reviewer_ids[index + 1])
            for index in range(0, len(reviewer_ids) - 1, 2)
        )
        if pair_block
        else ()
    )
    product_joint = np.stack(
        [np.outer(row, row) for row in base_likelihoods],
    )
    pair_likelihood_draws = {
        pair: np.stack(
            [
                rng.dirichlet(
                    product_joint[truth_index].reshape(-1) * 400.0,
                    size=draws,
                ).reshape(draws, 3, 3)
                for truth_index in range(2)
            ],
            axis=1,
        )
        for pair in pair_keys
    }
    dependence = DependenceModel(
        reviewer_ids=reviewer_ids,
        correlation=np.eye(len(reviewer_ids), dtype=np.float64),
        lineage_by_reviewer={
            reviewer_id: f"lineage-{index}"
            for index, reviewer_id in enumerate(reviewer_ids)
        },
    )
    return FusionContext(
        likelihood_draws=likelihood_draws,
        dependence=dependence,
        lineage_by_reviewer=dependence.lineage_by_reviewer,
        prior_pass=0.5,
        credible_mass=0.95,
        pair_likelihood_draws=pair_likelihood_draws,
    )


def _working_array_accounting(
    *,
    cases: int,
    reviewers: int,
    draws: int,
    chunk_size: int,
    pair_count: int,
) -> dict[str, int]:
    active_rows = min(cases, chunk_size)
    float_bytes = np.dtype(np.float64).itemsize
    int_bytes = np.dtype(np.int64).itemsize
    bool_bytes = np.dtype(np.bool_).itemsize

    inputs = cases * reviewers * (int_bytes + bool_bytes)
    context = reviewers * draws * 2 * 3 * float_bytes
    pair_context = pair_count * draws * 2 * 3 * 3 * float_bytes
    output_and_defensive_copies = 2 * cases * 6 * float_bytes
    mask_grouping = active_rows * (reviewers * bool_bytes + int_bytes + bool_bytes)
    # The shared kernel can have log masses, gathered likelihoods, stabilized
    # masses, posterior samples, and quantile work alive together. Ten full
    # chunk-by-draw float matrices is a conservative upper accounting.
    kernel_float_matrices = 10 * active_rows * draws * float_bytes
    pair_kernel_float_matrices = (
        2 * active_rows * draws * float_bytes if pair_count else 0
    )
    total = (
        inputs
        + context
        + pair_context
        + output_and_defensive_copies
        + mask_grouping
        + kernel_float_matrices
        + pair_kernel_float_matrices
    )
    return {
        "inputs": inputs,
        "context_likelihood_draws": context,
        "context_pair_likelihood_draws": pair_context,
        "output_and_defensive_copies": output_and_defensive_copies,
        "mask_grouping": mask_grouping,
        "kernel_float_matrices_conservative": kernel_float_matrices,
        "pair_kernel_float_matrices_conservative": pair_kernel_float_matrices,
        "total": total,
    }


def _environment_report() -> dict[str, Any]:
    return {
        "cpu_model": (
            platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER")
            or platform.machine()
        ),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_VARIABLES
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    reviewer_ids = tuple(f"reviewer-{index}" for index in range(args.reviewers))
    if args.pair_block and len(reviewer_ids) < 2:
        raise ValueError("--pair-block requires at least two reviewers")
    context = _build_context(
        reviewer_ids,
        args.draws,
        args.seed,
        pair_block=bool(args.pair_block),
    )
    rng = np.random.default_rng(args.seed)
    observations = rng.integers(
        0,
        3,
        size=(args.cases, args.reviewers),
        dtype=np.int64,
    )
    valid_mask = np.ones((args.cases, args.reviewers), dtype=np.bool_)

    started = time.perf_counter()
    result = fuse_review_matrix(
        observations,
        valid_mask,
        reviewer_ids,
        context,
        chunk_size=args.chunk_size,
    )
    elapsed_seconds = time.perf_counter() - started

    all_abstain_rows = np.all(observations == 2, axis=1)
    if (
        result.pass_probability.shape != (args.cases,)
        or not np.all(np.isfinite(result.pass_probability))
        or np.any(result.lower[all_abstain_rows] != 0.0)
        or np.any(result.upper[all_abstain_rows] != 1.0)
        or np.any(result.valid_reviewers != args.reviewers)
    ):
        raise RuntimeError("benchmark fusion returned invalid output")

    accounting = _working_array_accounting(
        cases=args.cases,
        reviewers=args.reviewers,
        draws=args.draws,
        chunk_size=args.chunk_size,
        pair_count=len(context.pair_likelihood_draws),
    )
    report: dict[str, Any] = {
        "workload": {
            "cases": args.cases,
            "reviewers": args.reviewers,
            "draws": args.draws,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "pair_block": bool(args.pair_block),
            "pair_keys": [list(pair) for pair in context.pair_likelihood_draws],
        },
        "elapsed_seconds": elapsed_seconds,
        "elapsed_method": "time.perf_counter",
        "max_seconds": args.max_seconds,
        "within_time_limit": elapsed_seconds <= args.max_seconds,
        "working_array_bytes": accounting,
        "working_array_limit_bytes": _MEMORY_LIMIT_BYTES,
        "within_working_array_limit": accounting["total"] < _MEMORY_LIMIT_BYTES,
        "environment": _environment_report(),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = run_benchmark(_parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["within_time_limit"]:
        return 1
    if not report["within_working_array_limit"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
