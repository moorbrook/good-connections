# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.3.2",
# ]
# ///
"""Inspect deterministic tie-breaking for v9 board-action thresholds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def threshold_curve(
    scores: np.ndarray, triple_targets: np.ndarray, has_set: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    best = scores.argmax(axis=1)
    board_scores = scores[np.arange(scores.shape[0]), best]
    selected_is_set = triple_targets[np.arange(scores.shape[0]), best]
    thresholds = np.unique(board_scores)
    predictions = board_scores[:, None] >= thresholds[None, :]
    correct = np.where(
        predictions,
        (has_set & selected_is_set)[:, None],
        (~has_set)[:, None],
    )
    return thresholds, correct.mean(axis=0)


def choices(
    scores: np.ndarray, triple_targets: np.ndarray, has_set: np.ndarray
) -> tuple[float, float, float, float]:
    thresholds, accuracies = threshold_curve(scores, triple_targets, has_set)
    best = accuracies.max()
    tied = thresholds[np.isclose(accuracies, best, rtol=0.0, atol=1.0e-12)]
    low = float(tied.min())
    high = float(tied.max())
    return low, high, (low + high) / 2.0, float(best)


def evaluate(
    scores: np.ndarray,
    threshold: float,
    triple_targets: np.ndarray,
    has_set: np.ndarray,
) -> tuple[float, float, float, int]:
    best = scores.argmax(axis=1)
    board_scores = scores[np.arange(scores.shape[0]), best]
    selected_is_set = triple_targets[np.arange(scores.shape[0]), best]
    correct = np.where(
        board_scores >= threshold,
        has_set & selected_is_set,
        ~has_set,
    )
    return (
        float(correct.mean()),
        float(correct[has_set].mean()),
        float(correct[~has_set].mean()),
        int(np.count_nonzero(~correct)),
    )


def report(
    name: str,
    scores: np.ndarray,
    triple_targets: np.ndarray,
    has_set: np.ndarray,
) -> None:
    midpoint = scores.shape[0] // 2
    development_scores = scores[:midpoint]
    development_targets = triple_targets[:midpoint]
    development_kinds = has_set[:midpoint]
    testing_scores = scores[midpoint:]
    testing_targets = triple_targets[midpoint:]
    testing_kinds = has_set[midpoint:]
    low, high, middle, development_accuracy = choices(
        development_scores, development_targets, development_kinds
    )
    first_high = choices(
        development_scores[: midpoint // 2],
        development_targets[: midpoint // 2],
        development_kinds[: midpoint // 2],
    )[1]
    second_high = choices(
        development_scores[midpoint // 2 :],
        development_targets[midpoint // 2 :],
        development_kinds[midpoint // 2 :],
    )[1]
    split_median = (first_high + second_high) / 2.0
    split_conservative = max(first_high, second_high)
    print(
        f"{name}: development={development_accuracy:.6f} "
        f"tie=[{low:.9f},{high:.9f}] split_high=[{first_high:.9f},{second_high:.9f}]"
    )
    for choice_name, threshold in (
        ("low", low),
        ("high", high),
        ("middle", middle),
        ("split_median", split_median),
        ("split_conservative", split_conservative),
    ):
        overall, set_accuracy, no_set_accuracy, failures = evaluate(
            testing_scores, threshold, testing_targets, testing_kinds
        )
        print(
            f"  {choice_name}: threshold={threshold:.9f} overall={overall:.6f} "
            f"set={set_accuracy:.6f} no_set={no_set_accuracy:.6f} failures={failures}"
        )
    _, _, _, oracle_accuracy = choices(testing_scores, testing_targets, testing_kinds)
    print(f"  threshold_oracle={oracle_accuracy:.6f}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: uv run vision/analyze_threshold_calibration.py board-existence-probe.npz"
        )
    data = np.load(Path(sys.argv[1]))
    targets = data["triple_targets"]
    has_set = data["has_set"]
    report("soft", data["soft_scores"], targets, has_set)
    exact = data["exact_scores"]
    for index, scores in enumerate(exact, start=1):
        report(f"exact_{index}", scores, targets, has_set)
    report("exact_ensemble", exact.mean(axis=0), targets, has_set)


if __name__ == "__main__":
    main()
