# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.3.2",
# ]
# ///
"""Characterize residual fill errors in exported v9 board evidence."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FillEvidence:
    atlases: list[str]
    has_set: np.ndarray
    logits: np.ndarray
    targets: np.ndarray
    card_targets: np.ndarray


def parse_vector(value: str) -> np.ndarray:
    return np.fromstring(value, dtype=np.float32, sep=",")


def read_fill_evidence(path: Path) -> FillEvidence:
    atlases: list[str] = []
    kinds: list[bool] = []
    logits: list[np.ndarray] = []
    card_targets: list[np.ndarray] = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            atlases.append(row["atlas"])
            kinds.append(row["expected_kind"] == "HasSet")
            attributes = parse_vector(row["attribute_logits"]).reshape(12, 4, 3)
            targets = parse_vector(row["card_targets"]).astype(np.int64).reshape(12, 4)
            logits.append(attributes[:, 3, :])
            card_targets.append(targets)
    all_targets = np.stack(card_targets)
    return FillEvidence(
        atlases=atlases,
        has_set=np.asarray(kinds, dtype=np.bool_),
        logits=np.stack(logits),
        targets=all_targets[:, :, 3],
        card_targets=all_targets,
    )


def atlas_profile(atlas: str) -> str:
    marker = "-p"
    if marker not in atlas:
        return "unknown"
    return atlas.split(marker, maxsplit=1)[1].split("-dpr", maxsplit=1)[0]


def card_id(attributes: np.ndarray) -> int:
    return int(sum(int(value) * 3**index for index, value in enumerate(attributes)))


def report_model(name: str, evidence: FillEvidence) -> np.ndarray:
    predictions = evidence.logits.argmax(axis=-1)
    wrong = predictions != evidence.targets
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (evidence.targets.reshape(-1), predictions.reshape(-1)), 1)
    probabilities = np.exp(
        evidence.logits - evidence.logits.max(axis=-1, keepdims=True)
    )
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    ranked = np.sort(probabilities, axis=-1)
    margins = ranked[..., -1] - ranked[..., -2]

    print(
        f"{name}: accuracy={1.0 - wrong.mean():.6f} errors={wrong.sum()} "
        f"wrong_margin_mean={margins[wrong].mean():.6f} "
        f"wrong_margin_p90={np.quantile(margins[wrong], 0.9):.6f}"
    )
    print(f"{name}_confusion_rows_true_cols_predicted=")
    for row in confusion:
        print("  " + " ".join(str(int(value)) for value in row))
    for target in range(3):
        mask = evidence.targets == target
        print(
            f"  true_{target}: accuracy={1.0 - wrong[mask].mean():.6f} "
            f"errors={wrong[mask].sum()} total={mask.sum()}"
        )

    profile_errors: Counter[str] = Counter()
    profile_totals: Counter[str] = Counter()
    atlas_errors: Counter[str] = Counter()
    atlas_totals: Counter[str] = Counter()
    card_errors: Counter[int] = Counter()
    card_totals: Counter[int] = Counter()
    kind_errors: Counter[str] = Counter()
    kind_totals: Counter[str] = Counter()
    for board, atlas in enumerate(evidence.atlases):
        profile = atlas_profile(atlas)
        kind = "has_set" if evidence.has_set[board] else "no_set"
        for card in range(12):
            identifier = card_id(evidence.card_targets[board, card])
            profile_totals[profile] += 1
            atlas_totals[atlas] += 1
            card_totals[identifier] += 1
            kind_totals[kind] += 1
            if wrong[board, card]:
                profile_errors[profile] += 1
                atlas_errors[atlas] += 1
                card_errors[identifier] += 1
                kind_errors[kind] += 1

    def ranked_rates(
        errors: Counter[str] | Counter[int],
        totals: Counter[str] | Counter[int],
        limit: int,
    ) -> list[tuple[object, int, int, float]]:
        rows = [
            (key, errors[key], total, errors[key] / total)
            for key, total in totals.items()
            if errors[key]
        ]
        return sorted(rows, key=lambda row: (row[3], row[1]), reverse=True)[:limit]

    print(f"{name}_errors_by_kind={dict(kind_errors)} totals={dict(kind_totals)}")
    print(f"{name}_worst_profiles={ranked_rates(profile_errors, profile_totals, 12)}")
    print(f"{name}_worst_cards={ranked_rates(card_errors, card_totals, 15)}")
    print(f"{name}_worst_atlases={ranked_rates(atlas_errors, atlas_totals, 15)}")
    return wrong


def main() -> None:
    if len(sys.argv) not in (2, 6):
        raise SystemExit(
            "usage: uv run vision/analyze_fill_errors.py "
            "MODEL.tsv\n"
            "   or: uv run vision/analyze_fill_errors.py "
            "ENSEMBLE.tsv MEMBER_1.tsv MEMBER_2.tsv MEMBER_3.tsv MEMBER_4.tsv"
        )
    paths = [Path(value) for value in sys.argv[1:]]
    streams = [read_fill_evidence(path) for path in paths]
    names = (
        ("model",)
        if len(paths) == 1
        else ("ensemble", "member_1", "member_2", "member_3", "member_4")
    )
    wrong_masks = [
        report_model(name, evidence)
        for name, evidence in zip(names, streams, strict=True)
    ]
    if len(wrong_masks) == 1:
        return
    member_wrong_count = np.stack(wrong_masks[1:]).sum(axis=0)
    ensemble_wrong = wrong_masks[0]
    overlap = defaultdict(lambda: [0, 0])
    for count, wrong in zip(
        member_wrong_count.reshape(-1), ensemble_wrong.reshape(-1), strict=True
    ):
        overlap[int(count)][0] += 1
        overlap[int(count)][1] += int(wrong)
    print("ensemble_error_by_wrong_member_count=")
    for count in sorted(overlap):
        total, errors = overlap[count]
        print(f"  wrong_members={count} total={total} ensemble_errors={errors}")


if __name__ == "__main__":
    main()
