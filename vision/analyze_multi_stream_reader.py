# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.3.2",
# ]
# ///
"""Train the contextual v9 reader on one board stream and test another."""

from __future__ import annotations

import sys
from pathlib import Path

import analyze_board_existence as existence
import analyze_contextual_reader as contextual
import numpy as np


def read_stream(
    paths: list[str],
) -> tuple[contextual.Evidence, list[contextual.Evidence]]:
    evidence = contextual.read_evidence(Path(paths[0]))
    members = [contextual.read_evidence(Path(path)) for path in paths[1:]]
    for member in members:
        if (
            not np.array_equal(member.triple_targets, evidence.triple_targets)
            or not np.array_equal(member.card_targets, evidence.card_targets)
            or not np.array_equal(member.has_set, evidence.has_set)
        ):
            raise ValueError("member evidence does not match its ensemble board stream")
    return evidence, members


def conservative_split_threshold(
    scores: np.ndarray,
    triple_targets: np.ndarray,
    has_set: np.ndarray,
) -> float:
    midpoint = scores.shape[0] // 2
    first = contextual.fast_best_threshold_and_accuracy(
        triple_targets[:midpoint], has_set[:midpoint], scores[:midpoint]
    )[0]
    second = contextual.fast_best_threshold_and_accuracy(
        triple_targets[midpoint:], has_set[midpoint:], scores[midpoint:]
    )[0]
    return max(first, second)


def main() -> None:
    if len(sys.argv) != 11:
        raise SystemExit(
            "usage: uv run vision/analyze_multi_stream_reader.py "
            "DEV_ENSEMBLE.tsv DEV_MEMBER_1.tsv ... DEV_MEMBER_4.tsv "
            "TEST_ENSEMBLE.tsv TEST_MEMBER_1.tsv ... TEST_MEMBER_4.tsv"
        )
    development, development_members = read_stream(sys.argv[1:6])
    testing, testing_members = read_stream(sys.argv[6:11])
    validation_start = development.attribute_logits.shape[0] * 3 // 4
    attribute_names = ("count", "colour", "shape", "fill")
    all_base_logits = np.concatenate(
        (development.attribute_logits, testing.attribute_logits), axis=0
    )
    stacked_logits = np.empty_like(all_base_logits)

    for attribute, name in enumerate(attribute_names):
        development_features, development_base = contextual.stacker_features_and_base(
            [
                member.attribute_logits[:, :, attribute, :]
                for member in development_members
            ]
        )
        testing_features, testing_base = contextual.stacker_features_and_base(
            [member.attribute_logits[:, :, attribute, :] for member in testing_members]
        )
        corrector, _, _, _, _ = contextual.train_feature_corrector(
            development_features[:validation_start],
            development_base[:validation_start],
            development.card_targets[:validation_start, :, attribute],
            development_features[validation_start:],
            development_base[validation_start:],
            development.card_targets[validation_start:, :, attribute],
            seed=50_000 + attribute,
        )
        stacked_logits[: development.attribute_logits.shape[0], :, attribute, :] = (
            contextual.apply_feature_corrector(
                development_features, development_base, corrector
            )
        )
        stacked_logits[development.attribute_logits.shape[0] :, :, attribute, :] = (
            contextual.apply_feature_corrector(
                testing_features, testing_base, corrector
            )
        )
        print(f"stacked_{name}=ready", flush=True)

    triple_scores = np.concatenate(
        (development.triple_scores, testing.triple_scores), axis=0
    )
    relation_logits = np.concatenate(
        (development.relation_logits, testing.relation_logits), axis=0
    )
    hypergraph_features = contextual.hypergraph_features(
        stacked_logits, triple_scores, relation_logits
    )
    corrector, _, _ = contextual.train_hypergraph_corrector(
        hypergraph_features[:validation_start],
        stacked_logits[:validation_start],
        development.card_targets[:validation_start],
        hypergraph_features[validation_start : development.attribute_logits.shape[0]],
        stacked_logits[validation_start : development.attribute_logits.shape[0]],
        development.card_targets[validation_start:],
        seed=55_008,
    )
    corrected_logits = contextual.apply_hypergraph_corrector(
        stacked_logits, hypergraph_features, corrector
    )
    development_count = development.attribute_logits.shape[0]
    contextual.report_attributes(
        "multi_stream_attributes_test",
        corrected_logits[development_count:],
        testing.card_targets,
    )

    exact_relations = existence.differentiable_relation_features(corrected_logits)
    development_members_scores: list[np.ndarray] = []
    testing_members_scores: list[np.ndarray] = []
    selected: tuple[float, int] | None = None
    for replica in range(8):
        model, mean, scale = contextual.train_binary_mlp(
            exact_relations[:development_count],
            development.triple_targets,
            hidden_width=24,
            seed=80_001 + replica,
            steps=1_500,
        )
        development_scores = contextual.predict_mlp(
            exact_relations[:development_count], model, mean, scale
        )[:, :, 0]
        testing_scores = contextual.predict_mlp(
            exact_relations[development_count:], model, mean, scale
        )[:, :, 0]
        development_members_scores.append(development_scores)
        testing_members_scores.append(testing_scores)
        threshold = conservative_split_threshold(
            development_scores, development.triple_targets, development.has_set
        )
        development_accuracy = float(
            contextual.action_correct(
                development.triple_targets,
                development.has_set,
                development_scores,
                threshold,
            ).mean()
        )
        candidate = (development_accuracy, -replica)
        if selected is None or candidate > selected:
            selected = candidate
        contextual.report_action(
            f"multi_stream_exact_relation_{replica + 1}_test",
            testing.triple_targets,
            testing.has_set,
            testing_scores,
            threshold,
        )
        oracle_threshold, oracle_accuracy = contextual.fast_best_threshold_and_accuracy(
            testing.triple_targets, testing.has_set, testing_scores
        )
        print(
            f"multi_stream_exact_relation_{replica + 1}: "
            f"development={development_accuracy:.6f} threshold={threshold:.9f} "
            f"test_oracle_threshold={oracle_threshold:.9f} "
            f"test_oracle_accuracy={oracle_accuracy:.6f}",
            flush=True,
        )

    if selected is None:
        raise RuntimeError("no exact-relation replicas were trained")
    print(
        f"development_selected_replica={-selected[1] + 1} "
        f"development_accuracy={selected[0]:.6f}",
        flush=True,
    )
    normalized_development: list[np.ndarray] = []
    normalized_testing: list[np.ndarray] = []
    for development_scores, testing_scores in zip(
        development_members_scores, testing_members_scores, strict=True
    ):
        mean = development_scores.mean()
        scale = max(float(development_scores.std()), 1.0e-6)
        normalized_development.append((development_scores - mean) / scale)
        normalized_testing.append((testing_scores - mean) / scale)
    for count in (2, 4, 8):
        development_scores = np.mean(normalized_development[:count], axis=0)
        testing_scores = np.mean(normalized_testing[:count], axis=0)
        threshold = conservative_split_threshold(
            development_scores, development.triple_targets, development.has_set
        )
        contextual.report_action(
            f"multi_stream_exact_ensemble_{count}_test",
            testing.triple_targets,
            testing.has_set,
            testing_scores,
            threshold,
        )
        if count == 4:
            contextual.report_joint_board_reader(
                "multi_stream_exact_ensemble_4_board_reader_test",
                development_scores,
                testing_scores,
                [*normalized_development[:count], development_scores],
                [*normalized_testing[:count], testing_scores],
                development.triple_targets,
                testing.triple_targets,
                development.has_set,
                testing.has_set,
            )
    contextual.search_weighted_score_ensemble(
        "multi_stream_exact_weighted_4_test",
        development_members_scores[:4],
        testing_members_scores[:4],
        development.triple_targets,
        testing.triple_targets,
        development.has_set,
        testing.has_set,
        units=10,
    )


if __name__ == "__main__":
    main()
