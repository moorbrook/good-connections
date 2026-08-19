# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.3.2",
# ]
# ///
"""Probe a permutation-invariant board-existence head on frozen v9 evidence."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import analyze_contextual_reader as contextual
import numpy as np

FloatArray = np.ndarray
BoolArray = np.ndarray


def differentiable_relation_features(attribute_logits: FloatArray) -> FloatArray:
    """Return exact soft same/different/invalid probabilities for each attribute."""
    probabilities = contextual.softmax(attribute_logits)
    cards = probabilities[:, contextual.TRIPLES]
    same = np.prod(cards, axis=2).sum(axis=-1)
    different = np.zeros_like(same)
    for permutation in itertools.permutations(range(3)):
        different += (
            cards[:, :, 0, :, permutation[0]]
            * cards[:, :, 1, :, permutation[1]]
            * cards[:, :, 2, :, permutation[2]]
        )
    invalid = np.clip(1.0 - same - different, 0.0, 1.0)
    return (
        np.stack((same, different, invalid), axis=-1)
        .reshape(attribute_logits.shape[0], contextual.TRIPLES.shape[0], 12)
        .astype(np.float32)
    )


def differentiable_set_scores(attribute_logits: FloatArray) -> FloatArray:
    """Return exact soft SET log-probabilities under independent card marginals."""
    relations = differentiable_relation_features(attribute_logits).reshape(
        attribute_logits.shape[0], contextual.TRIPLES.shape[0], 4, 3
    )
    valid = np.clip(relations[..., 0] + relations[..., 1], 1.0e-8, 1.0)
    return np.log(valid).sum(axis=2).astype(np.float32)


def differentiable_attribute_scores(attribute_logits: FloatArray) -> FloatArray:
    relations = differentiable_relation_features(attribute_logits).reshape(
        attribute_logits.shape[0], contextual.TRIPLES.shape[0], 4, 3
    )
    valid = np.clip(relations[..., 0] + relations[..., 1], 1.0e-8, 1.0)
    return np.log(valid).astype(np.float32)


def report_weighted_soft_relation(
    attribute_scores: FloatArray,
    evidence: contextual.Evidence,
    midpoint: int,
) -> None:
    candidates: list[tuple[float, float, float, float, float]] = []
    values = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    for shape_weight in values:
        for fill_weight in values:
            weights = np.asarray(
                (1.0, 1.0, shape_weight, fill_weight), dtype=np.float32
            )
            scores = (attribute_scores[:midpoint] * weights).sum(axis=2)
            threshold, accuracy = contextual.fast_best_threshold_and_accuracy(
                evidence.triple_targets[:midpoint],
                evidence.has_set[:midpoint],
                scores,
            )
            distance = (shape_weight - 1.0) ** 2 + (fill_weight - 1.0) ** 2
            candidates.append(
                (accuracy, -distance, shape_weight, fill_weight, threshold)
            )
    training_accuracy, _, shape_weight, fill_weight, threshold = max(candidates)
    weights = np.asarray((1.0, 1.0, shape_weight, fill_weight), dtype=np.float32)
    testing_scores = (attribute_scores[midpoint:] * weights).sum(axis=2)
    print(
        f"weighted_soft_relation_selection: shape={shape_weight:.2f} "
        f"fill={fill_weight:.2f} training_accuracy={training_accuracy:.6f}",
        flush=True,
    )
    contextual.report_action(
        "weighted_soft_relation",
        evidence.triple_targets[midpoint:],
        evidence.has_set[midpoint:],
        testing_scores,
        threshold,
    )


def hard_set_scores(attribute_logits: FloatArray) -> FloatArray:
    """Diagnostic ceiling for argmax attributes followed by the SET invariant."""
    attributes = attribute_logits.argmax(axis=-1)
    triples = attributes[:, contextual.TRIPLES]
    valid = (triples.sum(axis=2) % 3 == 0).all(axis=2)
    return valid.astype(np.float32)


def invariant_board_features(
    scores: FloatArray,
    relations: FloatArray,
    attribute_logits: FloatArray,
) -> FloatArray:
    """Summarize a board without exposing card or triple ordering."""
    ranked_scores = np.sort(scores, axis=1)[:, ::-1]
    valid_per_attribute = 1.0 - relations[..., 1]
    triple_validity = np.prod(valid_per_attribute, axis=2)
    ranked_validity = np.sort(triple_validity, axis=1)[:, ::-1]

    attribute_probabilities = contextual.softmax(attribute_logits)
    ranked_card_confidence = np.sort(
        attribute_probabilities.max(axis=-1).reshape(scores.shape[0], -1), axis=1
    )[:, ::-1]
    ranked_card_entropy = np.sort(
        (-attribute_probabilities * np.log(np.maximum(attribute_probabilities, 1.0e-8)))
        .sum(axis=-1)
        .reshape(scores.shape[0], -1),
        axis=1,
    )[:, ::-1]

    top_score_count = min(96, ranked_scores.shape[1])
    top_validity_count = min(96, ranked_validity.shape[1])
    return np.concatenate(
        (
            ranked_scores[:, :top_score_count],
            ranked_validity[:, :top_validity_count],
            ranked_card_confidence,
            ranked_card_entropy,
            contextual.board_summary_features(scores),
            contextual.board_summary_features(triple_validity),
        ),
        axis=1,
    ).astype(np.float32)


def fit_board_mlp(
    features: FloatArray,
    targets: BoolArray,
    seed: int,
) -> tuple[contextual.Mlp, FloatArray, FloatArray]:
    return contextual.train_binary_mlp(
        features[:, None, :],
        targets[:, None],
        hidden_width=64,
        seed=seed,
        steps=3_000,
    )


def predict_board_mlp(
    features: FloatArray,
    model: contextual.Mlp,
    mean: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    return contextual.predict_mlp(features[:, None, :], model, mean, scale)[:, 0, 0]


def action_accuracy(
    board_scores: FloatArray,
    threshold: float,
    selector_scores: FloatArray,
    triple_targets: BoolArray,
    has_set: BoolArray,
) -> tuple[float, float, float, int]:
    best = selector_scores.argmax(axis=1)
    selected_is_set = triple_targets[np.arange(triple_targets.shape[0]), best]
    predict_set = board_scores >= threshold
    correct = np.where(predict_set, has_set & selected_is_set, ~has_set)
    return (
        float(correct.mean()),
        float(correct[has_set].mean()),
        float(correct[~has_set].mean()),
        int(np.count_nonzero(~correct)),
    )


def best_board_threshold(
    board_scores: FloatArray,
    selector_scores: FloatArray,
    triple_targets: BoolArray,
    has_set: BoolArray,
) -> tuple[float, float]:
    best = selector_scores.argmax(axis=1)
    selected_is_set = triple_targets[np.arange(triple_targets.shape[0]), best]
    order = np.argsort(-board_scores)
    correct_count = int(np.count_nonzero(~has_set))
    best_correct = correct_count
    threshold = float(np.nextafter(board_scores.max(), np.inf))
    position = 0
    while position < order.size:
        end = position + 1
        score = board_scores[order[position]]
        while end < order.size and board_scores[order[end]] == score:
            end += 1
        group = order[position:end]
        correct_count += int(np.count_nonzero(has_set[group] & selected_is_set[group]))
        correct_count -= int(np.count_nonzero(~has_set[group]))
        if correct_count > best_correct:
            best_correct = correct_count
            threshold = float(score)
        position = end
    return threshold, best_correct / board_scores.size


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: uv run vision/analyze_board_existence.py ENSEMBLE.tsv "
            "MEMBER_1.tsv MEMBER_2.tsv MEMBER_3.tsv MEMBER_4.tsv"
        )
    evidence = contextual.read_evidence(Path(sys.argv[1]))
    members = [contextual.read_evidence(Path(path)) for path in sys.argv[2:]]
    midpoint = evidence.attribute_logits.shape[0] // 2
    calibration_start = midpoint * 3 // 4
    attribute_names = ("count", "colour", "shape", "fill")

    stacked_logits = np.empty_like(evidence.attribute_logits)
    for attribute, name in enumerate(attribute_names):
        features, base = contextual.stacker_features_and_base(
            [member.attribute_logits[:, :, attribute, :] for member in members]
        )
        corrector, _, _, _, _ = contextual.train_feature_corrector(
            features[:calibration_start],
            base[:calibration_start],
            evidence.card_targets[:calibration_start, :, attribute],
            features[calibration_start:midpoint],
            base[calibration_start:midpoint],
            evidence.card_targets[calibration_start:midpoint, :, attribute],
            seed=50_000 + attribute,
        )
        stacked_logits[:, :, attribute, :] = contextual.apply_feature_corrector(
            features, base, corrector
        )
        print(f"stacked_{name}=ready", flush=True)

    hypergraph_features = contextual.hypergraph_features(
        stacked_logits, evidence.triple_scores, evidence.relation_logits
    )
    corrector, _, _ = contextual.train_hypergraph_corrector(
        hypergraph_features[:calibration_start],
        stacked_logits[:calibration_start],
        evidence.card_targets[:calibration_start],
        hypergraph_features[calibration_start:midpoint],
        stacked_logits[calibration_start:midpoint],
        evidence.card_targets[calibration_start:midpoint],
        seed=55_008,
    )
    corrected_logits = contextual.apply_hypergraph_corrector(
        stacked_logits, hypergraph_features, corrector
    )
    for name, scores in (
        ("stacked_soft_relation", differentiable_set_scores(stacked_logits)),
        ("corrected_soft_relation", differentiable_set_scores(corrected_logits)),
        ("corrected_hard_relation", hard_set_scores(corrected_logits)),
    ):
        threshold = contextual.best_threshold(
            evidence.triple_targets[:midpoint],
            evidence.has_set[:midpoint],
            scores[:midpoint],
        )
        contextual.report_action(
            name,
            evidence.triple_targets[midpoint:],
            evidence.has_set[midpoint:],
            scores[midpoint:],
            threshold,
        )
    report_weighted_soft_relation(
        differentiable_attribute_scores(corrected_logits), evidence, midpoint
    )
    pipeline = contextual.run_typed_pipeline(
        "existence_selector",
        contextual.softmax(corrected_logits[:midpoint]),
        contextual.softmax(corrected_logits[midpoint:]),
        evidence.card_targets[:midpoint],
        evidence.card_targets[midpoint:],
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
        seed=60_008,
    )
    exact_relations = differentiable_relation_features(corrected_logits)
    exact_training_scores: list[FloatArray] = []
    exact_testing_scores: list[FloatArray] = []
    for replica in range(4):
        model, mean, scale = contextual.train_binary_mlp(
            exact_relations[:midpoint],
            evidence.triple_targets[:midpoint],
            hidden_width=24,
            seed=80_001 + replica,
            steps=1_500,
        )
        exact_training_scores.append(
            contextual.predict_mlp(exact_relations[:midpoint], model, mean, scale)[
                :, :, 0
            ]
        )
        exact_testing_scores.append(
            contextual.predict_mlp(exact_relations[midpoint:], model, mean, scale)[
                :, :, 0
            ]
        )
        threshold = contextual.best_threshold(
            evidence.triple_targets[:midpoint],
            evidence.has_set[:midpoint],
            exact_training_scores[-1],
        )
        contextual.report_action(
            f"learned_exact_relation_{replica + 1}",
            evidence.triple_targets[midpoint:],
            evidence.has_set[midpoint:],
            exact_testing_scores[-1],
            threshold,
        )
        oracle_threshold, oracle_accuracy = contextual.fast_best_threshold_and_accuracy(
            evidence.triple_targets[midpoint:],
            evidence.has_set[midpoint:],
            exact_testing_scores[-1],
        )
        print(
            f"learned_exact_relation_{replica + 1}_threshold_oracle: "
            f"accuracy={oracle_accuracy:.6f} threshold={oracle_threshold:.9f}",
            flush=True,
        )
    exact_ensemble_training = np.mean(exact_training_scores, axis=0)
    exact_ensemble_testing = np.mean(exact_testing_scores, axis=0)
    exact_ensemble_threshold = contextual.best_threshold(
        evidence.triple_targets[:midpoint],
        evidence.has_set[:midpoint],
        exact_ensemble_training,
    )
    contextual.report_action(
        "learned_exact_relation_ensemble",
        evidence.triple_targets[midpoint:],
        evidence.has_set[midpoint:],
        exact_ensemble_testing,
        exact_ensemble_threshold,
    )
    contextual.report_selected_blend(
        "exact_typed_blend",
        exact_ensemble_training,
        exact_ensemble_testing,
        pipeline.training_scores,
        pipeline.testing_scores,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )
    soft_scores = differentiable_set_scores(corrected_logits)
    contextual.report_selected_blend(
        "soft_typed_blend",
        soft_scores[:midpoint],
        soft_scores[midpoint:],
        pipeline.training_scores,
        pipeline.testing_scores,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )
    np.savez_compressed(
        Path(sys.argv[1]).parent / "board-existence-probe.npz",
        triple_targets=evidence.triple_targets,
        has_set=evidence.has_set,
        typed_scores=np.concatenate(
            (pipeline.training_scores, pipeline.testing_scores)
        ),
        soft_scores=soft_scores,
        exact_scores=np.stack(
            [
                np.concatenate((training, testing))
                for training, testing in zip(
                    exact_training_scores, exact_testing_scores, strict=True
                )
            ]
        ),
    )
    all_scores = np.concatenate((pipeline.training_scores, pipeline.testing_scores))
    all_relations = np.concatenate(
        (pipeline.training_relations, pipeline.testing_relations)
    ).reshape(-1, contextual.TRIPLES.shape[0], 4, 3)
    board_features = invariant_board_features(
        all_scores, all_relations, corrected_logits
    )

    fit_features = board_features[:calibration_start]
    calibration_features = board_features[calibration_start:midpoint]
    testing_features = board_features[midpoint:]
    calibration_scores: list[FloatArray] = []
    testing_scores: list[FloatArray] = []
    for replica in range(8):
        model, mean, scale = fit_board_mlp(
            fit_features, evidence.has_set[:calibration_start], 70_001 + replica
        )
        calibration_scores.append(
            predict_board_mlp(calibration_features, model, mean, scale)
        )
        testing_scores.append(predict_board_mlp(testing_features, model, mean, scale))
        threshold, calibration_accuracy = best_board_threshold(
            calibration_scores[-1],
            pipeline.training_scores[calibration_start:],
            evidence.triple_targets[calibration_start:midpoint],
            evidence.has_set[calibration_start:midpoint],
        )
        overall, has_set_accuracy, no_set_accuracy, failures = action_accuracy(
            testing_scores[-1],
            threshold,
            pipeline.testing_scores,
            evidence.triple_targets[midpoint:],
            evidence.has_set[midpoint:],
        )
        print(
            f"board_mlp_{replica + 1}: calibration={calibration_accuracy:.6f} "
            f"overall={overall:.6f} set={has_set_accuracy:.6f} "
            f"no_set={no_set_accuracy:.6f} failures={failures}",
            flush=True,
        )

    for count in (2, 4, 8):
        calibration_ensemble = np.mean(calibration_scores[:count], axis=0)
        testing_ensemble = np.mean(testing_scores[:count], axis=0)
        threshold, calibration_accuracy = best_board_threshold(
            calibration_ensemble,
            pipeline.training_scores[calibration_start:],
            evidence.triple_targets[calibration_start:midpoint],
            evidence.has_set[calibration_start:midpoint],
        )
        overall, has_set_accuracy, no_set_accuracy, failures = action_accuracy(
            testing_ensemble,
            threshold,
            pipeline.testing_scores,
            evidence.triple_targets[midpoint:],
            evidence.has_set[midpoint:],
        )
        print(
            f"board_mlp_ensemble_{count}: calibration={calibration_accuracy:.6f} "
            f"overall={overall:.6f} set={has_set_accuracy:.6f} "
            f"no_set={no_set_accuracy:.6f} failures={failures}",
            flush=True,
        )


if __name__ == "__main__":
    main()
