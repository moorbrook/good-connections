# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.2",
# ]
# ///

"""Probe learned triple readers over frozen typed-relation evidence.

This is an architecture-development tool, not deployed inference. It trains on
the first half of a deterministic mixed-board corpus and reports raw SET/NoSet
action accuracy on the untouched second half.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Evidence:
    features: FloatArray
    targets: BoolArray
    has_set: BoolArray


def parse_vector(value: str) -> FloatArray:
    return np.fromstring(value, dtype=np.float32, sep=",")


def read_evidence(path: Path) -> Evidence:
    feature_rows: list[FloatArray] = []
    target_rows: list[BoolArray] = []
    kinds: list[bool] = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            relation_classes = int(row["relation_class_count"])
            if relation_classes != 3:
                raise ValueError(f"expected three typed relation classes, got {relation_classes}")
            scores = parse_vector(row["triple_scores"])
            relations = parse_vector(row["relation_logits"])
            targets = parse_vector(row["triple_targets"]) >= 0.5
            triple_count = scores.size
            expected_relations = triple_count * 4 * relation_classes
            if relations.size != expected_relations or targets.size != triple_count:
                raise ValueError("inconsistent triple evidence row")
            relations = relations.reshape(triple_count, 4, relation_classes)
            relations -= relations.max(axis=2, keepdims=True)
            probabilities = np.exp(relations)
            probabilities /= probabilities.sum(axis=2, keepdims=True)
            features = np.concatenate(
                (scores[:, None], probabilities.reshape(triple_count, 12)), axis=1
            )
            feature_rows.append(features.astype(np.float32, copy=False))
            target_rows.append(targets)
            kinds.append(row["expected_kind"] == "HasSet")
    features = np.stack(feature_rows)
    targets = np.stack(target_rows)
    has_set = np.asarray(kinds, dtype=np.bool_)
    return Evidence(features=features, targets=targets, has_set=has_set)


def sigmoid(values: FloatArray) -> FloatArray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))).astype(np.float32)


class Adam:
    def __init__(self, arrays: list[FloatArray], learning_rate: float) -> None:
        self.arrays = arrays
        self.m = [np.zeros_like(array) for array in arrays]
        self.v = [np.zeros_like(array) for array in arrays]
        self.learning_rate = learning_rate
        self.step = 0

    def update(self, gradients: list[FloatArray]) -> None:
        self.step += 1
        correction1 = 1.0 - 0.9**self.step
        correction2 = 1.0 - 0.999**self.step
        for array, gradient, mean, variance in zip(
            self.arrays, gradients, self.m, self.v, strict=True
        ):
            mean *= 0.9
            mean += 0.1 * gradient
            variance *= 0.999
            variance += 0.001 * gradient * gradient
            array -= (
                self.learning_rate
                * (mean / correction1)
                / (np.sqrt(variance / correction2) + 1.0e-8)
            )


def train_reader(
    features: FloatArray,
    targets: BoolArray,
    hidden_width: int,
    seed: int,
    steps: int = 1_500,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_targets = targets.reshape(-1)
    positive = np.flatnonzero(flat_targets)
    negative = np.flatnonzero(~flat_targets)
    if positive.size == 0:
        raise ValueError("training split contains no positive triples")
    input_width = flat_features.shape[1]
    first = (rng.standard_normal((input_width, hidden_width)) * 0.08).astype(np.float32)
    first_bias = np.zeros(hidden_width, dtype=np.float32)
    second = (rng.standard_normal(hidden_width) * 0.08).astype(np.float32)
    second_bias = np.zeros(1, dtype=np.float32)
    optimizer = Adam([first, first_bias, second, second_bias], learning_rate=0.008)
    half_batch = 2_048
    for _ in range(steps):
        indices = np.concatenate(
            (
                rng.choice(positive, half_batch, replace=True),
                rng.choice(negative, half_batch, replace=False),
            )
        )
        rng.shuffle(indices)
        batch = flat_features[indices]
        expected = flat_targets[indices].astype(np.float32)
        preactivation = batch @ first + first_bias
        hidden = np.tanh(preactivation)
        logits = hidden @ second + second_bias[0]
        error = (sigmoid(logits) - expected) / expected.size
        second_gradient = hidden.T @ error + 2.0e-5 * second
        second_bias_gradient = np.asarray([error.sum()], dtype=np.float32)
        hidden_gradient = error[:, None] * second[None, :]
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 2.0e-5 * first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
    return first, first_bias, second, second_bias


def reader_scores(
    features: FloatArray,
    model: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
) -> FloatArray:
    first, first_bias, second, second_bias = model
    flat = features.reshape(-1, features.shape[-1])
    logits = np.tanh(flat @ first + first_bias) @ second + second_bias[0]
    return logits.reshape(features.shape[:2])


def stable_logmeanexp(values: FloatArray, temperature: float) -> FloatArray:
    scaled = values / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    return (
        temperature
        * (
            maximum[:, 0]
            + np.log(np.exp(scaled - maximum).mean(axis=1))
        )
    ).astype(np.float32)


def train_mil_reader(
    features: FloatArray,
    targets: BoolArray,
    has_set: BoolArray,
    hidden_width: int,
    temperature: float,
    seed: int,
    steps: int = 2_000,
    triple_auxiliary_weight: float = 0.05,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    input_width = features.shape[-1]
    first = (rng.standard_normal((input_width, hidden_width)) * 0.08).astype(np.float32)
    first_bias = np.zeros(hidden_width, dtype=np.float32)
    second = (rng.standard_normal(hidden_width) * 0.08).astype(np.float32)
    board_bias = np.zeros(1, dtype=np.float32)
    optimizer = Adam([first, first_bias, second, board_bias], learning_rate=0.004)
    batch_size = 64
    triple_count = targets.shape[1]
    positive_weight = float(np.count_nonzero(~targets)) / float(np.count_nonzero(targets))
    for _ in range(steps):
        indices = rng.choice(features.shape[0], batch_size, replace=False)
        batch = features[indices]
        expected_triples = targets[indices].astype(np.float32)
        expected_boards = has_set[indices].astype(np.float32)
        flat_batch = batch.reshape(-1, input_width)
        preactivation = flat_batch @ first + first_bias
        hidden = np.tanh(preactivation)
        triple_logits = (hidden @ second).reshape(batch_size, triple_count)

        scaled = triple_logits / temperature
        maximum = scaled.max(axis=1, keepdims=True)
        exponentials = np.exp(scaled - maximum)
        pooling_weights = exponentials / exponentials.sum(axis=1, keepdims=True)
        board_logits = (
            temperature
            * (
                maximum[:, 0]
                + np.log(exponentials.mean(axis=1))
            )
            + board_bias[0]
        )
        board_error = (sigmoid(board_logits) - expected_boards) / batch_size
        triple_gradient = board_error[:, None] * pooling_weights

        triple_probabilities = sigmoid(triple_logits)
        local_weights = np.where(expected_triples > 0.5, positive_weight, 1.0)
        local_gradient = (triple_probabilities - expected_triples) * local_weights
        local_gradient /= local_weights.sum()
        triple_gradient += triple_auxiliary_weight * local_gradient

        flat_triple_gradient = triple_gradient.reshape(-1)
        second_gradient = hidden.T @ flat_triple_gradient + 2.0e-5 * second
        hidden_gradient = flat_triple_gradient[:, None] * second[None, :]
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = flat_batch.T @ preactivation_gradient + 2.0e-5 * first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        board_bias_gradient = np.asarray([board_error.sum()], dtype=np.float32)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, board_bias_gradient]
        )
    return first, first_bias, second, board_bias


def mil_scores(
    features: FloatArray,
    model: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    temperature: float,
) -> tuple[FloatArray, FloatArray]:
    first, first_bias, second, board_bias = model
    flat = features.reshape(-1, features.shape[-1])
    triple_scores = (
        np.tanh(flat @ first + first_bias) @ second
    ).reshape(features.shape[:2])
    board_scores = stable_logmeanexp(triple_scores, temperature) + board_bias[0]
    return triple_scores, board_scores


def action_correct(
    targets: BoolArray, has_set: BoolArray, scores: FloatArray, threshold: float
) -> BoolArray:
    best = scores.argmax(axis=1)
    best_scores = scores[np.arange(scores.shape[0]), best]
    chosen_is_set = targets[np.arange(targets.shape[0]), best]
    predict_set = best_scores >= threshold
    return np.where(predict_set, has_set & chosen_is_set, ~has_set)


def action_correct_with_board_scores(
    targets: BoolArray,
    has_set: BoolArray,
    triple_scores: FloatArray,
    board_scores: FloatArray,
    threshold: float,
) -> BoolArray:
    best = triple_scores.argmax(axis=1)
    chosen_is_set = targets[np.arange(targets.shape[0]), best]
    predict_set = board_scores >= threshold
    return np.where(predict_set, has_set & chosen_is_set, ~has_set)


def best_threshold(targets: BoolArray, has_set: BoolArray, scores: FloatArray) -> float:
    best = scores.argmax(axis=1)
    board_scores = scores[np.arange(scores.shape[0]), best]
    candidates = np.concatenate(
        (
            np.asarray([np.nextafter(board_scores.min(), -np.inf)], dtype=np.float32),
            np.sort(np.unique(board_scores)),
            np.asarray([np.nextafter(board_scores.max(), np.inf)], dtype=np.float32),
        )
    )
    accuracies = np.asarray(
        [action_correct(targets, has_set, scores, float(value)).mean() for value in candidates]
    )
    return float(candidates[int(accuracies.argmax())])


def report(
    name: str,
    targets: BoolArray,
    has_set: BoolArray,
    scores: FloatArray,
    threshold: float,
) -> None:
    correct = action_correct(targets, has_set, scores, threshold)
    print(
        f"{name}: overall={correct.mean():.6f} "
        f"set={correct[has_set].mean():.6f} "
        f"no_set={correct[~has_set].mean():.6f} "
        f"threshold={threshold:.9f} failures={np.count_nonzero(~correct)}"
    )


def best_board_threshold(
    targets: BoolArray,
    has_set: BoolArray,
    triple_scores: FloatArray,
    board_scores: FloatArray,
) -> float:
    candidates = np.concatenate(
        (
            np.asarray([np.nextafter(board_scores.min(), -np.inf)], dtype=np.float32),
            np.sort(np.unique(board_scores)),
            np.asarray([np.nextafter(board_scores.max(), np.inf)], dtype=np.float32),
        )
    )
    accuracies = np.asarray(
        [
            action_correct_with_board_scores(
                targets, has_set, triple_scores, board_scores, float(value)
            ).mean()
            for value in candidates
        ]
    )
    return float(candidates[int(accuracies.argmax())])


def report_mil(
    name: str,
    targets: BoolArray,
    has_set: BoolArray,
    triple_scores: FloatArray,
    board_scores: FloatArray,
    threshold: float,
) -> None:
    correct = action_correct_with_board_scores(
        targets, has_set, triple_scores, board_scores, threshold
    )
    print(
        f"{name}: overall={correct.mean():.6f} "
        f"set={correct[has_set].mean():.6f} "
        f"no_set={correct[~has_set].mean():.6f} "
        f"threshold={threshold:.9f} failures={np.count_nonzero(~correct)}"
    )


def pooled_board_features(features: FloatArray) -> FloatArray:
    invalid_probabilities = features[:, :, [3, 6, 9, 12]]
    valid_probabilities = 1.0 - invalid_probabilities
    minimum_validity = valid_probabilities.min(axis=2, keepdims=True)
    product_validity = valid_probabilities.prod(axis=2, keepdims=True)
    log_validity = np.log(np.maximum(valid_probabilities, 1.0e-7)).sum(
        axis=2, keepdims=True
    )
    expanded = np.concatenate(
        (features, valid_probabilities, minimum_validity, product_validity, log_validity),
        axis=2,
    )
    quantiles = np.quantile(
        expanded,
        (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0),
        axis=1,
    ).transpose(1, 0, 2)
    summaries = np.concatenate(
        (
            quantiles.reshape(features.shape[0], -1),
            expanded.mean(axis=1),
            expanded.std(axis=1),
        ),
        axis=1,
    )
    ranked = np.sort(expanded[:, :, [0, 17, 18]], axis=1)[:, -20:, :]
    return np.concatenate((summaries, ranked.reshape(features.shape[0], -1)), axis=1).astype(
        np.float32
    )


def train_board_reader(
    features: FloatArray,
    has_set: BoolArray,
    hidden_width: int,
    seed: int,
    steps: int = 3_000,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    first = (rng.standard_normal((features.shape[1], hidden_width)) * 0.04).astype(np.float32)
    first_bias = np.zeros(hidden_width, dtype=np.float32)
    second = (rng.standard_normal(hidden_width) * 0.04).astype(np.float32)
    second_bias = np.zeros(1, dtype=np.float32)
    optimizer = Adam([first, first_bias, second, second_bias], learning_rate=0.003)
    for _ in range(steps):
        indices = rng.choice(features.shape[0], 256, replace=False)
        batch = features[indices]
        expected = has_set[indices].astype(np.float32)
        preactivation = batch @ first + first_bias
        hidden = np.tanh(preactivation)
        logits = hidden @ second + second_bias[0]
        error = (sigmoid(logits) - expected) / expected.size
        second_gradient = hidden.T @ error + 5.0e-5 * second
        second_bias_gradient = np.asarray([error.sum()], dtype=np.float32)
        hidden_gradient = error[:, None] * second[None, :]
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 5.0e-5 * first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
    return first, first_bias, second, second_bias


def board_reader_scores(
    features: FloatArray,
    model: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
) -> FloatArray:
    first, first_bias, second, second_bias = model
    return (np.tanh(features @ first + first_bias) @ second + second_bias[0]).astype(
        np.float32
    )


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: uv run vision/analyze_relation_reader.py BOARD_RELATION_EVIDENCE.tsv [DIVERSE_EVIDENCE.tsv]"
        )
    evidence = read_evidence(Path(sys.argv[1]))
    midpoint = evidence.features.shape[0] // 2
    training_features = evidence.features[:midpoint]
    testing_features = evidence.features[midpoint:]
    training_targets = evidence.targets[:midpoint]
    testing_targets = evidence.targets[midpoint:]
    training_kinds = evidence.has_set[:midpoint]
    testing_kinds = evidence.has_set[midpoint:]

    mean = training_features.reshape(-1, training_features.shape[-1]).mean(axis=0)
    scale = training_features.reshape(-1, training_features.shape[-1]).std(axis=0)
    scale = np.maximum(scale, 1.0e-5)
    training_features = (training_features - mean) / scale
    testing_features = (testing_features - mean) / scale

    print(
        f"boards: train={midpoint} test={evidence.features.shape[0] - midpoint} "
        f"triples_per_board={evidence.features.shape[1]} "
        f"train_positive_triples={np.count_nonzero(training_targets)}"
    )
    raw_training = evidence.features[:midpoint, :, 0]
    raw_testing = evidence.features[midpoint:, :, 0]
    report("raw_test", testing_targets, testing_kinds, raw_testing, 0.0)
    test_best = raw_testing.argmax(axis=1)
    test_selection_correct = testing_targets[np.arange(testing_targets.shape[0]), test_best]
    selection_ceiling = np.where(testing_kinds, test_selection_correct, True).mean()
    print(f"raw_selector_oracle_board_ceiling={selection_ceiling:.6f}")

    if len(sys.argv) == 3:
        diverse = read_evidence(Path(sys.argv[2]))
        if (
            not np.array_equal(evidence.targets, diverse.targets)
            or not np.array_equal(evidence.has_set, diverse.has_set)
            or evidence.features.shape != diverse.features.shape
        ):
            raise ValueError("diverse evidence corpus does not match the primary board sequence")
        candidates: list[tuple[float, float, float]] = []
        for primary_weight in np.linspace(0.0, 1.0, 21):
            blended_scores = (
                primary_weight * evidence.features[:, :, 0]
                + (1.0 - primary_weight) * diverse.features[:, :, 0]
            )
            blended_training = blended_scores[:midpoint]
            blended_testing = blended_scores[midpoint:]
            threshold = best_threshold(
                training_targets, training_kinds, blended_training
            )
            training_accuracy = action_correct(
                training_targets, training_kinds, blended_training, threshold
            ).mean()
            testing_accuracy = action_correct(
                testing_targets, testing_kinds, blended_testing, threshold
            ).mean()
            candidates.append((float(training_accuracy), float(primary_weight), threshold))
            print(
                f"blend_candidate: primary_weight={primary_weight:.2f} "
                f"train={training_accuracy:.6f} test={testing_accuracy:.6f} "
                f"threshold={threshold:.9f}"
            )
        _, selected_weight, selected_threshold = max(candidates)
        selected_testing = (
            selected_weight * evidence.features[midpoint:, :, 0]
            + (1.0 - selected_weight) * diverse.features[midpoint:, :, 0]
        )
        report(
            "selected_diverse_blend_test",
            testing_targets,
            testing_kinds,
            selected_testing,
            selected_threshold,
        )

    for hidden_width, seed in ((8, 9108), (16, 9116), (32, 9132)):
        model = train_reader(
            training_features, training_targets, hidden_width=hidden_width, seed=seed
        )
        training_scores = reader_scores(training_features, model)
        testing_scores = reader_scores(testing_features, model)
        threshold = best_threshold(
            training_targets, training_kinds, training_scores
        )
        report(
            f"relation_mlp_{hidden_width}_train",
            training_targets,
            training_kinds,
            training_scores,
            threshold,
        )
        report(
            f"relation_mlp_{hidden_width}_test",
            testing_targets,
            testing_kinds,
            testing_scores,
            threshold,
        )

    for hidden_width, temperature, seed in (
        (8, 0.25, 9208),
        (16, 0.5, 9216),
        (16, 1.0, 9316),
        (32, 2.0, 9232),
    ):
        model = train_mil_reader(
            training_features,
            training_targets,
            training_kinds,
            hidden_width=hidden_width,
            temperature=temperature,
            seed=seed,
        )
        training_triples, training_boards = mil_scores(
            training_features, model, temperature
        )
        testing_triples, testing_boards = mil_scores(
            testing_features, model, temperature
        )
        threshold = best_board_threshold(
            training_targets,
            training_kinds,
            training_triples,
            training_boards,
        )
        suffix = f"{hidden_width}_t{temperature:g}"
        report_mil(
            f"mil_{suffix}_train",
            training_targets,
            training_kinds,
            training_triples,
            training_boards,
            threshold,
        )
        report_mil(
            f"mil_{suffix}_test",
            testing_targets,
            testing_kinds,
            testing_triples,
            testing_boards,
            threshold,
        )

    pooled = pooled_board_features(evidence.features)
    pooled_training = pooled[:midpoint]
    pooled_testing = pooled[midpoint:]
    pooled_mean = pooled_training.mean(axis=0)
    pooled_scale = np.maximum(pooled_training.std(axis=0), 1.0e-5)
    pooled_training = (pooled_training - pooled_mean) / pooled_scale
    pooled_testing = (pooled_testing - pooled_mean) / pooled_scale
    for hidden_width, seed in ((8, 9408), (16, 9416), (32, 9432)):
        model = train_board_reader(
            pooled_training, training_kinds, hidden_width=hidden_width, seed=seed
        )
        training_boards = board_reader_scores(pooled_training, model)
        testing_boards = board_reader_scores(pooled_testing, model)
        threshold = best_board_threshold(
            training_targets,
            training_kinds,
            raw_training,
            training_boards,
        )
        report_mil(
            f"pooled_mlp_{hidden_width}_train",
            training_targets,
            training_kinds,
            raw_training,
            training_boards,
            threshold,
        )
        report_mil(
            f"pooled_mlp_{hidden_width}_test",
            testing_targets,
            testing_kinds,
            raw_testing,
            testing_boards,
            threshold,
        )


if __name__ == "__main__":
    main()
