# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Test whether existing board summaries support a learned No SET head."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    features: tuple[float, ...]
    has_set: bool
    strongest_is_set: bool


def read_examples(path: Path) -> list[Example]:
    with path.open(newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        return [
            Example(
                features=(
                    float(row["score"]),
                    float(row["score_margin"]),
                    float(row["mean_attribute_entropy"]),
                    float(row["min_attribute_margin"]),
                    float(row["third_score"]),
                    float(row["fifth_score"]),
                    float(row["tenth_score"]),
                    float(row["score_mean"]),
                    float(row["score_standard_deviation"]),
                    float(row["score_logsumexp"]),
                    float(row["nonnegative_count"]),
                    float(row["above_minus_one_count"]),
                ),
                has_set=row["expected_kind"] == "HasSet",
                strongest_is_set=row["strongest_is_set"] == "1",
            )
            for row in rows
        ]


def standardize(
    training: list[Example], examples: list[Example]
) -> tuple[list[list[float]], list[float], list[float]]:
    width = len(training[0].features)
    means = [sum(row.features[i] for row in training) / len(training) for i in range(width)]
    scales = []
    for i, mean in enumerate(means):
        variance = sum((row.features[i] - mean) ** 2 for row in training) / len(training)
        scales.append(max(math.sqrt(variance), 1.0e-9))
    transformed = [
        [1.0]
        + [
            (example.features[i] - means[i]) / scales[i]
            for i in range(width)
        ]
        for example in examples
    ]
    return transformed, means, scales


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def train_logistic(features: list[list[float]], examples: list[Example]) -> list[float]:
    weights = [0.0] * len(features[0])
    learning_rate = 0.08
    regularization = 1.0e-3
    for _ in range(5_000):
        gradients = [0.0] * len(weights)
        for row, example in zip(features, examples, strict=True):
            probability = sigmoid(sum(weight * value for weight, value in zip(weights, row, strict=True)))
            error = probability - float(example.has_set)
            for index, value in enumerate(row):
                gradients[index] += error * value
        for index in range(len(weights)):
            penalty = 0.0 if index == 0 else regularization * weights[index]
            weights[index] -= learning_rate * (gradients[index] / len(examples) + penalty)
    return weights


def probabilities(features: list[list[float]], weights: list[float]) -> list[float]:
    return [
        sigmoid(sum(weight * value for weight, value in zip(weights, row, strict=True)))
        for row in features
    ]


def train_mlp(
    features: list[list[float]], examples: list[Example], hidden_width: int = 8
) -> tuple[list[list[float]], list[float], list[float], float]:
    input_width = len(features[0])
    first = [
        [0.08 * math.sin((hidden + 1) * (column + 1)) for column in range(input_width)]
        for hidden in range(hidden_width)
    ]
    first_bias = [0.0] * hidden_width
    second = [0.08 * math.sin(hidden + 1) for hidden in range(hidden_width)]
    second_bias = 0.0
    learning_rate = 0.12
    regularization = 5.0e-4
    for _ in range(1_200):
        first_gradient = [[0.0] * input_width for _ in range(hidden_width)]
        first_bias_gradient = [0.0] * hidden_width
        second_gradient = [0.0] * hidden_width
        second_bias_gradient = 0.0
        for row, example in zip(features, examples, strict=True):
            hidden_values = [
                math.tanh(
                    sum(weight * value for weight, value in zip(weights, row, strict=True))
                    + bias
                )
                for weights, bias in zip(first, first_bias, strict=True)
            ]
            probability = sigmoid(
                sum(weight * value for weight, value in zip(second, hidden_values, strict=True))
                + second_bias
            )
            output_error = probability - float(example.has_set)
            second_bias_gradient += output_error
            for hidden, hidden_value in enumerate(hidden_values):
                second_gradient[hidden] += output_error * hidden_value
                hidden_error = output_error * second[hidden] * (1.0 - hidden_value**2)
                first_bias_gradient[hidden] += hidden_error
                for column, value in enumerate(row):
                    first_gradient[hidden][column] += hidden_error * value
        inverse_count = 1.0 / len(examples)
        second_bias -= learning_rate * second_bias_gradient * inverse_count
        for hidden in range(hidden_width):
            second[hidden] -= learning_rate * (
                second_gradient[hidden] * inverse_count + regularization * second[hidden]
            )
            first_bias[hidden] -= learning_rate * first_bias_gradient[hidden] * inverse_count
            for column in range(input_width):
                first[hidden][column] -= learning_rate * (
                    first_gradient[hidden][column] * inverse_count
                    + regularization * first[hidden][column]
                )
    return first, first_bias, second, second_bias


def mlp_probabilities(
    features: list[list[float]],
    model: tuple[list[list[float]], list[float], list[float], float],
) -> list[float]:
    first, first_bias, second, second_bias = model
    output = []
    for row in features:
        hidden_values = [
            math.tanh(
                sum(weight * value for weight, value in zip(weights, row, strict=True)) + bias
            )
            for weights, bias in zip(first, first_bias, strict=True)
        ]
        output.append(
            sigmoid(
                sum(weight * value for weight, value in zip(second, hidden_values, strict=True))
                + second_bias
            )
        )
    return output


def action_correct(example: Example, predict_set: bool) -> bool:
    if predict_set:
        return example.has_set and example.strongest_is_set
    return not example.has_set


def best_threshold(examples: list[Example], scores: list[float]) -> float:
    candidates = sorted(set(scores))
    return max(
        candidates,
        key=lambda threshold: sum(
            action_correct(example, score >= threshold)
            for example, score in zip(examples, scores, strict=True)
        ),
    )


def report(name: str, examples: list[Example], scores: list[float], threshold: float) -> None:
    decisions = [score >= threshold for score in scores]
    correct = sum(
        action_correct(example, decision)
        for example, decision in zip(examples, decisions, strict=True)
    )
    set_rows = [(example, decision) for example, decision in zip(examples, decisions, strict=True) if example.has_set]
    no_set_rows = [(example, decision) for example, decision in zip(examples, decisions, strict=True) if not example.has_set]
    set_correct = sum(action_correct(example, decision) for example, decision in set_rows)
    no_set_correct = sum(action_correct(example, decision) for example, decision in no_set_rows)
    print(
        f"{name}: overall={correct / len(examples):.6f} "
        f"set={set_correct / len(set_rows):.6f} "
        f"no_set={no_set_correct / len(no_set_rows):.6f} threshold={threshold:.9f}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run vision/analyze_board_head.py SAMPLES.tsv")
    examples = read_examples(Path(sys.argv[1]))
    midpoint = len(examples) // 2
    training = examples[:midpoint]
    testing = examples[midpoint:]
    all_features, _, _ = standardize(training, examples)
    training_features = all_features[:midpoint]
    testing_features = all_features[midpoint:]
    weights = train_logistic(training_features, training)
    training_probabilities = probabilities(training_features, weights)
    testing_probabilities = probabilities(testing_features, weights)
    threshold = best_threshold(training, training_probabilities)

    report("raw_test", testing, [example.features[0] for example in testing], 0.0)
    report("learned_train", training, training_probabilities, threshold)
    report("learned_test", testing, testing_probabilities, threshold)
    print("weights:", ",".join(f"{weight:.9f}" for weight in weights))

    mlp = train_mlp(training_features, training)
    mlp_training_probabilities = mlp_probabilities(training_features, mlp)
    mlp_testing_probabilities = mlp_probabilities(testing_features, mlp)
    mlp_threshold = best_threshold(training, mlp_training_probabilities)
    report("mlp_train", training, mlp_training_probabilities, mlp_threshold)
    report("mlp_test", testing, mlp_testing_probabilities, mlp_threshold)


if __name__ == "__main__":
    main()
