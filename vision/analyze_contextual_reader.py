# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.2",
# ]
# ///

"""Probe a learned permutation-equivariant 12-card context reader.

The first half of an exported deterministic board stream is development data.
Within it, 75% trains the card correctors and 25% selects whether each
specialist correction is retained. The second half is untouched evaluation.
All inference remains continuous: card context -> typed relation MLP -> SET
MLP. Procedural card and relation classes are used only as training targets.
"""

from __future__ import annotations

import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
TRIPLES = np.asarray(list(itertools.combinations(range(12), 3)), dtype=np.int64)


@dataclass(frozen=True)
class Evidence:
    triple_scores: FloatArray
    relation_logits: FloatArray
    triple_targets: BoolArray
    attribute_logits: FloatArray
    card_targets: IntArray
    has_set: BoolArray


@dataclass(frozen=True)
class PipelineResult:
    training_scores: FloatArray
    testing_scores: FloatArray
    training_relations: FloatArray
    testing_relations: FloatArray


@dataclass
class Mlp:
    first: FloatArray
    first_bias: FloatArray
    second: FloatArray
    second_bias: FloatArray

    def arrays(self) -> list[FloatArray]:
        return [self.first, self.first_bias, self.second, self.second_bias]

    def copy(self) -> Mlp:
        return Mlp(*(array.copy() for array in self.arrays()))


@dataclass(frozen=True)
class Corrector:
    model: Mlp
    mean: FloatArray
    scale: FloatArray
    enabled: bool


@dataclass(frozen=True)
class HypergraphCorrector:
    model: Mlp
    mean: FloatArray
    scale: FloatArray
    enabled_attributes: BoolArray


class Adam:
    def __init__(self, arrays: list[FloatArray], learning_rate: float) -> None:
        self.arrays = arrays
        self.first_moments = [np.zeros_like(array) for array in arrays]
        self.second_moments = [np.zeros_like(array) for array in arrays]
        self.learning_rate = learning_rate
        self.step = 0

    def update(self, gradients: list[FloatArray]) -> None:
        self.step += 1
        first_correction = 1.0 - 0.9**self.step
        second_correction = 1.0 - 0.999**self.step
        for array, gradient, first, second in zip(
            self.arrays,
            gradients,
            self.first_moments,
            self.second_moments,
            strict=True,
        ):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient * gradient
            array -= (
                self.learning_rate
                * (first / first_correction)
                / (np.sqrt(second / second_correction) + 1.0e-8)
            )


def parse_float_vector(value: str) -> FloatArray:
    return np.fromstring(value, dtype=np.float32, sep=",")


def read_evidence(path: Path) -> Evidence:
    scores: list[FloatArray] = []
    relation_logits: list[FloatArray] = []
    triple_targets: list[BoolArray] = []
    attribute_logits: list[FloatArray] = []
    card_targets: list[IntArray] = []
    kinds: list[bool] = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            row_scores = parse_float_vector(row["triple_scores"])
            row_triples = parse_float_vector(row["triple_targets"]) >= 0.5
            row_relations = parse_float_vector(row["relation_logits"])
            row_logits = parse_float_vector(row["attribute_logits"])
            row_cards = parse_float_vector(row["card_targets"]).astype(np.int64)
            if (
                row_scores.size != TRIPLES.shape[0]
                or row_triples.size != TRIPLES.shape[0]
                or row_relations.size != TRIPLES.shape[0] * 4 * 3
                or row_logits.size != 12 * 4 * 3
                or row_cards.size != 12 * 4
            ):
                raise ValueError("inconsistent contextual evidence row")
            scores.append(row_scores)
            triple_targets.append(row_triples)
            relation_logits.append(row_relations.reshape(TRIPLES.shape[0], 4, 3))
            attribute_logits.append(row_logits.reshape(12, 4, 3))
            card_targets.append(row_cards.reshape(12, 4))
            kinds.append(row["expected_kind"] == "HasSet")
    return Evidence(
        triple_scores=np.stack(scores),
        relation_logits=np.stack(relation_logits),
        triple_targets=np.stack(triple_targets),
        attribute_logits=np.stack(attribute_logits),
        card_targets=np.stack(card_targets),
        has_set=np.asarray(kinds, dtype=np.bool_),
    )


def softmax(values: FloatArray, axis: int = -1) -> FloatArray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=axis, keepdims=True)).astype(
        np.float32
    )


def sigmoid(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def initialize_mlp(
    input_width: int, hidden_width: int, output_width: int, seed: int
) -> Mlp:
    rng = np.random.default_rng(seed)
    return Mlp(
        first=(rng.standard_normal((input_width, hidden_width)) * 0.05).astype(
            np.float32
        ),
        first_bias=np.zeros(hidden_width, dtype=np.float32),
        second=(rng.standard_normal((hidden_width, output_width)) * 0.05).astype(
            np.float32
        ),
        second_bias=np.zeros(output_width, dtype=np.float32),
    )


def mlp_logits(features: FloatArray, model: Mlp) -> FloatArray:
    return (
        np.tanh(features @ model.first + model.first_bias) @ model.second
        + model.second_bias
    )


def cross_entropy(logits: FloatArray, targets: IntArray) -> float:
    probabilities = softmax(logits)
    selected = probabilities[np.arange(targets.size), targets]
    return float(-np.log(np.maximum(selected, 1.0e-8)).mean())


def context_features(logits: FloatArray) -> tuple[FloatArray, FloatArray]:
    centered = logits - logits.mean(axis=-1, keepdims=True)
    probabilities = softmax(centered)
    board_sum = probabilities.sum(axis=1, keepdims=True)
    board_square_sum = (probabilities * probabilities).sum(axis=1, keepdims=True)
    other_mean = (board_sum - probabilities) / 11.0
    other_variance = np.maximum(
        (board_square_sum - probabilities * probabilities) / 11.0
        - other_mean * other_mean,
        0.0,
    )
    other_standard_deviation = np.sqrt(other_variance)
    features = np.concatenate(
        (centered, probabilities, other_mean, other_standard_deviation), axis=-1
    )
    return features.astype(np.float32), centered.astype(np.float32)


def train_corrector(
    training_logits: FloatArray,
    training_targets: IntArray,
    validation_logits: FloatArray,
    validation_targets: IntArray,
    seed: int,
) -> tuple[Corrector, float, float, float, float]:
    training_features, training_base = context_features(training_logits)
    validation_features, validation_base = context_features(validation_logits)
    flat_training = training_features.reshape(-1, training_features.shape[-1])
    flat_targets = training_targets.reshape(-1)
    mean = flat_training.mean(axis=0)
    scale = np.maximum(flat_training.std(axis=0), 1.0e-5)
    flat_training = (flat_training - mean) / scale
    flat_validation = (
        validation_features.reshape(-1, validation_features.shape[-1]) - mean
    ) / scale
    training_base = training_base.reshape(-1, 3)
    validation_base = validation_base.reshape(-1, 3)
    validation_targets = validation_targets.reshape(-1)

    model = initialize_mlp(flat_training.shape[1], 16, 3, seed)
    optimizer = Adam(model.arrays(), learning_rate=0.004)
    rng = np.random.default_rng(seed ^ 0x5E7)
    best_model = model.copy()
    best_loss = float("inf")
    for step in range(1, 1_501):
        indices = rng.choice(flat_training.shape[0], 2_048, replace=False)
        batch = flat_training[indices]
        expected = flat_targets[indices]
        hidden = np.tanh(batch @ model.first + model.first_bias)
        logits = training_base[indices] + hidden @ model.second + model.second_bias
        probabilities = softmax(logits)
        probabilities[np.arange(expected.size), expected] -= 1.0
        output_gradient = probabilities / expected.size
        second_gradient = hidden.T @ output_gradient + 2.0e-5 * model.second
        second_bias_gradient = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ model.second.T
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 2.0e-5 * model.first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
        if step % 100 == 0:
            corrected = validation_base + mlp_logits(flat_validation, model)
            loss = cross_entropy(corrected, validation_targets)
            if loss < best_loss:
                best_loss = loss
                best_model = model.copy()

    base_validation_loss = cross_entropy(validation_base, validation_targets)
    corrected_validation = validation_base + mlp_logits(flat_validation, best_model)
    base_accuracy = float((validation_base.argmax(axis=1) == validation_targets).mean())
    corrected_accuracy = float(
        (corrected_validation.argmax(axis=1) == validation_targets).mean()
    )
    enabled = best_loss < base_validation_loss
    return (
        Corrector(model=best_model, mean=mean, scale=scale, enabled=enabled),
        base_validation_loss,
        best_loss,
        base_accuracy,
        corrected_accuracy,
    )


def apply_correctors(logits: FloatArray, correctors: list[Corrector]) -> FloatArray:
    corrected = np.empty_like(logits)
    for attribute, corrector in enumerate(correctors):
        features, base = context_features(logits[:, :, attribute, :])
        flat_features = (
            features.reshape(-1, features.shape[-1]) - corrector.mean
        ) / corrector.scale
        flat_base = base.reshape(-1, 3)
        if corrector.enabled:
            flat_base = flat_base + mlp_logits(flat_features, corrector.model)
        corrected[:, :, attribute, :] = flat_base.reshape(base.shape)
    return corrected


def stacker_features_and_base(
    member_logits: list[FloatArray],
) -> tuple[FloatArray, FloatArray]:
    stacked = np.stack(member_logits, axis=2)
    centered = stacked - stacked.mean(axis=-1, keepdims=True)
    probabilities = softmax(centered)
    features = np.concatenate(
        (
            centered.reshape(*centered.shape[:2], -1),
            probabilities.reshape(*probabilities.shape[:2], -1),
            centered.mean(axis=2),
            centered.std(axis=2),
            probabilities.mean(axis=2),
            probabilities.std(axis=2),
        ),
        axis=2,
    )
    return features.astype(np.float32), centered.mean(axis=2).astype(np.float32)


def train_feature_corrector(
    training_features: FloatArray,
    training_base: FloatArray,
    training_targets: IntArray,
    validation_features: FloatArray,
    validation_base: FloatArray,
    validation_targets: IntArray,
    seed: int,
) -> tuple[Corrector, float, float, float, float]:
    flat_training = training_features.reshape(-1, training_features.shape[-1])
    flat_validation = validation_features.reshape(-1, validation_features.shape[-1])
    mean = flat_training.mean(axis=0)
    scale = np.maximum(flat_training.std(axis=0), 1.0e-5)
    flat_training = (flat_training - mean) / scale
    flat_validation = (flat_validation - mean) / scale
    flat_training_base = training_base.reshape(-1, 3)
    flat_validation_base = validation_base.reshape(-1, 3)
    flat_training_targets = training_targets.reshape(-1)
    flat_validation_targets = validation_targets.reshape(-1)

    model = initialize_mlp(flat_training.shape[1], 32, 3, seed)
    optimizer = Adam(model.arrays(), learning_rate=0.004)
    rng = np.random.default_rng(seed ^ 0x57AC)
    best_model = model.copy()
    best_loss = float("inf")
    for step in range(1, 1_501):
        indices = rng.choice(flat_training.shape[0], 2_048, replace=False)
        batch = flat_training[indices]
        expected = flat_training_targets[indices]
        hidden = np.tanh(batch @ model.first + model.first_bias)
        logits = flat_training_base[indices] + hidden @ model.second + model.second_bias
        probabilities = softmax(logits)
        probabilities[np.arange(expected.size), expected] -= 1.0
        output_gradient = probabilities / expected.size
        second_gradient = hidden.T @ output_gradient + 2.0e-5 * model.second
        second_bias_gradient = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ model.second.T
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 2.0e-5 * model.first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
        if step % 100 == 0:
            corrected = flat_validation_base + mlp_logits(flat_validation, model)
            loss = cross_entropy(corrected, flat_validation_targets)
            if loss < best_loss:
                best_loss = loss
                best_model = model.copy()

    base_loss = cross_entropy(flat_validation_base, flat_validation_targets)
    corrected_validation = flat_validation_base + mlp_logits(
        flat_validation, best_model
    )
    base_accuracy = float(
        (flat_validation_base.argmax(axis=1) == flat_validation_targets).mean()
    )
    corrected_accuracy = float(
        (corrected_validation.argmax(axis=1) == flat_validation_targets).mean()
    )
    return (
        Corrector(
            model=best_model,
            mean=mean,
            scale=scale,
            enabled=best_loss < base_loss,
        ),
        base_loss,
        best_loss,
        base_accuracy,
        corrected_accuracy,
    )


def apply_feature_corrector(
    features: FloatArray, base: FloatArray, corrector: Corrector
) -> FloatArray:
    flat_features = (
        features.reshape(-1, features.shape[-1]) - corrector.mean
    ) / corrector.scale
    if not corrector.enabled:
        return base
    corrections = mlp_logits(flat_features, corrector.model).reshape(base.shape)
    return (base + corrections).astype(np.float32)


def report_attributes(name: str, logits: FloatArray, targets: IntArray) -> None:
    predictions = logits.argmax(axis=-1)
    by_kind = [
        float((predictions[:, :, index] == targets[:, :, index]).mean())
        for index in range(4)
    ]
    print(
        f"{name}: overall={(predictions == targets).mean():.6f} "
        f"count={by_kind[0]:.6f} colour={by_kind[1]:.6f} "
        f"shape={by_kind[2]:.6f} fill={by_kind[3]:.6f}"
    )


def hypergraph_features(
    attribute_logits: FloatArray,
    triple_scores: FloatArray,
    relation_logits: FloatArray,
) -> FloatArray:
    per_attribute_context = [
        context_features(attribute_logits[:, :, attribute, :])[0]
        for attribute in range(4)
    ]
    card_context = np.concatenate(per_attribute_context, axis=2)
    relation_probabilities = softmax(relation_logits).reshape(
        relation_logits.shape[0], relation_logits.shape[1], 12
    )
    triple_features = np.concatenate(
        (triple_scores[:, :, None], relation_probabilities), axis=2
    )
    card_features: list[FloatArray] = []
    for card in range(12):
        incident = np.flatnonzero((TRIPLES == card).any(axis=1))
        evidence = triple_features[:, incident, :]
        order = np.argsort(-evidence[:, :, 0], axis=1)
        ranked = np.take_along_axis(evidence, order[:, :, None], axis=1)
        top = ranked[:, :8, :].reshape(evidence.shape[0], -1)
        summaries = np.concatenate(
            (
                evidence.mean(axis=1),
                evidence.std(axis=1),
                evidence.max(axis=1),
                evidence.min(axis=1),
            ),
            axis=1,
        )
        card_features.append(
            np.concatenate((card_context[:, card, :], top, summaries), axis=1)
        )
    return np.stack(card_features, axis=1).astype(np.float32)


def attribute_cross_entropy_by_kind(
    logits: FloatArray, targets: IntArray
) -> FloatArray:
    probabilities = softmax(logits, axis=-1)
    selected = np.take_along_axis(probabilities, targets[..., None], axis=-1)[..., 0]
    return -np.log(np.maximum(selected, 1.0e-8)).mean(axis=(0, 1))


def train_hypergraph_corrector(
    training_features: FloatArray,
    training_logits: FloatArray,
    training_targets: IntArray,
    validation_features: FloatArray,
    validation_logits: FloatArray,
    validation_targets: IntArray,
    seed: int,
) -> tuple[HypergraphCorrector, FloatArray, FloatArray]:
    flat_training_features = training_features.reshape(-1, training_features.shape[-1])
    flat_validation_features = validation_features.reshape(
        -1, validation_features.shape[-1]
    )
    mean = flat_training_features.mean(axis=0)
    scale = np.maximum(flat_training_features.std(axis=0), 1.0e-5)
    flat_training_features = (flat_training_features - mean) / scale
    flat_validation_features = (flat_validation_features - mean) / scale
    training_base = training_logits - training_logits.mean(axis=-1, keepdims=True)
    validation_base = validation_logits - validation_logits.mean(axis=-1, keepdims=True)
    flat_training_base = training_base.reshape(-1, 4, 3)
    flat_training_targets = training_targets.reshape(-1, 4)

    model = initialize_mlp(flat_training_features.shape[1], 64, 12, seed)
    optimizer = Adam(model.arrays(), learning_rate=0.003)
    rng = np.random.default_rng(seed ^ 0xC07E)
    best_model = model.copy()
    best_loss = float("inf")
    for step in range(1, 2_001):
        indices = rng.choice(flat_training_features.shape[0], 2_048, replace=False)
        batch = flat_training_features[indices]
        expected = flat_training_targets[indices]
        hidden = np.tanh(batch @ model.first + model.first_bias)
        corrections = (hidden @ model.second + model.second_bias).reshape(-1, 4, 3)
        logits = flat_training_base[indices] + corrections
        probabilities = softmax(logits, axis=-1)
        row_indices = np.arange(expected.shape[0])[:, None]
        attribute_indices = np.arange(4)[None, :]
        probabilities[row_indices, attribute_indices, expected] -= 1.0
        output_gradient = (probabilities / (expected.shape[0] * 4)).reshape(-1, 12)
        second_gradient = hidden.T @ output_gradient + 3.0e-5 * model.second
        second_bias_gradient = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ model.second.T
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 3.0e-5 * model.first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
        if step % 100 == 0:
            corrections = mlp_logits(flat_validation_features, model).reshape(
                -1, 12, 4, 3
            )
            corrected = validation_base + corrections
            loss = float(
                attribute_cross_entropy_by_kind(corrected, validation_targets).mean()
            )
            if loss < best_loss:
                best_loss = loss
                best_model = model.copy()

    base_losses = attribute_cross_entropy_by_kind(validation_base, validation_targets)
    best_corrections = mlp_logits(flat_validation_features, best_model).reshape(
        -1, 12, 4, 3
    )
    corrected_losses = attribute_cross_entropy_by_kind(
        validation_base + best_corrections, validation_targets
    )
    enabled = corrected_losses < base_losses
    return (
        HypergraphCorrector(
            model=best_model,
            mean=mean,
            scale=scale,
            enabled_attributes=enabled,
        ),
        base_losses,
        corrected_losses,
    )


def apply_hypergraph_corrector(
    attribute_logits: FloatArray,
    features: FloatArray,
    corrector: HypergraphCorrector,
) -> FloatArray:
    base = attribute_logits - attribute_logits.mean(axis=-1, keepdims=True)
    flat_features = (
        features.reshape(-1, features.shape[-1]) - corrector.mean
    ) / corrector.scale
    corrections = mlp_logits(flat_features, corrector.model).reshape(base.shape)
    enabled = corrector.enabled_attributes[None, None, :, None]
    return np.where(enabled, base + corrections, base).astype(np.float32)


def triple_attribute_sums(probabilities: FloatArray) -> FloatArray:
    return (
        probabilities[:, TRIPLES[:, 0], :, :]
        + probabilities[:, TRIPLES[:, 1], :, :]
        + probabilities[:, TRIPLES[:, 2], :, :]
    ).astype(np.float32)


def typed_relation_targets(card_targets: IntArray) -> IntArray:
    first = card_targets[:, TRIPLES[:, 0], :]
    second = card_targets[:, TRIPLES[:, 1], :]
    third = card_targets[:, TRIPLES[:, 2], :]
    same = (first == second) & (second == third)
    different = (first != second) & (first != third) & (second != third)
    return np.where(same, 0, np.where(different, 1, 2)).astype(np.int64)


def train_multiclass_mlp(
    features: FloatArray,
    targets: IntArray,
    hidden_width: int,
    output_width: int,
    seed: int,
    steps: int,
) -> tuple[Mlp, FloatArray, FloatArray]:
    flat_features = features.reshape(-1, features.shape[-1])
    flat_targets = targets.reshape(-1)
    mean = flat_features.mean(axis=0)
    scale = np.maximum(flat_features.std(axis=0), 1.0e-5)
    flat_features = (flat_features - mean) / scale
    class_indices = [
        np.flatnonzero(flat_targets == value) for value in range(output_width)
    ]
    model = initialize_mlp(flat_features.shape[1], hidden_width, output_width, seed)
    optimizer = Adam(model.arrays(), learning_rate=0.006)
    rng = np.random.default_rng(seed ^ 0xA11CE)
    per_class = 1_024
    for _ in range(steps):
        indices = np.concatenate(
            [rng.choice(values, per_class, replace=True) for values in class_indices]
        )
        rng.shuffle(indices)
        batch = flat_features[indices]
        expected = flat_targets[indices]
        hidden = np.tanh(batch @ model.first + model.first_bias)
        logits = hidden @ model.second + model.second_bias
        probabilities = softmax(logits)
        probabilities[np.arange(expected.size), expected] -= 1.0
        output_gradient = probabilities / expected.size
        second_gradient = hidden.T @ output_gradient + 2.0e-5 * model.second
        second_bias_gradient = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ model.second.T
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 2.0e-5 * model.first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
    return model, mean, scale


def predict_mlp(
    features: FloatArray, model: Mlp, mean: FloatArray, scale: FloatArray
) -> FloatArray:
    flat = features.reshape(-1, features.shape[-1])
    outputs = []
    for start in range(0, flat.shape[0], 131_072):
        batch = (flat[start : start + 131_072] - mean) / scale
        outputs.append(mlp_logits(batch, model))
    return np.concatenate(outputs).reshape(*features.shape[:-1], model.second.shape[1])


def train_binary_mlp(
    features: FloatArray,
    targets: BoolArray,
    hidden_width: int,
    seed: int,
    steps: int,
) -> tuple[Mlp, FloatArray, FloatArray]:
    flat_features = features.reshape(-1, features.shape[-1])
    flat_targets = targets.reshape(-1)
    mean = flat_features.mean(axis=0)
    scale = np.maximum(flat_features.std(axis=0), 1.0e-5)
    flat_features = (flat_features - mean) / scale
    positive = np.flatnonzero(flat_targets)
    negative = np.flatnonzero(~flat_targets)
    model = initialize_mlp(flat_features.shape[1], hidden_width, 1, seed)
    optimizer = Adam(model.arrays(), learning_rate=0.006)
    rng = np.random.default_rng(seed ^ 0xB04D)
    half_batch = 2_048
    for _ in range(steps):
        indices = np.concatenate(
            (
                rng.choice(positive, half_batch, replace=True),
                rng.choice(negative, half_batch, replace=negative.size < half_batch),
            )
        )
        rng.shuffle(indices)
        batch = flat_features[indices]
        expected = flat_targets[indices].astype(np.float32)
        hidden = np.tanh(batch @ model.first + model.first_bias)
        logits = (hidden @ model.second + model.second_bias)[:, 0]
        output_gradient = ((sigmoid(logits) - expected) / expected.size)[:, None]
        second_gradient = hidden.T @ output_gradient + 2.0e-5 * model.second
        second_bias_gradient = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ model.second.T
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        first_gradient = batch.T @ preactivation_gradient + 2.0e-5 * model.first
        first_bias_gradient = preactivation_gradient.sum(axis=0)
        optimizer.update(
            [first_gradient, first_bias_gradient, second_gradient, second_bias_gradient]
        )
    return model, mean, scale


def action_correct(
    triple_targets: BoolArray,
    has_set: BoolArray,
    scores: FloatArray,
    threshold: float,
) -> BoolArray:
    best = scores.argmax(axis=1)
    best_scores = scores[np.arange(scores.shape[0]), best]
    selected_is_set = triple_targets[np.arange(scores.shape[0]), best]
    predict_set = best_scores >= threshold
    return np.where(predict_set, has_set & selected_is_set, ~has_set)


def best_threshold(
    triple_targets: BoolArray, has_set: BoolArray, scores: FloatArray
) -> float:
    board_scores = scores.max(axis=1)
    candidates = np.concatenate(
        (
            np.asarray([np.nextafter(board_scores.min(), -np.inf)], dtype=np.float32),
            np.sort(np.unique(board_scores)),
            np.asarray([np.nextafter(board_scores.max(), np.inf)], dtype=np.float32),
        )
    )
    accuracies = np.asarray(
        [
            action_correct(triple_targets, has_set, scores, float(threshold)).mean()
            for threshold in candidates
        ]
    )
    return float(candidates[int(accuracies.argmax())])


def report_action(
    name: str,
    triple_targets: BoolArray,
    has_set: BoolArray,
    scores: FloatArray,
    threshold: float,
) -> None:
    correct = action_correct(triple_targets, has_set, scores, threshold)
    print(
        f"{name}: overall={correct.mean():.6f} "
        f"set={correct[has_set].mean():.6f} "
        f"no_set={correct[~has_set].mean():.6f} "
        f"threshold={threshold:.9f} failures={np.count_nonzero(~correct)}"
    )


def run_typed_pipeline(
    name: str,
    training_probabilities: FloatArray,
    testing_probabilities: FloatArray,
    training_card_targets: IntArray,
    testing_card_targets: IntArray,
    training_triple_targets: BoolArray,
    testing_triple_targets: BoolArray,
    training_kinds: BoolArray,
    testing_kinds: BoolArray,
    seed: int,
) -> PipelineResult:
    training_sums = triple_attribute_sums(training_probabilities)
    testing_sums = triple_attribute_sums(testing_probabilities)
    training_relations = typed_relation_targets(training_card_targets)
    testing_relations = typed_relation_targets(testing_card_targets)
    relation_model, relation_mean, relation_scale = train_multiclass_mlp(
        training_sums,
        training_relations,
        hidden_width=24,
        output_width=3,
        seed=seed,
        steps=1_200,
    )
    training_relation_logits = predict_mlp(
        training_sums, relation_model, relation_mean, relation_scale
    )
    testing_relation_logits = predict_mlp(
        testing_sums, relation_model, relation_mean, relation_scale
    )
    relation_accuracy = float(
        (testing_relation_logits.argmax(axis=-1) == testing_relations).mean()
    )
    training_relation_probabilities = softmax(training_relation_logits).reshape(
        training_sums.shape[0], training_sums.shape[1], 12
    )
    testing_relation_probabilities = softmax(testing_relation_logits).reshape(
        testing_sums.shape[0], testing_sums.shape[1], 12
    )
    set_model, set_mean, set_scale = train_binary_mlp(
        training_relation_probabilities,
        training_triple_targets,
        hidden_width=24,
        seed=seed ^ 0x51E7,
        steps=1_500,
    )
    training_scores = predict_mlp(
        training_relation_probabilities, set_model, set_mean, set_scale
    )[:, :, 0]
    testing_scores = predict_mlp(
        testing_relation_probabilities, set_model, set_mean, set_scale
    )[:, :, 0]
    threshold = best_threshold(training_triple_targets, training_kinds, training_scores)
    print(f"{name}_relation_accuracy={relation_accuracy:.6f}")
    report_action(
        f"{name}_test",
        testing_triple_targets,
        testing_kinds,
        testing_scores,
        threshold,
    )
    return PipelineResult(
        training_scores=training_scores,
        testing_scores=testing_scores,
        training_relations=training_relation_probabilities,
        testing_relations=testing_relation_probabilities,
    )


def report_selected_blend(
    name: str,
    first_training: FloatArray,
    first_testing: FloatArray,
    second_training: FloatArray,
    second_testing: FloatArray,
    training_targets: BoolArray,
    testing_targets: BoolArray,
    training_kinds: BoolArray,
    testing_kinds: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    first_mean = first_training.mean()
    first_scale = max(float(first_training.std()), 1.0e-6)
    second_mean = second_training.mean()
    second_scale = max(float(second_training.std()), 1.0e-6)
    first_training = (first_training - first_mean) / first_scale
    first_testing = (first_testing - first_mean) / first_scale
    second_training = (second_training - second_mean) / second_scale
    second_testing = (second_testing - second_mean) / second_scale
    candidates: list[tuple[float, float, float]] = []
    for first_weight in np.linspace(0.0, 1.0, 21):
        training_scores = (
            first_weight * first_training + (1.0 - first_weight) * second_training
        )
        threshold = best_threshold(training_targets, training_kinds, training_scores)
        accuracy = float(
            action_correct(
                training_targets, training_kinds, training_scores, threshold
            ).mean()
        )
        candidates.append((accuracy, float(first_weight), threshold))
    training_accuracy, selected_weight, threshold = max(candidates)
    testing_scores = (
        selected_weight * first_testing + (1.0 - selected_weight) * second_testing
    )
    print(
        f"{name}_selection: first_weight={selected_weight:.2f} "
        f"training_accuracy={training_accuracy:.6f}"
    )
    report_action(name, testing_targets, testing_kinds, testing_scores, threshold)
    selected_training_scores = (
        selected_weight * first_training + (1.0 - selected_weight) * second_training
    )
    return selected_training_scores, testing_scores


def report_joint_board_reader(
    name: str,
    selector_training_scores: FloatArray,
    selector_testing_scores: FloatArray,
    feature_training_scores: list[FloatArray],
    feature_testing_scores: list[FloatArray],
    training_targets: BoolArray,
    testing_targets: BoolArray,
    training_kinds: BoolArray,
    testing_kinds: BoolArray,
) -> None:
    training_features = np.concatenate(
        [board_summary_features(scores) for scores in feature_training_scores], axis=1
    )
    testing_features = np.concatenate(
        [board_summary_features(scores) for scores in feature_testing_scores], axis=1
    )
    weights, bias, mean, scale = train_board_logistic(training_features, training_kinds)
    training_board_scores = ((training_features - mean) / scale) @ weights + bias
    testing_board_scores = ((testing_features - mean) / scale) @ weights + bias
    training_best = selector_training_scores.argmax(axis=1)
    training_selected_is_set = training_targets[
        np.arange(training_targets.shape[0]), training_best
    ]
    candidates = np.sort(np.unique(training_board_scores))
    accuracies = np.asarray(
        [
            np.where(
                training_board_scores >= threshold,
                training_kinds & training_selected_is_set,
                ~training_kinds,
            ).mean()
            for threshold in candidates
        ]
    )
    threshold = float(candidates[int(accuracies.argmax())])
    testing_best = selector_testing_scores.argmax(axis=1)
    testing_selected_is_set = testing_targets[
        np.arange(testing_targets.shape[0]), testing_best
    ]
    predict_set = testing_board_scores >= threshold
    correct = np.where(
        predict_set,
        testing_kinds & testing_selected_is_set,
        ~testing_kinds,
    )
    ceiling = np.where(testing_kinds, testing_selected_is_set, True).mean()
    print(
        f"{name}: overall={correct.mean():.6f} "
        f"set={correct[testing_kinds].mean():.6f} "
        f"no_set={correct[~testing_kinds].mean():.6f} "
        f"selector_oracle_ceiling={ceiling:.6f} failures={np.count_nonzero(~correct)}"
    )


def report_pipeline_ensemble(
    name: str,
    members: list[PipelineResult],
    training_targets: BoolArray,
    testing_targets: BoolArray,
    training_kinds: BoolArray,
    testing_kinds: BoolArray,
) -> None:
    normalized_training: list[FloatArray] = []
    normalized_testing: list[FloatArray] = []
    for member in members:
        mean = member.training_scores.mean()
        scale = max(float(member.training_scores.std()), 1.0e-6)
        normalized_training.append((member.training_scores - mean) / scale)
        normalized_testing.append((member.testing_scores - mean) / scale)
    training_scores = np.mean(normalized_training, axis=0)
    testing_scores = np.mean(normalized_testing, axis=0)
    threshold = best_threshold(training_targets, training_kinds, training_scores)
    report_action(name, testing_targets, testing_kinds, testing_scores, threshold)


def fast_best_threshold_and_accuracy(
    triple_targets: BoolArray,
    has_set: BoolArray,
    scores: FloatArray,
) -> tuple[float, float]:
    best_triples = scores.argmax(axis=1)
    board_scores = scores[np.arange(scores.shape[0]), best_triples]
    selected_is_set = triple_targets[np.arange(scores.shape[0]), best_triples]
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
    return threshold, best_correct / scores.shape[0]


def integer_compositions(total: int, width: int) -> list[tuple[int, ...]]:
    if width == 1:
        return [(total,)]
    output: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for remainder in integer_compositions(total - first, width - 1):
            output.append((first, *remainder))
    return output


def search_weighted_score_ensemble(
    name: str,
    training_members: list[FloatArray],
    testing_members: list[FloatArray],
    training_targets: BoolArray,
    testing_targets: BoolArray,
    training_kinds: BoolArray,
    testing_kinds: BoolArray,
    units: int,
) -> None:
    normalized_training: list[FloatArray] = []
    normalized_testing: list[FloatArray] = []
    for training, testing in zip(training_members, testing_members, strict=True):
        mean = training.mean()
        scale = max(float(training.std()), 1.0e-6)
        normalized_training.append((training - mean) / scale)
        normalized_testing.append((testing - mean) / scale)
    uniform = 1.0 / len(training_members)
    selected: tuple[float, float, FloatArray, float] | None = None
    for composition in integer_compositions(units, len(training_members)):
        weights = np.asarray(composition, dtype=np.float32) / units
        training_scores = sum(
            weight * scores
            for weight, scores in zip(weights, normalized_training, strict=True)
        )
        threshold, accuracy = fast_best_threshold_and_accuracy(
            training_targets, training_kinds, training_scores
        )
        uniform_distance = float(((weights - uniform) ** 2).sum())
        candidate = (accuracy, -uniform_distance, weights, threshold)
        if selected is None or candidate[:2] > selected[:2]:
            selected = candidate
    if selected is None:
        raise ValueError("weighted ensemble search produced no candidates")
    training_accuracy, _, weights, threshold = selected
    testing_scores = sum(
        weight * scores
        for weight, scores in zip(weights, normalized_testing, strict=True)
    )
    print(
        f"{name}_selection: weights="
        + ",".join(f"{weight:.3f}" for weight in weights)
        + f" training_accuracy={training_accuracy:.6f}"
    )
    report_action(name, testing_targets, testing_kinds, testing_scores, threshold)


def board_summary_features(scores: FloatArray) -> FloatArray:
    ranked = np.sort(scores, axis=1)[:, ::-1]
    rank_features = ranked[:, [0, 1, 2, 4, 9, 19, 49, 99, 219]]
    mean = scores.mean(axis=1)
    standard_deviation = np.maximum(scores.std(axis=1), 1.0e-6)
    centered = (scores - mean[:, None]) / standard_deviation[:, None]
    moments = np.stack(
        (
            mean,
            standard_deviation,
            (centered**3).mean(axis=1),
            (centered**4).mean(axis=1),
        ),
        axis=1,
    )
    gaps = np.stack(
        (
            ranked[:, 0] - ranked[:, 1],
            ranked[:, 0] - ranked[:, 2],
            ranked[:, 0] - ranked[:, 4],
            ranked[:, 0] - ranked[:, 9],
        ),
        axis=1,
    )
    aggregate_features: list[FloatArray] = []
    for temperature in (0.5, 1.0, 2.0, 4.0):
        scaled = scores / temperature
        maximum = scaled.max(axis=1)
        logmeanexp = temperature * (
            maximum + np.log(np.exp(scaled - maximum[:, None]).mean(axis=1))
        )
        aggregate_features.append(logmeanexp[:, None])
    tail_counts = np.stack(
        (
            (centered >= 0.0).mean(axis=1),
            (centered >= 1.0).mean(axis=1),
            (centered >= 2.0).mean(axis=1),
            (centered >= 3.0).mean(axis=1),
        ),
        axis=1,
    )
    return np.concatenate(
        (rank_features, moments, gaps, *aggregate_features, tail_counts), axis=1
    ).astype(np.float32)


def train_board_logistic(
    features: FloatArray, targets: BoolArray, steps: int = 5_000
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1.0e-5)
    standardized = (features - mean) / scale
    weights = np.zeros(features.shape[1], dtype=np.float32)
    bias = 0.0
    first_weight = np.zeros_like(weights)
    second_weight = np.zeros_like(weights)
    first_bias = 0.0
    second_bias = 0.0
    expected = targets.astype(np.float32)
    for step in range(1, steps + 1):
        logits = standardized @ weights + bias
        error = (sigmoid(logits) - expected) / expected.size
        weight_gradient = standardized.T @ error + 2.0e-3 * weights
        bias_gradient = float(error.sum())
        first_weight = 0.9 * first_weight + 0.1 * weight_gradient
        second_weight = (
            0.999 * second_weight + 0.001 * weight_gradient * weight_gradient
        )
        first_bias = 0.9 * first_bias + 0.1 * bias_gradient
        second_bias = 0.999 * second_bias + 0.001 * bias_gradient * bias_gradient
        weights -= (
            0.03
            * (first_weight / (1.0 - 0.9**step))
            / (np.sqrt(second_weight / (1.0 - 0.999**step)) + 1.0e-8)
        )
        bias -= (
            0.03
            * (first_bias / (1.0 - 0.9**step))
            / (np.sqrt(second_bias / (1.0 - 0.999**step)) + 1.0e-8)
        )
    return weights, bias, mean, scale


def report_learned_board_reader(
    name: str,
    member: PipelineResult,
    training_targets: BoolArray,
    testing_targets: BoolArray,
    training_kinds: BoolArray,
    testing_kinds: BoolArray,
) -> None:
    training_features = board_summary_features(member.training_scores)
    testing_features = board_summary_features(member.testing_scores)
    weights, bias, mean, scale = train_board_logistic(training_features, training_kinds)
    training_board_scores = ((training_features - mean) / scale) @ weights + bias
    testing_board_scores = ((testing_features - mean) / scale) @ weights + bias
    threshold_candidates = np.sort(np.unique(training_board_scores))
    best_triples = member.training_scores.argmax(axis=1)
    selected_is_set = training_targets[
        np.arange(training_targets.shape[0]), best_triples
    ]
    accuracies = np.asarray(
        [
            np.where(
                training_board_scores >= threshold,
                training_kinds & selected_is_set,
                ~training_kinds,
            ).mean()
            for threshold in threshold_candidates
        ]
    )
    threshold = float(threshold_candidates[int(accuracies.argmax())])
    testing_best = member.testing_scores.argmax(axis=1)
    testing_selected_is_set = testing_targets[
        np.arange(testing_targets.shape[0]), testing_best
    ]
    predict_set = testing_board_scores >= threshold
    correct = np.where(
        predict_set,
        testing_kinds & testing_selected_is_set,
        ~testing_kinds,
    )
    oracle_ceiling = np.where(testing_kinds, testing_selected_is_set, True).mean()
    print(
        f"{name}: overall={correct.mean():.6f} "
        f"set={correct[testing_kinds].mean():.6f} "
        f"no_set={correct[~testing_kinds].mean():.6f} "
        f"selector_oracle_ceiling={oracle_ceiling:.6f} failures={np.count_nonzero(~correct)}"
    )


def main() -> None:
    if len(sys.argv) not in (2, 6):
        raise SystemExit(
            "usage: uv run vision/analyze_contextual_reader.py ENSEMBLE_EVIDENCE.tsv [MEMBER_1.tsv MEMBER_2.tsv MEMBER_3.tsv MEMBER_4.tsv]"
        )
    evidence = read_evidence(Path(sys.argv[1]))
    member_evidence = [read_evidence(Path(path)) for path in sys.argv[2:]]
    for member in member_evidence:
        if (
            not np.array_equal(member.triple_targets, evidence.triple_targets)
            or not np.array_equal(member.card_targets, evidence.card_targets)
            or not np.array_equal(member.has_set, evidence.has_set)
        ):
            raise ValueError(
                "individual model evidence does not match the ensemble board stream"
            )
    midpoint = evidence.attribute_logits.shape[0] // 2
    corrector_training_end = midpoint * 3 // 4
    correctors: list[Corrector] = []
    attribute_names = ("count", "colour", "shape", "fill")
    for attribute, name in enumerate(attribute_names):
        corrector, base_loss, corrected_loss, base_accuracy, corrected_accuracy = (
            train_corrector(
                evidence.attribute_logits[:corrector_training_end, :, attribute, :],
                evidence.card_targets[:corrector_training_end, :, attribute],
                evidence.attribute_logits[
                    corrector_training_end:midpoint, :, attribute, :
                ],
                evidence.card_targets[corrector_training_end:midpoint, :, attribute],
                seed=10_000 + attribute,
            )
        )
        correctors.append(corrector)
        print(
            f"corrector_{name}: enabled={corrector.enabled} "
            f"validation_loss={base_loss:.6f}->{corrected_loss:.6f} "
            f"validation_accuracy={base_accuracy:.6f}->{corrected_accuracy:.6f}"
        )

    corrected_logits = apply_correctors(evidence.attribute_logits, correctors)
    all_hypergraph_features = hypergraph_features(
        evidence.attribute_logits,
        evidence.triple_scores,
        evidence.relation_logits,
    )
    hypergraph_corrector, hypergraph_base_losses, hypergraph_corrected_losses = (
        train_hypergraph_corrector(
            all_hypergraph_features[:corrector_training_end],
            evidence.attribute_logits[:corrector_training_end],
            evidence.card_targets[:corrector_training_end],
            all_hypergraph_features[corrector_training_end:midpoint],
            evidence.attribute_logits[corrector_training_end:midpoint],
            evidence.card_targets[corrector_training_end:midpoint],
            seed=15_001,
        )
    )
    hypergraph_logits = apply_hypergraph_corrector(
        evidence.attribute_logits,
        all_hypergraph_features,
        hypergraph_corrector,
    )
    print(
        "hypergraph_corrector: enabled="
        + ",".join(
            f"{name}:{bool(enabled)}"
            for name, enabled in zip(
                attribute_names, hypergraph_corrector.enabled_attributes, strict=True
            )
        )
    )
    print(
        "hypergraph_validation_losses: "
        + ",".join(
            f"{name}={before:.6f}->{after:.6f}"
            for name, before, after in zip(
                attribute_names,
                hypergraph_base_losses,
                hypergraph_corrected_losses,
                strict=True,
            )
        )
    )
    report_attributes(
        "raw_attributes_test",
        evidence.attribute_logits[midpoint:],
        evidence.card_targets[midpoint:],
    )
    report_attributes(
        "contextual_attributes_test",
        corrected_logits[midpoint:],
        evidence.card_targets[midpoint:],
    )
    report_attributes(
        "hypergraph_attributes_test",
        hypergraph_logits[midpoint:],
        evidence.card_targets[midpoint:],
    )
    report_action(
        "existing_policy_test",
        evidence.triple_targets[midpoint:],
        evidence.has_set[midpoint:],
        evidence.triple_scores[midpoint:],
        0.0,
    )

    raw_pipeline = run_typed_pipeline(
        "raw_typed_reader",
        softmax(evidence.attribute_logits[:midpoint]),
        softmax(evidence.attribute_logits[midpoint:]),
        evidence.card_targets[:midpoint],
        evidence.card_targets[midpoint:],
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
        seed=20_001,
    )
    contextual_pipeline = run_typed_pipeline(
        "contextual_typed_reader",
        softmax(corrected_logits[:midpoint]),
        softmax(corrected_logits[midpoint:]),
        evidence.card_targets[:midpoint],
        evidence.card_targets[midpoint:],
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
        seed=30_001,
    )
    hypergraph_pipeline = run_typed_pipeline(
        "hypergraph_typed_reader",
        softmax(hypergraph_logits[:midpoint]),
        softmax(hypergraph_logits[midpoint:]),
        evidence.card_targets[:midpoint],
        evidence.card_targets[midpoint:],
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
        seed=35_001,
    )
    hypergraph_logits_members = [hypergraph_logits]
    hypergraph_pipeline_members = [hypergraph_pipeline]
    for replica in range(2, 5):
        replica_corrector, _, _ = train_hypergraph_corrector(
            all_hypergraph_features[:corrector_training_end],
            evidence.attribute_logits[:corrector_training_end],
            evidence.card_targets[:corrector_training_end],
            all_hypergraph_features[corrector_training_end:midpoint],
            evidence.attribute_logits[corrector_training_end:midpoint],
            evidence.card_targets[corrector_training_end:midpoint],
            seed=15_000 + replica,
        )
        replica_logits = apply_hypergraph_corrector(
            evidence.attribute_logits,
            all_hypergraph_features,
            replica_corrector,
        )
        hypergraph_logits_members.append(replica_logits)
        report_attributes(
            f"hypergraph_replica_{replica}_attributes_test",
            replica_logits[midpoint:],
            evidence.card_targets[midpoint:],
        )
        hypergraph_pipeline_members.append(
            run_typed_pipeline(
                f"hypergraph_replica_{replica}_typed_reader",
                softmax(replica_logits[:midpoint]),
                softmax(replica_logits[midpoint:]),
                evidence.card_targets[:midpoint],
                evidence.card_targets[midpoint:],
                evidence.triple_targets[:midpoint],
                evidence.triple_targets[midpoint:],
                evidence.has_set[:midpoint],
                evidence.has_set[midpoint:],
                seed=35_000 + replica,
            )
        )
    averaged_hypergraph_logits = np.mean(hypergraph_logits_members, axis=0)
    report_attributes(
        "hypergraph_logit_ensemble_attributes_test",
        averaged_hypergraph_logits[midpoint:],
        evidence.card_targets[midpoint:],
    )
    averaged_hypergraph_pipeline = run_typed_pipeline(
        "hypergraph_logit_ensemble_typed_reader",
        softmax(averaged_hypergraph_logits[:midpoint]),
        softmax(averaged_hypergraph_logits[midpoint:]),
        evidence.card_targets[:midpoint],
        evidence.card_targets[midpoint:],
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
        seed=39_001,
    )
    report_pipeline_ensemble(
        "hypergraph_action_ensemble_test",
        hypergraph_pipeline_members,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )
    report_pipeline_ensemble(
        "hypergraph_all_stage_ensemble_test",
        [*hypergraph_pipeline_members, averaged_hypergraph_pipeline],
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )
    for index, member in enumerate(hypergraph_pipeline_members, start=1):
        report_learned_board_reader(
            f"hypergraph_replica_{index}_board_reader_test",
            member,
            evidence.triple_targets[:midpoint],
            evidence.triple_targets[midpoint:],
            evidence.has_set[:midpoint],
            evidence.has_set[midpoint:],
        )
    report_learned_board_reader(
        "hypergraph_logit_ensemble_board_reader_test",
        averaged_hypergraph_pipeline,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )

    report_selected_blend(
        "existing_contextual_blend_test",
        evidence.triple_scores[:midpoint],
        evidence.triple_scores[midpoint:],
        contextual_pipeline.training_scores,
        contextual_pipeline.testing_scores,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )
    report_selected_blend(
        "raw_contextual_blend_test",
        raw_pipeline.training_scores,
        raw_pipeline.testing_scores,
        contextual_pipeline.training_scores,
        contextual_pipeline.testing_scores,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )
    report_selected_blend(
        "existing_hypergraph_blend_test",
        evidence.triple_scores[:midpoint],
        evidence.triple_scores[midpoint:],
        hypergraph_pipeline.training_scores,
        hypergraph_pipeline.testing_scores,
        evidence.triple_targets[:midpoint],
        evidence.triple_targets[midpoint:],
        evidence.has_set[:midpoint],
        evidence.has_set[midpoint:],
    )

    existing_relations = softmax(evidence.relation_logits).reshape(
        evidence.relation_logits.shape[0], TRIPLES.shape[0], 12
    )
    fusion_training = np.concatenate(
        (
            evidence.triple_scores[:midpoint, :, None],
            existing_relations[:midpoint],
            contextual_pipeline.training_relations,
            hypergraph_pipeline.training_relations,
        ),
        axis=2,
    )
    fusion_testing = np.concatenate(
        (
            evidence.triple_scores[midpoint:, :, None],
            existing_relations[midpoint:],
            contextual_pipeline.testing_relations,
            hypergraph_pipeline.testing_relations,
        ),
        axis=2,
    )
    fusion_model, fusion_mean, fusion_scale = train_binary_mlp(
        fusion_training,
        evidence.triple_targets[:midpoint],
        hidden_width=32,
        seed=40_001,
        steps=2_000,
    )
    fusion_training_scores = predict_mlp(
        fusion_training, fusion_model, fusion_mean, fusion_scale
    )[:, :, 0]
    fusion_testing_scores = predict_mlp(
        fusion_testing, fusion_model, fusion_mean, fusion_scale
    )[:, :, 0]
    fusion_threshold = best_threshold(
        evidence.triple_targets[:midpoint],
        evidence.has_set[:midpoint],
        fusion_training_scores,
    )
    report_action(
        "learned_contextual_fusion_test",
        evidence.triple_targets[midpoint:],
        evidence.has_set[midpoint:],
        fusion_testing_scores,
        fusion_threshold,
    )

    if member_evidence:
        stacked_logits = np.empty_like(evidence.attribute_logits)
        for attribute, name in enumerate(attribute_names):
            features, base = stacker_features_and_base(
                [
                    member.attribute_logits[:, :, attribute, :]
                    for member in member_evidence
                ]
            )
            corrector, base_loss, corrected_loss, base_accuracy, corrected_accuracy = (
                train_feature_corrector(
                    features[:corrector_training_end],
                    base[:corrector_training_end],
                    evidence.card_targets[:corrector_training_end, :, attribute],
                    features[corrector_training_end:midpoint],
                    base[corrector_training_end:midpoint],
                    evidence.card_targets[
                        corrector_training_end:midpoint, :, attribute
                    ],
                    seed=50_000 + attribute,
                )
            )
            stacked_logits[:, :, attribute, :] = apply_feature_corrector(
                features, base, corrector
            )
            print(
                f"logit_stacker_{name}: enabled={corrector.enabled} "
                f"validation_loss={base_loss:.6f}->{corrected_loss:.6f} "
                f"validation_accuracy={base_accuracy:.6f}->{corrected_accuracy:.6f}"
            )
        report_attributes(
            "logit_stacker_attributes_test",
            stacked_logits[midpoint:],
            evidence.card_targets[midpoint:],
        )
        stacked_hypergraph_features = hypergraph_features(
            stacked_logits,
            evidence.triple_scores,
            evidence.relation_logits,
        )
        stacked_hypergraph_corrector, _, _ = train_hypergraph_corrector(
            stacked_hypergraph_features[:corrector_training_end],
            stacked_logits[:corrector_training_end],
            evidence.card_targets[:corrector_training_end],
            stacked_hypergraph_features[corrector_training_end:midpoint],
            stacked_logits[corrector_training_end:midpoint],
            evidence.card_targets[corrector_training_end:midpoint],
            seed=55_001,
        )
        stacked_hypergraph_logits = apply_hypergraph_corrector(
            stacked_logits,
            stacked_hypergraph_features,
            stacked_hypergraph_corrector,
        )
        report_attributes(
            "stacked_hypergraph_attributes_test",
            stacked_hypergraph_logits[midpoint:],
            evidence.card_targets[midpoint:],
        )
        stacked_hypergraph_pipeline = run_typed_pipeline(
            "stacked_hypergraph_typed_reader",
            softmax(stacked_hypergraph_logits[:midpoint]),
            softmax(stacked_hypergraph_logits[midpoint:]),
            evidence.card_targets[:midpoint],
            evidence.card_targets[midpoint:],
            evidence.triple_targets[:midpoint],
            evidence.triple_targets[midpoint:],
            evidence.has_set[:midpoint],
            evidence.has_set[midpoint:],
            seed=60_001,
        )
        stacked_hypergraph_members = [stacked_hypergraph_pipeline]
        for replica in range(2, 9):
            replica_corrector, _, _ = train_hypergraph_corrector(
                stacked_hypergraph_features[:corrector_training_end],
                stacked_logits[:corrector_training_end],
                evidence.card_targets[:corrector_training_end],
                stacked_hypergraph_features[corrector_training_end:midpoint],
                stacked_logits[corrector_training_end:midpoint],
                evidence.card_targets[corrector_training_end:midpoint],
                seed=55_000 + replica,
            )
            replica_logits = apply_hypergraph_corrector(
                stacked_logits,
                stacked_hypergraph_features,
                replica_corrector,
            )
            report_attributes(
                f"stacked_hypergraph_replica_{replica}_attributes_test",
                replica_logits[midpoint:],
                evidence.card_targets[midpoint:],
            )
            stacked_hypergraph_members.append(
                run_typed_pipeline(
                    f"stacked_hypergraph_replica_{replica}_typed_reader",
                    softmax(replica_logits[:midpoint]),
                    softmax(replica_logits[midpoint:]),
                    evidence.card_targets[:midpoint],
                    evidence.card_targets[midpoint:],
                    evidence.triple_targets[:midpoint],
                    evidence.triple_targets[midpoint:],
                    evidence.has_set[:midpoint],
                    evidence.has_set[midpoint:],
                    seed=60_000 + replica,
                )
            )
        report_pipeline_ensemble(
            "stacked_hypergraph_action_ensemble_test",
            stacked_hypergraph_members,
            evidence.triple_targets[:midpoint],
            evidence.triple_targets[midpoint:],
            evidence.has_set[:midpoint],
            evidence.has_set[midpoint:],
        )
        search_weighted_score_ensemble(
            "stacked_hypergraph_weighted_ensemble_test",
            [member.training_scores for member in stacked_hypergraph_members[:4]],
            [member.testing_scores for member in stacked_hypergraph_members[:4]],
            evidence.triple_targets[:midpoint],
            evidence.triple_targets[midpoint:],
            evidence.has_set[:midpoint],
            evidence.has_set[midpoint:],
            units=10,
        )
        search_weighted_score_ensemble(
            "existing_plus_stacked_weighted_ensemble_test",
            [
                evidence.triple_scores[:midpoint],
                *[member.training_scores for member in stacked_hypergraph_members[:4]],
            ],
            [
                evidence.triple_scores[midpoint:],
                *[member.testing_scores for member in stacked_hypergraph_members[:4]],
            ],
            evidence.triple_targets[:midpoint],
            evidence.triple_targets[midpoint:],
            evidence.has_set[:midpoint],
            evidence.has_set[midpoint:],
            units=8,
        )
        for index, member in enumerate(stacked_hypergraph_members, start=1):
            blend_training, blend_testing = report_selected_blend(
                f"existing_stacked_hypergraph_replica_{index}_blend_test",
                evidence.triple_scores[:midpoint],
                evidence.triple_scores[midpoint:],
                member.training_scores,
                member.testing_scores,
                evidence.triple_targets[:midpoint],
                evidence.triple_targets[midpoint:],
                evidence.has_set[:midpoint],
                evidence.has_set[midpoint:],
            )
            report_joint_board_reader(
                f"existing_stacked_hypergraph_replica_{index}_joint_board_test",
                blend_training,
                blend_testing,
                [
                    evidence.triple_scores[:midpoint],
                    member.training_scores,
                    blend_training,
                ],
                [
                    evidence.triple_scores[midpoint:],
                    member.testing_scores,
                    blend_testing,
                ],
                evidence.triple_targets[:midpoint],
                evidence.triple_targets[midpoint:],
                evidence.has_set[:midpoint],
                evidence.has_set[midpoint:],
            )
        report_learned_board_reader(
            "stacked_hypergraph_board_reader_test",
            stacked_hypergraph_pipeline,
            evidence.triple_targets[:midpoint],
            evidence.triple_targets[midpoint:],
            evidence.has_set[:midpoint],
            evidence.has_set[midpoint:],
        )


if __name__ == "__main__":
    main()
