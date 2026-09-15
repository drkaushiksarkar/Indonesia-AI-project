from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from itertools import pairwise


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mase_scale(values: list[float], season: int) -> float:
    lag = season if season > 1 and len(values) > season else 1
    differences = [abs(values[index] - values[index - lag]) for index in range(lag, len(values))]
    scale = statistics.fmean(differences) if differences else 0.0
    if scale <= 1e-9 and lag != 1:
        differences = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
        scale = statistics.fmean(differences) if differences else 0.0
    return max(scale, 1.0)


def rmsse_scale(values: list[float], season: int) -> float:
    lag = season if season > 1 and len(values) > season else 1
    differences = [(values[index] - values[index - lag]) ** 2 for index in range(lag, len(values))]
    scale = statistics.fmean(differences) if differences else 0.0
    if scale <= 1e-9 and lag != 1:
        differences = [(values[index] - values[index - 1]) ** 2 for index in range(1, len(values))]
        scale = statistics.fmean(differences) if differences else 0.0
    return max(scale, 1.0)


def smape(actual: float, predicted: float) -> float:
    denominator = abs(actual) + abs(predicted)
    return 0.0 if denominator == 0 else 200.0 * abs(actual - predicted) / denominator


def poisson_deviance(actual: float, predicted: float) -> float:
    estimate = max(predicted, 1e-9)
    if actual <= 0:
        return 2.0 * estimate
    return 2.0 * (actual * math.log(actual / estimate) - actual + estimate)


def normal_probability_above(threshold: float, mean: float, sigma: float) -> float:
    value = (threshold - mean) / max(sigma, 1e-9)
    probability = 0.5 * math.erfc(value / math.sqrt(2.0))
    return min(1.0 - 1e-9, max(1e-9, probability))


def auc_roc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auc_pr(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return math.nan
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in ranked:
        true_positive += label
        false_positive += 1 - label
        recall = true_positive / positives
        precision = true_positive / max(1, true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def binary_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    predictions = [int(score >= 0.5) for score in scores]
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predictions, strict=True)
    )
    true_negative = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predictions, strict=True)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, predictions, strict=True)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(labels, predictions, strict=True)
    )
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    return {
        "auroc": auc_roc(labels, scores),
        "auprc": auc_pr(labels, scores),
        "brier": statistics.fmean(
            (label - score) ** 2 for label, score in zip(labels, scores, strict=True)
        ),
        "log_loss": -statistics.fmean(
            label * math.log(score) + (1 - label) * math.log(1.0 - score)
            for label, score in zip(labels, scores, strict=True)
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "f1": 2.0 * precision * sensitivity / max(1e-9, precision + sensitivity),
        "event_rate": statistics.fmean(labels),
    }


def multiclass_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    recalls: list[float] = []
    f1_values: list[float] = []
    for category in (0, 1, 2):
        true_positive = sum(
            left == category and right == category
            for left, right in zip(labels, predictions, strict=True)
        )
        false_positive = sum(
            left != category and right == category
            for left, right in zip(labels, predictions, strict=True)
        )
        false_negative = sum(
            left == category and right != category
            for left, right in zip(labels, predictions, strict=True)
        )
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        recalls.append(recall)
        f1_values.append(2.0 * precision * recall / max(1e-9, precision + recall))
    return {
        "macro_f1": statistics.fmean(f1_values),
        "balanced_accuracy": statistics.fmean(recalls),
        "ordinal_mae": statistics.fmean(
            abs(left - right) for left, right in zip(labels, predictions, strict=True)
        ),
        "accuracy": statistics.fmean(
            left == right for left, right in zip(labels, predictions, strict=True)
        ),
    }


def bootstrap_skill_interval(
    errors: list[float], baseline_errors: list[float], seed: int, replicates: int = 400
) -> tuple[float, float, float]:
    if not errors or len(errors) != len(baseline_errors):
        return math.nan, math.nan, math.nan
    baseline = statistics.fmean(baseline_errors)
    estimate = 1.0 - statistics.fmean(errors) / max(baseline, 1e-9)
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        indices = [generator.randrange(len(errors)) for _ in errors]
        current = statistics.fmean(errors[index] for index in indices)
        reference = statistics.fmean(baseline_errors[index] for index in indices)
        values.append(1.0 - current / max(reference, 1e-9))
    return estimate, quantile(values, 0.025), quantile(values, 0.975)


def permutation_entropy(values: list[float], order: int) -> float:
    if len(values) < order + 2:
        return math.nan
    patterns: Counter[tuple[int, ...]] = Counter()
    for start in range(len(values) - order + 1):
        window = values[start : start + order]
        pattern = tuple(sorted(range(order), key=lambda index: (window[index], index)))
        patterns[pattern] += 1
    total = sum(patterns.values())
    entropy = -sum((count / total) * math.log(count / total) for count in patterns.values())
    return entropy / math.log(math.factorial(order))


def spectral_entropy(values: list[float]) -> float:
    size = len(values)
    if size < 4:
        return math.nan
    centered = [value - statistics.fmean(values) for value in values]
    powers: list[float] = []
    for frequency in range(1, size // 2 + 1):
        real = sum(
            value * math.cos(2.0 * math.pi * frequency * index / size)
            for index, value in enumerate(centered)
        )
        imaginary = -sum(
            value * math.sin(2.0 * math.pi * frequency * index / size)
            for index, value in enumerate(centered)
        )
        powers.append(real * real + imaginary * imaginary)
    total = sum(powers)
    if total <= 0:
        return 0.0
    probabilities = [power / total for power in powers if power > 0]
    return -sum(probability * math.log(probability) for probability in probabilities) / math.log(
        len(powers)
    )


def mutual_information(values: list[float], lag: int, bins: int = 4) -> float:
    if lag <= 0 or len(values) <= lag + bins:
        return math.nan
    edges = [quantile(values, index / bins) for index in range(1, bins)]
    categories = [sum(value > edge for edge in edges) for value in values]
    pairs = list(zip(categories[:-lag], categories[lag:], strict=True))
    joint = Counter(pairs)
    left = Counter(item[0] for item in pairs)
    right = Counter(item[1] for item in pairs)
    total = len(pairs)
    information = 0.0
    for (left_value, right_value), count in joint.items():
        probability = count / total
        information += probability * math.log(
            probability / ((left[left_value] / total) * (right[right_value] / total))
        )
    normalizer = math.log(bins)
    return information / normalizer if normalizer > 0 else math.nan


def intermittency(values: list[float]) -> tuple[float, float, float, str]:
    positive_positions = [index for index, value in enumerate(values) if value > 0]
    zero_rate = statistics.fmean(value == 0 for value in values)
    if len(positive_positions) < 2:
        return zero_rate, math.inf, math.inf, "lumpy"
    intervals = [right - left for left, right in pairwise(positive_positions)]
    positives = [values[index] for index in positive_positions]
    adi = statistics.fmean(intervals)
    mean_positive = statistics.fmean(positives)
    cv2 = (statistics.pstdev(positives) / max(mean_positive, 1e-9)) ** 2
    demand_class = (
        "smooth"
        if adi < 1.32 and cv2 < 0.49
        else "erratic"
        if adi < 1.32
        else "intermittent"
        if cv2 < 0.49
        else "lumpy"
    )
    return zero_rate, adi, cv2, demand_class
