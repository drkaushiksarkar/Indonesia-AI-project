from __future__ import annotations

import math
import random
import statistics


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mase_scale(values: list[float], season: int) -> float:
    lag = season if season > 1 and len(values) > season else 1
    differences = [abs(values[index] - values[index - lag]) for index in range(lag, len(values))]
    scale = statistics.fmean(differences) if differences else 0.0
    if scale <= 1e-9 and lag != 1:
        scale = statistics.fmean(abs(values[index] - values[index - 1]) for index in range(1, len(values)))
    return max(scale, 1.0)


def smape(actual: float, predicted: float) -> float:
    denominator = abs(actual) + abs(predicted)
    return 0.0 if denominator == 0 else 200.0 * abs(actual - predicted) / denominator


def poisson_deviance(actual: float, predicted: float) -> float:
    estimate = max(predicted, 1e-9)
    return 2.0 * estimate if actual <= 0 else 2.0 * (actual * math.log(actual / estimate) - actual + estimate)


def poisson_log_score(actual: float, predicted: float) -> float:
    estimate = max(predicted, 1e-9)
    return estimate - actual * math.log(estimate) + math.lgamma(actual + 1.0)


def pinball(actual: float, predicted: float, probability: float) -> float:
    error = actual - predicted
    return probability * error if error >= 0 else (probability - 1.0) * error


def interval_score(actual: float, lower: float, upper: float, alpha: float) -> float:
    penalty = upper - lower
    if actual < lower:
        penalty += 2.0 * (lower - actual) / alpha
    if actual > upper:
        penalty += 2.0 * (actual - upper) / alpha
    return penalty


def weighted_interval_score(actual: float, q10: float, q25: float, q50: float, q75: float, q90: float) -> float:
    return (
        0.5 * abs(actual - q50)
        + 0.1 * interval_score(actual, q10, q90, 0.2)
        + 0.25 * interval_score(actual, q25, q75, 0.5)
    ) / 0.85


def crps_quantile(actual: float, quantiles: list[tuple[float, float]]) -> float:
    return 2.0 * statistics.fmean(pinball(actual, value, probability) for probability, value in quantiles)


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
        rank = (index + 1 + end) / 2.0
        rank_sum += rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auc_pr(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return math.nan
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    true_positive = 0
    area = 0.0
    previous_recall = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        true_positive += label
        recall = true_positive / positives
        area += (recall - previous_recall) * true_positive / index
        previous_recall = recall
    return area


def binary_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    predictions = [int(score >= 0.5) for score in scores]
    true_positive = sum(left == 1 and right == 1 for left, right in zip(labels, predictions, strict=True))
    true_negative = sum(left == 0 and right == 0 for left, right in zip(labels, predictions, strict=True))
    false_positive = sum(left == 0 and right == 1 for left, right in zip(labels, predictions, strict=True))
    false_negative = sum(left == 1 and right == 0 for left, right in zip(labels, predictions, strict=True))
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    return {
        "auroc": auc_roc(labels, scores),
        "auprc": auc_pr(labels, scores),
        "brier": statistics.fmean((left - right) ** 2 for left, right in zip(labels, scores, strict=True)),
        "log_loss": -statistics.fmean(left * math.log(max(right, 1e-9)) + (1 - left) * math.log(max(1.0 - right, 1e-9)) for left, right in zip(labels, scores, strict=True)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "f1": 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-9),
        "false_alarms_per_100": 100.0 * false_positive / max(1, len(labels)),
        "event_rate": statistics.fmean(labels),
    }


def risk_metrics(labels: list[int], probabilities: list[tuple[float, float, float]]) -> dict[str, float]:
    predictions = [max(range(3), key=lambda index: values[index]) for values in probabilities]
    recalls: list[float] = []
    f1_values: list[float] = []
    for category in range(3):
        true_positive = sum(left == category and right == category for left, right in zip(labels, predictions, strict=True))
        false_positive = sum(left != category and right == category for left, right in zip(labels, predictions, strict=True))
        false_negative = sum(left == category and right != category for left, right in zip(labels, predictions, strict=True))
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        recalls.append(recall)
        f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-9))
    ranked_probability = statistics.fmean(
        sum((sum(values[: boundary + 1]) - float(label <= boundary)) ** 2 for boundary in (0, 1)) / 2.0
        for label, values in zip(labels, probabilities, strict=True)
    )
    return {
        "macro_f1": statistics.fmean(f1_values),
        "balanced_accuracy": statistics.fmean(recalls),
        "ordinal_mae": statistics.fmean(abs(left - right) for left, right in zip(labels, predictions, strict=True)),
        "accuracy": statistics.fmean(left == right for left, right in zip(labels, predictions, strict=True)),
        "ranked_probability_score": ranked_probability,
        "multiclass_log_loss": -statistics.fmean(math.log(max(values[label], 1e-9)) for label, values in zip(labels, probabilities, strict=True)),
    }


def block_bootstrap_skill(errors: list[float], baseline: list[float], seed: int, replicates: int = 400) -> tuple[float, float, float]:
    if not errors or len(errors) != len(baseline):
        return math.nan, math.nan, math.nan
    estimate = 1.0 - statistics.fmean(errors) / max(statistics.fmean(baseline), 1e-9)
    generator = random.Random(seed)
    block = max(1, round(len(errors) ** (1.0 / 3.0)))
    samples: list[float] = []
    for _ in range(replicates):
        indices: list[int] = []
        while len(indices) < len(errors):
            start = generator.randrange(len(errors))
            indices.extend((start + offset) % len(errors) for offset in range(block))
        indices = indices[: len(errors)]
        current = statistics.fmean(errors[index] for index in indices)
        reference = statistics.fmean(baseline[index] for index in indices)
        samples.append(1.0 - current / max(reference, 1e-9))
    return estimate, quantile(samples, 0.025), quantile(samples, 0.975)
