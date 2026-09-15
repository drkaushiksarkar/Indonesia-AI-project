from __future__ import annotations

import math
import statistics
from collections.abc import Callable

Forecast = Callable[[list[float], int, int], list[float]]

MODEL_REGISTRY = [
    ("naive", "baseline", False),
    ("seasonal_naive", "baseline", False),
    ("historical_mean", "level", False),
    ("historical_median", "level", False),
    ("moving_average", "level", False),
    ("drift", "trend", False),
    ("ses_optimized", "exponential_smoothing", False),
    ("holt_damped", "exponential_smoothing", False),
    ("autoregression_ridge", "autoregression", False),
    ("analog_knn", "nonlinear_analog", False),
    ("croston_sba", "intermittent", True),
    ("tsb", "intermittent", True),
    ("median_ensemble", "ensemble", False),
    ("online_selector", "adaptive_selection", False),
]


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _clip(values: list[float]) -> list[float]:
    return [max(0.0, value) if math.isfinite(value) else 0.0 for value in values]


def naive(values: list[float], horizon: int, season: int) -> list[float]:
    return [values[-1]] * horizon


def seasonal_naive(values: list[float], horizon: int, season: int) -> list[float]:
    if season <= 1 or len(values) < season:
        return naive(values, horizon, season)
    return [values[-season + index % season] for index in range(horizon)]


def historical_mean(values: list[float], horizon: int, season: int) -> list[float]:
    estimate = statistics.fmean(_finite(values))
    return [estimate] * horizon


def historical_median(values: list[float], horizon: int, season: int) -> list[float]:
    estimate = statistics.median(_finite(values))
    return [estimate] * horizon


def moving_average(values: list[float], horizon: int, season: int) -> list[float]:
    window = min(len(values), max(3, min(season, 12)))
    result = list(values)
    predictions: list[float] = []
    for _ in range(horizon):
        estimate = statistics.fmean(result[-window:])
        predictions.append(estimate)
        result.append(estimate)
    return _clip(predictions)


def drift(values: list[float], horizon: int, season: int) -> list[float]:
    slope = (values[-1] - values[0]) / max(1, len(values) - 1)
    return _clip([values[-1] + slope * step for step in range(1, horizon + 1)])


def _ses_fit(values: list[float], alpha: float) -> tuple[float, float]:
    level = values[0]
    error = 0.0
    for value in values[1:]:
        error += (value - level) ** 2
        level = alpha * value + (1.0 - alpha) * level
    return level, error


def ses_optimized(values: list[float], horizon: int, season: int) -> list[float]:
    candidates = [
        (alpha, _ses_fit(values, alpha)) for alpha in [index / 10 for index in range(1, 10)]
    ]
    level = min(candidates, key=lambda item: item[1][1])[1][0]
    return _clip([level] * horizon)


def _holt_fit(
    values: list[float], alpha: float, beta: float, phi: float
) -> tuple[float, float, float]:
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0.0
    error = 0.0
    for value in values[1:]:
        predicted = level + phi * trend
        error += (value - predicted) ** 2
        previous = level
        level = alpha * value + (1.0 - alpha) * predicted
        trend = beta * (level - previous) + (1.0 - beta) * phi * trend
    return level, trend, error


def holt_damped(values: list[float], horizon: int, season: int) -> list[float]:
    candidates = [
        (alpha, beta, phi, _holt_fit(values, alpha, beta, phi))
        for alpha in (0.2, 0.5, 0.8)
        for beta in (0.1, 0.3, 0.6)
        for phi in (0.8, 0.9, 0.98)
    ]
    level, trend, _ = min(candidates, key=lambda item: item[3][2])[3]
    predictions: list[float] = []
    damping = 0.0
    for step in range(1, horizon + 1):
        damping += 0.9**step
        predictions.append(level + damping * trend)
    return _clip(predictions)


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    size = len(target)
    augmented = [[*row, target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            continue
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * reference
                for current, reference in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][-1] for index in range(size)]


def _ridge_fit(rows: list[list[float]], target: list[float], penalty: float) -> list[float]:
    width = len(rows[0])
    matrix = [[0.0] * width for _ in range(width)]
    vector = [0.0] * width
    for features, value in zip(rows, target, strict=True):
        for left in range(width):
            vector[left] += features[left] * value
            for right in range(width):
                matrix[left][right] += features[left] * features[right]
    for index in range(1, width):
        matrix[index][index] += penalty
    return _solve(matrix, vector)


def autoregression_ridge(values: list[float], horizon: int, season: int) -> list[float]:
    lags = list(range(1, min(3, len(values) // 5) + 1))
    if season > 1 and season < len(values) // 2:
        lags.append(season)
    lags = sorted(set(lags))
    maximum = max(lags)
    rows = [[1.0] + [values[index - lag] for lag in lags] for index in range(maximum, len(values))]
    target = values[maximum:]
    if len(rows) < len(lags) + 2:
        return ses_optimized(values, horizon, season)
    scale = max(statistics.pstdev(values), 1.0)
    coefficients = _ridge_fit(rows, target, 0.1 * scale)
    result = list(values)
    predictions: list[float] = []
    for _ in range(horizon):
        features = [1.0] + [result[-lag] for lag in lags]
        estimate = sum(
            coefficient * feature
            for coefficient, feature in zip(coefficients, features, strict=True)
        )
        estimate = max(0.0, estimate)
        result.append(estimate)
        predictions.append(estimate)
    return predictions


def analog_knn(values: list[float], horizon: int, season: int) -> list[float]:
    dimension = min(max(2, min(5, len(values) // 8)), len(values) // 3)
    query = values[-dimension:]
    scale = max(statistics.pstdev(values), 1e-9)
    candidates: list[tuple[float, int]] = []
    for end in range(dimension, len(values) - horizon + 1):
        pattern = values[end - dimension : end]
        distance = math.sqrt(
            statistics.fmean(
                ((left - right) / scale) ** 2 for left, right in zip(query, pattern, strict=True)
            )
        )
        candidates.append((distance, end))
    if not candidates:
        return seasonal_naive(values, horizon, season)
    selected = sorted(candidates)[: min(5, len(candidates))]
    predictions: list[float] = []
    for step in range(horizon):
        weighted = [(1.0 / max(distance, 1e-6), values[end + step]) for distance, end in selected]
        predictions.append(
            sum(weight * value for weight, value in weighted)
            / sum(weight for weight, _ in weighted)
        )
    return _clip(predictions)


def croston_sba(values: list[float], horizon: int, season: int) -> list[float]:
    alpha = 0.1
    positives = [(index, value) for index, value in enumerate(values) if value > 0]
    if not positives:
        return [0.0] * horizon
    size = positives[0][1]
    interval = max(1.0, float(positives[0][0] + 1))
    previous = positives[0][0]
    for index, value in positives[1:]:
        size += alpha * (value - size)
        gap = index - previous
        interval += alpha * (gap - interval)
        previous = index
    estimate = (1.0 - alpha / 2.0) * size / max(interval, 1e-9)
    return [max(0.0, estimate)] * horizon


def tsb(values: list[float], horizon: int, season: int) -> list[float]:
    alpha = 0.1
    beta = 0.1
    positives = [value for value in values if value > 0]
    if not positives:
        return [0.0] * horizon
    size = positives[0]
    probability = 1.0 if values[0] > 0 else 0.0
    for value in values[1:]:
        occurrence = 1.0 if value > 0 else 0.0
        probability += beta * (occurrence - probability)
        if occurrence:
            size += alpha * (value - size)
    estimate = probability * size
    return [max(0.0, estimate)] * horizon


FORECASTERS: dict[str, Forecast] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "historical_mean": historical_mean,
    "historical_median": historical_median,
    "moving_average": moving_average,
    "drift": drift,
    "ses_optimized": ses_optimized,
    "holt_damped": holt_damped,
    "autoregression_ridge": autoregression_ridge,
    "analog_knn": analog_knn,
    "croston_sba": croston_sba,
    "tsb": tsb,
}
