from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

Forecast = Callable[[list[float], int, int], list[float]]
Surveillance = Callable[[list[float], int], float]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    scopes: tuple[str, ...]
    intermittent_only: bool = False
    pooled: bool = False


def _clip(values: list[float]) -> list[float]:
    return [max(0.0, value) if math.isfinite(value) else 0.0 for value in values]


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    size = len(target)
    augmented = [[*row, target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-10:
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


def _ridge(rows: list[list[float]], target: list[float], penalty: float) -> list[float]:
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


def naive(values: list[float], horizon: int, season: int) -> list[float]:
    return [values[-1]] * horizon


def seasonal_naive(values: list[float], horizon: int, season: int) -> list[float]:
    if season <= 1 or len(values) < season:
        return naive(values, horizon, season)
    return [values[-season + index % season] for index in range(horizon)]


def historical_median(values: list[float], horizon: int, season: int) -> list[float]:
    return [statistics.median(values)] * horizon


def moving_average(values: list[float], horizon: int, season: int) -> list[float]:
    width = min(len(values), max(3, min(season, 12)))
    state = list(values)
    result: list[float] = []
    for _ in range(horizon):
        estimate = statistics.fmean(state[-width:])
        result.append(estimate)
        state.append(estimate)
    return _clip(result)


def seasonal_window_average(values: list[float], horizon: int, season: int) -> list[float]:
    if season <= 1 or len(values) < season:
        return moving_average(values, horizon, season)
    result: list[float] = []
    for step in range(horizon):
        phase = (len(values) + step) % season
        candidates = [value for index, value in enumerate(values) if index % season == phase]
        result.append(statistics.fmean(candidates[-5:]))
    return _clip(result)


def _ses(values: list[float], alpha: float) -> tuple[float, float]:
    level = values[0]
    error = 0.0
    for value in values[1:]:
        error += (value - level) ** 2
        level += alpha * (value - level)
    return level, error


def ses_optimized(values: list[float], horizon: int, season: int) -> list[float]:
    level, _ = min((_ses(values, index / 20) for index in range(1, 20)), key=lambda item: item[1])
    return [max(0.0, level)] * horizon


def _holt(values: list[float], alpha: float, beta: float, phi: float) -> tuple[float, float, float]:
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0.0
    error = 0.0
    for value in values[1:]:
        expected = level + phi * trend
        error += (value - expected) ** 2
        previous = level
        level = alpha * value + (1.0 - alpha) * expected
        trend = beta * (level - previous) + (1.0 - beta) * phi * trend
    return level, trend, error


def _holt_forecast(values: list[float], horizon: int, phi: float) -> list[float]:
    fit = min(
        (
            _holt(values, alpha, beta, phi)
            for alpha in (0.2, 0.4, 0.6, 0.8)
            for beta in (0.05, 0.15, 0.3, 0.5)
        ),
        key=lambda item: item[2],
    )
    level, trend, _ = fit
    total = 0.0
    result: list[float] = []
    for step in range(1, horizon + 1):
        total += phi**step
        result.append(level + total * trend)
    return _clip(result)


def holt_linear(values: list[float], horizon: int, season: int) -> list[float]:
    return _holt_forecast(values, horizon, 1.0)


def holt_damped(values: list[float], horizon: int, season: int) -> list[float]:
    return _holt_forecast(values, horizon, 0.9)


def _seasonal_fit(values: list[float], season: int, alpha: float, beta: float, gamma: float) -> tuple[float, float, list[float], float]:
    level = statistics.fmean(values[:season])
    second = statistics.fmean(values[season : 2 * season]) if len(values) >= 2 * season else level
    trend = (second - level) / max(season, 1)
    seasonal = [values[index] - level for index in range(season)]
    error = 0.0
    for index, value in enumerate(values):
        phase = index % season
        expected = level + trend + seasonal[phase]
        error += (value - expected) ** 2
        previous = level
        level = alpha * (value - seasonal[phase]) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - previous) + (1.0 - beta) * trend
        seasonal[phase] = gamma * (value - level) + (1.0 - gamma) * seasonal[phase]
    return level, trend, seasonal, error


def holt_winters_additive(values: list[float], horizon: int, season: int) -> list[float]:
    if season <= 1 or len(values) < season + 4:
        return holt_damped(values, horizon, season)
    fit = min(
        (
            _seasonal_fit(values, season, alpha, beta, gamma)
            for alpha in (0.2, 0.5, 0.8)
            for beta in (0.05, 0.2)
            for gamma in (0.1, 0.3, 0.6)
        ),
        key=lambda item: item[3],
    )
    level, trend, seasonal, _ = fit
    return _clip(
        [level + step * trend + seasonal[(len(values) + step - 1) % season] for step in range(1, horizon + 1)]
    )


def theta(values: list[float], horizon: int, season: int) -> list[float]:
    size = len(values)
    x_mean = (size - 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(size))
    slope = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / max(denominator, 1e-9)
    intercept = y_mean - slope * x_mean
    residuals = [value - (intercept + slope * index) for index, value in enumerate(values)]
    level, _ = _ses(residuals, 0.2)
    return _clip([intercept + slope * (size + step) + level for step in range(horizon)])


def _lag_features(values: list[float], index: int, season: int, log: bool = False) -> list[float]:
    transform = math.log1p if log else float
    lags = [1, 2, 3]
    if season > 1 and season < index:
        lags.append(season)
    features = [1.0]
    features.extend(transform(values[index - lag]) for lag in sorted(set(lags)) if lag <= index)
    return features


def _ar_forecast(values: list[float], horizon: int, season: int, order: int, log: bool = False) -> list[float]:
    maximum = max(order, season if season > 1 and len(values) > 2 * season else order)
    lags = list(range(1, order + 1))
    if maximum == season:
        lags.append(season)
    transform = math.log1p if log else float
    rows = [[1.0, *(transform(values[index - lag]) for lag in lags)] for index in range(maximum, len(values))]
    target = [transform(values[index]) for index in range(maximum, len(values))]
    if len(rows) < len(lags) + 3:
        return ses_optimized(values, horizon, season)
    coefficients = _ridge(rows, target, 0.1)
    state = list(values)
    result: list[float] = []
    for _ in range(horizon):
        features = [1.0, *(transform(state[-lag]) for lag in lags)]
        estimate = sum(left * right for left, right in zip(coefficients, features, strict=True))
        estimate = math.expm1(estimate) if log else estimate
        state.append(max(0.0, estimate))
        result.append(state[-1])
    return result


def autoregression_ridge(values: list[float], horizon: int, season: int) -> list[float]:
    return _ar_forecast(values, horizon, season, 3)


def log_autoregression(values: list[float], horizon: int, season: int) -> list[float]:
    return _ar_forecast(values, horizon, season, 3, True)


def auto_autoregression(values: list[float], horizon: int, season: int) -> list[float]:
    candidates = [1, 2, 3, 5]
    validation = min(6, max(2, len(values) // 5))
    scores: list[tuple[float, int]] = []
    for order in candidates:
        errors: list[float] = []
        for end in range(len(values) - validation, len(values)):
            train = values[:end]
            if len(train) < max(10, order + 5):
                continue
            errors.append(abs(values[end] - _ar_forecast(train, 1, season, order)[0]))
        if errors:
            scores.append((statistics.fmean(errors), order))
    order = min(scores)[1] if scores else 1
    return _ar_forecast(values, horizon, season, order)


def harmonic_ridge(values: list[float], horizon: int, season: int) -> list[float]:
    harmonics = 2 if season >= 12 else 1
    rows: list[list[float]] = []
    for index in range(len(values)):
        features = [1.0, float(index)]
        if season > 1:
            for harmonic in range(1, harmonics + 1):
                angle = 2.0 * math.pi * harmonic * index / season
                features.extend([math.sin(angle), math.cos(angle)])
        rows.append(features)
    coefficients = _ridge(rows, values, 0.1)
    result: list[float] = []
    for index in range(len(values), len(values) + horizon):
        features = [1.0, float(index)]
        if season > 1:
            for harmonic in range(1, harmonics + 1):
                angle = 2.0 * math.pi * harmonic * index / season
                features.extend([math.sin(angle), math.cos(angle)])
        result.append(sum(left * right for left, right in zip(coefficients, features, strict=True)))
    return _clip(result)


def local_linear(values: list[float], horizon: int, season: int) -> list[float]:
    width = min(len(values), max(8, min(2 * season, 24)))
    recent = values[-width:]
    weights = [(index + 1) / width for index in range(width)]
    x_mean = sum(weight * index for index, weight in enumerate(weights)) / sum(weights)
    y_mean = sum(weight * value for weight, value in zip(weights, recent, strict=True)) / sum(weights)
    denominator = sum(weight * (index - x_mean) ** 2 for index, weight in enumerate(weights))
    slope = sum(weight * (index - x_mean) * (value - y_mean) for index, (weight, value) in enumerate(zip(weights, recent, strict=True))) / max(denominator, 1e-9)
    return _clip([y_mean + slope * (width - x_mean + step) for step in range(horizon)])


def _ingarch(values: list[float], horizon: int, season: int, negative_binomial: bool) -> list[float]:
    mean = max(statistics.fmean(values), 1e-6)
    variance = statistics.pvariance(values) if len(values) > 1 else mean
    dispersion = max(1.0, mean * mean / max(variance - mean, 1e-6))
    candidates: list[tuple[float, float, float, float, float]] = []
    for alpha in (0.1, 0.3, 0.5, 0.7):
        for beta in (0.1, 0.3, 0.5):
            if alpha + beta >= 0.95:
                continue
            omega = mean * (1.0 - alpha - beta)
            level = mean
            loss = 0.0
            for index, value in enumerate(values[1:], start=1):
                level = max(1e-6, omega + alpha * values[index - 1] + beta * level)
                if negative_binomial:
                    loss += -(math.lgamma(value + dispersion) - math.lgamma(dispersion) - math.lgamma(value + 1.0) + dispersion * math.log(dispersion / (dispersion + level)) + value * math.log(level / (dispersion + level)))
                else:
                    loss += level - value * math.log(level) + math.lgamma(value + 1.0)
            candidates.append((loss, alpha, beta, omega, level))
    _, alpha, beta, omega, level = min(candidates)
    state = values[-1]
    result: list[float] = []
    for _ in range(horizon):
        level = max(0.0, omega + alpha * state + beta * level)
        result.append(level)
        state = level
    return result


def poisson_ingarch(values: list[float], horizon: int, season: int) -> list[float]:
    return _ingarch(values, horizon, season, False)


def negative_binomial_ingarch(values: list[float], horizon: int, season: int) -> list[float]:
    return _ingarch(values, horizon, season, True)


def hurdle_occurrence(values: list[float], horizon: int, season: int) -> list[float]:
    alpha = 0.2
    probability = statistics.fmean(value > 0 for value in values[: min(6, len(values))])
    size = statistics.fmean([value for value in values if value > 0] or [0.0])
    for value in values:
        occurrence = float(value > 0)
        probability += alpha * (occurrence - probability)
        if occurrence:
            size += alpha * (value - size)
    return [max(0.0, probability * size)] * horizon


def markov_switching(values: list[float], horizon: int, season: int) -> list[float]:
    threshold = statistics.median(values)
    states = [int(value > threshold) for value in values]
    transitions = [[1.0, 1.0], [1.0, 1.0]]
    for left, right in pairwise(states):
        transitions[left][right] += 1.0
    probabilities = [row[1] / sum(row) for row in transitions]
    low = statistics.fmean([value for value in values if value <= threshold] or [threshold])
    high = statistics.fmean([value for value in values if value > threshold] or [threshold])
    current = float(states[-1])
    result: list[float] = []
    for _ in range(horizon):
        current = (1.0 - current) * probabilities[0] + current * probabilities[1]
        result.append((1.0 - current) * low + current * high)
    return _clip(result)


def croston_classic(values: list[float], horizon: int, season: int, alpha: float = 0.1) -> list[float]:
    positive = [(index, value) for index, value in enumerate(values) if value > 0]
    if not positive:
        return [0.0] * horizon
    size = positive[0][1]
    interval = max(1.0, positive[0][0] + 1.0)
    previous = positive[0][0]
    for index, value in positive[1:]:
        size += alpha * (value - size)
        interval += alpha * (index - previous - interval)
        previous = index
    return [max(0.0, size / max(interval, 1e-9))] * horizon


def croston_sba(values: list[float], horizon: int, season: int) -> list[float]:
    base = croston_classic(values, horizon, season, 0.1)[0]
    return [0.95 * base] * horizon


def croston_optimized(values: list[float], horizon: int, season: int) -> list[float]:
    validation = min(8, max(3, len(values) // 4))
    candidates: list[tuple[float, float]] = []
    for alpha in (0.05, 0.1, 0.2, 0.3, 0.5):
        errors = [abs(values[index] - croston_classic(values[:index], 1, season, alpha)[0]) for index in range(max(2, len(values) - validation), len(values))]
        candidates.append((statistics.fmean(errors), alpha))
    return croston_classic(values, horizon, season, min(candidates)[1])


def tsb(values: list[float], horizon: int, season: int) -> list[float]:
    alpha = 0.1
    beta = 0.1
    positives = [value for value in values if value > 0]
    if not positives:
        return [0.0] * horizon
    size = positives[0]
    probability = float(values[0] > 0)
    for value in values[1:]:
        occurrence = float(value > 0)
        probability += beta * (occurrence - probability)
        if occurrence:
            size += alpha * (value - size)
    return [max(0.0, probability * size)] * horizon


def adida(values: list[float], horizon: int, season: int, level: int = 3) -> list[float]:
    width = max(1, min(level, len(values) // 4))
    aggregates = [sum(values[index : index + width]) for index in range(0, len(values) - width + 1, width)]
    estimate = ses_optimized(aggregates, 1, max(1, season // width))[0] / width
    return [max(0.0, estimate)] * horizon


def imapa(values: list[float], horizon: int, season: int) -> list[float]:
    levels = range(1, min(6, max(2, len(values) // 6)) + 1)
    estimate = statistics.fmean(adida(values, 1, season, level)[0] for level in levels)
    return [max(0.0, estimate)] * horizon


def analog_knn(values: list[float], horizon: int, season: int) -> list[float]:
    dimension = min(6, max(2, len(values) // 8))
    query = values[-dimension:]
    scale = max(statistics.pstdev(values), 1.0)
    candidates: list[tuple[float, int]] = []
    for end in range(dimension, len(values) - horizon + 1):
        pattern = values[end - dimension : end]
        distance = math.sqrt(statistics.fmean(((left - right) / scale) ** 2 for left, right in zip(query, pattern, strict=True)))
        candidates.append((distance, end))
    if not candidates:
        return seasonal_naive(values, horizon, season)
    selected = sorted(candidates)[: min(7, len(candidates))]
    return _clip([
        sum(values[end + step] / max(distance, 1e-6) for distance, end in selected) / sum(1.0 / max(distance, 1e-6) for distance, _ in selected)
        for step in range(horizon)
    ])


def kernel_ridge_rbf(values: list[float], horizon: int, season: int) -> list[float]:
    dimension = min(5, max(2, len(values) // 10))
    rows = [values[index - dimension : index] for index in range(dimension, len(values))]
    target = values[dimension:]
    if len(rows) < 8:
        return analog_knn(values, horizon, season)
    step = max(1, len(rows) // 12)
    centers = rows[::step][-12:]
    scale = max(statistics.pstdev(values), 1.0)
    features = [[1.0, *(math.exp(-statistics.fmean(((left - right) / scale) ** 2 for left, right in zip(row, center, strict=True))) for center in centers)] for row in rows]
    coefficients = _ridge(features, target, 0.5)
    state = list(values)
    result: list[float] = []
    for _ in range(horizon):
        row = state[-dimension:]
        feature = [1.0, *(math.exp(-statistics.fmean(((left - right) / scale) ** 2 for left, right in zip(row, center, strict=True))) for center in centers)]
        estimate = max(0.0, sum(left * right for left, right in zip(coefficients, feature, strict=True)))
        state.append(estimate)
        result.append(estimate)
    return result


def _stump(rows: list[list[float]], target: list[float]) -> tuple[int, float, float, float]:
    best = (math.inf, 0, 0.0, statistics.fmean(target), statistics.fmean(target))
    for feature in range(len(rows[0])):
        values = sorted(row[feature] for row in rows)
        for position in (len(values) // 4, len(values) // 2, 3 * len(values) // 4):
            threshold = values[position]
            left = [value for row, value in zip(rows, target, strict=True) if row[feature] <= threshold]
            right = [value for row, value in zip(rows, target, strict=True) if row[feature] > threshold]
            if not left or not right:
                continue
            left_mean = statistics.fmean(left)
            right_mean = statistics.fmean(right)
            loss = sum((value - left_mean) ** 2 for value in left) + sum((value - right_mean) ** 2 for value in right)
            best = min(best, (loss, feature, threshold, left_mean, right_mean))
    _, feature, threshold, left_mean, right_mean = best
    return feature, threshold, left_mean, right_mean


def gradient_boosted_stumps(values: list[float], horizon: int, season: int) -> list[float]:
    dimension = min(5, max(2, len(values) // 10))
    rows = [values[index - dimension : index] for index in range(dimension, len(values))]
    target = values[dimension:]
    if len(rows) < 8:
        return moving_average(values, horizon, season)
    base = statistics.fmean(target)
    predictions = [base] * len(target)
    stumps: list[tuple[int, float, float, float]] = []
    for _ in range(12):
        residuals = [actual - predicted for actual, predicted in zip(target, predictions, strict=True)]
        stump = _stump(rows, residuals)
        stumps.append(stump)
        feature, threshold, left, right = stump
        predictions = [predicted + 0.15 * (left if row[feature] <= threshold else right) for row, predicted in zip(rows, predictions, strict=True)]
    state = list(values)
    result: list[float] = []
    for _ in range(horizon):
        row = state[-dimension:]
        estimate = base + sum(0.15 * (left if row[feature] <= threshold else right) for feature, threshold, left, right in stumps)
        state.append(max(0.0, estimate))
        result.append(state[-1])
    return result


def extra_trees(values: list[float], horizon: int, season: int) -> list[float]:
    dimension = min(5, max(2, len(values) // 10))
    rows = [values[index - dimension : index] for index in range(dimension, len(values))]
    target = values[dimension:]
    if len(rows) < 8:
        return analog_knn(values, horizon, season)
    generator = random.Random(len(values) * 7919 + round(sum(values)))
    state = list(values)
    result: list[float] = []
    for _ in range(horizon):
        query = state[-dimension:]
        estimates: list[float] = []
        for _ in range(11):
            feature = generator.randrange(dimension)
            threshold = generator.choice([row[feature] for row in rows])
            side = query[feature] <= threshold
            candidates = [value for row, value in zip(rows, target, strict=True) if (row[feature] <= threshold) == side]
            estimates.append(statistics.fmean(candidates or target))
        estimate = max(0.0, statistics.median(estimates))
        state.append(estimate)
        result.append(estimate)
    return result


def haar_multiscale(values: list[float], horizon: int, season: int) -> list[float]:
    widths = sorted({1, 2, 4, 8, min(12, season), min(24, len(values))})
    estimates = [statistics.fmean(values[-width:]) for width in widths if width <= len(values)]
    seasonal = seasonal_naive(values, horizon, season)
    level = statistics.median(estimates)
    return _clip([(level + seasonal[index]) / 2.0 for index in range(horizon)])


FORECASTERS: dict[str, Forecast] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "historical_median": historical_median,
    "moving_average": moving_average,
    "seasonal_window_average": seasonal_window_average,
    "ses_optimized": ses_optimized,
    "holt_linear": holt_linear,
    "holt_damped": holt_damped,
    "holt_winters_additive": holt_winters_additive,
    "theta": theta,
    "autoregression_ridge": autoregression_ridge,
    "log_autoregression": log_autoregression,
    "auto_autoregression": auto_autoregression,
    "harmonic_ridge": harmonic_ridge,
    "local_linear": local_linear,
    "poisson_ingarch": poisson_ingarch,
    "negative_binomial_ingarch": negative_binomial_ingarch,
    "hurdle_occurrence": hurdle_occurrence,
    "markov_switching": markov_switching,
    "croston_classic": croston_classic,
    "croston_sba": croston_sba,
    "croston_optimized": croston_optimized,
    "tsb": tsb,
    "adida": adida,
    "imapa": imapa,
    "analog_knn": analog_knn,
    "kernel_ridge_rbf": kernel_ridge_rbf,
    "gradient_boosted_stumps": gradient_boosted_stumps,
    "extra_trees": extra_trees,
    "haar_multiscale": haar_multiscale,
}


def _z_probability(value: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, 0.5 * math.erfc(-value / math.sqrt(2.0))))


def ears_c1(values: list[float], season: int) -> float:
    baseline = values[-8:-1] if len(values) >= 8 else values[:-1]
    if not baseline:
        return 0.5
    z = (values[-1] - statistics.fmean(baseline)) / max(statistics.pstdev(baseline), 1.0)
    return _z_probability(z - 3.0)


def ears_c2(values: list[float], season: int) -> float:
    baseline = values[-10:-3] if len(values) >= 10 else values[:-3]
    if not baseline:
        return ears_c1(values, season)
    z = (values[-1] - statistics.fmean(baseline)) / max(statistics.pstdev(baseline), 1.0)
    return _z_probability(z - 3.0)


def ears_c3(values: list[float], season: int) -> float:
    scores = [ears_c2(values[:end], season) for end in range(max(4, len(values) - 2), len(values) + 1)]
    return min(1.0 - 1e-9, sum(scores) / len(scores))


def ewma_alarm(values: list[float], season: int) -> float:
    level = values[0]
    residuals: list[float] = []
    for value in values[1:]:
        residuals.append(value - level)
        level += 0.2 * (value - level)
    z = (values[-1] - level) / max(statistics.pstdev(residuals) if len(residuals) > 1 else 1.0, 1.0)
    return _z_probability(z - 2.0)


def cusum_alarm(values: list[float], season: int) -> float:
    baseline = statistics.median(values)
    scale = max(statistics.pstdev(values), 1.0)
    score = 0.0
    for value in values[-min(12, len(values)):]:
        score = max(0.0, score + (value - baseline) / scale - 0.5)
    return _z_probability(score - 4.0)


def farrington_alarm(values: list[float], season: int) -> float:
    if season <= 1 or len(values) < season + 2:
        baseline = values[:-1]
    else:
        phase = (len(values) - 1) % season
        baseline = [value for index, value in enumerate(values[:-1]) if index % season == phase]
    baseline = baseline or values[:-1]
    mean = statistics.fmean(baseline)
    variance = statistics.pvariance(baseline) if len(baseline) > 1 else mean
    z = (values[-1] - mean) / math.sqrt(max(variance, mean, 1.0))
    return _z_probability(z - 2.5)


def bayesian_change(values: list[float], season: int) -> float:
    width = min(4, max(2, len(values) // 5))
    recent = statistics.fmean(values[-width:])
    previous = statistics.fmean(values[-2 * width : -width]) if len(values) >= 2 * width else statistics.fmean(values[:-width] or values)
    scale = math.sqrt(max(previous, 1.0) / width)
    return _z_probability((recent - previous) / scale - 2.0)


SURVEILLANCE: dict[str, Surveillance] = {
    "ears_c1": ears_c1,
    "ears_c2": ears_c2,
    "ears_c3": ears_c3,
    "ewma_alarm": ewma_alarm,
    "cusum_alarm": cusum_alarm,
    "farrington_alarm": farrington_alarm,
    "bayesian_change": bayesian_change,
}


SPECS = [
    *(ModelSpec(model_id, family, ("forecast", "outbreak", "risk"), intermittent) for model_id, family, intermittent in [
        ("naive", "baseline", False),
        ("seasonal_naive", "baseline", False),
        ("historical_median", "level", False),
        ("moving_average", "level", False),
        ("seasonal_window_average", "seasonal_level", False),
        ("ses_optimized", "exponential_smoothing", False),
        ("holt_linear", "exponential_smoothing", False),
        ("holt_damped", "exponential_smoothing", False),
        ("holt_winters_additive", "seasonal_exponential_smoothing", False),
        ("theta", "theta", False),
        ("autoregression_ridge", "autoregression", False),
        ("log_autoregression", "count_transform", False),
        ("auto_autoregression", "nested_autoregression", False),
        ("harmonic_ridge", "dynamic_harmonic_regression", False),
        ("local_linear", "local_regression", False),
        ("poisson_ingarch", "count_time_series", False),
        ("negative_binomial_ingarch", "count_time_series", False),
        ("hurdle_occurrence", "hurdle", True),
        ("markov_switching", "regime_switching", False),
        ("croston_classic", "intermittent", True),
        ("croston_sba", "intermittent", True),
        ("croston_optimized", "intermittent", True),
        ("tsb", "intermittent", True),
        ("adida", "intermittent_aggregation", True),
        ("imapa", "intermittent_aggregation", True),
        ("analog_knn", "nonlinear_analog", False),
        ("kernel_ridge_rbf", "kernel_machine", False),
        ("gradient_boosted_stumps", "machine_learning", False),
        ("extra_trees", "machine_learning", False),
        ("haar_multiscale", "multiscale", False),
    ]),
    ModelSpec("global_pooled_ridge", "global_target_only", ("forecast", "outbreak", "risk"), pooled=True),
    ModelSpec("median_ensemble", "ensemble", ("forecast", "outbreak", "risk")),
    ModelSpec("trimmed_mean_ensemble", "ensemble", ("forecast", "outbreak", "risk")),
    ModelSpec("online_selector", "adaptive_selection", ("forecast", "outbreak", "risk")),
    ModelSpec("online_weighted_ensemble", "adaptive_ensemble", ("forecast", "outbreak", "risk")),
    *(ModelSpec(model_id, "surveillance", ("outbreak",)) for model_id in SURVEILLANCE),
]


SPEC_BY_ID = {spec.model_id: spec for spec in SPECS}
