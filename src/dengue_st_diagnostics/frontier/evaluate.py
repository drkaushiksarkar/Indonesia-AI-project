from __future__ import annotations

import datetime as dt
import hashlib
import math
import statistics
from collections import defaultdict
from typing import Any

from dengue_st_diagnostics.frontier.metrics import (
    binary_metrics,
    block_bootstrap_skill,
    crps_quantile,
    mase_scale,
    poisson_deviance,
    poisson_log_score,
    quantile,
    risk_metrics,
    smape,
    weighted_interval_score,
)
from dengue_st_diagnostics.frontier.models import (
    FORECASTERS,
    SPEC_BY_ID,
    SPECS,
    SURVEILLANCE,
    _ridge,
    seasonal_naive,
)
from dengue_st_diagnostics.predictability.io import regularize
from dengue_st_diagnostics.predictability.metrics import intermittency

GRAIN_SETTINGS = {
    "week": {"season": 52, "minimum_train": 52, "horizons": [1, 2, 4, 8, 13], "maximum_origins": 12},
    "month": {"season": 12, "minimum_train": 18, "horizons": [1, 2, 3, 6, 12], "maximum_origins": 12},
    "year": {"season": 1, "minimum_train": 5, "horizons": [1, 2, 3], "maximum_origins": 8},
}

PROBABILITIES = (0.1, 0.25, 0.5, 0.75, 0.9)
NORMAL_MULTIPLIERS = (-1.2815515655, -0.6744897502, 0.0, 0.6744897502, 1.2815515655)


def _sample_origins(origins: list[int], maximum: int) -> list[int]:
    if len(origins) <= maximum:
        return origins
    positions = {round(index * (len(origins) - 1) / (maximum - 1)) for index in range(maximum)}
    return [origins[index] for index in sorted(positions)]


def _threshold(train: list[float], target_index: int, season: int) -> float:
    if season > 1:
        seasonal = [value for index, value in enumerate(train) if index % season == target_index % season]
        if len(seasonal) >= 3:
            return statistics.fmean(seasonal) + 2.0 * statistics.pstdev(seasonal)
    return quantile(train, 0.9)


def _risk_category(value: float, lower: float, upper: float) -> int:
    return 0 if value <= lower else 1 if value <= upper else 2


def _fallback_sigma(train: list[float], season: int) -> float:
    lag = season if season > 1 and len(train) > season else 1
    residuals = [train[index] - train[index - lag] for index in range(lag, len(train))]
    sigma = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0
    return max(sigma, math.sqrt(max(statistics.fmean(train), 0.0) + 1.0), 1.0)


def _distribution(
    predicted: float,
    residuals: list[float],
    train: list[float],
    season: int,
    threshold: float,
    lower: float,
    upper: float,
) -> tuple[list[float], float, tuple[float, float, float]]:
    if len(residuals) >= 5:
        values = sorted(max(0.0, predicted + quantile(residuals, probability)) for probability in PROBABILITIES)
        simulated = [max(0.0, predicted + value) for value in residuals]
    else:
        sigma = _fallback_sigma(train, season)
        values = sorted(max(0.0, predicted + multiplier * sigma) for multiplier in NORMAL_MULTIPLIERS)
        simulated = [max(0.0, predicted + multiplier * sigma) for multiplier in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)]
    outbreak = min(1.0 - 1e-9, max(1e-9, statistics.fmean(value > threshold for value in simulated)))
    risk = (
        statistics.fmean(value <= lower for value in simulated),
        statistics.fmean(lower < value <= upper for value in simulated),
        statistics.fmean(value > upper for value in simulated),
    )
    total = sum(risk)
    risk = tuple(max(1e-9, value / max(total, 1e-9)) for value in risk)
    normalization = sum(risk)
    return values, outbreak, tuple(value / normalization for value in risk)


def _global_features(values: list[float], index: int, season: int) -> list[float]:
    scale = max(statistics.fmean(values[max(0, index - max(season, 6)) : index]), 1.0)
    features = [1.0, values[index - 1] / scale, values[index - 2] / scale, values[index - 3] / scale]
    if season > 1 and index >= season:
        features.append(values[index - season] / scale)
        angle = 2.0 * math.pi * index / season
        features.extend([math.sin(angle), math.cos(angle)])
    return features


def _global_model(
    contexts: dict[str, dict[str, Any]],
    disease: str,
    grain: str,
    horizon: int,
    cutoff: dt.date,
    cache: dict[tuple[str, str, int, dt.date], list[float]],
) -> list[float]:
    key = (disease, grain, horizon, cutoff)
    if key in cache:
        return cache[key]
    season = GRAIN_SETTINGS[grain]["season"]
    rows: list[list[float]] = []
    target: list[float] = []
    for context in contexts.values():
        if context["disease"] != disease or context["grain"] != grain:
            continue
        dates = context["dates"]
        values = context["values"]
        start = max(3, season if season > 1 else 3)
        for index in range(start, len(values) - horizon + 1):
            target_index = index + horizon - 1
            if dates[target_index] > cutoff:
                break
            scale = max(statistics.fmean(values[max(0, index - max(season, 6)) : index]), 1.0)
            rows.append(_global_features(values, index, season))
            target.append(values[target_index] / scale)
    if not rows:
        cache[key] = []
        return []
    maximum = 800
    if len(rows) > maximum:
        positions = {round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum)}
        rows = [rows[index] for index in sorted(positions)]
        target = [target[index] for index in sorted(positions)]
    cache[key] = _ridge(rows, target, 1.0)
    return cache[key]


def _global_predict(
    contexts: dict[str, dict[str, Any]],
    series_id: str,
    train: list[float],
    horizon: int,
    cutoff: dt.date,
    cache: dict[tuple[str, str, int, dt.date], list[float]],
) -> float:
    context = contexts[series_id]
    season = GRAIN_SETTINGS[context["grain"]]["season"]
    coefficients = _global_model(contexts, context["disease"], context["grain"], horizon, cutoff, cache)
    if not coefficients:
        return seasonal_naive(train, horizon, season)[horizon - 1]
    scale = max(statistics.fmean(train[-max(season, 6) :]), 1.0)
    features = _global_features(train, len(train), season)
    return max(0.0, scale * sum(left * right for left, right in zip(coefficients, features, strict=True)))


def _row(
    common: dict[str, Any],
    model_id: str,
    horizon: int,
    origin: int,
    dates: list[dt.date],
    actual: float,
    predicted: float,
    quantiles: list[float],
    outbreak_threshold: float,
    outbreak_probability: float,
    risk_lower: float,
    risk_upper: float,
    risk_probabilities: tuple[float, float, float],
    mase: float,
) -> dict[str, Any]:
    q10, q25, q50, q75, q90 = quantiles
    error = actual - predicted
    return {
        **common,
        "model_id": model_id,
        "model_family": SPEC_BY_ID[model_id].family,
        "model_scopes": ";".join(SPEC_BY_ID[model_id].scopes),
        "horizon": horizon,
        "origin_index": origin,
        "training_periods": origin,
        "train_start": dates[0].isoformat(),
        "train_end": dates[origin - 1].isoformat(),
        "target_period": dates[origin + horizon - 1].isoformat(),
        "actual": actual,
        "predicted": predicted,
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "absolute_error": abs(error),
        "squared_error": error**2,
        "smape": smape(actual, predicted),
        "mase_scale": mase,
        "poisson_deviance": poisson_deviance(actual, predicted),
        "poisson_log_score": poisson_log_score(actual, predicted),
        "weighted_interval_score": weighted_interval_score(actual, q10, q25, q50, q75, q90),
        "crps_quantile": crps_quantile(actual, list(zip(PROBABILITIES, quantiles, strict=True))),
        "coverage_50": int(q25 <= actual <= q75),
        "coverage_80": int(q10 <= actual <= q90),
        "width_50": q75 - q25,
        "width_80": q90 - q10,
        "outbreak_threshold": outbreak_threshold,
        "outbreak_actual": int(actual > outbreak_threshold),
        "outbreak_probability": outbreak_probability,
        "risk_lower": risk_lower,
        "risk_upper": risk_upper,
        "risk_actual": _risk_category(actual, risk_lower, risk_upper),
        "risk_probability_low": risk_probabilities[0],
        "risk_probability_medium": risk_probabilities[1],
        "risk_probability_high": risk_probabilities[2],
    }


def _contexts(series: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in series:
        dates, values, present = regularize(value)
        result[value["series_id"]] = {
            "dates": dates,
            "values": values,
            "present": present,
            "disease": value["disease"],
            "grain": value["temporal_resolution"],
        }
    return result


def _base_rows(
    series: list[dict[str, Any]], run_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    contexts = _contexts(series)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    residual_history: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    global_cache: dict[tuple[str, str, int, dt.date], list[float]] = {}
    for value in series:
        series_id = value["series_id"]
        context = contexts[series_id]
        dates = context["dates"]
        values = context["values"]
        present = context["present"]
        settings = GRAIN_SETTINGS[value["temporal_resolution"]]
        season = settings["season"]
        observed = [item for item, available in zip(values, present, strict=True) if available]
        zero_rate, adi, cv2, demand_class = intermittency(observed)
        common = {
            "frontier_run_id": run_id,
            "series_id": series_id,
            "source_id": value["source_id"],
            "source_dataset_name": value["source_dataset_name"],
            "disease": value["disease"],
            "metric": value["metric"],
            "spatial_resolution": value["spatial_resolution"],
            "temporal_resolution": value["temporal_resolution"],
            "location_name": value["location_name"],
        }
        maximum_fold_count = 0
        for horizon in settings["horizons"]:
            origins = [origin for origin in range(settings["minimum_train"], len(values) - horizon + 1) if present[origin + horizon - 1]]
            origins = _sample_origins(origins, settings["maximum_origins"])
            maximum_fold_count = max(maximum_fold_count, len(origins))
            for origin in origins:
                train = values[:origin]
                actual = values[origin + horizon - 1]
                threshold = _threshold(train, origin + horizon - 1, season)
                lower = quantile(train, 1.0 / 3.0)
                upper = quantile(train, 2.0 / 3.0)
                scale = mase_scale(train, season)
                model_ids = list(FORECASTERS)
                if zero_rate < 0.2:
                    model_ids = [model_id for model_id in model_ids if not SPEC_BY_ID[model_id].intermittent_only]
                predictions = {model_id: FORECASTERS[model_id](train, horizon, season)[horizon - 1] for model_id in model_ids}
                predictions["global_pooled_ridge"] = _global_predict(contexts, series_id, train, horizon, dates[origin - 1], global_cache)
                prediction_rows: dict[str, dict[str, Any]] = {}
                for model_id, predicted in predictions.items():
                    history = residual_history[(series_id, model_id, horizon)]
                    quantiles, outbreak_probability, risk_probabilities = _distribution(predicted, history, train, season, threshold, lower, upper)
                    prediction_rows[model_id] = _row(common, model_id, horizon, origin, dates, actual, predicted, quantiles, threshold, outbreak_probability, lower, upper, risk_probabilities, scale)
                    rows.append(prediction_rows[model_id])
                    history.append(actual - predicted)
                baseline_prediction = predictions.get("seasonal_naive", seasonal_naive(train, horizon, season)[horizon - 1])
                baseline_row = prediction_rows["seasonal_naive"]
                baseline_quantiles = [baseline_row[field] for field in ("q10", "q25", "q50", "q75", "q90")]
                baseline_risk = (baseline_row["risk_probability_low"], baseline_row["risk_probability_medium"], baseline_row["risk_probability_high"])
                for model_id, detector in SURVEILLANCE.items():
                    probability = detector(train, season)
                    rows.append(_row(common, model_id, horizon, origin, dates, actual, baseline_prediction, baseline_quantiles, threshold, probability, lower, upper, baseline_risk, scale))
        tier = "robust" if maximum_fold_count >= 12 else "moderate" if maximum_fold_count >= 6 else "pilot"
        records.append({
            **common,
            "period_count": len(values),
            "observed_count": sum(present),
            "missing_count": len(values) - sum(present),
            "first_period": dates[0].isoformat(),
            "last_period": dates[-1].isoformat(),
            "completeness": sum(present) / len(present),
            "zero_rate": zero_rate,
            "average_demand_interval": adi,
            "positive_cv_squared": cv2,
            "demand_class": demand_class,
            "fold_count": maximum_fold_count,
            "validation_tier": tier,
        })
    return records, rows, contexts


def _ensemble_row(template: dict[str, Any], model_id: str, candidates: list[dict[str, Any]], weights: list[float] | None = None) -> dict[str, Any]:
    if weights is None:
        weights = [1.0] * len(candidates)
    total = sum(weights)
    fields = ["predicted", "q10", "q25", "q50", "q75", "q90", "outbreak_probability", "risk_probability_low", "risk_probability_medium", "risk_probability_high"]
    values = {field: sum(weight * row[field] for weight, row in zip(weights, candidates, strict=True)) / total for field in fields}
    if model_id == "median_ensemble":
        values = {field: statistics.median(row[field] for row in candidates) for field in fields}
    actual = template["actual"]
    predicted = values["predicted"]
    q10, q25, q50, q75, q90 = sorted(values[field] for field in ("q10", "q25", "q50", "q75", "q90"))
    risk = [values["risk_probability_low"], values["risk_probability_medium"], values["risk_probability_high"]]
    risk_total = sum(risk)
    result = dict(template)
    result.update({
        "model_id": model_id,
        "model_family": SPEC_BY_ID[model_id].family,
        "model_scopes": ";".join(SPEC_BY_ID[model_id].scopes),
        "predicted": predicted,
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "absolute_error": abs(actual - predicted),
        "squared_error": (actual - predicted) ** 2,
        "smape": smape(actual, predicted),
        "poisson_deviance": poisson_deviance(actual, predicted),
        "poisson_log_score": poisson_log_score(actual, predicted),
        "weighted_interval_score": weighted_interval_score(actual, q10, q25, q50, q75, q90),
        "crps_quantile": crps_quantile(actual, list(zip(PROBABILITIES, (q10, q25, q50, q75, q90), strict=True))),
        "coverage_50": int(q25 <= actual <= q75),
        "coverage_80": int(q10 <= actual <= q90),
        "width_50": q75 - q25,
        "width_80": q90 - q10,
        "outbreak_probability": values["outbreak_probability"],
        "risk_probability_low": risk[0] / max(risk_total, 1e-9),
        "risk_probability_medium": risk[1] / max(risk_total, 1e-9),
        "risk_probability_high": risk[2] / max(risk_total, 1e-9),
    })
    return result


def _derived_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "forecast" in row["model_scopes"]:
            grouped[(row["series_id"], row["horizon"], row["origin_index"])].append(row)
    history: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    excluded = {"naive", "seasonal_naive", "historical_median"}
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        candidates = grouped[key]
        ensemble = [row for row in candidates if row["model_id"] not in excluded]
        ensemble = ensemble or candidates
        template = candidates[0]
        result.append(_ensemble_row(template, "median_ensemble", ensemble))
        trimmed = sorted(ensemble, key=lambda row: row["predicted"])
        trim = len(trimmed) // 10
        trimmed = trimmed[trim : len(trimmed) - trim] if trim else trimmed
        result.append(_ensemble_row(template, "trimmed_mean_ensemble", trimmed))
        series_id, horizon, _ = key
        eligible = [(statistics.fmean(history[(series_id, horizon, row["model_id"])]), row) for row in candidates if len(history[(series_id, horizon, row["model_id"])]) >= 2]
        selected = min(eligible, key=lambda item: item[0])[1] if eligible else next((row for row in candidates if row["model_id"] == "seasonal_naive"), candidates[0])
        selector = dict(selected)
        selector.update({"model_id": "online_selector", "model_family": SPEC_BY_ID["online_selector"].family, "model_scopes": ";".join(SPEC_BY_ID["online_selector"].scopes)})
        result.append(selector)
        weighted_candidates = [row for row in candidates if history[(series_id, horizon, row["model_id"])]]
        if weighted_candidates:
            weights = [1.0 / max(statistics.fmean(history[(series_id, horizon, row["model_id"])]), 0.05) for row in weighted_candidates]
            result.append(_ensemble_row(template, "online_weighted_ensemble", weighted_candidates, weights))
        else:
            fallback = dict(selector)
            fallback.update({"model_id": "online_weighted_ensemble", "model_family": SPEC_BY_ID["online_weighted_ensemble"].family})
            result.append(fallback)
        for row in candidates:
            history[(series_id, horizon, row["model_id"])].append(row["absolute_error"] / row["mase_scale"])
    return result


def _metric_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["series_id"], row["model_id"], row["horizon"])].append(row)
    result: list[dict[str, Any]] = []
    for (series_id, model_id, horizon), values in sorted(grouped.items()):
        scopes = SPEC_BY_ID[model_id].scopes
        tasks: list[tuple[str, dict[str, float]]] = []
        if "forecast" in scopes:
            point = {
                "mae": statistics.fmean(row["absolute_error"] for row in values),
                "rmse": math.sqrt(statistics.fmean(row["squared_error"] for row in values)),
                "mase": statistics.fmean(row["absolute_error"] / row["mase_scale"] for row in values),
                "smape": statistics.fmean(row["smape"] for row in values),
                "poisson_deviance": statistics.fmean(row["poisson_deviance"] for row in values),
                "poisson_log_score": statistics.fmean(row["poisson_log_score"] for row in values),
            }
            probabilistic = {
                "weighted_interval_score": statistics.fmean(row["weighted_interval_score"] for row in values),
                "crps_quantile": statistics.fmean(row["crps_quantile"] for row in values),
                "coverage_50": statistics.fmean(row["coverage_50"] for row in values),
                "coverage_80": statistics.fmean(row["coverage_80"] for row in values),
                "width_50": statistics.fmean(row["width_50"] for row in values),
                "width_80": statistics.fmean(row["width_80"] for row in values),
            }
            tasks.extend([("forecast_point", point), ("forecast_probabilistic", probabilistic)])
        if "outbreak" in scopes:
            tasks.append(("outbreak", binary_metrics([row["outbreak_actual"] for row in values], [row["outbreak_probability"] for row in values])))
        if "risk" in scopes:
            probabilities = [(row["risk_probability_low"], row["risk_probability_medium"], row["risk_probability_high"]) for row in values]
            tasks.append(("risk", risk_metrics([row["risk_actual"] for row in values], probabilities)))
        for task, measurements in tasks:
            for metric_name, metric_value in measurements.items():
                result.append({
                    "frontier_run_id": run_id,
                    "series_id": series_id,
                    "model_id": model_id,
                    "horizon": horizon,
                    "task": task,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "sample_size": len(values),
                    "event_count": sum(row["outbreak_actual"] for row in values),
                })
    return result


def _frontier_rows(rows: list[dict[str, Any]], metrics: list[dict[str, Any]], records: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    tiers = {row["series_id"]: row["validation_tier"] for row in records}
    index = {(row["series_id"], row["model_id"], row["horizon"], row["task"], row["metric_name"]): row for row in metrics if math.isfinite(row["metric_value"])}
    fold_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fold_groups[(row["series_id"], row["model_id"], row["horizon"])].append(row)
    definitions = (
        ("forecast_point", "mase", "min"),
        ("forecast_probabilistic", "weighted_interval_score", "min"),
        ("outbreak", "brier", "min"),
        ("risk", "ranked_probability_score", "min"),
    )
    result: list[dict[str, Any]] = []
    for series_id, horizon in sorted({(row["series_id"], row["horizon"]) for row in rows}):
        for task, criterion, direction in definitions:
            candidates = [(row["metric_value"], row["model_id"]) for row in metrics if row["series_id"] == series_id and row["horizon"] == horizon and row["task"] == task and row["metric_name"] == criterion and math.isfinite(row["metric_value"])]
            if not candidates:
                continue
            value, model_id = min(candidates) if direction == "min" else max(candidates)
            baseline_id = "seasonal_naive"
            baseline_key = (series_id, baseline_id, horizon, task, criterion)
            baseline_value = index[baseline_key]["metric_value"] if baseline_key in index else math.nan
            skill = 1.0 - value / max(baseline_value, 1e-9) if math.isfinite(baseline_value) else math.nan
            lower = math.nan
            upper = math.nan
            if task == "forecast_point" and (series_id, model_id, horizon) in fold_groups and (series_id, baseline_id, horizon) in fold_groups:
                selected = fold_groups[(series_id, model_id, horizon)]
                baseline = {row["origin_index"]: row for row in fold_groups[(series_id, baseline_id, horizon)]}
                aligned = [(row, baseline[row["origin_index"]]) for row in selected if row["origin_index"] in baseline]
                seed = int(hashlib.sha256(f"{series_id}|{horizon}|v2".encode()).hexdigest()[:8], 16)
                skill, lower, upper = block_bootstrap_skill([left["absolute_error"] for left, _ in aligned], [right["absolute_error"] for _, right in aligned], seed)
            selected_rows = fold_groups.get((series_id, model_id, horizon), [])
            result.append({
                "frontier_run_id": run_id,
                "series_id": series_id,
                "horizon": horizon,
                "task": task,
                "frontier_model_id": model_id,
                "criterion": criterion,
                "criterion_value": value,
                "baseline_model_id": baseline_id,
                "baseline_value": baseline_value,
                "skill": skill,
                "skill_lower_95": lower,
                "skill_upper_95": upper,
                "fold_count": len(selected_rows),
                "event_count": sum(row["outbreak_actual"] for row in selected_rows),
                "validation_tier": tiers[series_id],
            })
    return result


def _learning_curves(series: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    models = ["seasonal_naive", "holt_winters_additive", "auto_autoregression", "negative_binomial_ingarch", "kernel_ridge_rbf", "gradient_boosted_stumps", "haar_multiscale"]
    caps = [12, 24, 36, 60, 120]
    result: list[dict[str, Any]] = []
    for value in series:
        dates, values, present = regularize(value)
        settings = GRAIN_SETTINGS[value["temporal_resolution"]]
        season = settings["season"]
        origins = [origin for origin in range(settings["minimum_train"], len(values)) if present[origin]]
        origins = _sample_origins(origins, 8)
        for cap in caps:
            if cap < settings["minimum_train"]:
                continue
            for model_id in models:
                errors: list[float] = []
                baseline_errors: list[float] = []
                for origin in origins:
                    if origin < cap:
                        continue
                    train = values[origin - cap : origin]
                    actual = values[origin]
                    predicted = FORECASTERS[model_id](train, 1, season)[0]
                    baseline = seasonal_naive(train, 1, season)[0]
                    errors.append(abs(actual - predicted))
                    baseline_errors.append(abs(actual - baseline))
                if len(errors) >= 3:
                    result.append({
                        "frontier_run_id": run_id,
                        "series_id": value["series_id"],
                        "model_id": model_id,
                        "history_periods": cap,
                        "temporal_resolution": value["temporal_resolution"],
                        "disease": value["disease"],
                        "spatial_resolution": value["spatial_resolution"],
                        "origin_count": len(errors),
                        "mae": statistics.fmean(errors),
                        "baseline_mae": statistics.fmean(baseline_errors),
                        "mae_skill": 1.0 - statistics.fmean(errors) / max(statistics.fmean(baseline_errors), 1e-9),
                        "first_target": dates[origins[0]].isoformat(),
                        "last_target": dates[origins[-1]].isoformat(),
                    })
    return result


def _aggregate(metrics: list[dict[str, Any]], records: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    metadata = {row["series_id"]: row for row in records}
    grouped: dict[tuple[str, str, str, str, int, str, str], list[float]] = defaultdict(list)
    for row in metrics:
        value = row["metric_value"]
        if not math.isfinite(value):
            continue
        series = metadata[row["series_id"]]
        key = (series["disease"], series["spatial_resolution"], series["temporal_resolution"], row["model_id"], row["horizon"], row["task"], row["metric_name"])
        grouped[key].append(value)
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        disease, spatial, temporal, model_id, horizon, task, metric_name = key
        result.append({
            "frontier_run_id": run_id,
            "disease": disease,
            "spatial_resolution": spatial,
            "temporal_resolution": temporal,
            "model_id": model_id,
            "horizon": horizon,
            "task": task,
            "metric_name": metric_name,
            "median_value": statistics.median(values),
            "mean_value": statistics.fmean(values),
            "series_count": len(values),
            "q25": quantile(values, 0.25),
            "q75": quantile(values, 0.75),
        })
    return result


def evaluate(series: list[dict[str, Any]], run_id: str) -> dict[str, list[dict[str, Any]]]:
    records, rows, _ = _base_rows(series, run_id)
    rows.extend(_derived_rows(rows))
    metrics = _metric_rows(rows, run_id)
    frontiers = _frontier_rows(rows, metrics, records, run_id)
    learning = _learning_curves(series, run_id)
    aggregates = _aggregate(metrics, records, run_id)
    return {
        "series": records,
        "folds": rows,
        "metrics": metrics,
        "frontiers": frontiers,
        "learning_curves": learning,
        "aggregates": aggregates,
    }


MODEL_COUNT = len(SPECS)
