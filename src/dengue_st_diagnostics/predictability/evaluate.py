from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from typing import Any

from dengue_st_diagnostics.predictability.io import inclusion_reason, regularize
from dengue_st_diagnostics.predictability.metrics import (
    binary_metrics,
    bootstrap_skill_interval,
    intermittency,
    mase_scale,
    multiclass_metrics,
    mutual_information,
    normal_probability_above,
    permutation_entropy,
    poisson_deviance,
    quantile,
    rmsse_scale,
    smape,
    spectral_entropy,
)
from dengue_st_diagnostics.predictability.models import FORECASTERS

GRAIN_SETTINGS = {
    "week": {
        "season": 52,
        "minimum_train": 52,
        "horizons": [1, 2, 4, 8, 13],
        "maximum_origins": 24,
    },
    "month": {
        "season": 12,
        "minimum_train": 18,
        "horizons": [1, 2, 3, 6, 12],
        "maximum_origins": 24,
    },
    "year": {"season": 1, "minimum_train": 5, "horizons": [1, 2, 3], "maximum_origins": 10},
}


def _sample_origins(origins: list[int], maximum: int) -> list[int]:
    if len(origins) <= maximum:
        return origins
    positions = {round(index * (len(origins) - 1) / (maximum - 1)) for index in range(maximum)}
    return [origins[index] for index in sorted(positions)]


def _threshold(train: list[float], target_index: int, season: int) -> float:
    if season > 1:
        seasonal = [
            value for index, value in enumerate(train) if index % season == target_index % season
        ]
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


def _base_fold_rows(
    series: dict[str, Any], run_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    included, reason = inclusion_reason(series)
    common = {
        "predictability_run_id": run_id,
        "series_id": series["series_id"],
        "source_id": series["source_id"],
        "source_dataset_name": series["source_dataset_name"],
        "upstream_source": series["upstream_source"],
        "disease": series["disease"],
        "metric": series["metric"],
        "unit": series["unit"],
        "spatial_resolution": series["spatial_resolution"],
        "temporal_resolution": series["temporal_resolution"],
        "location_name": series["location_name"],
    }
    if not included:
        return {
            **common,
            "include_status": "excluded",
            "exclusion_reason": reason,
            "period_count": series["expected_periods"],
            "observed_count": series["observations"],
            "missing_count": max(0, series["expected_periods"] - series["observations"]),
            "first_period": series["first_period"],
            "last_period": series["last_period"],
            "completeness": series["completeness"],
            "zero_rate": math.nan,
            "average_demand_interval": math.nan,
            "positive_cv_squared": math.nan,
            "demand_class": "not_assessed",
            "permutation_entropy_3": math.nan,
            "permutation_entropy_4": math.nan,
            "spectral_entropy": math.nan,
            "mutual_information_lag1": math.nan,
            "mutual_information_season": math.nan,
            "entropy_predictability": math.nan,
            "evaluated_horizons": "",
            "fold_count": 0,
            "validation_tier": "excluded",
        }, []
    dates, values, present = regularize(series)
    settings = GRAIN_SETTINGS[series["temporal_resolution"]]
    season = settings["season"]
    zero_rate, adi, cv2, demand_class = intermittency(
        [value for value, available in zip(values, present, strict=True) if available]
    )
    fold_rows: list[dict[str, Any]] = []
    residual_history: dict[tuple[str, int], list[float]] = defaultdict(list)
    evaluated_horizons: list[int] = []
    for horizon in settings["horizons"]:
        origins = [
            origin
            for origin in range(settings["minimum_train"], len(values) - horizon + 1)
            if present[origin + horizon - 1]
        ]
        origins = _sample_origins(origins, settings["maximum_origins"])
        if len(origins) < 2:
            continue
        evaluated_horizons.append(horizon)
        for origin in origins:
            train = values[:origin]
            target_index = origin + horizon - 1
            actual = values[target_index]
            threshold = _threshold(train, target_index, season)
            lower = quantile(train, 1.0 / 3.0)
            upper = quantile(train, 2.0 / 3.0)
            scale_mase = mase_scale(train, season)
            scale_rmsse = rmsse_scale(train, season)
            model_ids = list(FORECASTERS)
            if zero_rate < 0.2:
                model_ids = [
                    model_id for model_id in model_ids if model_id not in {"croston_sba", "tsb"}
                ]
            for model_id in model_ids:
                predicted = FORECASTERS[model_id](train, horizon, season)[horizon - 1]
                previous_errors = residual_history[(model_id, horizon)]
                sigma = (
                    statistics.pstdev(previous_errors)
                    if len(previous_errors) >= 3
                    else _fallback_sigma(train, season)
                )
                probability = normal_probability_above(threshold, predicted, sigma)
                error = actual - predicted
                row = {
                    **common,
                    "model_id": model_id,
                    "horizon": horizon,
                    "origin_index": origin,
                    "train_start": dates[0].isoformat(),
                    "train_end": dates[origin - 1].isoformat(),
                    "target_period": dates[target_index].isoformat(),
                    "actual": actual,
                    "predicted": predicted,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "smape": smape(actual, predicted),
                    "poisson_deviance": poisson_deviance(actual, predicted),
                    "mase_scale": scale_mase,
                    "rmsse_scale": scale_rmsse,
                    "outbreak_threshold": threshold,
                    "outbreak_actual": int(actual > threshold),
                    "outbreak_probability": probability,
                    "risk_lower": lower,
                    "risk_upper": upper,
                    "risk_actual": _risk_category(actual, lower, upper),
                    "risk_predicted": _risk_category(predicted, lower, upper),
                }
                fold_rows.append(row)
                previous_errors.append(error)
    maximum_folds = max(
        (
            sum(row["model_id"] == "naive" and row["horizon"] == horizon for row in fold_rows)
            for horizon in evaluated_horizons
        ),
        default=0,
    )
    validation_tier = (
        "robust"
        if maximum_folds >= 12
        else "moderate"
        if maximum_folds >= 6
        else "pilot"
        if maximum_folds >= 2
        else "insufficient"
    )
    record = {
        **common,
        "include_status": "evaluated" if fold_rows else "insufficient",
        "exclusion_reason": "" if fold_rows else "insufficient_history_for_rolling_origin",
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
        "permutation_entropy_3": permutation_entropy(values, 3),
        "permutation_entropy_4": permutation_entropy(values, 4),
        "spectral_entropy": spectral_entropy(values),
        "mutual_information_lag1": mutual_information(values, 1),
        "mutual_information_season": mutual_information(values, season),
        "entropy_predictability": 1.0 - permutation_entropy(values, 3),
        "evaluated_horizons": ";".join(str(value) for value in evaluated_horizons),
        "fold_count": maximum_folds,
        "validation_tier": validation_tier,
    }
    return record, fold_rows


def _derived_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["series_id"], row["horizon"], row["origin_index"])].append(row)
    history: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        candidates = grouped[key]
        nonbaseline = [
            row for row in candidates if row["model_id"] not in {"naive", "seasonal_naive"}
        ]
        ensemble_candidates = nonbaseline or candidates
        template = candidates[0]
        predicted = statistics.median(row["predicted"] for row in ensemble_candidates)
        sigma = max(
            statistics.pstdev(row["predicted"] for row in ensemble_candidates),
            math.sqrt(template["actual"] + 1.0),
            1.0,
        )
        ensemble = dict(template)
        ensemble.update(
            {
                "model_id": "median_ensemble",
                "predicted": predicted,
                "absolute_error": abs(template["actual"] - predicted),
                "squared_error": (template["actual"] - predicted) ** 2,
                "smape": smape(template["actual"], predicted),
                "poisson_deviance": poisson_deviance(template["actual"], predicted),
                "outbreak_probability": normal_probability_above(
                    template["outbreak_threshold"], predicted, sigma
                ),
                "risk_predicted": _risk_category(
                    predicted, template["risk_lower"], template["risk_upper"]
                ),
            }
        )
        result.append(ensemble)
        series_id, horizon, _ = key
        eligible = []
        for row in candidates:
            previous = history[(series_id, horizon, row["model_id"])]
            if len(previous) >= 2:
                eligible.append((statistics.fmean(previous), row["model_id"]))
        selected_id = min(eligible)[1] if eligible else "seasonal_naive"
        selected = next(
            (row for row in candidates if row["model_id"] == selected_id), candidates[0]
        )
        selector = dict(selected)
        selector["model_id"] = "online_selector"
        result.append(selector)
        for row in candidates:
            history[(series_id, horizon, row["model_id"])].append(
                row["absolute_error"] / row["mase_scale"]
            )
    return result


def _metric_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["series_id"], row["model_id"], row["horizon"])].append(row)
    result: list[dict[str, Any]] = []
    for (series_id, model_id, horizon), values in sorted(grouped.items()):
        measurements = {
            "mae": statistics.fmean(row["absolute_error"] for row in values),
            "rmse": math.sqrt(statistics.fmean(row["squared_error"] for row in values)),
            "mase": statistics.fmean(row["absolute_error"] / row["mase_scale"] for row in values),
            "rmsse": math.sqrt(
                statistics.fmean(row["squared_error"] / row["rmsse_scale"] for row in values)
            ),
            "smape": statistics.fmean(row["smape"] for row in values),
            "poisson_deviance": statistics.fmean(row["poisson_deviance"] for row in values),
        }
        outbreak = binary_metrics(
            [row["outbreak_actual"] for row in values],
            [row["outbreak_probability"] for row in values],
        )
        risk = multiclass_metrics(
            [row["risk_actual"] for row in values],
            [row["risk_predicted"] for row in values],
        )
        for task, task_values in (
            ("forecast", measurements),
            ("outbreak", outbreak),
            ("risk", risk),
        ):
            for metric_name, metric_value in task_values.items():
                result.append(
                    {
                        "predictability_run_id": run_id,
                        "series_id": series_id,
                        "model_id": model_id,
                        "horizon": horizon,
                        "task": task,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "sample_size": len(values),
                        "event_count": sum(row["outbreak_actual"] for row in values),
                    }
                )
    return result


def _frontier_rows(
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    series_records: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    tiers = {row["series_id"]: row["validation_tier"] for row in series_records}
    metric_index = {
        (row["series_id"], row["model_id"], row["horizon"], row["task"], row["metric_name"]): row
        for row in metrics
        if math.isfinite(row["metric_value"])
    }
    fold_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fold_groups[(row["series_id"], row["model_id"], row["horizon"])].append(row)
    combinations = sorted({(row["series_id"], row["horizon"]) for row in rows})
    result: list[dict[str, Any]] = []
    definitions = (
        ("forecast", "mase", "min"),
        ("outbreak", "brier", "min"),
        ("risk", "macro_f1", "max"),
    )
    for series_id, horizon in combinations:
        models = sorted(
            {
                model_id
                for candidate_series, model_id, candidate_horizon in fold_groups
                if candidate_series == series_id and candidate_horizon == horizon
            }
        )
        for task, criterion, direction in definitions:
            candidates = [
                (
                    metric_index[(series_id, model_id, horizon, task, criterion)]["metric_value"],
                    model_id,
                )
                for model_id in models
                if (series_id, model_id, horizon, task, criterion) in metric_index
            ]
            if not candidates:
                continue
            value, model_id = min(candidates) if direction == "min" else max(candidates)
            baseline_id = (
                "seasonal_naive"
                if (series_id, "seasonal_naive", horizon) in fold_groups
                else "naive"
            )
            baseline_key = (series_id, baseline_id, horizon, task, criterion)
            baseline_value = (
                metric_index[baseline_key]["metric_value"]
                if baseline_key in metric_index
                else math.nan
            )
            skill = math.nan
            lower = math.nan
            upper = math.nan
            if task == "forecast":
                selected_rows = fold_groups[(series_id, model_id, horizon)]
                baseline_rows = {
                    row["origin_index"]: row
                    for row in fold_groups[(series_id, baseline_id, horizon)]
                }
                aligned = [
                    (row, baseline_rows[row["origin_index"]])
                    for row in selected_rows
                    if row["origin_index"] in baseline_rows
                ]
                seed = int(hashlib.sha256(f"{series_id}|{horizon}".encode()).hexdigest()[:8], 16)
                skill, lower, upper = bootstrap_skill_interval(
                    [left["absolute_error"] for left, _ in aligned],
                    [right["absolute_error"] for _, right in aligned],
                    seed,
                )
            elif math.isfinite(baseline_value):
                skill = (
                    1.0 - value / max(baseline_value, 1e-9)
                    if direction == "min"
                    else value - baseline_value
                )
            selected = fold_groups[(series_id, model_id, horizon)]
            result.append(
                {
                    "predictability_run_id": run_id,
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
                    "fold_count": len(selected),
                    "event_count": sum(row["outbreak_actual"] for row in selected),
                    "validation_tier": tiers.get(series_id, "insufficient"),
                }
            )
    return result


def _aggregate_rows(
    metric_rows: list[dict[str, Any]], series_records: list[dict[str, Any]], run_id: str
) -> list[dict[str, Any]]:
    metadata = {row["series_id"]: row for row in series_records}
    grouped: dict[tuple[str, str, str, str, int, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        series = metadata[row["series_id"]]
        if series["validation_tier"] == "excluded" or not math.isfinite(row["metric_value"]):
            continue
        key = (
            series["disease"],
            series["spatial_resolution"],
            series["temporal_resolution"],
            row["model_id"],
            row["horizon"],
            row["task"],
            row["metric_name"],
        )
        grouped[key].append(row["metric_value"])
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        disease, spatial, temporal, model_id, horizon, task, metric_name = key
        result.append(
            {
                "predictability_run_id": run_id,
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
            }
        )
    return result


def evaluate(series: list[dict[str, Any]], run_id: str) -> dict[str, list[dict[str, Any]]]:
    series_records: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for value in series:
        record, rows = _base_fold_rows(value, run_id)
        series_records.append(record)
        fold_rows.extend(rows)
    fold_rows.extend(_derived_rows(fold_rows))
    metric_rows = _metric_rows(fold_rows, run_id)
    frontier_rows = _frontier_rows(fold_rows, metric_rows, series_records, run_id)
    aggregate_rows = _aggregate_rows(metric_rows, series_records, run_id)
    return {
        "series": series_records,
        "folds": fold_rows,
        "metrics": metric_rows,
        "frontiers": frontier_rows,
        "aggregates": aggregate_rows,
    }
