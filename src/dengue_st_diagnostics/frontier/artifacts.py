from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dengue_st_diagnostics.predictability.artifacts import (
    _bar_chart,
    _line_chart,
    _scatter_chart,
)


def _clean(value: Any) -> Any:
    return "" if isinstance(value, float) and not math.isfinite(value) else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _clean(value) for key, value in row.items()} for row in rows)


def _median(rows: list[dict[str, Any]], key: str, value: str) -> dict[Any, float]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        if isinstance(row[value], (int, float)) and math.isfinite(row[value]):
            grouped[row[key]].append(row[value])
    return {group: statistics.median(values) for group, values in grouped.items()}


def create_images(root: Path, results: dict[str, list[dict[str, Any]]]) -> list[Path]:
    image_root = root / "images"
    metadata = {row["series_id"]: row for row in results["series"]}
    metrics = results["metrics"]
    frontiers = results["frontiers"]
    paths: list[Path] = []
    province_ids = {series_id for series_id, row in metadata.items() if row["disease"] == "dengue" and row["spatial_resolution"] == "adm1" and row["temporal_resolution"] == "month"}
    point = [row for row in metrics if row["series_id"] in province_ids and row["task"] == "forecast_point" and row["metric_name"] == "mase" and row["model_id"] in {"seasonal_naive", "online_selector", "online_weighted_ensemble", "median_ensemble", "global_pooled_ridge"}]
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    grouped_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in point:
        grouped_values[(row["model_id"], row["horizon"])].append(row["metric_value"])
    baseline = {horizon: statistics.median(values) for (model_id, horizon), values in grouped_values.items() if model_id == "seasonal_naive"}
    for (model_id, horizon), values in grouped_values.items():
        if model_id != "seasonal_naive" and horizon in baseline:
            groups[model_id].append((float(horizon), 1.0 - statistics.median(values) / max(baseline[horizon], 1e-9)))
    path = image_root / "province_forecast_skill.svg"
    _line_chart(path, "Province-month dengue forecast skill", "Median MASE skill versus seasonal-naive across eligible province series", groups, "MASE skill")
    paths.append(path)
    probability = [row for row in metrics if row["series_id"] in province_ids and row["task"] == "forecast_probabilistic" and row["metric_name"] == "coverage_80" and row["model_id"] in {"seasonal_naive", "online_weighted_ensemble", "median_ensemble", "negative_binomial_ingarch"}]
    probability_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    probability_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in probability:
        probability_values[(row["model_id"], row["horizon"])].append(row["metric_value"])
    for (model_id, horizon), values in probability_values.items():
        probability_groups[model_id].append((float(horizon), statistics.median(values)))
    path = image_root / "province_interval_coverage.svg"
    _line_chart(path, "Province-month 80% interval coverage", "Empirical coverage across rolling-origin folds; nominal target is 0.80", probability_groups, "Empirical coverage")
    paths.append(path)
    outbreak = [row for row in metrics if row["series_id"] in province_ids and row["task"] == "outbreak" and row["metric_name"] == "auprc" and row["horizon"] == 1]
    outbreak_rank = sorted(_median(outbreak, "model_id", "metric_value").items(), key=lambda item: item[1], reverse=True)[:15]
    path = image_root / "outbreak_model_leaderboard.svg"
    _bar_chart(path, "One-month outbreak model leaderboard", "Median AUPRC across province-month dengue series", [item[0] for item in outbreak_rank], [item[1] for item in outbreak_rank], value_format=".3f")
    paths.append(path)
    wins = Counter(row["frontier_model_id"] for row in frontiers)
    ranked_wins = wins.most_common(15)
    path = image_root / "frontier_model_wins.svg"
    _bar_chart(path, "Algorithms on the expanded frontier", "Wins across series, horizons and four decision tasks", [item[0] for item in ranked_wins], [float(item[1]) for item in ranked_wins], value_format=".0f")
    paths.append(path)
    learning = [row for row in results["learning_curves"] if row["disease"] == "dengue" and row["spatial_resolution"] == "adm1" and row["temporal_resolution"] == "month"]
    learning_grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in learning:
        learning_grouped[(row["model_id"], row["history_periods"])].append(row["mae_skill"])
    learning_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (model_id, history), values in learning_grouped.items():
        learning_groups[model_id].append((float(history), statistics.median(values)))
    path = image_root / "history_learning_curve.svg"
    _line_chart(path, "Forecast skill by available history", "One-month province dengue skill under fixed trailing training windows", learning_groups, "MAE skill", "Available training periods")
    paths.append(path)
    facility = [row for row in frontiers if metadata[row["series_id"]]["spatial_resolution"] == "facility" and metadata[row["series_id"]]["temporal_resolution"] == "month" and row["task"] == "forecast_point" and row["horizon"] == 1]
    points = [(float(metadata[row["series_id"]]["period_count"]), row["skill"], metadata[row["series_id"]]["disease"]) for row in facility]
    path = image_root / "facility_skill_vs_history.svg"
    _scatter_chart(path, "Facility forecast skill versus history", "One-month empirical frontier; each point is a facility series", points, "Available monthly periods", "Frontier MAE skill")
    paths.append(path)
    return paths


def save_outputs(
    root: Path, results: dict[str, list[dict[str, Any]]], run_id: str
) -> tuple[list[dict[str, Any]], Path]:
    quantitative = root / "quantitative"
    files = {
        "series": "series.csv",
        "folds": "fold_predictions.csv",
        "metrics": "model_metrics.csv",
        "frontiers": "frontiers.csv",
        "learning_curves": "learning_curves.csv",
        "aggregates": "aggregate_metrics.csv",
    }
    paths: list[tuple[str, Path, str]] = []
    for key, filename in files.items():
        path = quantitative / filename
        _write_csv(path, results[key])
        paths.append((key, path, "text/csv"))
    for path in create_images(root, results):
        paths.append((path.stem, path, "image/svg+xml"))
    created = dt.datetime.now(dt.UTC).isoformat()
    artifacts: list[dict[str, Any]] = []
    for role, path, media_type in paths:
        payload = path.read_bytes()
        artifacts.append({
            "artifact_id": str(uuid.uuid4()),
            "frontier_run_id": run_id,
            "artifact_role": role,
            "media_type": media_type,
            "absolute_path": str(path.resolve()),
            "relative_path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "created_at": created,
        })
    catalog = quantitative / "artifact_catalog.csv"
    _write_csv(catalog, artifacts)
    catalog_payload = catalog.read_bytes()
    manifest_artifacts = [
        {"relative_path": row["relative_path"], "sha256": row["sha256"], "byte_count": row["byte_count"]}
        for row in artifacts
    ]
    manifest_artifacts.append({"relative_path": str(catalog.relative_to(root)), "sha256": hashlib.sha256(catalog_payload).hexdigest(), "byte_count": len(catalog_payload)})
    manifest = {
        "frontier_run_id": run_id,
        "created_at": created,
        "artifact_count": len(manifest_artifacts),
        "artifacts": manifest_artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts, manifest_path
