from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import math
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INK = "#172033"
BLUE = "#007C91"
GOLD = "#E3A018"
ORANGE = "#E8752A"
OLIVE = "#708238"
PINK = "#C65A8E"
GRID = "#D9DEE7"
MUTED = "#667085"
PALETTE = [BLUE, GOLD, ORANGE, OLIVE, PINK]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _clean(value) for key, value in row.items()} for row in rows)


def _svg_start(title: str, subtitle: str, width: int = 1200, height: int = 720) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="70" y="58" fill="{INK}" font-family="Arial, sans-serif" font-size="28" font-weight="700">{html.escape(title)}</text>',
        f'<text x="70" y="88" fill="{MUTED}" font-family="Arial, sans-serif" font-size="15">{html.escape(subtitle)}</text>',
    ]


def _svg_end(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _empty(parts: list[str]) -> None:
    parts.append(
        f'<text x="600" y="370" text-anchor="middle" fill="{MUTED}" font-family="Arial, sans-serif" font-size="20">Insufficient eligible evidence</text>'
    )


def _bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str] | None = None,
    value_format: str = ".2f",
) -> None:
    parts = _svg_start(title, subtitle)
    if not values:
        _empty(parts)
        _svg_end(parts, path)
        return
    left = 310
    right = 1110
    top = 125
    bottom = 650
    maximum = max(max(values), 1e-9)
    row_height = (bottom - top) / len(values)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = top + index * row_height
        width = (right - left) * value / maximum
        color = (colors or PALETTE)[index % len(colors or PALETTE)]
        parts.extend(
            [
                f'<line x1="{left}" y1="{y + row_height * 0.72:.1f}" x2="{right}" y2="{y + row_height * 0.72:.1f}" stroke="{GRID}" stroke-width="1"/>',
                f'<text x="{left - 16}" y="{y + row_height * 0.52:.1f}" text-anchor="end" fill="{INK}" font-family="Arial, sans-serif" font-size="14">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y + row_height * 0.18:.1f}" width="{width:.1f}" height="{row_height * 0.45:.1f}" fill="{color}" stroke="{INK}" stroke-width="0.5"/>',
                f'<text x="{min(right - 4, left + width + 10):.1f}" y="{y + row_height * 0.52:.1f}" fill="{INK}" font-family="Courier New, monospace" font-size="13">{format(value, value_format)}</text>',
            ]
        )
    _svg_end(parts, path)


def _line_chart(
    path: Path,
    title: str,
    subtitle: str,
    groups: dict[str, list[tuple[float, float]]],
    y_label: str,
    x_label: str = "Forecast horizon in native periods",
) -> None:
    parts = _svg_start(title, subtitle)
    points = [(x, y) for values in groups.values() for x, y in values if math.isfinite(y)]
    if not points:
        _empty(parts)
        _svg_end(parts, path)
        return
    left, right, top, bottom = 115, 1110, 155, 610
    x_values = [value[0] for value in points]
    y_values = [value[1] for value in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min = min(0.0, min(y_values))
    y_max = max(max(y_values), y_min + 1e-6)

    def x_scale(value: float) -> float:
        return left + (right - left) * (value - x_min) / max(x_max - x_min, 1e-9)

    def y_scale(value: float) -> float:
        return bottom - (bottom - top) * (value - y_min) / max(y_max - y_min, 1e-9)

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_scale(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>',
                f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" fill="{MUTED}" font-family="Courier New, monospace" font-size="12">{value:.2f}</text>',
            ]
        )
    for index, (label, values) in enumerate(sorted(groups.items())):
        color = PALETTE[index % len(PALETTE)]
        ordered = sorted((x, y) for x, y in values if math.isfinite(y))
        coordinates = " ".join(f"{x_scale(x):.1f},{y_scale(y):.1f}" for x, y in ordered)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x, y in ordered:
            parts.append(
                f'<circle cx="{x_scale(x):.1f}" cy="{y_scale(y):.1f}" r="5" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>'
            )
        legend_x = 390 + index % 3 * 245
        legend_y = 108 + index // 3 * 20
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 32}" y="{legend_y + 5}" fill="{INK}" font-family="Arial, sans-serif" font-size="12">{html.escape(label)}</text>',
            ]
        )
    for value in sorted(set(x_values)):
        parts.append(
            f'<text x="{x_scale(value):.1f}" y="636" text-anchor="middle" fill="{MUTED}" font-family="Courier New, monospace" font-size="12">{value:g}</text>'
        )
    parts.extend(
        [
            f'<text x="612" y="662" text-anchor="middle" fill="{INK}" font-family="Arial, sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="32" y="370" transform="rotate(-90 32 370)" text-anchor="middle" fill="{INK}" font-family="Arial, sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    _svg_end(parts, path)


def _scatter_chart(
    path: Path,
    title: str,
    subtitle: str,
    points: list[tuple[float, float, str]],
    x_label: str,
    y_label: str,
) -> None:
    parts = _svg_start(title, subtitle)
    points = [point for point in points if math.isfinite(point[0]) and math.isfinite(point[1])]
    if not points:
        _empty(parts)
        _svg_end(parts, path)
        return
    left, right, top, bottom = 115, 1110, 125, 610
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(min(y_values), 0.0), max(max(y_values), 0.0)

    def x_scale(value: float) -> float:
        return left + (right - left) * (value - x_min) / max(x_max - x_min, 1e-9)

    def y_scale(value: float) -> float:
        return bottom - (bottom - top) * (value - y_min) / max(y_max - y_min, 1e-9)

    categories = sorted({point[2] for point in points})
    colors = {category: PALETTE[index % len(PALETTE)] for index, category in enumerate(categories)}
    for tick in range(6):
        x_value = x_min + (x_max - x_min) * tick / 5
        y_value = y_min + (y_max - y_min) * tick / 5
        x = x_scale(x_value)
        y = y_scale(y_value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}" font-family="Courier New, monospace" font-size="11">{y_value:.2f}</text>',
                f'<text x="{x:.1f}" y="634" text-anchor="middle" fill="{MUTED}" font-family="Courier New, monospace" font-size="11">{x_value:.2f}</text>',
            ]
        )
    for x_value, y_value, category in points:
        parts.append(
            f'<circle cx="{x_scale(x_value):.1f}" cy="{y_scale(y_value):.1f}" r="5" fill="{colors[category]}" fill-opacity="0.68" stroke="{INK}" stroke-width="0.5"/>'
        )
    for index, category in enumerate(categories):
        parts.extend(
            [
                f'<circle cx="{720 + index * 115}" cy="111" r="5" fill="{colors[category]}"/>',
                f'<text x="{732 + index * 115}" y="116" fill="{INK}" font-family="Arial, sans-serif" font-size="13">{html.escape(category)}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="612" y="662" text-anchor="middle" fill="{INK}" font-family="Arial, sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="32" y="370" transform="rotate(-90 32 370)" text-anchor="middle" fill="{INK}" font-family="Arial, sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    _svg_end(parts, path)


def _median_groups(
    rows: list[dict[str, Any]], key_fields: tuple[str, ...], value_field: str
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row[value_field]
        if isinstance(value, (int, float)) and math.isfinite(value):
            grouped[tuple(row[field] for field in key_fields)].append(value)
    return {key: statistics.median(values) for key, values in grouped.items() if values}


def create_images(root: Path, results: dict[str, list[dict[str, Any]]]) -> list[Path]:
    image_root = root / "images"
    series = results["series"]
    metrics = results["metrics"]
    frontiers = results["frontiers"]
    metadata = {row["series_id"]: row for row in series}
    paths: list[Path] = []
    coverage = Counter(
        (row["temporal_resolution"], row["validation_tier"])
        for row in series
        if row["include_status"] != "excluded"
    )
    labels = [f"{grain} · {tier}" for grain, tier in sorted(coverage)]
    values = [float(coverage[key]) for key in sorted(coverage)]
    path = image_root / "coverage_by_grain.svg"
    _bar_chart(
        path,
        "Univariate validation coverage",
        "Series counts by native temporal grain and evidence tier",
        labels,
        values,
        value_format=".0f",
    )
    paths.append(path)
    eligible_ids = {
        row["series_id"] for row in series if row["validation_tier"] in {"robust", "moderate"}
    }
    mase_rows = [
        row
        for row in metrics
        if row["task"] == "forecast"
        and row["metric_name"] == "mase"
        and row["horizon"] == 1
        and row["series_id"] in eligible_ids
    ]
    leaderboard = _median_groups(mase_rows, ("model_id",), "metric_value")
    ranked = sorted(leaderboard.items(), key=lambda item: item[1])[:12]
    path = image_root / "model_leaderboard_h1.svg"
    _bar_chart(
        path,
        "One-period forecast model leaderboard",
        "Median rolling-origin MASE across robust and moderate series; lower is better",
        [item[0][0] for item in ranked],
        [item[1] for item in ranked],
        [BLUE] * len(ranked),
    )
    paths.append(path)
    skill_rows = [
        {**row, "grain": metadata[row["series_id"]]["temporal_resolution"]}
        for row in frontiers
        if row["task"] == "forecast" and row["validation_tier"] in {"robust", "moderate"}
    ]
    skill_groups = _median_groups(skill_rows, ("grain", "horizon"), "skill")
    line_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (grain, horizon), value in skill_groups.items():
        line_groups[grain].append((float(horizon), value))
    path = image_root / "forecast_skill_by_horizon.svg"
    _line_chart(
        path,
        "Empirical forecastability frontier by horizon",
        "Best post-hoc algorithm relative to seasonal-naive; robust and moderate evidence only",
        dict(line_groups),
        "MAE skill",
    )
    paths.append(path)
    entropy_points = [
        (
            metadata[row["series_id"]]["entropy_predictability"],
            row["skill"],
            metadata[row["series_id"]]["temporal_resolution"],
        )
        for row in frontiers
        if row["task"] == "forecast"
        and row["horizon"] == 1
        and row["validation_tier"] != "insufficient"
    ]
    path = image_root / "entropy_vs_empirical_skill.svg"
    _scatter_chart(
        path,
        "Information structure versus achieved forecast skill",
        "Permutation-entropy redundancy is diagnostic, while rolling-origin skill is the empirical ceiling",
        entropy_points,
        "1 - normalized permutation entropy",
        "Frontier MAE skill",
    )
    paths.append(path)
    for task, metric_name, title, y_label in (
        (
            "outbreak",
            "auprc",
            "Deployable outbreak discrimination",
            "Area under precision-recall curve",
        ),
        ("risk", "macro_f1", "Deployable within-series risk stratification", "Macro F1"),
    ):
        selected = [
            {**row, "grain": metadata[row["series_id"]]["temporal_resolution"]}
            for row in metrics
            if row["task"] == task
            and row["metric_name"] == metric_name
            and row["model_id"] == "online_selector"
            and metadata[row["series_id"]]["validation_tier"] in {"robust", "moderate"}
        ]
        grouped = _median_groups(selected, ("grain", "horizon"), "metric_value")
        values_by_grain: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (grain, horizon), value in grouped.items():
            values_by_grain[grain].append((float(horizon), value))
        path = image_root / f"{task}_deployable_by_horizon.svg"
        _line_chart(
            path,
            title,
            "Online model selection uses only errors available before each forecast origin",
            dict(values_by_grain),
            y_label,
        )
        paths.append(path)
    puskesmas = [
        (float(row["period_count"]), float(row["fold_count"]), row["validation_tier"])
        for row in series
        if row["temporal_resolution"] == "month"
        and row["spatial_resolution"] in {"facility", "adm3", "adm4"}
    ]
    path = image_root / "puskesmas_history_vs_validation.svg"
    _scatter_chart(
        path,
        "Local monthly history versus validation depth",
        "Facility, subdistrict and village series; fold count is the usable rolling-origin evidence",
        puskesmas,
        "Available monthly periods",
        "Usable one-period origins",
    )
    paths.append(path)
    model_wins = Counter(
        row["frontier_model_id"]
        for row in frontiers
        if row["task"] == "forecast" and row["validation_tier"] in {"robust", "moderate"}
    )
    wins = model_wins.most_common(12)
    path = image_root / "frontier_model_wins.svg"
    _bar_chart(
        path,
        "Algorithm families on the empirical frontier",
        "Number of series-horizon combinations won under rolling-origin evaluation",
        [label for label, _ in wins],
        [float(value) for _, value in wins],
        [ORANGE] * len(wins),
        value_format=".0f",
    )
    paths.append(path)
    return paths


def _file_record(path: Path, root: Path, run_id: str, role: str) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "artifact_id": str(uuid.uuid4()),
        "predictability_run_id": run_id,
        "artifact_role": role,
        "media_type": "image/svg+xml"
        if path.suffix == ".svg"
        else "text/csv"
        if path.suffix == ".csv"
        else "application/json",
        "absolute_path": str(path.resolve()),
        "relative_path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
    }


def save_outputs(
    root: Path, results: dict[str, list[dict[str, Any]]], run_id: str
) -> tuple[list[dict[str, Any]], Path]:
    quantitative = root / "quantitative"
    files: list[tuple[Path, str]] = []
    for key, filename in (
        ("series", "series_predictability.csv"),
        ("folds", "fold_predictions.csv"),
        ("metrics", "model_metrics.csv"),
        ("frontiers", "predictability_frontiers.csv"),
        ("aggregates", "aggregate_metrics.csv"),
    ):
        path = quantitative / filename
        write_csv(path, results[key])
        files.append((path, key))
    image_paths = create_images(root, results)
    files.extend((path, path.stem) for path in image_paths)
    records = [_file_record(path, root, run_id, role) for path, role in files]
    artifact_path = quantitative / "artifact_catalog.csv"
    write_csv(artifact_path, records)
    records.append(_file_record(artifact_path, root, run_id, "artifact_catalog"))
    manifest = {
        "predictability_run_id": run_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "artifact_count": len(records),
        "artifacts": [
            {
                "relative_path": record["relative_path"],
                "sha256": record["sha256"],
                "byte_count": record["byte_count"],
            }
            for record in records
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return records, manifest_path
