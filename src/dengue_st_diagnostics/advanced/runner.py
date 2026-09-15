from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from dengue_st_diagnostics.advanced.artifacts import (
    artifact_metadata,
    overview_artifacts,
    panel_artifacts,
    spatial_artifact,
    temporal_artifacts,
)
from dengue_st_diagnostics.advanced.methods import method_registry
from dengue_st_diagnostics.advanced.panels import diagnose_panel
from dengue_st_diagnostics.advanced.series import (
    build_series_catalog,
    read_native_observations,
)
from dengue_st_diagnostics.advanced.spatial import (
    diagnose_spatial_panel,
    load_boundaries,
)
from dengue_st_diagnostics.advanced.storage import (
    apply_schema,
    complete_run,
    export_frames,
    input_snapshot,
    load_results,
    prepare_statistics,
    register_methods,
    register_run,
)
from dengue_st_diagnostics.advanced.temporal import diagnose_series
from dengue_st_diagnostics.config import Settings


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=columns)


def _manifest(root: Path) -> Path:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest}
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_advanced_diagnostics(settings: Settings) -> dict[str, Path]:
    started_at = datetime.now(UTC)
    diagnostic_run_id = str(uuid.uuid4())
    lake = settings.section("lakehouse")
    dsn = os.getenv(lake["dsn_environment"], f"dbname={lake['database']}")
    schema_path = Path(__file__).resolve().parents[3] / "sql" / "medallion.sql"
    output_root = settings.paths.root.resolve()
    run_root = output_root / "diagnostics" / diagnostic_run_id
    image_root = run_root / "images"
    quantitative_root = run_root / "quantitative"
    configuration = {
        "input_relation": "gold.native_observation",
        "minimum_temporal_periods": 8,
        "minimum_panel_locations": 4,
        "minimum_panel_periods": 5,
        "spatial_permutations": int(settings.section("analysis")["monte_carlo_permutations"]),
        "random_seed": int(settings.section("analysis")["random_seed"]),
        "native_grain_only": True,
        "cross_source_pooling": False,
        "temporal_resampling": False,
        "diagnostic_gap_interpolation": "linear_with_fraction_recorded",
        "false_discovery_rate": "benjamini_hochberg_within_method_statistic",
    }
    software = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scipy": _version("scipy"),
        "statsmodels": _version("statsmodels"),
        "scikit_learn": _version("scikit-learn"),
        "pywavelets": _version("PyWavelets"),
        "ruptures": _version("ruptures"),
        "esda": _version("esda"),
        "libpysal": _version("libpysal"),
    }
    methods = method_registry()
    with psycopg.connect(dsn) as connection:
        apply_schema(connection, schema_path)
        snapshot = input_snapshot(connection)
        register_run(connection, diagnostic_run_id, started_at, snapshot, configuration, software)
        register_methods(connection, methods)
        observations = read_native_observations(connection)
        connection.commit()
    catalog, inputs = build_series_catalog(observations)
    random_seed = configuration["random_seed"]
    statistics_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    grouped_series = dict(list(observations.groupby("series_id", sort=True)))
    for _, series in catalog.sort_values("series_id").iterrows():
        statistics, components, events, plot_data = diagnose_series(
            series,
            grouped_series[str(series["series_id"])],
            random_seed,
        )
        statistics_records.extend(statistics)
        component_records.extend(components)
        event_records.extend(events)
        artifact_records.extend(temporal_artifacts(image_root, series, plot_data))
    panel_plot_data: dict[str, dict[str, Any]] = {}
    grouped_panels = dict(list(observations.groupby("panel_id", sort=True)))
    for panel_id, panel in grouped_panels.items():
        statistics, components, events, plot_data = diagnose_panel(
            str(panel_id),
            panel,
            catalog,
        )
        statistics_records.extend(statistics)
        component_records.extend(components)
        event_records.extend(events)
        first = panel.iloc[0]
        title = (
            f"{first['upstream_source'] or first['source_dataset_name']} · "
            f"{first['disease'] or 'Unspecified'} · {first['metric']} · "
            f"{first['spatial_resolution']}/{first['temporal_resolution_normalized']}"
        )
        artifact_records.extend(panel_artifacts(image_root, str(panel_id), title, plot_data))
        panel_plot_data[str(panel_id)] = plot_data
    boundaries = load_boundaries(output_root / "raw" / "boundaries")
    permutations = configuration["spatial_permutations"]
    for panel_id, panel in grouped_panels.items():
        spatial_resolution = str(panel["spatial_resolution"].iloc[0])
        statistics, events, plot_data = diagnose_spatial_panel(
            str(panel_id),
            panel,
            spatial_resolution,
            boundaries,
            permutations,
            random_seed,
        )
        if statistics:
            statistics_records.extend(statistics)
            event_records.extend(events)
            first = panel.iloc[0]
            title = (
                f"{first['upstream_source'] or first['source_dataset_name']} · "
                f"{first['disease'] or 'Unspecified'} · {first['metric']}"
            )
            artifact_records.extend(spatial_artifact(image_root, str(panel_id), title, plot_data))
            panel_plot_data.setdefault(str(panel_id), {}).update(plot_data)
    statistic_columns = [
        "series_id",
        "panel_id",
        "method_id",
        "statistic_name",
        "estimate",
        "p_value",
        "q_value",
        "confidence_lower",
        "confidence_upper",
        "unit",
        "status",
        "interpretation",
        "parameters",
    ]
    component_columns = [
        "series_id",
        "panel_id",
        "method_id",
        "component_name",
        "component_index",
        "coordinate_name",
        "coordinate_value",
        "period_start",
        "location_name",
        "value",
        "component_metadata",
    ]
    event_columns = [
        "series_id",
        "panel_id",
        "method_id",
        "period_start",
        "period_end",
        "location_name",
        "event_type",
        "score",
        "threshold",
        "direction",
        "severity",
        "p_value",
        "q_value",
        "event_metadata",
    ]
    statistics = prepare_statistics(
        _frame(statistics_records, statistic_columns), diagnostic_run_id
    )
    components = _frame(component_records, component_columns)
    events = _frame(event_records, event_columns)
    artifact_records.extend(overview_artifacts(image_root, catalog, statistics, events))
    method_frame = pd.DataFrame.from_records(methods)
    summary = pd.DataFrame.from_records(
        [
            {
                "diagnostic_run_id": diagnostic_run_id,
                "input_snapshot_sha256": snapshot,
                "native_observations": len(observations),
                "series": len(catalog),
                "regular_series": int(catalog["regular_time"].sum()),
                "panels": int(catalog["panel_id"].nunique()),
                "statistics": len(statistics),
                "components": len(components),
                "events": len(events),
                "images": len(artifact_records),
            }
        ]
    )
    core_frames = {
        "diagnostic_summary": summary,
        "method_registry": method_frame,
        "series_catalog": catalog,
        "diagnostic_inputs": inputs,
        "diagnostic_statistics": statistics,
        "diagnostic_components": components,
        "diagnostic_events": events,
    }
    checkpoint_root = run_root / "checkpoint"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for name, frame in core_frames.items():
        frame.to_pickle(checkpoint_root / f"{name}.pkl.gz", compression="gzip")
    quantitative_paths = export_frames(quantitative_root, core_frames)
    artifact_records.extend(
        {
            "path": path,
            "method_id": None,
            "series_id": None,
            "panel_id": None,
            "artifact_role": f"quantitative_{path.name.split('.')[0]}",
        }
        for path in quantitative_paths
    )
    artifacts = artifact_metadata(output_root, diagnostic_run_id, artifact_records)
    export_frames(quantitative_root, {"artifact_catalog": artifacts})
    completed_at = datetime.now(UTC)
    run_metadata = {
        "native_observations": len(observations),
        "series": len(catalog),
        "regular_series": int(catalog["regular_time"].sum()),
        "panels": int(catalog["panel_id"].nunique()),
        "statistics": len(statistics),
        "components": len(components),
        "events": len(events),
        "artifacts": len(artifacts),
    }
    with psycopg.connect(dsn) as connection:
        load_results(
            connection,
            diagnostic_run_id,
            catalog,
            inputs,
            statistics,
            components,
            events,
            artifacts,
        )
        complete_run(connection, diagnostic_run_id, completed_at, run_metadata)
        connection.commit()
    manifest = _manifest(run_root)
    latest = output_root / "diagnostics" / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "diagnostic_run_id": diagnostic_run_id,
                "run_root": str(run_root),
                "manifest": str(manifest),
                "completed_at": completed_at.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"run_root": run_root, "manifest": manifest, "latest": latest}
