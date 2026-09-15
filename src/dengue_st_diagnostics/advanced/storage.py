from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

SERIES_COLUMNS = [
    "diagnostic_run_id",
    "series_id",
    "panel_id",
    "source_id",
    "source_dataset_id",
    "source_dataset_name",
    "upstream_source",
    "disease",
    "metric",
    "unit",
    "spatial_resolution",
    "spatial_unit_type",
    "temporal_resolution",
    "admin_0",
    "admin_1",
    "admin_2",
    "admin_3",
    "facility_name",
    "facility_type",
    "source_subunit",
    "source_aggregation_level",
    "first_period",
    "last_period",
    "observations",
    "expected_periods",
    "completeness",
    "duplicate_periods",
    "regular_time",
    "eligibility",
]

STATISTIC_COLUMNS = [
    "diagnostic_run_id",
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
    "result_hash",
]

COMPONENT_COLUMNS = [
    "diagnostic_run_id",
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

EVENT_COLUMNS = [
    "diagnostic_run_id",
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


def apply_schema(connection: psycopg.Connection[Any], schema_path: Path) -> None:
    connection.execute(schema_path.read_text(encoding="utf-8"))


def input_snapshot(connection: psycopg.Connection[Any]) -> str:
    digest = hashlib.sha256()
    with connection.cursor(name="diagnostic_snapshot") as cursor:
        cursor.execute(
            "select observation_id, fact_hash from gold.native_observation order by observation_id"
        )
        for observation_id, fact_hash in cursor:
            digest.update(f"{observation_id}|{fact_hash}\n".encode())
    return digest.hexdigest()


def false_discovery_rate(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    if "q_value" not in value:
        value["q_value"] = np.nan
    groups = (
        ["method_id", "statistic_name"]
        if "statistic_name" in value
        else ["method_id", "event_type"]
    )
    for indices in value.loc[value["p_value"].notna()].groupby(groups).groups.values():
        selected = value.loc[indices, "p_value"].astype(float)
        order = np.argsort(selected.to_numpy())
        ranked = selected.to_numpy()[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty(len(adjusted))
        restored[order] = np.clip(adjusted, 0, 1)
        value.loc[indices, "q_value"] = restored
    return value


def prepare_statistics(frame: pd.DataFrame, diagnostic_run_id: str) -> pd.DataFrame:
    value = false_discovery_rate(frame)
    value["diagnostic_run_id"] = diagnostic_run_id
    value["result_hash"] = value.apply(
        lambda row: hashlib.sha256(
            json.dumps(
                {
                    "diagnostic_run_id": diagnostic_run_id,
                    "series_id": row.get("series_id"),
                    "panel_id": row.get("panel_id"),
                    "method_id": row.get("method_id"),
                    "statistic_name": row.get("statistic_name"),
                    "estimate": row.get("estimate"),
                    "p_value": row.get("p_value"),
                    "parameters": row.get("parameters"),
                },
                default=str,
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        axis=1,
    )
    return value


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return Jsonb(value)
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    return [
        tuple(_clean(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    ]


def _insert_statement(table: str, columns: list[str]) -> str:
    names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    return f"insert into {table} ({names}) values ({placeholders})"


def register_run(
    connection: psycopg.Connection[Any],
    diagnostic_run_id: str,
    started_at: Any,
    snapshot: str,
    configuration: dict[str, Any],
    software_environment: dict[str, Any],
) -> None:
    connection.execute(
        """
        insert into metadata.diagnostic_run (
            diagnostic_run_id, started_at, status, input_relation,
            input_snapshot_sha256, code_version, configuration, software_environment
        ) values (%s, %s, 'running', 'gold.native_observation', %s, '1.0.0', %s, %s)
        """,
        (
            diagnostic_run_id,
            started_at,
            snapshot,
            Jsonb(configuration),
            Jsonb(software_environment),
        ),
    )


def register_methods(connection: psycopg.Connection[Any], methods: list[dict[str, Any]]) -> None:
    columns = [
        "method_id",
        "method_family",
        "method_name",
        "method_version",
        "minimum_periods",
        "requires_regular_time",
        "requires_multiple_locations",
        "requires_geometry",
        "description",
        "citation_url",
        "default_parameters",
    ]
    connection.cursor().executemany(
        """
        insert into metadata.diagnostic_method (
            method_id, method_family, method_name, method_version, minimum_periods,
            requires_regular_time, requires_multiple_locations, requires_geometry,
            description, citation_url, default_parameters
        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (method_id) do update set
            method_family = excluded.method_family,
            method_name = excluded.method_name,
            method_version = excluded.method_version,
            minimum_periods = excluded.minimum_periods,
            requires_regular_time = excluded.requires_regular_time,
            requires_multiple_locations = excluded.requires_multiple_locations,
            requires_geometry = excluded.requires_geometry,
            description = excluded.description,
            citation_url = excluded.citation_url,
            default_parameters = excluded.default_parameters
        """,
        [
            tuple(
                Jsonb(item[column]) if column == "default_parameters" else item[column]
                for column in columns
            )
            for item in methods
        ],
    )


def load_results(
    connection: psycopg.Connection[Any],
    diagnostic_run_id: str,
    catalog: pd.DataFrame,
    inputs: pd.DataFrame,
    statistics: pd.DataFrame,
    components: pd.DataFrame,
    events: pd.DataFrame,
    artifacts: pd.DataFrame,
) -> None:
    series = catalog.copy()
    series["diagnostic_run_id"] = diagnostic_run_id
    connection.cursor().executemany(
        _insert_statement("metadata.diagnostic_series", SERIES_COLUMNS),
        _rows(series, SERIES_COLUMNS),
    )
    input_frame = inputs.copy()
    input_frame["diagnostic_run_id"] = diagnostic_run_id
    connection.cursor().executemany(
        _insert_statement(
            "metadata.diagnostic_input",
            ["diagnostic_run_id", "series_id", "observation_id"],
        ),
        _rows(input_frame, ["diagnostic_run_id", "series_id", "observation_id"]),
    )
    if not statistics.empty:
        connection.cursor().executemany(
            _insert_statement("gold.diagnostic_statistic", STATISTIC_COLUMNS),
            _rows(statistics, STATISTIC_COLUMNS),
        )
    if not components.empty:
        component_frame = components.copy()
        component_frame["diagnostic_run_id"] = diagnostic_run_id
        connection.cursor().executemany(
            _insert_statement("gold.diagnostic_component", COMPONENT_COLUMNS),
            _rows(component_frame, COMPONENT_COLUMNS),
        )
    if not events.empty:
        event_frame = false_discovery_rate(events)
        event_frame["diagnostic_run_id"] = diagnostic_run_id
        connection.cursor().executemany(
            _insert_statement("gold.diagnostic_event", EVENT_COLUMNS),
            _rows(event_frame, EVENT_COLUMNS),
        )
    if not artifacts.empty:
        artifact_columns = list(artifacts.columns)
        connection.cursor().executemany(
            _insert_statement("metadata.diagnostic_artifact", artifact_columns),
            _rows(artifacts, artifact_columns),
        )


def complete_run(
    connection: psycopg.Connection[Any],
    diagnostic_run_id: str,
    completed_at: Any,
    metadata: dict[str, Any],
) -> None:
    connection.execute(
        """
        update metadata.diagnostic_run
        set completed_at = %s, status = 'completed', run_metadata = %s
        where diagnostic_run_id = %s
        """,
        (completed_at, Jsonb(metadata), diagnostic_run_id),
    )


def export_frames(root: Path, frames: dict[str, pd.DataFrame]) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, frame in frames.items():
        exportable = frame.copy()
        for column in exportable.select_dtypes(include=["object", "str"]).columns:
            contains_nested = (
                exportable[column].map(lambda value: isinstance(value, (dict, list, tuple))).any()
            )
            if contains_nested:
                exportable[column] = exportable[column].map(
                    lambda value: (
                        json.dumps(
                            value,
                            default=str,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                )
        parquet_path = root / f"{name}.parquet"
        csv_path = root / f"{name}.csv.gz"
        exportable.to_parquet(parquet_path, index=False, compression="zstd")
        with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as handle:
            exportable.to_csv(handle, index=False)
        paths.extend([parquet_path, csv_path])
    return paths
