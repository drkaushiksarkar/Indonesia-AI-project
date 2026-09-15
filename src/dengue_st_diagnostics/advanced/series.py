from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import psycopg

SERIES_DIMENSIONS = [
    "source_id",
    "source_dataset_id",
    "source_dataset_name",
    "upstream_source",
    "disease",
    "metric",
    "unit",
    "spatial_resolution",
    "spatial_unit_type",
    "temporal_resolution_normalized",
    "admin_0",
    "admin_1",
    "admin_2",
    "admin_3",
    "facility_name",
    "facility_type",
    "source_subunit",
    "source_aggregation_level",
    "sex",
    "age_group",
    "species",
    "case_definition",
    "confirmation_status",
]

PANEL_DIMENSIONS = [
    "source_id",
    "source_dataset_id",
    "source_dataset_name",
    "upstream_source",
    "disease",
    "metric",
    "unit",
    "spatial_resolution",
    "spatial_unit_type",
    "temporal_resolution_normalized",
    "source_aggregation_level",
    "sex",
    "age_group",
    "species",
    "case_definition",
    "confirmation_status",
]

FREQUENCIES = {"week": "7D", "month": "MS", "quarter": "QS", "year": "YS"}


def _identifier(prefix: str, values: dict[str, Any]) -> str:
    encoded = json.dumps(values, default=str, ensure_ascii=False, sort_keys=True).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _location(row: pd.Series) -> str:
    values = [
        row.get("admin_0"),
        row.get("admin_1"),
        row.get("admin_2"),
        row.get("admin_3"),
        row.get("facility_name"),
        row.get("source_subunit"),
    ]
    clean = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
    return " | ".join(dict.fromkeys(clean)) or "Unspecified"


def read_native_observations(connection: psycopg.Connection[Any]) -> pd.DataFrame:
    cursor = connection.execute(
        """
        select
            observation_id,
            source_id,
            source_dataset_id,
            source_dataset_name,
            upstream_source,
            upstream_source_record_id,
            disease,
            metric,
            value,
            unit,
            period_start,
            period_end_effective,
            spatial_resolution,
            spatial_unit_type,
            temporal_resolution_normalized,
            admin_0,
            admin_1,
            admin_2,
            admin_3,
            facility_name,
            facility_type,
            source_subunit,
            source_aggregation_level,
            sex,
            age_group,
            species,
            case_definition,
            confirmation_status
        from gold.modeling_series
        order by source_id, source_dataset_id, disease, metric, period_start, observation_id
        """
    )
    columns = [column.name for column in cursor.description or []]
    frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end_effective"] = pd.to_datetime(frame["period_end_effective"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["location_name"] = frame.apply(_location, axis=1)
    frame["series_id"] = frame.apply(
        lambda row: _identifier(
            "series",
            {column: row.get(column) for column in SERIES_DIMENSIONS},
        ),
        axis=1,
    )
    frame["panel_id"] = frame.apply(
        lambda row: _identifier(
            "panel",
            {column: row.get(column) for column in PANEL_DIMENSIONS},
        ),
        axis=1,
    )
    return frame


def _expected_periods(group: pd.DataFrame, grain: str) -> int | None:
    frequency = FREQUENCIES.get(grain)
    if frequency is None or group.empty:
        return None
    return len(
        pd.date_range(group["period_start"].min(), group["period_start"].max(), freq=frequency)
    )


def build_series_catalog(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for series_id, group in frame.groupby("series_id", sort=True, dropna=False):
        group = group.sort_values("period_start")
        first = group.iloc[0]
        grain = str(first["temporal_resolution_normalized"])
        expected = _expected_periods(group, grain)
        duplicate_periods = int(group["period_start"].duplicated(keep=False).sum())
        observations = len(group)
        unique_periods = int(group["period_start"].nunique())
        completeness = unique_periods / expected if expected and expected > 0 else np.nan
        intervals = group["period_start"].drop_duplicates().sort_values().diff().dropna().dt.days
        regular = bool(
            duplicate_periods == 0
            and expected is not None
            and completeness >= 0.8
            and (intervals.nunique() <= 2 or grain in {"month", "quarter", "year"})
        )
        eligibility = {
            "profile": True,
            "temporal": regular and unique_periods >= 8,
            "stl": regular and grain in {"week", "month", "quarter"},
            "emd_hht": regular and unique_periods >= 24,
            "panel": True,
            "reason": "eligible"
            if regular
            else "duplicate_periods"
            if duplicate_periods > 0
            else "irregular_or_sparse_time",
        }
        rows.append(
            {
                "series_id": series_id,
                "panel_id": first["panel_id"],
                "source_id": first["source_id"],
                "source_dataset_id": int(first["source_dataset_id"]),
                "source_dataset_name": first["source_dataset_name"],
                "upstream_source": first["upstream_source"],
                "disease": first["disease"],
                "metric": first["metric"],
                "unit": first["unit"],
                "spatial_resolution": first["spatial_resolution"],
                "spatial_unit_type": first["spatial_unit_type"],
                "temporal_resolution": grain,
                "admin_0": first["admin_0"],
                "admin_1": first["admin_1"],
                "admin_2": first["admin_2"],
                "admin_3": first["admin_3"],
                "facility_name": first["facility_name"],
                "facility_type": first["facility_type"],
                "source_subunit": first["source_subunit"],
                "source_aggregation_level": first["source_aggregation_level"],
                "location_name": first["location_name"],
                "first_period": group["period_start"].min().date(),
                "last_period": group["period_end_effective"].max().date(),
                "observations": observations,
                "expected_periods": expected,
                "completeness": completeness,
                "duplicate_periods": duplicate_periods,
                "regular_time": regular,
                "eligibility": eligibility,
            }
        )
        inputs.extend(
            {
                "series_id": series_id,
                "observation_id": int(observation_id),
            }
            for observation_id in group["observation_id"]
        )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(inputs)


def regular_series(group: pd.DataFrame, grain: str) -> pd.DataFrame:
    frequency = FREQUENCIES[grain]
    values = group.set_index("period_start")["value"].sort_index()
    index = pd.date_range(values.index.min(), values.index.max(), freq=frequency)
    result = values.reindex(index).rename("observed_value").to_frame()
    result.index.name = "period_start"
    result["analysis_value"] = result["observed_value"].interpolate(
        method="linear",
        limit_direction="both",
    )
    result["imputed"] = result["observed_value"].isna()
    return result.reset_index()
