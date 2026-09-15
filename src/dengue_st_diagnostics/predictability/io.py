from __future__ import annotations

import csv
import datetime as dt
import io
import math
import subprocess
from collections import defaultdict
from typing import Any


def psql(database: str, sql: str) -> str:
    result = subprocess.run(
        ["psql", "-X", "-d", database, "-v", "ON_ERROR_STOP=1", "-At", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def execute_sql(database: str, sql: str) -> None:
    subprocess.run(
        ["psql", "-X", "-d", database, "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
    )


def latest_diagnostic_run(database: str) -> str:
    return psql(
        database,
        "select diagnostic_run_id from metadata.diagnostic_run where status='completed' order by completed_at desc limit 1",
    )


def diagnostic_snapshot(database: str, diagnostic_run_id: str) -> str:
    return psql(
        database,
        f"select input_snapshot_sha256 from metadata.diagnostic_run where diagnostic_run_id='{diagnostic_run_id}'",
    )


def _copy_query(database: str, query: str) -> list[dict[str, str]]:
    command = f"copy ({query}) to stdout with (format csv, header true)"
    result = subprocess.run(
        ["psql", "-X", "-d", database, "-v", "ON_ERROR_STOP=1", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def load_native_series(
    database: str, diagnostic_run_id: str, series_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    series_filter = ""
    if series_ids:
        values = ",".join(f"'{value}'" for value in sorted(series_ids))
        series_filter = f"and s.series_id in ({values})"
    query = f"""
        select
            s.series_id,
            s.source_id,
            s.source_dataset_id,
            s.source_dataset_name,
            coalesce(s.upstream_source, '') as upstream_source,
            coalesce(s.disease, '') as disease,
            s.metric,
            coalesce(s.unit, '') as unit,
            s.spatial_resolution,
            s.spatial_unit_type,
            s.temporal_resolution,
            coalesce(s.admin_0, '') as admin_0,
            coalesce(s.admin_1, '') as admin_1,
            coalesce(s.admin_2, '') as admin_2,
            coalesce(s.admin_3, '') as admin_3,
            coalesce(s.facility_name, '') as facility_name,
            coalesce(s.source_subunit, '') as source_subunit,
            concat_ws(' | ', nullif(s.admin_0, ''), nullif(s.admin_1, ''), nullif(s.admin_2, ''), nullif(s.admin_3, ''), nullif(s.facility_name, ''), nullif(s.source_subunit, '')) as location_name,
            s.first_period,
            s.last_period,
            s.observations,
            coalesce(s.expected_periods, s.observations) as expected_periods,
            coalesce(s.completeness, 0) as completeness,
            s.duplicate_periods,
            s.regular_time,
            o.period_start,
            o.value
        from metadata.diagnostic_series s
        join metadata.diagnostic_input i
          on i.diagnostic_run_id = s.diagnostic_run_id
         and i.series_id = s.series_id
        join gold.native_observation o on o.observation_id = i.observation_id
        where s.diagnostic_run_id = '{diagnostic_run_id}'
        {series_filter}
        order by s.series_id, o.period_start, o.observation_id
    """
    rows = _copy_query(database, query)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["series_id"]].append(row)
    series: list[dict[str, Any]] = []
    for series_id, records in grouped.items():
        first = records[0]
        series.append(
            {
                "series_id": series_id,
                "source_id": first["source_id"],
                "source_dataset_id": int(first["source_dataset_id"]),
                "source_dataset_name": first["source_dataset_name"],
                "upstream_source": first["upstream_source"],
                "disease": first["disease"],
                "metric": first["metric"],
                "unit": first["unit"],
                "spatial_resolution": first["spatial_resolution"],
                "spatial_unit_type": first["spatial_unit_type"],
                "temporal_resolution": first["temporal_resolution"],
                "admin_0": first["admin_0"],
                "admin_1": first["admin_1"],
                "admin_2": first["admin_2"],
                "admin_3": first["admin_3"],
                "facility_name": first["facility_name"],
                "source_subunit": first["source_subunit"],
                "location_name": first["location_name"] or "Unspecified",
                "first_period": first["first_period"],
                "last_period": first["last_period"],
                "observations": int(first["observations"]),
                "expected_periods": int(first["expected_periods"]),
                "completeness": float(first["completeness"]),
                "duplicate_periods": int(first["duplicate_periods"]),
                "regular_time": first["regular_time"] == "t",
                "observed": {
                    dt.date.fromisoformat(record["period_start"]): float(record["value"])
                    for record in records
                    if record["value"] and math.isfinite(float(record["value"]))
                },
            }
        )
    return series


def increment_period(value: dt.date, grain: str) -> dt.date:
    if grain == "week":
        return value + dt.timedelta(days=7)
    if grain == "month":
        year = value.year + (1 if value.month == 12 else 0)
        month = 1 if value.month == 12 else value.month + 1
        return dt.date(year, month, 1)
    if grain == "year":
        return dt.date(value.year + 1, value.month, value.day)
    raise ValueError(grain)


def regularize(series: dict[str, Any]) -> tuple[list[dt.date], list[float], list[bool]]:
    observed = series["observed"]
    dates: list[dt.date] = []
    values: list[float] = []
    present: list[bool] = []
    current = min(observed)
    final = max(observed)
    previous = observed[current]
    while current <= final:
        available = current in observed
        if available:
            previous = observed[current]
        dates.append(current)
        values.append(previous)
        present.append(available)
        current = increment_period(current, series["temporal_resolution"])
    return dates, values, present


def inclusion_reason(series: dict[str, Any]) -> tuple[bool, str]:
    if series["disease"] not in {"dengue", "malaria"}:
        return False, "non_target_disease"
    if series["temporal_resolution"] not in {"week", "month", "year"}:
        return False, "non_incident_or_snapshot_grain"
    if not series["regular_time"]:
        return False, "irregular_or_sparse_time"
    if series["duplicate_periods"]:
        return False, "duplicate_periods"
    if series["upstream_source"] == "opendengue_temporal":
        return False, "non_independent_opendengue_copy"
    return True, "included"
