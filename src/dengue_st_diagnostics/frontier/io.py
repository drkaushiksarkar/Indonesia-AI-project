from __future__ import annotations

from typing import Any

from dengue_st_diagnostics.predictability.io import load_native_series, psql


def latest_parent_run(database: str) -> str:
    return psql(
        database,
        "select predictability_run_id from metadata.predictability_run where status='completed' order by completed_at desc limit 1",
    )


def diagnostic_run_for_parent(database: str, parent_run_id: str) -> str:
    return psql(
        database,
        f"select diagnostic_run_id from metadata.predictability_run where predictability_run_id='{parent_run_id}'",
    )


def eligible_ids(database: str, parent_run_id: str) -> set[str]:
    value = psql(
        database,
        f"select string_agg(series_id, ',') from metadata.predictability_series where predictability_run_id='{parent_run_id}' and validation_tier in ('robust','moderate')",
    )
    return {item for item in value.split(",") if item}


def load_eligible_series(
    database: str, diagnostic_run_id: str, parent_run_id: str
) -> list[dict[str, Any]]:
    return load_native_series(database, diagnostic_run_id, eligible_ids(database, parent_run_id))
