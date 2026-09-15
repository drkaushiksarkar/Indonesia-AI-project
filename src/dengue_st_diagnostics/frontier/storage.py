from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from dengue_st_diagnostics.frontier.models import SPECS


def _run(database: str, sql: str) -> None:
    subprocess.run(
        ["psql", "-X", "-d", database, "-v", "ON_ERROR_STOP=1"],
        input=sql,
        check=True,
        capture_output=True,
        text=True,
    )


def apply_schema(database: str, path: Path) -> None:
    subprocess.run(
        ["psql", "-X", "-d", database, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def register_run(
    database: str,
    run_id: str,
    parent_run_id: str,
    diagnostic_run_id: str,
    started_at: str,
    snapshot: str,
    configuration: dict[str, Any],
) -> None:
    payload = json.dumps(configuration, sort_keys=True)
    _run(
        database,
        f"""
        insert into metadata.univariate_frontier_run (
            frontier_run_id, parent_predictability_run_id, diagnostic_run_id,
            started_at, status, input_relation, input_snapshot_sha256,
            code_version, configuration
        ) values (
            '{run_id}', '{parent_run_id}', '{diagnostic_run_id}', '{started_at}',
            'running', 'gold.native_observation', '{snapshot}', '2.0.0',
            $json${payload}$json$::jsonb
        );
        """,
    )


def register_models(database: str) -> None:
    values: list[str] = []
    for spec in SPECS:
        scopes = ",".join(f"'{scope}'" for scope in spec.scopes)
        description = spec.model_id.replace("_", " ")
        values.append(
            f"('{spec.model_id}', '{spec.family}', array[{scopes}]::text[], {str(spec.intermittent_only).lower()}, {str(spec.pooled).lower()}, '{description}')"
        )
    _run(
        database,
        f"""
        insert into metadata.univariate_frontier_model (
            model_id, model_family, model_scopes, intermittent_only, pooled,
            model_description
        ) values {", ".join(values)}
        on conflict (model_id) do update set
            model_family = excluded.model_family,
            model_scopes = excluded.model_scopes,
            intermittent_only = excluded.intermittent_only,
            pooled = excluded.pooled,
            model_description = excluded.model_description;
        """,
    )


def _copy(database: str, table: str, path: Path, columns: list[str]) -> None:
    escaped = str(path.resolve()).replace("'", "''")
    _run(
        database,
        f"\\copy {table} ({', '.join(columns)}) from '{escaped}' with (format csv, header true, null '')\n",
    )


def load_results(
    database: str,
    root: Path,
    results: dict[str, list[dict[str, Any]]],
    artifacts: list[dict[str, Any]],
) -> None:
    mapping = (
        ("metadata.univariate_frontier_series", "series", "series.csv"),
        ("gold.univariate_frontier_fold", "folds", "fold_predictions.csv"),
        ("gold.univariate_frontier_model_metric", "metrics", "model_metrics.csv"),
        ("gold.univariate_frontier", "frontiers", "frontiers.csv"),
        ("gold.univariate_frontier_learning_curve", "learning_curves", "learning_curves.csv"),
        ("gold.univariate_frontier_aggregate", "aggregates", "aggregate_metrics.csv"),
    )
    for table, key, filename in mapping:
        rows = results[key]
        if rows:
            _copy(database, table, root / "quantitative" / filename, list(rows[0]))
    if artifacts:
        _copy(
            database,
            "metadata.univariate_frontier_artifact",
            root / "quantitative" / "artifact_catalog.csv",
            list(artifacts[0]),
        )


def complete_run(database: str, run_id: str, completed_at: str, metadata: dict[str, Any]) -> None:
    payload = json.dumps(metadata, sort_keys=True)
    _run(
        database,
        f"""
        update metadata.univariate_frontier_run
        set completed_at = '{completed_at}', status = 'completed',
            run_metadata = $json${payload}$json$::jsonb
        where frontier_run_id = '{run_id}';
        """,
    )


def fail_run(database: str, run_id: str, completed_at: str, reason: str) -> None:
    payload = json.dumps({"failure_reason": reason}, sort_keys=True)
    _run(
        database,
        f"""
        update metadata.univariate_frontier_run
        set completed_at = '{completed_at}', status = 'failed',
            run_metadata = $json${payload}$json$::jsonb
        where frontier_run_id = '{run_id}';
        """,
    )
