from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from dengue_st_diagnostics.predictability.models import MODEL_REGISTRY


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
    diagnostic_run_id: str,
    started_at: str,
    snapshot: str,
    configuration: dict[str, Any],
) -> None:
    payload = json.dumps(configuration, sort_keys=True)
    sql = f"""
        insert into metadata.predictability_run (
            predictability_run_id, diagnostic_run_id, started_at, status,
            input_relation, input_snapshot_sha256, code_version, configuration
        ) values (
            '{run_id}', '{diagnostic_run_id}', '{started_at}', 'running',
            'gold.native_observation', '{snapshot}', '1.0.0', $json${payload}$json$::jsonb
        );
    """
    _run(database, sql)


def register_models(database: str) -> None:
    descriptions = {
        "naive": "Last observation carried forward",
        "seasonal_naive": "Last observation from the matching seasonal phase",
        "historical_mean": "Expanding-window historical mean",
        "historical_median": "Expanding-window historical median",
        "moving_average": "Recursive short moving average",
        "drift": "Random walk with global drift",
        "ses_optimized": "Simple exponential smoothing with training-only alpha selection",
        "holt_damped": "Damped Holt trend with training-only grid selection",
        "autoregression_ridge": "Regularized autoregression with short and seasonal lags",
        "analog_knn": "Nearest historical trajectory analog forecast",
        "croston_sba": "Bias-adjusted Croston intermittent-demand forecast",
        "tsb": "Teunter-Syntetos-Babai intermittent-demand forecast",
        "median_ensemble": "Median of non-baseline model forecasts",
        "online_selector": "Model selected from only previously observed fold errors",
    }
    values = []
    for model_id, family, intermittent_only in MODEL_REGISTRY:
        description = descriptions[model_id].replace("'", "''")
        values.append(
            f"('{model_id}', '{family}', {str(intermittent_only).lower()}, '{description}')"
        )
    sql = f"""
        insert into metadata.predictability_model (
            model_id, model_family, intermittent_only, model_description
        ) values {", ".join(values)}
        on conflict (model_id) do update set
            model_family = excluded.model_family,
            intermittent_only = excluded.intermittent_only,
            model_description = excluded.model_description;
    """
    _run(database, sql)


def _copy(database: str, table: str, path: Path, columns: list[str]) -> None:
    escaped = str(path.resolve()).replace("'", "''")
    sql = f"\\copy {table} ({', '.join(columns)}) from '{escaped}' with (format csv, header true, null '')\n"
    _run(database, sql)


def load_results(
    database: str,
    root: Path,
    results: dict[str, list[dict[str, Any]]],
    artifacts: list[dict[str, Any]],
) -> None:
    mapping = (
        ("metadata.predictability_series", "series", "series_predictability.csv"),
        ("gold.predictability_fold", "folds", "fold_predictions.csv"),
        ("gold.predictability_model_metric", "metrics", "model_metrics.csv"),
        ("gold.predictability_frontier", "frontiers", "predictability_frontiers.csv"),
        ("gold.predictability_aggregate", "aggregates", "aggregate_metrics.csv"),
    )
    for table, key, filename in mapping:
        rows = results[key]
        if rows:
            _copy(database, table, root / "quantitative" / filename, list(rows[0]))
    if artifacts:
        path = root / "quantitative" / "artifact_catalog.csv"
        _copy(database, "metadata.predictability_artifact", path, list(artifacts[0]))


def complete_run(database: str, run_id: str, completed_at: str, metadata: dict[str, Any]) -> None:
    payload = json.dumps(metadata, sort_keys=True)
    sql = f"""
        update metadata.predictability_run
        set completed_at = '{completed_at}', status = 'completed', run_metadata = $json${payload}$json$::jsonb
        where predictability_run_id = '{run_id}';
    """
    _run(database, sql)
