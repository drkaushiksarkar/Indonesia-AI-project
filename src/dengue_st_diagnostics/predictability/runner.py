from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from pathlib import Path

from dengue_st_diagnostics.predictability.artifacts import save_outputs
from dengue_st_diagnostics.predictability.evaluate import GRAIN_SETTINGS, evaluate
from dengue_st_diagnostics.predictability.io import (
    diagnostic_snapshot,
    latest_diagnostic_run,
    load_native_series,
)
from dengue_st_diagnostics.predictability.storage import (
    apply_schema,
    complete_run,
    load_results,
    register_models,
    register_run,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="indonesia_vector_lake")
    parser.add_argument("--diagnostic-run-id")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/predictability"))
    parser.add_argument("--schema", type=Path, default=Path("sql/medallion.sql"))
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    started = dt.datetime.now(dt.UTC)
    run_id = str(uuid.uuid4())
    diagnostic_run_id = arguments.diagnostic_run_id or latest_diagnostic_run(arguments.database)
    snapshot = diagnostic_snapshot(arguments.database, diagnostic_run_id)
    root = arguments.output_root.resolve() / run_id
    configuration = {
        "design": "expanding_window_rolling_origin",
        "selection": ["posthoc_oracle_frontier", "online_selector", "median_ensemble"],
        "tasks": ["forecast", "outbreak", "risk"],
        "grains": GRAIN_SETTINGS,
        "outbreak_definition": "training_seasonal_mean_plus_2sd_else_training_p90",
        "risk_definition": "training_tertiles_within_series",
        "duplicate_policy": "exclude_opendengue_temporal_copy",
        "imputation_policy": "past_only_forward_fill",
    }
    series = load_native_series(arguments.database, diagnostic_run_id)
    results = evaluate(series, run_id)
    artifacts, manifest_path = save_outputs(root, results, run_id)
    summary = {
        "series_catalog_count": len(results["series"]),
        "evaluated_series_count": sum(
            row["include_status"] == "evaluated" for row in results["series"]
        ),
        "robust_series_count": sum(row["validation_tier"] == "robust" for row in results["series"]),
        "moderate_series_count": sum(
            row["validation_tier"] == "moderate" for row in results["series"]
        ),
        "pilot_series_count": sum(row["validation_tier"] == "pilot" for row in results["series"]),
        "fold_prediction_count": len(results["folds"]),
        "model_metric_count": len(results["metrics"]),
        "frontier_count": len(results["frontiers"]),
        "aggregate_metric_count": len(results["aggregates"]),
        "artifact_count": len(artifacts),
        "manifest_path": str(manifest_path),
    }
    root.joinpath("run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    apply_schema(arguments.database, arguments.schema)
    register_run(
        arguments.database,
        run_id,
        diagnostic_run_id,
        started.isoformat(),
        snapshot,
        configuration,
    )
    register_models(arguments.database)
    load_results(arguments.database, root, results, artifacts)
    complete_run(arguments.database, run_id, dt.datetime.now(dt.UTC).isoformat(), summary)
    latest = arguments.output_root.resolve() / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "predictability_run_id": run_id,
                "diagnostic_run_id": diagnostic_run_id,
                "root": str(root),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
