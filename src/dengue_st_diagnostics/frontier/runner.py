from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from pathlib import Path

from dengue_st_diagnostics.frontier.artifacts import save_outputs
from dengue_st_diagnostics.frontier.evaluate import (
    GRAIN_SETTINGS,
    MODEL_COUNT,
    evaluate,
)
from dengue_st_diagnostics.frontier.io import (
    diagnostic_run_for_parent,
    latest_parent_run,
    load_eligible_series,
)
from dengue_st_diagnostics.frontier.storage import (
    apply_schema,
    complete_run,
    fail_run,
    load_results,
    register_models,
    register_run,
)
from dengue_st_diagnostics.predictability.io import diagnostic_snapshot


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="indonesia_vector_lake")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/univariate_frontier"))
    parser.add_argument("--schema", type=Path, default=Path("sql/medallion.sql"))
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    started = dt.datetime.now(dt.UTC)
    run_id = str(uuid.uuid4())
    parent_run_id = arguments.parent_run_id or latest_parent_run(arguments.database)
    diagnostic_run_id = diagnostic_run_for_parent(arguments.database, parent_run_id)
    snapshot = diagnostic_snapshot(arguments.database, diagnostic_run_id)
    root = arguments.output_root.resolve() / run_id
    configuration = {
        "design": "expanding_window_rolling_origin_with_nested_training_only_selection",
        "eligibility": "parent_run_robust_or_moderate_series",
        "model_count": MODEL_COUNT,
        "tasks": ["forecast_point", "forecast_probabilistic", "outbreak", "risk"],
        "grains": GRAIN_SETTINGS,
        "probability_levels": [0.1, 0.25, 0.5, 0.75, 0.9],
        "uncertainty": "online_residual_quantiles_with_training_only_fallback",
        "outbreak_definition": "training_seasonal_mean_plus_2sd_else_training_p90",
        "risk_definition": "training_tertiles_within_series",
        "selection": ["posthoc_library_oracle", "online_selector", "online_weighted_ensemble"],
        "global_model": "pooled_target_only_ridge_with_training_targets_not_after_origin",
        "learning_curve_windows": [12, 24, 36, 60, 120],
    }
    apply_schema(arguments.database, arguments.schema)
    register_run(
        arguments.database,
        run_id,
        parent_run_id,
        diagnostic_run_id,
        started.isoformat(),
        snapshot,
        configuration,
    )
    register_models(arguments.database)
    try:
        series = load_eligible_series(arguments.database, diagnostic_run_id, parent_run_id)
        results = evaluate(series, run_id)
        artifacts, manifest_path = save_outputs(root, results, run_id)
        summary = {
            "eligible_series_count": len(series),
            "model_count": MODEL_COUNT,
            "fold_prediction_count": len(results["folds"]),
            "model_metric_count": len(results["metrics"]),
            "frontier_count": len(results["frontiers"]),
            "learning_curve_count": len(results["learning_curves"]),
            "aggregate_metric_count": len(results["aggregates"]),
            "artifact_count": len(artifacts),
            "manifest_path": str(manifest_path),
        }
        root.joinpath("run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        load_results(arguments.database, root, results, artifacts)
        complete_run(arguments.database, run_id, dt.datetime.now(dt.UTC).isoformat(), summary)
        arguments.output_root.resolve().joinpath("latest.json").write_text(
            json.dumps(
                {
                    "frontier_run_id": run_id,
                    "parent_predictability_run_id": parent_run_id,
                    "diagnostic_run_id": diagnostic_run_id,
                    "root": str(root),
                    "manifest": str(manifest_path),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except Exception as error:
        fail_run(
            arguments.database,
            run_id,
            dt.datetime.now(dt.UTC).isoformat(),
            f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
