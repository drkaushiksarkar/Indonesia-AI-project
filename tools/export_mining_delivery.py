"""Export analysis files, coverage and checksums; no raw registers or failed source charts."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("../outputs/extensive_mining")
if __name__ == "__main__":
    out = ROOT / "normalized"
    data = pd.read_parquet(out / "observations.parquet")
    cases = data[
        (data.metric == "cases")
        & (data.sex == "all")
        & ~data.quality.isin(["source_validation_failed", "archived_snapshot"])
    ].copy()
    cases.drop(columns=["raw_path"], errors="ignore").to_csv(
        out / "case_series_with_quality_flags.csv", index=False
    )
    features = pd.read_parquet(out / "monthly_features.parquet")
    features = features[features.target_history_complete]
    features.to_parquet(out / "forecast_candidates.parquet", index=False)
    features.drop(columns=["raw_path"], errors="ignore").to_csv(
        out / "forecast_candidates.csv", index=False
    )
    shutil.copyfile("MINING_AUDIT.md", ROOT / "MINING_AUDIT.md")
    manifest = []
    for name in [
        "case_series_with_quality_flags.csv",
        "forecast_candidates.parquet",
        "forecast_candidates.csv",
        "series_readiness.csv",
        "baseline_comparison.csv",
        "summary.json",
    ]:
        path = out / name
        manifest.append(
            {
                "file": "normalized/" + name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (ROOT / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Case rows with quality flags", len(cases), "complete-history candidates", len(features))
