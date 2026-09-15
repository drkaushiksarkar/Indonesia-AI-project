from __future__ import annotations

import numpy as np
import pandas as pd


def _result(
    scope: str,
    check: str,
    value: float,
    denominator: float,
    severity: str,
) -> dict[str, object]:
    rate = value / denominator if denominator else np.nan
    return {
        "scope": scope,
        "check": check,
        "value": value,
        "denominator": denominator,
        "rate": rate,
        "severity": severity,
    }


def _expected_periods(group: pd.DataFrame, grain: str) -> int:
    start = group["period_start"].min()
    end = group["period_start"].max()
    if pd.isna(start) or pd.isna(end):
        return 0
    frequencies = {"day": "D", "week": "W-SUN", "month": "MS", "quarter": "QS", "year": "YS"}
    return len(pd.date_range(start, end, freq=frequencies.get(grain, "YS")))


def diagnose_quality(
    cases: pd.DataFrame, multiscale: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    rows = len(cases)
    if rows:
        records.extend(
            [
                _result(
                    "all",
                    "missing_period_start",
                    float(cases["period_start"].isna().sum()),
                    rows,
                    "critical",
                ),
                _result(
                    "all", "missing_cases", float(cases["cases"].isna().sum()), rows, "critical"
                ),
                _result(
                    "all", "negative_cases", float(cases["cases"].lt(0).sum()), rows, "critical"
                ),
                _result(
                    "all",
                    "missing_location",
                    float(cases["location_name"].isna().sum()),
                    rows,
                    "high",
                ),
                _result(
                    "all",
                    "missing_population",
                    float(cases["population"].isna().sum()),
                    rows,
                    "medium",
                ),
            ]
        )
        keys = ["source", "location_id", "period_start", "period_end"]
        duplicates = cases.duplicated(keys, keep=False).sum()
        records.append(_result("all", "duplicate_grain_keys", float(duplicates), rows, "critical"))
        overlap = cases["period_end"].lt(cases["period_start"]).sum()
        records.append(_result("all", "reversed_intervals", float(overlap), rows, "critical"))
    coverage: list[dict[str, object]] = []
    if not multiscale.empty:
        groups = multiscale.groupby(["panel_id", "location_name"], dropna=False)
        for (panel_id, location), group in groups:
            grain = str(group["temporal_resolution"].iloc[0])
            observed = int(group["period_start"].nunique())
            expected = _expected_periods(group, grain)
            coverage.append(
                {
                    "panel_id": panel_id,
                    "location_name": location,
                    "temporal_resolution": grain,
                    "observed_periods": observed,
                    "expected_periods": expected,
                    "completeness": observed / expected if expected else np.nan,
                    "zero_rate": group["cases"].eq(0).mean(),
                    "population_coverage": group["population"].notna().mean(),
                    "first_period": group["period_start"].min(),
                    "last_period": group["period_start"].max(),
                }
            )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(coverage)
