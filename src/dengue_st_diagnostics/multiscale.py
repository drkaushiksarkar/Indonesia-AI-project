from __future__ import annotations

import re

import pandas as pd

from dengue_st_diagnostics.schema import valid_cases

SPATIAL_ORDER = {"admin0": 0, "admin1": 1, "admin2": 2, "admin3": 3}
TEMPORAL_ORDER = {"day": 0, "week": 1, "month": 2, "quarter": 3, "year": 4}


def _temporal_resolution(frame: pd.DataFrame) -> pd.Series:
    supplied = (
        frame["temporal_resolution"]
        .astype("string")
        .str.casefold()
        .str.replace(r"[^a-z]", "", regex=True)
    )
    mapping = {
        "d": "day",
        "daily": "day",
        "day": "day",
        "w": "week",
        "weekly": "week",
        "week": "week",
        "m": "month",
        "monthly": "month",
        "month": "month",
        "q": "quarter",
        "quarterly": "quarter",
        "quarter": "quarter",
        "y": "year",
        "annual": "year",
        "annually": "year",
        "year": "year",
    }
    result = supplied.map(mapping)
    days = (frame["period_end"] - frame["period_start"]).dt.days.add(1)
    inferred = pd.Series("year", index=frame.index, dtype="string")
    inferred = inferred.mask(days.le(92), "quarter")
    inferred = inferred.mask(days.le(32), "month")
    inferred = inferred.mask(days.le(8), "week")
    inferred = inferred.mask(days.le(1), "day")
    return result.fillna(inferred)


def _period_start(dates: pd.Series, grain: str) -> pd.Series:
    if grain == "day":
        return dates.dt.floor("D")
    if grain == "week":
        return dates.dt.to_period("W-SUN").dt.start_time
    if grain == "month":
        return dates.dt.to_period("M").dt.start_time
    if grain == "quarter":
        return dates.dt.to_period("Q").dt.start_time
    return dates.dt.to_period("Y").dt.start_time


def _period_end(starts: pd.Series, grain: str) -> pd.Series:
    if grain == "day":
        return starts
    if grain == "week":
        return starts + pd.Timedelta(days=6)
    if grain == "month":
        return starts + pd.offsets.MonthEnd(0)
    if grain == "quarter":
        return starts + pd.offsets.QuarterEnd(startingMonth=12)
    return starts + pd.offsets.YearEnd(0)


def _location(frame: pd.DataFrame, level: str) -> pd.Series:
    if level == "admin0":
        return frame["admin_0"].fillna("Indonesia")
    columns = [f"admin_{index}" for index in range(1, int(level[-1]) + 1)]
    values = frame[columns].astype("string")
    return values.apply(
        lambda row: " | ".join(str(value) for value in row if pd.notna(value)),
        axis=1,
    ).replace("", pd.NA)


def build_multiscale(frame: pd.DataFrame, temporal_grains: list[str]) -> pd.DataFrame:
    source = valid_cases(frame)
    if source.empty:
        return pd.DataFrame()
    source["native_temporal_resolution"] = _temporal_resolution(source)
    records: list[pd.DataFrame] = []
    partition_columns = ["source", "admin_level", "native_temporal_resolution"]
    for keys, partition in source.groupby(partition_columns, dropna=False):
        source_name, native_spatial, native_temporal = keys
        spatial_order = SPATIAL_ORDER.get(str(native_spatial), 0)
        temporal_order = TEMPORAL_ORDER.get(str(native_temporal), 4)
        spatial_targets = [
            level for level, order in SPATIAL_ORDER.items() if order <= spatial_order
        ]
        temporal_targets = [
            grain for grain in temporal_grains if TEMPORAL_ORDER.get(grain, 4) >= temporal_order
        ]
        for spatial_target in spatial_targets:
            location = _location(partition, spatial_target)
            valid = partition.loc[location.notna()].copy()
            if valid.empty:
                continue
            valid["location_name"] = location.loc[valid.index].astype("string")
            for temporal_target in temporal_targets:
                valid["period_start_derived"] = _period_start(
                    valid["period_start"],
                    temporal_target,
                )
                groups = ["location_name", "period_start_derived"]
                aggregated = (
                    valid.groupby(groups, dropna=False)
                    .agg(
                        cases=("cases", "sum"),
                        population=("population", lambda values: values.sum(min_count=1)),
                        contributing_records=("cases", "size"),
                        contributing_locations=("location_id", "nunique"),
                    )
                    .reset_index()
                )
                aggregated = aggregated.rename(columns={"period_start_derived": "period_start"})
                aggregated["period_end"] = _period_end(
                    aggregated["period_start"],
                    temporal_target,
                )
                aggregated["source"] = source_name
                aggregated["native_spatial_resolution"] = native_spatial
                aggregated["native_temporal_resolution"] = native_temporal
                aggregated["spatial_resolution"] = spatial_target
                aggregated["temporal_resolution"] = temporal_target
                aggregated["panel_id"] = (
                    str(source_name)
                    + "__"
                    + str(native_spatial)
                    + "__"
                    + str(native_temporal)
                    + "__"
                    + spatial_target
                    + "__"
                    + temporal_target
                )
                records.append(aggregated)
    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True)
    result["panel_id"] = result["panel_id"].str.replace(r"[^a-zA-Z0-9_]+", "_", regex=True)
    result["incidence_per_100k"] = 100_000 * result["cases"] / result["population"]
    result.loc[result["population"].le(0), "incidence_per_100k"] = pd.NA
    return result


def grain_inventory(multiscale: pd.DataFrame) -> pd.DataFrame:
    if multiscale.empty:
        return pd.DataFrame()
    grouped = multiscale.groupby(
        [
            "panel_id",
            "source",
            "native_spatial_resolution",
            "native_temporal_resolution",
            "spatial_resolution",
            "temporal_resolution",
        ],
        dropna=False,
    )
    result = grouped.agg(
        rows=("cases", "size"),
        locations=("location_name", "nunique"),
        periods=("period_start", "nunique"),
        start_date=("period_start", "min"),
        end_date=("period_end", "max"),
        total_cases=("cases", "sum"),
        population_coverage=("population", lambda values: values.notna().mean()),
    )
    return result.reset_index()


def sanitize_panel_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value)
