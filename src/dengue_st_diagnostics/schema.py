from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "source",
    "source_record_id",
    "location_id",
    "location_name",
    "admin_level",
    "admin_0",
    "admin_1",
    "admin_2",
    "admin_3",
    "period_start",
    "period_end",
    "temporal_resolution",
    "cases",
    "population",
    "case_definition",
    "confirmation_status",
    "native_record",
]


def empty_cases() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    result = result[CANONICAL_COLUMNS]
    result["period_start"] = pd.to_datetime(result["period_start"], errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    result["cases"] = pd.to_numeric(result["cases"], errors="coerce")
    result["population"] = pd.to_numeric(result["population"], errors="coerce")
    result["native_record"] = result["native_record"].fillna(True).astype(bool)
    text_columns = [
        column
        for column in CANONICAL_COLUMNS
        if column
        not in {
            "period_start",
            "period_end",
            "cases",
            "population",
            "native_record",
        }
    ]
    for column in text_columns:
        result[column] = result[column].astype("string")
    return result


def combine(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    available = [canonicalize(frame) for frame in frames if not frame.empty]
    if not available:
        return empty_cases()
    result = pd.concat(available, ignore_index=True)
    key = ["source", "source_record_id", "location_id", "period_start", "period_end"]
    return result.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def valid_cases(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame["period_start"].notna()
        & frame["cases"].notna()
        & np.isfinite(frame["cases"].astype(float))
        & frame["cases"].ge(0)
    )
    return frame.loc[mask].copy()
