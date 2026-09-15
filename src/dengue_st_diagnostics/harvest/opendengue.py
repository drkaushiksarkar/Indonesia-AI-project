from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import download, sha256
from dengue_st_diagnostics.schema import canonicalize, combine, empty_cases


def _archive_name(extract: str, version: str) -> str:
    suffix = version.replace(".", "_")
    return f"{extract}_extract_V{suffix}.zip"


def _value(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    lookup = {str(column).casefold(): column for column in frame.columns}
    for name in names:
        column = lookup.get(name.casefold())
        if column is not None:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _admin_level(frame: pd.DataFrame) -> pd.Series:
    supplied = _value(frame, ["S_res", "spatial_resolution"])
    inferred = pd.Series("admin0", index=frame.index, dtype="string")
    inferred = inferred.mask(_value(frame, ["adm_1_name"]).notna(), "admin1")
    inferred = inferred.mask(_value(frame, ["adm_2_name"]).notna(), "admin2")
    cleaned = supplied.astype("string").str.lower().str.replace(r"[^0-9a-z]", "", regex=True)
    mapped = cleaned.map({"0": "admin0", "1": "admin1", "2": "admin2"})
    mapped = mapped.fillna(cleaned.where(cleaned.str.startswith("admin")))
    return mapped.fillna(inferred)


def _normalize_chunk(frame: pd.DataFrame, extract: str) -> pd.DataFrame:
    country = _value(frame, ["adm_0_name", "country", "country_name"])
    selected = frame.loc[country.astype("string").str.casefold().eq("indonesia")].copy()
    if selected.empty:
        return empty_cases()
    admin_0 = _value(selected, ["adm_0_name", "country", "country_name"])
    admin_1 = _value(selected, ["adm_1_name", "province"])
    admin_2 = _value(selected, ["adm_2_name", "district"])
    location = _value(selected, ["full_name", "location_name"])
    location = location.fillna(admin_2).fillna(admin_1).fillna(admin_0)
    location_id = _value(
        selected,
        ["FAO_GAUL_code", "RNE_iso_code", "location_id", "full_name"],
    ).fillna(location)
    record_id = _value(selected, ["UUID", "uuid", "record_id"])
    index_values = pd.Series(selected.index.astype(str), index=selected.index, dtype="string")
    record_id = record_id.fillna(index_values)
    result = pd.DataFrame(
        {
            "source": f"opendengue_{extract.casefold()}",
            "source_record_id": record_id.astype("string"),
            "location_id": location_id.astype("string"),
            "location_name": location.astype("string"),
            "admin_level": _admin_level(selected),
            "admin_0": admin_0,
            "admin_1": admin_1,
            "admin_2": admin_2,
            "admin_3": pd.NA,
            "period_start": _value(selected, ["calendar_start_date", "start_date"]),
            "period_end": _value(selected, ["calendar_end_date", "end_date"]),
            "temporal_resolution": _value(selected, ["T_res", "temporal_resolution"]),
            "cases": _value(selected, ["Dengue_total", "dengue_total", "cases"]),
            "population": pd.NA,
            "case_definition": _value(
                selected,
                ["case_definition_standardised", "case_definition"],
            ),
            "confirmation_status": pd.NA,
            "native_record": True,
        }
    )
    return canonicalize(result)


def _read_archive(path: Path, extract: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.casefold().endswith(".csv")]
        for member in members:
            with archive.open(member) as handle:
                for chunk in pd.read_csv(handle, chunksize=250_000, low_memory=False):
                    normalized = _normalize_chunk(chunk, extract)
                    if not normalized.empty:
                        frames.append(normalized)
    return combine(frames)


def harvest_opendengue(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = settings.section("opendengue")
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for extract in config["extracts"]:
        name = _archive_name(extract, config["version"])
        url = f"{config['base_url'].rstrip('/')}/{name}"
        path = settings.paths.raw / "opendengue" / name
        status = "cached" if path.exists() else "disabled"
        error = ""
        try:
            if enabled and not path.exists():
                download(session, url, path)
                status = "downloaded"
            if path.exists():
                frames.append(_read_archive(path, extract))
                status = "parsed"
        except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "source": f"opendengue_{extract.casefold()}",
                "url": url,
                "path": str(path),
                "status": status,
                "sha256": sha256(path) if path.exists() else "",
                "error": error,
            }
        )
    return combine(frames), pd.DataFrame.from_records(records)
