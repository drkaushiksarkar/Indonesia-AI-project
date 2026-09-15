from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from dengue_st_diagnostics.schema import canonicalize, combine, empty_cases

MONTHS = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}


def _name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _read_bytes(data: bytes, suffix: str) -> list[pd.DataFrame]:
    if suffix == ".csv":
        for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                return [pd.read_csv(io.BytesIO(data), encoding=encoding)]
            except UnicodeDecodeError:
                continue
        return []
    if suffix in {".xls", ".xlsx"}:
        return list(pd.read_excel(io.BytesIO(data), sheet_name=None).values())
    if suffix in {".json", ".geojson"}:
        value = pd.read_json(io.BytesIO(data))
        return [value]
    return []


def _read(path: Path) -> list[pd.DataFrame]:
    if path.suffix.casefold() == ".zip":
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                suffix = Path(name).suffix.casefold()
                if suffix in {".csv", ".xls", ".xlsx", ".json", ".geojson"}:
                    frames.extend(_read_bytes(archive.read(name), suffix))
        return frames
    return _read_bytes(path.read_bytes(), path.suffix.casefold())


def _column(frame: pd.DataFrame, patterns: list[str]) -> pd.Series:
    for pattern in patterns:
        for column in frame.columns:
            if re.search(pattern, str(column)):
                return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = [pd.to_numeric(frame[column], errors="coerce") for column in columns]
    if not values:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.concat(values, axis=1).sum(axis=1, min_count=1)


def _measure_columns(frame: pd.DataFrame) -> list[str]:
    include = re.compile(r"(dengue|dbd|demam_berdarah|jumlah_kasus|jumlah_penderita|cases?)")
    exclude = re.compile(r"(kematian|meninggal|death|fatal|rate|rasio|persen|cfr|ir)")
    values = [str(column) for column in frame.columns if include.search(str(column))]
    values = [column for column in values if not exclude.search(column)]
    return [
        column
        for column in values
        if pd.to_numeric(frame[column], errors="coerce").notna().mean() >= 0.25
    ]


def _wide_years(frame: pd.DataFrame) -> pd.DataFrame:
    year_columns = [
        str(column) for column in frame.columns if re.fullmatch(r"20\d{2}", str(column))
    ]
    if len(year_columns) < 2:
        return frame
    identifiers = [column for column in frame.columns if str(column) not in year_columns]
    return frame.melt(
        id_vars=identifiers,
        value_vars=year_columns,
        var_name="tahun",
        value_name="jumlah_kasus",
    )


def _dates(frame: pd.DataFrame, path: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    date = pd.to_datetime(
        _column(frame, [r"^(tanggal|date|period_start|waktu)$"]),
        errors="coerce",
        dayfirst=True,
    )
    year = pd.to_numeric(_column(frame, [r"^(tahun|year)$"]), errors="coerce")
    inferred = re.search(r"(?:19|20)\d{2}", path.name)
    if inferred is not None:
        year = year.fillna(int(inferred.group()))
    month_raw = _column(frame, [r"^(bulan|month)$"])
    month = pd.to_numeric(month_raw, errors="coerce")
    month = month.fillna(month_raw.astype("string").str.casefold().map(MONTHS))
    week = pd.to_numeric(_column(frame, [r"^(minggu|week|pekan)$"]), errors="coerce")
    annual = pd.to_datetime(year.astype("Int64").astype("string") + "-01-01", errors="coerce")
    monthly = pd.to_datetime(
        year.astype("Int64").astype("string")
        + "-"
        + month.astype("Int64").astype("string")
        + "-01",
        errors="coerce",
    )
    weekly = pd.to_datetime(
        year.astype("Int64").astype("string")
        + "-W"
        + week.astype("Int64").astype("string").str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    start = date.fillna(weekly).fillna(monthly).fillna(annual)
    resolution = pd.Series("year", index=frame.index, dtype="string")
    resolution = resolution.mask(month.notna(), "month").mask(week.notna(), "week")
    resolution = resolution.mask(date.notna(), "day")
    end = start.copy()
    end = end.mask(resolution.eq("day"), start)
    end = end.mask(resolution.eq("week"), start + pd.Timedelta(days=6))
    end = end.mask(resolution.eq("month"), start + pd.offsets.MonthEnd(0))
    end = end.mask(resolution.eq("year"), start + pd.offsets.YearEnd(0))
    return start, end, resolution


def _normalize(frame: pd.DataFrame, path: Path, index: int) -> pd.DataFrame:
    value = frame.copy()
    value.columns = [_name(column) for column in value.columns]
    value = _wide_years(value)
    measures = _measure_columns(value)
    if not measures:
        return empty_cases()
    cases = _numeric(value, measures)
    admin_1 = _column(value, [r"^(provinsi|province|admin_1)"])
    admin_2 = _column(value, [r"^(kabupaten|kota|kabupaten_kota|regency|district|admin_2)"])
    admin_3 = _column(value, [r"^(kecamatan|puskesmas|desa|kelurahan|admin_3)"])
    admin_level = pd.Series("admin0", index=value.index, dtype="string")
    admin_level = admin_level.mask(admin_1.notna(), "admin1")
    admin_level = admin_level.mask(admin_2.notna(), "admin2")
    admin_level = admin_level.mask(admin_3.notna(), "admin3")
    location = admin_3.fillna(admin_2).fillna(admin_1).fillna("Indonesia")
    start, end, resolution = _dates(value, path)
    population = pd.to_numeric(
        _column(value, [r"^(population|populasi|jumlah_penduduk|penduduk)$"]),
        errors="coerce",
    )
    result = pd.DataFrame(
        {
            "source": f"local_{_name(path.stem)}_{index}",
            "source_record_id": value.index.astype(str),
            "location_id": location.astype("string"),
            "location_name": location.astype("string"),
            "admin_level": admin_level,
            "admin_0": "Indonesia",
            "admin_1": admin_1,
            "admin_2": admin_2,
            "admin_3": admin_3,
            "period_start": start,
            "period_end": end,
            "temporal_resolution": resolution,
            "cases": cases,
            "population": population,
            "case_definition": pd.NA,
            "confirmation_status": pd.NA,
            "native_record": True,
        }
    )
    return canonicalize(result)


def normalize_paths(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for path in paths:
        status = "parsed"
        error = ""
        row_count = 0
        try:
            normalized = [_normalize(frame, path, index) for index, frame in enumerate(_read(path))]
            result = combine(normalized)
            row_count = len(result)
            if not result.empty:
                frames.append(result)
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "path": str(path),
                "status": status,
                "normalized_rows": row_count,
                "error": error,
            }
        )
    return combine(frames), pd.DataFrame.from_records(records)
