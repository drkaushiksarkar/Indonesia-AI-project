from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import download, request_json, safe_name, sha256
from dengue_st_diagnostics.schema import canonicalize

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
MALANG_CATALOG = "https://data.malangkota.go.id"
PURBALINGGA_CATALOG = "https://data.jatengprov.go.id"
PURBALINGGA_PACKAGE = "tabel-21"
MALANG_SOURCE = "Malang City Open Data Puskesmas"
PURBALINGGA_SOURCE = "Purbalingga Open Data Puskesmas"
SUBUNITS_ARJOWINANGUN = ["Arjowinangun", "Bumiayu", "Mergosono", "Tlogowaru"]
SUBUNITS_MOJOLANGU = ["Mojolangu", "Tunjungsekar", "Tunggulwulung", "Tasikmadu"]
SUBUNITS_JANTI = ["Bandungrejosari", "Sukun", "Tanjungrejo"]


def _period(year: int, month: int | None) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if month is None:
        start = pd.Timestamp(year=year, month=1, day=1)
        return start, pd.Timestamp(year=year, month=12, day=31), "year"
    start = pd.Timestamp(year=year, month=month, day=1)
    return start, start + pd.offsets.MonthEnd(0), "month"


def _month_year(value: str) -> tuple[int, int]:
    lowered = value.casefold()
    year_match = re.search(r"\b(20\d{2})\b", lowered)
    month = next((number for name, number in MONTHS.items() if name in lowered), None)
    if year_match is None or month is None:
        raise ValueError("month or year unavailable")
    return int(year_match.group(1)), month


def _record(
    source: str,
    record_id: str,
    province: str,
    district: str,
    puskesmas: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    temporal_resolution: str,
    cases: float,
    source_url: str,
    raw_path: Path,
    subunit: str | None = None,
    kecamatan: str | None = None,
    deaths: float | None = None,
    male_cases: float | None = None,
    female_cases: float | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_record_id": record_id,
        "province": province,
        "district_city": district,
        "kecamatan": kecamatan,
        "puskesmas": puskesmas,
        "subunit": subunit,
        "aggregation_level": "kelurahan" if subunit else "puskesmas",
        "period_start": period_start,
        "period_end": period_end,
        "temporal_resolution": temporal_resolution,
        "cases": cases,
        "deaths": deaths,
        "male_cases": male_cases,
        "female_cases": female_cases,
        "case_definition": "reported DBD cases",
        "confirmation_status": "not stated in source resource",
        "source_url": source_url,
        "source_format": raw_path.suffix.casefold().lstrip("."),
        "raw_path": str(raw_path),
        "privacy_class": "aggregate",
    }


def _extension(resource: dict[str, Any]) -> str:
    declared = str(resource.get("format") or "").casefold().strip(" .")
    if declared in {"csv", "pdf", "xls", "xlsx"}:
        return declared
    return Path(urlparse(str(resource.get("url") or "")).path).suffix.casefold().lstrip(".")


def _resource_path(settings: Settings, resource: dict[str, Any]) -> Path:
    resource_id = str(resource.get("id") or resource.get("name") or "resource")
    return settings.paths.raw / "puskesmas" / f"{safe_name(resource_id)}.{_extension(resource)}"


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _numeric_cells(values: pd.Series) -> list[float]:
    numbers = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return numbers.tolist()


def _read_excel(path: Path) -> list[pd.DataFrame]:
    workbook = pd.ExcelFile(path)
    return [pd.read_excel(path, sheet_name=sheet, header=None) for sheet in workbook.sheet_names]


def _parse_purbalingga(
    path: Path,
    year: int,
    url: str,
    resource_id: str,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {
        "Nama Kecamatan",
        "UPT Puskesmas",
        "Jumlah Pasien Laki - laki",
        "Jumlah Pasien Perempuan",
        "Jumlah Pasien Meninggal Laki - laki",
        "Jumlah Pasien Meninggal Perempuan",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Purbalingga schema mismatch")
    start, end, resolution = _period(year, None)
    rows = []
    for index, row in frame.iterrows():
        male = float(row["Jumlah Pasien Laki - laki"])
        female = float(row["Jumlah Pasien Perempuan"])
        deaths = float(row["Jumlah Pasien Meninggal Laki - laki"]) + float(
            row["Jumlah Pasien Meninggal Perempuan"]
        )
        facility = str(row["UPT Puskesmas"]).strip()
        rows.append(
            _record(
                PURBALINGGA_SOURCE,
                f"{resource_id}:{index}:{facility}:{year}",
                "Jawa Tengah",
                "Purbalingga",
                facility,
                start,
                end,
                resolution,
                male + female,
                url,
                path,
                kecamatan=str(row["Nama Kecamatan"]).strip(),
                deaths=deaths,
                male_cases=male,
                female_cases=female,
            )
        )
    return rows


def _parse_arjowinangun(
    path: Path,
    title: str,
    url: str,
    resource_id: str,
) -> list[dict[str, Any]]:
    text = _pdf_text(path)
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if "PUSKESMAS ARJOWINANGUN" not in normalized.upper() or "DBD" not in normalized.upper():
        raise ValueError("resource content does not match DBD Arjowinangun")
    year, month = _month_year(title)
    start, end, resolution = _period(year, month)
    rows = []
    for subunit in SUBUNITS_ARJOWINANGUN:
        modern = re.search(
            rf"(?mi)^\s*\d+\s+{re.escape(subunit)}\s+(\d+)\s*$",
            normalized,
        )
        legacy = re.search(
            rf"(?mi)^\s*{re.escape(subunit)}\s+(\d+)\s+(\d+)(?:\s+(\d+))?\s*$",
            normalized,
        )
        deaths = None
        if modern is not None:
            cases = float(modern.group(1))
        elif legacy is not None:
            recovered = float(legacy.group(1))
            deaths = float(legacy.group(2))
            cases = float(legacy.group(3)) if legacy.group(3) else recovered + deaths
        else:
            raise ValueError(f"missing Arjowinangun subunit {subunit}")
        rows.append(
            _record(
                MALANG_SOURCE,
                f"{resource_id}:{subunit}:{year}-{month:02d}",
                "Jawa Timur",
                "Kota Malang",
                "Arjowinangun",
                start,
                end,
                resolution,
                cases,
                url,
                path,
                subunit=subunit,
                deaths=deaths,
            )
        )
    facility = _record(
        MALANG_SOURCE,
        f"{resource_id}:Arjowinangun:{year}-{month:02d}",
        "Jawa Timur",
        "Kota Malang",
        "Arjowinangun",
        start,
        end,
        resolution,
        sum(float(row["cases"]) for row in rows),
        url,
        path,
        deaths=sum(float(row["deaths"] or 0) for row in rows)
        if any(row["deaths"] is not None for row in rows)
        else None,
    )
    return [*rows, facility]


def _indicator_value(path: Path, phrase: str) -> float:
    if path.suffix.casefold() == ".pdf":
        text = _pdf_text(path)
        match = re.search(rf"{re.escape(phrase)}[^\n]*", text, re.IGNORECASE)
        if match is None:
            raise ValueError(f"indicator unavailable: {phrase}")
        values = re.findall(r"(?<![\d.,])\d+(?:[.,]\d+)?", match.group(0))
        if not values:
            raise ValueError(f"indicator value unavailable: {phrase}")
        return float(values[-1].replace(",", "."))
    for frame in _read_excel(path):
        for _, row in frame.iterrows():
            text = " ".join(str(value) for value in row if pd.notna(value))
            if phrase.casefold() in text.casefold():
                values = _numeric_cells(row)
                if values:
                    return values[-1]
    raise ValueError(f"indicator unavailable: {phrase}")


def _parse_indicator_month(
    path: Path,
    title: str,
    url: str,
    resource_id: str,
    puskesmas: str,
) -> list[dict[str, Any]]:
    year, month = _month_year(title)
    start, end, resolution = _period(year, month)
    cases = _indicator_value(path, "Penderita DBD ditangani")
    return [
        _record(
            MALANG_SOURCE,
            f"{resource_id}:{puskesmas}:{year}-{month:02d}",
            "Jawa Timur",
            "Kota Malang",
            puskesmas,
            start,
            end,
            resolution,
            cases,
            url,
            path,
        )
    ]


def _find_indicator_matrix(path: Path) -> tuple[list[float], int]:
    for frame in _read_excel(path):
        for index, row in frame.iterrows():
            text = " ".join(str(value) for value in row if pd.notna(value))
            if "penderita dbd ditangani" not in text.casefold():
                continue
            header = frame.iloc[max(0, index - 3) : index]
            positions: list[tuple[int, int]] = []
            for column in range(frame.shape[1]):
                header_text = " ".join(
                    str(value) for value in header.iloc[:, column] if pd.notna(value)
                ).casefold()
                month = next(
                    (number for name, number in MONTHS.items() if name in header_text),
                    None,
                )
                if month is not None:
                    positions.append((column, month))
            if len(positions) != 12:
                continue
            values = [
                float(pd.to_numeric(row.iloc[column], errors="raise")) for column, _ in positions
            ]
            year_text = " ".join(
                str(value)
                for value in frame.iloc[: index + 1].to_numpy().ravel()
                if pd.notna(value)
            )
            year_match = re.search(r"\b(20\d{2})\b", year_text)
            if year_match is None:
                raise ValueError("indicator matrix year unavailable")
            ordered = [
                value
                for _, value in sorted(zip([month for _, month in positions], values, strict=True))
            ]
            return ordered, int(year_match.group(1))
    raise ValueError("indicator matrix unavailable")


def _parse_indicator_matrix(
    path: Path,
    url: str,
    resource_id: str,
    puskesmas: str,
) -> list[dict[str, Any]]:
    values, year = _find_indicator_matrix(path)
    rows = []
    for month, cases in enumerate(values, start=1):
        if not np.isfinite(cases):
            continue
        start, end, resolution = _period(year, month)
        rows.append(
            _record(
                MALANG_SOURCE,
                f"{resource_id}:{puskesmas}:{year}-{month:02d}",
                "Jawa Timur",
                "Kota Malang",
                puskesmas,
                start,
                end,
                resolution,
                cases,
                url,
                path,
            )
        )
    return rows


def _parse_mojolangu(
    path: Path,
    title: str,
    url: str,
    resource_id: str,
) -> list[dict[str, Any]]:
    year, month = _month_year(title)
    start, end, resolution = _period(year, month)
    frame = _read_excel(path)[0]
    total = frame.loc[
        frame.apply(
            lambda row: row.astype("string").str.fullmatch("TOTAL", case=False, na=False).any(),
            axis=1,
        )
    ]
    if total.empty:
        raise ValueError("Mojolangu total row unavailable")
    values = pd.to_numeric(total.iloc[0, 1:5], errors="coerce").fillna(0).astype(float).tolist()
    rows = []
    for subunit, cases in zip(SUBUNITS_MOJOLANGU, values, strict=True):
        rows.append(
            _record(
                MALANG_SOURCE,
                f"{resource_id}:{subunit}:{year}-{month:02d}",
                "Jawa Timur",
                "Kota Malang",
                "Mojolangu",
                start,
                end,
                resolution,
                cases,
                url,
                path,
                subunit=subunit,
            )
        )
    rows.append(
        _record(
            MALANG_SOURCE,
            f"{resource_id}:Mojolangu:{year}-{month:02d}",
            "Jawa Timur",
            "Kota Malang",
            "Mojolangu",
            start,
            end,
            resolution,
            sum(values),
            url,
            path,
        )
    )
    return rows


def _parse_janti(path: Path, url: str, resource_id: str) -> list[dict[str, Any]]:
    frame = _read_excel(path)[0]
    rows = []
    for month_name, month in MONTHS.items():
        matching = frame.loc[
            frame.apply(
                lambda row, expected=month_name: (
                    row.astype("string").str.casefold().str.fullmatch(expected, na=False).any()
                ),
                axis=1,
            )
        ]
        if matching.empty:
            raise ValueError(f"Janti month unavailable: {month_name}")
        values = pd.to_numeric(matching.iloc[0, 3:6], errors="coerce").fillna(0).astype(float)
        start, end, resolution = _period(2025, month)
        for subunit, cases in zip(SUBUNITS_JANTI, values, strict=True):
            rows.append(
                _record(
                    MALANG_SOURCE,
                    f"{resource_id}:{subunit}:2025-{month:02d}",
                    "Jawa Timur",
                    "Kota Malang",
                    "Janti",
                    start,
                    end,
                    resolution,
                    float(cases),
                    url,
                    path,
                    subunit=subunit,
                )
            )
        rows.append(
            _record(
                MALANG_SOURCE,
                f"{resource_id}:Janti:2025-{month:02d}",
                "Jawa Timur",
                "Kota Malang",
                "Janti",
                start,
                end,
                resolution,
                float(values.sum()),
                url,
                path,
            )
        )
    return rows


def _parse_polowijen(path: Path, url: str, resource_id: str) -> list[dict[str, Any]]:
    text = _pdf_text(path)
    number_rows = [
        [float(value) for value in re.findall(r"\d+(?:\.\d+)?", line)]
        for line in text.splitlines()
        if re.fullmatch(r"\s*\d+(?:\s+\d+){5,}\s*", line)
    ]
    if len(number_rows) < 6 or len(number_rows[1]) < 7 or len(number_rows[4]) < 5:
        raise ValueError("Polowijen monthly matrix unavailable")
    values = [*number_rows[1][:7], *number_rows[4][:5]]
    rows = []
    for month, cases in enumerate(values, start=1):
        start, end, resolution = _period(2022, month)
        rows.append(
            _record(
                MALANG_SOURCE,
                f"{resource_id}:Polowijen:2022-{month:02d}",
                "Jawa Timur",
                "Kota Malang",
                "Polowijen",
                start,
                end,
                resolution,
                cases,
                url,
                path,
            )
        )
    return rows


def _malang_kind(title: str) -> tuple[str, str] | None:
    lowered = title.casefold()
    if title == "Data Pelayanan DBD, Rabies & Malaria Tahun 2022 Puskesmas Kendalkerep":
        return "indicator_matrix", "Kendalkerep"
    if title == "DATA PELAYANAN DBD PUSKESMAS KENDALKEREP":
        return "indicator_matrix_resources", "Kendalkerep"
    if title == "DATA DBD 2022 PUSKESMAS POLOWIJEN":
        return "polowijen", "Polowijen"
    if title == "DATA KASUS DBD PUSKESMAS JANTI":
        return "janti", "Janti"
    if lowered.startswith("laporan kasus dbd puskesmas mojolangu"):
        return "mojolangu", "Mojolangu"
    if lowered.startswith("data penyakit dbd puskesmas gribig bulan"):
        return "indicator_month", "Gribig"
    if lowered.startswith("pelayanan dbd ") and lowered.endswith("puskesmas kendalkerep"):
        return "indicator_month", "Kendalkerep"
    if (
        "arjowinangun" in lowered
        and ("dbd" in lowered or "demam berdarah dengue" in lowered)
        and "tahun 2025" in lowered
        and any(month in lowered for month in MONTHS)
    ):
        return "arjowinangun", "Arjowinangun"
    if "puskesmas arjowinangun" in lowered and "bulan" in lowered and "2023" in lowered:
        return "arjowinangun", "Arjowinangun"
    if lowered.startswith("data kasus dbd bulan") and "tahun 2024" in lowered:
        return "arjowinangun", "Arjowinangun"
    return None


def _parse_malang_resource(
    kind: str,
    facility: str,
    path: Path,
    title: str,
    url: str,
    resource_id: str,
) -> list[dict[str, Any]]:
    if kind == "arjowinangun":
        return _parse_arjowinangun(path, title, url, resource_id)
    if kind == "indicator_month":
        return _parse_indicator_month(path, title, url, resource_id, facility)
    if kind in {"indicator_matrix", "indicator_matrix_resources"}:
        return _parse_indicator_matrix(path, url, resource_id, facility)
    if kind == "mojolangu":
        return _parse_mojolangu(path, title, url, resource_id)
    if kind == "janti":
        return _parse_janti(path, url, resource_id)
    if kind == "polowijen":
        return _parse_polowijen(path, url, resource_id)
    raise ValueError(kind)


def _catalog_row(
    catalog: str,
    title: str,
    resource: dict[str, Any],
    path: Path,
    status: str,
    records: int,
    error: str,
) -> dict[str, Any]:
    return {
        "catalog": catalog,
        "dataset_title": title,
        "resource_id": resource.get("id"),
        "resource_name": resource.get("name"),
        "format": _extension(resource),
        "url": resource.get("url"),
        "path": str(path),
        "status": status,
        "records": records,
        "sha256": sha256(path) if path.exists() else "",
        "error": error,
    }


def _harvest_purbalingga(
    settings: Settings,
    session: requests.Session,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = request_json(
        session,
        f"{PURBALINGGA_CATALOG}/api/3/action/package_show",
        {"id": PURBALINGGA_PACKAGE},
        timeout=120,
    )
    package = payload["result"]
    rows = []
    catalog = []
    for resource in package.get("resources") or []:
        year_match = re.search(r"\b(20\d{2})\b", str(resource.get("name") or ""))
        if year_match is None or _extension(resource) != "csv":
            continue
        path = _resource_path(settings, resource)
        url = str(resource.get("url") or "")
        status = "cached" if path.exists() else "downloaded"
        error = ""
        parsed: list[dict[str, Any]] = []
        try:
            if not path.exists():
                download(session, url, path)
            parsed = _parse_purbalingga(
                path,
                int(year_match.group(1)),
                url,
                str(resource.get("id") or "resource"),
            )
            rows.extend(parsed)
        except (OSError, ValueError, KeyError, requests.RequestException) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        catalog.append(
            _catalog_row(
                PURBALINGGA_CATALOG,
                str(package.get("title") or ""),
                resource,
                path,
                status,
                len(parsed),
                error,
            )
        )
    return rows, catalog


def _harvest_malang(
    settings: Settings,
    session: requests.Session,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packages: dict[str, dict[str, Any]] = {}
    for query in ("DBD", "Arjowinangun DBD"):
        payload = request_json(
            session,
            f"{MALANG_CATALOG}/api/3/action/package_search",
            {"q": query, "rows": 1000},
            timeout=120,
        )
        for package in payload["result"]["results"]:
            packages[str(package.get("id") or package.get("name"))] = package
    rows = []
    catalog = []
    for package in packages.values():
        title = str(package.get("title") or "")
        identified = _malang_kind(title)
        if identified is None:
            continue
        kind, facility = identified
        resources = package.get("resources") or []
        for resource in resources:
            if _extension(resource) not in {"pdf", "xls", "xlsx"}:
                continue
            path = _resource_path(settings, resource)
            url = str(resource.get("url") or "")
            status = "cached" if path.exists() else "downloaded"
            error = ""
            parsed: list[dict[str, Any]] = []
            try:
                if not path.exists():
                    download(session, url, path)
                parsed = _parse_malang_resource(
                    kind,
                    facility,
                    path,
                    title,
                    url,
                    str(resource.get("id") or "resource"),
                )
                rows.extend(parsed)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                requests.RequestException,
            ) as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            catalog.append(
                _catalog_row(
                    MALANG_CATALOG,
                    title,
                    resource,
                    path,
                    status,
                    len(parsed),
                    error,
                )
            )
    return rows, catalog


def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    key = ["source", "puskesmas", "subunit", "period_start", "temporal_resolution"]
    value = frame.assign(_available=frame["cases"].notna().astype(int)).sort_values("_available")
    return value.drop_duplicates(key, keep="last").drop(columns="_available").reset_index(drop=True)


def _facility_cases(frame: pd.DataFrame) -> pd.DataFrame:
    facility = frame.loc[frame["aggregation_level"].eq("puskesmas")].copy()
    if facility.empty:
        return canonicalize(pd.DataFrame())
    source_ids = facility["source_record_id"].astype("string")
    canonical = pd.DataFrame(
        {
            "source": facility["source"],
            "source_record_id": source_ids,
            "location_id": facility["district_city"].astype("string")
            + "|"
            + facility["puskesmas"].astype("string"),
            "location_name": facility["puskesmas"],
            "admin_level": "admin3",
            "admin_0": "Indonesia",
            "admin_1": facility["province"],
            "admin_2": facility["district_city"],
            "admin_3": facility["puskesmas"],
            "period_start": facility["period_start"],
            "period_end": facility["period_end"],
            "temporal_resolution": facility["temporal_resolution"],
            "cases": facility["cases"],
            "population": pd.NA,
            "case_definition": facility["case_definition"],
            "confirmation_status": facility["confirmation_status"],
            "native_record": True,
        }
    )
    return canonicalize(canonical)


def puskesmas_readiness(frame: pd.DataFrame) -> pd.DataFrame:
    facility = frame.loc[frame["aggregation_level"].eq("puskesmas")].copy()
    if facility.empty:
        return pd.DataFrame()
    rows = []
    groups = facility.groupby(
        ["source", "province", "district_city", "puskesmas", "temporal_resolution"]
    )
    for keys, group in groups:
        group = group.loc[group["cases"].notna()].copy()
        if group.empty:
            continue
        grain = str(keys[-1])
        frequency = "MS" if grain == "month" else "YS"
        expected = len(
            pd.date_range(group["period_start"].min(), group["period_start"].max(), freq=frequency)
        )
        observed = int(group["period_start"].nunique())
        completeness = observed / expected if expected else np.nan
        if grain == "month" and observed >= 60 and completeness >= 0.9:
            use = "standalone seasonal forecasting candidate"
        elif grain == "month" and observed >= 24 and completeness >= 0.8:
            use = "hierarchical or transfer-learning pilot"
        elif grain == "month":
            use = "pooled modeling or validation only"
        else:
            use = "annual trend and spatial-prior modeling"
        rows.append(
            {
                "source": keys[0],
                "province": keys[1],
                "district_city": keys[2],
                "puskesmas": keys[3],
                "temporal_resolution": grain,
                "first_period": group["period_start"].min(),
                "last_period": group["period_start"].max(),
                "observed_periods": observed,
                "expected_periods": expected,
                "completeness": completeness,
                "total_cases": group["cases"].sum(),
                "zero_rate": group["cases"].eq(0).mean(),
                "duplicate_periods": int(group["period_start"].duplicated().sum()),
                "recommended_use": use,
                "confirmation_status": "not stated in source resource",
            }
        )
    return pd.DataFrame.from_records(rows)


def harvest_puskesmas(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not enabled:
        empty = pd.DataFrame()
        return canonicalize(empty), empty, empty, empty
    rows: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    for harvester in (_harvest_purbalingga, _harvest_malang):
        try:
            harvested, resources = harvester(settings, session)
            rows.extend(harvested)
            catalog.extend(resources)
        except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
            catalog.append(
                {
                    "catalog": harvester.__name__,
                    "dataset_title": "",
                    "resource_id": "",
                    "resource_name": "",
                    "format": "",
                    "url": "",
                    "path": "",
                    "status": "failed",
                    "records": 0,
                    "sha256": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    granular = _deduplicate(pd.DataFrame.from_records(rows))
    if not granular.empty:
        granular["period_start"] = pd.to_datetime(granular["period_start"])
        granular["period_end"] = pd.to_datetime(granular["period_end"])
    return (
        _facility_cases(granular),
        granular,
        pd.DataFrame.from_records(catalog),
        puskesmas_readiness(granular),
    )
