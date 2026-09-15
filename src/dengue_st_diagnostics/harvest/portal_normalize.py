from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

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


def _month(value: str) -> int | None:
    text = value.casefold()
    return next((number for name, number in MONTHS.items() if name in text), None)


def _integer(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if str(value).strip().casefold() == "nihil":
        return 0
    number = pd.to_numeric(value, errors="coerce")
    return int(number) if pd.notna(number) else None


def _percentage(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).replace("%", "").replace(",", ".").strip()
    number = pd.to_numeric(text, errors="coerce")
    return float(number) if pd.notna(number) else None


def _species(resources: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = resources.loc[
        resources["dataset_title"].str.contains(
            "Proporsi Kasus Berdasarkan Jenis Plasmodium", case=False, na=False
        )
        & resources["status"].eq("parsed")
    ]
    names = {
        "falsiparum": "P. falciparum",
        "falciparum": "P. falciparum",
        "vivax": "P. vivax",
        "ovale": "P. ovale",
        "malariae": "P. malariae",
        "knowlesi": "P. knowlesi",
        "mix": "mixed infection",
    }
    for resource in selected.itertuples(index=False):
        with Path(resource.path).open(encoding="utf-8-sig", errors="replace") as handle:
            parsed = list(csv.reader(handle))
        semicolon = any(";" in cell for row in parsed for cell in row)
        if semicolon:
            with Path(resource.path).open(encoding="utf-8-sig", errors="replace") as handle:
                parsed = list(csv.reader(handle, delimiter=";"))
        for values in parsed[1:]:
            if len(values) < 2:
                continue
            species = names.get(values[0].strip().casefold())
            if species is None:
                continue
            parts = [part.strip() for part in values[1].split(",")]
            cases = _integer(parts[0])
            percent = _percentage(parts[1]) if len(parts) > 1 else None
            rows.append(
                {
                    "source": "Central Java Open Data",
                    "province": "Jawa Tengah",
                    "period_start": pd.Timestamp(2026, 1, 1),
                    "period_end": pd.Timestamp(2026, 3, 31),
                    "temporal_resolution": "quarter",
                    "species": species,
                    "cases": cases,
                    "percentage": percent,
                    "zoonotic": species == "P. knowlesi",
                    "dataset_title": resource.dataset_title,
                    "source_url": resource.url,
                    "path": resource.path,
                }
            )
    return pd.DataFrame.from_records(rows).drop_duplicates() if rows else pd.DataFrame()


def _annual_malaria(resource: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sheets = pd.read_excel(resource.path, sheet_name=None, header=None)
    metrics = {
        "dilakukan pemeriksaan sd": "malaria_examined",
        "diobati sesuai standar": "confirmed_malaria_treated",
        "di follow up": "confirmed_malaria_followed_up",
        "penyeelidikan epidemiologi": "confirmed_malaria_investigated",
    }
    for frame in sheets.values():
        text = " ".join(frame.fillna("").astype(str).to_numpy().ravel())
        year_match = re.search(r"(?:TAHUN|TARGET)\s+(20\d{2})", text, re.IGNORECASE)
        if year_match is None:
            year_match = re.search(
                r"(20\d{2})", f"{resource.dataset_title} {resource.resource_name}"
            )
        facility_match = re.search(r"PUSKESMAS\s+([A-Z]+)", text, re.IGNORECASE)
        if facility_match is None:
            facility_match = re.search(
                r"PUSKESMAS\s+([A-Z]+)", str(resource.dataset_title), re.IGNORECASE
            )
        if year_match is None or facility_match is None:
            continue
        year = int(year_match.group(1))
        facility = facility_match.group(1).title()
        header_map: dict[int, int] = {}
        for _, header in frame.iterrows():
            current: dict[int, int] = {}
            for column, value in header.items():
                token = str(value).strip().casefold()
                month = MONTHS.get(token)
                if month is None:
                    numeric = pd.to_numeric(value, errors="coerce")
                    month = int(numeric) if pd.notna(numeric) and 1 <= numeric <= 12 else None
                if month is not None:
                    current[int(column)] = month
            if len(set(current.values())) >= 6:
                header_map = current
        if not header_map:
            continue
        for _, value_row in frame.iterrows():
            label = " ".join(
                str(value) for value in value_row.iloc[:3] if pd.notna(value)
            ).casefold()
            metric = next((name for term, name in metrics.items() if term in label), None)
            if metric is None:
                continue
            for column, month in header_map.items():
                value = _integer(value_row.get(column))
                if value is not None:
                    rows.append(
                        {
                            "source": "Malang Open Data",
                            "puskesmas": facility,
                            "period_start": pd.Timestamp(year, month, 1),
                            "temporal_resolution": "month",
                            "metric": metric,
                            "value": value,
                            "case_definition": "laboratory_confirmed"
                            if metric.startswith("confirmed")
                            else "tested",
                            "source_url": resource.url,
                            "path": resource.path,
                        }
                    )
    return rows


def _mojolangu(resource: Any) -> dict[str, Any] | None:
    month = _month(str(resource.dataset_title))
    year_match = re.search(r"(20\d{2})", str(resource.dataset_title))
    if month is None or year_match is None:
        return None
    sheets = pd.read_excel(resource.path, sheet_name=None, header=None)
    values = [str(value).strip().casefold() for frame in sheets.values() for value in frame.stack()]
    if "nihil" not in values:
        return None
    return {
        "source": "Malang Open Data",
        "puskesmas": "Mojolangu",
        "period_start": pd.Timestamp(int(year_match.group(1)), month, 1),
        "temporal_resolution": "month",
        "metric": "malaria_cases",
        "value": 0,
        "case_definition": "reported_zero",
        "source_url": resource.url,
        "path": resource.path,
    }


def _malaria(resources: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = resources.loc[resources["status"].eq("parsed") & resources["category"].eq("malaria")]
    for resource in selected.itertuples(index=False):
        title = str(resource.dataset_title)
        if "laporan kasus malaria puskesmas mojolangu" in title.casefold():
            row = _mojolangu(resource)
            if row is not None:
                rows.append(row)
        if title.startswith("DATA PELAYANAN"):
            rows.extend(_annual_malaria(resource))
    frame = pd.DataFrame.from_records(rows)
    return (
        frame.drop_duplicates(["puskesmas", "period_start", "metric", "value"]) if rows else frame
    )


def _mojolangu_vectors(resource: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sheets = pd.read_excel(resource.path, sheet_name=None, header=None)
    for frame in sheets.values():
        for _, value in frame.iloc[3:].iterrows():
            month = _integer(value.iloc[0])
            if month is None or not 1 <= month <= 12:
                continue
            locality = str(value.iloc[1]).strip()
            rows.append(
                {
                    "source": "Malang Open Data",
                    "puskesmas": "Mojolangu",
                    "locality": locality,
                    "period_start": pd.Timestamp(2025, month, 1),
                    "temporal_resolution": "month",
                    "aedes_houses_inspected": _integer(value.iloc[3]),
                    "aedes_houses_positive": _integer(value.iloc[4]),
                    "aedes_house_index_free_percent": _percentage(value.iloc[5]),
                    "anopheles_habitats_inspected": _integer(value.iloc[7]),
                    "anopheles_habitats_positive": _integer(value.iloc[8]),
                    "anopheles_habitat_index": _percentage(value.iloc[9]),
                    "source_url": resource.url,
                    "path": resource.path,
                }
            )
    return rows


def _arjowinangun_vectors(resource: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = pd.read_excel(resource.path, header=None)
    for _, value in frame.iloc[8:12].iterrows():
        locality = str(value.iloc[1]).strip()
        for month in range(1, 13):
            inspected = _integer(value.iloc[2 + 2 * (month - 1)])
            positive = _integer(value.iloc[3 + 2 * (month - 1)])
            if inspected is None or positive is None:
                continue
            free_percent = 100 * (inspected - positive) / inspected if inspected else None
            rows.append(
                {
                    "source": "Malang Open Data",
                    "puskesmas": "Arjowinangun",
                    "locality": locality,
                    "period_start": pd.Timestamp(2022, month, 1),
                    "temporal_resolution": "month",
                    "aedes_houses_inspected": inspected,
                    "aedes_houses_positive": positive,
                    "aedes_house_index_free_percent": free_percent,
                    "anopheles_habitats_inspected": pd.NA,
                    "anopheles_habitats_positive": pd.NA,
                    "anopheles_habitat_index": pd.NA,
                    "source_url": resource.url,
                    "path": resource.path,
                }
            )
    return rows


def _vectors(resources: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = resources.loc[
        resources["status"].eq("parsed") & resources["category"].eq("entomology")
    ]
    for resource in selected.itertuples(index=False):
        title = str(resource.dataset_title)
        if title.startswith("Angka Bebas Jentik Kelurahan") and "Mojolangu" in title:
            rows.extend(_mojolangu_vectors(resource))
        if title == "Data Angka Bebas Jentik":
            rows.extend(_arjowinangun_vectors(resource))
    frame = pd.DataFrame.from_records(rows)
    return frame.drop_duplicates(["puskesmas", "locality", "period_start"]) if rows else frame


def normalize_portal_records(
    resources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if resources.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return _species(resources), _malaria(resources), _vectors(resources)
