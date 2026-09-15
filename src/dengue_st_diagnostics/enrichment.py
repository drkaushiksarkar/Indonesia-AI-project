from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from PIL import Image, ImageDraw, ImageFont

from dengue_st_diagnostics.config import Settings

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
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

DISEASE_PATTERN = re.compile(r"\b(dbd|dengue|malaria)\b", re.IGNORECASE)
TIME_PATTERN = re.compile(r"\b(tahun|year|bulan|month|minggu|week|tanggal|date|periode)\b")
LOCATION_PATTERN = re.compile(
    r"(provinsi|province|kabupaten|kota|district|kecamatan|kelurahan|desa|puskesmas|fasilitas|faskes|sarana_kesehatan|wilayah)"
)
MEASURE_PATTERN = re.compile(
    r"(jumlah|kasus|penderita|pasien|positif|postif|meninggal|kematian|suspek|pemeriksaan|diperiksa|pengobatan|cfr|crf|fatality|incidence|api|persentase|cakupan|ditangani|ditemukan|fogging|jentik|larva|aedes|anopheles|habitat|rumah|indeks|abj|mikroskop|rdt|konfirmasi)"
)
EXCLUDE_MEASURE_PATTERN = re.compile(
    r"(^|_)(id|no|nomor|kode|tahun|year|bulan|month|minggu|week|tanggal|date)($|_)|penduduk|population|longitude|latitude|koordinat"
)
GENERIC_VALUE_PATTERN = re.compile(r"^(jumlah|nilai|value|total)$")
DESCRIPTOR_PATTERN = re.compile(
    r"(jenis|indikator|uraian|kasus_dbd|persentase|penyakit|kategori|category|measure|metric)"
)


def _label(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _text(value: Any) -> str:
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip()


def _number(value: Any) -> float | None:
    text = _text(value).replace("%", "").replace(" ", "")
    if not text or text.casefold() in {"na", "n/a", "nan", "null", "-", "--"}:
        return None
    if not re.fullmatch(r"[-+]?[0-9]+(?:[.,][0-9]+)*", text):
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") == 1:
        text = text.replace(",", ".")
    elif text.count(",") > 1:
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _disease(*values: Any) -> str | None:
    text = " ".join(_text(value).casefold() for value in values)
    has_dengue = bool(re.search(r"\b(dbd|dengue)\b", text))
    has_malaria = bool(re.search(r"\bmalaria\b", text))
    if has_dengue and not has_malaria:
        return "dengue"
    if has_malaria and not has_dengue:
        return "malaria"
    return None


def _metric(value: str, disease: str) -> str:
    name = _label(value)
    if re.search(r"cfr|crf|fatality", name):
        return "case_fatality_rate"
    if re.search(r"meninggal|kematian|death", name):
        return "deaths"
    if re.search(r"suspek|suspect", name):
        return "suspected_cases"
    if re.search(r"falciparum", name):
        return "plasmodium_falciparum"
    if re.search(r"vivax", name):
        return "plasmodium_vivax"
    if re.search(r"knowlesi", name):
        return "plasmodium_knowlesi"
    if re.search(r"malariae", name):
        return "plasmodium_malariae"
    if re.search(r"ovale", name):
        return "plasmodium_ovale"
    if re.search(r"konfirmasi.*mikroskop|mikroskop.*konfirmasi", name):
        return "malaria_microscopy_confirmed"
    if re.search(r"konfirmasi.*rdt|rdt.*konfirmasi|konfirmasi.*rapid_diagnostic", name):
        return "malaria_rdt_confirmed"
    if re.search(r"mikroskop", name):
        return "malaria_microscopy_examinations"
    if re.search(r"rdt|rapid_diagnostic", name):
        return "malaria_rdt_examinations"
    if re.search(r"pemeriksaan|diperiksa|exam", name):
        return f"{disease}_examinations"
    if re.search(r"pengobatan|pengaobatan|treatment", name):
        return f"{disease}_standard_treatment"
    if re.search(r"fogging", name):
        return "fogging_locations"
    if re.search(r"angka_bebas_jentik|\babj\b", name):
        return "aedes_larva_free_index"
    if re.search(r"jentik|larva", name):
        return "larval_surveillance_value"
    if re.search(r"anopheles.*habitat|habitat.*anopheles", name):
        return "anopheles_habitat_index"
    if re.search(r"incidence|angka_kesakitan|\bapi\b", name):
        return "incidence_rate"
    if re.search(r"positif|postif|confirmed|konfirmasi", name):
        return "confirmed_cases"
    if re.search(r"persentase|percent|cakupan", name):
        return name
    if re.search(r"kasus|penderita|pasien|case", name):
        return "cases"
    return name or f"{disease}_value"


def _sex(value: Any) -> str | None:
    name = _label(value)
    if re.search(r"total|seluruh|laki_laki_dan_perempuan", name):
        return "all"
    if re.search(r"laki|pria|male", name) and not re.search(r"perempuan|wanita|female", name):
        return "male"
    if re.search(r"perempuan|wanita|female", name):
        return "female"
    return None


def _species(value: Any) -> str | None:
    name = _label(value)
    for species in ["falciparum", "vivax", "knowlesi", "malariae", "ovale"]:
        if species in name:
            return f"Plasmodium {species}"
    return None


def _year(values: Iterable[Any]) -> int | None:
    for value in values:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", _text(value))
        if match:
            year = int(match.group())
            if 1900 <= year <= date.today().year + 1:
                return year
    return None


def _month(value: Any) -> int | None:
    integer = _integer(value)
    if integer is not None and 1 <= integer <= 12:
        return integer
    name = _label(value)
    for month_name, month_number in MONTHS.items():
        if name == month_name or name.startswith(f"{month_name}_"):
            return month_number
    return None


def _period(
    row: dict[str, Any],
    title: str,
    resource_name: str,
    sheet: str,
) -> tuple[str | None, str | None, str | None, bool]:
    year_columns = [name for name in row if name in {"tahun", "year", "tahun_laporan"}]
    month_columns = [name for name in row if name in {"bulan", "month", "bulan_laporan"}]
    week_columns = [name for name in row if name in {"minggu", "week", "pekan", "minggu_ke"}]
    date_columns = [name for name in row if re.search(r"(^|_)(tanggal|date)($|_)", name)]
    for column in date_columns:
        raw = _text(row.get(column))
        parsed = pd.to_datetime(
            raw,
            errors="coerce",
            dayfirst=not bool(re.match(r"^\d{4}[-/]", raw)),
        )
        if pd.notna(parsed):
            value = parsed.date()
            return value.isoformat(), value.isoformat(), "day", True
    year = _year([*(row.get(column) for column in year_columns), title, resource_name, sheet])
    if year is None:
        return None, None, None, False
    if not week_columns and re.search(r"mingguan|weekly", f"{title} {resource_name}", re.I):
        week_columns = [name for name in row if name in {"category", "kategori"}]
    for column in week_columns:
        week = _integer(row.get(column))
        if week is not None and 1 <= week <= 53:
            try:
                start = date.fromisocalendar(year, week, 1)
            except ValueError:
                continue
            return start.isoformat(), (start + timedelta(days=6)).isoformat(), "week", True
    for column in month_columns:
        month = _month(row.get(column))
        if month is not None:
            start = date(year, month, 1)
            end = date(year, month, calendar.monthrange(year, month)[1])
            return start.isoformat(), end.isoformat(), "month", True
    explicit = bool(year_columns and any(_text(row.get(column)) for column in year_columns))
    return f"{year}-01-01", f"{year}-12-31", "year", explicit


def _first(row: dict[str, Any], patterns: Iterable[str]) -> str:
    for pattern in patterns:
        for name, value in row.items():
            if re.search(pattern, name) and _text(value):
                return _text(value)
    return ""


def _locations(row: dict[str, Any], title: str) -> dict[str, str]:
    admin_1 = _first(row, [r"nama_provinsi", r"^provinsi$", r"^province$"])
    admin_2 = _first(
        row,
        [r"nama_kabupaten_kota", r"nama_kabupatenkota", r"nama_kabupaten", r"^kabupaten$", r"^kota$", r"^district$"],
    )
    admin_3 = _first(row, [r"nama_kecamatan", r"^kecamatan$", r"^district_name$"])
    admin_4 = _first(row, [r"^kelurahan$", r"^desa$", r"nama_kelurahan", r"nama_desa"])
    facility = _first(
        row,
        [r"nama_puskesmas", r"puskesmas", r"fasilitas", r"faskes", r"sarana_kesehatan"],
    )
    if not admin_2:
        match = re.search(r"\b(Kabupaten|Kota)\s+([A-Za-z][A-Za-z .'-]+?)(?:\s+Tahun|\s+20\d{2}|$)", title)
        if match:
            admin_2 = f"{match.group(1)} {match.group(2).strip()}"
    return {
        "admin_0": "Indonesia",
        "admin_1": admin_1,
        "admin_2": admin_2,
        "admin_3": admin_3,
        "admin_4": admin_4,
        "facility_name": facility,
        "facility_type": "puskesmas" if facility and "puskesmas" in " ".join(row) else "health_facility" if facility else "",
    }


def _header_score(values: dict[int, str]) -> float:
    labels = [_label(value) for value in values.values()]
    joined = " ".join(labels)
    score = 0.0
    score += 2.0 if TIME_PATTERN.search(joined) else 0.0
    score += 2.0 if LOCATION_PATTERN.search(joined) else 0.0
    score += 2.0 if MEASURE_PATTERN.search(joined) else 0.0
    score += min(len(set(labels)), 10) / 10
    score -= sum(_number(value) is not None for value in values.values()) * 0.25
    return score


def _headers(values: dict[int, str]) -> dict[int, str]:
    result: dict[int, str] = {}
    used: defaultdict[str, int] = defaultdict(int)
    for column, value in values.items():
        name = _label(value) or f"column_{column}"
        used[name] += 1
        result[column] = name if used[name] == 1 else f"{name}_{used[name]}"
    return result


def _measurement_columns(headers: dict[int, str]) -> list[int]:
    return [
        column
        for column, name in headers.items()
        if MEASURE_PATTERN.search(name) and not EXCLUDE_MEASURE_PATTERN.search(name)
    ]


def _descriptor(row: dict[str, Any], measure_name: str, title: str) -> str:
    parts = [
        _text(value)
        for name, value in row.items()
        if DESCRIPTOR_PATTERN.search(name) and _text(value)
    ]
    return " ".join([*parts, measure_name, title])


def _unit(row: dict[str, Any], metric: str) -> str:
    value = _first(row, [r"^satuan", r"^unit"])
    if value:
        return value
    if "rate" in metric or "percent" in metric or "persentase" in metric or "index" in metric:
        return "percent"
    return "count"


def _observation_hash(row: dict[str, Any]) -> str:
    fields = [
        "source_dataset_id",
        "disease",
        "metric",
        "value_numeric",
        "unit",
        "period_start",
        "period_end",
        "temporal_resolution",
        "admin_0",
        "admin_1",
        "admin_2",
        "admin_3",
        "admin_4",
        "facility_name",
        "sex",
        "species",
    ]
    payload = {field: row.get(field) for field in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _classify_table(row: dict[str, Any]) -> str:
    title = _text(row.get("dataset_title")).casefold()
    source = _text(row.get("source")).casefold()
    sheet = _text(row.get("sheet")).casefold()
    if "global compendium" in title and "aedes" in title:
        return "vector occurrence coordinates"
    if source == "data.go.id" and DISEASE_PATTERN.search(title):
        if sheet.startswith("page_"):
            return "government PDF tables requiring table review"
        return "government disease tables eligible for reconstruction"
    if source in {"zenodo", "researchdata.jcu.edu.au"}:
        if re.search(r"knowledge|attitude|practice|co-infection|risk factor|patient|clinical", title):
            return "research participant or survey data requiring governance review"
        return "research and ecological data requiring study-specific parsing"
    if DISEASE_PATTERN.search(title):
        return "other disease-related tables requiring review"
    return "non-disease or contextual tables"


def _fetch_tables(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        select payload
        from bronze.record b
        join metadata.dataset d using (dataset_id)
        where d.dataset_name = 'public_tabular_tables'
        """
    ).fetchall()
    return [dict(row[0]) for row in rows]


def _fetch_candidate_cells(connection: psycopg.Connection[Any]) -> list[tuple[Any, ...]]:
    return connection.execute(
        """
        select
            b.bronze_record_id,
            payload->>'source',
            payload->>'dataset_id',
            payload->>'dataset_title',
            payload->>'resource_id',
            payload->>'resource_name',
            payload->>'source_url',
            payload->>'member',
            payload->>'sheet',
            (payload->>'row_number')::integer,
            (payload->>'column_number')::integer,
            payload->>'value'
        from bronze.record b
        join metadata.dataset d using (dataset_id)
        where d.dataset_name = 'public_tabular_cells'
          and payload->>'source' = 'data.go.id'
          and coalesce(payload->>'dataset_title', '') ~* '\\m(dbd|dengue|malaria)\\M'
        order by
            payload->>'dataset_id',
            payload->>'resource_id',
            payload->>'member',
            payload->>'sheet',
            (payload->>'row_number')::integer,
            (payload->>'column_number')::integer
        """
    ).fetchall()


def _fetch_organizations(connection: psycopg.Connection[Any]) -> dict[str, str]:
    rows = connection.execute(
        """
        select payload->>'dataset_id', max(payload->>'organization')
        from bronze.record b
        join metadata.dataset d using (dataset_id)
        where d.dataset_name = 'public_dataset_catalog'
          and payload->>'source' = 'data.go.id'
        group by payload->>'dataset_id'
        """
    ).fetchall()
    return {_text(dataset_id): _text(organization) for dataset_id, organization in rows}


def _fetch_database_reconciliation(connection: psycopg.Connection[Any]) -> pd.DataFrame:
    cursor = connection.execute(
        """
        with ranked as (
            select
                dataset_id,
                dataset_name,
                data_role,
                modeling_eligible,
                row_count,
                loaded_at,
                row_number() over (
                    partition by dataset_name
                    order by loaded_at desc, dataset_id desc
                ) as version_rank
            from metadata.dataset
            where medallion_layer = 'bronze'
        ),
        silver_counts as (
            select d.dataset_name, count(*) as normalized_measurements
            from silver.observation s
            join metadata.dataset d using (dataset_id)
            group by d.dataset_name
        )
        select
            r.data_role,
            r.modeling_eligible,
            r.dataset_name,
            r.row_count as stored_rows_or_cells,
            coalesce(s.normalized_measurements, 0) as normalized_measurements,
            r.loaded_at
        from ranked r
        left join silver_counts s using (dataset_name)
        where r.version_rank = 1
        order by r.data_role, r.row_count desc, r.dataset_name
        """
    )
    columns = [column.name for column in cursor.description or []]
    return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)


def _database_role_summary(reconciliation: pd.DataFrame) -> pd.DataFrame:
    return (
        reconciliation.groupby(["data_role", "modeling_eligible"], dropna=False)
        .agg(
            registered_datasets=("dataset_name", "size"),
            stored_rows_or_cells=("stored_rows_or_cells", "sum"),
            normalized_measurements=("normalized_measurements", "sum"),
        )
        .reset_index()
        .sort_values("stored_rows_or_cells", ascending=False)
    )


def _table_groups(rows: Iterable[tuple[Any, ...]]) -> Iterable[list[tuple[Any, ...]]]:
    current_key: tuple[Any, ...] | None = None
    values: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row[2:6]) + tuple(row[7:9])
        if current_key is not None and key != current_key:
            yield values
            values = []
        current_key = key
        values.append(row)
    if values:
        yield values


def _parse_table(
    rows: list[tuple[Any, ...]], organizations: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = rows[0]
    source, dataset_id, title, resource_id, resource_name, source_url, member, sheet = first[1:9]
    matrix: defaultdict[int, dict[int, tuple[int, str]]] = defaultdict(dict)
    for bronze_id, *_, row_number, column_number, value in rows:
        matrix[row_number][column_number] = (bronze_id, value)
    candidates = {
        row_number: _header_score({column: value for column, (_, value) in columns.items()})
        for row_number, columns in matrix.items()
        if row_number <= min(max(matrix), 25)
    }
    header_row = max(candidates, key=candidates.get)
    score = candidates[header_row]
    organization = organizations.get(_text(dataset_id), "")
    semantic_score = score
    semantic_score += 1.0 if _year([title, resource_name, sheet]) else 0.0
    semantic_score += 1.5 if re.match(r"^(kabupaten|kota|provinsi)\b", organization, re.I) else 0.0
    headers = _headers({column: value for column, (_, value) in matrix[header_row].items()})
    measures = _measurement_columns(headers)
    audit = {
        "source": source,
        "source_dataset_id": dataset_id,
        "dataset_title": title,
        "resource_id": resource_id,
        "resource_name": resource_name,
        "member": member,
        "sheet": sheet,
        "raw_cells": len(rows),
        "raw_rows": len(matrix),
        "header_row": header_row,
        "header_score": round(score, 3),
        "semantic_score": round(semantic_score, 3),
        "source_organization": organization,
        "measurement_columns": len(measures),
        "status": "eligible" if semantic_score >= 4.5 and measures else "review",
    }
    if semantic_score < 4.5 or not measures:
        return [], audit
    observations: list[dict[str, Any]] = []
    for row_number in sorted(matrix):
        if row_number <= header_row:
            continue
        values = {
            headers[column]: value
            for column, (_, value) in matrix[row_number].items()
            if column in headers
        }
        period_start, period_end, temporal_resolution, explicit_time = _period(
            values, title, resource_name, sheet
        )
        if period_start is None:
            continue
        locations = _locations(values, title)
        if not locations["admin_1"] and organization.casefold().startswith("provinsi "):
            locations["admin_1"] = organization
        if not locations["admin_2"] and re.match(r"^(kabupaten|kota)\b", organization, re.I):
            locations["admin_2"] = organization
        spatial_resolution = (
            "facility"
            if locations["facility_name"]
            else "admin4"
            if locations["admin_4"]
            else "admin3"
            if locations["admin_3"]
            else "admin2"
            if locations["admin_2"]
            else "admin1"
            if locations["admin_1"]
            else "admin0"
        )
        for column in measures:
            if column not in matrix[row_number]:
                continue
            bronze_id, raw_value = matrix[row_number][column]
            number = _number(raw_value)
            if number is None or number < 0:
                continue
            measure_name = headers[column]
            descriptor = _descriptor(values, measure_name, title)
            disease = _disease(measure_name, descriptor) or _disease(title)
            if disease is None:
                continue
            metric = _metric(descriptor if GENERIC_VALUE_PATTERN.fullmatch(measure_name) else measure_name, disease)
            metric_sex = _sex(measure_name)
            row_sex = _sex(_first(values, [r"jenis_kelamin", r"^sex$"]))
            sex = row_sex or metric_sex
            species = _species(f"{measure_name} {descriptor}")
            confidence = 0.65
            confidence += 0.1 if semantic_score >= 6 else 0.05
            confidence += 0.1 if explicit_time else 0.05
            confidence += 0.1 if spatial_resolution != "admin0" else 0.0
            confidence += 0.05 if not GENERIC_VALUE_PATTERN.fullmatch(measure_name) else 0.0
            if sheet.startswith("page_"):
                confidence = min(confidence, 0.8)
            if metric in {
                "jumlah_lp",
                "jumlah_seluruhnya",
                "nilai_cakupan",
                "cakupan_riil",
            }:
                confidence = min(confidence, 0.8)
            confidence = min(confidence, 0.99)
            result = {
                "source_id": "existing_public_corpus",
                "source": source,
                "source_dataset_id": dataset_id,
                "dataset_title": title,
                "resource_id": resource_id,
                "resource_name": resource_name,
                "source_url": source_url,
                "source_organization": organization,
                "member": member,
                "sheet": sheet,
                "source_row_number": row_number,
                "source_column_number": column,
                "source_header": measure_name,
                "upstream_bronze_record_id": bronze_id,
                "disease": disease,
                "metric": metric,
                "value_numeric": number,
                "value_text": "",
                "unit": _unit(values, metric),
                "period_start": period_start,
                "period_end": period_end,
                "temporal_resolution": temporal_resolution,
                **locations,
                "spatial_resolution": spatial_resolution,
                "sex": sex or "",
                "age_group": "",
                "species": species or "",
                "case_definition": "aggregate value as labelled by source",
                "confirmation_status": "not stated in source table",
                "privacy_class": "aggregate",
                "normalization_confidence": round(confidence, 3),
                "review_status": "model_ready" if confidence >= 0.85 else "review_required",
                "parser_version": "public_table_semantic_parser_1.0.0",
                "accounting_category": _classify_table(
                    {"source": source, "dataset_title": title, "sheet": sheet}
                ),
            }
            result["observation_hash"] = _observation_hash(result)
            observations.append(result)
    audit["candidate_observations"] = len(observations)
    audit["model_ready_observations"] = sum(
        row["review_status"] == "model_ready" for row in observations
    )
    return observations, audit


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["observation_hash"]].append(row)
    result: list[dict[str, Any]] = []
    for observation_hash, values in grouped.items():
        primary = max(
            values,
            key=lambda row: (
                row["normalization_confidence"],
                not row["sheet"].startswith("page_"),
                row["resource_name"].casefold().endswith(".csv"),
                row["sheet"] == "csv",
            ),
        ).copy()
        primary["observation_hash"] = observation_hash
        primary["source_representation_count"] = len(values)
        primary["source_coordinates_json"] = json.dumps(
            [
                {
                    "resource_id": row["resource_id"],
                    "resource_name": row["resource_name"],
                    "member": row["member"],
                    "sheet": row["sheet"],
                    "row": row["source_row_number"],
                    "column": row["source_column_number"],
                    "bronze_record_id": row["upstream_bronze_record_id"],
                }
                for row in values
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        result.append(primary)
    return result


def _reconciliation(tables: list[dict[str, Any]], enriched: pd.DataFrame) -> pd.DataFrame:
    table_frame = pd.DataFrame(tables)
    table_frame["accounting_category"] = table_frame.apply(
        lambda row: _classify_table(row.to_dict()), axis=1
    )
    table_frame["nonempty_cells"] = pd.to_numeric(
        table_frame["nonempty_cells"], errors="coerce"
    ).fillna(0)
    table_frame["rows"] = pd.to_numeric(table_frame["rows"], errors="coerce").fillna(0)
    result = (
        table_frame.groupby("accounting_category", dropna=False)
        .agg(
            source_tables=("sheet", "size"),
            source_rows=("rows", "sum"),
            extracted_cells=("nonempty_cells", "sum"),
        )
        .reset_index()
    )
    result["normalized_measurements"] = result["accounting_category"].map(
        enriched[enriched["review_status"].eq("model_ready")]
        .groupby("accounting_category")
        .size()
        .to_dict()
    ).fillna(0).astype(int)
    result["measurements_awaiting_review"] = result["accounting_category"].map(
        enriched[enriched["review_status"].eq("review_required")]
        .groupby("accounting_category")
        .size()
        .to_dict()
    ).fillna(0).astype(int)
    result["interpretation"] = result["accounting_category"].map(
        {
            "government disease tables eligible for reconstruction": "Aggregate Indonesian disease tables assessed for model-ready measurements",
            "government PDF tables requiring table review": "Tables extracted from reports whose layouts require page-level validation",
            "vector occurrence coordinates": "Individual mosquito occurrence fields, including coordinates and environmental attributes",
            "research participant or survey data requiring governance review": "Study-level participant or questionnaire fields retained outside aggregate surveillance",
            "research and ecological data requiring study-specific parsing": "Scientific study tables needing their own data dictionaries and spatial filters",
            "other disease-related tables requiring review": "Disease-related tables not yet safe for automated normalization",
            "non-disease or contextual tables": "Contextual material whose cells are not disease measurements",
        }
    )
    return result.sort_values("extracted_cells", ascending=False)


def _quality(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    return (
        enriched.groupby(
            [
                "review_status",
                "disease",
                "metric",
                "spatial_resolution",
                "temporal_resolution",
            ],
            dropna=False,
        )
        .agg(
            measurements=("observation_hash", "size"),
            earliest=("period_start", "min"),
            latest=("period_end", "max"),
            source_datasets=("source_dataset_id", "nunique"),
            source_resources=("resource_id", "nunique"),
            minimum_confidence=("normalization_confidence", "min"),
            median_confidence=("normalization_confidence", "median"),
        )
        .reset_index()
        .sort_values("measurements", ascending=False)
    )


def _figure(reconciliation: pd.DataFrame, enriched: pd.DataFrame, path: Path) -> None:
    width = 1800
    height = 920
    left = reconciliation.sort_values("extracted_cells", ascending=False).reset_index(drop=True)
    maximum = max(float(left["extracted_cells"].max()), 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F9FC"/>',
        '<text x="70" y="65" font-family="Arial" font-size="32" font-weight="700" fill="#16324F">Public table reconciliation and enrichment</text>',
        '<text x="70" y="112" font-family="Arial" font-size="22" font-weight="700" fill="#24547A">What the extracted spreadsheet cells contain</text>',
    ]
    for index, row in left.iterrows():
        y = 155 + index * 72
        value = float(row["extracted_cells"])
        bar_width = 560 * math.log10(value + 1) / math.log10(maximum + 1)
        label = html.escape(str(row["accounting_category"]))
        parts.extend(
            [
                f'<text x="70" y="{y}" font-family="Arial" font-size="15" fill="#243B53">{label}</text>',
                f'<rect x="70" y="{y + 12}" width="{bar_width:.1f}" height="24" rx="4" fill="#24547A"/>',
                f'<text x="{82 + bar_width:.1f}" y="{y + 31}" font-family="Arial" font-size="15" fill="#102A43">{int(value):,}</text>',
            ]
        )
    ready = enriched[enriched["review_status"].eq("model_ready")]
    parts.append(
        '<text x="1020" y="112" font-family="Arial" font-size="22" font-weight="700" fill="#24547A">New model-ready measurements by native grain</text>'
    )
    if not ready.empty:
        grouped = (
            ready.groupby(["disease", "spatial_resolution"]).size().reset_index(name="count")
        )
        maximum_ready = max(float(grouped["count"].max()), 1.0)
        colors = {"dengue": "#D1495B", "malaria": "#2A9D8F"}
        for index, row in grouped.iterrows():
            y = 160 + index * 58
            count = int(row["count"])
            bar_width = 560 * count / maximum_ready
            label = html.escape(f"{row['disease']} · {row['spatial_resolution']}")
            parts.extend(
                [
                    f'<text x="1020" y="{y}" font-family="Arial" font-size="16" fill="#243B53">{label}</text>',
                    f'<rect x="1020" y="{y + 10}" width="{bar_width:.1f}" height="24" rx="4" fill="{colors.get(str(row["disease"]), "#6C757D")}"/>',
                    f'<text x="{1032 + bar_width:.1f}" y="{y + 29}" font-family="Arial" font-size="15" fill="#102A43">{count:,}</text>',
                ]
            )
    parts.extend(
        [
            '<text x="70" y="885" font-family="Arial" font-size="15" fill="#52667A">Cell bars use a logarithmic width scale. Measurements are deduplicated source values, not spreadsheet cells.</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _figure_png(reconciliation: pd.DataFrame, enriched: pd.DataFrame, path: Path) -> None:
    width, height = 1800, 920
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    heading_font = _font(22, bold=True)
    label_font = _font(15)
    value_font = _font(15, bold=True)
    note_font = _font(14)
    draw.text((65, 35), "Public table reconciliation and enrichment", fill="#16324F", font=title_font)
    draw.text((65, 95), "What the extracted spreadsheet cells contain", fill="#24547A", font=heading_font)
    left = reconciliation.sort_values("extracted_cells", ascending=False).reset_index(drop=True)
    maximum = max(float(left["extracted_cells"].max()), 1.0)
    for index, row in left.iterrows():
        y = 145 + index * 98
        value = float(row["extracted_cells"])
        bar_width = 650 * math.log10(value + 1) / math.log10(maximum + 1)
        draw.text((65, y), str(row["accounting_category"]), fill="#243B53", font=label_font)
        draw.rounded_rectangle((65, y + 27, 65 + bar_width, y + 55), radius=5, fill="#2B6087")
        draw.text((77 + bar_width, y + 31), f"{int(value):,}", fill="#102A43", font=value_font)
    draw.text((975, 95), "New model-ready measurements by native grain", fill="#24547A", font=heading_font)
    ready = enriched[enriched["review_status"].eq("model_ready")]
    grouped = ready.groupby(["disease", "spatial_resolution"]).size().reset_index(name="count")
    maximum_ready = max(float(grouped["count"].max()), 1.0)
    colors = {"dengue": "#D1495B", "malaria": "#2A9D8F"}
    for index, row in grouped.iterrows():
        y = 145 + index * 63
        count = int(row["count"])
        bar_width = 650 * count / maximum_ready
        label = f"{row['disease']} · {row['spatial_resolution']}"
        draw.text((975, y), label, fill="#243B53", font=label_font)
        draw.rounded_rectangle(
            (975, y + 25, 975 + bar_width, y + 50),
            radius=5,
            fill=colors.get(str(row["disease"]), "#6C757D"),
        )
        draw.text((987 + bar_width, y + 28), f"{count:,}", fill="#102A43", font=value_font)
    draw.text(
        (65, 875),
        "Cell bars use a logarithmic width scale. Measurements are deduplicated source values, not spreadsheet cells.",
        fill="#52667A",
        font=note_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def run(settings: Settings) -> dict[str, Path]:
    config = settings.section("lakehouse")
    dsn = os.getenv(config["dsn_environment"], f"dbname={config['database']}")
    with psycopg.connect(dsn) as connection:
        tables = _fetch_tables(connection)
        cell_rows = _fetch_candidate_cells(connection)
        organizations = _fetch_organizations(connection)
        database_reconciliation = _fetch_database_reconciliation(connection)
    parsed: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for group in _table_groups(cell_rows):
        observations, audit = _parse_table(group, organizations)
        parsed.extend(observations)
        audits.append(audit)
    deduplicated = _deduplicate(parsed)
    enriched = pd.DataFrame.from_records(deduplicated)
    model_ready = enriched[enriched["review_status"].eq("model_ready")].copy()
    reconciliation = _reconciliation(tables, enriched)
    quality = _quality(enriched)
    table_audit = pd.DataFrame.from_records(audits)
    database_summary = _database_role_summary(database_reconciliation)
    paths = {
        "observations_csv": settings.paths.quantitative / "public_enriched_observations.csv",
        "observations_parquet": settings.paths.quantitative
        / "public_enriched_observations.parquet",
        "candidates_csv": settings.paths.quantitative / "public_enrichment_candidates.csv",
        "candidates_parquet": settings.paths.quantitative / "public_enrichment_candidates.parquet",
        "reconciliation_csv": settings.paths.quantitative / "public_cell_reconciliation.csv",
        "reconciliation_parquet": settings.paths.quantitative / "public_cell_reconciliation.parquet",
        "quality_csv": settings.paths.quantitative / "public_enrichment_quality.csv",
        "quality_parquet": settings.paths.quantitative / "public_enrichment_quality.parquet",
        "table_audit_csv": settings.paths.quantitative / "public_table_enrichment_audit.csv",
        "table_audit_parquet": settings.paths.quantitative / "public_table_enrichment_audit.parquet",
        "image": settings.paths.images / "public_cell_reconciliation.svg",
        "image_png": settings.paths.images / "public_cell_reconciliation.png",
        "database_reconciliation_csv": settings.paths.quantitative
        / "lake_record_reconciliation.csv",
        "database_reconciliation_parquet": settings.paths.quantitative
        / "lake_record_reconciliation.parquet",
        "database_summary_csv": settings.paths.quantitative
        / "lake_record_role_summary.csv",
        "database_summary_parquet": settings.paths.quantitative
        / "lake_record_role_summary.parquet",
    }
    model_ready.to_csv(paths["observations_csv"], index=False)
    model_ready.to_parquet(paths["observations_parquet"], index=False)
    enriched.to_csv(paths["candidates_csv"], index=False)
    enriched.to_parquet(paths["candidates_parquet"], index=False)
    reconciliation.to_csv(paths["reconciliation_csv"], index=False)
    reconciliation.to_parquet(paths["reconciliation_parquet"], index=False)
    quality.to_csv(paths["quality_csv"], index=False)
    quality.to_parquet(paths["quality_parquet"], index=False)
    table_audit.to_csv(paths["table_audit_csv"], index=False)
    table_audit.to_parquet(paths["table_audit_parquet"], index=False)
    database_reconciliation.to_csv(paths["database_reconciliation_csv"], index=False)
    database_reconciliation.to_parquet(paths["database_reconciliation_parquet"], index=False)
    database_summary.to_csv(paths["database_summary_csv"], index=False)
    database_summary.to_parquet(paths["database_summary_parquet"], index=False)
    _figure(reconciliation, enriched, paths["image"])
    _figure_png(reconciliation, enriched, paths["image_png"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    settings = Settings.load(arguments.config, arguments.output)
    run(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
