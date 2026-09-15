from __future__ import annotations

import re
from typing import Any

import pandas as pd


def source_catalog() -> pd.DataFrame:
    rows = [
        {
            "source_system": "SATUSEHAT FHIR",
            "operator": "Ministry of Health Indonesia",
            "diseases": "dengue; malaria",
            "data_types": "condition; observation; diagnostic report; specimen",
            "native_spatial_grain": "healthcare organization",
            "native_temporal_grain": "event timestamp",
            "access_class": "credentialed_api",
            "anonymous_public_data": False,
            "contains_person_level_data": True,
            "connector_status": "environment_credentials_required",
            "url": "https://satusehat.kemkes.go.id/platform/docs/id/api-catalogue/",
        },
        {
            "source_system": "SKDR EBS and IBS",
            "operator": "Ministry of Health Indonesia",
            "diseases": "dengue; malaria; outbreak-prone diseases",
            "data_types": "event signals; alerts; suspected cases; confirmed malaria",
            "native_spatial_grain": "reporting facility to province",
            "native_temporal_grain": "event and week",
            "access_class": "public_reports_and_restricted_system",
            "anonymous_public_data": True,
            "contains_person_level_data": False,
            "connector_status": "public_reports_harvested",
            "url": "https://surveilans.kemkes.go.id/kategori/buletin",
        },
        {
            "source_system": "SISMAL V3",
            "operator": "Ministry of Health Indonesia",
            "diseases": "malaria",
            "data_types": "cases; species; investigation; laboratory; vector; logistics",
            "native_spatial_grain": "person and health facility",
            "native_temporal_grain": "event or month depending elimination status",
            "access_class": "credentialed_system",
            "anonymous_public_data": False,
            "contains_person_level_data": True,
            "connector_status": "no_public_api_documentation",
            "url": "https://malaria.kemkes.go.id/node/142",
        },
        {
            "source_system": "WHO Global Health Observatory",
            "operator": "World Health Organization",
            "diseases": "malaria",
            "data_types": "cases; species; tests; intervention coverage",
            "native_spatial_grain": "country",
            "native_temporal_grain": "year",
            "access_class": "open_api",
            "anonymous_public_data": True,
            "contains_person_level_data": False,
            "connector_status": "implemented",
            "url": "https://www.who.int/data/gho/info/gho-odata-api",
        },
        {
            "source_system": "NCBI Nucleotide",
            "operator": "National Center for Biotechnology Information",
            "diseases": "dengue; malaria",
            "data_types": "sequence sample metadata; serotype; species",
            "native_spatial_grain": "sample location when submitted",
            "native_temporal_grain": "collection date when submitted",
            "access_class": "open_api",
            "anonymous_public_data": True,
            "contains_person_level_data": False,
            "connector_status": "implemented",
            "url": "https://www.ncbi.nlm.nih.gov/books/NBK25501/",
        },
        {
            "source_system": "GBIF",
            "operator": "Global Biodiversity Information Facility",
            "diseases": "dengue; malaria",
            "data_types": "vector occurrence; coordinates; sampling metadata",
            "native_spatial_grain": "occurrence coordinates when submitted",
            "native_temporal_grain": "event date when submitted",
            "access_class": "open_api",
            "anonymous_public_data": True,
            "contains_person_level_data": False,
            "connector_status": "implemented",
            "url": "https://techdocs.gbif.org/en/openapi/",
        },
        {
            "source_system": "Indonesian local CKAN portals",
            "operator": "municipal and provincial governments",
            "diseases": "dengue; malaria",
            "data_types": "cases; services; larval indices; plasmodium species",
            "native_spatial_grain": "Puskesmas to province",
            "native_temporal_grain": "month to year",
            "access_class": "open_api_and_download",
            "anonymous_public_data": True,
            "contains_person_level_data": False,
            "connector_status": "implemented",
            "url": "https://data.malangkota.go.id/api/3/action/package_search",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _number(value: str) -> int:
    return int(re.sub(r"\D", "", value))


def _labels(text: str, start: int, count: int) -> list[str]:
    labels: list[str] = []
    for line in text[start:].splitlines():
        value = line.strip()
        if not value:
            continue
        if re.match(r"^(?:M-?\d+|\d|Kasus ISPA Tahun)", value, re.IGNORECASE):
            break
        labels.append(value)
        if len(labels) == count:
            break
    return labels


def parse_skdr(documents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly: list[dict[str, Any]] = []
    ebs: list[dict[str, Any]] = []
    if documents.empty:
        return pd.DataFrame(), pd.DataFrame()
    disease_pattern = re.compile(
        r"Kasus Suspek Dengue Tahun\s+\d{4}\s*\n"
        r"(?P<values>(?:\s*[\d.,]+\s*\n){7,})\s*(?=ISPA)",
        re.IGNORECASE,
    )
    ebs_pattern = re.compile(
        r"Pada M0?1\s*-\s*0?(\d+)\s+(\d{4}),\s*dilaporkan\s+([\d.,]+)\s+"
        r"laporan EBS dengan\s+([\d.,]+)\s+kejadian indikasi KLB dari\s+"
        r"([\d.,]+)\s+provinsi.*?dengan\s+([\d.,]+)\s+kejadian(?: KLB)? "
        r"ditangani dalam waktu",
        re.IGNORECASE | re.DOTALL,
    )
    for document in documents.itertuples(index=False):
        title = str(document.title)
        text = str(document.text)
        report_match = re.search(r"\bM(\d{1,2})\s*(\d{4})\b", title, re.IGNORECASE)
        report_week = int(report_match.group(1)) if report_match else pd.NA
        report_year = int(report_match.group(2)) if report_match else pd.NA
        disease_match = disease_pattern.search(text)
        if disease_match:
            values = re.findall(r"\d[\d.,]*", disease_match.group("values"))
            labels = _labels(text, disease_match.end(), len(values))
            for label, value in zip(labels, values, strict=False):
                weekly.append(
                    {
                        "source": "SKDR public weekly report",
                        "report_year": report_year,
                        "report_week": report_week,
                        "indicator": label,
                        "cumulative_reports": _number(value),
                        "case_definition": "suspected"
                        if label.startswith("Suspek")
                        else "reported",
                        "spatial_resolution": "Indonesia",
                        "source_title": title,
                        "source_url": document.url,
                    }
                )
        ebs_match = ebs_pattern.search(text)
        if ebs_match:
            ebs.append(
                {
                    "source": "SKDR EBS public weekly report",
                    "report_year": report_year,
                    "report_week": report_week,
                    "coverage_end_week": int(ebs_match.group(1)),
                    "coverage_year": int(ebs_match.group(2)),
                    "cumulative_ebs_reports": _number(ebs_match.group(3)),
                    "cumulative_possible_outbreak_events": _number(ebs_match.group(4)),
                    "provinces_with_possible_outbreak_events": _number(ebs_match.group(5)),
                    "events_handled_within_7_days": _number(ebs_match.group(6)),
                    "confirmation_status": "requires_epidemiological_investigation",
                    "source_title": title,
                    "source_url": document.url,
                }
            )
    weekly_frame = pd.DataFrame.from_records(weekly)
    ebs_frame = pd.DataFrame.from_records(ebs)
    if not weekly_frame.empty:
        weekly_frame = weekly_frame.drop_duplicates(
            ["report_year", "report_week", "indicator", "cumulative_reports"]
        )
    if not ebs_frame.empty:
        ebs_frame = ebs_frame.drop_duplicates(
            ["coverage_year", "coverage_end_week", "cumulative_ebs_reports"]
        )
    return weekly_frame, ebs_frame
