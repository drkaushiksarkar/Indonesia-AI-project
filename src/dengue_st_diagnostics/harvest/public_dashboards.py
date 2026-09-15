from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

from dengue_st_diagnostics.config import Settings


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result[:80] or "asset"


def _suffix(content_type: str, url: str) -> str:
    path_suffix = Path(urlsplit(url).path).suffix.casefold()
    if path_suffix in {".csv", ".doc", ".docx", ".html", ".json", ".pdf", ".xlsx"}:
        return path_suffix
    values = {
        "application/json": ".json",
        "application/pdf": ".pdf",
        "text/csv": ".csv",
        "text/html": ".html",
    }
    return values.get(content_type.split(";")[0].casefold(), ".bin")


def _capture(
    session: requests.Session,
    root: Path,
    source_id: str,
    dataset_name: str,
    url: str,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> tuple[dict[str, Any], bytes]:
    retrieved_at = _now()
    request_values = params if method == "GET" else data
    try:
        response = session.request(
            method,
            url,
            params=params,
            data=data,
            headers=headers,
            timeout=timeout,
        )
        body = response.content
        content_type = response.headers.get("Content-Type", "")
        digest = hashlib.sha256(body).hexdigest()
        directory = root / _slug(source_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_slug(dataset_name)}-{digest[:16]}{_suffix(content_type, url)}"
        if not path.exists():
            path.write_bytes(body)
        status = "harvested" if response.ok else "http_error"
        record = {
            "source_id": source_id,
            "dataset_name": dataset_name,
            "retrieved_at": retrieved_at,
            "request_url": response.url,
            "request_method": method,
            "request_parameters": json.dumps(
                request_values or {}, ensure_ascii=False, sort_keys=True
            ),
            "http_status": response.status_code,
            "content_type": content_type,
            "byte_count": len(body),
            "sha256": digest,
            "path": str(path.resolve()),
            "access_class": "anonymous_public",
            "license": "source_terms_apply",
            "status": status,
            "error": "" if response.ok else response.reason,
        }
        return record, body
    except requests.RequestException as exc:
        record = {
            "source_id": source_id,
            "dataset_name": dataset_name,
            "retrieved_at": retrieved_at,
            "request_url": url,
            "request_method": method,
            "request_parameters": json.dumps(
                request_values or {}, ensure_ascii=False, sort_keys=True
            ),
            "http_status": 0,
            "content_type": "",
            "byte_count": 0,
            "sha256": "",
            "path": "",
            "access_class": "anonymous_public",
            "license": "source_terms_apply",
            "status": "request_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return record, b""


def _json_records(
    body: bytes,
    asset: dict[str, Any],
    dataset_name: str,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    values = value if isinstance(value, list) else [value]
    return [
        {
            "source_id": asset["source_id"],
            "dataset_name": dataset_name,
            "record_index": index,
            "record_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
            "asset_sha256": asset["sha256"],
            "retrieved_at": asset["retrieved_at"],
        }
        for index, item in enumerate(values)
    ]


def _csv_records(
    body: bytes,
    asset: dict[str, Any],
    dataset_name: str,
) -> list[dict[str, Any]]:
    try:
        text = body.decode("utf-8-sig")
        values = list(csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error):
        return []
    return [
        {
            "source_id": asset["source_id"],
            "dataset_name": dataset_name,
            "record_index": index,
            "record_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
            "asset_sha256": asset["sha256"],
            "retrieved_at": asset["retrieved_at"],
        }
        for index, item in enumerate(values)
    ]


def _malaria(
    settings: Settings,
    session: requests.Session,
    assets: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    config = settings.section("open_dashboards")
    base = config["malaria_base_url"].rstrip("/")
    root = settings.paths.raw / "open_dashboards"
    endpoints = {
        "malaria_case_graph": f"{base}/api/sismal/case/graph",
        "malaria_menu_graph": f"{base}/api/sismal/menu/graph",
        "malaria_lab_graph": f"{base}/api/sismal/lab/graph",
    }
    for name, url in endpoints.items():
        asset, body = _capture(session, root, "malaria_public_sismal", name, url)
        assets.append(asset)
        records.extend(_json_records(body, asset, name))
    current_year = datetime.now(UTC).year
    for year in range(int(config["malaria_start_year"]), current_year + 1):
        for name, template in {
            "malaria_national_detail": f"{base}/api/sismal/menu/{{year}}",
            "malaria_district_map": f"{base}/api/sismal/chartMap/{{year}}",
        }.items():
            dataset_name = f"{name}_{year}"
            asset, body = _capture(
                session,
                root,
                "malaria_public_sismal",
                dataset_name,
                template.format(year=year),
            )
            assets.append(asset)
            records.extend(_json_records(body, asset, dataset_name))
    library_asset, library_body = _capture(
        session,
        root,
        "malaria_public_library",
        "malaria_library_index",
        config["malaria_library_url"],
    )
    assets.append(library_asset)
    if library_asset["status"] != "harvested":
        return
    soup = BeautifulSoup(library_body, "html.parser")
    links: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(config["malaria_library_url"], str(anchor["href"]))
        suffix = Path(urlsplit(url).path).suffix.casefold()
        if suffix in {".csv", ".doc", ".docx", ".pdf", ".xls", ".xlsx"}:
            links[url] = anchor.get_text(" ", strip=True) or Path(urlsplit(url).path).name
    for index, (url, title) in enumerate(sorted(links.items())):
        asset, _ = _capture(
            session,
            root,
            "malaria_public_library",
            f"library_{index}_{title}",
            url,
            timeout=240,
        )
        assets.append(asset)


def _tableau_export_url(value: str, extension: str) -> str:
    base = value.split("?", maxsplit=1)[0]
    return f"{base}.{extension}?:showVizHome=no"


def _satusehat(
    settings: Settings,
    session: requests.Session,
    assets: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    config = settings.section("open_dashboards")
    root = settings.paths.raw / "open_dashboards"
    for index, page_url in enumerate(config["satusehat_pages"]):
        page_name = f"satusehat_dashboard_{index + 1}"
        page_asset, body = _capture(
            session,
            root,
            "satusehat_public_tableau",
            f"{page_name}_page",
            page_url,
        )
        assets.append(page_asset)
        text = html.unescape(body.decode("utf-8", errors="replace")).replace("\\u0026", "&")
        matches = re.findall(
            r"https://dashboard\.kemkes\.go\.id/(?:#/)?views/[^\"'<>\\]+",
            text,
        )
        if not matches:
            continue
        view_url = matches[0].replace("/#/views/", "/views/")
        for extension in ["csv", "pdf"]:
            dataset_name = f"{page_name}_{extension}"
            asset, export_body = _capture(
                session,
                root,
                "satusehat_public_tableau",
                dataset_name,
                _tableau_export_url(view_url, extension),
                timeout=240,
            )
            assets.append(asset)
            if extension == "csv":
                records.extend(_csv_records(export_body, asset, dataset_name))
        looker_matches = re.findall(
            r"https://lookerstudio\.google\.com/embed/reporting/[^\"'<>\\]+",
            text,
        )
        for looker_index, looker_url in enumerate(looker_matches):
            asset, _ = _capture(
                session,
                root,
                "satusehat_public_tableau",
                f"{page_name}_looker_{looker_index + 1}",
                looker_url,
                timeout=240,
            )
            assets.append(asset)


def _skdr(
    settings: Settings,
    session: requests.Session,
    assets: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    config = settings.section("kemkes")
    root = settings.paths.raw / "open_dashboards"
    page_asset, body = _capture(
        session,
        root,
        "skdr_public_live",
        "skdr_live_dashboard",
        config["live_dashboard_url"],
    )
    assets.append(page_asset)
    text = body.decode("utf-8", errors="replace")
    token_match = re.search(r"'_csrf':\s*'([^']+)'", text)
    if page_asset["status"] != "harvested" or token_match is None:
        return
    token = token_match.group(1)
    headers = {
        "Referer": config["live_dashboard_url"],
        "X-Requested-With": "XMLHttpRequest",
    }
    current_year = datetime.now(UTC).year
    endpoints = [
        "https://surkarkes.kemkes.go.id/index.php/api-skk/grafik-suspek-ibs",
        "https://surkarkes.kemkes.go.id/index.php/api-skk/grafik-prov-ibs",
    ]
    for year in range(int(config["live_start_year"]), current_year + 1):
        for disease_code in ["12MAL", "16SDF"]:
            for endpoint in endpoints:
                name = endpoint.rsplit("/", maxsplit=1)[-1]
                data = {
                    "_csrf": token,
                    "kode_penyakit": disease_code,
                    "tahun": str(year),
                    "kd_prop": "",
                    "kd_kab": "",
                    "jenis_waktu": "1",
                    "minggu1": "",
                    "minggu2": "",
                    "tahun1": "",
                    "tahun2": "",
                    "multi_penyakit": "0",
                }
                asset, response_body = _capture(
                    session,
                    root,
                    "skdr_public_live",
                    f"{name}_{disease_code}_{year}",
                    endpoint,
                    method="POST",
                    data=data,
                    headers=headers,
                )
                assets.append(asset)
                response_text = response_body.decode("utf-8", errors="replace")
                parsed: list[dict[str, Any]] = []
                if name == "grafik-prov-ibs":
                    for province, value in re.findall(
                        r"\['([^']+)',\s*([\d.]+)\]",
                        response_text,
                    ):
                        parsed.append(
                            {
                                "year": year,
                                "disease_code": disease_code,
                                "province": province.strip(),
                                "cases": value,
                                "temporal_resolution": "year_to_date",
                            }
                        )
                else:
                    pattern = re.compile(
                        r"'label':\s*'Mi-(\d+)-(\d+)'\s*,\s*"
                        r"'value':\s*([\d.]+)\s*,\s*'value2':\s*([\d.]+)",
                        re.DOTALL,
                    )
                    for week, short_year, value, outbreak_value in pattern.findall(response_text):
                        parsed.append(
                            {
                                "year": 2000 + int(short_year),
                                "week": int(week),
                                "disease_code": disease_code,
                                "cases": value,
                                "outbreak_cases": outbreak_value,
                                "temporal_resolution": "week",
                            }
                        )
                for record_index, item in enumerate(parsed):
                    records.append(
                        {
                            "source_id": asset["source_id"],
                            "dataset_name": asset["dataset_name"],
                            "record_index": record_index,
                            "record_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
                            "asset_sha256": asset["sha256"],
                            "retrieved_at": asset["retrieved_at"],
                        }
                    )


def _silantor(
    settings: Settings,
    session: requests.Session,
    assets: list[dict[str, Any]],
) -> None:
    config = settings.section("open_dashboards")
    root = settings.paths.raw / "open_dashboards"
    page_bodies: dict[str, bytes] = {}
    for index, page_url in enumerate(config["silantor_pages"]):
        asset, body = _capture(
            session,
            root,
            "silantor_public",
            f"silantor_page_{index + 1}",
            page_url,
        )
        assets.append(asset)
        page_bodies[page_url] = body
    dashboard_url = config["silantor_pages"][0]
    soup = BeautifulSoup(page_bodies.get(dashboard_url, b""), "html.parser")
    years = [option.get("value") for option in soup.select("#tahun option[value]")]
    months = [option.get("value") for option in soup.select("#bulan option[value]")]
    years = [value for value in years if value and str(value).isdigit()]
    months = [value for value in months if value and str(value).isdigit()]
    if not years:
        years = [str(datetime.now(UTC).year)]
    if not months:
        months = [f"{datetime.now(UTC).month:02d}"]
    headers = {"Referer": dashboard_url, "X-Requested-With": "XMLHttpRequest"}
    consecutive_errors = 0
    for year in sorted(set(years)):
        for month in sorted(set(months)):
            for endpoint in ["data-dashboard", "data-posbindu-dashboard"]:
                asset, _ = _capture(
                    session,
                    root,
                    "silantor_public",
                    f"{endpoint}_{year}_{month}",
                    f"https://silantor.kemkes.go.id/apis/{endpoint}",
                    params={"year": year, "month": month},
                    headers=headers,
                )
                assets.append(asset)
                consecutive_errors = consecutive_errors + 1 if asset["status"] != "harvested" else 0
            if consecutive_errors >= 6:
                break
        if consecutive_errors >= 6:
            break
    for endpoint, params in [
        ("dashboard-keragaman-spesies", {"pulau": "", "genus_id": ""}),
        (
            "dashboard-bank-data",
            {
                "year": years[-1],
                "month": months[-1],
                "province_id": "",
                "city_id": "",
                "jenis_bank_data_id": "",
                "kategori_bank_data_id": "",
            },
        ),
    ]:
        asset, _ = _capture(
            session,
            root,
            "silantor_public",
            endpoint,
            f"https://silantor.kemkes.go.id/apis/{endpoint}",
            params=params,
            headers=headers,
        )
        assets.append(asset)


def harvest_open_dashboards(settings: Settings) -> dict[str, pd.DataFrame]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "*/*",
            "User-Agent": "IndonesiaVectorDataLake/1.0 public-health-research",
        }
    )
    assets: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    _malaria(settings, session, assets, records)
    _satusehat(settings, session, assets, records)
    _skdr(settings, session, assets, records)
    _silantor(settings, session, assets)
    blocked = pd.DataFrame.from_records(
        [
            {
                "source_id": "satusehat_fhir",
                "access_class": "credentialed_api",
                "reason": "client_credentials_required",
                "endpoint": "https://api-satusehat.kemkes.go.id/fhir-r4/v1",
            },
            {
                "source_id": "sismal_internal_api",
                "access_class": "credentialed_api",
                "reason": "authorization_token_required",
                "endpoint": "https://sismal.kemkes.go.id/api",
            },
        ]
    )
    return {
        "open_dashboard_assets": pd.DataFrame.from_records(assets),
        "open_dashboard_records": pd.DataFrame.from_records(records),
        "credential_blocked_sources": blocked,
    }
