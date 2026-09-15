from __future__ import annotations

import io
import json
import math
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import safe_name, sha256

FORMATS = {
    "csv": "csv",
    "doc": "doc",
    "docx": "docx",
    "geojson": "geojson",
    "json": "json",
    "pdf": "pdf",
    "sav": "sav",
    "xls": "xls",
    "xlsx": "xlsx",
    "xml": "xml",
    "zip": "zip",
}
USER_AGENT = "dengue-st-diagnostics/0.1"


def _response(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    value = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=(20, 120),
    )
    value.raise_for_status()
    return value


def _flight_text(html: str) -> str:
    values: list[str] = []
    decoder = json.JSONDecoder()
    marker = "self.__next_f.push("
    for script in BeautifulSoup(html, "html.parser").find_all("script"):
        text = script.string or script.get_text()
        position = 0
        while True:
            start = text.find(marker, position)
            if start < 0:
                break
            offset = start + len(marker)
            try:
                payload, consumed = decoder.raw_decode(text[offset:])
                if len(payload) > 1 and isinstance(payload[1], str):
                    values.append(payload[1])
                position = offset + consumed
            except (json.JSONDecodeError, TypeError):
                position = offset + 1
    return "".join(values)


def _data_go_package(html: str) -> dict[str, Any]:
    flight = _flight_text(html)
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", flight)]
    for start in reversed(starts):
        try:
            value, _ = decoder.raw_decode(flight[start:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("resources"), list)
            and value.get("name")
            and value.get("title")
        ):
            return value
    raise ValueError("dataset package unavailable")


def _data_go_links(html: str, base_url: str) -> tuple[list[str], int]:
    soup = BeautifulSoup(html, "html.parser")
    links = sorted(
        {
            urljoin(base_url, str(anchor["href"]))
            for anchor in soup.find_all("a", href=True)
            if str(anchor["href"]).startswith("/dataset/dataset/")
        }
    )
    text = soup.get_text(" ", strip=True)
    match = re.search(r"([\d.,]+)\s+Datasets Found", text)
    count = int(re.sub(r"\D", "", match.group(1))) if match else len(links)
    return links, count


def _search_data_go(
    base_url: str,
    query: str,
    page: int,
) -> tuple[str, int, list[str], str]:
    try:
        html = _response(
            f"{base_url.rstrip('/')}/dataset",
            {"q": query, "perPage": 20, "page": page},
        ).text
        links, count = _data_go_links(html, base_url)
        return query, count, links, ""
    except (requests.RequestException, ValueError, TypeError) as exc:
        return query, 0, [], f"{type(exc).__name__}: {exc}"


def _cache_package(value: tuple[str, Path]) -> tuple[str, dict[str, Any] | None, str]:
    url, path = value
    try:
        if path.exists():
            html = path.read_text(encoding="utf-8")
        else:
            html = _response(url).text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
        return url, _data_go_package(html), ""
    except (requests.RequestException, OSError, ValueError, TypeError) as exc:
        return url, None, f"{type(exc).__name__}: {exc}"


def _format(resource: dict[str, Any], label: str = "") -> str:
    declared = str(resource.get("format") or "").casefold().strip(" .")
    suffix = Path(urlparse(str(resource.get("url") or "")).path).suffix.casefold().strip(".")
    label_suffix = Path(label).suffix.casefold().strip(".")
    return FORMATS.get(declared) or FORMATS.get(suffix) or FORMATS.get(label_suffix) or ""


def _download_url(url: str, extension: str) -> str:
    sheet = re.search(r"docs\.google\.com/spreadsheets/d/([^/]+)", url)
    drive = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if sheet is not None:
        return f"https://docs.google.com/spreadsheets/d/{sheet.group(1)}/export?format=xlsx"
    if drive is not None:
        return f"https://drive.google.com/uc?export=download&id={drive.group(1)}"
    data_go = re.search(r"https?://data\.go\.id/api-be/file/download/(.+)$", url)
    if data_go is not None:
        return f"https://file.data.go.id/sdi/backend/migrated/upload-dir/{data_go.group(1)}"
    if extension == "xlsx" and "docs.google.com" in url:
        return url.replace("/edit", "/export?format=xlsx").split("?", maxsplit=1)[0]
    return url


def _disease(title: str) -> str:
    value = title.casefold()
    dengue = any(term in value for term in ["dengue", "dbd", "demam berdarah", "aedes", "jentik"])
    malaria = any(term in value for term in ["malaria", "plasmodium", "anopheles", "knowlesi"])
    if dengue and malaria:
        return "dengue;malaria"
    if dengue:
        return "dengue"
    if malaria:
        return "malaria"
    return "vector"


def _data_go(
    settings: Settings,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    base_url = str(config["data_go_base_url"])
    queries = list(config["data_go_queries"])
    workers = int(config["download_workers"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        first = list(
            executor.map(
                lambda query: _search_data_go(base_url, query, 1),
                queries,
            )
        )
    tasks = [
        (base_url, query, page)
        for query, count, _, error in first
        if not error
        for page in range(2, math.ceil(count / 20) + 1)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        remaining = list(executor.map(lambda item: _search_data_go(*item), tasks))
    all_results = [*first, *remaining]
    links = sorted({link for _, _, found, _ in all_results for link in found})
    search_rows = []
    for query in queries:
        selected = [row for row in all_results if row[0] == query]
        search_rows.append(
            {
                "source": "data.go.id",
                "query": query,
                "reported_datasets": max((row[1] for row in selected), default=0),
                "discovered_links": len({link for row in selected for link in row[2]}),
                "pages_requested": len(selected),
                "failed_pages": sum(bool(row[3]) for row in selected),
                "error": "; ".join(row[3] for row in selected if row[3]),
            }
        )
    cached = {
        url: settings.paths.raw
        / "public_sources"
        / "data_go"
        / "pages"
        / f"{safe_name(url.rsplit('/', 1)[-1])}.html"
        for url in links
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        packages = list(executor.map(_cache_package, cached.items()))
    catalog_rows = []
    resources: list[dict[str, Any]] = []
    for url, package, error in packages:
        if package is None:
            catalog_rows.append(
                {
                    "source": "data.go.id",
                    "dataset_id": url.rsplit("/", 1)[-1],
                    "dataset_title": "",
                    "organization": "",
                    "disease": "",
                    "landing_url": url,
                    "access_level": "",
                    "resource_count": 0,
                    "status": "failed",
                    "error": error,
                }
            )
            continue
        title = str(package.get("title") or "")
        organization = package.get("organization") or {}
        dataset_id = str(package.get("id") or package.get("name") or url.rsplit("/", 1)[-1])
        catalog_rows.append(
            {
                "source": "data.go.id",
                "dataset_id": dataset_id,
                "dataset_title": title,
                "organization": organization.get("title") or organization.get("name") or "",
                "disease": _disease(title),
                "landing_url": url,
                "access_level": next(
                    (
                        item.get("value")
                        for item in package.get("extras") or []
                        if item.get("key") == "accesslevel"
                    ),
                    "",
                ),
                "resource_count": len(package.get("resources") or []),
                "status": "parsed",
                "error": "",
            }
        )
        for resource in package.get("resources") or []:
            resource_url = str(resource.get("download") or resource.get("url") or "")
            extension = _format(resource, str(resource.get("name") or ""))
            resource_id = str(resource.get("id") or safe_name(resource_url))
            resources.append(
                {
                    "source": "data.go.id",
                    "dataset_id": dataset_id,
                    "dataset_title": title,
                    "organization": organization.get("title") or organization.get("name") or "",
                    "disease": _disease(title),
                    "resource_id": resource_id,
                    "resource_name": resource.get("name") or resource.get("description") or "",
                    "format": extension,
                    "url": resource_url,
                    "download_url": _download_url(resource_url, extension),
                    "landing_url": url,
                    "path": str(
                        settings.paths.raw
                        / "public_sources"
                        / "data_go"
                        / "resources"
                        / f"{safe_name(resource_id)}.{extension or 'html'}"
                    ),
                    "status": "discovered",
                    "bytes": pd.NA,
                    "sha256": "",
                    "error": "",
                }
            )
    return (
        pd.DataFrame.from_records(search_rows),
        pd.DataFrame.from_records(catalog_rows),
        resources,
    )


def _landing_resource_rows(
    settings: Settings,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resources: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    for index, url in enumerate(config["landing_pages"]):
        path = settings.paths.raw / "public_sources" / "landing_pages" / f"page_{index}.html"
        status = "cached" if path.exists() else "downloaded"
        error = ""
        try:
            if path.exists():
                html = path.read_text(encoding="utf-8")
            else:
                html = _response(url).text
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(html, encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")
            for anchor_index, anchor in enumerate(soup.find_all("a", href=True)):
                label = anchor.get_text(" ", strip=True)
                resource_url = urljoin(url, str(anchor["href"]))
                extension = _format({"url": resource_url}, label)
                if not extension:
                    continue
                resource_id = safe_name(f"landing_{index}_{anchor_index}_{label}")
                resources.append(
                    {
                        "source": urlparse(url).netloc,
                        "dataset_id": f"landing_{index}",
                        "dataset_title": (
                            soup.title.get_text(" ", strip=True) if soup.title else url
                        ),
                        "organization": urlparse(url).netloc,
                        "disease": _disease(f"{label} {url}"),
                        "resource_id": resource_id,
                        "resource_name": label,
                        "format": extension,
                        "url": resource_url,
                        "download_url": _download_url(resource_url, extension),
                        "landing_url": url,
                        "path": str(
                            settings.paths.raw
                            / "public_sources"
                            / "landing_files"
                            / f"{resource_id}.{extension}"
                        ),
                        "status": "discovered",
                        "bytes": pd.NA,
                        "sha256": "",
                        "error": "",
                    }
                )
        except (requests.RequestException, OSError, ValueError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        pages.append(
            {
                "source": urlparse(url).netloc,
                "url": url,
                "path": str(path),
                "status": status,
                "sha256": sha256(path) if path.exists() else "",
                "error": error,
            }
        )
    return resources, pages


def _zenodo_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    searches: list[dict[str, Any]] = []
    page_size = int(config["repository_page_size"])
    for query in config["repository_queries"]:
        page = 1
        total = 0
        error = ""
        while True:
            try:
                payload = _response(
                    "https://zenodo.org/api/records",
                    {"q": query, "size": page_size, "page": page},
                ).json()
                hits = payload.get("hits", {}).get("hits") or []
                total = int(payload.get("hits", {}).get("total") or total)
                for hit in hits:
                    records[int(hit["id"])] = hit
                if not hits or len(hits) < page_size:
                    break
                page += 1
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
        searches.append(
            {
                "source": "zenodo",
                "query": query,
                "reported_records": total,
                "pages_requested": page,
                "error": error,
            }
        )
    for record_id in config["zenodo_record_ids"]:
        try:
            records[int(record_id)] = _response(
                f"https://zenodo.org/api/records/{int(record_id)}"
            ).json()
        except (requests.RequestException, ValueError, TypeError):
            continue
    return list(records.values()), searches


def _repository(
    settings: Settings,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    records, searches = _zenodo_records(config)
    catalog = []
    resources: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        title = str(metadata.get("title") or record.get("title") or "")
        record_id = str(record.get("id"))
        description = BeautifulSoup(str(metadata.get("description") or ""), "html.parser").get_text(
            " "
        )
        catalog.append(
            {
                "source": "zenodo",
                "record_id": record_id,
                "title": title,
                "disease": _disease(f"{title} {description}"),
                "publication_date": metadata.get("publication_date"),
                "resource_type": (metadata.get("resource_type") or {}).get("type"),
                "access_right": metadata.get("access_right"),
                "doi": metadata.get("doi") or record.get("doi"),
                "landing_url": (record.get("links") or {}).get("self_html"),
                "file_count": len(record.get("files") or []),
            }
        )
        for file_index, file_value in enumerate(record.get("files") or []):
            name = str(file_value.get("key") or f"file_{file_index}")
            extension = _format({"url": name}, name)
            if not extension:
                continue
            resource_id = safe_name(f"zenodo_{record_id}_{file_index}_{name}")
            resources.append(
                {
                    "source": "zenodo",
                    "dataset_id": record_id,
                    "dataset_title": title,
                    "organization": "Zenodo",
                    "disease": _disease(title),
                    "resource_id": resource_id,
                    "resource_name": name,
                    "format": extension,
                    "url": (file_value.get("links") or {}).get("self") or "",
                    "download_url": (file_value.get("links") or {}).get("self") or "",
                    "landing_url": (record.get("links") or {}).get("self_html") or "",
                    "path": str(
                        settings.paths.raw
                        / "public_sources"
                        / "repositories"
                        / "zenodo"
                        / record_id
                        / name
                    ),
                    "status": "discovered",
                    "bytes": file_value.get("size"),
                    "sha256": "",
                    "error": "",
                }
            )
    return (
        pd.DataFrame.from_records(catalog),
        pd.DataFrame.from_records(searches),
        resources,
    )


def _download_resource(value: tuple[dict[str, Any], int]) -> dict[str, Any]:
    row, max_bytes = value
    result = dict(row)
    url = str(row.get("download_url") or "")
    path = Path(str(row["path"]))
    if not url:
        result["status"] = "failed"
        result["error"] = "missing download URL"
        return result
    if path.exists() and path.stat().st_size > 0:
        result["status"] = "cached"
        result["bytes"] = path.stat().st_size
        result["sha256"] = sha256(path)
        return result
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=(15, 180),
            stream=True,
        ) as response:
            response.raise_for_status()
            reported = int(response.headers.get("Content-Length") or 0)
            if reported > max_bytes:
                result["status"] = "skipped_size"
                result["bytes"] = reported
                result["error"] = f"content length exceeds {max_bytes}"
                return result
            path.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise OverflowError(f"download exceeds {max_bytes}")
                    handle.write(chunk)
        partial.replace(path)
        result["status"] = "downloaded"
        result["bytes"] = path.stat().st_size
        result["sha256"] = sha256(path)
    except (requests.RequestException, OSError, OverflowError, ValueError) as exc:
        partial.unlink(missing_ok=True)
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _flatten(child, f"{prefix}.{key}".strip("."))
        ]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in _flatten(child, f"{prefix}[{index}]")
        ]
    return [(prefix, value)]


def _dashboard_call(
    value: tuple[str, int | None, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    url, year, path = value
    status = "downloaded"
    error = ""
    records: list[dict[str, Any]] = []
    try:
        response = _response(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        try:
            payload = response.json()
            status = "empty" if payload in ([], {}, None) else "parsed"
            records = [
                {
                    "endpoint": url,
                    "year": year,
                    "record_path": record_path,
                    "value": item,
                }
                for record_path, item in _flatten(payload)
            ]
        except requests.JSONDecodeError:
            status = "non_json"
            error = response.headers.get("Content-Type") or ""
    except (requests.RequestException, OSError, ValueError, TypeError) as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return (
        {
            "source": urlparse(url).netloc,
            "endpoint": url,
            "year": year,
            "path": str(path),
            "status": status,
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256(path) if path.exists() else "",
            "error": error,
        },
        records,
    )


def _dashboards(
    settings: Settings,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_year = pd.Timestamp.now().year
    calls = [
        (
            str(url),
            None,
            settings.paths.raw / "public_sources" / "dashboards" / f"fixed_{index}.json",
        )
        for index, url in enumerate(config["dashboard_endpoints"])
    ]
    calls.extend(
        (
            str(template).format(year=year),
            year,
            settings.paths.raw
            / "public_sources"
            / "dashboards"
            / f"year_{template_index}_{year}.json",
        )
        for template_index, template in enumerate(config["dashboard_year_templates"])
        for year in range(int(config["dashboard_start_year"]), current_year + 1)
    )
    with ThreadPoolExecutor(max_workers=int(config["download_workers"])) as executor:
        results = list(executor.map(_dashboard_call, calls))
    return (
        pd.DataFrame.from_records(row for row, _ in results),
        pd.DataFrame.from_records(record for _, values in results for record in values),
    )


def _frames(data: bytes, suffix: str) -> dict[str, pd.DataFrame]:
    if suffix == ".csv":
        for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                return {
                    "csv": pd.read_csv(
                        io.BytesIO(data),
                        header=None,
                        dtype="string",
                        sep=None,
                        engine="python",
                        encoding=encoding,
                    )
                }
            except UnicodeDecodeError:
                continue
        return {}
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype="string")
    if suffix in {".json", ".geojson"}:
        value = json.loads(data)
        if isinstance(value, list):
            return {"json": pd.json_normalize(value)}
        if isinstance(value, dict):
            candidates = [(key, item) for key, item in value.items() if isinstance(item, list)]
            if candidates:
                return {str(key): pd.json_normalize(item) for key, item in candidates}
            return {"json": pd.json_normalize(value)}
    return {}


def _frame_records(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    sheet: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = frame.dropna(how="all").dropna(axis=1, how="all")
    table = {
        **metadata,
        "sheet": sheet,
        "rows": len(value),
        "columns": len(value.columns),
        "nonempty_cells": int(value.notna().sum().sum()),
    }
    cells = [
        {
            **metadata,
            "sheet": sheet,
            "row_number": int(row_index) + 1,
            "column_number": int(column_index) + 1,
            "value": str(item),
        }
        for row_index, row in value.iterrows()
        for column_index, item in enumerate(row)
        if pd.notna(item)
    ]
    return table, cells


def _pdf_records(
    data: bytes,
    metadata: dict[str, Any],
    extract_tables: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(data)) as document:
        for page_number, page in enumerate(document.pages, start=1):
            pages.append(
                {
                    **metadata,
                    "page_number": page_number,
                    "text": page.extract_text() or "",
                }
            )
            if extract_tables:
                for table_number, values in enumerate(page.extract_tables(), start=1):
                    frame = pd.DataFrame(values)
                    table, extracted = _frame_records(
                        frame,
                        metadata,
                        f"page_{page_number}_table_{table_number}",
                    )
                    tables.append(table)
                    cells.extend(extracted)
    return pages, tables, cells


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n".join(text for text in root.itertext() if text.strip())


def _extract_one(
    value: tuple[dict[str, Any], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    row, extract_tables = value
    path = Path(str(row["path"]))
    metadata = {
        "source": row.get("source"),
        "dataset_id": row.get("dataset_id"),
        "dataset_title": row.get("dataset_title"),
        "disease": row.get("disease"),
        "resource_id": row.get("resource_id"),
        "resource_name": row.get("resource_name"),
        "source_url": row.get("url"),
        "path": str(path),
        "member": "",
    }
    pages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    status = {**metadata, "status": "parsed", "error": ""}
    try:
        data = path.read_bytes()
        suffix = path.suffix.casefold()
        if suffix in {".csv", ".xls", ".xlsx", ".json", ".geojson"}:
            for sheet, frame in _frames(data, suffix).items():
                table, extracted = _frame_records(frame, metadata, sheet)
                tables.append(table)
                cells.extend(extracted)
        elif suffix == ".pdf":
            pages, tables, cells = _pdf_records(data, metadata, extract_tables)
        elif suffix == ".docx":
            pages.append({**metadata, "page_number": pd.NA, "text": _docx_text(data)})
        elif suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.namelist():
                    member_suffix = Path(member).suffix.casefold()
                    member_metadata = {**metadata, "member": member}
                    member_data = archive.read(member)
                    if member_suffix in {".csv", ".xls", ".xlsx", ".json", ".geojson"}:
                        for sheet, frame in _frames(member_data, member_suffix).items():
                            table, extracted = _frame_records(frame, member_metadata, sheet)
                            tables.append(table)
                            cells.extend(extracted)
                    elif member_suffix == ".pdf":
                        member_pages, member_tables, member_cells = _pdf_records(
                            member_data,
                            member_metadata,
                            extract_tables,
                        )
                        pages.extend(member_pages)
                        tables.extend(member_tables)
                        cells.extend(member_cells)
                    elif member_suffix == ".docx":
                        pages.append(
                            {
                                **member_metadata,
                                "page_number": pd.NA,
                                "text": _docx_text(member_data),
                            }
                        )
        else:
            status["status"] = "unsupported_extraction"
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
    return pages, tables, cells, status


def harvest_public_sources(
    settings: Settings,
    enabled: bool = True,
) -> dict[str, pd.DataFrame]:
    names = [
        "public_data_go_search_status",
        "public_dataset_catalog",
        "public_repository_catalog",
        "public_repository_search_status",
        "public_landing_page_status",
        "public_source_resources",
        "public_dashboard_status",
        "public_dashboard_records",
        "public_extraction_status",
        "public_tabular_tables",
        "public_tabular_cells",
        "public_document_pages",
    ]
    if not enabled:
        return {name: pd.DataFrame() for name in names}
    config = settings.section("public_harvest")
    data_go_search, data_go_catalog, data_go_resources = _data_go(settings, config)
    repository_catalog, repository_search, repository_resources = _repository(settings, config)
    landing_resources, landing_status = _landing_resource_rows(settings, config)
    discovered = [*data_go_resources, *repository_resources, *landing_resources]
    harvested = _download_extract(discovered, config)
    dashboard_status, dashboard_records = _dashboards(settings, config)
    return {
        "public_data_go_search_status": data_go_search,
        "public_dataset_catalog": data_go_catalog,
        "public_repository_catalog": repository_catalog,
        "public_repository_search_status": repository_search,
        "public_landing_page_status": pd.DataFrame.from_records(landing_status),
        "public_source_resources": harvested["resources"],
        "public_dashboard_status": dashboard_status,
        "public_dashboard_records": dashboard_records,
        "public_extraction_status": harvested["extraction_status"],
        "public_tabular_tables": harvested["tables"],
        "public_tabular_cells": harvested["cells"],
        "public_document_pages": harvested["pages"],
    }


def _download_extract(
    discovered: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    unique: dict[str, dict[str, Any]] = {}
    for row in discovered:
        key = str(row.get("download_url") or row.get("url") or row.get("resource_id"))
        unique.setdefault(key, row)
    with ThreadPoolExecutor(max_workers=int(config["download_workers"])) as executor:
        downloaded = list(
            executor.map(
                _download_resource,
                [(row, int(config["max_file_bytes"])) for row in unique.values()],
            )
        )
    extractable = [
        row
        for row in downloaded
        if row["status"] in {"cached", "downloaded"} and Path(str(row["path"])).exists()
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(config["download_workers"]) // 2)) as executor:
        extracted = list(
            executor.map(
                _extract_one,
                [(row, bool(config["pdf_table_extraction"])) for row in extractable],
            )
        )
    return {
        "resources": pd.DataFrame.from_records(downloaded),
        "extraction_status": pd.DataFrame.from_records(row for *_, row in extracted),
        "tables": pd.DataFrame.from_records(
            table for _, tables, _, _ in extracted for table in tables
        ),
        "cells": pd.DataFrame.from_records(cell for _, _, cells, _ in extracted for cell in cells),
        "pages": pd.DataFrame.from_records(page for pages, _, _, _ in extracted for page in pages),
    }


def harvest_public_repositories(settings: Settings) -> dict[str, pd.DataFrame]:
    config = settings.section("public_harvest")
    catalog, search, resources = _repository(settings, config)
    harvested = _download_extract(resources, config)
    return {
        "public_repository_catalog": catalog,
        "public_repository_search_status": search,
        "public_source_resources": harvested["resources"],
        "public_extraction_status": harvested["extraction_status"],
        "public_tabular_tables": harvested["tables"],
        "public_tabular_cells": harvested["cells"],
        "public_document_pages": harvested["pages"],
    }
