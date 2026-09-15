from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import download, safe_name, sha256

FORMATS = {
    ".csv": "csv",
    ".pdf": "pdf",
    ".xls": "xls",
    ".xlsx": "xlsx",
    "csv": "csv",
    "excell": "xlsx",
    "pdf": "pdf",
    "xls": "xls",
    "xlsx": "xlsx",
}


def _packages(session: requests.Session, catalog: str, query: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    start = 0
    while True:
        response = session.get(
            f"{catalog.rstrip('/')}/api/3/action/package_search",
            params={"q": query, "rows": 100, "start": start},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()["result"]
        batch = result.get("results") or []
        values.extend(batch)
        start += len(batch)
        if not batch or start >= int(result.get("count") or start):
            return values


def _format(resource: dict[str, Any]) -> str:
    declared = str(resource.get("format") or "").casefold().strip()
    suffix = Path(urlparse(str(resource.get("url") or "")).path).suffix.casefold()
    return FORMATS.get(suffix) or FORMATS.get(declared) or ""


def _category(value: str) -> str:
    text = value.casefold()
    if any(term in text for term in ["jentik", "abj", "larva", "aedes", "nyamuk"]):
        return "entomology"
    if any(term in text for term in ["malaria", "plasmodium"]):
        return "malaria"
    return "dengue"


def _cache_resource(value: tuple[str, Path]) -> tuple[str, str, str]:
    url, path = value
    if path.exists():
        return url, "cached", ""
    session = requests.Session()
    session.headers.update({"User-Agent": "dengue-st-diagnostics/0.1"})
    try:
        download(session, url, path, timeout=45)
        return url, "downloaded", ""
    except (requests.RequestException, OSError, ValueError) as exc:
        return url, "failed", f"{type(exc).__name__}: {exc}"


def _tabular(
    path: Path,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    if path.suffix.casefold() == ".csv":
        sheets = {"csv": pd.read_csv(path, header=None, dtype="string", sep=None, engine="python")}
    else:
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype="string")
        except ValueError:
            sheets = {
                "delimited": pd.read_csv(
                    path,
                    header=None,
                    dtype="string",
                    sep=None,
                    engine="python",
                    encoding_errors="replace",
                )
            }
    for sheet, frame in sheets.items():
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        tables.append(
            {
                **metadata,
                "sheet": sheet,
                "rows": len(frame),
                "columns": len(frame.columns),
                "nonempty_cells": int(frame.notna().sum().sum()),
            }
        )
        for row_index, row in frame.iterrows():
            for column_index, value in row.items():
                if pd.notna(value):
                    cells.append(
                        {
                            **metadata,
                            "sheet": sheet,
                            "row_number": int(row_index) + 1,
                            "column_number": int(column_index) + 1,
                            "value": str(value),
                        }
                    )
    return cells, tables


def harvest_portals(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = settings.section("surveillance_portals")
    packages: dict[str, dict[str, Any]] = {}
    searches: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    if not enabled:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    for catalog in config["catalogs"]:
        for query in config["queries"]:
            status = "completed"
            error = ""
            count = 0
            try:
                found = _packages(session, catalog, query)
                count = len(found)
                for package in found:
                    key = f"{catalog}|{package.get('id') or package.get('name')}"
                    packages[key] = {"catalog": catalog, **package}
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            searches.append(
                {
                    "catalog": catalog,
                    "query": query,
                    "status": status,
                    "records": count,
                    "error": error,
                }
            )
    candidates: dict[str, Path] = {}
    for package in packages.values():
        for resource in package.get("resources") or []:
            url = str(resource.get("url") or "")
            extension = _format(resource)
            if not extension or not url or url in candidates:
                continue
            resource_id = str(resource.get("id") or resource.get("name") or "resource")
            host = safe_name(urlparse(package["catalog"]).netloc)
            candidates[url] = (
                settings.paths.raw
                / "surveillance_portals"
                / host
                / (safe_name(resource_id) + f".{extension}")
            )
    previous_path = settings.paths.quantitative / "surveillance_portal_resources.parquet"
    previous = pd.read_parquet(previous_path) if previous_path.exists() else pd.DataFrame()
    previous_failures = (
        previous.loc[previous["status"].eq("failed")].set_index("url")["error"].to_dict()
        if not previous.empty and not config["retry_failed_downloads"]
        else {}
    )
    scheduled = {
        url: path
        for url, path in candidates.items()
        if path.exists() or url not in previous_failures
    }
    with ThreadPoolExecutor(max_workers=int(config["download_workers"])) as executor:
        cached = {
            url: (status, error)
            for url, status, error in executor.map(_cache_resource, scheduled.items())
        }
    cached.update(
        {
            url: ("failed", error)
            for url, error in previous_failures.items()
            if url in candidates and not candidates[url].exists()
        }
    )
    seen: set[str] = set()
    for package in packages.values():
        title = str(package.get("title") or "")
        category = _category(title)
        for resource in package.get("resources") or []:
            url = str(resource.get("url") or "")
            extension = _format(resource)
            if not extension or not url or url in seen:
                continue
            seen.add(url)
            resource_id = str(resource.get("id") or resource.get("name") or "resource")
            host = safe_name(urlparse(package["catalog"]).netloc)
            path = (
                settings.paths.raw
                / "surveillance_portals"
                / host
                / (safe_name(resource_id) + f".{extension}")
            )
            status, error = cached.get(url, ("failed", "resource_not_scheduled"))
            try:
                if not path.exists():
                    raise OSError(error)
                if extension in {"csv", "xls", "xlsx"}:
                    metadata = {
                        "catalog": package["catalog"],
                        "category": category,
                        "dataset_title": title,
                        "resource_name": resource.get("name"),
                        "source_url": url,
                        "path": str(path),
                    }
                    resource_cells, resource_tables = _tabular(path, metadata)
                    cells.extend(resource_cells)
                    tables.extend(resource_tables)
                    status = "parsed"
            except (requests.RequestException, OSError, ValueError, TypeError) as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            resources.append(
                {
                    "catalog": package["catalog"],
                    "category": category,
                    "dataset_id": package.get("id") or package.get("name"),
                    "dataset_title": title,
                    "resource_id": resource_id,
                    "resource_name": resource.get("name"),
                    "format": extension,
                    "url": url,
                    "path": str(path),
                    "status": status,
                    "sha256": sha256(path) if path.exists() else "",
                    "error": error,
                }
            )
    return (
        pd.DataFrame.from_records(resources),
        pd.DataFrame.from_records(tables),
        pd.DataFrame.from_records(cells),
        pd.DataFrame.from_records(searches),
    )
