from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import download, safe_name, sha256

SUPPORTED_FORMATS = {"csv", "geojson", "json", "xls", "xlsx", "zip"}


def _packages(
    session: requests.Session,
    catalog: str,
    query: str,
) -> list[dict[str, Any]]:
    start = 0
    rows = 100
    values: list[dict[str, Any]] = []
    while True:
        response = session.get(
            f"{catalog.rstrip('/')}/api/3/action/package_search",
            params={"q": query, "rows": rows, "start": start},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload["result"]
        batch = result.get("results") or []
        values.extend(batch)
        start += len(batch)
        if not batch or start >= int(result.get("count", start)):
            return values


def _extension(resource: dict[str, Any]) -> str:
    declared = str(resource.get("format") or "").casefold().strip(" .")
    if declared in SUPPORTED_FORMATS:
        return declared
    suffix = Path(urlparse(str(resource.get("url") or "")).path).suffix.casefold().strip(".")
    return suffix if suffix in SUPPORTED_FORMATS else ""


def harvest_ckan(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    config = settings.section("ckan")
    packages: dict[str, dict[str, Any]] = {}
    searches: list[dict[str, Any]] = []
    files: list[Path] = []
    resources: list[dict[str, Any]] = []
    if not enabled:
        return files, pd.DataFrame(), pd.DataFrame()
    for catalog in config["catalogs"]:
        for query in config["queries"]:
            error = ""
            status = "completed"
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
    for package in packages.values():
        for resource in package.get("resources") or []:
            extension = _extension(resource)
            if not extension:
                continue
            resource_id = str(resource.get("id") or resource.get("name") or "resource")
            package_id = str(package.get("id") or package.get("name") or "dataset")
            name = safe_name(f"{package_id}_{resource_id}") + f".{extension}"
            path = settings.paths.raw / "ckan" / name
            url = str(resource.get("url") or "")
            status = "cached" if path.exists() else "pending"
            error = ""
            try:
                if not path.exists():
                    download(session, url, path)
                    status = "downloaded"
                files.append(path)
            except (requests.RequestException, OSError, ValueError) as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            resources.append(
                {
                    "catalog": package["catalog"],
                    "dataset_id": package_id,
                    "dataset_title": package.get("title"),
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
    return files, pd.DataFrame.from_records(searches), pd.DataFrame.from_records(resources)
