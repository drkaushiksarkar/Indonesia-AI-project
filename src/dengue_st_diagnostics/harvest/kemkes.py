from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import download, safe_name, sha256


def _rows(html: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    values: list[dict[str, str]] = []
    for row in soup.select("tr"):
        cells = row.find_all("td")
        anchor = row.find("a", href=True)
        if len(cells) < 4 or anchor is None:
            continue
        values.append(
            {
                "publication_date": cells[1].get_text(" ", strip=True),
                "title": cells[2].get_text(" ", strip=True),
                "url": urljoin(base_url, str(anchor["href"])),
            }
        )
    return values


def _relevant(title: str, url: str) -> bool:
    text = f"{title} {url}".casefold()
    terms = ["dengue", "dbd", "demam berdarah", "penyakit vektor", "laporan mingguan situasi"]
    return any(term in text for term in terms) and url.casefold().endswith(".pdf")


def _pdf_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages), len(pages)


def harvest_kemkes(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = settings.section("kemkes")
    documents: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    if not enabled:
        return pd.DataFrame(), pd.DataFrame()
    try:
        response = session.get(config["publication_url"], timeout=120)
        response.raise_for_status()
        rows = _rows(response.text, config["publication_url"])
    except requests.RequestException as exc:
        return pd.DataFrame(), pd.DataFrame.from_records(
            [
                {
                    "source": "kemkes_publications",
                    "url": config["publication_url"],
                    "path": "",
                    "status": "failed",
                    "sha256": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        )
    existing_path = settings.paths.quantitative / "kemkes_document_text.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    existing_by_url = (
        existing.drop_duplicates("url").set_index("url").to_dict("index")
        if not existing.empty
        else {}
    )
    for row in rows:
        if not _relevant(row["title"], row["url"]):
            continue
        name = safe_name(Path(row["url"]).name)
        path = settings.paths.raw / "kemkes" / name
        status = "cached" if path.exists() else "pending"
        error = ""
        text = ""
        pages = 0
        try:
            if not path.exists():
                download(session, row["url"], path)
                status = "downloaded"
            cached = existing_by_url.get(row["url"])
            if cached is not None:
                text = str(cached.get("text") or "")
                pages = int(cached.get("pages") or 0)
                status = "parsed_cached"
            else:
                text, pages = _pdf_text(path)
                status = "parsed"
        except (requests.RequestException, OSError, ValueError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        documents.append({**row, "path": str(path), "pages": pages, "text": text})
        sources.append(
            {
                "source": "kemkes_publications",
                "url": row["url"],
                "path": str(path),
                "status": status,
                "sha256": sha256(path) if path.exists() else "",
                "error": error,
            }
        )
    return pd.DataFrame.from_records(documents), pd.DataFrame.from_records(sources)
