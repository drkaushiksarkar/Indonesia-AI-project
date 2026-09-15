from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings


def _abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(position, word) for word, values in index.items() for position in values]
    return " ".join(word for _, word in sorted(positions))


def _row(work: dict[str, Any], query: str) -> dict[str, Any]:
    source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    authorships = work.get("authorships") or []
    authors = [
        str((authorship.get("author") or {}).get("display_name") or "")
        for authorship in authorships
    ]
    concepts = [str(topic.get("display_name") or "") for topic in work.get("topics") or []]
    location = work.get("best_oa_location") or work.get("primary_location") or {}
    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("display_name"),
        "publication_date": work.get("publication_date"),
        "publication_year": work.get("publication_year"),
        "journal": source,
        "authors": "; ".join(value for value in authors if value),
        "topics": "; ".join(value for value in concepts if value),
        "cited_by_count": work.get("cited_by_count"),
        "is_open_access": (work.get("open_access") or {}).get("is_oa"),
        "article_url": location.get("landing_page_url") or work.get("doi") or work.get("id"),
        "pdf_url": location.get("pdf_url"),
        "abstract": _abstract(work.get("abstract_inverted_index")),
        "matched_query": query,
    }


def _search(
    session: requests.Session,
    query: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    cursor = "*"
    rows: list[dict[str, Any]] = []
    while cursor:
        params = {
            "search": query,
            "filter": (
                f"from_publication_date:{start_date},to_publication_date:{end_date},type:article"
            ),
            "per-page": 200,
            "cursor": cursor,
            "select": (
                "id,doi,display_name,publication_date,publication_year,primary_location,"
                "best_oa_location,authorships,topics,cited_by_count,open_access,"
                "abstract_inverted_index"
            ),
        }
        response = None
        maximum_attempts = 2
        for attempt in range(maximum_attempts):
            candidate = session.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=120,
            )
            if candidate.status_code not in {429, 500, 502, 503, 504}:
                response = candidate
                break
            if attempt < maximum_attempts - 1:
                retry_after = candidate.headers.get("Retry-After")
                delay = min(2.0, float(retry_after)) if retry_after else 1.0
                time.sleep(delay)
        if response is None:
            response = candidate
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("results") or []
        rows.extend(_row(work, query) for work in batch)
        cursor = (payload.get("meta") or {}).get("next_cursor") if batch else None
        time.sleep(0.15)
    return rows


EUROPE_PMC_QUERIES = {
    "infectious disease emerging hotspot analysis": (
        "TITLE_ABS:(infectious AND disease AND emerging AND hotspot)"
    ),
    "infectious disease spatiotemporal cluster detection comparison": (
        "TITLE_ABS:(infectious AND disease AND spatiotemporal AND cluster)"
    ),
    "spatiotemporal epidemiology diagnostic methods": (
        "TITLE_ABS:(spatiotemporal AND epidemiology AND diagnostic)"
    ),
    "spatial disaggregation disease incidence": (
        'TITLE_ABS:("spatial disaggregation" AND disease)'
    ),
    "transfer learning spatiotemporal epidemiology": (
        'TITLE_ABS:("transfer learning" AND spatiotemporal)'
    ),
    "distributed lag nonlinear dengue": 'TITLE_ABS:("distributed lag nonlinear" AND dengue)',
    "dengue generalized additive spatiotemporal": (
        'TITLE_ABS:(dengue AND "generalized additive" AND spatiotemporal)'
    ),
    "dengue spatial temporal varying coefficient": (
        'TITLE_ABS:(dengue AND coefficient AND (spatiotemporal OR "space-time"))'
    ),
    "dengue change point seasonality diagnostics": (
        'TITLE_ABS:(dengue AND ("change point" OR changepoint OR "regime shift"))'
    ),
}


def _europe_pmc_row(work: dict[str, Any], query: str) -> dict[str, Any]:
    doi = str(work.get("doi") or "")
    urls = (work.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    pdf = next(
        (value.get("url") for value in urls if value.get("documentStyle") == "pdf"),
        None,
    )
    publication_date = str(work.get("firstPublicationDate") or "")
    return {
        "openalex_id": pd.NA,
        "doi": f"https://doi.org/{doi}" if doi else pd.NA,
        "title": work.get("title"),
        "publication_date": publication_date,
        "publication_year": int(publication_date[:4]) if publication_date else pd.NA,
        "journal": work.get("journalTitle"),
        "authors": work.get("authorString"),
        "topics": "",
        "cited_by_count": work.get("citedByCount"),
        "is_open_access": str(work.get("isOpenAccess") or "").casefold() == "y",
        "article_url": (f"https://europepmc.org/article/{work.get('source')}/{work.get('id')}"),
        "pdf_url": pdf,
        "abstract": str(work.get("abstractText") or ""),
        "matched_query": query,
    }


def _search_europe_pmc(
    session: requests.Session,
    query: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    cursor = "*"
    rows: list[dict[str, Any]] = []
    while cursor:
        response = session.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": (
                    f"{EUROPE_PMC_QUERIES.get(query, query)} "
                    f"AND FIRST_PDATE:[{start_date} TO {end_date}]"
                ),
                "pageSize": 1000,
                "cursorMark": cursor,
                "resultType": "core",
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        batch = (payload.get("resultList") or {}).get("result") or []
        rows.extend(_europe_pmc_row(work, query) for work in batch)
        cursor = payload.get("nextCursorMark") if len(batch) == 1000 else None
        time.sleep(0.2)
    return rows


def _existing(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = settings.paths.quantitative
    literature_path = root / "literature_2020_2025.csv"
    status_path = root / "literature_search_status.csv"
    literature = pd.read_csv(literature_path) if literature_path.exists() else pd.DataFrame()
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    return literature, status


def harvest_literature(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = settings.section("literature")
    rows: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    existing, existing_status = _existing(settings)
    if not enabled:
        return existing, existing_status
    for query in config["queries"]:
        completed = (
            not existing_status.empty
            and existing_status.loc[
                existing_status["query"].eq(query),
                "status",
            ]
            .astype("string")
            .str.startswith("completed")
            .any()
        )
        if completed and not existing.empty:
            matched = existing["matched_query"].fillna("").str.contains(query, regex=False)
            found = existing.loc[matched].copy()
            found["matched_query"] = query
            rows.extend(found.to_dict("records"))
            status.append(
                {"query": query, "status": "completed", "records": len(found), "error": ""}
            )
            continue
        error = ""
        count = 0
        value = "completed"
        try:
            found = _search(session, query, config["start_date"], config["end_date"])
            rows.extend(found)
            count = len(found)
        except (requests.RequestException, ValueError, TypeError) as exc:
            try:
                found = _search_europe_pmc(
                    session,
                    query,
                    config["start_date"],
                    config["end_date"],
                )
                rows.extend(found)
                count = len(found)
                value = "completed_fallback_europe_pmc"
                error = f"OpenAlex {type(exc).__name__}: {exc}"
            except (requests.RequestException, ValueError, TypeError, KeyError) as fallback_exc:
                value = "failed"
                error = (
                    f"OpenAlex {type(exc).__name__}: {exc}; "
                    f"Europe PMC {type(fallback_exc).__name__}: {fallback_exc}"
                )
        status.append({"query": query, "status": value, "records": count, "error": error})
    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame["identity"] = frame["doi"].fillna(frame["openalex_id"])
        grouped = frame.groupby("identity", dropna=False, sort=False)
        queries = grouped["matched_query"].agg(lambda values: "; ".join(sorted(set(values))))
        frame = frame.drop_duplicates("identity").set_index("identity")
        frame["matched_query"] = queries
        frame = frame.reset_index(drop=True)
        frame = frame.sort_values(
            ["publication_date", "cited_by_count"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return frame, pd.DataFrame.from_records(status)
