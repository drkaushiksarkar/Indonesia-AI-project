from __future__ import annotations

from typing import Any

import pandas as pd


def surveillance_quality(
    who: pd.DataFrame,
    sequences: pd.DataFrame,
    vectors: pd.DataFrame,
    portal_resources: pd.DataFrame,
    skdr_weekly: pd.DataFrame,
    malaria_species: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(dataset: str, dimension: str, total: int, usable: int, issue: str) -> None:
        rows.append(
            {
                "dataset": dataset,
                "dimension": dimension,
                "records": total,
                "usable_records": usable,
                "usable_proportion": usable / total if total else pd.NA,
                "issue": issue,
            }
        )

    who_usable = int(who["value"].notna().sum()) if "value" in who else 0
    add("WHO malaria", "numeric_value", len(who), who_usable, "")
    collected = (
        sequences["collection_date"].fillna("").astype(str).str.strip().ne("")
        if "collection_date" in sequences
        else pd.Series(dtype=bool)
    )
    located = (
        sequences["geo_location"].fillna("").astype(str).str.strip().ne("")
        if "geo_location" in sequences
        else pd.Series(dtype=bool)
    )
    add(
        "NCBI pathogen sequences",
        "collection_date",
        len(sequences),
        int(collected.sum()),
        "submission sampling is not incidence",
    )
    add(
        "NCBI pathogen sequences",
        "geo_location",
        len(sequences),
        int(located.sum()),
        "location precision varies by submitter",
    )
    georeferenced = (
        vectors["latitude"].notna() & vectors["longitude"].notna()
        if {"latitude", "longitude"}.issubset(vectors)
        else pd.Series(dtype=bool)
    )
    dated = (
        vectors["event_date"].fillna("").astype(str).str.strip().ne("")
        if "event_date" in vectors
        else pd.Series(dtype=bool)
    )
    add(
        "GBIF vector occurrences",
        "coordinates",
        len(vectors),
        int(georeferenced.sum()),
        "occurrence is not vector density",
    )
    add(
        "GBIF vector occurrences",
        "event_date",
        len(vectors),
        int(dated.sum()),
        "collection effort is heterogeneous",
    )
    parsed = (
        portal_resources["status"].isin(["parsed", "downloaded", "cached"])
        if "status" in portal_resources
        else pd.Series(dtype=bool)
    )
    add(
        "Indonesian surveillance portals",
        "retrieval_or_parse",
        len(portal_resources),
        int(parsed.sum()),
        "stale and protected resource URLs remain cataloged",
    )
    duplicate_reports = (
        skdr_weekly.duplicated(["indicator", "cumulative_reports"], keep=False)
        if {"indicator", "cumulative_reports"}.issubset(skdr_weekly)
        else pd.Series(dtype=bool)
    )
    add(
        "SKDR public weekly reports",
        "unique_cumulative_observation",
        len(skdr_weekly),
        int((~duplicate_reports).sum()),
        "reports are cumulative snapshots and some editions repeat prior data",
    )
    conflict_count = (
        int(malaria_species.duplicated(["species", "period_start"], keep=False).sum())
        if {"species", "period_start"}.issubset(malaria_species)
        else 0
    )
    add(
        "Central Java malaria species",
        "nonconflicting_species_period",
        len(malaria_species),
        len(malaria_species) - conflict_count,
        "two official Q1 2026 datasets disagree and must not be summed",
    )
    return pd.DataFrame.from_records(rows)
