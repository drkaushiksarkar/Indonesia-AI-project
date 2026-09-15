from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings


def _record(value: dict[str, Any], requested_name: str, key: int) -> dict[str, Any]:
    return {
        "source": "GBIF",
        "requested_species": requested_name,
        "taxon_key": key,
        "species": value.get("species") or value.get("scientificName"),
        "gbif_occurrence_key": value.get("key"),
        "event_date": value.get("eventDate"),
        "year": value.get("year"),
        "month": value.get("month"),
        "latitude": value.get("decimalLatitude"),
        "longitude": value.get("decimalLongitude"),
        "coordinate_uncertainty_m": value.get("coordinateUncertaintyInMeters"),
        "state_province": value.get("stateProvince"),
        "county": value.get("county"),
        "locality": value.get("locality"),
        "basis_of_record": value.get("basisOfRecord"),
        "sampling_protocol": value.get("samplingProtocol"),
        "life_stage": value.get("lifeStage"),
        "sex": value.get("sex"),
        "organism_quantity": value.get("organismQuantity"),
        "organism_quantity_type": value.get("organismQuantityType"),
        "individual_count": value.get("individualCount"),
        "occurrence_status": value.get("occurrenceStatus"),
        "dataset_name": value.get("datasetTitle"),
        "publisher": value.get("publishingOrgKey"),
        "license": value.get("license"),
        "issues": "; ".join(value.get("issues") or []),
        "record_url": f"https://www.gbif.org/occurrence/{value.get('key')}",
    }


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for species, group in frame.groupby("requested_species", dropna=False):
        dated = pd.to_datetime(group["event_date"], errors="coerce", utc=True)
        georeferenced = group["latitude"].notna() & group["longitude"].notna()
        rows.append(
            {
                "species": species,
                "occurrences": len(group),
                "georeferenced_records": int(georeferenced.sum()),
                "dated_records": int(dated.notna().sum()),
                "first_event_date": dated.min(),
                "last_event_date": dated.max(),
                "datasets": group["dataset_name"].nunique(),
                "sampling_protocol_records": int(group["sampling_protocol"].notna().sum()),
                "quantity_records": int(group["organism_quantity"].notna().sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def harvest_gbif(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = settings.section("gbif")
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    if not enabled:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = config["base_url"].rstrip("/")
    page_size = int(config["page_size"])
    for name in config["species"]:
        status = "completed"
        error = ""
        expected = 0
        harvested = 0
        key = 0
        try:
            match = session.get(f"{base}/species/match", params={"name": name}, timeout=60)
            match.raise_for_status()
            key = int(match.json()["usageKey"])
            offset = 0
            while True:
                response = session.get(
                    f"{base}/occurrence/search",
                    params={
                        "country": "ID",
                        "taxon_key": key,
                        "limit": page_size,
                        "offset": offset,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                values = payload.get("results") or []
                expected = int(payload.get("count") or 0)
                rows.extend(_record(value, name, key) for value in values)
                harvested += len(values)
                offset += len(values)
                if not values or payload.get("endOfRecords") or offset >= expected:
                    break
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        statuses.append(
            {
                "source": "GBIF",
                "species": name,
                "taxon_key": key,
                "url": f"{base}/occurrence/search",
                "access_class": "open_api",
                "status": status,
                "expected_records": expected,
                "records": harvested,
                "error": error,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("gbif_occurrence_key").reset_index(drop=True)
    return frame, _summary(frame), pd.DataFrame.from_records(statuses)
