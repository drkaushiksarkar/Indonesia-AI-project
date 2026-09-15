from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings


def _qualifiers(sequence: ET.Element) -> dict[str, str]:
    values: dict[str, list[str]] = {}
    for qualifier in sequence.findall(".//GBQualifier"):
        name = qualifier.findtext("GBQualifier_name") or ""
        value = qualifier.findtext("GBQualifier_value") or ""
        if name and value:
            values.setdefault(name, []).append(value)
    return {name: "; ".join(dict.fromkeys(items)) for name, items in values.items()}


def _sequences(content: bytes, taxon: dict[str, str]) -> list[dict[str, Any]]:
    root = ET.fromstring(content)
    rows: list[dict[str, Any]] = []
    for sequence in root.findall(".//GBSeq"):
        qualifiers = _qualifiers(sequence)
        location = qualifiers.get("geo_loc_name") or qualifiers.get("country") or ""
        rows.append(
            {
                "source": "NCBI Nucleotide",
                "disease": taxon["disease"],
                "pathogen": taxon["pathogen"],
                "classification": taxon["classification"],
                "accession": sequence.findtext("GBSeq_accession-version")
                or sequence.findtext("GBSeq_primary-accession"),
                "organism": sequence.findtext("GBSeq_organism"),
                "collection_date": qualifiers.get("collection_date"),
                "geo_location": location,
                "host": qualifiers.get("host"),
                "isolation_source": qualifiers.get("isolation_source"),
                "isolate": qualifiers.get("isolate"),
                "strain": qualifiers.get("strain"),
                "serotype": qualifiers.get("serotype") or taxon.get("serotype"),
                "segment": qualifiers.get("segment"),
                "molecule_type": sequence.findtext("GBSeq_moltype"),
                "sequence_length": sequence.findtext("GBSeq_length"),
                "record_created": sequence.findtext("GBSeq_create-date"),
                "record_updated": sequence.findtext("GBSeq_update-date"),
            }
        )
    return rows


def harvest_ncbi(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = settings.section("ncbi")
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    if not enabled:
        return pd.DataFrame(), pd.DataFrame()
    base = config["base_url"].rstrip("/")
    batch_size = int(config["batch_size"])
    delay = float(config["request_delay_seconds"])
    for taxon in config["taxa"]:
        status = "completed"
        error = ""
        expected = 0
        harvested = 0
        try:
            search = session.get(
                f"{base}/esearch.fcgi",
                params={
                    "db": "nuccore",
                    "term": taxon["query"],
                    "retmode": "json",
                    "retmax": 0,
                    "usehistory": "y",
                },
                timeout=120,
            )
            search.raise_for_status()
            result = search.json()["esearchresult"]
            expected = int(result["count"])
            for start in range(0, expected, batch_size):
                time.sleep(delay)
                response = session.get(
                    f"{base}/efetch.fcgi",
                    params={
                        "db": "nuccore",
                        "query_key": result["querykey"],
                        "WebEnv": result["webenv"],
                        "retstart": start,
                        "retmax": batch_size,
                        "retmode": "xml",
                    },
                    timeout=180,
                )
                response.raise_for_status()
                batch = _sequences(response.content, taxon)
                rows.extend(batch)
                harvested += len(batch)
        except (requests.RequestException, ValueError, KeyError, ET.ParseError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        statuses.append(
            {
                "source": "NCBI Nucleotide",
                "pathogen": taxon["pathogen"],
                "classification": taxon["classification"],
                "query": taxon["query"],
                "url": f"{base}/esearch.fcgi",
                "access_class": "open_api",
                "status": status,
                "expected_records": expected,
                "records": harvested,
                "error": error,
            }
        )
        time.sleep(delay)
    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(["accession", "classification"]).reset_index(drop=True)
        frame["sequence_length"] = pd.to_numeric(frame["sequence_length"], errors="coerce")
        frame["record_url"] = "https://www.ncbi.nlm.nih.gov/nuccore/" + frame["accession"].fillna(
            ""
        )
    return frame, pd.DataFrame.from_records(statuses)
