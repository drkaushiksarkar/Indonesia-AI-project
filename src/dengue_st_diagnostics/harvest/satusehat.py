from __future__ import annotations

import os
from collections import Counter
from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings


def _coding(resource: dict[str, Any]) -> tuple[str, str]:
    codings = resource.get("code", {}).get("coding") or []
    if not codings:
        return "", ""
    value = codings[0]
    return str(value.get("code") or ""), str(value.get("display") or "")


def harvest_satusehat(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = settings.section("satusehat")
    client_id = os.getenv(config["client_id_environment"], "")
    client_secret = os.getenv(config["client_secret_environment"], "")
    if not enabled or not client_id or not client_secret:
        return pd.DataFrame(), pd.DataFrame.from_records(
            [
                {
                    "source": "SATUSEHAT FHIR",
                    "access_class": "credentialed_api",
                    "status": "credentials_required",
                    "records": 0,
                    "error": "",
                }
            ]
        )
    rows: Counter[tuple[str, str, str, str]] = Counter()
    status = "completed"
    error = ""
    try:
        token_response = session.post(
            config["token_url"],
            params={"grant_type": "client_credentials"},
            data={"client_id": client_id, "client_secret": client_secret},
            timeout=60,
        )
        token_response.raise_for_status()
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        for code in config["condition_codes"]:
            url = f"{config['fhir_base_url'].rstrip('/')}/Condition"
            params: dict[str, Any] | None = {
                "code": code,
                "_count": int(config["page_size"]),
            }
            while url:
                response = session.get(url, params=params, headers=headers, timeout=120)
                response.raise_for_status()
                bundle = response.json()
                for entry in bundle.get("entry") or []:
                    resource = entry.get("resource") or {}
                    diagnosis_code, display = _coding(resource)
                    date = str(resource.get("onsetDateTime") or resource.get("recordedDate") or "")
                    month = date[:7] if len(date) >= 7 else "unknown"
                    organization = str(resource.get("asserter", {}).get("reference") or "unknown")
                    rows[(month, organization, diagnosis_code, display)] += 1
                next_links = [
                    link.get("url")
                    for link in bundle.get("link") or []
                    if link.get("relation") == "next"
                ]
                url = str(next_links[0]) if next_links else ""
                params = None
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    frame = pd.DataFrame.from_records(
        [
            {
                "source": "SATUSEHAT FHIR",
                "month": key[0],
                "organization_reference": key[1],
                "diagnosis_code": key[2],
                "diagnosis_display": key[3],
                "condition_count": count,
            }
            for key, count in rows.items()
        ]
    )
    status_frame = pd.DataFrame.from_records(
        [
            {
                "source": "SATUSEHAT FHIR",
                "access_class": "credentialed_api",
                "status": status,
                "records": int(sum(rows.values())),
                "error": error,
            }
        ]
    )
    return frame, status_frame
