from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings


def harvest_who(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = settings.section("who")
    observations: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    if not enabled:
        return pd.DataFrame(), pd.DataFrame()
    for indicator in config["indicators"]:
        code = indicator["code"]
        url = f"{config['base_url'].rstrip('/')}/{code}"
        status = "completed"
        error = ""
        count = 0
        try:
            response = session.get(
                url,
                params={"$filter": "SpatialDim eq 'IDN'", "$format": "json"},
                timeout=120,
            )
            response.raise_for_status()
            values = response.json().get("value") or []
            count = len(values)
            for value in values:
                observations.append(
                    {
                        "source": "WHO Global Health Observatory",
                        "disease": "malaria",
                        "indicator_code": code,
                        "indicator_name": indicator["name"],
                        "year": value.get("TimeDim"),
                        "value": value.get("NumericValue"),
                        "display_value": value.get("Value"),
                        "lower_bound": value.get("Low"),
                        "upper_bound": value.get("High"),
                        "dimension_1_type": value.get("Dim1Type"),
                        "dimension_1": value.get("Dim1"),
                        "dimension_2_type": value.get("Dim2Type"),
                        "dimension_2": value.get("Dim2"),
                        "data_source": value.get("DataSourceDim"),
                        "date_updated": value.get("Date"),
                        "url": url,
                    }
                )
        except (requests.RequestException, ValueError, TypeError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        statuses.append(
            {
                "source": "WHO Global Health Observatory",
                "indicator_code": code,
                "url": url,
                "access_class": "open_api",
                "status": status,
                "records": count,
                "error": error,
            }
        )
    return pd.DataFrame.from_records(observations), pd.DataFrame.from_records(statuses)
