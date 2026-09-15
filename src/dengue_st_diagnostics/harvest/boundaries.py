from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.io import download, request_json, sha256


def harvest_boundaries(
    settings: Settings,
    session: requests.Session,
    enabled: bool = True,
) -> tuple[dict[str, gpd.GeoDataFrame], pd.DataFrame]:
    config = settings.section("boundaries")
    frames: dict[str, gpd.GeoDataFrame] = {}
    records: list[dict[str, Any]] = []
    for level in config["administrative_levels"]:
        api_url = config["api_template"].format(level=level)
        path = settings.paths.raw / "boundaries" / f"indonesia_{level.casefold()}.geojson"
        download_url = ""
        status = "cached" if path.exists() else "disabled"
        error = ""
        try:
            if enabled and not path.exists():
                metadata = request_json(session, api_url)
                download_url = str(metadata["gjDownloadURL"])
                download(session, download_url, path)
                status = "downloaded"
            if path.exists():
                frames[level.casefold()] = gpd.read_file(path).to_crs(4326)
                status = "parsed"
        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "source": f"geoboundaries_{level.casefold()}",
                "url": download_url or api_url,
                "path": str(path),
                "status": status,
                "sha256": sha256(path) if path.exists() else "",
                "error": error,
            }
        )
    return frames, pd.DataFrame.from_records(records)
