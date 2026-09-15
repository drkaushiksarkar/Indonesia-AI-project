from __future__ import annotations

import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from esda import G_Local, Geary, Moran, Moran_Local
from libpysal.weights import Queen
from scipy.spatial.distance import pdist
from shapely.geometry import shape

PROVINCE_ALIASES = {
    "bangkabelitungislands": "babel",
    "centraljava": "jawatengah",
    "centralkalimantan": "kalimantantengah",
    "centralsulawesi": "sulawesitengah",
    "eastjava": "jawatimur",
    "eastkalimantan": "kalimantantimur",
    "eastnusatenggara": "nusatenggaratimur",
    "jakartaspecialcapitalregion": "dkijakarta",
    "northkalimantan": "kalimantanutara",
    "northmaluku": "malukuutara",
    "northsulawesi": "sulawesiutara",
    "northsumatra": "sumaterautara",
    "riauarchipelago": "kepulauanriau",
    "riauislands": "kepulauanriau",
    "southkalimantan": "kalimantanselatan",
    "southsulawesi": "sulawesiselatan",
    "southsumatra": "sumateraselatan",
    "southeastsulawesi": "sulawesitenggara",
    "specialregionofyogyakarta": "diyogya",
    "westjava": "jawabarat",
    "westkalimantan": "kalimantanbarat",
    "westnusatenggara": "nusatenggarabarat",
    "westpapua": "papuabarat",
    "westsulawesi": "sulawesibarat",
    "westsumatra": "sumaterabarat",
}


def _name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(
        r"\b(provinsi|province|kabupaten|kota|regency|city|district)\b", "", text.casefold()
    )
    normalized = re.sub(r"[^a-z0-9]+", "", text)
    return PROVINCE_ALIASES.get(normalized, normalized)


def load_boundaries(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for level in ("adm1", "adm2"):
        path = root / f"indonesia_{level}.geojson"
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for feature in document.get("features", []):
            name = feature.get("properties", {}).get("shapeName")
            geometry = shape(feature.get("geometry"))
            records.append({"name": name, "key": _name(name), "geometry": geometry})
        result[level] = records
    return result


def _stat(
    panel_id: str,
    method_id: str,
    name: str,
    estimate: Any,
    p_value: Any = None,
    parameters: dict[str, Any] | None = None,
    status: str = "computed",
) -> dict[str, Any]:
    return {
        "series_id": None,
        "panel_id": panel_id,
        "method_id": method_id,
        "statistic_name": name,
        "estimate": float(estimate) if estimate is not None and np.isfinite(estimate) else None,
        "p_value": float(np.clip(p_value, 0, 1))
        if p_value is not None and np.isfinite(p_value)
        else None,
        "q_value": None,
        "confidence_lower": None,
        "confidence_upper": None,
        "unit": None,
        "status": status,
        "interpretation": None,
        "parameters": parameters or {},
    }


def _event(
    panel_id: str,
    method_id: str,
    period: Any,
    location: str,
    event_type: str,
    score: Any,
    p_value: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "series_id": None,
        "panel_id": panel_id,
        "method_id": method_id,
        "period_start": pd.Timestamp(period).date(),
        "period_end": pd.Timestamp(period).date(),
        "location_name": location,
        "event_type": event_type,
        "score": float(score) if np.isfinite(score) else None,
        "threshold": 0.05,
        "direction": None,
        "severity": "high" if p_value <= 0.01 else "medium" if p_value <= 0.05 else "informational",
        "p_value": float(np.clip(p_value, 0, 1)) if np.isfinite(p_value) else None,
        "q_value": None,
        "event_metadata": metadata,
    }


def _semivariogram(geometries: list[Any], values: np.ndarray) -> tuple[float | None, float | None]:
    coordinates = np.array([[geometry.centroid.x, geometry.centroid.y] for geometry in geometries])
    distances = pdist(coordinates)
    semivariance = 0.5 * pdist(values[:, None], metric="sqeuclidean")
    if len(distances) < 3 or np.max(distances) == 0:
        return None, None
    bins = np.linspace(0, np.max(distances), min(11, max(5, len(values))))
    groups = np.digitize(distances, bins)
    means = np.array(
        [
            np.mean(semivariance[groups == index]) if np.any(groups == index) else np.nan
            for index in range(1, len(bins))
        ]
    )
    centers = (bins[:-1] + bins[1:]) / 2
    if not np.isfinite(means).any():
        return None, None
    sill = float(np.nanmax(means))
    reached = np.flatnonzero(means >= 0.95 * sill)
    spatial_range = float(centers[reached[0]]) if len(reached) else float(centers[-1])
    return spatial_range, sill


def diagnose_spatial_panel(
    panel_id: str,
    panel: pd.DataFrame,
    spatial_resolution: str,
    boundaries: dict[str, list[dict[str, Any]]],
    permutations: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    statistics: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    plot_data: dict[str, Any] = {}
    if spatial_resolution not in {"adm1", "adm2"} or spatial_resolution not in boundaries:
        statistics.append(_stat(panel_id, "moran_global", "eligible", 0, status="missing_geometry"))
        return statistics, events, plot_data
    location_column = "admin_1" if spatial_resolution == "adm1" else "admin_2"
    boundary = boundaries[spatial_resolution]
    boundary_by_key = {record["key"]: record for record in boundary}
    valid_periods = 0
    latest: dict[str, Any] | None = None
    for period, current in panel.groupby("period_start", sort=True):
        current = current.loc[current[location_column].notna()].copy()
        current["match_key"] = current[location_column].map(_name)
        current = current.drop_duplicates("match_key", keep=False)
        current = current.loc[current["match_key"].isin(boundary_by_key)]
        if len(current) < 4 or current["value"].nunique() < 2:
            continue
        current = current.sort_values("match_key")
        records = [boundary_by_key[key] for key in current["match_key"]]
        geometries = [record["geometry"] for record in records]
        weights = Queen.from_iterable(geometries, silence_warnings=True)
        weights.transform = "r"
        if weights.s0 <= 0:
            continue
        values = current["value"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            moran = Moran(values, weights, permutations=permutations)
            geary = Geary(values, weights, permutations=permutations)
            local = Moran_Local(values, weights, permutations=permutations, seed=random_seed)
            hotspot = G_Local(
                values,
                weights,
                permutations=permutations,
                transform="B",
                star=True,
                keep_simulations=False,
                n_jobs=1,
                seed=random_seed,
            )
        period_name = pd.Timestamp(period).date().isoformat()
        statistics.extend(
            [
                _stat(
                    panel_id,
                    "moran_global",
                    f"moran_i_{period_name}",
                    moran.I,
                    moran.p_sim,
                    {"period_start": period_name, "locations": len(values)},
                ),
                _stat(
                    panel_id,
                    "geary_c",
                    f"geary_c_{period_name}",
                    geary.C,
                    geary.p_sim,
                    {"period_start": period_name, "locations": len(values)},
                ),
            ]
        )
        spatial_range, sill = _semivariogram(geometries, values)
        statistics.extend(
            [
                _stat(
                    panel_id,
                    "semivariogram",
                    f"range_degrees_{period_name}",
                    spatial_range,
                    parameters={"period_start": period_name},
                ),
                _stat(
                    panel_id,
                    "semivariogram",
                    f"sill_{period_name}",
                    sill,
                    parameters={"period_start": period_name},
                ),
            ]
        )
        for index, row in enumerate(current.itertuples(index=False)):
            local_p = float(np.clip(local.p_sim[index], 0, 1))
            hotspot_p = float(np.clip(hotspot.p_sim[index], 0, 1))
            quadrant = int(local.q[index])
            cluster = {1: "high_high", 2: "low_high", 3: "low_low", 4: "high_low"}.get(
                quadrant, "unclassified"
            )
            if local_p <= 0.1:
                events.append(
                    _event(
                        panel_id,
                        "moran_local",
                        period,
                        str(getattr(row, location_column)),
                        cluster,
                        local.Is[index],
                        local_p,
                        {"quadrant": quadrant, "value": float(row.value)},
                    )
                )
            if hotspot_p <= 0.1:
                direction = "hotspot" if hotspot.Zs[index] > 0 else "coldspot"
                events.append(
                    _event(
                        panel_id,
                        "getis_ord",
                        period,
                        str(getattr(row, location_column)),
                        direction,
                        hotspot.Zs[index],
                        hotspot_p,
                        {"value": float(row.value)},
                    )
                )
        valid_periods += 1
        latest = {
            "period": period,
            "records": records,
            "values": values,
            "names": current[location_column].astype(str).tolist(),
            "local_i": local.Is,
            "local_p": np.clip(local.p_sim, 0, 1),
            "hotspot_z": hotspot.Zs,
            "hotspot_p": np.clip(hotspot.p_sim, 0, 1),
        }
    statistics.append(
        _stat(
            panel_id,
            "moran_global",
            "eligible_periods",
            valid_periods,
            status="computed" if valid_periods else "insufficient_matches",
        )
    )
    if latest is not None:
        plot_data["spatial"] = latest
    return statistics, events, plot_data
