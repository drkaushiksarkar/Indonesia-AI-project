from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from esda import G_Local, Geary, Moran, Moran_Local
from libpysal.weights import Queen
from scipy.spatial.distance import pdist


def _name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(
        r"\b(provinsi|province|kabupaten|kota|regency|city|district)\b", "", text.casefold()
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def _boundary_frame(
    level: str,
    boundaries: dict[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame | None:
    frame = boundaries.get(level)
    if frame is None or frame.empty:
        return None
    result = frame.copy()
    if "shapeName" not in result:
        return None
    result["boundary_location_name"] = result["shapeName"].astype("string")
    if level == "adm2" and "adm1" in boundaries:
        parent = boundaries["adm1"][["shapeName", "geometry"]].copy()
        points = result[["shapeName", "geometry"]].copy()
        points.geometry = points.representative_point()
        joined = gpd.sjoin(
            points, parent, how="left", predicate="within", lsuffix="child", rsuffix="parent"
        )
        parent_name = joined["shapeName_parent"].astype("string")
        child_name = joined["shapeName_child"].astype("string")
        result.loc[joined.index, "boundary_location_name"] = parent_name + " | " + child_name
    result["match_key"] = result["boundary_location_name"].map(_name)
    return result.drop_duplicates("match_key")


def _weights(frame: gpd.GeoDataFrame) -> Queen:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = Queen.from_dataframe(frame, use_index=False, silence_warnings=True)
    weights.transform = "r"
    return weights


def _semivariogram_range(frame: gpd.GeoDataFrame, values: np.ndarray) -> float:
    if len(values) < 4 or np.std(values) == 0:
        return np.nan
    projected = frame.to_crs(3857)
    coordinates = np.column_stack(
        [projected.geometry.centroid.x.to_numpy(), projected.geometry.centroid.y.to_numpy()]
    )
    distances = pdist(coordinates) / 1000
    semivariance = 0.5 * pdist(values[:, None], metric="sqeuclidean")
    if not np.isfinite(distances).all() or np.nanmax(distances) == 0:
        return np.nan
    bins = np.linspace(0, np.nanmax(distances), 11)
    indices = np.digitize(distances, bins)
    means = np.array(
        [
            np.nanmean(semivariance[indices == index]) if np.any(indices == index) else np.nan
            for index in range(1, 11)
        ]
    )
    centers = (bins[:-1] + bins[1:]) / 2
    sill = np.nanmax(means)
    reached = np.flatnonzero(means >= 0.95 * sill)
    return float(centers[reached[0]]) if reached.size else float(centers[-1])


def _global_rows(
    panel_id: str,
    period: pd.Timestamp,
    values: np.ndarray,
    weights: Queen,
    frame: gpd.GeoDataFrame,
    permutations: int,
) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moran = Moran(values, weights, permutations=permutations)
        geary = Geary(values, weights, permutations=permutations)
    return [
        {
            "panel_id": panel_id,
            "period_start": period,
            "method": "global_moran",
            "estimate": moran.I,
            "expected": moran.EI,
            "p_value": moran.p_sim,
            "locations": len(values),
            "status": "computed",
        },
        {
            "panel_id": panel_id,
            "period_start": period,
            "method": "geary_c",
            "estimate": geary.C,
            "expected": geary.EC,
            "p_value": geary.p_sim,
            "locations": len(values),
            "status": "computed",
        },
        {
            "panel_id": panel_id,
            "period_start": period,
            "method": "semivariogram",
            "estimate": _semivariogram_range(frame, values),
            "expected": np.nan,
            "p_value": np.nan,
            "locations": len(values),
            "status": "computed",
        },
    ]


def _local_rows(
    panel_id: str,
    period: pd.Timestamp,
    names: pd.Series,
    values: np.ndarray,
    weights: Queen,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local = Moran_Local(
            values,
            weights,
            permutations=permutations,
            seed=seed,
            alternative="two-sided",
        )
        hotspot = G_Local(
            values,
            weights,
            permutations=permutations,
            transform="B",
            star=True,
            keep_simulations=False,
            n_jobs=1,
            seed=seed,
            alternative="two-sided",
        )
    return [
        {
            "panel_id": panel_id,
            "period_start": period,
            "location_name": name,
            "cases": value,
            "local_moran_i": local.Is[index],
            "local_moran_p": local.p_sim[index],
            "local_moran_quadrant": local.q[index],
            "getis_ord_z": hotspot.Zs[index],
            "getis_ord_p": hotspot.p_sim[index],
        }
        for index, (name, value) in enumerate(zip(names, values, strict=True))
    ]


def _scan_score(observed: float, expected: float, total: float) -> float:
    if observed <= expected or expected <= 0 or expected >= total or observed >= total:
        return 0.0
    outside_observed = total - observed
    outside_expected = total - expected
    return float(
        observed * np.log(observed / expected)
        + outside_observed * np.log(outside_observed / outside_expected)
    )


def _maximum_scan(
    values: np.ndarray,
    expected: np.ndarray,
    neighborhoods: list[list[int]],
    maximum_window: int,
) -> tuple[float, int, int, int]:
    total = float(values.sum())
    best = (0.0, 0, 0, 0)
    for center, neighbors in enumerate(neighborhoods):
        observed_series = values[neighbors].sum(axis=0)
        expected_series = expected[neighbors].sum(axis=0)
        observed_cumulative = np.concatenate(([0.0], np.cumsum(observed_series)))
        expected_cumulative = np.concatenate(([0.0], np.cumsum(expected_series)))
        for window in range(1, maximum_window + 1):
            observed_windows = observed_cumulative[window:] - observed_cumulative[:-window]
            expected_windows = expected_cumulative[window:] - expected_cumulative[:-window]
            scores = np.array(
                [
                    _scan_score(observed, expected_value, total)
                    for observed, expected_value in zip(
                        observed_windows,
                        expected_windows,
                        strict=True,
                    )
                ]
            )
            index = int(np.argmax(scores))
            if scores[index] > best[0]:
                best = (float(scores[index]), center, index, window)
    return best


def _space_time_scan(
    panel_id: str,
    panel: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    permutations: int,
    seed: int,
) -> dict[str, Any] | None:
    values = panel.groupby(["match_key", "period_start"], as_index=False)["cases"].sum()
    observed_keys = sorted(set(values["match_key"]) & set(geometry["match_key"]))
    periods = sorted(values["period_start"].dropna().unique())
    if len(observed_keys) < 4 or len(observed_keys) > 100 or len(periods) < 8:
        return None
    selected = geometry.loc[geometry["match_key"].isin(observed_keys)].copy()
    selected = selected.sort_values("match_key").reset_index(drop=True)
    weights = _weights(selected)
    pivot = values.pivot_table(
        index="match_key",
        columns="period_start",
        values="cases",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=selected["match_key"], columns=periods, fill_value=0)
    array = pivot.to_numpy(dtype=float)
    total = array.sum()
    if total <= 0:
        return None
    expected = np.outer(array.sum(axis=1), array.sum(axis=0)) / total
    neighborhoods = [
        sorted({index, *weights.neighbors.get(index, [])}) for index in range(len(selected))
    ]
    maximum_window = min(4, len(periods))
    score, center, start, window = _maximum_scan(
        array,
        expected,
        neighborhoods,
        maximum_window,
    )
    rng = np.random.default_rng(seed)
    simulated = np.array(
        [
            _maximum_scan(
                rng.poisson(expected),
                expected,
                neighborhoods,
                maximum_window,
            )[0]
            for _ in range(permutations)
        ]
    )
    return {
        "panel_id": panel_id,
        "period_start": periods[start],
        "period_end": periods[start + window - 1],
        "method": "space_time_scan_statistic",
        "estimate": score,
        "expected": np.nan,
        "p_value": (1 + np.sum(simulated >= score)) / (permutations + 1),
        "locations": len(observed_keys),
        "center_location": selected.iloc[center]["boundary_location_name"],
        "window_periods": window,
        "status": "computed",
    }


def diagnose_spatial(
    multiscale: pd.DataFrame,
    boundaries: dict[str, gpd.GeoDataFrame],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    if multiscale.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    np.random.seed(seed)
    for panel_id, panel in multiscale.groupby("panel_id", dropna=False):
        level = str(panel["spatial_resolution"].iloc[0])
        if level not in {"admin1", "admin2"}:
            continue
        geometry = _boundary_frame(level.replace("admin", "adm"), boundaries)
        if geometry is None:
            match_rows.append(
                {
                    "panel_id": panel_id,
                    "matched": 0,
                    "observed": panel["location_name"].nunique(),
                    "status": "missing_input",
                }
            )
            continue
        panel = panel.copy()
        panel["match_key"] = panel["location_name"].map(_name)
        observed = int(panel["location_name"].nunique())
        matched = int(
            panel.loc[panel["match_key"].isin(geometry["match_key"]), "location_name"].nunique()
        )
        match_rows.append(
            {"panel_id": panel_id, "matched": matched, "observed": observed, "status": "computed"}
        )
        scan = _space_time_scan(str(panel_id), panel, geometry, permutations, seed)
        if scan is not None:
            global_rows.append(scan)
        for period, current in panel.groupby("period_start"):
            values = current.groupby("match_key", as_index=False).agg(
                cases=("cases", "sum"),
                location_name=("location_name", "first"),
            )
            joined = geometry.merge(values, on="match_key", how="inner")
            joined = joined.loc[joined.geometry.notna() & joined["cases"].notna()].copy()
            if len(joined) < 4 or joined["cases"].nunique() < 2:
                continue
            try:
                weights = _weights(joined)
                if weights.s0 <= 0:
                    continue
                array = joined["cases"].astype(float).to_numpy()
                global_rows.extend(
                    _global_rows(str(panel_id), period, array, weights, joined, permutations)
                )
                local_rows.extend(
                    _local_rows(
                        str(panel_id),
                        period,
                        joined["location_name"],
                        array,
                        weights,
                        permutations,
                        seed,
                    )
                )
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                continue
    return (
        pd.DataFrame.from_records(global_rows),
        pd.DataFrame.from_records(local_rows),
        pd.DataFrame.from_records(match_rows),
    )
