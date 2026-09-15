from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _pivot(panel: pd.DataFrame) -> pd.DataFrame:
    value = panel.pivot_table(
        index="location_name",
        columns="period_start",
        values="cases",
        aggfunc="sum",
    )
    value = value.sort_index(axis=1)
    value = value.interpolate(axis=1, limit_direction="both")
    return value.fillna(0)


def _dtw(first: np.ndarray, second: np.ndarray) -> float:
    length = max(len(first), len(second))
    radius = max(2, int(np.ceil(length * 0.1)))
    matrix = np.full((len(first) + 1, len(second) + 1), np.inf)
    matrix[0, 0] = 0
    for left in range(1, len(first) + 1):
        start = max(1, left - radius)
        end = min(len(second), left + radius)
        for right in range(start, end + 1):
            cost = abs(first[left - 1] - second[right - 1])
            matrix[left, right] = cost + min(
                matrix[left - 1, right],
                matrix[left, right - 1],
                matrix[left - 1, right - 1],
            )
    return float(matrix[-1, -1])


def _dtw_clusters(
    panel_id: str, pivot: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(pivot) < 4 or len(pivot.columns) < 8 or len(pivot) > 100:
        return [], [
            {"panel_id": panel_id, "method": "dtw_clustering", "status": "insufficient_data"}
        ]
    values = StandardScaler().fit_transform(pivot.to_numpy().T).T
    distances = np.zeros((len(values), len(values)))
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            distance = _dtw(values[left], values[right])
            distances[left, right] = distance
            distances[right, left] = distance
    candidates: list[tuple[float, int, np.ndarray]] = []
    for clusters in range(2, min(6, len(values) - 1) + 1):
        labels = AgglomerativeClustering(
            n_clusters=clusters,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distances)
        score = silhouette_score(distances, labels, metric="precomputed")
        candidates.append((float(score), clusters, labels))
    score, clusters, labels = max(candidates, key=lambda item: item[0])
    memberships = [
        {
            "panel_id": panel_id,
            "location_name": location,
            "cluster": int(labels[index]),
            "clusters": clusters,
            "silhouette": score,
        }
        for index, location in enumerate(pivot.index)
    ]
    return memberships, [{"panel_id": panel_id, "method": "dtw_clustering", "status": "computed"}]


def _interaction(panel_id: str, pivot: pd.DataFrame) -> list[dict[str, Any]]:
    values = np.log1p(pivot.to_numpy(dtype=float))
    grand = np.mean(values)
    total = np.sum((values - grand) ** 2)
    if total <= 0:
        return []
    spatial = values.shape[1] * np.sum((np.mean(values, axis=1) - grand) ** 2)
    temporal = values.shape[0] * np.sum((np.mean(values, axis=0) - grand) ** 2)
    interaction = max(0.0, total - spatial - temporal)
    return [
        {"panel_id": panel_id, "component": "spatial", "variance_fraction": spatial / total},
        {"panel_id": panel_id, "component": "temporal", "variance_fraction": temporal / total},
        {
            "panel_id": panel_id,
            "component": "interaction",
            "variance_fraction": interaction / total,
        },
    ]


def _similarity(panel_id: str, pivot: pd.DataFrame) -> list[dict[str, Any]]:
    correlations = pivot.T.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(correlations.index):
        for right in correlations.index[left_index + 1 :]:
            rows.append(
                {
                    "panel_id": panel_id,
                    "source_location": left,
                    "target_location": right,
                    "spearman_similarity": correlations.loc[left, right],
                }
            )
    return rows


def _phase(panel_id: str, panel: pd.DataFrame) -> list[dict[str, Any]]:
    grain = str(panel["temporal_resolution"].iloc[0])
    if grain not in {"week", "month", "quarter"}:
        return []
    divisor = {"week": 52, "month": 12, "quarter": 4}[grain]
    values = panel.copy()
    if grain == "week":
        values["season_index"] = values["period_start"].dt.isocalendar().week.astype(int)
    elif grain == "month":
        values["season_index"] = values["period_start"].dt.month
    else:
        values["season_index"] = values["period_start"].dt.quarter
    seasonal = values.groupby(["location_name", "season_index"], as_index=False)["cases"].mean()
    peak = seasonal.loc[seasonal.groupby("location_name")["cases"].idxmax()].copy()
    peak["phase_angle"] = 2 * np.pi * (peak["season_index"] - 1) / divisor
    peak["panel_id"] = panel_id
    return peak[["panel_id", "location_name", "season_index", "phase_angle", "cases"]].to_dict(
        "records"
    )


def _emerging(local_spatial: pd.DataFrame) -> pd.DataFrame:
    if local_spatial.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (panel_id, location), group in local_spatial.sort_values("period_start").groupby(
        ["panel_id", "location_name"],
        dropna=False,
    ):
        z = group["getis_ord_z"].astype(float).to_numpy()
        hot = z > 1.96
        cold = z < -1.96
        tau, p_value = kendalltau(np.arange(len(z)), z) if len(z) >= 4 else (np.nan, np.nan)
        if hot.mean() >= 0.9:
            category = "persistent_hotspot"
        elif hot[-1] and hot[:-1].sum() == 0:
            category = "new_hotspot"
        elif hot[-min(3, len(hot)) :].all():
            category = "consecutive_hotspot"
        elif hot.mean() >= 0.5 and p_value < 0.05 and tau > 0:
            category = "intensifying_hotspot"
        elif hot.mean() >= 0.5 and p_value < 0.05 and tau < 0:
            category = "diminishing_hotspot"
        elif hot.any():
            category = "sporadic_hotspot"
        elif cold.mean() >= 0.5:
            category = "coldspot"
        else:
            category = "no_pattern"
        rows.append(
            {
                "panel_id": panel_id,
                "location_name": location,
                "category": category,
                "hot_fraction": hot.mean(),
                "cold_fraction": cold.mean(),
                "trend_tau": tau,
                "trend_p_value": p_value,
                "latest_z": z[-1],
            }
        )
    return pd.DataFrame.from_records(rows)


def _readiness(panel_id: str, panel: pd.DataFrame) -> list[dict[str, Any]]:
    periods = panel["period_start"].nunique()
    locations = panel["location_name"].nunique()
    zero_rate = panel["cases"].eq(0).mean()
    population = panel["population"].notna().mean()
    scores = {
        "forecasting": 100
        * (0.5 * min(periods / 60, 1) + 0.3 * (1 - zero_rate) + 0.2 * min(locations / 10, 1)),
        "transmission_modeling": 100
        * (0.4 * min(periods / 36, 1) + 0.3 * min(locations / 20, 1) + 0.3 * population),
        "risk_stratification": 100
        * (0.5 * min(locations / 30, 1) + 0.3 * min(periods / 12, 1) + 0.2 * population),
        "transfer_learning": 100 * (0.5 * min(locations / 20, 1) + 0.5 * min(periods / 36, 1)),
        "disaggregation": 100
        * (0.5 * min(locations / 50, 1) + 0.3 * population + 0.2 * min(periods / 24, 1)),
    }
    return [
        {
            "panel_id": panel_id,
            "use_case": use_case,
            "readiness_score": score,
            "periods": periods,
            "locations": locations,
            "zero_rate": zero_rate,
            "population_coverage": population,
        }
        for use_case, score in scores.items()
    ]


def diagnose_spatiotemporal(
    multiscale: pd.DataFrame,
    local_spatial: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    clusters: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    similarities: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    readiness: list[dict[str, Any]] = []
    if not multiscale.empty:
        for panel_id, panel in multiscale.groupby("panel_id", dropna=False):
            pivot = _pivot(panel)
            membership, status = _dtw_clusters(str(panel_id), pivot)
            clusters.extend(membership)
            statuses.extend(status)
            interactions.extend(_interaction(str(panel_id), pivot))
            similarities.extend(_similarity(str(panel_id), pivot))
            phases.extend(_phase(str(panel_id), panel))
            readiness.extend(_readiness(str(panel_id), panel))
    return {
        "dtw_clusters": pd.DataFrame.from_records(clusters),
        "method_status": pd.DataFrame.from_records(statuses),
        "variance_decomposition": pd.DataFrame.from_records(interactions),
        "transfer_similarity": pd.DataFrame.from_records(similarities),
        "seasonal_phase": pd.DataFrame.from_records(phases),
        "emerging_hotspots": _emerging(local_spatial),
        "modeling_readiness": pd.DataFrame.from_records(readiness),
    }
