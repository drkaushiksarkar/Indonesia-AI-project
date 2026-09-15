from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _stat(
    panel_id: str,
    method_id: str,
    name: str,
    estimate: Any,
    status: str = "computed",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = float(estimate) if estimate is not None and np.isfinite(estimate) else None
    return {
        "series_id": None,
        "panel_id": panel_id,
        "method_id": method_id,
        "statistic_name": name,
        "estimate": value,
        "p_value": None,
        "q_value": None,
        "confidence_lower": None,
        "confidence_upper": None,
        "unit": None,
        "status": status,
        "interpretation": None,
        "parameters": parameters or {},
    }


def _component(
    panel_id: str,
    method_id: str,
    name: str,
    value: Any,
    index: int | None = None,
    coordinate_name: str | None = None,
    coordinate_value: Any = None,
    period_start: Any = None,
    location_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "series_id": None,
        "panel_id": panel_id,
        "method_id": method_id,
        "component_name": name,
        "component_index": index,
        "coordinate_name": coordinate_name,
        "coordinate_value": float(coordinate_value)
        if coordinate_value is not None and np.isfinite(coordinate_value)
        else None,
        "period_start": pd.Timestamp(period_start).date() if pd.notna(period_start) else None,
        "location_name": location_name,
        "value": float(value) if value is not None and np.isfinite(value) else None,
        "component_metadata": metadata or {},
    }


def _event(
    panel_id: str,
    method_id: str,
    event_type: str,
    score: Any,
    location_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "series_id": None,
        "panel_id": panel_id,
        "method_id": method_id,
        "period_start": None,
        "period_end": None,
        "location_name": location_name,
        "event_type": event_type,
        "score": float(score) if np.isfinite(score) else None,
        "threshold": None,
        "direction": None,
        "severity": "informational",
        "p_value": None,
        "q_value": None,
        "event_metadata": metadata,
    }


def _panel_matrix(
    panel: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]] | None:
    eligible = catalog.loc[
        catalog["regular_time"] & catalog["series_id"].isin(panel["series_id"]),
        ["series_id", "location_name", "completeness"],
    ]
    eligible = eligible.loc[eligible["completeness"].fillna(0).ge(0.7)]
    if len(eligible) < 4:
        return None
    selected = panel.loc[panel["series_id"].isin(eligible["series_id"])].copy()
    duplicates = selected.duplicated(["series_id", "period_start"], keep=False)
    selected = selected.loc[~duplicates]
    pivot = selected.pivot(index="series_id", columns="period_start", values="value")
    pivot = pivot.sort_index(axis=1)
    pivot = pivot.interpolate(axis=1, limit_direction="both")
    keep_rows = pivot.notna().mean(axis=1).ge(0.7)
    keep_columns = pivot.notna().mean(axis=0).ge(0.7)
    pivot = pivot.loc[keep_rows, keep_columns]
    pivot = pivot.apply(lambda row: row.fillna(row.median()), axis=1)
    labels = eligible.set_index("series_id")["location_name"].astype(str).to_dict()
    if len(pivot) < 4 or len(pivot.columns) < 5:
        return None
    return pivot, labels


def _transfer_entropy(source: np.ndarray, target: np.ndarray, bins: int = 4) -> float:
    quantiles_source = np.unique(np.quantile(source, np.linspace(0, 1, bins + 1)))
    quantiles_target = np.unique(np.quantile(target, np.linspace(0, 1, bins + 1)))
    if len(quantiles_source) < 3 or len(quantiles_target) < 3:
        return 0.0
    source_codes = np.digitize(source, quantiles_source[1:-1])
    target_codes = np.digitize(target, quantiles_target[1:-1])
    triples = pd.DataFrame(
        {
            "future": target_codes[1:],
            "target_past": target_codes[:-1],
            "source_past": source_codes[:-1],
        }
    )
    joint = triples.value_counts(normalize=True)
    target_transition = triples[["future", "target_past"]].value_counts(normalize=True)
    past_joint = triples[["target_past", "source_past"]].value_counts(normalize=True)
    target_past = triples["target_past"].value_counts(normalize=True)
    value = 0.0
    for key, probability in joint.items():
        future, target_previous, source_previous = key
        numerator = probability * target_past[target_previous]
        denominator = (
            target_transition[future, target_previous]
            * past_joint[target_previous, source_previous]
        )
        if numerator > 0 and denominator > 0:
            value += probability * np.log2(numerator / denominator)
    return float(max(value, 0.0))


def _lead_lag(first: np.ndarray, second: np.ndarray, maximum_lag: int) -> tuple[int, float]:
    candidates: list[tuple[float, int, float]] = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        left = first[max(0, lag) : len(first) + min(0, lag)]
        right = second[max(0, -lag) : len(second) - max(0, lag)]
        if len(left) < 5 or np.std(left) == 0 or np.std(right) == 0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        candidates.append((abs(correlation), lag, correlation))
    if not candidates:
        return 0, np.nan
    _, lag, correlation = max(candidates)
    return lag, correlation


def diagnose_panel(
    panel_id: str,
    panel: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    statistics: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    plot_data: dict[str, Any] = {}
    prepared = _panel_matrix(panel, catalog)
    if prepared is None:
        statistics.append(_stat(panel_id, "hgd", "eligible", 0, status="insufficient_data"))
        return statistics, components, events, plot_data
    pivot, labels = prepared
    array = np.log1p(np.maximum(pivot.to_numpy(dtype=float), 0))
    row_mean = np.mean(array, axis=1, keepdims=True)
    row_scale = np.std(array, axis=1, keepdims=True)
    row_scale = np.where(row_scale > 0, row_scale, 1.0)
    standardized = (array - row_mean) / row_scale
    component_count = min(10, len(standardized), standardized.shape[1])
    left, singular_values, loadings = np.linalg.svd(standardized, full_matrices=False)
    scores = left[:, :component_count] * singular_values[:component_count]
    loadings = loadings[:component_count]
    denominator = max(len(standardized) - 1, 1)
    eigenvalues = singular_values[:component_count] ** 2 / denominator
    total_variance = np.sum(singular_values**2 / denominator)
    explained_variance_ratio = eigenvalues / total_variance
    probabilities = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    effective_rank = float(np.exp(entropy(probabilities))) if probabilities.sum() > 0 else 0.0
    participation = (
        float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        if np.sum(eigenvalues**2) > 0
        else 0.0
    )
    statistics.extend(
        [
            _stat(panel_id, "hgd", "eligible", 1),
            _stat(panel_id, "hgd", "locations", len(pivot)),
            _stat(panel_id, "hgd", "periods", len(pivot.columns)),
            _stat(panel_id, "hgd", "effective_rank", effective_rank),
            _stat(panel_id, "hgd", "participation_ratio", participation),
            _stat(panel_id, "hgd", "first_component_concentration", explained_variance_ratio[0]),
            _stat(panel_id, "pca_biplot", "pc1_explained_variance", explained_variance_ratio[0]),
            _stat(
                panel_id,
                "pca_biplot",
                "pc2_explained_variance",
                explained_variance_ratio[1] if component_count > 1 else None,
            ),
        ]
    )
    for component_index in range(component_count):
        for row_index, series_id in enumerate(pivot.index):
            components.append(
                _component(
                    panel_id,
                    "pca_biplot",
                    f"pc{component_index + 1}_score",
                    scores[row_index, component_index],
                    index=component_index + 1,
                    location_name=labels.get(series_id, series_id),
                    metadata={"series_id": series_id},
                )
            )
        for column_index, period in enumerate(pivot.columns):
            components.append(
                _component(
                    panel_id,
                    "pca_biplot",
                    f"pc{component_index + 1}_loading",
                    loadings[component_index, column_index],
                    index=component_index + 1,
                    period_start=period,
                )
            )
    plot_data["pca"] = {
        "scores": scores[:, :2],
        "labels": [labels.get(value, value) for value in pivot.index],
        "loadings": loadings[:2],
        "periods": pivot.columns,
        "variance": explained_variance_ratio[:2],
    }
    correlations = np.corrcoef(standardized)
    upper = correlations[np.triu_indices_from(correlations, k=1)]
    statistics.extend(
        [
            _stat(panel_id, "panel_synchrony", "median_pairwise_correlation", np.nanmedian(upper)),
            _stat(
                panel_id,
                "panel_synchrony",
                "correlation_interquartile_range",
                np.nanquantile(upper, 0.75) - np.nanquantile(upper, 0.25),
            ),
            _stat(panel_id, "panel_synchrony", "negative_correlation_fraction", np.mean(upper < 0)),
        ]
    )
    pairs: list[tuple[float, int, int, float]] = []
    for left in range(len(pivot)):
        for right in range(left + 1, len(pivot)):
            value = correlations[left, right]
            if np.isfinite(value):
                pairs.append((abs(value), left, right, value))
    maximum_lag = min(12, max(1, len(pivot.columns) // 4))
    for _, left, right, correlation in sorted(pairs, reverse=True)[: min(100, len(pairs))]:
        lag, lag_correlation = _lead_lag(standardized[left], standardized[right], maximum_lag)
        left_name = labels.get(pivot.index[left], pivot.index[left])
        right_name = labels.get(pivot.index[right], pivot.index[right])
        events.append(
            _event(
                panel_id,
                "panel_synchrony",
                "lead_lag_pair",
                abs(lag_correlation),
                f"{left_name} → {right_name}",
                {
                    "source_location": left_name,
                    "target_location": right_name,
                    "contemporaneous_correlation": correlation,
                    "maximum_correlation": lag_correlation,
                    "lag_periods": lag,
                },
            )
        )
        if len(pivot.columns) >= 16:
            forward = _transfer_entropy(standardized[left], standardized[right])
            reverse = _transfer_entropy(standardized[right], standardized[left])
            events.append(
                _event(
                    panel_id,
                    "transfer_entropy",
                    "directional_information_pair",
                    forward - reverse,
                    f"{left_name} → {right_name}",
                    {
                        "source_location": left_name,
                        "target_location": right_name,
                        "forward_bits": forward,
                        "reverse_bits": reverse,
                        "net_bits": forward - reverse,
                    },
                )
            )
    plot_data["correlation"] = {
        "matrix": correlations,
        "labels": [labels.get(value, value) for value in pivot.index],
    }
    snapshots = standardized.T
    if snapshots.shape[0] >= 8 and snapshots.shape[1] >= 4:
        first = snapshots[:-1].T
        second = snapshots[1:].T
        left, singular, right = np.linalg.svd(first, full_matrices=False)
        rank = min(10, int(np.sum(singular > singular[0] * 1e-8)))
        if rank >= 1:
            reduced = left[:, :rank].T @ second @ right[:rank].T @ np.diag(1 / singular[:rank])
            eigenvalues_dmd, eigenvectors = np.linalg.eig(reduced)
            modes = second @ right[:rank].T @ np.diag(1 / singular[:rank]) @ eigenvectors
            amplitudes = np.linalg.norm(modes, axis=0)
            order = np.argsort(amplitudes)[::-1]
            statistics.append(_stat(panel_id, "dynamic_mode_decomposition", "mode_count", rank))
            for mode_index in order[: min(6, rank)]:
                eigenvalue = eigenvalues_dmd[mode_index]
                statistics.extend(
                    [
                        _stat(
                            panel_id,
                            "dynamic_mode_decomposition",
                            f"mode_{mode_index + 1}_growth",
                            np.log(max(abs(eigenvalue), 1e-12)),
                        ),
                        _stat(
                            panel_id,
                            "dynamic_mode_decomposition",
                            f"mode_{mode_index + 1}_frequency",
                            np.angle(eigenvalue) / (2 * np.pi),
                        ),
                    ]
                )
                for location_index, series_id in enumerate(pivot.index):
                    components.append(
                        _component(
                            panel_id,
                            "dynamic_mode_decomposition",
                            f"mode_{mode_index + 1}_real",
                            np.real(modes[location_index, mode_index]),
                            index=int(mode_index + 1),
                            location_name=labels.get(series_id, series_id),
                        )
                    )
            plot_data["dmd"] = {
                "eigenvalues": eigenvalues_dmd,
                "amplitudes": amplitudes,
                "modes": modes,
                "labels": [labels.get(value, value) for value in pivot.index],
            }
    return statistics, components, events, plot_data
