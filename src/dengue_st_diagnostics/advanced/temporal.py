from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pywt
import ruptures as rpt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, hilbert, periodogram
from scipy.stats import chi2, kendalltau, kurtosis, linregress, skew
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from dengue_st_diagnostics.advanced.series import regular_series

SEASONAL_PERIODS = {"week": 52, "month": 12, "quarter": 4}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _stat(
    series_id: str,
    panel_id: str,
    method_id: str,
    name: str,
    estimate: Any,
    p_value: Any = None,
    unit: str | None = None,
    status: str = "computed",
    interpretation: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "panel_id": panel_id,
        "method_id": method_id,
        "statistic_name": name,
        "estimate": _finite(estimate),
        "p_value": _finite(p_value),
        "q_value": None,
        "confidence_lower": None,
        "confidence_upper": None,
        "unit": unit,
        "status": status,
        "interpretation": interpretation,
        "parameters": parameters or {},
    }


def _component(
    series_id: str,
    panel_id: str,
    method_id: str,
    name: str,
    value: Any,
    index: int | None = None,
    coordinate_name: str | None = None,
    coordinate_value: Any = None,
    period_start: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "panel_id": panel_id,
        "method_id": method_id,
        "component_name": name,
        "component_index": index,
        "coordinate_name": coordinate_name,
        "coordinate_value": _finite(coordinate_value),
        "period_start": pd.Timestamp(period_start).date() if pd.notna(period_start) else None,
        "location_name": None,
        "value": _finite(value),
        "component_metadata": metadata or {},
    }


def _event(
    series_id: str,
    panel_id: str,
    method_id: str,
    period_start: Any,
    event_type: str,
    score: Any,
    threshold: Any,
    direction: str | None,
    severity: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "panel_id": panel_id,
        "method_id": method_id,
        "period_start": pd.Timestamp(period_start).date(),
        "period_end": pd.Timestamp(period_start).date(),
        "location_name": None,
        "event_type": event_type,
        "score": _finite(score),
        "threshold": _finite(threshold),
        "direction": direction,
        "severity": severity,
        "p_value": None,
        "q_value": None,
        "event_metadata": metadata or {},
    }


def _safe_test(function: Any, values: np.ndarray, position: int, **kwargs: Any) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = function(values, **kwargs)
        return _finite(result[position])
    except (ValueError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError):
        return None


def _sample_entropy(values: np.ndarray, order: int = 2) -> float | None:
    if len(values) < order + 3 or np.std(values) == 0:
        return None
    tolerance = 0.2 * np.std(values)

    def count(length: int) -> float:
        templates = np.array(
            [values[index : index + length] for index in range(len(values) - length + 1)]
        )
        distances = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        return float(np.sum((distances <= tolerance) & (~np.eye(len(templates), dtype=bool))))

    first = count(order)
    second = count(order + 1)
    return float(-np.log(second / first)) if first > 0 and second > 0 else None


def _permutation_entropy(values: np.ndarray, order: int = 3) -> float | None:
    if len(values) < order + 1:
        return None
    patterns = [
        tuple(np.argsort(values[index : index + order], kind="stable"))
        for index in range(len(values) - order + 1)
    ]
    counts = pd.Series(patterns).value_counts(normalize=True).to_numpy(dtype=float)
    entropy = -np.sum(counts * np.log(counts))
    return float(entropy / np.log(math.factorial(order)))


def _hurst(values: np.ndarray) -> float | None:
    maximum = min(20, len(values) // 2)
    lags = np.arange(2, maximum + 1)
    if len(lags) < 4:
        return None
    deviations = np.array([np.std(values[lag:] - values[:-lag]) for lag in lags])
    valid = deviations > 0
    if valid.sum() < 4:
        return None
    slope = np.polyfit(np.log(lags[valid]), np.log(deviations[valid]), 1)[0]
    return float(slope)


def _higuchi(values: np.ndarray) -> float | None:
    maximum = min(12, len(values) // 4)
    if maximum < 3:
        return None
    lengths: list[float] = []
    scales: list[int] = []
    size = len(values)
    for scale in range(1, maximum + 1):
        segments: list[float] = []
        for offset in range(scale):
            count = int(np.floor((size - offset - 1) / scale))
            if count < 2:
                continue
            selected = values[offset : offset + count * scale + 1 : scale]
            distance = np.abs(np.diff(selected)).sum()
            normalized = distance * (size - 1) / (count * scale * scale)
            segments.append(float(normalized))
        if segments and np.mean(segments) > 0:
            lengths.append(float(np.mean(segments)))
            scales.append(scale)
    if len(lengths) < 3:
        return None
    return float(np.polyfit(np.log(1 / np.asarray(scales)), np.log(lengths), 1)[0])


def _recurrence(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(values) < 24 or np.std(values) == 0:
        return None, None, None
    normalized = (values - np.mean(values)) / np.std(values)
    embedded = np.column_stack([normalized[:-2], normalized[1:-1], normalized[2:]])
    distances = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=2)
    threshold = np.quantile(distances[np.triu_indices_from(distances, k=1)], 0.1)
    recurrence = distances <= threshold
    np.fill_diagonal(recurrence, False)
    rate = float(recurrence.mean())
    diagonal_lengths: list[int] = []
    vertical_lengths: list[int] = []
    for offset in range(-len(recurrence) + 1, len(recurrence)):
        sequence = np.diag(recurrence, k=offset).astype(int)
        padded = np.pad(sequence, (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        diagonal_lengths.extend(int(value) for value in ends - starts if value >= 2)
    for column in recurrence.T:
        padded = np.pad(column.astype(int), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        vertical_lengths.extend(int(value) for value in ends - starts if value >= 2)
    recurrent_points = recurrence.sum()
    determinism = sum(diagonal_lengths) / recurrent_points if recurrent_points else None
    laminarity = sum(vertical_lengths) / recurrent_points if recurrent_points else None
    return rate, _finite(determinism), _finite(laminarity)


def _emd(
    values: np.ndarray, maximum_imfs: int = 6, maximum_sifts: int = 30
) -> tuple[list[np.ndarray], np.ndarray]:
    residue = values.astype(float).copy()
    imfs: list[np.ndarray] = []
    positions = np.arange(len(values))
    for _ in range(maximum_imfs):
        if len(find_peaks(residue)[0]) + len(find_peaks(-residue)[0]) < 3:
            break
        candidate = residue.copy()
        for _ in range(maximum_sifts):
            maxima = np.unique(
                np.concatenate(([0], find_peaks(candidate)[0], [len(candidate) - 1]))
            )
            minima = np.unique(
                np.concatenate(([0], find_peaks(-candidate)[0], [len(candidate) - 1]))
            )
            if len(maxima) < 3 or len(minima) < 3:
                break
            upper = CubicSpline(maxima, candidate[maxima], bc_type="natural")(positions)
            lower = CubicSpline(minima, candidate[minima], bc_type="natural")(positions)
            updated = candidate - (upper + lower) / 2
            denominator = np.sum(candidate**2)
            change = np.sum((candidate - updated) ** 2) / denominator if denominator > 0 else 0
            candidate = updated
            if change < 0.05:
                break
        imfs.append(candidate)
        residue = residue - candidate
    return imfs, residue


def _profile(
    series_id: str,
    panel_id: str,
    values: np.ndarray,
    expected: int | None,
    completeness: float | None,
    duplicate_periods: int,
    regular: bool,
) -> list[dict[str, Any]]:
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if len(values) > 1 else None
    entries = [
        ("observations", len(values), None),
        ("expected_periods", expected, None),
        ("completeness", completeness, "proportion"),
        ("duplicate_period_records", duplicate_periods, None),
        ("regular_time", int(regular), "boolean"),
        ("mean", mean, None),
        ("median", np.median(values), None),
        ("standard_deviation", np.std(values, ddof=1) if len(values) > 1 else None, None),
        (
            "coefficient_of_variation",
            np.std(values, ddof=1) / mean if mean != 0 and len(values) > 1 else None,
            None,
        ),
        ("zero_rate", np.mean(values == 0), "proportion"),
        (
            "variance_mean_ratio",
            variance / mean if variance is not None and mean > 0 else None,
            None,
        ),
        ("skewness", skew(values, bias=False) if len(values) >= 3 else None, None),
        ("excess_kurtosis", kurtosis(values, bias=False) if len(values) >= 4 else None, None),
    ]
    return [
        _stat(series_id, panel_id, "series_profile", name, value, unit=unit)
        for name, value, unit in entries
    ]


def diagnose_series(
    series: pd.Series,
    group: pd.DataFrame,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    series_id = str(series["series_id"])
    panel_id = str(series["panel_id"])
    raw_values = group.sort_values("period_start")["value"].to_numpy(dtype=float)
    statistics = _profile(
        series_id,
        panel_id,
        raw_values,
        int(series["expected_periods"]) if pd.notna(series["expected_periods"]) else None,
        float(series["completeness"]) if pd.notna(series["completeness"]) else None,
        int(series["duplicate_periods"]),
        bool(series["regular_time"]),
    )
    components: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    plot_data: dict[str, Any] = {}
    if not bool(series["regular_time"]) or int(series["observations"]) < 8:
        statistics.append(
            _stat(
                series_id,
                panel_id,
                "series_profile",
                "advanced_temporal_eligibility",
                0,
                status="insufficient_data",
                interpretation=str(series["eligibility"].get("reason")),
            )
        )
        return statistics, components, events, plot_data
    grain = str(series["temporal_resolution"])
    prepared = regular_series(group, grain)
    values = prepared["analysis_value"].to_numpy(dtype=float)
    dates = prepared["period_start"]
    imputed_fraction = float(prepared["imputed"].mean())
    statistics.append(
        _stat(
            series_id,
            panel_id,
            "series_profile",
            "imputed_fraction_for_diagnostics",
            imputed_fraction,
            unit="proportion",
        )
    )
    trend = linregress(np.arange(len(values)), values)
    tau, tau_p = kendalltau(np.arange(len(values)), values)
    statistics.extend(
        [
            _stat(
                series_id,
                panel_id,
                "series_profile",
                "linear_trend_slope",
                trend.slope,
                p_value=trend.pvalue,
                unit="value_per_period",
            ),
            _stat(series_id, panel_id, "series_profile", "mann_kendall_tau", tau, p_value=tau_p),
            _stat(
                series_id,
                panel_id,
                "adf",
                "p_value",
                _safe_test(adfuller, values, 1),
                p_value=_safe_test(adfuller, values, 1),
            ),
            _stat(
                series_id,
                panel_id,
                "kpss",
                "p_value",
                _safe_test(kpss, values, 1, regression="c", nlags="auto"),
                p_value=_safe_test(kpss, values, 1, regression="c", nlags="auto"),
            ),
        ]
    )
    maximum_lag = min(24, max(1, len(values) // 4))
    if np.std(values) > 0:
        acf_values = acf(values, nlags=maximum_lag, fft=True)
        pacf_lags = min(maximum_lag, len(values) // 2 - 1)
        pacf_values = pacf(values, nlags=max(1, pacf_lags), method="ywm")
        for lag, value in enumerate(acf_values):
            components.append(
                _component(
                    series_id,
                    panel_id,
                    "acf_pacf",
                    "acf",
                    value,
                    index=lag,
                    coordinate_name="lag",
                    coordinate_value=lag,
                )
            )
        for lag, value in enumerate(pacf_values):
            components.append(
                _component(
                    series_id,
                    panel_id,
                    "acf_pacf",
                    "pacf",
                    value,
                    index=lag,
                    coordinate_name="lag",
                    coordinate_value=lag,
                )
            )
        ljung_lag = min(10, max(1, len(values) // 5))
        ljung = acorr_ljungbox(values, lags=[ljung_lag], return_df=True).iloc[0]
        statistics.append(
            _stat(
                series_id,
                panel_id,
                "ljung_box",
                "lb_statistic",
                ljung["lb_stat"],
                p_value=ljung["lb_pvalue"],
                parameters={"lag": ljung_lag},
            )
        )
        if len(values) >= 16:
            arch = het_arch(values - np.mean(values), nlags=min(5, len(values) // 5))
            statistics.append(
                _stat(series_id, panel_id, "arch_lm", "lm_statistic", arch[0], p_value=arch[1])
            )
    seasonal_period = SEASONAL_PERIODS.get(grain)
    residual = values - np.median(values)
    if seasonal_period is not None and len(values) >= 2 * seasonal_period and np.std(values) > 0:
        fit = STL(values, period=seasonal_period, robust=True).fit()
        residual = fit.resid
        remainder_variance = np.var(fit.resid)
        seasonal_strength = max(0.0, 1 - remainder_variance / np.var(fit.resid + fit.seasonal))
        trend_strength = max(0.0, 1 - remainder_variance / np.var(fit.resid + fit.trend))
        statistics.extend(
            [
                _stat(series_id, panel_id, "stl", "seasonal_strength", seasonal_strength),
                _stat(series_id, panel_id, "stl", "trend_strength", trend_strength),
            ]
        )
        for name, array in (
            ("trend", fit.trend),
            ("seasonal", fit.seasonal),
            ("remainder", fit.resid),
        ):
            components.extend(
                _component(series_id, panel_id, "stl", name, value, period_start=date)
                for date, value in zip(dates, array, strict=True)
            )
        plot_data["stl"] = {
            "dates": dates,
            "observed": values,
            "trend": fit.trend,
            "seasonal": fit.seasonal,
            "residual": fit.resid,
        }
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    robust_scores = 0.67448975 * (residual - median) / mad if mad > 0 else np.zeros(len(residual))
    for index in np.flatnonzero(np.abs(robust_scores) >= 3.5):
        events.append(
            _event(
                series_id,
                panel_id,
                "robust_residual_anomaly",
                dates.iloc[index],
                "point_anomaly",
                robust_scores[index],
                3.5,
                "high" if robust_scores[index] > 0 else "low",
                "high" if abs(robust_scores[index]) >= 5 else "medium",
            )
        )
    statistics.append(
        _stat(
            series_id,
            panel_id,
            "robust_residual_anomaly",
            "anomaly_count",
            np.sum(np.abs(robust_scores) >= 3.5),
        )
    )
    features = np.column_stack([values, np.r_[0, np.diff(values)], np.abs(robust_scores)])
    if len(values) >= 12 and np.std(features[:, 0]) > 0:
        centers = np.median(features, axis=0)
        scales = 1.4826 * np.median(np.abs(features - centers), axis=0)
        scales = np.where(scales > 0, scales, 1.0)
        scores = np.sqrt(np.sum(((features - centers) / scales) ** 2, axis=1))
        threshold = float(np.sqrt(chi2.ppf(0.997, df=features.shape[1])))
        predictions = scores >= threshold
        for index in np.flatnonzero(predictions):
            events.append(
                _event(
                    series_id,
                    panel_id,
                    "robust_multivariate_anomaly",
                    dates.iloc[index],
                    "multivariate_point_anomaly",
                    scores[index],
                    threshold,
                    "high" if values[index] >= np.median(values) else "low",
                    "medium",
                )
            )
        statistics.append(
            _stat(
                series_id,
                panel_id,
                "robust_multivariate_anomaly",
                "anomaly_count",
                np.sum(predictions),
                parameters={"threshold_probability": 0.997},
            )
        )
    standardized = (
        (residual - np.mean(residual)) / np.std(residual)
        if np.std(residual) > 0
        else np.zeros(len(residual))
    )
    ewma = pd.Series(standardized).ewm(alpha=0.2, adjust=False).mean().to_numpy()
    for index in np.flatnonzero(np.abs(ewma) >= 2.5):
        events.append(
            _event(
                series_id,
                panel_id,
                "ewma",
                dates.iloc[index],
                "sustained_shift_signal",
                ewma[index],
                2.5,
                "high" if ewma[index] > 0 else "low",
                "medium",
                {"alpha": 0.2},
            )
        )
    statistics.append(
        _stat(
            series_id,
            panel_id,
            "ewma",
            "signal_count",
            np.sum(np.abs(ewma) >= 2.5),
            parameters={"alpha": 0.2, "threshold": 2.5},
        )
    )
    if np.std(values) > 0 and len(values) >= 12:
        penalty = max(np.log(len(values)) * np.var(values), 1e-9)
        changes = (
            rpt.Pelt(model="rbf", min_size=max(3, seasonal_period // 4 if seasonal_period else 3))
            .fit(values)
            .predict(pen=penalty)
        )
        changes = [value for value in changes if value < len(values)]
        for index in changes:
            events.append(
                _event(
                    series_id,
                    panel_id,
                    "pelt",
                    dates.iloc[index],
                    "change_point",
                    None,
                    penalty,
                    None,
                    "medium",
                )
            )
        statistics.append(
            _stat(
                series_id,
                panel_id,
                "pelt",
                "change_point_count",
                len(changes),
                parameters={"penalty": penalty},
            )
        )
    frequencies, power = periodogram(values - np.mean(values))
    valid = frequencies > 0
    if valid.any() and np.sum(power[valid]) > 0:
        probabilities = power[valid] / np.sum(power[valid])
        spectral_entropy = (
            -np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))
            if len(probabilities) > 1
            else 0
        )
        dominant_index = int(np.argmax(power[valid]))
        dominant_frequency = frequencies[valid][dominant_index]
        statistics.extend(
            [
                _stat(
                    series_id,
                    panel_id,
                    "periodogram",
                    "dominant_period",
                    1 / dominant_frequency,
                    unit="periods",
                ),
                _stat(series_id, panel_id, "entropy", "spectral_entropy", spectral_entropy),
            ]
        )
        for frequency, value in zip(frequencies[valid], power[valid], strict=True):
            components.append(
                _component(
                    series_id,
                    panel_id,
                    "periodogram",
                    "power",
                    value,
                    coordinate_name="frequency",
                    coordinate_value=frequency,
                )
            )
    if len(values) >= 24 and np.std(values) > 0:
        scales = np.arange(2, min(128, max(3, len(values) // 2)) + 1)
        coefficients, wavelet_frequencies = pywt.cwt(
            values - np.mean(values), scales, "cmor1.5-1.0"
        )
        scale_power = np.mean(np.abs(coefficients) ** 2, axis=1)
        for scale, frequency, value in zip(scales, wavelet_frequencies, scale_power, strict=True):
            components.append(
                _component(
                    series_id,
                    panel_id,
                    "wavelet",
                    "mean_power",
                    value,
                    index=int(scale),
                    coordinate_name="scale",
                    coordinate_value=scale,
                    metadata={"frequency": float(frequency)},
                )
            )
        dominant = int(np.argmax(scale_power))
        statistics.append(
            _stat(
                series_id, panel_id, "wavelet", "dominant_scale", scales[dominant], unit="periods"
            )
        )
        plot_data["wavelet"] = {
            "dates": dates,
            "scales": scales,
            "power": np.abs(coefficients) ** 2,
        }
        imfs, residue_values = _emd(values)
        total_energy = float(np.sum(values**2))
        for imf_index, imf in enumerate(imfs, start=1):
            analytic = hilbert(imf)
            phase = np.unwrap(np.angle(analytic))
            frequency = np.diff(phase) / (2 * np.pi)
            energy = float(np.sum(imf**2))
            statistics.extend(
                [
                    _stat(
                        series_id,
                        panel_id,
                        "emd_hht",
                        f"imf_{imf_index}_energy_fraction",
                        energy / total_energy if total_energy > 0 else None,
                    ),
                    _stat(
                        series_id,
                        panel_id,
                        "emd_hht",
                        f"imf_{imf_index}_median_instantaneous_frequency",
                        np.median(frequency) if len(frequency) else None,
                        unit="cycles_per_period",
                    ),
                ]
            )
            components.extend(
                _component(
                    series_id,
                    panel_id,
                    "emd_hht",
                    f"imf_{imf_index}",
                    value,
                    index=imf_index,
                    period_start=date,
                )
                for date, value in zip(dates, imf, strict=True)
            )
        components.extend(
            _component(
                series_id,
                panel_id,
                "emd_hht",
                "residue",
                value,
                index=len(imfs) + 1,
                period_start=date,
            )
            for date, value in zip(dates, residue_values, strict=True)
        )
        statistics.append(_stat(series_id, panel_id, "emd_hht", "imf_count", len(imfs)))
        plot_data["emd"] = {
            "dates": dates,
            "observed": values,
            "imfs": imfs,
            "residue": residue_values,
        }
    recurrence_rate, determinism, laminarity = _recurrence(values)
    statistics.extend(
        [
            _stat(series_id, panel_id, "hurst", "hurst_exponent", _hurst(values)),
            _stat(series_id, panel_id, "higuchi_fd", "fractal_dimension", _higuchi(values)),
            _stat(series_id, panel_id, "entropy", "sample_entropy", _sample_entropy(values)),
            _stat(
                series_id, panel_id, "entropy", "permutation_entropy", _permutation_entropy(values)
            ),
            _stat(series_id, panel_id, "recurrence", "recurrence_rate", recurrence_rate),
            _stat(series_id, panel_id, "recurrence", "determinism", determinism),
            _stat(series_id, panel_id, "recurrence", "laminarity", laminarity),
        ]
    )
    plot_data["anomaly"] = {
        "dates": dates,
        "observed": values,
        "scores": robust_scores,
        "events": events,
    }
    return statistics, components, events, plot_data
