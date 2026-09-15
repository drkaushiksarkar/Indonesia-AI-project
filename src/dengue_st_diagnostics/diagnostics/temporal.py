from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pywt
import ruptures as rpt
from scipy.signal import periodogram
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

SEASONAL_PERIODS = {"week": 52, "month": 12, "quarter": 4}


def _metric(
    panel_id: str,
    location: str,
    method: str,
    metric: str,
    value: float | None,
    status: str = "computed",
) -> dict[str, object]:
    return {
        "panel_id": panel_id,
        "location_name": location,
        "method": method,
        "metric": metric,
        "value": value,
        "status": status,
    }


def _safe_test(
    function: object,
    values: np.ndarray,
    index: int,
) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = function(values)
        return float(result[index])
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return np.nan


def _seasonality(values: np.ndarray, period: int) -> tuple[float, float]:
    if len(values) < 2 * period:
        return np.nan, np.nan
    result = STL(values, period=period, robust=True).fit()
    remainder = np.var(result.resid)
    seasonal = max(0.0, 1.0 - remainder / np.var(result.resid + result.seasonal))
    trend = max(0.0, 1.0 - remainder / np.var(result.resid + result.trend))
    return float(seasonal), float(trend)


def _dominant_period(values: np.ndarray) -> float:
    frequencies, power = periodogram(values - np.mean(values))
    valid = frequencies > 0
    if not valid.any() or np.nanmax(power[valid]) <= 0:
        return np.nan
    frequency = frequencies[valid][np.argmax(power[valid])]
    return float(1.0 / frequency)


def _wavelet(values: np.ndarray) -> tuple[float, float]:
    if np.std(values) == 0:
        return np.nan, np.nan
    maximum = min(128, max(3, len(values) // 2))
    scales = np.arange(2, maximum + 1)
    coefficients, _ = pywt.cwt(values - np.mean(values), scales, "cmor1.5-1.0")
    power = np.abs(coefficients) ** 2
    scale_power = power.mean(axis=1)
    selected = int(np.argmax(scale_power))
    average_power = power.mean()
    concentration = scale_power[selected] / average_power if average_power > 0 else np.nan
    return float(scales[selected]), float(concentration)


def _changes(values: np.ndarray) -> list[int]:
    if len(values) < 8 or np.std(values) == 0:
        return []
    penalty = max(np.log(len(values)) * np.var(values), 1e-9)
    detected = rpt.Pelt(model="rbf", min_size=3).fit(values).predict(pen=penalty)
    return [int(value) for value in detected if value < len(values)]


def _series_metrics(
    panel_id: str,
    location: str,
    grain: str,
    values: np.ndarray,
    minimum_periods: int,
) -> list[dict[str, object]]:
    if len(values) < minimum_periods:
        return [
            _metric(
                panel_id,
                location,
                "temporal_profile",
                "observations",
                len(values),
                "insufficient_data",
            )
        ]
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if len(values) > 1 else np.nan
    lag_one = float(acf(values, nlags=1, fft=False)[1]) if np.std(values) > 0 else np.nan
    ljung_lag = max(1, min(10, len(values) // 5))
    ljung = np.nan
    if np.std(values) > 0:
        ljung = float(acorr_ljungbox(values, lags=[ljung_lag], return_df=True)["lb_pvalue"].iloc[0])
    seasonal_period = SEASONAL_PERIODS.get(grain)
    seasonal_strength, trend_strength = (
        _seasonality(values, seasonal_period) if seasonal_period is not None else (np.nan, np.nan)
    )
    wavelet_scale, wavelet_concentration = _wavelet(values)
    changes = _changes(values)
    return [
        _metric(panel_id, location, "temporal_profile", "observations", len(values)),
        _metric(panel_id, location, "temporal_profile", "mean", mean),
        _metric(panel_id, location, "temporal_profile", "variance", variance),
        _metric(panel_id, location, "temporal_profile", "zero_rate", float(np.mean(values == 0))),
        _metric(
            panel_id,
            location,
            "temporal_profile",
            "variance_mean_ratio",
            variance / mean if mean > 0 else np.nan,
        ),
        _metric(panel_id, location, "stationarity_adf", "p_value", _safe_test(adfuller, values, 1)),
        _metric(panel_id, location, "stationarity_kpss", "p_value", _safe_test(kpss, values, 1)),
        _metric(panel_id, location, "autocorrelation", "lag_1", lag_one),
        _metric(panel_id, location, "ljung_box", "p_value", ljung),
        _metric(panel_id, location, "stl", "seasonal_strength", seasonal_strength),
        _metric(panel_id, location, "stl", "trend_strength", trend_strength),
        _metric(panel_id, location, "periodogram", "dominant_period", _dominant_period(values)),
        _metric(panel_id, location, "wavelet", "dominant_scale", wavelet_scale),
        _metric(panel_id, location, "wavelet", "power_concentration", wavelet_concentration),
        _metric(panel_id, location, "pelt", "change_point_count", len(changes)),
        _metric(
            panel_id, location, "pelt", "last_change_index", changes[-1] if changes else np.nan
        ),
    ]


def diagnose_temporal(multiscale: pd.DataFrame, minimum_periods: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if multiscale.empty:
        return pd.DataFrame()
    groups = multiscale.sort_values("period_start").groupby(
        ["panel_id", "location_name"],
        dropna=False,
    )
    for (panel_id, location), group in groups:
        grain = str(group["temporal_resolution"].iloc[0])
        values = group["cases"].astype(float).to_numpy()
        records.extend(
            _series_metrics(str(panel_id), str(location), grain, values, minimum_periods)
        )
    return pd.DataFrame.from_records(records)
