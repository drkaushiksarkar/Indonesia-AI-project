import math

from dengue_st_diagnostics.frontier.metrics import risk_metrics, weighted_interval_score
from dengue_st_diagnostics.frontier.models import (
    adida,
    auto_autoregression,
    negative_binomial_ingarch,
    seasonal_window_average,
)


def test_seasonal_window_average_preserves_phase() -> None:
    values = [float(index % 12) for index in range(48)]
    assert seasonal_window_average(values, 3, 12) == [0.0, 1.0, 2.0]


def test_auto_autoregression_is_nonnegative() -> None:
    values = [float(index % 6) for index in range(40)]
    result = auto_autoregression(values, 4, 12)
    assert len(result) == 4
    assert all(value >= 0 for value in result)


def test_negative_binomial_ingarch_is_finite() -> None:
    values = [0.0, 1.0, 7.0, 2.0, 12.0, 3.0] * 6
    result = negative_binomial_ingarch(values, 3, 12)
    assert len(result) == 3
    assert all(math.isfinite(value) and value >= 0 for value in result)


def test_adida_handles_zero_series() -> None:
    assert adida([0.0] * 24, 2, 12) == [0.0, 0.0]


def test_probabilistic_scores_reward_exact_forecast() -> None:
    assert weighted_interval_score(5.0, 5.0, 5.0, 5.0, 5.0, 5.0) == 0.0
    result = risk_metrics([0, 1, 2], [(0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)])
    assert result["accuracy"] == 1.0
