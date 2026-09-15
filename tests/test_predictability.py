import math

from dengue_st_diagnostics.predictability.metrics import (
    auc_pr,
    auc_roc,
    bootstrap_skill_interval,
    permutation_entropy,
)
from dengue_st_diagnostics.predictability.models import (
    autoregression_ridge,
    croston_sba,
    seasonal_naive,
)


def test_seasonal_naive_uses_native_cycle() -> None:
    values = [float(value) for value in range(1, 25)]
    assert seasonal_naive(values, 3, 12) == [13.0, 14.0, 15.0]


def test_autoregression_returns_nonnegative_horizon() -> None:
    values = [float(value % 12) for value in range(48)]
    result = autoregression_ridge(values, 6, 12)
    assert len(result) == 6
    assert all(value >= 0 for value in result)


def test_croston_handles_all_zero_series() -> None:
    assert croston_sba([0.0] * 20, 3, 12) == [0.0, 0.0, 0.0]


def test_binary_ranking_metrics_are_perfect() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auc_roc(labels, scores) == 1.0
    assert auc_pr(labels, scores) == 1.0


def test_entropy_and_skill_interval_are_bounded() -> None:
    entropy = permutation_entropy([0.0, 1.0] * 20, 3)
    estimate, lower, upper = bootstrap_skill_interval([1.0] * 10, [2.0] * 10, 7)
    assert 0 <= entropy <= 1
    assert math.isclose(estimate, 0.5)
    assert lower <= estimate <= upper
