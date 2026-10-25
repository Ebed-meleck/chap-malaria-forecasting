from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chap_malaria_forecasting.evaluation import empirical_crps, interval_coverage, interval_score
from chap_malaria_forecasting.forecasting import (
    HorizonResidualSampler,
    InsufficientResidualEvidence,
)


def _backtest() -> pd.DataFrame:
    rows = []
    for horizon in (1, 2, 3):
        for index in range(30):
            rows.append(
                {
                    "actual": 10 + index % 7,
                    "prediction": 9 + (index * 3) % 8,
                    "horizon": horizon,
                    "forecast_origin": pd.Timestamp("2025-01-01") + pd.Timedelta(weeks=index),
                    "is_out_of_sample": True,
                }
            )
    return pd.DataFrame(rows)


def test_residual_samples_are_reproducible_non_identical_and_labelled_honestly() -> None:
    points = pd.DataFrame(
        {"location": ["A", "A", "A"], "horizon": [1, 2, 3], "prediction": [10, 11, 12]}
    )
    first = HorizonResidualSampler(n_samples=50, seed=9).fit(_backtest()).add_samples(points)
    second = HorizonResidualSampler(n_samples=50, seed=9).fit(_backtest()).add_samples(points)
    sample_columns = [f"sample_{index}" for index in range(50)]
    assert np.array_equal(first[sample_columns].to_numpy(), second[sample_columns].to_numpy())
    assert np.unique(first.loc[0, sample_columns].to_numpy()).size > 1
    assert first["probabilistic_method"].eq("empirical_oos_residual").all()
    assert not first["probabilistic_calibrated"].any()
    actual = np.array([11, 10, 14], dtype=float)
    samples = first[sample_columns].to_numpy(dtype=float)
    assert empirical_crps(actual, samples) >= 0
    lower = np.quantile(samples, 0.05, axis=1)
    upper = np.quantile(samples, 0.95, axis=1)
    assert 0 <= interval_coverage(actual, lower, upper) <= 1
    assert interval_score(actual, lower, upper) >= 0


def test_in_sample_or_insufficient_residuals_are_rejected() -> None:
    invalid = _backtest()
    invalid.loc[0, "is_out_of_sample"] = False
    with pytest.raises(ValueError, match="out-of-sample"):
        HorizonResidualSampler().fit(invalid)
    with pytest.raises(InsufficientResidualEvidence):
        HorizonResidualSampler(min_residuals_per_horizon=50).fit(_backtest())
