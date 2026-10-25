"""Predictive samples from strictly out-of-sample historical residuals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class InsufficientResidualEvidence(ValueError):
    pass


@dataclass
class HorizonResidualSampler:
    """Sample horizon-specific backtest residuals around point forecasts.

    This is labelled ``empirical_oos_residual`` rather than calibrated: finite
    rolling-origin residuals provide an uncertainty approximation but do not,
    by themselves, prove calibration under temporal dependence.
    """

    n_samples: int = 100
    min_residuals_per_horizon: int = 20
    seed: int = 20261025
    residuals_: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    evaluated_through_: pd.Timestamp | None = field(default=None, init=False)

    def fit(self, backtest_results: pd.DataFrame) -> HorizonResidualSampler:
        required = {
            "actual",
            "prediction",
            "horizon",
            "forecast_origin",
            "is_out_of_sample",
        }
        missing = required.difference(backtest_results.columns)
        if missing:
            raise ValueError(f"backtest results missing columns: {sorted(missing)}")
        if not backtest_results["is_out_of_sample"].eq(True).all():  # noqa: E712
            raise ValueError("residual sampling requires exclusively out-of-sample predictions")
        pools: dict[int, np.ndarray] = {}
        for horizon, group in backtest_results.groupby("horizon", sort=True):
            horizon_number = int(str(horizon))
            actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
            predicted = pd.to_numeric(group["prediction"], errors="coerce").to_numpy(dtype=float)
            residual = actual - predicted
            residual = residual[np.isfinite(residual)]
            if len(residual) < self.min_residuals_per_horizon:
                raise InsufficientResidualEvidence(
                    f"horizon {horizon_number} has {len(residual)} residuals; "
                    f"at least {self.min_residuals_per_horizon} required"
                )
            if np.unique(residual).size < 2:
                raise InsufficientResidualEvidence(
                    f"horizon {horizon_number} residual distribution has no variation"
                )
            pools[horizon_number] = residual
        if not pools:
            raise InsufficientResidualEvidence("no usable horizon residuals")
        self.residuals_ = pools
        self.evaluated_through_ = pd.to_datetime(backtest_results["forecast_origin"]).max()
        return self

    def add_samples(self, point_forecasts: pd.DataFrame) -> pd.DataFrame:
        required = {"prediction", "horizon"}
        missing = required.difference(point_forecasts.columns)
        if missing:
            raise ValueError(f"point forecasts missing columns: {sorted(missing)}")
        if not self.residuals_ or self.evaluated_through_ is None:
            raise InsufficientResidualEvidence("residual sampler has not been fitted")
        if self.n_samples < 2:
            raise ValueError("at least two predictive samples are required")
        output = point_forecasts.copy()
        rng = np.random.default_rng(self.seed)
        samples = np.empty((len(output), self.n_samples), dtype=float)
        for row_index, (_, row) in enumerate(output.iterrows()):
            horizon = int(row["horizon"])
            if horizon not in self.residuals_:
                raise InsufficientResidualEvidence(
                    f"no residual distribution for horizon {horizon}"
                )
            sampled = rng.choice(self.residuals_[horizon], size=self.n_samples, replace=True)
            samples[row_index] = np.maximum(float(row["prediction"]) + sampled, 0.0)
        for sample_index in range(self.n_samples):
            output[f"sample_{sample_index}"] = samples[:, sample_index]
        output["probabilistic_method"] = "empirical_oos_residual"
        output["probabilistic_calibrated"] = False
        output["residuals_evaluated_through"] = self.evaluated_through_
        return output


@dataclass
class ResidualSampleForecaster:
    point_forecaster: object
    sampler: HorizonResidualSampler

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.point_forecaster, "predict"):
            raise TypeError("point_forecaster must implement predict")
        return self.sampler.add_samples(self.point_forecaster.predict(history))
