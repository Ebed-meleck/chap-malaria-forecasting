"""Feature generation that never looks beyond the row's forecast origin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from chap_malaria_forecasting.data import ValidationConfig, validate_history


@dataclass(frozen=True)
class FeatureConfig:
    case_lags: tuple[int, ...] = (1, 2, 3, 4, 8, 52)
    case_windows: tuple[int, ...] = (4, 8, 12)
    climate_lags: tuple[int, ...] = tuple(range(1, 13))
    climate_windows: tuple[int, ...] = (4, 8, 12)
    climate_columns: tuple[str, ...] = ("rainfall", "mean_temperature")
    include_cases_at_origin: bool = True


class WeeklyFeatureBuilder:
    """Build numerical features independently within each health zone.

    Rolling summaries are shifted by one week. ``cases_at_origin`` is the one
    observation allowed at the forecast origin and is only used to predict
    strictly later weeks in the direct forecasting pipeline.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.location_mapping_: dict[str, int] | None = None
        self.parent_mapping_: dict[str, int] | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.time_origin_: pd.Timestamp | None = None

    @property
    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def fit(self, history: pd.DataFrame) -> WeeklyFeatureBuilder:
        frame = validate_history(history, ValidationConfig(require_contiguous_weeks=False))
        locations = sorted(frame["location"].dropna().astype(str).unique())
        self.location_mapping_ = {value: index for index, value in enumerate(locations)}
        if "parent" in frame:
            parents = sorted(frame["parent"].dropna().astype(str).unique())
            self.parent_mapping_ = {value: index for index, value in enumerate(parents)}
        else:
            self.parent_mapping_ = {}
        self.time_origin_ = frame["time_period"].min()
        features = self._build(frame)
        self.feature_names_ = tuple(features.columns)
        return self

    def fit_transform(self, history: pd.DataFrame) -> pd.DataFrame:
        self.fit(history)
        return self.transform(history)

    def transform(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.location_mapping_ is None or self.parent_mapping_ is None:
            raise RuntimeError("WeeklyFeatureBuilder must be fitted before transform")
        frame = validate_history(history, ValidationConfig(require_contiguous_weeks=False))
        features = self._build(frame)
        if self.feature_names_ is None:
            self.feature_names_ = tuple(features.columns)
        missing = set(self.feature_names_).difference(features.columns)
        if missing:
            raise ValueError(f"feature schema changed; missing: {sorted(missing)}")
        return features.loc[:, self.feature_names_]

    def _build(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.location_mapping_ is not None
        assert self.parent_mapping_ is not None
        assert self.time_origin_ is not None
        group = frame.groupby("location", sort=False, observed=True)
        cases = frame["disease_cases"].astype(float)
        result = pd.DataFrame(index=frame.index)

        if self.config.include_cases_at_origin:
            result["cases_at_origin"] = cases
            result["cases_at_origin_missing"] = cases.isna().astype(np.int8)
        for lag in self.config.case_lags:
            values = group["disease_cases"].shift(lag).astype(float)
            result[f"cases_lag_{lag}"] = values
            result[f"cases_lag_{lag}_missing"] = values.isna().astype(np.int8)

        shifted_cases = group["disease_cases"].shift(1).astype(float)
        shifted_group = shifted_cases.groupby(frame["location"], sort=False, observed=True)
        for window in self.config.case_windows:
            rolling = shifted_group.rolling(window=window, min_periods=1)
            result[f"cases_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            result[f"cases_std_{window}"] = (
                rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            )

        lag_1 = result["cases_lag_1"]
        lag_4 = result["cases_lag_4"]
        result["cases_trend_1_4"] = lag_1 - lag_4
        denominator = lag_4.abs().where(lag_4.abs() > 0)
        result["cases_relative_change_1_4"] = (lag_1 - lag_4) / denominator

        if "population" in frame:
            population = frame["population"].astype(float)
            result["log_population"] = np.log(population)
            result["incidence_lag_1_per_1000"] = 1_000.0 * lag_1 / population
            result["population_missing"] = population.isna().astype(np.int8)

        for column in self.config.climate_columns:
            if column not in frame:
                continue
            values = frame[column].astype(float)
            climate_group = values.groupby(frame["location"], sort=False, observed=True)
            for lag in self.config.climate_lags:
                lagged = climate_group.shift(lag)
                result[f"{column}_lag_{lag}"] = lagged
                result[f"{column}_lag_{lag}_missing"] = lagged.isna().astype(np.int8)
            shifted = climate_group.shift(1)
            shifted_climate_group = shifted.groupby(frame["location"], sort=False, observed=True)
            for window in self.config.climate_windows:
                result[f"{column}_mean_{window}"] = (
                    shifted_climate_group.rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        iso = frame["time_period"].dt.isocalendar()
        week = iso.week.astype(float)
        result["week_sin"] = np.sin(2.0 * np.pi * week / 52.1775)
        result["week_cos"] = np.cos(2.0 * np.pi * week / 52.1775)
        result["time_index"] = (frame["time_period"] - self.time_origin_).dt.days / 7.0
        result["location_code"] = (
            frame["location"].astype(str).map(self.location_mapping_).fillna(-1).astype(float)
        )
        result["location_unknown"] = (
            ~frame["location"].astype(str).isin(self.location_mapping_)
        ).astype(np.int8)
        if "parent" in frame:
            parents = frame["parent"].astype("string")
            result["parent_code"] = (
                parents.astype(str).map(self.parent_mapping_).fillna(-1).astype(float)
            )
            result["parent_missing"] = parents.isna().astype(np.int8)
            result["parent_unknown"] = (
                parents.notna() & ~parents.astype(str).isin(self.parent_mapping_)
            ).astype(np.int8)
        if "completeness" in frame:
            result["completeness_at_origin"] = frame["completeness"].astype(float)
            result["completeness_missing"] = frame["completeness"].isna().astype(np.int8)

        return result.astype(float)
