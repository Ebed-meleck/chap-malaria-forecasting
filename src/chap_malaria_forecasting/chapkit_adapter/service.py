"""CHAPKit 2.x service adapter.

The business library does not import CHAPKit outside this module.  The value
returned by ``on_train`` is a self-contained descriptor: native XGBoost/LightGBM
text files are embedded with SHA-256 checksums, so the CHAPKit database is the
only state to back up.  Heavy computation runs in a worker thread because
CHAPKit awaits the callbacks on the event loop that also serves ``/health``
and the ServiceKit registration keepalive.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog
from chapkit import BaseConfig
from chapkit.api import (
    AssessedStatus,
    MLServiceBuilder,
    MLServiceInfo,
    ModelMetadata,
    PeriodType,
)
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner, ValidationDiagnostic
from geojson_pydantic import FeatureCollection
from pydantic import Field, HttpUrl

from chap_malaria_forecasting.artifacts import pack_forecaster, unpack_forecaster
from chap_malaria_forecasting.data import DataValidationError, normalize_week, validate_history
from chap_malaria_forecasting.evaluation import (
    BacktestConfig,
    RollingOriginBacktester,
    point_metrics,
)
from chap_malaria_forecasting.forecasting import (
    DirectForecaster,
    HorizonResidualSampler,
    InsufficientResidualEvidence,
)
from chap_malaria_forecasting.models.ensemble import (
    AutoSelectionPolicy,
    ChampionChallengerPolicy,
    EnsembleForecaster,
    FallbackForecaster,
    HistoricalScore,
    InsufficientEvidenceError,
    SelectionDecision,
)
from chap_malaria_forecasting.models.lightgbm import LightGBMConfig, LightGBMRegressor
from chap_malaria_forecasting.models.naive import SeasonalNaiveForecaster
from chap_malaria_forecasting.models.xgboost import XGBoostConfig, XGBoostRegressor

log = structlog.get_logger()

SERVICE_ID = "chap-malaria-forecasting"
REPOSITORY_URL = HttpUrl("https://github.com/Ebed-meleck/chap-malaria-forecasting")
MODEL_VERSION = "0.2.0"
DESCRIPTOR_VERSION = 2
SUPPORTED_ENGINES = ("xgboost", "lightgbm")


class MalariaModelConfig(BaseConfig):
    """Options exposed as CHAP configured-model user options."""

    prediction_periods: int = Field(default=3, ge=1, le=3)
    engine: Literal["xgboost", "lightgbm", "auto"] = "auto"
    auto_mode: Literal["select", "ensemble", "champion_challenger"] = "select"
    champion: Literal["xgboost", "lightgbm"] = "xgboost"
    n_estimators: int = Field(default=120, ge=10, le=1_000)
    n_threads: int = Field(default=1, ge=1, le=12)
    random_seed: int = 20261025
    selection_backtest_splits: int = Field(default=3, ge=2, le=7)
    selection_min_train_weeks: int = Field(default=104, ge=26, le=180)
    selection_min_observations: int = Field(default=100, ge=20)
    switch_margin: float = Field(default=0.02, ge=0.0, le=0.5)
    prediction_samples: int = Field(default=100, ge=2, le=1_000)
    minimum_residuals_per_horizon: int = Field(default=20, ge=10)
    seasonal_fallback: bool = True


def _engine_factory(config: MalariaModelConfig, engine: str) -> Callable[[], Any]:
    if engine == "xgboost":
        xgboost_config = XGBoostConfig(
            n_estimators=config.n_estimators,
            n_jobs=config.n_threads,
            random_state=config.random_seed,
        )
        return lambda: XGBoostRegressor(xgboost_config)
    if engine == "lightgbm":
        lightgbm_config = LightGBMConfig(
            n_estimators=config.n_estimators,
            n_jobs=config.n_threads,
            random_state=config.random_seed,
        )
        return lambda: LightGBMRegressor(lightgbm_config)
    raise ValueError(f"unsupported engine: {engine!r}")


def _backtest(
    config: MalariaModelConfig,
    history: pd.DataFrame,
    engine: str,
) -> pd.DataFrame:
    evaluator = RollingOriginBacktester(
        BacktestConfig(
            n_splits=config.selection_backtest_splits,
            horizon=3,
            min_train_weeks=config.selection_min_train_weeks,
        )
    )
    factory = _engine_factory(config, engine)
    return evaluator.run(history, lambda: DirectForecaster(factory))


def _historical_score(engine: str, result: pd.DataFrame) -> HistoricalScore:
    metrics = point_metrics(result["actual"], result["prediction"])
    return HistoricalScore(
        model_name=engine,
        mae=float(metrics["mae"]),
        n_observations=int(metrics["n"]),
        evaluated_through=pd.to_datetime(result["forecast_origin"]).max(),
        scope="global",
    )


def _decision_to_json(decision: Any) -> dict[str, Any]:
    return {
        "policy": decision.policy,
        "selected_models": list(decision.selected_models),
        "weights": list(decision.weights),
        "reason": decision.reason,
        "scores": [
            {
                "model_name": score.model_name,
                "mae": score.mae,
                "n_observations": score.n_observations,
                "evaluated_through": pd.Timestamp(score.evaluated_through).isoformat(),
                "scope": score.scope,
            }
            for score in decision.scores
        ],
    }


def _choose_auto(
    config: MalariaModelConfig,
    scores: list[HistoricalScore],
    origin: pd.Timestamp,
    trained: list[str],
) -> Any:
    if config.auto_mode == "champion_challenger":
        challenger = "lightgbm" if config.champion == "xgboost" else "xgboost"
        decision = ChampionChallengerPolicy(
            champion=config.champion,
            challenger=challenger,
            minimum_observations=config.selection_min_observations,
            promotion_margin=config.switch_margin,
        ).select(scores, origin)
    else:
        mode = "ensemble" if config.auto_mode == "ensemble" else "select"
        try:
            decision = AutoSelectionPolicy(
                mode=mode,
                minimum_observations=config.selection_min_observations,
                incumbent=config.champion,
                switch_margin=config.switch_margin,
            ).select(scores, origin)
        except InsufficientEvidenceError:
            decision = SelectionDecision(
                policy=f"auto_{mode}",
                selected_models=(config.champion,),
                weights=(1.0,),
                reason=(
                    "insufficient historical out-of-sample evidence "
                    f"(< {config.selection_min_observations} observations); champion retained"
                ),
                scores=tuple(scores),
            )
    if decision.selected_models[0] not in trained:
        # The champion itself failed to train: keep the decision auditable.
        replacement = trained[0]
        decision = SelectionDecision(
            policy=decision.policy,
            selected_models=(replacement,),
            weights=(1.0,),
            reason=f"{decision.reason}; {decision.selected_models[0]} unavailable, "
            f"{replacement} used instead",
            scores=decision.scores,
        )
    return decision


def _combine_backtests(
    results: dict[str, pd.DataFrame], selected_models: list[str], weights: list[float]
) -> pd.DataFrame:
    keys = ["split", "location", "time_period", "horizon", "forecast_origin"]
    combined: pd.DataFrame | None = None
    for name in selected_models:
        current = results[name][keys + ["actual", "prediction", "is_out_of_sample"]].rename(
            columns={"prediction": f"prediction_{name}"}
        )
        if combined is None:
            combined = current
        else:
            combined = combined.merge(
                current.drop(columns=["actual", "is_out_of_sample"]), on=keys, how="inner"
            )
    if combined is None or combined.empty:
        raise InsufficientResidualEvidence("no aligned backtest results for uncertainty")
    combined["prediction"] = sum(
        combined[f"prediction_{name}"] * weight
        for name, weight in zip(selected_models, weights, strict=True)
    )
    return combined


def _fit_sampler(
    config: MalariaModelConfig,
    result: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    sampler = HorizonResidualSampler(
        n_samples=config.prediction_samples,
        min_residuals_per_horizon=config.minimum_residuals_per_horizon,
        seed=config.random_seed,
    )
    try:
        sampler.fit(result)
    except (InsufficientResidualEvidence, ValueError) as exc:
        log.warning("uncertainty_unavailable", reason=str(exc))
        return None
    assert sampler.evaluated_through_ is not None
    return {
        "method": "empirical_oos_residual",
        "calibrated": False,
        "n_samples": sampler.n_samples,
        "seed": sampler.seed,
        "evaluated_through": sampler.evaluated_through_.isoformat(),
        "residuals": {
            str(horizon): values.astype(float).tolist()
            for horizon, values in sampler.residuals_.items()
        },
    }


def _restore_sampler(descriptor: dict[str, Any]) -> HorizonResidualSampler | None:
    uncertainty = descriptor.get("uncertainty")
    if uncertainty is None:
        return None
    if not isinstance(uncertainty, dict) or uncertainty.get("method") != "empirical_oos_residual":
        raise ValueError("unsupported uncertainty descriptor")
    sampler = HorizonResidualSampler(
        n_samples=int(uncertainty["n_samples"]),
        min_residuals_per_horizon=1,
        seed=int(uncertainty["seed"]),
    )
    sampler.residuals_ = {
        int(horizon): np.asarray(values, dtype=float)
        for horizon, values in uncertainty["residuals"].items()
    }
    if not sampler.residuals_ or any(
        values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
        for values in sampler.residuals_.values()
    ):
        raise ValueError("invalid residual samples in training descriptor")
    sampler.evaluated_through_ = pd.Timestamp(uncertainty["evaluated_through"])
    return sampler


def _train_sync(config: MalariaModelConfig, raw: pd.DataFrame) -> dict[str, Any]:
    history = validate_history(raw)
    requested = [config.engine] if config.engine in SUPPORTED_ENGINES else list(SUPPORTED_ENGINES)
    training_id = f"train-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    models: dict[str, dict[str, Any]] = {}
    backtests: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for engine in requested:
        try:
            backtests[engine] = _backtest(config, history, engine)
        except Exception as exc:  # evidence failure is retained, never hidden
            errors[f"{engine}_backtest"] = f"{type(exc).__name__}: {exc}"
            log.warning("model_backtest_failed", engine=engine, error=str(exc))
        try:
            model = DirectForecaster(_engine_factory(config, engine)).fit(history)
            models[engine] = pack_forecaster(model)
        except Exception as exc:
            errors[f"{engine}_training"] = f"{type(exc).__name__}: {exc}"
            log.exception("model_training_failed", engine=engine)

    if not models:
        raise RuntimeError(f"all requested model trainings failed: {errors}")

    if config.engine == "auto":
        scores = [
            _historical_score(engine, result)
            for engine, result in backtests.items()
            if engine in models
        ]
        decision = _choose_auto(config, scores, history["time_period"].max(), list(models))
        selected_models = list(decision.selected_models)
        weights = list(decision.weights)
        decision_json = _decision_to_json(decision)
    else:
        selected_models = [config.engine]
        weights = [1.0]
        decision_json = {
            "policy": "manual",
            "selected_models": selected_models,
            "weights": weights,
            "reason": "engine explicitly selected by configured model",
            "scores": [],
        }

    selected_backtest: pd.DataFrame | None = None
    if all(name in backtests for name in selected_models):
        selected_backtest = _combine_backtests(backtests, selected_models, weights)
    descriptor: dict[str, Any] = {
        "format_version": DESCRIPTOR_VERSION,
        "service_id": SERVICE_ID,
        "model_version": MODEL_VERSION,
        "training_id": training_id,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(),
        "data_through": pd.Timestamp(history["time_period"].max()).isoformat(),
        "locations": int(history["location"].nunique()),
        "rows": len(history),
        "config": config.model_dump(mode="json"),
        "models": models,
        "selection": decision_json,
        "training_errors": errors,
        "uncertainty": _fit_sampler(config, selected_backtest),
    }
    log.info(
        "malaria_models_trained",
        training_id=training_id,
        engines=list(models),
        selected_models=selected_models,
        rows=len(history),
        locations=descriptor["locations"],
        uncertainty_available=descriptor["uncertainty"] is not None,
    )
    return descriptor


async def on_train(
    config: MalariaModelConfig,
    data: DataFrame,
    geo: FeatureCollection[Any] | None = None,
) -> dict[str, Any]:
    """Train requested engines and return a self-contained, checksummed descriptor."""

    del geo
    return await asyncio.to_thread(_train_sync, config, data.to_pandas())


def _validate_descriptor(model: Any) -> dict[str, Any]:
    if type(model) is not dict or model.get("format_version") != DESCRIPTOR_VERSION:
        raise ValueError(
            "invalid or incompatible training descriptor; retrain with this service version"
        )
    if model.get("service_id") != SERVICE_ID:
        raise ValueError("training descriptor belongs to a different service")
    models = model.get("models")
    selection = model.get("selection")
    if not isinstance(models, dict) or not isinstance(selection, dict):
        raise ValueError("training descriptor is missing models or selection")
    if not models or any(name not in SUPPORTED_ENGINES for name in models):
        raise ValueError("training descriptor contains unsupported engine references")
    selected = selection.get("selected_models")
    weights = selection.get("weights")
    if (
        not isinstance(selected, list)
        or not isinstance(weights, list)
        or len(selected) != len(weights)
    ):
        raise ValueError("invalid selection decision in training descriptor")
    return model


def _prediction_orchestrator(descriptor: dict[str, Any]) -> FallbackForecaster:
    loaded: dict[str, Any] = {}
    load_errors: dict[str, str] = {}
    for engine, bundle in descriptor["models"].items():
        try:
            loaded[engine] = unpack_forecaster(bundle)
        except Exception as exc:
            load_errors[engine] = f"{type(exc).__name__}: {exc}"
    selection = descriptor["selection"]
    selected = list(selection["selected_models"])
    weights = [float(value) for value in selection["weights"]]
    candidates: dict[str, Any] = dict(loaded)
    primary: str
    if len(selected) > 1 and all(name in loaded for name in selected):
        primary = "weighted_ensemble"
        candidates[primary] = EnsembleForecaster(
            models={name: loaded[name] for name in selected},
            weights=dict(zip(selected, weights, strict=True)),
        )
    else:
        primary = selected[0]
    alternatives = tuple(name for name in SUPPORTED_ENGINES if name != primary and name in loaded)
    if descriptor["config"].get("seasonal_fallback", True):
        candidates["naive_seasonal"] = SeasonalNaiveForecaster()
        alternatives = (*alternatives, "naive_seasonal")
    if primary not in candidates:
        log.error("primary_artifact_unavailable", primary=primary, errors=load_errors)
    return FallbackForecaster(models=candidates, primary=primary, alternatives=alternatives)


def _align_to_future(future: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if "location" not in future or "time_period" not in future:
        raise ValueError("future data requires location and time_period")
    output = future.copy()
    output["_normalized_period"] = output["time_period"].map(normalize_week)
    predicted = predictions.copy()
    predicted["_normalized_period"] = predicted["time_period"].map(normalize_week)
    sample_columns = [
        column
        for column in predicted.columns
        if isinstance(column, str) and column.startswith("sample_")
    ]
    if not sample_columns:
        predicted["sample_0"] = predicted["prediction"]
        sample_columns = ["sample_0"]
    values = predicted[["location", "_normalized_period", *sample_columns]]
    merged = output.merge(
        values,
        on=["location", "_normalized_period"],
        how="left",
        validate="one_to_one",
    )
    if merged[sample_columns].isna().any().any():
        missing = merged.loc[merged[sample_columns].isna().any(axis=1), ["location", "time_period"]]
        preview = missing.head(5).to_dict("records")
        raise ValueError(f"no forecast produced for future rows: {preview}")
    return merged.drop(columns=["_normalized_period"])


def _predict_sync(model: Any, historic: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    descriptor = _validate_descriptor(model)
    history = validate_history(historic)
    orchestrator = _prediction_orchestrator(descriptor)
    predictions = orchestrator.predict(history)
    used_fallback = bool(predictions["used_fallback"].any())
    sampler = None if used_fallback else _restore_sampler(descriptor)
    if sampler is not None:
        predictions = sampler.add_samples(predictions)
    output = _align_to_future(future, predictions)
    log.info(
        "malaria_predictions_created",
        training_id=descriptor["training_id"],
        selected_model=str(predictions["selected_model"].iloc[0]),
        requested_model=str(predictions["requested_model"].iloc[0]),
        used_fallback=used_fallback,
        fallback_reason=str(predictions["fallback_reason"].iloc[0]),
        rows=len(output),
        samples=len(
            [
                column
                for column in output.columns
                if isinstance(column, str) and column.startswith("sample_")
            ]
        ),
        uncertainty_method=(descriptor.get("uncertainty") or {}).get("method"),
        probabilistic_calibrated=False,
    )
    return output


async def on_predict(
    config: MalariaModelConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection[Any] | None = None,
) -> DataFrame:
    """Load verified native models and return CHAP `sample_N` columns."""

    del config, geo
    output = await asyncio.to_thread(_predict_sync, model, historic.to_pandas(), future.to_pandas())
    return DataFrame.from_pandas(output)


def _diagnostic_error(code: str, message: str, field: str) -> ValidationDiagnostic:
    return ValidationDiagnostic.error(code=code, message=message, field=field)


async def on_validate_train(
    config: MalariaModelConfig,
    data: DataFrame,
    geo: FeatureCollection[Any] | None = None,
) -> list[ValidationDiagnostic]:
    del config, geo
    try:
        frame = validate_history(data.to_pandas())
    except (DataValidationError, TypeError, ValueError) as exc:
        return [_diagnostic_error("invalid_weekly_history", str(exc), "data")]
    diagnostics: list[ValidationDiagnostic] = []
    weeks = frame.groupby("location", observed=True)["time_period"].nunique()
    if int(weeks.min()) < 55:
        diagnostics.append(
            ValidationDiagnostic.warning(
                code="short_location_history",
                message=(
                    "At least one location has fewer than 55 weeks; lag-52 evidence is limited."
                ),
                field="data",
            )
        )
    return diagnostics


async def on_validate_predict(
    config: MalariaModelConfig,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection[Any] | None = None,
) -> list[ValidationDiagnostic]:
    del geo
    try:
        history = validate_history(historic.to_pandas())
    except (DataValidationError, TypeError, ValueError) as exc:
        return [_diagnostic_error("invalid_historic_context", str(exc), "historic")]
    future_frame = future.to_pandas()
    required = {"location", "time_period"}
    missing = required.difference(future_frame.columns)
    if missing:
        return [
            _diagnostic_error(
                "future_columns_missing", f"future is missing columns: {sorted(missing)}", "future"
            )
        ]
    diagnostics: list[ValidationDiagnostic] = []
    try:
        normalized = future_frame.copy()
        normalized["time_period"] = normalized["time_period"].map(normalize_week)
    except ValueError as exc:
        return [_diagnostic_error("invalid_future_period", str(exc), "future.time_period")]
    unknown = sorted(set(normalized["location"].astype(str)) - set(history["location"].astype(str)))
    if unknown:
        diagnostics.append(
            _diagnostic_error(
                "unknown_future_location",
                f"future contains location(s) absent from history: {unknown[:5]}",
                "future.location",
            )
        )
    max_requested = normalized.groupby("location", observed=True)["time_period"].nunique().max()
    if int(max_requested) > config.prediction_periods or int(max_requested) > 3:
        diagnostics.append(
            _diagnostic_error(
                "unsupported_forecast_horizon",
                "future requests more periods than the configured maximum of three.",
                "future.time_period",
            )
        )
    return diagnostics


def create_app() -> Any:
    """Build the official CHAPKit application."""

    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
    if database_url.startswith("sqlite") and ":///" in database_url:
        db_path = Path(database_url.split("///", 1)[1])
        db_path.parent.mkdir(parents=True, exist_ok=True)
    info = MLServiceInfo(
        id=SERVICE_ID,
        display_name="Malaria XGBoost / LightGBM / Auto",
        version=MODEL_VERSION,
        description=(
            "Weekly direct multi-horizon malaria forecasts with explicit model selection, "
            "traceable fallback and empirical out-of-sample residual samples."
        ),
        model_metadata=ModelMetadata(
            author="HISP DRC engineering team",
            author_note=(
                "Software validated on synthetic data only; not scientifically validated "
                "for national use."
            ),
            author_assessed_status=AssessedStatus.orange,
            organization="HISP DRC",
            # Also the AGPL-3.0 section 13 source offer to users reaching the service.
            repository_url=HttpUrl(REPOSITORY_URL),
            documentation_url=HttpUrl(f"{REPOSITORY_URL}#readme"),
        ),
        period_type=PeriodType.weekly,
        min_prediction_periods=1,
        max_prediction_periods=3,
        required_covariates=["rainfall", "mean_temperature", "population"],
        allow_free_additional_continuous_covariates=False,
        requires_geo=False,
        target="disease_cases",
    )
    hierarchy = ArtifactHierarchy(
        name="chap_malaria_forecasting",
        level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
    )
    runner: FunctionalModelRunner[MalariaModelConfig] = FunctionalModelRunner(
        on_train=on_train,
        on_predict=on_predict,
        on_validate_train=on_validate_train,
        on_validate_predict=on_validate_predict,
    )
    return (
        MLServiceBuilder(
            info=info,
            config_schema=MalariaModelConfig,
            hierarchy=hierarchy,
            runner=runner,
            database_url=database_url,
        )
        .with_monitoring()
        .with_registration()
        .build()
    )


app = create_app()
