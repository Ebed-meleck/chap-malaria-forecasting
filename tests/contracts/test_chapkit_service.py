from __future__ import annotations

import asyncio
import pickle

import pandas as pd
import pytest
from chapkit.data import DataFrame
from fastapi.testclient import TestClient

from chap_malaria_forecasting.chapkit_adapter.service import (
    MalariaModelConfig,
    create_app,
    on_predict,
    on_train,
)
from chap_malaria_forecasting.data import add_iso_weeks
from chap_malaria_forecasting.synthetic import SyntheticConfig, generate_synthetic_history


def _small_history(n_locations: int = 2) -> pd.DataFrame:
    # Two backtest splits give n_locations * 2 residuals per horizon and
    # n_locations * 6 selection observations.
    return generate_synthetic_history(
        SyntheticConfig(
            n_locations=n_locations,
            n_weeks=36,
            missing_rate=0.0,
            short_history_fraction=0.0,
            seed=71,
        )
    )


def _future_from(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for location, group in history.groupby("location", observed=True):
        last = group.sort_values("time_period").iloc[-1]
        for horizon in (1, 2, 3):
            rows.append(
                {
                    "location": str(location),
                    "time_period": add_iso_weeks(last["time_period"], horizon),
                    "rainfall": float(last["rainfall"]),
                    "mean_temperature": float(last["mean_temperature"]),
                    "population": float(last["population"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def test_service_contract_exposes_health_info_and_config_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'chapkit.db'}")
    application = create_app()

    with TestClient(application) as client:
        health = client.get("/health")
        info = client.get("/api/v1/info")
        schema = client.get("/api/v1/configs/$schema")

    assert health.status_code == 200
    assert info.status_code == 200
    assert info.json()["id"] == "chap-malaria-forecasting"
    assert info.json()["period_type"] == "weekly"
    assert schema.status_code == 200
    engine_schema = schema.json()["properties"]["engine"]
    assert engine_schema["enum"] == ["xgboost", "lightgbm", "auto"]


def _small_config(engine: str) -> MalariaModelConfig:
    return MalariaModelConfig(
        engine=engine,
        n_estimators=10,
        n_threads=1,
        selection_backtest_splits=2,
        selection_min_train_weeks=26,
        selection_min_observations=20,
        prediction_samples=10,
        minimum_residuals_per_horizon=10,
    )


@pytest.mark.asyncio
async def test_callbacks_train_real_model_and_return_chap_samples(tmp_path, monkeypatch):
    # CHAPKit copies the working directory into every workspace zip, so the
    # callbacks must not write model files under it.
    monkeypatch.chdir(tmp_path)
    history = _small_history(n_locations=6)
    config = _small_config("xgboost")

    descriptor = await on_train(config, DataFrame.from_pandas(history))
    # CHAPKit persists the training result with pickle and reloads it for predict.
    restored = pickle.loads(pickle.dumps(descriptor))  # noqa: S301 - self-produced bytes
    prediction = await on_predict(
        config,
        restored,
        DataFrame.from_pandas(history),
        DataFrame.from_pandas(_future_from(history)),
    )
    output = prediction.to_pandas()

    assert list(tmp_path.iterdir()) == []
    assert descriptor["selection"]["policy"] == "manual"
    assert descriptor["models"].keys() == {"xgboost"}
    assert descriptor["uncertainty"]["calibrated"] is False
    assert len(output) == 18
    sample_columns = [column for column in output.columns if column.startswith("sample_")]
    assert sample_columns == [f"sample_{index}" for index in range(10)]
    assert output[sample_columns].notna().all().all()
    assert (output[sample_columns] >= 0).all().all()


@pytest.mark.asyncio
async def test_auto_engine_trains_both_and_records_selection():
    history = _small_history(n_locations=6)
    descriptor = await on_train(_small_config("auto"), DataFrame.from_pandas(history))

    selection = descriptor["selection"]
    assert descriptor["models"].keys() == {"xgboost", "lightgbm"}
    assert selection["policy"] == "auto_select"
    assert selection["reason"] in {
        "lowest eligible historical out-of-sample MAE",
        "incumbent retained to avoid a switch below the hysteresis margin",
    }
    assert {score["model_name"] for score in selection["scores"]} == {"xgboost", "lightgbm"}


@pytest.mark.asyncio
async def test_auto_engine_retains_champion_when_evidence_is_insufficient():
    history = _small_history(n_locations=2)
    config = _small_config("auto").model_copy(update={"champion": "lightgbm"})

    descriptor = await on_train(config, DataFrame.from_pandas(history))
    prediction = await on_predict(
        config,
        descriptor,
        DataFrame.from_pandas(history),
        DataFrame.from_pandas(_future_from(history)),
    )

    assert descriptor["selection"]["selected_models"] == ["lightgbm"]
    assert "insufficient historical out-of-sample evidence" in descriptor["selection"]["reason"]
    assert len(prediction.to_pandas()) == 6


@pytest.mark.asyncio
async def test_training_does_not_block_the_event_loop():
    # /health and the ServiceKit keepalive share the loop that awaits on_train;
    # Core drops a service whose keepalive is silent for 30 seconds.
    history = DataFrame.from_pandas(_small_history())
    done = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.01)
            if not done.is_set():
                ticks += 1

    async def train() -> None:
        await on_train(_small_config("xgboost"), history)
        done.set()

    await asyncio.gather(ticker(), train())

    assert ticks >= 3


@pytest.mark.asyncio
async def test_predict_rejects_descriptor_from_previous_format():
    history = DataFrame.from_pandas(_small_history())
    legacy = {"format_version": 1, "service_id": "chap-malaria-forecasting", "artifacts": {}}

    with pytest.raises(ValueError, match="retrain"):
        await on_predict(_small_config("xgboost"), legacy, history, history)
