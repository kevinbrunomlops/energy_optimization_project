from fastapi import FastAPI, HTTPException, Query
import pandas as pd

from src.data_loader import load_data
from src.forecasting import load_model, predict_energy_usage, train_model
from src.optimizer import optimize_schedule
from src.schemas import OptimizeRequest, OptimizeResponse
from src.config import (
    DATA_PATH,
    ENERGY_COLUMN,
    LOAD_COLUMN,
    MODEL_PATH,
    PRICE_COLUMN,
    TEMPERATURE_COLUMN,
    TIMESTAMP_COLUMN,
)

FORECAST_COLUMN = "predicted_energy_usage_kwh"

app = FastAPI(
    title="Energy Optimization API",
    description="API for forecasting industrial energy usage and optimizing production run windows.",
    version="0.1.0",
)


def _serialize_record(record):
    serialized = dict(record)
    timestamp = serialized.get(TIMESTAMP_COLUMN)

    if hasattr(timestamp, "isoformat"):
        serialized[TIMESTAMP_COLUMN] = timestamp.isoformat()

    return serialized


def _load_forecast_model(df):
    model_exists = MODEL_PATH.exists()

    try:
        model = load_model()
        model_source = "saved_model"
    except FileNotFoundError:
        model = train_model(df)
        model_source = "trained_in_memory"

    if model_exists and model_source != "saved_model":
        model_source = "trained_in_memory"

    return model, model_source


def _add_energy_forecast(df):
    model, model_source = _load_forecast_model(df)
    forecast_df = predict_energy_usage(model, df)
    forecast_df[FORECAST_COLUMN] = forecast_df[FORECAST_COLUMN].round(2)

    return forecast_df, model_source


def _filter_forecast_date(df, target_date=None):
    if target_date is None:
        selected_date = df[TIMESTAMP_COLUMN].dt.date.max()
    else:
        try:
            selected_date = pd.to_datetime(target_date).date()
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="date must use YYYY-MM-DD format.",
            ) from exc

    day_df = df[df[TIMESTAMP_COLUMN].dt.date == selected_date].reset_index(drop=True)

    if day_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for date {selected_date.isoformat()}.",
        )

    return day_df, selected_date


def _to_optimize_response(result):
    date = result.get("date")

    return {
        **result,
        "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
        "recommended_start": result["recommended_start"].isoformat(),
        "recommended_end": result["recommended_end"].isoformat(),
    }


def _run_optimization(request: OptimizeRequest):
    df = load_data()
    forecast_df, _ = _add_energy_forecast(df)

    try:
        result = optimize_schedule(
            df=forecast_df,
            process_duration_hours=request.process_duration_hours,
            earliest_start=request.earliest_start,
            latest_end=request.latest_end,
            baseline_start=request.baseline_start,
            baseline_end=request.baseline_end,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_optimize_response(result)


@app.get("/")
def root():
    return {
        "service": "Energy Optimization API",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "GET /health",
            "GET /data",
            "GET /forecast",
            "POST /optimize",
            "GET /recommendation",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
    }


@app.get("/data")
def get_data(limit: int = Query(default=24, ge=1, le=500)):
    df = load_data()
    records = df.tail(limit).to_dict(orient="records")

    return {
        "rows": len(df),
        "returned": len(records),
        "data": [_serialize_record(record) for record in records],
    }


@app.get("/forecast")
def get_forecast(
    date: str | None = Query(default=None, description="Forecast date in YYYY-MM-DD format."),
    limit: int = Query(default=24, ge=1, le=500),
):
    df = load_data()
    forecast_df, model_source = _add_energy_forecast(df)
    day_df, selected_date = _filter_forecast_date(forecast_df, date)
    day_df = day_df.head(limit)
    columns = [
        TIMESTAMP_COLUMN,
        PRICE_COLUMN,
        TEMPERATURE_COLUMN,
        LOAD_COLUMN,
        ENERGY_COLUMN,
        FORECAST_COLUMN,
    ]

    records = day_df[columns].to_dict(orient="records")
    total_predicted_usage = day_df[FORECAST_COLUMN].sum()
    estimated_total_cost = (
        day_df[PRICE_COLUMN]
        * day_df[FORECAST_COLUMN]
    ).sum()

    return {
        "date": selected_date.isoformat(),
        "rows": len(day_df),
        "returned": len(records),
        "model_source": model_source,
        "summary": {
            "average_predicted_energy_usage_kwh": float(
                round(day_df[FORECAST_COLUMN].mean(), 2)
            ),
            "total_predicted_energy_usage_kwh": float(round(total_predicted_usage, 2)),
            "estimated_total_cost": float(round(estimated_total_cost, 2)),
        },
        "forecast": [_serialize_record(record) for record in records],
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    return _run_optimization(request)


@app.get("/recommendation", response_model=OptimizeResponse)
def recommendation():
    return _run_optimization(OptimizeRequest())
