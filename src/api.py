from fastapi import FastAPI, HTTPException, Query

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

    try:
        result = optimize_schedule(
            df=df,
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
def get_forecast(limit: int = Query(default=24, ge=1, le=500)):
    df = load_data()

    try:
        model = load_model()
    except FileNotFoundError:
        model = train_model(df)

    forecast_df = predict_energy_usage(model, df).tail(limit)
    columns = [
        TIMESTAMP_COLUMN,
        PRICE_COLUMN,
        TEMPERATURE_COLUMN,
        LOAD_COLUMN,
        ENERGY_COLUMN,
        "predicted_energy_usage_kwh",
    ]

    records = forecast_df[columns].to_dict(orient="records")

    return {
        "rows": len(df),
        "returned": len(records),
        "model_source": "saved_model" if MODEL_PATH.exists() else "trained_in_memory",
        "forecast": [_serialize_record(record) for record in records],
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    return _run_optimization(request)


@app.get("/recommendation", response_model=OptimizeResponse)
def recommendation():
    return _run_optimization(OptimizeRequest())
