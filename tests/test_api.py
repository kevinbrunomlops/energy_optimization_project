import pytest
from fastapi import HTTPException

from src.api import _filter_forecast_date, get_data, health, root
from src.data_loader import load_data


def test_root_endpoint_returns_service_metadata():
    response = root()

    assert response["status"] == "running"
    assert "GET /health" in response["endpoints"]


def test_health_endpoint_returns_ok_status():
    response = health()

    assert response["status"] == "ok"
    assert response["data_path"].endswith("hourly_energy_data.csv")


def test_data_endpoint_returns_limited_rows():
    response = get_data(limit=3)

    assert response["returned"] == 3
    assert len(response["data"]) == 3


def test_forecast_date_filter_raises_404_for_missing_date():
    df = load_data()

    with pytest.raises(HTTPException) as exc_info:
        _filter_forecast_date(df, target_date="1900-01-01")

    assert exc_info.value.status_code == 404
