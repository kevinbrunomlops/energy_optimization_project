import pandas as pd
import pytest

from src.config import PRICE_COLUMN, TIMESTAMP_COLUMN
from src.optimizer import (
    PREDICTED_ENERGY_COLUMN,
    find_cheapest_window,
    optimize_daily_schedule,
)


def make_day(date="2026-01-01", cheap_start=10):
    timestamps = pd.date_range(date, periods=24, freq="h")
    prices = [2.0] * 24

    for hour in range(cheap_start, cheap_start + 4):
        prices[hour] = 0.5

    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            PRICE_COLUMN: prices,
            PREDICTED_ENERGY_COLUMN: [100.0] * 24,
        }
    )


def test_find_cheapest_window_selects_lowest_four_hour_slot():
    df = make_day(cheap_start=10)

    result = find_cheapest_window(
        df,
        duration_hours=4,
        earliest_start=6,
        latest_end=22,
    )

    assert result["recommended_start"].hour == 10
    assert result["recommended_end"].hour == 14
    assert result["estimated_cost"] == 200.0


def test_find_cheapest_window_rejects_impossible_constraints():
    df = make_day()

    with pytest.raises(ValueError, match="No valid optimization window"):
        find_cheapest_window(
            df,
            duration_hours=4,
            earliest_start=20,
            latest_end=22,
        )


def test_optimize_daily_schedule_returns_one_result_per_day():
    df = pd.concat(
        [
            make_day("2026-01-01", cheap_start=10),
            make_day("2026-01-02", cheap_start=12),
        ],
        ignore_index=True,
    )

    results = optimize_daily_schedule(
        df,
        process_duration_hours=4,
        earliest_start=6,
        latest_end=22,
        baseline_start=8,
        baseline_end=12,
    )

    assert len(results) == 2
    assert results["recommended_start"].dt.hour.tolist() == [10, 12]
    assert {"estimated_cost", "baseline_cost", "saving_percent"}.issubset(results.columns)
