import pandas as pd

from src.config import (
    PROCESS_DURATION_HOURS,
    ENERGY_USAGE_PER_HOUR_KWH,
    PRICE_COLUMN,
    TIMESTAMP_COLUMN,
)

def calculate_window_cost(window_df):
    """
    Calculate process cost for a time window.
    """

    return (
        window_df[PRICE_COLUMN].sum()
        * ENERGY_USAGE_PER_HOUR_KWH
    )

def find_cheapest_window(
    df,
    duration_hours=PROCESS_DURATION_HOURS,
    earliest_start=6,
    latest_end=22,
):
    """
    Find the cheapest consecutive time window.
    """

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    best_window = None
    lowest_cost = float("inf")

    for i in range(len(df) - duration_hours + 1):

        window = df.iloc[i:i + duration_hours]

        start_hour = window[TIMESTAMP_COLUMN].iloc[0].hour
        end_hour = start_hour + duration_hours

        if start_hour < earliest_start:
            continue

        if end_hour > latest_end:
            continue

        cost = calculate_window_cost(window)

        if cost < lowest_cost:

            lowest_cost = cost

            best_window = {
                "recommended_start": window[TIMESTAMP_COLUMN].iloc[0],
                "recommended_end": window[TIMESTAMP_COLUMN].iloc[-1]
                + pd.Timedelta(hours=1),
                "estimated_cost": round(cost, 2),
            }

    if best_window is None:
        raise ValueError("No valid optimization window found for the provided constraints.")

    return best_window

def calculate_baseline_cost(
    df,
    baseline_start=8,
    baseline_end=12,
):
    """
    Calculate the cost of the baseline schedule.
    """

    baseline = df[
        (df[TIMESTAMP_COLUMN].dt.hour >= baseline_start)
        & (df[TIMESTAMP_COLUMN].dt.hour < baseline_end)
    ]

    return calculate_window_cost(baseline)

def calculate_savings(
    optimized_cost,
    baseline_cost,
):
    """
    Calculate savings compared to baseline.
    """

    saving = baseline_cost - optimized_cost

    saving_percent = (
        saving / baseline_cost * 100
        if baseline_cost > 0
        else 0
    )

    return {
        "saving": round(saving, 2),
        "saving_percent": round(saving_percent, 2),
    }

def optimize_schedule(
    df,
    process_duration_hours=PROCESS_DURATION_HOURS,
    earliest_start=6,
    latest_end=22,
    baseline_start=8,
    baseline_end=12,
):
    """
    Run the complete optimization.
    """

    optimized = find_cheapest_window(
        df=df,
        duration_hours=process_duration_hours,
        earliest_start=earliest_start,
        latest_end=latest_end,
    )

    baseline_cost = calculate_baseline_cost(
        df,
        baseline_start,
        baseline_end,
    )

    savings = calculate_savings(
        optimized["estimated_cost"],
        baseline_cost,
    )

    return {
        **optimized,
        "baseline_cost": round(baseline_cost, 2),
        **savings,
    }
