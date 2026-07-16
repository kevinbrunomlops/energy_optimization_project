import pandas as pd

from src.config import (
    ENERGY_COLUMN,
    PROCESS_DURATION_HOURS,
    ENERGY_USAGE_PER_HOUR_KWH,
    PRICE_COLUMN,
    TIMESTAMP_COLUMN,
)

PREDICTED_ENERGY_COLUMN = "predicted_energy_usage_kwh"


def calculate_window_cost(window_df):
    """
    Calculate process cost for a time window.
    """

    if PREDICTED_ENERGY_COLUMN in window_df.columns:
        energy_usage = window_df[PREDICTED_ENERGY_COLUMN]
    elif ENERGY_COLUMN in window_df.columns:
        energy_usage = window_df[ENERGY_COLUMN]
    else:
        energy_usage = ENERGY_USAGE_PER_HOUR_KWH

    if hasattr(energy_usage, "sum"):
        return (window_df[PRICE_COLUMN] * energy_usage).sum()

    return window_df[PRICE_COLUMN].sum() * energy_usage


def find_cheapest_window(
    df,
    duration_hours=PROCESS_DURATION_HOURS,
    earliest_start=6,
    latest_end=22,
):
    """
    Find the cheapest consecutive time window for one day.
    """

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if df[TIMESTAMP_COLUMN].dt.date.nunique() > 1:
        raise ValueError("find_cheapest_window expects data for one day only.")

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


def optimize_day_schedule(
    df,
    process_duration_hours=PROCESS_DURATION_HOURS,
    earliest_start=6,
    latest_end=22,
    baseline_start=8,
    baseline_end=12,
):
    """
    Optimize the process schedule for one day.
    """

    df = df.copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    date = df[TIMESTAMP_COLUMN].dt.date.iloc[0]

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
        "date": date,
        **optimized,
        "baseline_cost": round(baseline_cost, 2),
        **savings,
    }


def optimize_daily_schedule(
    df,
    process_duration_hours=PROCESS_DURATION_HOURS,
    earliest_start=6,
    latest_end=22,
    baseline_start=8,
    baseline_end=12,
):
    """
    Optimize the process schedule independently for each day in the dataset.
    """

    df = df.copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df["date"] = df[TIMESTAMP_COLUMN].dt.date

    results = []

    for _, day_df in df.groupby("date", sort=True):
        result = optimize_day_schedule(
            df=day_df,
            process_duration_hours=process_duration_hours,
            earliest_start=earliest_start,
            latest_end=latest_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
        )
        results.append(result)

    return pd.DataFrame(results)


def optimize_schedule(
    df,
    process_duration_hours=PROCESS_DURATION_HOURS,
    earliest_start=6,
    latest_end=22,
    baseline_start=8,
    baseline_end=12,
):
    """
    Run daily optimization and return the latest day's recommendation.
    """

    daily_results = optimize_daily_schedule(
        df=df,
        process_duration_hours=process_duration_hours,
        earliest_start=earliest_start,
        latest_end=latest_end,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )

    if daily_results.empty:
        raise ValueError("No daily optimization results found.")

    return daily_results.iloc[-1].to_dict()
