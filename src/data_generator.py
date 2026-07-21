from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH


def generate_energy_data(
    days=90,
    start_date="2024-01-01",
    output_path=DATA_PATH,
    overwrite=True,
    random_state=42,
): 
    """
    Generate realistic simulated hourly energy data.

    Columns:
    - timestamp
    - electricity_price
    - temperature
    - production_load
    - historical_energy_usage

    """

    np.random.seed(random_state)

    timestamps = pd.date_range(
        start=start_date,
        periods=days * 24,
        freq="h",
    )

    df = pd.DataFrame({"timestamp": timestamps})

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # Temperature: seasonal + daily pattern + noise
    seasonal_temp = 10 + 10 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365)
    daily_temp = 3 * np.sin(2 * np.pi * (df["hour"] - 6) / 24)
    temp_noise = np.random.normal(0, 2, len(df))

    df["temperature"] = seasonal_temp + daily_temp + temp_noise

    # Electricity price: higher morning/evening, lower night/midday 
    base_price = 0.55

    morning_peak = np.where(
        (df["hour"] >= 7) & (df["hour"] <= 10),
        0.35,
        0,
    )

    evening_peak = np.where(
        (df["hour"] >= 17) & (df["hour"] <= 20),
        0.45,
        0,
    )

    night_discount = np.where(
        (df["hour"] >= 0) & (df["hour"] <= 5),
        -0.20,
        0,
    )

    midday_discount = np.where(
        (df["hour"] >= 12) & (df["hour"] <= 15),
        -0.10,
        0,
    )

    price_noise = np.random.normal(0, 0.08, len(df))

    df["electricity_price"] = (
        base_price
        + morning_peak
        + evening_peak
        + night_discount
        + midday_discount
        + price_noise
    )

    df["electricity_price"] = df["electricity_price"].clip(lower=0.05).round(3)

    # Production load
    load_probabilities = [0.25, 0.50, 0.25]
    df["production_load"] = np.random.choice(
        ["low", "medium", "high"],
        size=len(df),
        p=load_probabilities,
    )

    load_effect = df["production_load"].map({
        "low": 70,
        "medium": 120,
        "high": 180,
    })

    # Energy usage: affected by production load, temperature and noise 

    temperature_effect = np.where(
        df["temperature"] < 5,
        (5 - df["temperature"]) * 2.5,
        0,
    )
    
    energy_noise = np.random.normal(0, 10, len(df))

    df["historical_energy_usage"] = (
        load_effect
        + temperature_effect
        + energy_noise
    )

    df["historical_energy_usage"] = df["historical_energy_usage"].clip(lower=20).round(2)
    df["temperature"] = df["temperature"].round(1)

    final_df = df[
        [
            "timestamp",
            "electricity_price",
            "temperature",
            "production_load",
            "historical_energy_usage",
        ]
    ].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists."
        )

    final_df.to_csv(output_path, index=False)

    return final_df


if __name__ == "__main__":
    df = generate_energy_data(days=90)
    print(f"Generated {len(df)} rows")
    print(df.head())
    
    
