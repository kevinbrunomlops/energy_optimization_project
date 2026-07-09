import pandas as pd 

from src.config import (
    TIMESTAMP_COLUMN,
    LOAD_COLUMN,
    ENERGY_COLUMN,
)

LOAD_MAPPING = {
    "low": 0,
    "medium": 1, 
    "high": 2,
}

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month", 
    "production_load_encoded",
]

def create_features(df):
    """
    Create model features from raw energy data.
    """

    df = df.copy()

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

    df["hour"] = df[TIMESTAMP_COLUMN].dt.hour
    df["day_of_week"] = df[TIMESTAMP_COLUMN].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df[TIMESTAMP_COLUMN].dt.month

    df["production_load_encoded"] = df[LOAD_COLUMN].map(LOAD_MAPPING)

    if df["production_load_encoded"].isna().any():
        raise ValueError("Unknown production_load value found.")
    
    return df

def get_model_input(df):
    """
    Return X features for model prediction/training.
    """

    df = create_features(df)

    X = df[FEATURE_COLUMNS]

    return X

def get_target(df):
    """
    Return y target variable.
    """

    if ENERGY_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {ENERGY_COLUMN}")
    
    return df[ENERGY_COLUMN]