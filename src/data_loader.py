import pandas as pd

from src.config import (
    DATA_PATH, 
    TIMESTAMP_COLUMN,
    PRICE_COLUMN, 
    TEMPERATURE_COLUMN,
    LOAD_COLUMN,
    ENERGY_COLUMN
)

REQUIRED_COLUMNS = [
    TIMESTAMP_COLUMN,
    PRICE_COLUMN,
    TEMPERATURE_COLUMN,
    LOAD_COLUMN,
    ENERGY_COLUMN,
]

def load_data(file_path=DATA_PATH):
    """
    Load energy data from CSV and return a clean DataFrame.
    """

    df = pd.read_csv(file_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COLUMN])
    
    df = df.sort_values(TIMESTAMP_COLUMN)

    df = df.drop_duplicates()

    df = df.dropna()

    df = df.reset_indes(drop=True)

    return df