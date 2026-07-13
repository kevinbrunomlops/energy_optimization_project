from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data/simulated"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_PATH = DATA_DIR / "hourly_energy_data.csv"
MODEL_PATH = PROJECT_ROOT / "energy_model.pkl"

# Process settings
PROCESS_DURATION_HOURS = 4
ENERGY_USAGE_PER_HOUR_KWH = 120

# Standard run window
DEFAULT_START_HOUR = 8
DEFAULT_END_HOUR = 12

# Allowed optimization window
ALLOWED_START_HOUR = 6
ALLOWED_END_HOUR = 22

# Dataset columns
TIMESTAMP_COLUMN = "timestamp"
PRICE_COLUMN = "electricity_price"
TEMPERATURE_COLUMN = "temperature"
LOAD_COLUMN = "production_load"
ENERGY_COLUMN = "historical_energy_usage"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
