import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from src.config import MODEL_PATH, RANDOM_STATE
from src.features import get_model_input, get_target

def train_linear_regression(df):
    """ 
    Train a Linear Regression model.
    """

    X = get_model_input(df)
    y = get_target(df)

    model = LinearRegression()
    model.fit(X, y)

    return model

def train_random_forest(df):
    """
    Train a Random Forest model.
    """

    X = get_model_input(df)
    y = get_target(df)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
    )

    model.fit(X, y)

    return model

def evaluate_model(model, df):
    """
    Evaluate model using MAE and RMSE
    """

    X = get_model_input(df)
    y = get_target(df)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return {
        "mae": mae,
        "rmse": rmse,
    }

def save_model(model, model_path=MODEL_PATH):
    """
    Save trained model to disk
    """

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

def load_model(model_path=MODEL_PATH):
    """
    Load trained model from disk
    """

    return joblib.load(model_path)

def predict_energy_usage(model, df):
    """
    Predict energy usage in kWh.
    """

    X = get_model_input(df)

    predictions = model.predict(X)

    result_df = df.copy()
    result_df["predicted_energy_usage_kwh"] = predictions

    return result_df