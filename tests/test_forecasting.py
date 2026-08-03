from src.data_generator import generate_energy_data
from src.features import FEATURE_COLUMNS, get_model_input
from src.forecasting import predict_energy_usage, train_model


def test_feature_engineering_returns_expected_columns(tmp_path):
    df = generate_energy_data(days=7, output_path=tmp_path / "energy.csv")

    features = get_model_input(df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(df)


def test_linear_model_returns_numeric_energy_predictions(tmp_path):
    df = generate_energy_data(days=14, output_path=tmp_path / "energy.csv")
    model = train_model(df, model_type="linear")

    predictions = predict_energy_usage(model, df.head(24))

    assert "predicted_energy_usage_kwh" in predictions.columns
    assert len(predictions) == 24
    assert predictions["predicted_energy_usage_kwh"].notna().all()
    assert predictions["predicted_energy_usage_kwh"].dtype.kind in {"f", "i"}
