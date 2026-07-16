from src.config import MODEL_PATH
from src.data_loader import load_data
from src.forecasting import (
    evaluate_model,
    save_model,
    train_model,
)


def main():
    """
    Train, evaluate and save the energy forecasting model.
    """

    df = load_data()

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    model = train_model(train_df, model_type="random_forest")

    metrics = evaluate_model(model, test_df)

    save_model(model, MODEL_PATH)

    print("Model training complete.")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
