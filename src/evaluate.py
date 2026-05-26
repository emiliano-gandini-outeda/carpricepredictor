import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def run(data_path: str = "data/processed/auto_features.csv",
        model_path: str = "models/model.pkl",
        scaler_path: str = "models/scaler.pkl",
        reports_dir: str = "reports") -> dict:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    auto = pd.read_csv(data_path)
    X = auto.drop("price", axis=1)
    y = auto["price"]

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_test_scaled = scaler.transform(X_test)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    r2 = r2_score(y_test_actual, y_pred)
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = root_mean_squared_error(y_test_actual, y_pred)

    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    plt.scatter(y_test_actual, y_pred, alpha=0.6)
    plt.plot([y_test_actual.min(), y_test_actual.max()],
             [y_test_actual.min(), y_test_actual.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.savefig(f"{reports_dir}/actual_vs_predicted.png")
    plt.close()

    residuals = y_test_actual - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residuals")
    plt.savefig(f"{reports_dir}/residuals.png")
    plt.close()

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "plots": ["actual_vs_predicted.png", "residuals.png"],
    }


if __name__ == "__main__":
    run()
