import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# load
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

auto = pd.read_csv("data/processed/auto_features.csv")
X = auto.drop("price", axis=1)
y = auto["price"]

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_test_scaled = scaler.transform(X_test)
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# metrics
print(f"R²:   {r2_score(y_test_actual, y_pred):.3f}")
print(f"MAE:  {mean_absolute_error(y_test_actual, y_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test_actual, y_pred):.2f}")

# actual vs predicted plot
plt.scatter(y_test_actual, y_pred, alpha=0.6)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.savefig("reports/actual_vs_predicted.png")

# residuals plot
residuals = y_test_actual - y_pred
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residuals")
plt.savefig("reports/residuals.png")