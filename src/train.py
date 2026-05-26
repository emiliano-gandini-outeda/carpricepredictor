from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import json


def run(input_path: str = "data/processed/auto_features.csv",
        model_path: str = "models/model.pkl",
        scaler_path: str = "models/scaler.pkl",
        columns_path: str = "models/feature_columns.json") -> dict:
    auto = pd.read_csv(input_path)

    X = auto.drop("price", axis=1)
    y = auto["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    r2_train = model.score(X_train_scaled, y_train)
    r2_test = model.score(X_test_scaled, y_test)

    print('Coefficient of determination (train):{:.3f}'.format(r2_train))
    print('Coefficient of determination (test):{:.3f}'.format(r2_test))
    print('\nRegression coefficients\n{}'.format(pd.Series(model.coef_, index=X.columns)))
    print('Intercept: {:.3f}'.format(model.intercept_))

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(columns_path, "w") as f:
        json.dump(list(X.columns), f)

    return {
        "r2_train": round(r2_train, 4),
        "r2_test": round(r2_test, 4),
        "intercept": round(model.intercept_, 4),
        "feature_count": len(X.columns),
    }


if __name__ == "__main__":
    run()
