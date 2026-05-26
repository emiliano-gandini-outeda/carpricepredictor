import pandas as pd
import numpy as np
import joblib
import json


def run(features: dict = None,
        model_path: str = "models/model.pkl",
        scaler_path: str = "models/scaler.pkl",
        columns_path: str = "models/feature_columns.json") -> float:
    if features is None:
        features = {
            'symboling': 1, 'height': 55.0, 'curb-weight': 2500,
            'engine-size': 130, 'bore': 3.5, 'stroke': 3.0,
            'horsepower': 110.0, 'peak-rpm': 5500.0,
            'horsepower_per_kg': 110.0 / 2500,
            'engine_per_kg': 130 / 2500,
            'mpg_avg': (25 + 30) / 2,
            'footprint': 170 * 65,
            'make_audi': 0, 'make_bmw': 0, 'make_chevrolet': 0,
            'make_dodge': 0, 'make_honda': 1, 'make_jaguar': 0,
            'make_mazda': 0, 'make_mercedes-benz': 0, 'make_mitsubishi': 0,
            'make_nissan': 0, 'make_peugot': 0, 'make_plymouth': 0,
            'make_porsche': 0, 'make_saab': 0, 'make_subaru': 0,
            'make_toyota': 0, 'make_volkswagen': 0, 'make_volvo': 0,
            'aspiration_turbo': 0,
            'body-style_hardtop': 0, 'body-style_hatchback': 0,
            'body-style_sedan': 1,
            'drive-wheels_rwd': 0, 'engine-location_rear': 0,
            'engine-type_ohc': 1, 'engine-type_ohcv': 0,
            'num-of-cylinders_five': 0, 'num-of-cylinders_four': 1,
            'num-of-cylinders_six': 0, 'num-of-cylinders_three': 0,
            'num-of-cylinders_twelve': 0,
            'fuel-system_2bbl': 0, 'fuel-system_idi': 0,
            'fuel-system_mpfi': 0, 'fuel-system_spdi': 0,
        }

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = json.load(open(columns_path))

    sample = pd.DataFrame([features])

    log_cols = ['engine-size', 'horsepower', 'horsepower_per_kg', 'engine_per_kg']
    for col in log_cols:
        if col in sample.columns:
            sample[col] = np.log1p(sample[col])

    sample = sample[feature_cols]

    sample_scaled = scaler.transform(sample)
    log_prediction = model.predict(sample_scaled)
    price = float(np.expm1(log_prediction)[0])

    return price


if __name__ == "__main__":
    price = run()
    print(f"Predicted price: ${price:,.2f}")
