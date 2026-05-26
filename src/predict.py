import pandas as pd
import numpy as np
import joblib
import json

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = json.load(open("models/feature_columns.json"))

sample = pd.DataFrame([{
    # numeric
    'symboling':          1,
    'height':             55.0,
    'curb-weight':        2500,
    'engine-size':        130,
    'bore':               3.5,
    'stroke':             3.0,
    'horsepower':         110.0,
    'peak-rpm':           5500.0,
    # engineered
    'horsepower_per_kg':  110.0 / 2500,
    'engine_per_kg':      130 / 2500,
    'mpg_avg':            (25 + 30) / 2,
    'footprint':          170 * 65,
    # make — set the one that applies to 1, rest are 0
    'make_audi':          0, 'make_bmw':           0, 'make_chevrolet':     0,
    'make_dodge':         0, 'make_honda':         1, 'make_jaguar':        0,
    'make_mazda':         0, 'make_mercedes-benz': 0, 'make_mitsubishi':    0,
    'make_nissan':        0, 'make_peugot':        0, 'make_plymouth':      0,
    'make_porsche':       0, 'make_saab':          0, 'make_subaru':        0,
    'make_toyota':        0, 'make_volkswagen':    0, 'make_volvo':         0,
    # other categoricals
    'aspiration_turbo':   0,
    'body-style_hardtop': 0, 'body-style_hatchback': 0, 'body-style_sedan': 1,
    'drive-wheels_rwd':   0,
    'engine-location_rear': 0,
    'engine-type_ohc':    1, 'engine-type_ohcv':   0,
    'num-of-cylinders_five': 0, 'num-of-cylinders_four': 1,
    'num-of-cylinders_six':  0, 'num-of-cylinders_three': 0,
    'num-of-cylinders_twelve': 0,
    'fuel-system_2bbl':   0, 'fuel-system_idi':    0,
    'fuel-system_mpfi':   0, 'fuel-system_spdi':   0,
}])

# apply log transform to the same skewed numeric cols as in features.py
log_cols = ['curb-weight', 'engine-size', 'horsepower', 'horsepower_per_kg',
            'engine_per_kg', 'footprint', 'mpg_avg']
for col in log_cols:
    if col in sample.columns:
        sample[col] = np.log1p(sample[col])

# reorder columns to match training
sample = sample[feature_cols]

# scale and predict
sample_scaled = scaler.transform(sample)
log_prediction = model.predict(sample_scaled)
price = np.expm1(log_prediction)

print(f"Predicted price: ${price[0]:,.2f}")