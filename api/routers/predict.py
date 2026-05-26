from fastapi import APIRouter, HTTPException
from api.schemas import PredictRequest, PredictResponse
from src.pipeline import run_prediction
import json

router = APIRouter()

FEATURE_COLS = None


def _load_feature_cols():
    global FEATURE_COLS
    if FEATURE_COLS is None:
        with open("models/feature_columns.json") as f:
            FEATURE_COLS = json.load(f)
    return FEATURE_COLS


def _build_feature_schema() -> dict:
    cols = _load_feature_cols()
    groups = {
        "make": [],
        "body-style": [],
        "drive-wheels": [],
        "engine-type": [],
        "num-of-cylinders": [],
        "fuel-system": [],
    }
    for col in cols:
        for group in groups:
            if col.startswith(f"{group}_"):
                groups[group].append(col)
                break
    return groups


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    schema = _build_feature_schema()
    feature_cols = _load_feature_cols()
    features = {}

    # Numeric fields
    features['symboling'] = req.symboling
    features['height'] = req.height
    features['curb-weight'] = req.curb_weight
    features['engine-size'] = req.engine_size
    features['bore'] = req.bore
    features['stroke'] = req.stroke
    features['horsepower'] = req.horsepower
    features['peak-rpm'] = req.peak_rpm

    # One-hot categoricals
    for col in schema.get("make", []):
        suffix = col.split("make_", 1)[1]
        features[col] = 1 if suffix == req.make else 0
    features['aspiration_turbo'] = req.aspiration_turbo
    for col in schema.get("body-style", []):
        suffix = col.split("body-style_", 1)[1]
        features[col] = 1 if suffix == req.body_style else 0
    for col in schema.get("drive-wheels", []):
        suffix = col.split("drive-wheels_", 1)[1]
        features[col] = 1 if suffix == req.drive_wheels else 0
    features['engine-location_rear'] = req.engine_location_rear
    for col in schema.get("engine-type", []):
        suffix = col.split("engine-type_", 1)[1]
        features[col] = 1 if suffix == req.engine_type else 0
    for col in schema.get("num-of-cylinders", []):
        suffix = col.split("num-of-cylinders_", 1)[1]
        features[col] = 1 if suffix == req.num_of_cylinders else 0
    for col in schema.get("fuel-system", []):
        suffix = col.split("fuel-system_", 1)[1]
        features[col] = 1 if suffix == req.fuel_system else 0

    # Derived features
    features['horsepower_per_kg'] = req.horsepower / req.curb_weight
    features['engine_per_kg'] = req.engine_size / req.curb_weight
    features['mpg_avg'] = req.mpg_avg
    features['footprint'] = req.footprint

    try:
        price = run_prediction(features)
        return PredictResponse(predicted_price=round(price, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
