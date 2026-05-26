from fastapi import APIRouter, HTTPException
from api.schemas import PredictRequest, PredictResponse
from src.pipeline import run_prediction
import pandas as pd
import json

router = APIRouter()


def _resolve_onehot(feature_schema: dict, field: str, value: str) -> dict:
    """Given a categorical field like 'make' and value 'honda',
    return the one-hot dict like {'make_honda': 1, 'make_audi': 0, ...}"""
    result = {}
    for col in feature_schema.get(field, []):
        prefix = f"{field}_"
        suffix = col[len(prefix):]
        result[col] = 1 if suffix == value else 0
    return result


def _build_feature_schema() -> dict:
    """Parse feature_columns.json into categorical groups + numeric list."""
    with open("models/feature_columns.json") as f:
        cols = json.load(f)

    groups = {
        "make": [],
        "body-style": [],
        "drive-wheels": [],
        "engine-type": [],
        "num-of-cylinders": [],
        "fuel-system": [],
    }
    numeric = []

    for col in cols:
        matched = False
        for group in groups:
            prefix = f"{group}_"
            if col.startswith(prefix):
                groups[group].append(col)
                matched = True
                break
        if not matched:
            numeric.append(col)

    return {"groups": groups, "numeric": numeric}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import json
    import numpy as np

    schema = _build_feature_schema()
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

    # Computed numeric
    features['horsepower_per_kg'] = req.horsepower / req.curb_weight
    features['engine_per_kg'] = req.engine_size / req.curb_weight

    with open("models/feature_columns.json") as f:
        feature_cols = json.load(f)
    has_mpg = "mpg_avg" in feature_cols
    has_footprint = "footprint" in feature_cols

    if has_mpg and has_footprint:
        default_mpg = 28.0
        default_length = 170.0
        default_width = 65.0
        features['mpg_avg'] = default_mpg
        features['footprint'] = default_length * default_width

    # One-hot categoricals
    features.update(_resolve_onehot(schema["groups"], "make", req.make))
    features['aspiration_turbo'] = req.aspiration_turbo
    features.update(_resolve_onehot(schema["groups"], "body-style", req.body_style))
    features.update(_resolve_onehot(schema["groups"], "drive-wheels", req.drive_wheels))
    features['engine-location_rear'] = req.engine_location_rear
    features.update(_resolve_onehot(schema["groups"], "engine-type", req.engine_type))
    features.update(_resolve_onehot(schema["groups"], "num-of-cylinders", req.num_of_cylinders))
    features.update(_resolve_onehot(schema["groups"], "fuel-system", req.fuel_system))

    try:
        price = run_prediction(features)
        return PredictResponse(predicted_price=round(price, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
