import json
from pathlib import Path
from fastapi import APIRouter
from api.schemas import StatusResponse

router = APIRouter()


def _build_feature_schema() -> list[dict]:
    cols_path = Path("models/feature_columns.json")
    if not cols_path.exists():
        return []
    with open(cols_path) as f:
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

    schema = []
    for group, members in groups.items():
        schema.append({
            "name": group,
            "type": "onehot",
            "values": [m.split(f"{group}_", 1)[1] for m in members],
            "columns": members,
        })
    schema.append({
        "name": "numeric_features",
        "type": "numeric",
        "values": numeric,
        "columns": numeric,
    })
    schema.append({
        "name": "derived",
        "type": "derived",
        "values": ["horsepower_per_kg", "engine_per_kg", "mpg_avg", "footprint"],
        "columns": ["horsepower_per_kg", "engine_per_kg", "mpg_avg", "footprint"],
    })
    return schema


@router.get("/status", response_model=StatusResponse)
def status():
    model_exists = Path("models/model.pkl").exists()
    data_exists = Path("data/processed/auto_features.csv").exists()
    feature_schema = _build_feature_schema() if model_exists else None

    return StatusResponse(
        model_trained=model_exists,
        data_exists=data_exists,
        feature_count=len(feature_schema[0]["columns"]) + len(feature_schema[1]["columns"]) + len(feature_schema[2]["columns"]) if feature_schema else None,
        features=[col for entry in feature_schema for col in entry["columns"]] if feature_schema else None,
        feature_schema=feature_schema if feature_schema else None,
    )
