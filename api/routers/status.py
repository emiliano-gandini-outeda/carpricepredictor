import json
from pathlib import Path
from fastapi import APIRouter
from api.schemas import StatusResponse

router = APIRouter()

DERIVED_COLS = {"horsepower_per_kg", "engine_per_kg", "mpg_avg", "footprint"}


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
    derived = []

    for col in cols:
        matched = False
        for group in groups:
            prefix = f"{group}_"
            if col.startswith(prefix):
                groups[group].append(col)
                matched = True
                break
        if matched:
            continue
        if col in DERIVED_COLS:
            derived.append(col)
        else:
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
        "name": "numeric",
        "type": "numeric",
        "values": numeric,
        "columns": numeric,
    })
    schema.append({
        "name": "derived",
        "type": "derived",
        "values": derived,
        "columns": derived,
    })
    return schema


@router.get("/status", response_model=StatusResponse)
def status():
    model_exists = Path("models/model.pkl").exists()
    data_exists = Path("data/processed/auto_features.csv").exists()
    feature_schema = _build_feature_schema() if model_exists else None

    flat_features = []
    total = 0
    if feature_schema:
        for entry in feature_schema:
            flat_features.extend(entry["columns"])
            total += len(entry["columns"])

    return StatusResponse(
        model_trained=model_exists,
        data_exists=data_exists,
        feature_count=total if feature_schema else None,
        features=flat_features if feature_schema else None,
        feature_schema=feature_schema if feature_schema else None,
    )
