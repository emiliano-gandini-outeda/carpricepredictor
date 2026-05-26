from pydantic import BaseModel


class PredictRequest(BaseModel):
    symboling: int = 1
    height: float = 55.0
    curb_weight: float = 2500.0
    engine_size: float = 130.0
    bore: float = 3.5
    stroke: float = 3.0
    horsepower: float = 110.0
    peak_rpm: float = 5500.0
    make: str = "honda"
    aspiration_turbo: int = 0
    body_style: str = "sedan"
    drive_wheels: str = "rwd"
    engine_location_rear: int = 0
    engine_type: str = "ohc"
    num_of_cylinders: str = "four"
    fuel_system: str = "mpfi"


class PredictResponse(BaseModel):
    predicted_price: float


class SetupResponse(BaseModel):
    status: str
    metrics: dict | None = None


class EvaluateResponse(BaseModel):
    status: str
    metrics: dict | None = None


class StatusResponse(BaseModel):
    model_trained: bool
    data_exists: bool
    feature_count: int | None = None
    features: list[str] | None = None
    feature_schema: list[dict] | None = None
