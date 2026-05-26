from fastapi import APIRouter
from src.pipeline import run_full_setup

router = APIRouter()


@router.post("/setup")
def setup():
    try:
        metrics = run_full_setup()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}
