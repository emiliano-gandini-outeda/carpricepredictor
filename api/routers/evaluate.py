from fastapi import APIRouter
from src.pipeline import run_evaluation

router = APIRouter()


@router.post("/evaluate")
def evaluate():
    try:
        metrics = run_evaluation()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}
