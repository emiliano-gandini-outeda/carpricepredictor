from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .routers import setup, evaluate, predict, status, notebooks

app = FastAPI(title="Car Price Predictor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(setup.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(notebooks.router, prefix="/api")

reports_dir = Path(__file__).resolve().parent.parent / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir), html=True), name="reports")

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
