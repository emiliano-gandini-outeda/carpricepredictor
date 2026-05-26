import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()

NOTEBOOKS_DIR = Path("notebooks")


@router.get("/notebooks")
def list_notebooks():
    if not NOTEBOOKS_DIR.exists():
        return {"notebooks": []}
    files = sorted(f.name for f in NOTEBOOKS_DIR.iterdir() if f.suffix == ".ipynb")
    return {"notebooks": files}


@router.get("/notebooks/{name}")
def get_notebook(name: str):
    path = NOTEBOOKS_DIR / name
    if not path.exists() or path.suffix != ".ipynb":
        raise HTTPException(status_code=404, detail="Notebook not found")
    with open(path) as f:
        return json.load(f)
