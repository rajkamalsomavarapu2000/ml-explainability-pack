from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.train import train_router
from app.predict import predict_router
from app.explain import generate_global_explanation, generate_local_explanation
import shutil
import pandas as pd
from pathlib import Path
import json

app = FastAPI(title="ML Model API", version="1.0")

# Serve the frontend at root
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

app.include_router(train_router, prefix="/train", tags=["training"])
app.include_router(predict_router, prefix="/predict", tags=["prediction"])

from app.storage import UPLOAD_DIR, MODEL_DIR
import uuid
import os

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    dataset_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{dataset_id}.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"dataset_id": dataset_id, "filename": file.filename}

@app.get("/datasets")
async def list_datasets():
    datasets = []
    # Sample dataset
    sample_path = "data/sample.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        datasets.append({
            "dataset_id": "sample",
            "filename": "sample.csv",
            "rows": len(df),
            "columns": len(df.columns),
            "has_target": "target" in df.columns
        })
    # Uploaded datasets
    for csv_file in UPLOAD_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        dataset_id = csv_file.stem
        datasets.append({
            "dataset_id": dataset_id,
            "filename": f"{dataset_id}.csv",
            "rows": len(df),
            "columns": len(df.columns),
            "has_target": "target" in df.columns
        })
    return {"datasets": datasets}

@app.get("/datasets/{dataset_id}/columns")
async def get_dataset_columns(dataset_id: str):
    if dataset_id == "sample":
        csv_path = "data/sample.csv"
    else:
        csv_path = UPLOAD_DIR / f"{dataset_id}.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_csv(csv_path)
    return {"columns": list(df.columns)}

@app.get("/models")
async def list_models():
    models = []
    models_dir = Path("/testbed/ml-explainability-pack/models")
    for meta_file in models_dir.glob("*_meta.json"):
        try:
            with open(meta_file, 'r') as f:
                content = f.read().strip().replace("NaN", "null")
                models.append(json.loads(content))
        except Exception as e:
            # Skip corrupted meta files
            continue
    return {"models": models}

@app.post("/explain/global/{model_id}")
async def explain_global(model_id: str):
    try:
        explanation = generate_global_explanation(model_id)
        return {"model_id": model_id, "global_feature_importance": explanation}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/explain/local/{model_id}")
async def explain_local(model_id: str, row_index: int):
    try:
        explanation = generate_local_explanation(model_id, row_index)
        return {"model_id": model_id, "local_explanation": explanation}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# Serve static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")
