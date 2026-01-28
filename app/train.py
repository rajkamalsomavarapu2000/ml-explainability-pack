import uuid
import json
import math
import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from app.storage import MODEL_DIR, UPLOAD_DIR
from pydantic import BaseModel

train_router = APIRouter()

class TrainRequest(BaseModel):
    dataset_id: str = "sample"
    model_type: str = "logistic"
    target_column: str = None

@train_router.post("/")
def train_model(request: TrainRequest):
    if request.dataset_id == "sample":
        csv_path = "data/sample.csv"
    else:
        csv_path = UPLOAD_DIR / f"{request.dataset_id}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(csv_path)

    supervised = request.model_type in ["logistic", "random_forest"]
    if supervised:
        if not request.target_column:
            raise HTTPException(status_code=400, detail="Target column required for supervised models")
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not found in dataset")
        X = df.drop(request.target_column, axis=1)
        y = df[request.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # For clustering, use all numeric columns
        X = df.select_dtypes(include=[np.number])
        X_train = X

    if request.model_type == "logistic":
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
    elif request.model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
    elif request.model_type == "kmeans":
        model = KMeans(n_clusters=3, random_state=42)  # Default 3 clusters
        model.fit(X_train)
        score = model.inertia_  # Use inertia as score
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    score_value = round(score, 4) if isinstance(score, (int, float)) and not math.isnan(score) else None
    meta = {
        "model_id": model_id,
        "model_type": request.model_type,
        "dataset_id": request.dataset_id,
        "target_column": request.target_column if supervised else None,
        "score": score_value
    }
    meta_path = MODEL_DIR / f"{model_id}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    return meta
