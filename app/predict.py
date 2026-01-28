import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.storage import MODEL_DIR

predict_router = APIRouter()

@predict_router.post("/{model_id}")
def predict(model_id: str, payload: dict):
    model_path = MODEL_DIR / f"{model_id}.joblib"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    model = joblib.load(model_path)

    X = pd.DataFrame([payload])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": probability
    }
