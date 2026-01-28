import uuid
import pandas as pd
import joblib
from fastapi import APIRouter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from app.storage import MODEL_DIR

train_router = APIRouter()

@train_router.post("/")
def train_model():
    df = pd.read_csv("data/sample.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}.joblib"
    joblib.dump(model, model_path)

    return {
        "model_id": model_id,
        "accuracy": round(acc, 4)
    }
