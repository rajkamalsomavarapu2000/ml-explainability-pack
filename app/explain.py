import json
import numpy as np
import pandas as pd
import joblib
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from app.storage import MODEL_DIR, EXPLAIN_DIR

def generate_global_explanation(model_id: str):
    if not SHAP_AVAILABLE:
        raise Exception("SHAP is not installed. Please install shap to use explanations.")

    # Load meta
    meta_path = MODEL_DIR / f"{model_id}_meta.json"
    if not meta_path.exists():
        raise ValueError(f"Model {model_id} not found")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    if meta["model_type"] not in ["logistic", "random_forest"]:
        return {"message": "SHAP explanations are only supported for supervised classification models."}

    # Load the dataset
    if meta["dataset_id"] == "sample":
        csv_path = "data/sample.csv"
    else:
        csv_path = UPLOAD_DIR / f"{meta['dataset_id']}.csv"
    df = pd.read_csv(csv_path)
    X = df.drop(meta["target_column"], axis=1) if meta["target_column"] else df

    # Load the model
    model_path = MODEL_DIR / f"{model_id}.joblib"
    model = joblib.load(model_path)

    # Background sample
    background = X.sample(n=min(100, len(X)), random_state=42)

    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(background)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_importance = {}
    for i, feature in enumerate(X.columns):
        feature_importance[feature] = float(abs(shap_values[:, i]).mean())

    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return sorted_importance

def generate_local_explanation(model_id: str, row_index: int):
    if not SHAP_AVAILABLE:
        raise Exception("SHAP is not installed. Please install shap to use explanations.")

    # Load meta
    meta_path = MODEL_DIR / f"{model_id}_meta.json"
    if not meta_path.exists():
        raise ValueError(f"Model {model_id} not found")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    if meta["model_type"] not in ["logistic", "random_forest"]:
        return {"message": "SHAP explanations are only supported for supervised classification models."}

    # Load the dataset
    if meta["dataset_id"] == "sample":
        csv_path = "data/sample.csv"
    else:
        csv_path = UPLOAD_DIR / f"{meta['dataset_id']}.csv"
    df = pd.read_csv(csv_path)
    X = df.drop(meta["target_column"], axis=1) if meta["target_column"] else df

    if row_index >= len(X):
        raise ValueError("Row index out of range")

    instance = X.iloc[row_index]

    # Load the model
    model_path = MODEL_DIR / f"{model_id}.joblib"
    model = joblib.load(model_path)

    # Background sample
    background = X.sample(n=min(100, len(X)), random_state=42)

    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(instance.values.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    local_explanation = {}
    for i, feature in enumerate(X.columns):
        local_explanation[feature] = float(shap_values[0, i])

    return local_explanation