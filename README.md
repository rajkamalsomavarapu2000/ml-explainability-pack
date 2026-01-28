# ML Model API

This repository provides a minimal machine learning service that supports:

- Training a binary classification model using scikit-learn
- Saving trained model artifacts
- Running predictions via an API

## Current Features
- Logistic Regression training
- Model persistence
- Prediction endpoint

## Planned Extensions
- Model explainability (SHAP-based global and local explanations)
- Partial dependence plots
- Explainability report generation

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
